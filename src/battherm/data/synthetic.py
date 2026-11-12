"""Synthetic cell data from a lumped electro-thermal model.

Purpose: an offline, fully known "ground truth" for debugging the pipeline and unit
tests. It is NOT a substitute for measured data in any reported result.

Electrical model (1-RC equivalent circuit, charge current > 0):
    V   = OCV(SOC) + I * R0(T) + V1
    dV1/dt = (-V1 + I * R1(T)) / tau1
Thermal model (single lumped node, surface temperature = core temperature):
    C_th * dT/dt = I^2 * R0(T) + V1^2 / R1(T) - hA * (T - T_amb)
Resistances follow an Arrhenius-type law:  R(T) = R_ref * exp(Ea * (1/T_K - 1/298.15)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from battherm.data.schema import COLUMNS


@dataclass(frozen=True)
class CellParams:
    capacity_Ah: float = 3.0
    r0_ref_ohm: float = 0.020
    r1_ref_ohm: float = 0.015
    activation_K: float = 2500.0
    tau1_s: float = 30.0
    heat_capacity_J_per_K: float = 45.0  # ~45 g 18650 cell, cp ~ 1000 J/(kg K)
    h_a_W_per_K: float = 0.08  # convective conductance to chamber air
    h_a_jitter: float = 0.05  # per-cycle relative variation of hA (airflow differences)
    temp_noise_C: float = 0.05
    volt_noise_V: float = 0.002
    curr_noise_A: float = 0.01


def ocv(soc: np.ndarray | float) -> np.ndarray | float:
    """Smooth, monotonic open-circuit voltage curve (illustrative, not a real chemistry)."""
    return 3.0 + 1.15 * soc - 0.15 * np.exp(-15.0 * soc) + 0.05 * np.tanh(10.0 * (soc - 0.5))


def arrhenius_factor(temp_C: float, activation_K: float) -> float:
    return float(np.exp(activation_K * (1.0 / (temp_C + 273.15) - 1.0 / 298.15)))


# ---------------------------------------------------------------- current profiles
def profile_pulse(rng: np.random.Generator, n: int, cap: float) -> np.ndarray:
    """Piecewise-constant pulses and rests (HPPC-like)."""
    levels = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0]) * cap
    out = np.empty(n)
    k = 0
    while k < n:
        dur = int(rng.integers(20, 180))
        out[k : k + dur] = rng.choice(levels)
        k += dur
    return out


def profile_drive(rng: np.random.Generator, n: int, cap: float) -> np.ndarray:
    """Drive-cycle-like current: filtered noise with accelerations and regen braking."""
    x = np.zeros(n)
    phi = 0.97
    eps = rng.normal(0.0, 1.0, n)
    for k in range(1, n):
        x[k] = phi * x[k - 1] + eps[k]
    x /= x.std() + 1e-9
    target_rms = rng.uniform(0.7, 1.8) * cap
    i = -(0.4 * target_rms + target_rms * x)  # mostly discharge, negative excursions = regen
    stop = rng.random(n) < 0.002  # occasional stops
    for s in np.where(stop)[0]:
        i[s : s + int(rng.integers(10, 60))] = 0.0
    return np.clip(i, -5.0 * cap, 2.0 * cap)


def profile_cc(rng: np.random.Generator, n: int, cap: float) -> np.ndarray:
    """Rest, constant-current discharge, rest (cool-down)."""
    out = np.zeros(n)
    rate = rng.uniform(0.5, 2.0) * cap
    start = int(rng.integers(30, 120))
    stop = start + int(rng.integers(n // 3, (2 * n) // 3))
    out[start:stop] = -rate
    return out


PROFILES = {"pulse": profile_pulse, "drive": profile_drive, "cc": profile_cc}


# ---------------------------------------------------------------- simulation
def simulate_cycle(
    current: np.ndarray,
    ambient_C: float,
    params: CellParams,
    rng: np.random.Generator,
    soc0: float = 1.0,
    dt_s: float = 1.0,
) -> dict[str, np.ndarray]:
    """Explicit-Euler simulation. Row k holds the state at the START of step k; current[k]
    is applied during step k, so it affects temperature_C[k+1] (forecast convention)."""
    p = params
    n = len(current)
    h_a = p.h_a_W_per_K * float(np.exp(rng.normal(0.0, p.h_a_jitter)))
    temp = ambient_C + rng.normal(0.0, 0.2)
    soc, v1 = soc0, 0.0
    T = np.empty(n)
    V = np.empty(n)
    end = n
    for k in range(n):
        i = current[k]
        f = arrhenius_factor(temp, p.activation_K)
        r0, r1 = p.r0_ref_ohm * f, p.r1_ref_ohm * f
        T[k] = temp
        V[k] = ocv(soc) + i * r0 + v1
        q_gen = i * i * r0 + v1 * v1 / r1
        temp += dt_s * (q_gen - h_a * (temp - ambient_C)) / p.heat_capacity_J_per_K
        v1 += dt_s * (-v1 + i * r1) / p.tau1_s
        soc = min(soc + i * dt_s / (3600.0 * p.capacity_Ah), 1.0)
        if soc < 0.02:  # cut-off: stop the cycle
            end = k + 1
            break
    return {
        "current_A": current[:end] + rng.normal(0.0, p.curr_noise_A, end),
        "voltage_V": V[:end] + rng.normal(0.0, p.volt_noise_V, end),
        "temperature_C": T[:end] + rng.normal(0.0, p.temp_noise_C, end),
    }


def generate_dataset(
    ambients_C: tuple[float, ...] = (-10.0, 0.0, 10.0, 25.0, 40.0),
    cycles_per_ambient: int = 9,
    duration_s: int = 3600,
    dt_s: float = 1.0,
    seed: int = 0,
    params: CellParams | None = None,
) -> pd.DataFrame:
    """Generate cycles for every ambient, rotating through the three profile types."""
    params = params or CellParams()
    rng = np.random.default_rng(seed)
    n = int(duration_s / dt_s)
    kinds = list(PROFILES)
    frames = []
    for amb in ambients_C:
        for j in range(cycles_per_ambient):
            kind = kinds[j % len(kinds)]
            current = PROFILES[kind](rng, n, params.capacity_Ah)
            sig = simulate_cycle(current, amb, params, rng, soc0=rng.uniform(0.7, 1.0), dt_s=dt_s)
            m = len(sig["temperature_C"])
            frames.append(
                pd.DataFrame(
                    {
                        "cycle_id": f"syn_{amb:+.0f}C_{j:02d}_{kind}",
                        "profile": kind,
                        "time_s": np.arange(m) * dt_s,
                        **sig,
                        "ambient_C": float(amb),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)[COLUMNS]
