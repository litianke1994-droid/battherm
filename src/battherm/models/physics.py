"""Grey-box lumped thermal model: the physics baseline and the base of the hybrid model.

    dT/dt = q(I, T) - k * (T - T_amb) + c0 + c1 * T_amb
    k = hA / C_th                                   (1/k = thermal time constant)
    c0 + c1 * T_amb                                 chamber-sensor offset: the real air temperature
                                                    differs from the set point (HG2: about +0.3 K at
                                                    -20 C, -1.2 K at 25 C)

Two forms of the Joule-heat term q = I^2 R(T) / C_th, with d = T - 25 C:

    heat = "poly"        q = I^2 (a0 + a1 d + a2 d^2)    v1. Flexible inside the training range but
                                                         NOT monotone: fitted on -20..25 C it bends
                                                         upward again above 25 C.
    heat = "arrhenius"   q = I^2 a exp(-beta d), beta>=0 v2. Resistance falls monotonically with
                                                         temperature (Arrhenius-like), so it
                                                         extrapolates to hotter cells sensibly.

For fixed beta every parameter is linear, so the model is fitted by least squares on the integral
form  T[t+m] - T[t] = sum_{j=t}^{t+m-1} phi_j * dt  over train cycles only (integrating over m
steps averages out sensor noise that would dominate a finite-difference dT/dt). beta is chosen by
a 1-D grid search on the same train residual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARAM_NAMES = {"poly": ("a0", "a1", "a2", "k", "c0", "c1"), "arrhenius": ("a", "k", "c0", "c1")}
BETA_GRID = np.round(np.arange(0.0, 0.1501, 0.0025), 4)  # 1/K


def _phi(i: np.ndarray, T: np.ndarray, T_amb: np.ndarray, heat: str, beta: float = 0.0) -> np.ndarray:
    d = T - 25.0
    i2 = i * i
    rest = [-(T - T_amb), np.ones_like(T), T_amb]
    if heat == "poly":
        return np.stack([i2, i2 * d, i2 * d * d, *rest], axis=-1)
    return np.stack([i2 * np.exp(-beta * d), *rest], axis=-1)


class LumpedThermalModel:
    def __init__(self, dt_s: float, heat: str = "poly"):
        if heat not in PARAM_NAMES:
            raise ValueError(f"heat must be one of {list(PARAM_NAMES)}")
        self.dt_s, self.heat = dt_s, heat
        self.theta: np.ndarray | None = None
        self.beta = 0.0

    def _design(self, train_df: pd.DataFrame, fit_steps: int, beta: float) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for _, g in train_df.groupby("cycle_id", sort=False):
            T = g["temperature_C"].to_numpy(float)
            if len(T) <= fit_steps:
                continue
            phi = _phi(g["current_A"].to_numpy(float), T, g["ambient_C"].to_numpy(float), self.heat, beta)
            cs = np.vstack([np.zeros((1, phi.shape[1])), np.cumsum(phi, axis=0)])
            starts = np.arange(0, len(T) - fit_steps, max(fit_steps // 2, 1))
            X.append((cs[starts + fit_steps] - cs[starts]) * self.dt_s)
            y.append(T[starts + fit_steps] - T[starts])
        if not X:
            raise ValueError("no train cycle is longer than fit_steps")
        return np.vstack(X), np.concatenate(y)

    def fit(self, train_df: pd.DataFrame, fit_steps: int) -> "LumpedThermalModel":
        best = (np.inf, 0.0, None)
        for beta in BETA_GRID if self.heat == "arrhenius" else [0.0]:
            X, y = self._design(train_df, fit_steps, beta)
            theta, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse = float(np.sum((X @ theta - y) ** 2))
            if sse < best[0]:
                best = (sse, float(beta), theta)
        _, self.beta, self.theta = best
        return self

    def rate(self, i: np.ndarray, T: np.ndarray, T_amb: np.ndarray) -> np.ndarray:
        d = T - 25.0
        if self.heat == "poly":
            a0, a1, a2, k, c0, c1 = self.theta
            heat = np.maximum(i * i * (a0 + a1 * d + a2 * d * d), 0.0)  # Joule heat cannot be negative
        else:
            a, k, c0, c1 = self.theta
            heat = np.maximum(a, 0.0) * i * i * np.exp(-self.beta * d)
        return heat - k * (T - T_amb) + c0 + c1 * T_amb

    def predict(self, t_now: np.ndarray, ambient: np.ndarray, future_current: np.ndarray) -> np.ndarray:
        """Roll the ODE forward from T[t] with the known future current. Returns (N, H) rise in K."""
        if self.theta is None:
            raise RuntimeError("model is not fitted")
        T = t_now.astype(float).copy()
        out = np.empty(future_current.shape, dtype=float)
        for h in range(future_current.shape[1]):
            T = T + self.dt_s * self.rate(future_current[:, h].astype(float), T, ambient)
            out[:, h] = T - t_now
        return out.astype(np.float32)

    def params(self) -> dict[str, float]:
        p = dict(zip(PARAM_NAMES[self.heat], map(float, self.theta)))
        if self.heat == "arrhenius":
            p["beta_per_K"] = self.beta
        p["time_constant_s"] = 1.0 / p["k"] if p["k"] > 0 else float("inf")
        return p
