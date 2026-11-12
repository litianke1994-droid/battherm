"""Grey-box lumped thermal model: the physics baseline and the base of the hybrid model.

    dT/dt = q(I, T) - k * (T - T_amb) + c0 + c1 * T_amb
    q(I, T) = I^2 * (a0 + a1 * d + a2 * d^2),   d = T - 25 C     (Joule heat / C_th, with
                                                                  a temperature-dependent resistance)
    k = hA / C_th                                                 (1/k = thermal time constant)
    c0 + c1 * T_amb                                               chamber-sensor offset: the real air
                                                                  temperature differs from the set point
                                                                  (HG2: about +0.3 K at -20 C, -1.2 K at 25 C)

All six parameters are linear, so they are fitted by least squares on the integral
form  T[t+m] - T[t] = sum_{j=t}^{t+m-1} phi_j * dt  over train cycles only. Integrating
over m steps averages out sensor noise that would dominate a finite-difference dT/dt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARAM_NAMES = ("a0", "a1", "a2", "k", "c0", "c1")


def _phi(i: np.ndarray, T: np.ndarray, T_amb: np.ndarray) -> np.ndarray:
    d = T - 25.0
    i2 = i * i
    return np.stack([i2, i2 * d, i2 * d * d, -(T - T_amb), np.ones_like(T), T_amb], axis=-1)


class LumpedThermalModel:
    def __init__(self, dt_s: float):
        self.dt_s = dt_s
        self.theta: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame, fit_steps: int) -> "LumpedThermalModel":
        X, y = [], []
        for _, g in train_df.groupby("cycle_id", sort=False):
            T = g["temperature_C"].to_numpy(float)
            if len(T) <= fit_steps:
                continue
            phi = _phi(g["current_A"].to_numpy(float), T, g["ambient_C"].to_numpy(float))
            cs = np.vstack([np.zeros((1, phi.shape[1])), np.cumsum(phi, axis=0)])
            starts = np.arange(0, len(T) - fit_steps, max(fit_steps // 2, 1))
            X.append((cs[starts + fit_steps] - cs[starts]) * self.dt_s)
            y.append(T[starts + fit_steps] - T[starts])
        if not X:
            raise ValueError("no train cycle is longer than fit_steps")
        self.theta, *_ = np.linalg.lstsq(np.vstack(X), np.concatenate(y), rcond=None)
        return self

    def rate(self, i: np.ndarray, T: np.ndarray, T_amb: np.ndarray) -> np.ndarray:
        a0, a1, a2, k, c0, c1 = self.theta
        d = T - 25.0
        heat = np.maximum(i * i * (a0 + a1 * d + a2 * d * d), 0.0)  # Joule heat cannot be negative
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
        p = dict(zip(PARAM_NAMES, map(float, self.theta)))
        p["time_constant_s"] = 1.0 / p["k"] if p["k"] > 0 else float("inf")
        return p
