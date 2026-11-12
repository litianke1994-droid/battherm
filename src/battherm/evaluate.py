"""Metrics in physical units (K). Errors on the rise T[t+h]-T[t] equal errors on T[t+h]."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "rmse_K": float(np.sqrt(np.mean(err**2))),
        "mae_K": float(np.mean(np.abs(err))),
        "max_abs_K": float(np.max(np.abs(err))),
        "rmse_final_step_K": float(np.sqrt(np.mean(err[:, -1] ** 2))),
        # Skill vs persistence on the final step: 1 = perfect, 0 = no better than "no change"
        "skill_final_step": float(1.0 - np.mean(err[:, -1] ** 2) / max(np.mean(y_true[:, -1] ** 2), 1e-12)),
    }


def rmse_per_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))


def grouped(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, name: str) -> pd.DataFrame:
    rows = []
    for g in pd.unique(groups):
        m = groups == g
        rows.append({name: g, "n_windows": int(m.sum()), **summary(y_true[m], y_pred[m])})
    return pd.DataFrame(rows).sort_values(name).reset_index(drop=True)
