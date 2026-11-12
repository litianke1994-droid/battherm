"""Non-deep baselines: persistence and ridge regression on the flattened windows."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from battherm.data.windows import Scalers, WindowSet


class Persistence:
    """T[t+h] = T[t]  ->  predicted rise is zero. Any useful model must beat this."""

    def predict(self, ws: WindowSet) -> np.ndarray:
        return np.zeros_like(ws.y)


class RidgeForecaster:
    """Linear multi-output model. Past window is subsampled to keep the design matrix small;
    the full future current profile is kept because it drives the temperature rise."""

    def __init__(self, scalers: Scalers, alphas: list[float], past_subsample: int = 10):
        self.scalers = scalers
        self.alphas = alphas
        self.past_subsample = past_subsample
        self.model: Ridge | None = None
        self.alpha: float | None = None
        self.val_rmse: dict[float, float] = {}

    def _design(self, ws: WindowSet) -> np.ndarray:
        past = self.scalers.past.transform(ws.past)
        past = past[:, ::-1][:, :: self.past_subsample][:, ::-1]  # keep the most recent row
        future = self.scalers.future.transform(ws.future)
        return np.hstack([past.reshape(len(ws), -1), future.reshape(len(ws), -1)])

    def fit(self, train: WindowSet, val: WindowSet) -> "RidgeForecaster":
        Xtr, Xva = self._design(train), self._design(val)
        ytr = train.y / self.scalers.y_std
        best = None
        for a in self.alphas:  # alpha is chosen on VALIDATION, never on test
            m = Ridge(alpha=a).fit(Xtr, ytr)
            rmse = float(np.sqrt(np.mean((m.predict(Xva) * self.scalers.y_std - val.y) ** 2)))
            self.val_rmse[a] = rmse
            if best is None or rmse < best[0]:
                best = (rmse, a, m)
        _, self.alpha, self.model = best
        return self

    def predict(self, ws: WindowSet) -> np.ndarray:
        return (self.model.predict(self._design(ws)) * self.scalers.y_std).astype(np.float32)
