"""Forecasting task: windows, cycle-level splits and train-only scaling.

Task definition (at forecast time index t, uniform step dt):
    inputs  past    rows t-W+1 .. t      of PAST_FEATURES      (what the BMS has measured)
            future  rows t   .. t+H-1    of FUTURE_FEATURES    (planned / predicted load)
    target  y[h-1] = T[t+h] - T[t],  h = 1..H                  (temperature rise in K)

current[t+h-1] is applied during step t+h-1 and therefore drives T[t+h]; the target
temperatures T[t+1..t+H] never appear in the inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PAST_FEATURES = ("current_A", "voltage_V", "temperature_C", "ambient_C", "current_sq")
FUTURE_FEATURES = ("current_A", "current_sq")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["current_sq"] = out["current_A"] ** 2  # Joule heating scales with I^2
    return out


@dataclass(frozen=True)
class WindowSpec:
    past_steps: int
    horizon_steps: int
    stride_steps: int


@dataclass
class WindowSet:
    past: np.ndarray  # (N, W, len(PAST_FEATURES))  raw physical units
    future: np.ndarray  # (N, H, len(FUTURE_FEATURES))
    y: np.ndarray  # (N, H)  temperature rise in K
    t_now: np.ndarray  # (N,)   T[t]
    ambient: np.ndarray  # (N,)
    time_s: np.ndarray  # (N,)   time of t within its cycle
    cycle_id: np.ndarray  # (N,)
    profile: np.ndarray  # (N,)
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)


def make_windows(df: pd.DataFrame, spec: WindowSpec) -> WindowSet:
    """Slide windows inside each cycle independently; no window crosses a cycle boundary."""
    W, H, S = spec.past_steps, spec.horizon_steps, spec.stride_steps
    parts: dict[str, list] = {k: [] for k in ("past", "future", "y", "t_now", "ambient", "time_s", "cycle_id", "profile")}
    for cid, g in df.groupby("cycle_id", sort=False):
        n = len(g)
        if n < W + H:
            continue
        p = g[list(PAST_FEATURES)].to_numpy(np.float32)
        f = g[list(FUTURE_FEATURES)].to_numpy(np.float32)
        T = g["temperature_C"].to_numpy(np.float32)
        ends = np.arange(W - 1, n - H, S)  # forecast origin t; requires t + H <= n - 1
        parts["past"].append(p[ends[:, None] + np.arange(-W + 1, 1)[None, :]])
        parts["future"].append(f[ends[:, None] + np.arange(0, H)[None, :]])
        parts["y"].append(T[ends[:, None] + np.arange(1, H + 1)[None, :]] - T[ends][:, None])
        parts["t_now"].append(T[ends])
        parts["ambient"].append(np.full(len(ends), g["ambient_C"].iloc[0], np.float32))
        parts["time_s"].append(g["time_s"].to_numpy()[ends])
        parts["cycle_id"].append(np.full(len(ends), cid, dtype=object))
        parts["profile"].append(np.full(len(ends), g["profile"].iloc[0], dtype=object))
    if not parts["y"]:
        raise ValueError("no windows: every cycle is shorter than past + horizon")
    return WindowSet(**{k: np.concatenate(v) for k, v in parts.items()})


def relative_temperature(ws: WindowSet) -> WindowSet:
    """Replace the absolute temperatures in the past inputs by relative ones (v2 option):

        temperature_C -> T[j] - T[t]      (recent history relative to the forecast origin)
        ambient_C     -> T[t] - T_amb     (how far the cell is above its surroundings)

    The target (a rise) does not depend on the absolute level either, so the network has no
    input outside the training range when it is applied at an unseen ambient temperature.
    Information about the temperature-dependent resistance is still in the voltage response.
    The physics model keeps using `t_now` and `ambient`, which are not changed."""
    past = ws.past.copy()
    iT, iA = PAST_FEATURES.index("temperature_C"), PAST_FEATURES.index("ambient_C")
    past[:, :, iT] = ws.past[:, :, iT] - ws.t_now[:, None]
    past[:, :, iA] = (ws.t_now - ws.ambient)[:, None]
    return WindowSet(past, ws.future, ws.y, ws.t_now, ws.ambient, ws.time_s, ws.cycle_id, ws.profile, dict(ws.extras))


# ---------------------------------------------------------------- splits
def split_cycles(cycles: pd.DataFrame, cfg: dict, seed: int) -> dict[str, list[str]]:
    """Assign whole cycles to train / val / test (the group split that prevents leakage).

    1. Explicit rules first: `test_profiles`, `test_ambients`, `val_profiles`, `val_ambients`.
    2. Then `test_fraction` / `val_fraction` of the remaining cycles, drawn at random
       per ambient so every ambient temperature stays represented in training.
    """
    rng = np.random.default_rng(seed)
    prof, amb = cycles["profile"], cycles["ambient_C"]
    test = prof.isin(cfg.get("test_profiles", [])) | amb.isin(cfg.get("test_ambients", []))
    val = (prof.isin(cfg.get("val_profiles", [])) | amb.isin(cfg.get("val_ambients", []))) & ~test
    remaining = cycles[~test & ~val]

    drawn: dict[str, list[str]] = {"test": [], "val": []}
    for _, g in remaining.groupby("ambient_C"):
        ids = list(rng.permutation(g["cycle_id"].to_numpy()))
        for name in ("test", "val"):
            k = int(round(cfg.get(f"{name}_fraction", 0.0) * len(g)))
            drawn[name] += ids[:k]
            ids = ids[k:]

    split = {
        "test": sorted(set(cycles.loc[test, "cycle_id"]) | set(drawn["test"])),
        "val": sorted(set(cycles.loc[val, "cycle_id"]) | set(drawn["val"])),
    }
    used = set(split["test"]) | set(split["val"])
    split["train"] = sorted(c for c in cycles["cycle_id"] if c not in used)
    for name, ids in split.items():
        if not ids:
            raise ValueError(f"split '{name}' is empty; adjust the [split] config")
    assert not (set(split["train"]) & set(split["val"]) | set(split["train"]) & set(split["test"]) | set(split["val"]) & set(split["test"]))
    return split


# ---------------------------------------------------------------- scaling
@dataclass
class Standardizer:
    """Per-feature z-scaling. Fit on TRAIN windows only, then apply everywhere."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
        std = flat.std(axis=0)
        return cls(flat.mean(axis=0), np.where(std < 1e-8, 1.0, std))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


@dataclass
class Scalers:
    past: Standardizer
    future: Standardizer
    y_std: float  # targets are scaled by one scalar so the loss stays proportional to K^2

    @classmethod
    def fit(cls, train: WindowSet) -> "Scalers":
        return cls(Standardizer.fit(train.past), Standardizer.fit(train.future), float(train.y.std()) or 1.0)

    def to_dict(self) -> dict:
        return {"past": self.past.to_dict(), "future": self.future.to_dict(), "y_std": self.y_std,
                "past_features": list(PAST_FEATURES), "future_features": list(FUTURE_FEATURES)}
