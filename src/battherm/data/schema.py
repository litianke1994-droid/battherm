"""Canonical data schema shared by every data source.

Every loader (synthetic generator, LG 18650HG2 loader, your own data) must return
a long-format DataFrame with exactly these columns, one row per time step:

    cycle_id       str    unique id of one continuous measurement (the split unit)
    profile        str    load profile name, e.g. "UDDS", "US06", "pulse"
    time_s         float  seconds since cycle start, uniform step after resampling
    current_A      float  cell current; sign convention: charge > 0, discharge < 0
    voltage_V      float  terminal voltage
    temperature_C  float  cell surface temperature (the quantity we forecast)
    ambient_C      float  ambient / chamber temperature, constant within a cycle
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["cycle_id", "profile", "time_s", "current_A", "voltage_V", "temperature_C", "ambient_C"]
NUMERIC = ["time_s", "current_A", "voltage_V", "temperature_C", "ambient_C"]
SIGNALS = ["current_A", "voltage_V", "temperature_C"]


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError if df violates the canonical schema."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    nan_counts = df[NUMERIC].isna().sum()
    if nan_counts.any():
        raise ValueError(f"NaN values found:\n{nan_counts[nan_counts > 0]}")
    for cid, g in df.groupby("cycle_id", sort=False):
        if not g["time_s"].is_monotonic_increasing:
            raise ValueError(f"time_s not increasing in cycle {cid}")
        if g["ambient_C"].nunique() != 1:
            raise ValueError(f"ambient_C not constant in cycle {cid}")
        if g["profile"].nunique() != 1:
            raise ValueError(f"profile not constant in cycle {cid}")


def infer_dt(df: pd.DataFrame) -> float:
    """Median time step over all cycles."""
    diffs = df.groupby("cycle_id", sort=False)["time_s"].diff().dropna()
    return float(np.median(diffs))


def split_on_gaps(t: np.ndarray, max_gap_s: float) -> list[np.ndarray]:
    """Return index arrays of contiguous segments; a new segment starts after a gap > max_gap_s."""
    breaks = np.where(np.diff(t) > max_gap_s)[0] + 1
    return np.split(np.arange(len(t)), breaks)


def resample_uniform(cycle: pd.DataFrame, dt_s: float) -> pd.DataFrame:
    """Bin-average one gap-free cycle onto a uniform grid t = 0, dt, 2dt, ...

    Bin-averaging current preserves charge throughput. Note: mean(I)^2 <= mean(I^2),
    so Joule heat computed from resampled current is slightly underestimated when the
    raw signal fluctuates faster than dt_s (documented limitation).
    """
    g = cycle.sort_values("time_s")
    t = g["time_s"].to_numpy(dtype=float)
    t = t - t[0]
    bins = np.floor(t / dt_s + 1e-9).astype(np.int64)
    binned = g[SIGNALS].groupby(bins).mean()
    full = binned.reindex(np.arange(bins.max() + 1)).interpolate(limit_direction="both")
    out = full.reset_index(drop=True)
    out.insert(0, "time_s", np.arange(len(out)) * dt_s)
    out["ambient_C"] = float(g["ambient_C"].iloc[0])
    out["cycle_id"] = g["cycle_id"].iloc[0]
    out["profile"] = g["profile"].iloc[0]
    return out[COLUMNS]


def cycle_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per cycle: profile, ambient, length and temperature range (for audits and splits)."""
    return (
        df.groupby("cycle_id", sort=False)
        .agg(
            profile=("profile", "first"),
            ambient_C=("ambient_C", "first"),
            n_steps=("time_s", "size"),
            duration_s=("time_s", "max"),
            t_min_C=("temperature_C", "min"),
            t_max_C=("temperature_C", "max"),
            i_min_A=("current_A", "min"),
            i_max_A=("current_A", "max"),
        )
        .reset_index()
    )
