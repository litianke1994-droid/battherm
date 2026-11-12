"""Loader for the public LG 18650HG2 dataset (McMaster University, Kollmeyer et al., 2020).

Source (verify version and licence before use): Mendeley Data, "LG 18650HG2 Li-ion Battery
Data and Example Deep Neural Network xEV SOC Estimator Script", doi:10.17632/cp3473x7xv.

Expected layout of the *original* dataset (待核实 against the files you download):
    <raw_dir>/<ambient folder, e.g. "25degC" or "n10degC">/<id>_<PROFILE>.csv
Each CSV starts with metadata lines, then a header row containing columns such as
"Prog Time", "Voltage", "Current", "Temperature", sometimes followed by a units row.

The loader is deliberately defensive: it finds the header row by content, matches
columns by keyword, drops non-numeric rows, splits on logging gaps and resamples
to a uniform grid. Always read the audit table it prints before training.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from battherm.data.schema import COLUMNS, resample_uniform, split_on_gaps

AMBIENT_RE = re.compile(r"(n|-|neg)?\s*(\d{1,2})\s*deg\s*c", re.IGNORECASE)
PROFILE_RE = re.compile(r"(UDDS|HWFET|LA92|US06|NN|Mixed\s*\d+|HPPC|Charge|Dis\w*|Paus\w*)", re.IGNORECASE)
DRIVE_CYCLE_RE = r"^(UDDS|HWFET|LA92|US06|NN|Mixed\d+)$"


def parse_ambient(path: Path) -> float:
    """Ambient temperature from the deepest path component that encodes it ("n10degC" -> -10)."""
    for part in reversed(path.parts):
        m = AMBIENT_RE.search(part)
        if m:
            value = float(m.group(2))
            return -value if m.group(1) else value
    raise ValueError(f"cannot parse ambient temperature from path: {path}")


def parse_profile(path: Path) -> str:
    m = PROFILE_RE.search(path.stem)
    if not m:
        return path.stem
    name = re.sub(r"\s+", "", m.group(1))
    return name[0].upper() + name[1:] if name.lower().startswith("mixed") else name.upper()


def find_header_line(path: Path, max_lines: int = 200) -> int:
    with open(path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            low = line.lower()
            if "voltage" in low and "current" in low and "temp" in low:
                return i
            if i >= max_lines:
                break
    raise ValueError(f"no header row with voltage/current/temperature in {path}")


def _pick(columns: list[str], *keywords: str) -> str | None:
    for c in columns:
        low = c.lower()
        if all(k in low for k in keywords):
            return c
    return None


def _time_seconds(raw: pd.DataFrame) -> np.ndarray:
    cols = list(raw.columns)
    prog = _pick(cols, "prog", "time")
    if prog is not None:
        td = pd.to_timedelta(raw[prog].astype(str).str.strip(), errors="coerce")
        if td.notna().mean() > 0.9:
            return td.dt.total_seconds().to_numpy()
    stamp = _pick(cols, "time", "stamp")
    if stamp is not None:
        ts = pd.to_datetime(raw[stamp], errors="coerce")
        if ts.notna().mean() > 0.9:
            return (ts - ts.dropna().iloc[0]).dt.total_seconds().to_numpy()
    raise ValueError("no parseable time column (looked for 'Prog Time' and 'Time Stamp')")


def load_hg2_csv(path: Path, dt_s: float = 1.0, max_gap_s: float = 5.0) -> pd.DataFrame:
    """Load one HG2 CSV into the canonical schema (possibly several gap-free segments)."""
    path = Path(path)
    header = find_header_line(path)
    raw = pd.read_csv(path, skiprows=header, low_memory=False, encoding_errors="ignore")
    raw.columns = [str(c).strip() for c in raw.columns]
    cols = list(raw.columns)
    col_v, col_i, col_t = _pick(cols, "voltage"), _pick(cols, "current"), _pick(cols, "temp")
    if None in (col_v, col_i, col_t):
        raise ValueError(f"missing voltage/current/temperature columns in {path}: {cols}")

    df = pd.DataFrame(
        {
            "time_s": _time_seconds(raw),
            "current_A": pd.to_numeric(raw[col_i], errors="coerce"),
            "voltage_V": pd.to_numeric(raw[col_v], errors="coerce"),
            "temperature_C": pd.to_numeric(raw[col_t], errors="coerce"),
        }
    ).dropna()  # removes units rows and broken lines
    df = df.sort_values("time_s").drop_duplicates("time_s")

    ambient = parse_ambient(path)
    profile = parse_profile(path)
    base_id = f"hg2_{ambient:+.0f}C_{path.stem}"
    segments = []
    for k, idx in enumerate(split_on_gaps(df["time_s"].to_numpy(), max_gap_s)):
        seg = df.iloc[idx].copy()
        if len(seg) < 10:
            continue
        seg["ambient_C"] = ambient
        seg["profile"] = profile
        seg["cycle_id"] = base_id if k == 0 else f"{base_id}_seg{k}"
        segments.append(resample_uniform(seg, dt_s))
    if not segments:
        raise ValueError(f"no usable segments in {path}")
    return pd.concat(segments, ignore_index=True)[COLUMNS]


def quality_filter(
    df: pd.DataFrame, min_current_rms_A: float = 0.1, max_start_offset_K: float = 5.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop cycles that cannot be used, by rules fixed BEFORE any model is trained.

    * no load: current RMS below `min_current_rms_A` (a pure rest recording)
    * wrong ambient label: median temperature of the first 60 s differs from the folder's
      ambient by more than `max_start_offset_K` (the chamber was not at the labelled
      temperature, e.g. 40degC/562_Mixed4..8 start at ~23 C)
    Returns (kept data, table of every cycle with its checks and exclusion reason).
    """
    g = df.groupby("cycle_id", sort=False)
    checks = pd.DataFrame(
        {
            "current_rms_A": g["current_A"].apply(lambda s: float(np.sqrt(np.mean(s**2)))),
            "start_offset_K": g["temperature_C"].apply(lambda s: float(s.iloc[:60].median())) - g["ambient_C"].first(),
        }
    )
    reason = pd.Series("", index=checks.index)
    reason[checks["current_rms_A"] < min_current_rms_A] = "no load"
    reason[checks["start_offset_K"].abs() > max_start_offset_K] = "ambient label mismatch"
    checks["excluded"] = reason
    keep = checks.index[reason == ""]
    return df[df["cycle_id"].isin(keep)].reset_index(drop=True), checks.reset_index()


def load_hg2_dir(
    raw_dir: Path,
    dt_s: float = 1.0,
    include: str = DRIVE_CYCLE_RE,
    max_gap_s: float = 5.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Load every CSV whose parsed profile matches `include`. Returns (data, skipped-file messages)."""
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files under {raw_dir}")
    pattern = re.compile(include, re.IGNORECASE)
    frames, skipped = [], []
    for f in files:
        profile = parse_profile(f)
        if not pattern.match(profile):
            skipped.append(f"{f.name}: profile '{profile}' not included")
            continue
        try:
            frames.append(load_hg2_csv(f, dt_s=dt_s, max_gap_s=max_gap_s))
        except ValueError as e:
            skipped.append(f"{f.name}: {e}")
    if not frames:
        raise ValueError("no files loaded; check `include` pattern and folder layout")
    return pd.concat(frames, ignore_index=True), skipped
