"""Tuning sweeps (validation split only) and the final multi-seed evaluation (test split)."""

from __future__ import annotations

import itertools
import json
import tomllib
from datetime import datetime
from pathlib import Path

import pandas as pd

from battherm.pipeline import load_config, run


def _parse_value(text: str):
    return tomllib.loads(f"v = {text}")["v"]


def _nested(dotted: str, value) -> dict:
    out: dict = {}
    cur = out
    keys = dotted.split(".")
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
    return out


def _merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        a[k] = _merge(a.get(k, {}), v) if isinstance(v, dict) else v
    return a


def parse_grid(settings: list[str]) -> list[dict]:
    """["lstm.hidden=32,64", "train.lr=1e-3"] -> list of nested override dicts (cartesian product)."""
    axes = []
    for s in settings:
        key, values = s.split("=", 1)
        axes.append([(key, _parse_value(v)) for v in values.split(";" if ";" in values else ",")])
    return [_merge_all(combo) for combo in itertools.product(*axes)] if axes else [{}]


def _merge_all(combo) -> dict:
    out: dict = {}
    for key, value in combo:
        _merge(out, _nested(key, value))
    return out


def sweep(config: Path, settings: list[str], models: list[str] | None, root: Path = Path(".")) -> pd.DataFrame:
    """Grid search evaluated on the VALIDATION split. The test split is never built."""
    name = load_config(config)["experiment"]["name"]
    rows = []
    for k, ov in enumerate(parse_grid(settings), 1):
        tag = json.dumps(ov, separators=(",", ":"))
        print(f"\n=== sweep {k}: {tag}")
        ov = _merge(ov, {"experiment": {"eval_split": "val"}})
        if models:
            ov = _merge(ov, {"models": {"enabled": models}})
        out, overall = run(config, root, ov)
        rows.append(overall.assign(setting=tag, run_dir=str(out)))
    table = pd.concat(rows, ignore_index=True)
    dest = root / "outputs" / "sweeps" / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest, index=False)
    print("\nvalidation RMSE (K) per setting:\n" + table.pivot_table(index="setting", columns="model", values="rmse_K").round(4).to_string())
    print(f"\nsweep table: {dest}")
    return table


def final(config: Path, seeds: list[int], root: Path = Path(".")) -> pd.DataFrame:
    """Evaluate the frozen config on the TEST split with several seeds; report mean and std."""
    name = load_config(config)["experiment"]["name"]
    rows, dirs = [], []
    for s in seeds:
        out, overall = run(config, root, {"experiment": {"eval_split": "test", "seed": s}})
        rows.append(overall)
        dirs.append(str(out))
    table = pd.concat(rows, ignore_index=True)
    metrics = ["rmse_K", "mae_K", "max_abs_K", "rmse_final_step_K", "skill_final_step"]
    agg = table.groupby("model", sort=False)[metrics].agg(["mean", "std"])
    dest = root / "outputs" / "final" / name
    dest.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest / "per_seed.csv", index=False)
    agg.to_csv(dest / "summary.csv")
    (dest / "runs.json").write_text(json.dumps({"seeds": seeds, "run_dirs": dirs}, indent=2))
    lines = ["| model | RMSE (K) | MAE (K) | max abs (K) | RMSE @ horizon (K) | skill @ horizon |", "|---|---:|---:|---:|---:|---:|"]
    for m, r in agg.iterrows():
        cells = [f"{r[(c, 'mean')]:.3f} ± {0 if pd.isna(r[(c, 'std')]) else r[(c, 'std')]:.3f}" for c in metrics]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    (dest / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nfinal results: {dest}")
    return agg
