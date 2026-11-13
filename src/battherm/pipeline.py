"""End-to-end experiment: load -> split -> window -> scale -> fit models -> evaluate -> report."""

from __future__ import annotations

import json
import copy
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from battherm import evaluate, plots
from battherm.data.schema import cycle_table, infer_dt, validate
from battherm.data.windows import (
    FUTURE_FEATURES,
    PAST_FEATURES,
    Scalers,
    WindowSet,
    WindowSpec,
    add_features,
    make_windows,
    split_cycles,
)
from battherm.models.baselines import Persistence, RidgeForecaster
from battherm.models.nets import build_model
from battherm.models.physics import LumpedThermalModel
from battherm.train import Arrays, fit, predict, resolve_device, set_seed

NET_MODELS = ("lstm", "cnn", "hybrid_lstm")


def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _arrays(ws: WindowSet, sc: Scalers, base_K: np.ndarray | None, base_as_input: bool = False) -> Arrays:
    """Scale one window set. For the hybrid model, `base_K` is the physics forecast: the network
    output is added to it (residual learning) and, if `base_as_input`, it is also fed to the
    decoder as an extra future channel so the network knows what it is correcting."""
    base = np.zeros_like(ws.y) if base_K is None else (base_K / sc.y_std).astype(np.float32)
    future = sc.future.transform(ws.future)
    if base_K is not None and base_as_input:
        future = np.concatenate([future, base[:, :, None]], axis=2)
    return Arrays(sc.past.transform(ws.past), future, ws.y / sc.y_std, base)


def deep_update(base: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def run(config_path: Path, root: Path = Path("."), overrides: dict | None = None) -> tuple[Path, pd.DataFrame]:
    """Run one experiment. `overrides` (nested dict) patches the TOML config, e.g. for sweeps.

    `experiment.eval_split` = "val" evaluates on the validation split and never touches the
    test split (use it for every tuning run); "test" is for the final, pre-registered run.
    """
    cfg = deep_update(load_config(config_path), copy.deepcopy(overrides or {}))
    exp, seed = cfg["experiment"], cfg["experiment"]["seed"]
    ev = exp.get("eval_split", "test")
    if ev not in ("val", "test"):
        raise ValueError("experiment.eval_split must be 'val' or 'test'")
    set_seed(seed)
    out = root / exp.get("output_dir", "outputs") / f"{exp['name']}_{ev}_s{seed}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    (out / "figures").mkdir(parents=True)
    (out / "models").mkdir()
    (out / "config_resolved.json").write_text(json.dumps(cfg, indent=2))
    log_file = open(out / "log.txt", "w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    # ---------------------------------------------------------------- data
    df = pd.read_csv(root / exp["data_path"])
    validate(df)
    dt = infer_dt(df)
    log(f"data: {len(df):,} rows, {df['cycle_id'].nunique()} cycles, dt = {dt:g} s")
    df = add_features(df)
    cycles = cycle_table(df)

    split = split_cycles(cycles, cfg["split"], seed)
    (out / "split.json").write_text(json.dumps(split, indent=2))
    for name, ids in split.items():
        sub = cycles[cycles["cycle_id"].isin(ids)]
        log(f"split {name:5s}: {len(ids):3d} cycles | ambients {sorted(sub['ambient_C'].unique().tolist())} | profiles {sorted(sub['profile'].unique().tolist())}")

    w = cfg["windows"]
    steps = lambda s: max(int(round(s / dt)), 1)  # noqa: E731
    W, H = steps(w["past_s"]), steps(w["horizon_s"])
    spec_train = WindowSpec(W, H, steps(w["train_stride_s"]))
    spec_eval = WindowSpec(W, H, steps(w["eval_stride_s"]))
    ws = {
        name: make_windows(df[df["cycle_id"].isin(ids)], spec_train if name == "train" else spec_eval)
        for name, ids in split.items()
        if name != "test" or ev == "test"  # tuning runs never even build test windows
    }
    log(f"windows: past {W} steps, horizon {H} steps | " + ", ".join(f"{k} {len(v):,}" for k, v in ws.items()))

    scalers = Scalers.fit(ws["train"])  # TRAIN ONLY
    (out / "models" / "scalers.json").write_text(json.dumps(scalers.to_dict(), indent=2))

    # ---------------------------------------------------------------- models
    enabled = cfg["models"]["enabled"]
    preds: dict[str, np.ndarray] = {}
    info: dict[str, dict] = {}

    if "persistence" in enabled:
        preds["persistence"] = Persistence().predict(ws[ev])

    physics = None
    if "physics" in enabled or "hybrid_lstm" in enabled:
        physics = LumpedThermalModel(dt).fit(df[df["cycle_id"].isin(split["train"])], steps(cfg["physics"]["fit_window_s"]))
        info["physics"] = physics.params()
        log(f"physics params: {json.dumps({k: round(v, 6) for k, v in info['physics'].items()})}")
        phys_pred = {k: physics.predict(v.t_now, v.ambient, v.future[:, :, FUTURE_FEATURES.index("current_A")]) for k, v in ws.items()}
        if "physics" in enabled:
            preds["physics"] = phys_pred[ev]

    if "ridge" in enabled:
        ridge = RidgeForecaster(scalers, cfg["ridge"]["alphas"], cfg["ridge"].get("past_subsample", 10)).fit(ws["train"], ws["val"])
        preds["ridge"] = ridge.predict(ws[ev])
        info["ridge"] = {"alpha": ridge.alpha, "val_rmse_K_by_alpha": {str(k): v for k, v in ridge.val_rmse.items()}}
        log(f"ridge: chosen alpha {ridge.alpha} (validation)")

    device = resolve_device(cfg["train"].get("device", "auto"))
    histories = {}
    for kind in [m for m in enabled if m in NET_MODELS]:
        set_seed(seed)
        log(f"training {kind} on {device} ...")
        hybrid = kind.startswith("hybrid")
        base_in = hybrid and cfg.get("hybrid", {}).get("physics_as_input", True)
        arr = {k: _arrays(v, scalers, phys_pred[k] if hybrid else None, base_in) for k, v in ws.items()}
        model = build_model(kind, len(PAST_FEATURES), arr["train"].future.shape[2], cfg)
        histories[kind] = fit(model, arr["train"], arr["val"], cfg["train"], device, seed, log)
        preds[kind] = predict(model, arr[ev], device) * scalers.y_std
        torch.save(model.state_dict(), out / "models" / f"{kind}.pt")
        info[kind] = {"n_params": sum(p.numel() for p in model.parameters()), "best_epoch": int(np.argmin(histories[kind]["val_rmse"]) + 1)}
    (out / "histories.json").write_text(json.dumps(histories, indent=2))
    (out / "model_info.json").write_text(json.dumps(info, indent=2))

    # ---------------------------------------------------------------- evaluation (one split only)
    test = ws[ev]
    overall = pd.DataFrame([{"model": m, **evaluate.summary(test.y, p)} for m, p in preds.items()])
    by_amb = pd.concat([evaluate.grouped(test.y, p, test.ambient, "ambient_C").assign(model=m) for m, p in preds.items()])
    by_prof = pd.concat([evaluate.grouped(test.y, p, test.profile, "profile").assign(model=m) for m, p in preds.items()])
    per_h = {m: evaluate.rmse_per_horizon(test.y, p) for m, p in preds.items()}

    overall.to_csv(out / "metrics_overall.csv", index=False)
    by_amb.to_csv(out / "metrics_by_ambient.csv", index=False)
    by_prof.to_csv(out / "metrics_by_profile.csv", index=False)
    pd.DataFrame(per_h).assign(horizon_s=np.arange(1, H + 1) * dt).to_csv(out / "rmse_per_horizon.csv", index=False)
    log(f"\n{ev.upper()} metrics (K):\n" + overall.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------------------------------------------------------------- figures
    fig = out / "figures"
    plots.rmse_vs_horizon(per_h, dt, fig / "rmse_vs_horizon.png", ev)
    plots.rmse_by_group(by_amb, "ambient_C", fig / "rmse_by_ambient.png", ev)
    plots.rmse_by_group(by_prof, "profile", fig / "rmse_by_profile.png", ev)
    if histories:
        plots.learning_curves(histories, fig / "learning_curves.png")
    cid = _example_cycle(test, df)
    g = df[df["cycle_id"] == cid]
    m = test.cycle_id == cid
    t_target = test.time_s[m] + H * dt
    ex = {k: (t_target, test.t_now[m] + p[m, -1]) for k, p in preds.items() if k != "persistence"}
    plots.example_cycle(g["time_s"], g["current_A"], g["temperature_C"], ex, f"{cid}: forecast {H * dt:g} s ahead", fig / "example_cycle.png")

    log(f"\nresults written to {out}")
    log_file.close()
    overall.insert(0, "eval_split", ev)
    overall.insert(1, "seed", seed)
    return out, overall


def _example_cycle(test: WindowSet, df: pd.DataFrame) -> str:
    """Test cycle with the largest temperature swing (the most informative to look at)."""
    ids = pd.unique(test.cycle_id)
    swing = df[df["cycle_id"].isin(ids)].groupby("cycle_id")["temperature_C"].agg(lambda s: s.max() - s.min())
    return str(swing.idxmax())
