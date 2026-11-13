"""Static figures for the README / report. Colours follow the model, never its rank."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed model -> colour mapping (categorical slots 1..6 of a CVD-validated palette)
MODEL_COLORS = {
    "persistence": "#898781",  # muted grey: reference, not a contender
    "physics": "#2a78d6",
    "ridge": "#eb6834",
    "lstm": "#1baf7a",
    "cnn": "#eda100",
    "hybrid_lstm": "#e87ba4",
}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK_2)


def rmse_vs_horizon(per_h: dict[str, np.ndarray], dt_s: float, path: Path, split: str = "test") -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, r in per_h.items():
        x = np.arange(1, len(r) + 1) * dt_s
        ax.plot(x, r, color=MODEL_COLORS.get(name, INK), linewidth=2, label=name)
        ax.annotate(name, (x[-1], r[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
    _style(ax, f"{split.capitalize()} RMSE vs forecast horizon", "horizon (s)", "RMSE (K)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def rmse_by_group(table: pd.DataFrame, group: str, path: Path, split: str = "test") -> None:
    """Grouped bars: one group per ambient/profile, one bar per model."""
    models = list(table["model"].unique())
    groups = list(table[group].unique())
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(groups)), 4))
    x = np.arange(len(groups))
    for k, m in enumerate(models):
        sub = table[table["model"] == m].set_index(group).reindex(groups)
        ax.bar(x + k * width - 0.4 + width / 2, sub["rmse_K"], width * 0.92, color=MODEL_COLORS.get(m, INK), label=m)
    ax.set_xticks(x, [str(g) for g in groups])
    _style(ax, f"{split.capitalize()} RMSE by {group}", group, "RMSE (K)")
    ax.legend(frameon=False, fontsize=8, ncol=min(len(models), 6))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def learning_curves(histories: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, h in histories.items():
        c = MODEL_COLORS.get(name, INK)
        ax.plot(h["epoch"], h["train_rmse"], color=c, linewidth=1, linestyle="--")
        ax.plot(h["epoch"], h["val_rmse"], color=c, linewidth=2, label=f"{name} (val)")
    _style(ax, "Learning curves (dashed = train)", "epoch", "RMSE (scaled units)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def example_cycle(time_s, current, temp, preds: dict[str, tuple[np.ndarray, np.ndarray]], title: str, path: Path) -> None:
    """Measured temperature vs each model's forecast at the final horizon step, plus current."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(time_s, temp, color=INK, linewidth=1.2, label="measured")
    for name, (t, y) in preds.items():
        ax1.plot(t, y, color=MODEL_COLORS.get(name, INK), linewidth=1.5, label=name)
    _style(ax1, title, "", "temperature (°C)")
    ax1.legend(frameon=False, fontsize=8, ncol=4)
    ax2.plot(time_s, current, color=INK_2, linewidth=0.8)
    _style(ax2, "", "time (s)", "current (A)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
