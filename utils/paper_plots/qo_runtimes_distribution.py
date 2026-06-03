from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.paper_plots.utils import _COLORS

_BINS = [0.3, 0.9, 1.1, 2, 10, float("inf")]
_BIN_LABELS = ["[0.3,0.9)", "[0.9,1.1)", "[1.1,2)", "[2,10)", "≥10"]


def _bin_slowdowns(slowdowns: pd.Series) -> np.ndarray:
    counts = np.zeros(len(_BIN_LABELS))
    for i in range(len(_BIN_LABELS)):
        lo = _BINS[i]
        hi = _BINS[i + 1]
        counts[i] = ((slowdowns >= lo) & (slowdowns < hi)).sum()
    return counts / len(slowdowns) * 100


_RED = "#FF0048"
_GREEN = "#26d62f"
_HATCH = "xxxx"


def _plot_ax(ax: plt.Axes, wl_df: pd.DataFrame, workload: str) -> list[mpatches.Patch]:
    pg_pct = _bin_slowdowns(wl_df["pg_time"] / wl_df["true_time"])
    bespoke_pct = _bin_slowdowns(wl_df["bespoke_time"] / wl_df["true_time"])

    x = np.arange(len(_BIN_LABELS))
    width = 0.35

    ax.bar(
        x - width / 2,
        pg_pct,
        width,
        label="PG Card",
        color=_COLORS["pg"][0],
        edgecolor=_COLORS["pg"][1],
        linewidth=1.5,
    )
    ax.bar(
        x + width / 2,
        bespoke_pct,
        width,
        label="Bespoke Card",
        color=_COLORS["bespoke"][0],
        edgecolor=_COLORS["bespoke"][1],
        linewidth=1.5,
        alpha=0.95,
    )

    # Overlay hatches on the excess portion of the larger bar per bin.
    # Bins with lower-bound >= 1.1 are "bad" slowdowns → red hatches; <1.1 → green hatches.
    for i, lo in enumerate(_BINS[:-1]):
        highlight = _RED if lo >= 1.1 else _GREEN
        pg_val, bespoke_val = pg_pct[i], bespoke_pct[i]
        lo_val, hi_val = min(pg_val, bespoke_val), max(pg_val, bespoke_val)
        excess = hi_val - lo_val
        if excess < 1e-9:
            continue
        bar_x = (x[i] - width / 2) if pg_val > bespoke_val else (x[i] + width / 2)
        ax.bar(
            bar_x,
            excess,
            width,
            bottom=lo_val,
            facecolor="none",
            edgecolor=highlight,
            linewidth=0,
            hatch=_HATCH,
            zorder=3,
        )

    # Annotate each pair with the signed delta (bespoke − pg).
    # >1.1 bins: up = bad (red), down = good (green). <1.1: opposite.
    y_max = max(max(pg_pct), max(bespoke_pct))
    for i, lo in enumerate(_BINS[:-1]):
        pg_val, bespoke_val = pg_pct[i], bespoke_pct[i]
        if pg_val < 1e-9:
            continue
        rel = (bespoke_val - pg_val) / pg_val * 100
        if abs(rel) < 1e-9:
            continue
        going_up = rel > 0
        color = _GREEN if going_up else _RED
        sign = "+" if going_up else ""
        ax.text(
            x[i],
            max(pg_val, bespoke_val) + y_max * 0.03,
            f"{sign}{rel:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    ax.set_title(workload, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(_BIN_LABELS, fontsize=10, fontweight="bold")
    ax.set_xlabel(
        "Slowdown w.r.t. True Cardinalities",
        fontsize=12,
        fontweight="bold",
        labelpad=8,
    )
    ax.set_ylabel("Queries (%)", fontsize=12, fontweight="bold", labelpad=8)
    ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return [
        mpatches.Patch(
            facecolor="none",
            edgecolor=_RED,
            hatch=_HATCH,
            label="Queries getting faster",
        ),
        mpatches.Patch(
            facecolor="none",
            edgecolor=_GREEN,
            hatch=_HATCH,
            label="Queries improved by bespoke",
        ),
    ]


def plot_slowdown_distribution(df: pd.DataFrame, out_folder: Path) -> None:
    for machine, machine_df in df.groupby("host"):
        workloads = sorted(machine_df["workload"].unique())
        n = len(workloads)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 3), squeeze=False)

        hatch_handles = None
        for ax, workload in zip(axes[0], workloads):
            wl_df = machine_df[machine_df["workload"] == workload]
            hatch_handles = _plot_ax(ax, wl_df, workload)
        assert hatch_handles is not None

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            frameon=False,
            fontsize=11,
            loc="lower center",
            ncols=2,
            bbox_to_anchor=(0.5, -0.12),
        )

        plt.tight_layout()
        plt.savefig(
            out_folder / f"slowdown_distribution_{machine}.pdf",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
