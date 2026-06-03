from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.paper_plots.utils import _COLORS

_PG_DARK = _COLORS["pg"][1]
_BS_DARK = _COLORS["bespoke"][1]


def _render_distribution_on_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_ylabel: bool = True,
) -> None:
    pg_errors = df[df["estimator"] == "postgres"]["q_error"].to_numpy(dtype=float)
    bs_errors = df[df["estimator"] == "bespoke"]["q_error"].to_numpy(dtype=float)

    fractions = np.linspace(0, 1, 1000)
    pg_quantiles = np.quantile(pg_errors, fractions)
    bs_quantiles = np.quantile(bs_errors, fractions)

    ax.fill_between(pg_quantiles, fractions, 0, color=_PG_DARK, alpha=0.10)
    ax.fill_between(bs_quantiles, fractions, 0, color=_BS_DARK, alpha=0.10)
    ax.plot(pg_quantiles, fractions, color=_PG_DARK, linewidth=2.4)
    ax.plot(bs_quantiles, fractions, color=_BS_DARK, linewidth=2.4)

    ax.set_xscale("log")
    ax.set_xlim(left=1.0)
    ax.set_ylim(0, 1)

    # Direct curve labels: left curve labeled on the left, right curve on the right
    # Use per-curve fractions so each label sits where the curve is away from the axes
    curve_label_idxs = {
        "pg": 700,
        "bs": 880,
    }
    pg_idx, bs_idx = curve_label_idxs["pg"], curve_label_idxs["bs"]
    if pg_quantiles[pg_idx] >= bs_quantiles[bs_idx]:
        pg_ha, pg_xoff = "left", 5
        bs_ha, bs_xoff = "right", -5
    else:
        pg_ha, pg_xoff = "right", -5
        bs_ha, bs_xoff = "left", 5
    for quantiles, color, name, ha, xoff, idx, ma in [
        (pg_quantiles, _PG_DARK, "PostgreSQL", pg_ha, pg_xoff, pg_idx, pg_ha),
        (bs_quantiles, _BS_DARK, "Bespoke-\nCard", bs_ha, bs_xoff, bs_idx, "center"),
    ]:
        ax.annotate(
            name,
            xy=(quantiles[idx], fractions[idx]),
            xytext=(xoff, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            fontweight="bold",
            ha=ha,
            va="center",
            multialignment=ma,
            clip_on=False,
        )

    ax.set_xlabel("Q-Error", fontsize=11, fontweight="bold", labelpad=6)
    if show_ylabel:
        ax.set_ylabel(
            r"$\bf{Fraction\ of\ subplans}$" + "\n(cumulative)",
            fontsize=11,
            labelpad=6,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_distribution(df: pd.DataFrame, out_path: Path) -> None:
    workloads = df["workload"].unique().tolist() if "workload" in df.columns else [None]
    n = len(workloads)

    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 2.8), sharey=(n > 1))
    if n == 1:
        axes = [axes]

    for i, (wl, ax) in enumerate(zip(workloads, axes)):
        sub = df[df["workload"] == wl] if wl is not None else df
        _render_distribution_on_ax(ax, sub, show_ylabel=(i == 0))
        if wl is not None:
            ax.set_title(wl, fontsize=11, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_q_error_distribution_base(df: pd.DataFrame, out_folder: Path) -> None:
    sub = df[df["num_tables"] == 1]
    _plot_distribution(sub, out_folder / "q_error_distribution_base.pdf")


def plot_q_error_distribution_all(df: pd.DataFrame, out_folder: Path) -> None:
    _plot_distribution(df, out_folder / "q_error_distribution_all.pdf")
