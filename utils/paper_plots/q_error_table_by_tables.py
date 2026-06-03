from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.paper_plots.utils import _COLORS

_PERCENTILES = [50, 90, 95, 100]
_PG_LIGHT = _COLORS["pg"][0]
_PG_DARK = _COLORS["pg"][1]
_BS_LIGHT = _COLORS["bespoke"][0]
_BS_DARK = _COLORS["bespoke"][1]


def _render_table_on_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str = "",
    percentiles: list[int] = _PERCENTILES,
) -> None:
    pg_errors = df[df["estimator"] == "postgres"]["q_error"].to_numpy(dtype=float)
    bs_errors = df[df["estimator"] == "bespoke"]["q_error"].to_numpy(dtype=float)

    col_labels = [f"q{p}" if p < 100 else "max" for p in percentiles]

    pg_vals = [np.percentile(pg_errors, p) for p in percentiles]
    bs_vals = [np.percentile(bs_errors, p) for p in percentiles]

    cell_text = [
        [f"{v:.2f}" for v in pg_vals],
        [f"{v:.2f}" for v in bs_vals],
    ]
    cell_colors = [
        [_PG_LIGHT + "55"] * len(percentiles),
        [_BS_LIGHT + "55"] * len(percentiles),
    ]

    ax.set_axis_off()

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=["PostgreSQL", "Bespoke"],
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 2.2)

    for j in range(len(percentiles)):
        cell = tbl[0, j]
        cell.set_facecolor("#444444")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    tbl[1, -1].set_facecolor(_PG_DARK)
    tbl[1, -1].set_text_props(color="white", fontweight="bold")
    tbl[1, -1].set_edgecolor("white")
    tbl[2, -1].set_facecolor(_BS_DARK)
    tbl[2, -1].set_text_props(color="white", fontweight="bold")
    tbl[2, -1].set_edgecolor("white")

    for j in range(len(percentiles)):
        tbl[1, j].set_edgecolor("white")
        tbl[2, j].set_edgecolor("white")
        if bs_vals[j] <= pg_vals[j]:
            tbl[2, j].set_text_props(fontweight="bold")
        else:
            tbl[1, j].set_text_props(fontweight="bold")


def _plot_table(
    df: pd.DataFrame,
    base_title: str,
    out_path: Path,
    percentiles: list[int] = _PERCENTILES,
) -> None:
    workloads = df["workload"].unique().tolist() if "workload" in df.columns else [None]
    n = len(workloads)

    fig, axes = plt.subplots(1, n, figsize=(len(percentiles) * 1.6 * n + 1.4, 2.2))
    if n == 1:
        axes = [axes]

    for wl, ax in zip(workloads, axes):
        sub = df[df["workload"] == wl] if wl is not None else df
        title = wl if wl is not None else base_title
        _render_table_on_ax(ax, sub, title=title, percentiles=percentiles)

    if n == 1:
        fig.text(
            0.5, 0.97, base_title, ha="center", va="top", fontsize=11, fontweight="bold"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.92])
    else:
        fig.suptitle(base_title, fontsize=11, fontweight="bold", y=1.02)
        plt.tight_layout()

    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_q_error_table_by_tables(df: pd.DataFrame, out_folder: Path) -> None:
    sub = df[df["num_tables"] == 1]
    _plot_table(
        sub,
        base_title="Q-Error  [max(est/act, act/est)]  —  single-table subplans",
        out_path=out_folder / "q_error_table_by_tables.pdf",
    )


def plot_q_error_table_all(df: pd.DataFrame, out_folder: Path) -> None:
    _plot_table(
        df,
        base_title="Q-Error  [max(est/act, act/est)]  —  all subplans",
        out_path=out_folder / "q_error_table_all.pdf",
        percentiles=[5, 25, 50, 90, 95, 100],
    )
