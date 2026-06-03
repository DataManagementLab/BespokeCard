from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.paper_plots.utils import _COLORS


def _signed_ratio(q_error: pd.Series, over_estimate: pd.Series) -> np.ndarray:
    """Return est/true: >1 = overestimate, <1 = underestimate, 1 = exact."""
    ratio = np.where(over_estimate, q_error, 1.0 / q_error)
    return ratio.astype(float)


def _render_boxplot_on_ax(
    ax: plt.Axes, df: pd.DataFrame, show_ylabel: bool = True
) -> None:
    df = df.copy()
    df["ratio"] = _signed_ratio(df["q_error"], df["over_estimate"])

    table_counts = sorted(df["num_tables"].unique())
    n_groups = len(table_counts)

    bespoke_data = [
        df[(df["num_tables"] == n) & (df["estimator"] == "bespoke")]["ratio"].values
        for n in table_counts
    ]
    pg_data = [
        df[(df["num_tables"] == n) & (df["estimator"] == "postgres")]["ratio"].values
        for n in table_counts
    ]

    positions_pg = [i * 3 - 0.6 for i in range(n_groups)]
    positions_bespoke = [i * 3 + 0.6 for i in range(n_groups)]

    rng = np.random.default_rng(0)

    def draw_boxes(data, positions, face_color, edge_color, label):
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.9,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            boxprops=dict(facecolor=face_color, edgecolor=edge_color, linewidth=1.5),
            medianprops=dict(color=edge_color, linewidth=2.0),
            whiskerprops=dict(color=edge_color, linewidth=1.5, linestyle="--"),
            capprops=dict(color=edge_color, linewidth=1.5),
        )
        bp["boxes"][0].set_label(label)

        for arr, pos in zip(data, positions):
            arr = np.asarray(arr, dtype=float)
            p5, p95 = np.percentile(arr, [5, 95])
            outliers = arr[(arr < p5) | (arr > p95)]
            if len(outliers) > 50:
                outliers = rng.choice(outliers, 50, replace=False)
            if len(outliers):
                ax.scatter(
                    np.full(len(outliers), pos),
                    outliers,
                    s=6,
                    color=face_color,
                    edgecolors=edge_color,
                    linewidths=0.5,
                    alpha=0.5,
                    zorder=3,
                )

        return bp

    draw_boxes(
        pg_data, positions_pg, _COLORS["pg"][0], _COLORS["pg"][1], "PostgreSQL Card-Est"
    )
    draw_boxes(
        bespoke_data,
        positions_bespoke,
        _COLORS["bespoke"][0],
        _COLORS["bespoke"][1],
        "Bespoke Card-Est",
    )

    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e8)
    ax.axhline(y=1.0, color="#555555", linestyle="--", linewidth=1.0, zorder=0)

    from matplotlib.ticker import FixedLocator, FuncFormatter

    ax.yaxis.set_major_locator(FixedLocator([10**k for k in range(-8, 9, 2)]))

    def _fmt_tick(x, _):
        if x <= 0:
            return ""
        val = x if x >= 1 else 1.0 / x
        exp = round(np.log10(val))
        return "1" if exp == 0 else f"1e{exp}"

    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_tick))

    ax.axhspan(1.0, 1e9, color="#FF4444", alpha=0.04, zorder=0)
    ax.axhspan(1e-9, 1.0, color="#2288FF", alpha=0.04, zorder=0)

    ax.text(
        ax.get_xlim()[1],
        1.0 * 1.05,
        "over",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#AA2222",
    )
    ax.text(
        ax.get_xlim()[1],
        1.0 * 0.95,
        "under",
        ha="right",
        va="top",
        fontsize=8,
        color="#1155AA",
    )

    tick_positions = [i * 3 for i in range(n_groups)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(n) for n in table_counts], fontsize=10, fontweight="bold")
    ax.set_xlim(tick_positions[0] - 2, tick_positions[-1] + 2)

    ax.set_xlabel("Number of Tables", fontsize=12, fontweight="bold", labelpad=8)
    if show_ylabel:
        ax.set_ylabel(
            "← underestimation   1   overestimation →",
            fontsize=9,
            labelpad=8,
        )

    ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_error_boxplot_by_tables(df: pd.DataFrame, out_folder: Path) -> None:
    workloads = df["workload"].unique().tolist() if "workload" in df.columns else [None]
    n = len(workloads)

    fig, axes = plt.subplots(1, n, figsize=(10 * n, 4), sharey=(n > 1))
    if n == 1:
        axes = [axes]

    for i, (wl, ax) in enumerate(zip(workloads, axes)):
        sub = df[df["workload"] == wl] if wl is not None else df
        _render_boxplot_on_ax(ax, sub, show_ylabel=(i == 0))
        if wl is not None:
            ax.set_title(wl, fontsize=12, fontweight="bold", pad=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=11,
        loc="lower center",
        ncols=2,
        bbox_to_anchor=(0.5, 1.0),
        borderaxespad=0,
    )

    _draw_box_anatomy(axes[-1])

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = out_folder / "error_boxplot_by_tables.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_box_anatomy(ax: plt.Axes) -> None:
    """Small inset explaining box-plot anatomy, placed to the right of ax."""
    inset = ax.inset_axes([1.03, 0.25, 0.035, 0.60])
    inset.set_axis_off()

    q1, med, q3 = 0.25, 0.55, 0.75
    iqr = q3 - q1
    lo_whisker = q1 - 0.8 * iqr
    hi_whisker = q3 + 0.9 * iqr

    cx = 0.3
    bw = 0.28

    gray = "#555555"
    lw = 1.0

    inset.plot([cx, cx], [lo_whisker, q1], color=gray, lw=lw, linestyle="--")
    inset.plot([cx, cx], [q3, hi_whisker], color=gray, lw=lw, linestyle="--")
    for y in (lo_whisker, hi_whisker):
        inset.plot([cx - bw, cx + bw], [y, y], color=gray, lw=lw)
    from matplotlib.patches import FancyBboxPatch

    inset.add_patch(
        FancyBboxPatch(
            (cx - bw, q1),
            2 * bw,
            iqr,
            boxstyle="square,pad=0",
            facecolor="#DDDDDD",
            edgecolor=gray,
            linewidth=lw,
        )
    )
    inset.plot([cx - bw, cx + bw], [med, med], color=gray, lw=lw + 0.4)
    label_x = cx + bw + 0.05
    fs = 5.5
    lc = "#333333"

    def annotate(y, text):
        inset.annotate(
            text,
            xy=(cx + bw, y),
            xytext=(label_x, y),
            fontsize=fs,
            color=lc,
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.6),
        )

    annotate(hi_whisker, "95th percentile")
    annotate(q3, "75th percentile")
    annotate(med, "median")
    annotate(q1, "25th percentile")
    annotate(lo_whisker, "5th percentile")

    inset.set_xlim(cx - bw - 0.05, label_x + 0.8)
    inset.set_ylim(lo_whisker - 0.1, hi_whisker + 0.1)
