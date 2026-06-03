from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.paper_plots.utils import _COLORS


def plot_total_runtime_by_machine(df: pd.DataFrame, out_folder: Path) -> None:
    labels = ["PG Card", "Bespoke Card", "Act Card"]
    keys = ["pg_time", "bespoke_time", "true_time"]
    face_colors = [_COLORS["pg"][0], _COLORS["bespoke"][0], _COLORS["true"][0]]
    edge_colors = [_COLORS["pg"][1], _COLORS["bespoke"][1], _COLORS["true"][1]]

    for machine, machine_df in df.groupby("host"):
        workloads = sorted(machine_df["workload"].unique())
        n = len(workloads)
        fig, axes = plt.subplots(1, n, figsize=(3.35 * n, 3), squeeze=False)
        # fig.suptitle(
        #     f"Total Runtime — Machine: {machine}", fontsize=12, fontweight="bold"
        # )

        for i, (ax, workload) in enumerate(zip(axes[0], workloads)):
            wl_df = machine_df[machine_df["workload"] == workload]
            times = [wl_df[c].sum() for c in keys]
            x = np.arange(len(labels))
            for xi, (h, fc, ec) in enumerate(zip(times, face_colors, edge_colors)):
                ax.bar(xi, h, width=0.55, color=fc, edgecolor=ec, linewidth=2.0)
                ax.text(
                    xi,
                    h * 0.5,
                    f"{h:.0f}s",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                )
            ax.set_title(workload, fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
            if i == 0:
                ax.set_ylabel(
                    r"$\bf{Postgres\ Execution\ Time}$" "\n(seconds)",
                    fontsize=11,
                )
            ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Diagonal arrows showing runtime reduction (top-right of left bar → top-left of right bar)
            t_pg, t_bespoke, t_act = times
            arrow_color = "#FF0048"
            arrow_color = "#201F1F"
            half_w = 0.55 / 2

            def draw_reduction_arrow(ax, x0, y0, x1, y1, reduction):
                ax.annotate(
                    "",
                    xy=(x1 - half_w + 0.03, y1),
                    xytext=(x0 + half_w + 0.03, y0),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=arrow_color,
                        lw=1.5,
                    ),
                )
                ax.text(
                    (x0 + half_w + x1 - half_w) / 2 + 0.45,
                    (y0 + y1) / 2,
                    f"-{reduction:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    color=arrow_color,
                    fontweight="bold",
                )

            reduction_bespoke = (t_pg - t_bespoke) / t_pg
            draw_reduction_arrow(ax, 0, t_pg, 1, t_bespoke, reduction_bespoke)

            reduction_act = (t_bespoke - t_act) / t_pg
            draw_reduction_arrow(ax, 1, t_bespoke, 2, t_act, reduction_act)

            ax.set_ylim(top=max(times) * 1.15)

        plt.tight_layout()
        out_path = out_folder / f"total_runtime_{machine}.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def plot_per_query_exec_times(
    df: pd.DataFrame,
    timeout_threshold: int | None,
    out_folder: Path,
) -> None:
    assert isinstance(df, pd.DataFrame), (
        f"Input must be a pandas DataFrame: got {type(df)}"
    )
    for machine, machine_df in df.groupby("host"):
        workloads = sorted(machine_df["workload"].unique())
        query_counts = [
            max(1, len(machine_df[machine_df["workload"] == workload]))
            for workload in workloads
        ]
        width_ratios = query_counts
        if len(query_counts) == 2:
            total_queries = sum(query_counts)
            width_ratios = [
                min(0.65, max(0.35, count / total_queries)) for count in query_counts
            ]
        n = len(workloads)
        fig, axes = plt.subplots(
            1,
            n,
            figsize=(5 * n, 3.5),
            squeeze=False,
            gridspec_kw={"width_ratios": width_ratios, "wspace": 0.08},
        )

        for i, (ax, workload) in enumerate(zip(axes[0], workloads)):
            wl_df = machine_df[machine_df["workload"] == workload].sort_values(
                by="pg_time"
            )
            pg_times = wl_df["pg_time"].tolist()
            bespoke_times = wl_df["bespoke_time"].tolist()
            true_times = wl_df["true_time"].tolist()

            x = np.arange(len(pg_times))
            width = 0.2
            edge_linewidth = 1

            ax.bar(
                x - width,
                pg_times,
                width,
                label="PG Card",
                color=_COLORS["pg"][0],
                edgecolor=_COLORS["pg"][1],
                linewidth=edge_linewidth,
            )
            ax.bar(
                x,
                bespoke_times,
                width,
                label="Bespoke Card",
                color=_COLORS["bespoke"][0],
                edgecolor=_COLORS["bespoke"][1],
                linewidth=edge_linewidth,
                alpha=0.95,
            )
            ax.bar(
                x + width,
                true_times,
                width,
                label="Act Card",
                color=_COLORS["true"][0],
                edgecolor=_COLORS["true"][1],
                linewidth=edge_linewidth,
                alpha=0.95,
            )

            if timeout_threshold is not None:
                ax.axhline(
                    y=timeout_threshold,
                    color="#CC3333",
                    linestyle="--",
                    linewidth=1.2,
                    label="Timeout Threshold",
                )

            ax.set_title(workload, fontsize=11, fontweight="bold")
            ax.set_xlabel(
                "Queries (sorted by PostgreSQL latency)",
                fontsize=12,
                fontweight="bold",
                labelpad=8,
            )
            if i == 0:
                ax.set_ylabel(
                    r"$\bf{Postgres\ Execution\ Time}$" "\n(seconds)",
                    fontsize=11,
                )
            ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            frameon=False,
            fontsize=11,
            loc="lower center",
            ncols=len(labels),
            bbox_to_anchor=(0.5, -0.08),
        )

        fig.subplots_adjust(wspace=0.08, bottom=0.25)
        plt.savefig(
            out_folder / f"per_query_exec_times_{machine}.pdf",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
