import json
import os
import shutil

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import numpy as np
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import dotenv

dotenv.load_dotenv()

sns.set_theme(
    style="whitegrid",
    palette="muted",
    rc={
        "patch.edgecolor": "none",
        "patch.linewidth": 0,
        "axes.grid.axis": "y",
        "axes.grid.which": "major",
    },
)


def create_plots_directories():
    """Creates directories for plots if they don't exist."""
    if os.path.exists("plots"):
        shutil.rmtree("plots")
    os.makedirs("plots")
    os.makedirs("plots/distributions")
    os.makedirs("plots/boxplots")
    os.makedirs("plots/regressions")
    os.makedirs("plots/optimization")


def q_error_plots(
    q_errors: list[float],
    q_errors_pg: list[float],
    # grouped: bool = True,
    all: bool = False,
) -> None:
    """Plots the Q-Error distribution."""

    if all:
        # sample 10% of queries to make visualization possible
        np.random.seed(42)
        sample_size = max(1, len(q_errors) // 10)
        sampled_indices = np.random.choice(
            len(q_errors), size=sample_size, replace=False
        )
        q_errors = [q_errors[i] for i in sampled_indices]
        q_errors_pg = [q_errors_pg[i] for i in sampled_indices]

    plt.figure(figsize=(16, 6))
    for i, grouped in enumerate([True, False]):
        xlabel = "10% of all logical plans" if all else "Root node logical plans"
        if grouped:
            # sort only pg q_errors and rearrange ours accordingly
            sorted_indices = sorted(
                range(len(q_errors_pg)), key=lambda i: q_errors_pg[i]
            )
            q_errors_d = [q_errors[i] for i in sorted_indices]
            q_errors_pg_d = [q_errors_pg[i] for i in sorted_indices]

            xlabel += " (sorted by Postgres Q-Error)"
        else:
            # sort both individually
            q_errors_d = sorted(q_errors)
            q_errors_pg_d = sorted(q_errors_pg)
            xlabel += " (sorted individually)"

        plt.subplot(1, 2, i + 1)
        x = np.arange(len(q_errors))  # JOB queries
        width = 0.4  # width of each bar
        plt.bar(
            x - width / 2,
            q_errors_d,
            width=width,
            color="steelblue",
            alpha=0.7,
            label="Bespoke Estimator",
        )
        plt.bar(
            x + width / 2,
            q_errors_pg_d,
            width=width,
            color="darkred",
            alpha=0.7,
            label="Postgres Estimator",
        )
        plt.yscale("log")
        plt.xlabel(xlabel)
        plt.ylabel("Q-Error")
        plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"plots/distributions/q_error_distributions_{'all' if all else 'root'}.pdf"
    )

    # plot one more distribution where we show the improvement / degradation of q error by dividing
    diff1 = [q_pg / q for q, q_pg in zip(q_errors, q_errors_pg)]
    diff2 = [q / q_pg for q, q_pg in zip(q_errors, q_errors_pg)]
    diff = [
        d1 if d1 >= 1 else -d2 for d1, d2 in zip(diff1, diff2)
    ]  # positive means improvement, negative means degradation

    plt.figure(figsize=(12, 6))
    plt.bar(
        width=0.7,
        x=range(0, len(q_errors)),
        height=(sorted(diff)),
        color=["#2ecc70bf" if d >= 1 else "#e74d3cbc" for d in sorted(diff)],
        label="Improvement in Q-Error",
    )

    plt.yscale("symlog", linthresh=1)
    plt.xlabel("10% of all plans" if all else "Root node plans")
    plt.ylabel("Improvement Factor (higher is better)")
    plt.tight_layout()

    plt.savefig(
        f"plots/distributions/q_error_improvement_{'all' if all else 'root'}.pdf"
    )
    plt.close()


def boxplots(
    q_errors_per_category: dict[int, dict[str, list]],
    category: str,
    over_under: bool = False,
):
    """Plots boxplot of Q-Errors per number of category in one plot. If over_under is False, plots absolute q errors. Else, separates between over- and underestimation."""

    # per join, two boxplots side by side, x axis is num_joins, y axis is q error (log scale) all in one plot
    plt.figure(figsize=(12, 6))
    keys = sorted(q_errors_per_category.keys())
    bespoke_data = [q_errors_per_category[k]["bespoke"] for k in keys]
    pg_data = [q_errors_per_category[k]["postgres"] for k in keys]
    box_width = 0.4
    positions_bespoke = [k - box_width / 2 for k in keys]
    positions_pg = [k + box_width / 2 for k in keys]
    plt.boxplot(
        bespoke_data,
        positions=positions_bespoke,
        widths=box_width,
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.5),
        medianprops=dict(color="black"),
        labels=[str(k) for k in keys],
    )
    plt.boxplot(
        pg_data,
        positions=positions_pg,
        widths=box_width,
        patch_artist=True,
        boxprops=dict(facecolor="darkred", alpha=0.7),
        medianprops=dict(color="black"),
        labels=[str(k) for k in keys],
    )
    plt.yscale("log")
    plt.xlabel(f"Number of {category}")
    plt.xticks(keys)
    legend_handles = [
        mpatches.Patch(facecolor="steelblue", alpha=0.5, label="Bespoke Estimator"),
        mpatches.Patch(facecolor="darkred", alpha=0.7, label="Postgres Estimator"),
    ]
    plt.legend(handles=legend_handles)

    if over_under:
        plt.ylabel(
            r"$\leftarrow$ underestimation  Q-Error  overestimation $\rightarrow$"
        )
        plt.axhline(1, color="#34495e", linewidth=1.2, zorder=1, linestyle="--")
        # determine ticks for log scale that are symmetric around 1
        max_q_error = max(
            max(
                q_errors_per_category[k]["bespoke"]
                + q_errors_per_category[k]["postgres"]
            )
            for k in keys
        )
        min_q_error = min(
            min(
                q_errors_per_category[k]["bespoke"]
                + q_errors_per_category[k]["postgres"]
            )
            for k in keys
        )
        max_q_error = max(max_q_error, 1 / min_q_error)

        ticks = [
            1 / max_q_error,
            1 / math.sqrt(max_q_error),
            1,
            math.sqrt(max_q_error),
            max_q_error,
        ]
        tick_labels = [
            f"$10^{{{int(math.log10(max_q_error))}}}$",
            f"$10^{{{int(math.log10(math.sqrt(max_q_error)))}}}$",
            "1",
            f"$10^{{{int(math.log10(math.sqrt(max_q_error)))}}}$",
            f"$10^{{{int(math.log10(max_q_error))}}}$",
        ]
        # ticks = [1e4, 1e2, 1, 1e-2, 1e-4]
        # tick_labels = ["$10^4$", "$10^2$", "1", "$10^2$", "$10^4$"]
        plt.yticks(ticks, tick_labels)
        plt.tight_layout()
        plt.savefig(f"plots/boxplots/q_error_per_num_{category.lower()}_rel.pdf")
    else:
        plt.ylabel("Q-Error")
        plt.tight_layout()
        plt.savefig(f"plots/boxplots/q_error_per_num_{category.lower()}_abs.pdf")
    plt.close()


def regression_plot(
    regression_count: dict[str, int],
    total_counts: dict[str, int],
    x_label: str,
    x_values: list,
    tag: str,
):
    """Plot relative and absolute overlay bar charts of regression count and total counts per key."""
    x = sorted(regression_count.keys())

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)

    plt.bar(
        x_values,
        [total_counts[key] for key in x],
        color="steelblue",
        alpha=0.5,
        label="Occurences",
    )
    plt.bar(
        x_values,
        [regression_count[key] for key in x],
        color="darkred",
        alpha=0.7,
        label="Regressions",
    )
    plt.xlabel(x_label)
    if len(x_values) < 20:
        plt.xticks(x_values)
    if max([len(str(key)) for key in x_values]) > 3:
        plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    regression_rates = [
        regression_count[key] / total_counts[key] if total_counts[key] > 0 else 0
        for key in x
    ]
    cmap = cm.get_cmap("YlOrRd")  # RdYlGn_r
    norm = mcolors.Normalize(vmin=0, vmax=1)
    bar_colors = [cmap(norm(rate)) for rate in regression_rates]

    plt.bar(x_values, regression_rates, color=bar_colors, alpha=0.7)
    plt.xlabel(x_label)
    if len(x_values) < 20:
        plt.xticks(x_values)
    if max([len(str(key)) for key in x_values]) > 3:
        plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.1)
    plt.ylabel("Regression Rate")

    plt.tight_layout()
    plt.savefig(f"plots/regressions/regression_rate_per_{tag}.pdf")
    plt.close()


def regression_plot_vertical(
    regression_count: dict[str, int],
    total_counts: dict[str, int],
    x_label: str,
    x_values: list,
    tag: str,
):
    """Plot relative and absolute overlay bar charts of regression count and total counts per key."""
    x = sorted(regression_count.keys())

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)

    plt.bar(
        x_values,
        [total_counts[key] for key in x],
        color="steelblue",
        alpha=0.5,
        label="Occurences",
    )
    plt.bar(
        x_values,
        [regression_count[key] for key in x],
        color="darkred",
        alpha=0.7,
        label="Regressions",
    )
    plt.xlabel("")
    plt.xticks([])
    plt.legend()
    plt.ylabel("Count")

    plt.subplot(2, 1, 2)
    regression_rates = [
        regression_count[key] / total_counts[key] if total_counts[key] > 0 else 0
        for key in x
    ]
    cmap = cm.get_cmap("YlOrRd")  # RdYlGn_r
    norm = mcolors.Normalize(vmin=0, vmax=1)
    bar_colors = [cmap(norm(rate)) for rate in regression_rates]

    plt.bar(x_values, regression_rates, color=bar_colors, alpha=0.7)
    plt.xlabel(x_label)
    if len(x_values) < 20:
        plt.xticks(x_values)
    if max([len(str(key)) for key in x_values]) > 3:
        plt.xticks(rotation=90, fontsize=10)
    plt.ylim(0, 1.1)
    plt.ylabel("Regression Rate")

    plt.tight_layout()
    plt.savefig(f"plots/regressions/regression_rate_per_{tag}.pdf")
    plt.close()


def hierarchical_regression_plot(tables_stats, tag):
    # Flatten data for plotting
    plot_labels = []
    plot_rates = []
    counts = []
    regressions = []
    group_boundaries = []
    current_idx = 0

    for t_name, t_data in sorted(tables_stats.items()):
        start_idx = current_idx
        for c_name, c_stats in sorted(t_data["cols"].items()):
            cnt = c_stats["counts"]
            reg = c_stats["regressions"]
            rate = reg / cnt if cnt > 0 else 0

            plot_labels.append(c_name)
            plot_rates.append(rate)
            counts.append(cnt)
            regressions.append(reg)

            current_idx += 1
        group_boundaries.append((t_name, start_idx, current_idx - 1))

    # Setup Colors
    cmap = cm.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=0, vmax=1)
    colors = [cmap(norm(r)) for r in plot_rates]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )

    # -------------------------
    # TOP PLOT: Overlaid Counts + Regressions
    # -------------------------
    x = range(len(plot_labels))

    ax_top.bar(x, counts, color="steelblue", alpha=0.5, label="Occurences")

    ax_top.bar(x, regressions, color="darkred", alpha=0.7, label="Regressions")

    ax_top.set_ylabel("Counts", fontweight="bold")
    ax_top.legend()

    # -------------------------
    # BOTTOM PLOT: Regression Rate
    # -------------------------
    bars = ax_bottom.bar(x, plot_rates, color=colors, alpha=0.7)

    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(plot_labels, rotation=90, fontsize=10)
    ax_bottom.set_ylabel("Regression Rate", fontweight="bold")
    ax_bottom.set_ylim(0, 1.1)

    # Table labels under x-axis
    for i, (t_name, start, end) in enumerate(group_boundaries):
        midpoint = (start + end) / 2
        y_pos = -0.25 if i % 2 == 1 else -0.3

        ax_bottom.text(
            midpoint,
            y_pos,
            t_name,
            transform=ax_bottom.get_xaxis_transform(),
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=10,
            color="#333333",
            clip_on=False,
        )

        if end < len(plot_rates) - 1:
            ax_bottom.axvline(
                end + 0.5,
                color="gray",
                linestyle="-",
                alpha=0.2,
                ymin=-0.15,
                clip_on=False,
            )
            ax_top.axvline(
                end + 0.5,
                color="gray",
                linestyle="-",
                alpha=0.2,
                ymin=-0.15,
                clip_on=False,
            )

    plt.tight_layout()
    plt.savefig(f"plots/regressions/regression_rate_per_col.pdf")
    plt.close()


def plot_optimization_progress(
    q_error_percentiles: dict[str, list[int]], pg_percentiles: dict[str, int]
):
    # Sort percentiles to ensure they map consistently to the grid
    percentiles = sorted(q_error_percentiles.keys())
    rounds = range(1, len(q_error_percentiles[percentiles[0]]) + 1)

    # Create a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()  # Flatten to iterate easily

    for i, p in enumerate(percentiles):
        if i >= 4:
            break  # Cap at 4 for a 2x2 grid

        ax = axes_flat[i]

        # Plot the optimization progress
        ax.plot(
            rounds,
            q_error_percentiles[p],
            marker="o",
            markersize=4,
            label=f"Bespoke ({p}th)",
        )

        # Plot the Postgres baseline
        ax.axhline(
            y=pg_percentiles[p],
            color="darkred",
            linestyle="--",
            label=f"Postgres ({p}th)",
        )

        # Formatting individual subplot
        ax.set_yscale("log")
        ax.set_title(f"{p}th Percentile Q-Error")
        ax.set_xlabel("Optimization Round")
        ax.set_ylabel("Q-Error")
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()

    # Remove any empty subplots if percentiles < 4
    for j in range(i + 1, 4):
        fig.delaxes(axes_flat[j])

    plt.suptitle("Optimization Progress vs. Postgres Baseline", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
    plt.savefig(f"plots/optimization/optimization_progress.pdf")
    plt.close()


def optimization_plot_helper(path_to_logs: str):
    q_error_percentiles = {}
    pg_percentiles = {}
    for p in [50, 90, 95, 99]:
        q_error_percentiles[p] = []
        pg_percentiles[p] = 0
    # count interations by counting iteration folders in logs
    rounds = len(
        [
            folder
            for folder in os.listdir(path_to_logs)
            if folder.startswith("iteration_")
        ]
    )
    # read md files from logs and extract q error percentiles
    for round in range(rounds):
        with open(f"{path_to_logs}/iteration_{round}/feedback.json", "r") as f:
            feedback = json.load(f)
            # extract q error percentiles from content
            q_errors = feedback.get("q_error_percentiles", {})
            if q_error_percentiles == {}:
                continue
            else:
                bespoke_errors = q_errors.get("bespoke", {})
                bespoke_errors = {int(k[:2]): v for k, v in bespoke_errors.items()}
                pg_errors = q_errors.get("pg", {})
                pg_errors = {int(k[:2]): v for k, v in pg_errors.items()}
                for p in [50, 90, 95, 99]:
                    q_error_percentiles[p].append(bespoke_errors.get(p, None))
                    if (
                        pg_percentiles[p] == 0
                    ):  # only set once, should be the same across rounds
                        pg_percentiles[p] = pg_errors.get(p, None)

    return q_error_percentiles, pg_percentiles


def plot_execution_times():
    with open("outputs/end_to_end_results.json", "r") as f:
        results = json.load(f)

    pg_times = [r["pg_time"] for r in results.values()]
    bespoke_times = [r["bespoke_time"] for r in results.values()]
    true_times = [r["true_time"] for r in results.values()]

    # plot execution times, 3 bars per query
    # sort by pg times and rearrange others accordingly
    sorted_indices = sorted(range(len(pg_times)), key=lambda i: pg_times[i])
    pg_times = [pg_times[i] for i in sorted_indices]
    bespoke_times = [bespoke_times[i] for i in sorted_indices]
    true_times = [true_times[i] for i in sorted_indices]

    x = np.arange(len(pg_times))  # Use numpy for easier coordinate math
    width = 0.25  # Slightly narrower to create breathing room between groups

    # Use a modern style context
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 7))

    # Plotting with explicit offsets
    rects1 = ax.bar(x - width, pg_times, width, label="PostgreSQL", color="darkred")
    rects2 = ax.bar(x, bespoke_times, width, label="Bespoke Hints", color="steelblue")
    rects3 = ax.bar(x + width, true_times, width, label="True Hints", color="gold")

    # ax.set_yscale("log")
    # add red, dotted vertical line indicating timeout threshold
    timeout_threshold = int(os.getenv("end2end_timeout"))
    ax.axhline(
        y=timeout_threshold,
        color="darkred",
        linestyle="--",
        label=f"Timeout Threshold",
        # position the label above the line on the right side
    )

    # Labeling
    ax.set_xlabel("Queries (Sorted by PostgreSQL latency)", fontsize=12, labelpad=10)
    ax.set_ylabel("Execution Time (seconds)", fontsize=12, labelpad=10)

    ax.grid(True, which="both", ls="-", alpha=0.2)  # Show minor lines for log scale
    ax.set_axisbelow(True)  # Put grid behind bars

    ax.legend(frameon=True, facecolor="white", loc="upper left", fontsize=11)

    plt.tight_layout()
    plt.savefig("plots/e2e_execution_times.pdf", dpi=300)
