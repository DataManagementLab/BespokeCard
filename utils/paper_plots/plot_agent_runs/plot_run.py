"""Plots for analyzed runs. Reads CSVs produced by analyze_run.py and writes
PDFs into paper_plots/output/agent_plots/{workload}/.

Produces the following plots per workload (prefixed with workload name):
  1. {w}_q_error_progression.pdf  — q-error percentiles per iteration vs PG baseline
  2. {w}_cost_by_phase.pdf        — stacked cost split per agent x phase
  3. {w}_tool_usage.pdf           — coder tool call counts + runtimes
  4. {w}_iteration_activity.pdf   — coder turns / tool calls per iteration, with phase bands
  5. {w}_q_error_by_stage.pdf     — bespoke q-error percentiles at end of each stage
  6. {w}_calls_by_stage.pdf       — coder LLM turns and tool calls aggregated per stage

Style follows paper_plots/qo_plot.py and qo_runtimes.py conventions.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.paper_plots.utils import _COLORS, setup_plotting  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
OUT_BASE = Path(__file__).resolve().parents[1] / "output" / "agent_plots"


def find_log_dir(workload: str) -> Path:
    candidates = sorted((RESULTS_DIR / workload).glob("log_*"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No log_* directories in {RESULTS_DIR / workload}")
    return candidates[0]


WORKLOADS = {
    "job": find_log_dir("job"),
    "job-complex": find_log_dir("job-complex"),
}

WORKLOAD_LABEL = {
    "job": "JOB",
    "job-complex": "JOB-Complex",
}

# Color palette: phases/tools each get a (face, edge) tuple matching the
# (face, edge) convention used in paper_plots/utils._COLORS.
PHASE_COLORS = {
    "implement_estimator": ("#9ECAE1", "#3182BD"),
    "join_rounds": ("#A1D99B", "#31A354"),
    "filter_rounds": ("#FDAE6B", "#E6550D"),
    "final_rounds": ("#BCBDDC", "#6A51A3"),
    "create_plan": ("#9ECAE1", "#3182BD"),
    "coder_question": ("#A1D99B", "#31A354"),
    "identify_best": ("#BCBDDC", "#6A51A3"),
}
# (face, edge, hatch)
TOOL_COLORS = {
    "apply_patch": ("#9ECAE1", "#3182BD", "/"),
    "evaluate_estimator": ("#A1D99B", "#31A354", "\\"),
    "ask_agent": ("#FDAE6B", "#E6550D", "xx"),
    "shell": ("#BCBDDC", "#6A51A3", "//"),
    "query_db": ("#74C2C9", "#2E8B9A", ".."),
    "text response (no tool)": ("#D9D9D9", "#737373", ""),
}
TOOL_LABELS = {
    "apply_patch": "Apply Patch",
    "evaluate_estimator": "Run Generated Code",
    "ask_agent": "Ask Planner",
    "shell": "Shell",
    "query_db": "Query DB",
    "text response (no tool)": "Text Response",
}

EDGE_LW = 1.2


# ---------- helpers ----------


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def style_axes(ax) -> None:
    ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


STAGE_ORDER = ["implement_estimator", "join_rounds", "filter_rounds", "final_rounds"]
STAGE_LABEL = {
    "implement_estimator": "No\nFeedback",  # q_error_by_stage
    "join_rounds": "+ Join\nFeedback",
    "filter_rounds": "+ Filter\nFeedback",
    "final_rounds": "+ Full\nFeedback",
}

STAGE_LABEL_CALLS = {
    "implement_estimator": "Implement\n(Initial)",
    "join_rounds": "Join\nFeedback",
    "filter_rounds": "Filter\nFeedback",
    "final_rounds": "Full\nFeedback",
}


# ---------- plot 1: q-error progression per iteration ----------


def plot_q_error_progression(
    per_iter: list[dict], workload: str, out_dir: Path
) -> None:
    iters = [int(r["iteration"]) for r in per_iter]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    markers = {"p50": "o", "p90": "s", "p95": "D", "p99": "^"}
    for pct in ("p50", "p90", "p95", "p99"):
        bes = [float(r[f"bes_{pct}"]) for r in per_iter]
        pg = float(per_iter[0][f"pg_{pct}"])
        ax.plot(
            iters,
            bes,
            marker=markers[pct],
            color=_COLORS["bespoke"][1],
            markerfacecolor=_COLORS["bespoke"][0],
            markeredgecolor=_COLORS["bespoke"][1],
            markersize=6,
            lw=1.4,
            label=f"Bespoke {pct}",
        )
        ax.axhline(pg, color=_COLORS["pg"][1], ls="--", lw=0.9, alpha=0.75)
        ax.text(
            iters[-1] + 0.2,
            pg,
            f"PG {pct} = {pg:,.0f}",
            color=_COLORS["pg"][1],
            fontsize=8,
            va="center",
            fontweight="bold",
        )

    ax.set_yscale("log")
    ax.set_xticks(iters)
    ax.set_xlabel("Iteration", fontsize=11, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\bf{q\text{-}error}$" "\n(log scale)", fontsize=11)
    ax.set_title("Bespoke q-error across iterations", fontsize=11, fontweight="bold")
    style_axes(ax)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=10,
        loc="lower center",
        ncols=len(labels),
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.subplots_adjust(bottom=0.22)
    savefig(fig, out_dir, f"{workload}_q_error_progression.pdf")


# ---------- plot 2: cost & time by phase ----------


def plot_cost_by_phase(per_phase: list[dict], workload: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    for ax, metric, ylabel in (
        (axes[0], "cost", r"$\bf{Cost}$" "\n(USD)"),
        (
            axes[1],
            "response_time",
            r"$\bf{LLM\ wall\text{-}clock\ time}$" "\n(seconds)",
        ),
    ):
        agents = ["coder", "planner"]
        bottoms = {a: 0.0 for a in agents}
        seen_labels: set[str] = set()
        for row in per_phase:
            agent = row["agent"]
            phase = row["phase"]
            val = float(row[metric])
            fc, ec = PHASE_COLORS.get(phase, ("#cccccc", "#666666"))
            label_str = phase if phase not in seen_labels else None
            seen_labels.add(phase)
            ax.bar(
                agent,
                val,
                width=0.55,
                bottom=bottoms[agent],
                color=fc,
                edgecolor=ec,
                linewidth=EDGE_LW,
                label=label_str,
            )
            ax.text(
                agent,
                bottoms[agent] + val / 2,
                phase,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#222",
            )
            bottoms[agent] += val
        for a in agents:
            txt = f"{bottoms[a]:.2f}" if metric == "cost" else f"{bottoms[a]:.0f}s"
            ax.text(
                a,
                bottoms[a],
                txt,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels(agents, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel.replace("\n", " "), fontsize=11, fontweight="bold")
        style_axes(ax)
        ax.set_ylim(top=max(bottoms.values()) * 1.15)

    fig.tight_layout()
    savefig(fig, out_dir, f"{workload}_cost_by_phase.pdf")


# ---------- plot 3: coder tool usage ----------


def plot_tool_usage(per_tool: list[dict], workload: str, out_dir: Path) -> None:
    coder_rows = [r for r in per_tool if r["agent"] == "coder"]
    coder_rows.sort(key=lambda r: int(r["turns"]), reverse=True)

    planner_rows = [r for r in per_tool if r["agent"] == "planner"]
    planner_query_db = next(
        (float(r["tool_runtime"]) for r in planner_rows if r["tool"] == "query_db"), 0.0
    )
    planner_query_db_calls = next(
        (int(r["turns"]) for r in planner_rows if r["tool"] == "query_db"), 0
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))

    names = [r["tool"] for r in coder_rows]
    turns = [int(r["turns"]) for r in coder_rows]
    runtimes = [float(r["tool_runtime"]) for r in coder_rows]

    face = [TOOL_COLORS.get(n, ("#cccccc", "#666666", ""))[0] for n in names]
    edge = [TOOL_COLORS.get(n, ("#cccccc", "#666666", ""))[1] for n in names]
    hatch = [TOOL_COLORS.get(n, ("#cccccc", "#666666", ""))[2] for n in names]

    for ax, vals, fmt, title in (
        (axes[0], turns, lambda v: f"{v}", "Tool calls"),
        (axes[1], runtimes, lambda v: f"{v:.1f}s", "Cumulative tool runtime"),
    ):
        for i, (n, v, fc, ec, ht) in enumerate(zip(names, vals, face, edge, hatch)):
            ax.barh(
                i, v, height=0.6, color=fc, edgecolor=ec, linewidth=EDGE_LW, hatch=ht
            )
            ax.text(
                v,
                i,
                f"  {fmt(v)}",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
                color="#222",
            )
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10, fontweight="bold")
        ax.set_xlabel(title, fontsize=11, fontweight="bold", labelpad=6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(right=max(vals) * 1.18)

    fig.suptitle(
        f"Coder tool ecosystem  "
        f"(planner: {planner_query_db_calls} query_db calls, {planner_query_db:.1f} s total)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    savefig(fig, out_dir, f"{workload}_tool_usage.pdf")


# ---------- plot 4: iteration activity + q-error ----------


def plot_iteration_activity(
    per_iter_coder: list[dict], per_iter: list[dict], workload: str, out_dir: Path
) -> None:
    iters = [int(r["iteration"]) for r in per_iter_coder]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8.5, 5.8), sharex=True)

    # ----- top: stacked tool calls per iteration -----
    width = 0.7
    bottoms = np.zeros(len(iters))
    for tool in ("apply_patch", "evaluate_estimator", "ask_agent", "shell"):
        vals = np.array([int(r[tool]) for r in per_iter_coder])
        fc, ec, hatch = TOOL_COLORS[tool]
        ax_top.bar(
            iters,
            vals,
            width,
            bottom=bottoms,
            label=tool,
            color=fc,
            edgecolor=ec,
            linewidth=EDGE_LW,
            hatch=hatch,
        )
        bottoms = bottoms + vals
    ax_top.set_ylabel(r"$\bf{Coder\ tool\ calls}$", fontsize=11)
    ax_top.set_title(
        "Coder activity and resulting q-error per iteration",
        fontsize=11,
        fontweight="bold",
    )
    style_axes(ax_top)

    # phase bands
    phase_changes = [r["phase"] for r in per_iter_coder]
    boundaries: list[tuple[int, int, str]] = []
    start = 0
    for i in range(1, len(phase_changes) + 1):
        if i == len(phase_changes) or phase_changes[i] != phase_changes[start]:
            boundaries.append((start, i - 1, phase_changes[start]))
            start = i
    y_top = ax_top.get_ylim()[1]
    for s, e, ph in boundaries:
        fc = PHASE_COLORS.get(ph, ("#999", "#666"))[0]
        ax_top.axvspan(s - 0.5, e + 0.5, color=fc, alpha=0.10)
        ax_top.text(
            (s + e) / 2,
            y_top * 0.95,
            ph,
            ha="center",
            fontsize=8,
            fontweight="bold",
            color="#444",
        )

    # ----- bottom: q-error per iteration -----
    markers = {"p50": "o", "p90": "s", "p99": "^"}
    for pct in ("p50", "p90", "p99"):
        ys = [float(r[f"bes_{pct}"]) for r in per_iter]
        ax_bot.plot(
            iters,
            ys,
            marker=markers[pct],
            lw=1.4,
            markersize=6,
            color=_COLORS["bespoke"][1],
            markerfacecolor=_COLORS["bespoke"][0],
            markeredgecolor=_COLORS["bespoke"][1],
            label=f"Bespoke {pct}",
            alpha={"p50": 1.0, "p90": 0.75, "p99": 0.55}[pct],
        )
    for pct, ls in zip(("p50", "p90", "p99"), (":", "--", "-.")):
        pg = float(per_iter[0][f"pg_{pct}"])
        ax_bot.axhline(
            pg, color=_COLORS["pg"][1], ls=ls, lw=0.9, alpha=0.75, label=f"PG {pct}"
        )
    ax_bot.set_yscale("log")
    ax_bot.set_xticks(iters)
    ax_bot.set_xlabel(r"$\bf{Iteration}$", fontsize=11)
    ax_bot.set_ylabel(r"$\bf{q\text{-}error}$" "\n(log scale)", fontsize=11)
    style_axes(ax_bot)
    for s, e, ph in boundaries:
        fc = PHASE_COLORS.get(ph, ("#999", "#666"))[0]
        ax_bot.axvspan(s - 0.5, e + 0.5, color=fc, alpha=0.10)

    h_top, l_top = ax_top.get_legend_handles_labels()
    h_bot, l_bot = ax_bot.get_legend_handles_labels()
    fig.legend(
        h_top + h_bot,
        l_top + l_bot,
        frameon=False,
        fontsize=8,
        loc="lower center",
        ncols=5,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.subplots_adjust(bottom=0.18, hspace=0.18)
    savefig(fig, out_dir, f"{workload}_iteration_activity.pdf")


# ---------- plot 5: q-error at end of each stage ----------


def plot_q_error_by_stage(
    per_iter: list[dict], per_iter_coder: list[dict], workload: str, out_dir: Path
) -> None:
    stage_to_last_iter: dict[str, int] = {}
    for r in per_iter_coder:
        stage_to_last_iter[r["phase"]] = int(r["iteration"])
    iter_by_idx = {int(r["iteration"]): r for r in per_iter}

    stages = [s for s in STAGE_ORDER if s in stage_to_last_iter]
    percentiles = ["p50", "p90", "p99"]

    PCT_COLORS = {
        "p50": ("#9ECAE1", "#3182BD"),
        "p90": ("#FDAE6B", "#E6550D"),
        "p99": ("#A1D99B", "#31A354"),
    }

    fig, ax = plt.subplots(figsize=(5, 3))

    bar_w = 0.20
    n_pct = len(percentiles)
    loff = -(n_pct * bar_w) / 2

    def bes_x(si, pi):
        return si + loff + (pi + 0.5) * bar_w

    pg0 = iter_by_idx[0]

    for si, stage in enumerate(stages):
        for pi, pct in enumerate(percentiles):
            fc, ec = PCT_COLORS[pct]
            val_bes = float(iter_by_idx[stage_to_last_iter[stage]][f"bes_{pct}"])
            ax.bar(
                bes_x(si, pi), val_bes, bar_w, color=fc, edgecolor=ec, linewidth=EDGE_LW
            )

    for pct in percentiles:
        _, ec = PCT_COLORS[pct]
        val_pg = float(pg0[f"pg_{pct}"])
        ax.axhline(val_pg, color=ec, lw=1.6, linestyle="--", zorder=3)

    ax.set_yscale("log")
    ax.set_ylim(bottom=1)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([STAGE_LABEL[s] for s in stages], fontsize=10, fontweight="bold")
    ax.set_xlim(-0.5, len(stages) - 0.5)
    ax.set_ylabel(r"$\bf{Q\text{-}Error}$" " (log scale)", fontsize=11)
    style_axes(ax)

    pct_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=PCT_COLORS[p][0],
            edgecolor=PCT_COLORS[p][1],
            linewidth=EDGE_LW,
            label=p,
        )
        for p in percentiles
    ]
    sys_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#bbb",
            edgecolor="#555",
            linewidth=EDGE_LW,
            label="Bespoke",
        ),
        Line2D([0], [0], color="#555", lw=1.6, linestyle="--", label="Postgres"),
    ]
    leg_kw = dict(
        frameon=False,
        fontsize=9,
        title_fontsize=9,
        handletextpad=0.4,
        columnspacing=0.8,
        labelspacing=0.25,
    )
    leg1 = fig.legend(
        handles=pct_handles,
        title="Percentile",
        loc="upper left",
        bbox_to_anchor=(0.15, 1.0),
        ncols=3,
        **leg_kw,
    )
    leg1.get_title().set_fontweight("bold")
    leg2 = fig.legend(
        handles=sys_handles,
        title="System",
        loc="upper left",
        bbox_to_anchor=(0.6, 1.0),
        ncols=2,
        **leg_kw,
    )
    leg2.get_title().set_fontweight("bold")

    fig.tight_layout()
    fig.subplots_adjust(top=0.82, bottom=0.18)
    savefig(fig, out_dir, f"{workload}_q_error_by_stage.pdf")

    # raw values backing the plot
    lines = [
        f"# {workload}: q-error by stage",
        "",
        "Bespoke q-error percentiles at the end of each optimisation stage.",
        "",
        "| Stage | " + " | ".join(percentiles) + " |",
        "| --- | " + " | ".join("---" for _ in percentiles) + " |",
    ]
    for stage in stages:
        row = iter_by_idx[stage_to_last_iter[stage]]
        label = STAGE_LABEL[stage].replace("\n", " ")
        vals = " | ".join(f"{float(row[f'bes_{p}']):.4f}" for p in percentiles)
        lines.append(f"| {label} | {vals} |")
    lines += [
        "",
        "Postgres reference (dashed lines):",
        "",
        "| System | " + " | ".join(percentiles) + " |",
        "| --- | " + " | ".join("---" for _ in percentiles) + " |",
        "| Postgres | "
        + " | ".join(f"{float(pg0[f'pg_{p}']):.4f}" for p in percentiles)
        + " |",
        "",
    ]
    (out_dir / f"{workload}_q_error_by_stage.md").write_text("\n".join(lines))


# ---------- plot 6: stacked LLM turns by tool segment, per stage ----------


def _planner_stage_agg(log_dir: Path, per_iter_coder: list[dict]) -> dict:
    """Aggregate planner LLM turns and tool calls per optimisation stage."""
    with (log_dir / "planner_usage.json").open() as f:
        planner = json.load(f)

    ask_per_stage: dict[str, int] = defaultdict(int)
    for r in per_iter_coder:
        ask_per_stage[r["phase"]] += int(r["ask_agent"])

    agg = {
        s: {"turns": 0, "query_db": 0, "text response (no tool)": 0, "cost": 0.0}
        for s in STAGE_ORDER
    }

    cq_groups: list[list] = []
    for group in planner["requests"]:
        turns, phase = group
        if phase == "create_plan":
            # The planner is retagged to "coder_question" only after the first
            # coder run. Any ask_agent call during implement_estimator is
            # therefore recorded as create_plan and belongs to this stage.
            agg["implement_estimator"]["turns"] += len(turns)
            agg["implement_estimator"]["cost"] += sum(t.get("cost", 0.0) for t in turns)
            for t in turns:
                k = (t.get("tool") or {}).get("name")
                if k == "query_db":
                    agg["implement_estimator"]["query_db"] += 1
                else:
                    agg["implement_estimator"]["text response (no tool)"] += 1
        elif phase == "coder_question":
            cq_groups.append(turns)
        elif phase == "identify_best":
            agg["final_rounds"]["turns"] += len(turns)
            agg["final_rounds"]["cost"] += sum(t.get("cost", 0.0) for t in turns)
            agg["final_rounds"]["text response (no tool)"] += len(turns)

    ask_per_stage["implement_estimator"] = 0
    if sum(ask_per_stage.values()) != len(cq_groups):
        raise ValueError(
            "Planner coder_question groups do not match post-implementation "
            f"ask_agent calls: {len(cq_groups)} planner groups vs "
            f"{sum(ask_per_stage.values())} coder calls"
        )

    gi = 0
    for stage in STAGE_ORDER:
        for _ in range(ask_per_stage.get(stage, 0)):
            agg[stage]["cost"] += sum(t.get("cost", 0.0) for t in cq_groups[gi])
            for t in cq_groups[gi]:
                agg[stage]["turns"] += 1
                k = (t.get("tool") or {}).get("name")
                if k == "query_db":
                    agg[stage]["query_db"] += 1
                else:
                    agg[stage]["text response (no tool)"] += 1
            gi += 1

    return agg


def plot_calls_by_stage(
    per_iter_coder: list[dict],
    per_phase: list[dict],
    log_dir: Path,
    workload: str,
    out_dir: Path,
) -> None:
    """Grouped stacked bars per stage: Coder (left) and Planner (right)."""
    stages = STAGE_ORDER

    coder_agg = {
        s: {
            "turns": 0,
            "apply_patch": 0,
            "evaluate_estimator": 0,
            "ask_agent": 0,
            "shell": 0,
        }
        for s in stages
    }
    for r in per_iter_coder:
        s = r["phase"]
        if s not in coder_agg:
            continue
        coder_agg[s]["turns"] += int(r["turns"])
        for k in ("apply_patch", "evaluate_estimator", "ask_agent", "shell"):
            coder_agg[s][k] += int(r[k])
    for s in stages:
        tool_sum = sum(
            coder_agg[s][k]
            for k in ("apply_patch", "evaluate_estimator", "ask_agent", "shell")
        )
        coder_agg[s]["text response (no tool)"] = coder_agg[s]["turns"] - tool_sum

    planner_agg = _planner_stage_agg(log_dir, per_iter_coder)

    coder_cost = {
        r["phase"]: float(r["cost"]) for r in per_phase if r["agent"] == "coder"
    }
    combined_cost = {s: coder_cost.get(s, 0.0) + planner_agg[s]["cost"] for s in stages}

    coder_segments = [
        "apply_patch",
        "evaluate_estimator",
        "ask_agent",
        "shell",
        "text response (no tool)",
    ]
    planner_segments = ["query_db", "text response (no tool)"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(stages))
    w = 0.35
    gap = 0.04

    def draw_stacked(xs, agg_data, segments):
        bottoms = np.zeros(len(stages))
        for seg in segments:
            vals = np.array([agg_data[s].get(seg, 0) for s in stages], dtype=float)
            fc, ec, hatch = TOOL_COLORS[seg]
            bars = ax.bar(
                xs,
                vals,
                w,
                bottom=bottoms,
                color=fc,
                edgecolor=ec,
                linewidth=EDGE_LW,
                hatch=hatch,
            )
            for b, v, btm in zip(bars, vals, bottoms):
                if v > 0:
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        btm + v / 2,
                        f"{int(v)}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="medium",
                        color="black",
                    )
            bottoms += vals
        return bottoms

    planner_tops = draw_stacked(x - w / 2 - gap / 2, planner_agg, planner_segments)
    coder_tops = draw_stacked(x + w / 2 + gap / 2, coder_agg, coder_segments)

    for i, (pt, ct) in enumerate(zip(planner_tops, coder_tops)):
        ax.text(
            i - w / 2 - gap / 2,
            pt + 0.25,
            f"{int(pt)}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
        ax.text(
            i + w / 2 + gap / 2,
            ct + 0.25,
            f"{int(ct)}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_ylabel(r"$\bf{LLM\ turns}$", fontsize=11)
    style_axes(ax)
    ax.set_ylim(top=max(planner_tops.max(), coder_tops.max()) * 1.18)

    ax.set_xticks(x)
    ax.tick_params(axis="x", pad=20)

    def _stage_tick(stage, cost):
        parts = STAGE_LABEL_CALLS[stage].split("\n")
        bold = "\n".join(f"$\\bf{{{p}}}$" for p in parts)
        return f"{bold}\n{cost:.2f}\\$"

    ax.set_xticklabels(
        [_stage_tick(s, combined_cost[s]) for s in stages],
        fontsize=10,
    )
    y_sub = -ax.get_ylim()[1] * 0.032
    for i in range(len(stages)):
        ax.text(
            i - w / 2 - gap / 2,
            y_sub,
            "Planner",
            ha="center",
            va="top",
            fontsize=9.5,
            color="#444",
        )
        ax.text(
            i + w / 2 + gap / 2,
            y_sub,
            "Coder",
            ha="center",
            va="top",
            fontsize=9.5,
            color="#444",
        )

    all_segs = dict.fromkeys(coder_segments + planner_segments)
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=TOOL_COLORS[s][0],
            edgecolor=TOOL_COLORS[s][1],
            linewidth=EDGE_LW,
            hatch=TOOL_COLORS[s][2],
            label=TOOL_LABELS.get(s, s),
        )
        for s in all_segs
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=10,
        loc="upper center",
        ncols=(len(handles) + 1) // 2,
        bbox_to_anchor=(0.5, 0.88),
    )
    fig.subplots_adjust(top=0.78, bottom=0.2)
    savefig(fig, out_dir, f"{workload}_calls_by_stage.pdf")


def plot_workload(workload: str, log_dir: Path) -> None:
    out_dir = OUT_BASE / workload
    out_dir.mkdir(parents=True, exist_ok=True)

    per_iter = read_csv(out_dir / "per_iteration.csv")
    per_phase = read_csv(out_dir / "per_phase.csv")
    per_tool = read_csv(out_dir / "per_tool.csv")
    per_iter_coder = read_csv(out_dir / "per_iteration_coder.csv")

    plot_q_error_progression(per_iter, workload, out_dir)
    plot_cost_by_phase(per_phase, workload, out_dir)
    plot_tool_usage(per_tool, workload, out_dir)
    plot_iteration_activity(per_iter_coder, per_iter, workload, out_dir)
    plot_q_error_by_stage(per_iter, per_iter_coder, workload, out_dir)
    plot_calls_by_stage(per_iter_coder, per_phase, log_dir, workload, out_dir)

    print(f"[{workload}] Wrote 6 PDFs to {out_dir}")


def main() -> None:
    setup_plotting()
    for workload, log_dir in WORKLOADS.items():
        plot_workload(workload, log_dir)


if __name__ == "__main__":
    main()
