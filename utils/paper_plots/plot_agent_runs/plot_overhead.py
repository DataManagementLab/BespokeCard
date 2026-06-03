"""Resource-overhead plots, sourced ENTIRELY from the archived agent logs
(no estimator code is executed). Companion to plot_run.py; shares its palette
and output convention (paper_plots/output/agent_plots/{workload}/, files
prefixed with the workload name).

Only two of the overhead signals are actually present in the logs:

  1. {w}_estimator_size.pdf  -- statistics memory footprint (asizeof), read from
                                `estimator_size` in each full-eval feedback.json.
  2. {w}_eval_runtime.pdf    -- wall time of the full evaluate.py subprocess per
                                iteration, read from the `evaluate_estimator` tool
                                runtime in coder_usage.json. This is a BUNDLED
                                number: setup() + estimate() over all subplans +
                                the 9 analyses + plotting. It is NOT separable
                                into pure setup time or pure per-estimate latency
                                -- those were only ever logged to stdout and were
                                never archived, so they cannot be recovered here.

Run after the runs are archived under results/{job,job-complex}/log_*/:
    python -m paper_plots.plot_agent_runs.plot_overhead
    python paper_plots/plot_agent_runs/plot_overhead.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.paper_plots.utils import _COLORS, setup_plotting  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
OUT_BASE = Path(__file__).resolve().parents[1] / "output" / "agent_plots"

WORKLOADS = ["job", "job-complex"]
WORKLOAD_LABEL = {"job": "JOB", "job-complex": "JOB-Complex"}

SIZE_CAP_MB = 1000  # base_card_estimator hard cap
EDGE_LW = 1.2

# (face, edge) per optimisation phase — same palette as plot_run.py
PHASE_COLORS = {
    "implement_estimator": ("#9ECAE1", "#3182BD"),
    "join_rounds": ("#A1D99B", "#31A354"),
    "filter_rounds": ("#FDAE6B", "#E6550D"),
    "final_rounds": ("#BCBDDC", "#6A51A3"),
}
PHASE_LABEL = {
    "implement_estimator": "Implement",
    "join_rounds": "Join rounds",
    "filter_rounds": "Filter rounds",
    "final_rounds": "Final rounds",
}


# ---------- helpers (shared style with plot_run.py) ----------


def find_log_dir(workload: str) -> Path:
    candidates = sorted((RESULTS_DIR / workload).glob("log_*"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No log_* directories in {RESULTS_DIR / workload}")
    return candidates[0]


def style_axes(ax) -> None:
    ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------- log readers ----------


def read_estimator_sizes(log_dir: Path) -> list[tuple[int, float]]:
    """(iteration, size_mb) for every iteration whose feedback.json recorded a
    size. Only full-eval iterations (those that ran setup()) have one."""
    points = []
    for it in sorted(
        log_dir.glob("iteration_*"), key=lambda p: int(p.name.split("_")[1])
    ):
        fb_path = it / "feedback.json"
        if not fb_path.exists():
            continue
        fb = json.loads(fb_path.read_text())
        raw = fb.get("estimator_size")
        if not raw:
            continue
        # stored as e.g. "31.45 MB"
        points.append((int(it.name.split("_")[1]), float(str(raw).split()[0])))
    return points


def read_eval_runtimes(log_dir: Path) -> list[tuple[int, str, float]]:
    """(iteration, phase, subprocess_runtime_s) for every coder group that ran
    the evaluate_estimator tool. Each runtime is one full evaluate.py call."""
    coder = json.loads((log_dir / "coder_usage.json").read_text())
    out = []
    for gi, (turns, phase) in enumerate(coder["requests"]):
        rt = sum(
            (t.get("tool") or {}).get("runtime", 0.0)
            for t in turns
            if (t.get("tool") or {}).get("name") == "evaluate_estimator"
        )
        n = sum(
            1
            for t in turns
            if (t.get("tool") or {}).get("name") == "evaluate_estimator"
        )
        if n:
            out.append((gi, phase, rt))
    return out


# ---------- plot 1: estimator memory footprint ----------


def plot_estimator_size(
    points: list[tuple[int, float]], workload: str, out_dir: Path
) -> None:
    if not points:
        print(
            f"[{workload}] no estimator_size in any feedback.json — skipping size plot"
        )
        return
    iters = [i for i, _ in points]
    vals = [v for _, v in points]
    final = vals[-1]
    fc, ec = _COLORS["bespoke"]

    fig, ax = plt.subplots(figsize=(4.6, 3.7))
    xpos = np.arange(len(iters))
    bars = ax.bar(
        xpos, vals, width=0.6, color=fc, edgecolor=ec, linewidth=EDGE_LW, zorder=3
    )
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.axhline(SIZE_CAP_MB, color=_COLORS["pg"][1], ls="--", lw=1.4, zorder=2)
    ax.text(
        xpos[-1],
        SIZE_CAP_MB,
        f"cap {SIZE_CAP_MB:,} MB",
        ha="right",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=_COLORS["pg"][1],
    )

    ax.set_yscale("log")
    ax.set_ylim(top=SIZE_CAP_MB * 3)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"iter {i}" for i in iters], fontsize=10, fontweight="bold")
    ax.set_ylabel(r"$\bf{Statistics\ memory}$" "\n(MB, log scale)", fontsize=11)
    ax.set_title(
        f"{WORKLOAD_LABEL[workload]} estimator footprint "
        f"({100 * final / SIZE_CAP_MB:.1f}% of cap)",
        fontsize=11,
        fontweight="bold",
    )
    style_axes(ax)
    fig.tight_layout()
    savefig(fig, out_dir, f"{workload}_estimator_size.pdf")


# ---------- plot 2: full evaluate.py subprocess runtime per iteration ----------


def plot_eval_runtime(
    runtimes: list[tuple[int, str, float]], workload: str, out_dir: Path
) -> None:
    if not runtimes:
        print(f"[{workload}] no evaluate_estimator runtimes — skipping runtime plot")
        return
    iters = [g for g, _, _ in runtimes]
    phases = [p for _, p, _ in runtimes]
    vals = [rt for _, _, rt in runtimes]

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    xpos = np.arange(len(iters))
    face = [PHASE_COLORS.get(p, ("#cccccc", "#666666"))[0] for p in phases]
    edge = [PHASE_COLORS.get(p, ("#cccccc", "#666666"))[1] for p in phases]
    bars = ax.bar(
        xpos, vals, width=0.66, color=face, edgecolor=edge, linewidth=EDGE_LW, zorder=3
    )
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v:.0f}s",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    mean = float(np.mean(vals))
    ax.axhline(mean, color=_COLORS["pg"][1], ls="--", lw=1.1, zorder=2)
    ax.text(
        xpos[-1] + 0.4,
        mean,
        f"mean {mean:.0f}s",
        ha="right",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=_COLORS["pg"][1],
    )

    ax.set_xticks(xpos)
    ax.set_xticklabels([f"iter {i}" for i in iters], fontsize=10, fontweight="bold")
    ax.set_ylabel(r"$\bf{evaluate.py\ wall\ time}$" "\n(seconds)", fontsize=11)
    ax.set_ylim(top=max(vals) * 1.18)
    ax.set_title(
        f"{WORKLOAD_LABEL[workload]} per-iteration evaluation runtime\n"
        "(setup + estimate over all subplans + analyses; bundled)",
        fontsize=11,
        fontweight="bold",
    )
    style_axes(ax)

    seen = []
    handles = []
    for p in phases:
        if p in seen:
            continue
        seen.append(p)
        fcp, ecp = PHASE_COLORS.get(p, ("#cccccc", "#666666"))
        handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=fcp,
                edgecolor=ecp,
                linewidth=EDGE_LW,
                label=PHASE_LABEL.get(p, p),
            )
        )
    ax.legend(
        handles=handles, frameon=False, fontsize=9, ncols=len(handles), loc="upper left"
    )

    fig.tight_layout()
    savefig(fig, out_dir, f"{workload}_eval_runtime.pdf")


def plot_workload(workload: str) -> None:
    log_dir = find_log_dir(workload)
    out_dir = OUT_BASE / workload
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_estimator_size(read_estimator_sizes(log_dir), workload, out_dir)
    plot_eval_runtime(read_eval_runtimes(log_dir), workload, out_dir)
    print(f"[{workload}] log_dir: {log_dir.name}  ->  {out_dir}")


def main() -> None:
    setup_plotting()
    for workload in WORKLOADS:
        plot_workload(workload)


if __name__ == "__main__":
    main()
