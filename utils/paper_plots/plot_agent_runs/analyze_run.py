"""Analyze agent-loop runs for each workload under results/{job,job-complex}/log_<ts>/.

Reads coder_usage.json, planner_usage.json, and per-iteration feedback.json,
then writes per-workload summary artifacts to paper_plots/output/agent_plots/{job,job-complex}/.

Run from repo root:
    python -m paper_plots.plot_agent_runs.analyze_run
or:
    python paper_plots/plot_agent_runs/analyze_run.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import defaultdict, Counter

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


def load_usage(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def flatten_turns(usage: dict) -> list[dict]:
    """Yield one dict per LLM turn, tagged with its phase (request_type)."""
    rows = []
    for group_idx, group in enumerate(usage["requests"]):
        turns, phase = group
        for turn_idx, turn in enumerate(turns):
            rows.append(
                {
                    "phase": phase,
                    "group_idx": group_idx,
                    "turn_idx": turn_idx,
                    "cost": turn.get("cost", 0.0),
                    "response_time": turn.get("response_time", 0.0),
                    "from_cache": bool(turn.get("from_cache", False)),
                    "input_tokens": turn["tokens"].get("input_tokens", 0),
                    "cached_tokens": turn["tokens"].get("cached_tokens", 0),
                    "output_tokens": turn["tokens"].get("output_tokens", 0),
                    "reasoning_tokens": turn["tokens"].get("reasoning_tokens", 0),
                    "response_type": turn.get("response_type", ""),
                    "tool_name": (turn.get("tool") or {}).get("name"),
                    "tool_runtime": (turn.get("tool") or {}).get("runtime", 0.0),
                }
            )
    return rows


def agg(rows: list[dict], key: str) -> dict[str, dict]:
    """Aggregate per-turn rows by the given key ('phase' or 'tool_name')."""
    by = defaultdict(
        lambda: {
            "turns": 0,
            "groups": set(),
            "cost": 0.0,
            "response_time": 0.0,
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "tool_runtime": 0.0,
            "from_cache_count": 0,
            "tool_counts": Counter(),
        }
    )
    for r in rows:
        k = r.get(key)
        if k is None:
            continue
        b = by[k]
        b["turns"] += 1
        b["groups"].add(r["group_idx"])
        b["cost"] += r["cost"]
        b["response_time"] += r["response_time"]
        b["input_tokens"] += r["input_tokens"]
        b["cached_tokens"] += r["cached_tokens"]
        b["output_tokens"] += r["output_tokens"]
        b["reasoning_tokens"] += r["reasoning_tokens"]
        b["tool_runtime"] += r["tool_runtime"]
        if r["from_cache"]:
            b["from_cache_count"] += 1
        if r["tool_name"]:
            b["tool_counts"][r["tool_name"]] += 1
    out = {}
    for k, v in by.items():
        v["groups"] = len(v["groups"])
        v["cache_hit_rate"] = v["from_cache_count"] / v["turns"] if v["turns"] else 0
        token_total = v["input_tokens"]
        v["token_cache_rate"] = v["cached_tokens"] / token_total if token_total else 0
        v["tool_counts"] = dict(v["tool_counts"])
        out[k] = v
    return out


def load_feedbacks(log_dir: Path) -> list[dict]:
    rows = []
    iters = sorted(log_dir.glob("iteration_*"), key=lambda p: int(p.name.split("_")[1]))
    for it in iters:
        idx = int(it.name.split("_")[1])
        fb = json.loads((it / "feedback.json").read_text())
        bes = fb["q_error_percentiles"]["bespoke"]
        pg = fb["q_error_percentiles"]["pg"]
        rows.append(
            {
                "iteration": idx,
                "bes_p50": bes["50th"],
                "bes_p90": bes["90th"],
                "bes_p95": bes["95th"],
                "bes_p99": bes["99th"],
                "pg_p50": pg["50th"],
                "pg_p90": pg["90th"],
                "pg_p95": pg["95th"],
                "pg_p99": pg["99th"],
                "regression_rate": fb.get("total_regression_rate"),
                "estimator_size": fb.get("estimator_size"),
                "outlier_over": fb.get("outliers", {}).get("count_over_estimates"),
                "outlier_under": fb.get("outliers", {}).get("count_under_estimates"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def analyze_workload(name: str, log_dir: Path) -> None:
    out_dir = OUT_BASE / name
    out_dir.mkdir(parents=True, exist_ok=True)

    coder = load_usage(log_dir / "coder_usage.json")
    planner = load_usage(log_dir / "planner_usage.json")

    coder_turns = flatten_turns(coder)
    planner_turns = flatten_turns(planner)

    coder_by_phase = agg(coder_turns, "phase")
    planner_by_phase = agg(planner_turns, "phase")
    coder_by_tool = agg(coder_turns, "tool_name")
    planner_by_tool = agg(planner_turns, "tool_name")

    # ---------- summary.json ----------
    coder_total_cost = sum(p["cost"] for p in coder_by_phase.values())
    planner_total_cost = sum(p["cost"] for p in planner_by_phase.values())
    coder_total_time = sum(p["response_time"] for p in coder_by_phase.values())
    planner_total_time = sum(p["response_time"] for p in planner_by_phase.values())
    coder_total_turns = sum(p["turns"] for p in coder_by_phase.values())
    planner_total_turns = sum(p["turns"] for p in planner_by_phase.values())
    coder_total_in = sum(p["input_tokens"] for p in coder_by_phase.values())
    coder_total_cached = sum(p["cached_tokens"] for p in coder_by_phase.values())
    coder_total_out = sum(p["output_tokens"] for p in coder_by_phase.values())
    planner_total_in = sum(p["input_tokens"] for p in planner_by_phase.values())
    planner_total_cached = sum(p["cached_tokens"] for p in planner_by_phase.values())
    planner_total_out = sum(p["output_tokens"] for p in planner_by_phase.values())

    feedbacks = load_feedbacks(log_dir)
    first = feedbacks[0]
    last = feedbacks[-1]
    best_p50 = min(f["bes_p50"] for f in feedbacks)
    best_idx = next(f["iteration"] for f in feedbacks if f["bes_p50"] == best_p50)

    summary = {
        "workload": name,
        "log_dir": str(log_dir),
        "num_iterations": len(feedbacks),
        "coder": {
            "total_cost_usd": round(coder_total_cost, 4),
            "total_response_time_s": round(coder_total_time, 1),
            "total_turns": coder_total_turns,
            "num_iteration_groups": len(coder["requests"]),
            "input_tokens": coder_total_in,
            "cached_tokens": coder_total_cached,
            "output_tokens": coder_total_out,
            "token_cache_rate": (
                round(coder_total_cached / coder_total_in, 4) if coder_total_in else 0
            ),
            "tool_counts": coder["tool_count_analysis"][0],
            "tool_total": coder["tool_count_analysis"][1],
        },
        "planner": {
            "total_cost_usd": round(planner_total_cost, 4),
            "total_response_time_s": round(planner_total_time, 1),
            "total_turns": planner_total_turns,
            "num_iteration_groups": len(planner["requests"]),
            "input_tokens": planner_total_in,
            "cached_tokens": planner_total_cached,
            "output_tokens": planner_total_out,
            "token_cache_rate": (
                round(planner_total_cached / planner_total_in, 4)
                if planner_total_in
                else 0
            ),
            "tool_counts": planner["tool_count_analysis"][0],
            "tool_total": planner["tool_count_analysis"][1],
        },
        "combined": {
            "total_cost_usd": round(coder_total_cost + planner_total_cost, 4),
            "total_response_time_s": round(coder_total_time + planner_total_time, 1),
            "total_turns": coder_total_turns + planner_total_turns,
        },
        "quality": {
            "initial_iteration": first,
            "final_iteration": last,
            "best_p50_iteration": best_idx,
            "best_p50_value": best_p50,
            "p50_improvement_factor": round(first["bes_p50"] / last["bes_p50"], 2),
            "p99_improvement_factor": round(first["bes_p99"] / last["bes_p99"], 2),
            "regression_rate_initial": first["regression_rate"],
            "regression_rate_final": last["regression_rate"],
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---------- per-phase CSV ----------
    phase_fields = [
        "agent",
        "phase",
        "groups",
        "turns",
        "cost",
        "response_time",
        "input_tokens",
        "cached_tokens",
        "output_tokens",
        "reasoning_tokens",
        "tool_runtime",
        "from_cache_count",
        "cache_hit_rate",
        "token_cache_rate",
    ]
    phase_rows = []
    for phase, v in coder_by_phase.items():
        phase_rows.append(
            {"agent": "coder", "phase": phase, **{k: v[k] for k in phase_fields[2:]}}
        )
    for phase, v in planner_by_phase.items():
        phase_rows.append(
            {"agent": "planner", "phase": phase, **{k: v[k] for k in phase_fields[2:]}}
        )
    write_csv(out_dir / "per_phase.csv", phase_rows, phase_fields)

    # ---------- per-tool CSV ----------
    tool_fields = ["agent", "tool", "turns", "tool_runtime", "cost", "response_time"]
    tool_rows = []
    for tool, v in coder_by_tool.items():
        tool_rows.append(
            {
                "agent": "coder",
                "tool": tool,
                "turns": v["turns"],
                "tool_runtime": v["tool_runtime"],
                "cost": v["cost"],
                "response_time": v["response_time"],
            }
        )
    for tool, v in planner_by_tool.items():
        tool_rows.append(
            {
                "agent": "planner",
                "tool": tool,
                "turns": v["turns"],
                "tool_runtime": v["tool_runtime"],
                "cost": v["cost"],
                "response_time": v["response_time"],
            }
        )
    write_csv(out_dir / "per_tool.csv", tool_rows, tool_fields)

    # ---------- per-iteration q-error CSV ----------
    fb_fields = [
        "iteration",
        "bes_p50",
        "bes_p90",
        "bes_p95",
        "bes_p99",
        "pg_p50",
        "pg_p90",
        "pg_p95",
        "pg_p99",
        "regression_rate",
        "estimator_size",
        "outlier_over",
        "outlier_under",
    ]
    write_csv(out_dir / "per_iteration.csv", feedbacks, fb_fields)

    # ---------- per-iteration coder activity ----------
    iter_fields = [
        "iteration",
        "phase",
        "turns",
        "tool_calls",
        "ask_agent",
        "apply_patch",
        "shell",
        "evaluate_estimator",
        "cost",
        "response_time",
    ]
    iter_rows = []
    for group_idx, group in enumerate(coder["requests"]):
        turns, phase = group
        tools = Counter(
            (t.get("tool") or {}).get("name") for t in turns if t.get("tool")
        )
        iter_rows.append(
            {
                "iteration": group_idx,
                "phase": phase,
                "turns": len(turns),
                "tool_calls": sum(tools.values()),
                "ask_agent": tools.get("ask_agent", 0),
                "apply_patch": tools.get("apply_patch", 0),
                "shell": tools.get("shell", 0),
                "evaluate_estimator": tools.get("evaluate_estimator", 0),
                "cost": round(sum(t.get("cost", 0.0) for t in turns), 4),
                "response_time": round(
                    sum(t.get("response_time", 0.0) for t in turns), 1
                ),
            }
        )
    write_csv(out_dir / "per_iteration_coder.csv", iter_rows, iter_fields)

    print(f"[{name}] log_dir: {log_dir.name}  →  {out_dir}")
    print(f"  summary.json  ({len(feedbacks)} iterations)")
    print(f"  per_phase.csv ({len(phase_rows)} rows)")
    print(f"  per_tool.csv  ({len(tool_rows)} rows)")
    print("  per_iteration.csv")
    print("  per_iteration_coder.csv")


def main() -> None:
    for name, log_dir in WORKLOADS.items():
        analyze_workload(name, log_dir)


if __name__ == "__main__":
    main()
