# Run analysis — `log_20260527_170232` (job-complex)

Analysis of the two-agent synthesis loop archived under
[results/job-complex/log_20260527_170232/](../job-complex/log_20260527_170232/).
The loop ran the planner once for plan creation, then the coder for ten
iterations (`iteration_0`..`iteration_9`), with the planner re-entering six
times via `AskAgentTool` and once for final selection.

## Scripts

- [analyze_run.py](analyze_run.py) — reads `coder_usage.json`,
  `planner_usage.json`, and every `iteration_N/feedback.json`, then writes
  the CSVs and `summary.json` under `output/`.
- [plot_run.py](plot_run.py) — reads those CSVs and writes four PDFs under
  `output/`.

Run from the repo root:
```bash
python results/analysis/analyze_run.py
python results/analysis/plot_run.py
```

## Outputs (`output/`)

| File | Contents |
|---|---|
| `summary.json` | Headline totals + per-iteration quality summary. |
| `per_phase.csv` | One row per `(agent, request_type)`: turns, cost, tokens, response time, tool runtime, cache hit rate. |
| `per_tool.csv` | One row per `(agent, tool_name)`: calls, runtime, cost incurred on the same turn. |
| `per_iteration.csv` | One row per iteration: bespoke + PG q-error percentiles, regression rate, outlier counts, estimator size. |
| `per_iteration_coder.csv` | One row per coder iteration: phase, turn count, tool-call breakdown, cost, wall-clock. |
| `q_error_progression.pdf` | Bespoke p50/p90/p95/p99 across iterations, log y, with PG baselines dashed. |
| `cost_by_phase.pdf` | Stacked cost and LLM wall-clock time by agent × phase. |
| `tool_usage.pdf` | Coder tool calls and cumulative tool runtime (planner reported in caption). |
| `iteration_activity.pdf` | Stacked coder tool calls per iteration with phase bands, paired with q-error per iteration. |
| `q_error_by_stage.pdf` | Bespoke p50/p90/p95/p99 at the **end of each optimisation stage** (`implement_estimator` → `join_rounds` → `filter_rounds` → `final_rounds`), with PG baselines. |
| `calls_by_stage.pdf` | Coder LLM turns and tool-call mix **aggregated over the iterations of each stage**. |

## Headline findings

**Spend.** Total LLM cost was **$1.64** over **820 s of model time**: coder
$0.88 (358 s), planner $0.75 (462 s). The two agents make 86 LLM turns and
67 tool calls between them (33 coder, 34 planner — every planner turn
except the final `identify_best` and a few `coder_question` re-entries is a
`query_db` call).

**Cache.** This run was a cached replay: 95.0 % of coder input tokens and
94.1 % of planner input tokens came from the on-disk LLM cache.
`stop_on_cache_miss` did not fire, so the recorded trace is faithful to
the original.

**Where the budget goes.** For the coder, `join_rounds` is the most
expensive phase ($0.38, 142 s, 19 turns over 5 iterations); the initial
`implement_estimator` step is second ($0.18, 116 s, 11 turns). `filter_rounds`
and `final_rounds` are roughly tied near $0.16 each. For the planner,
**plan creation alone takes 35 LLM turns** ($0.43, 272 s) — it dominates
planner spend, while the six `coder_question` re-entries together cost
$0.31. Selection (`identify_best`) is one turn, $0.01.

**Tools.** Coder tool mix: 11 `apply_patch`, 8 `shell`, 7 `evaluate_estimator`,
7 `ask_agent`. Each iteration applies exactly one patch except the first
(which applied two) and `iteration_2` which applied none — that iteration
reverted via `feedback.json` without committing code. `evaluate_estimator`
accounts for 1 105 s of tool runtime (the harness subprocess); the LLM
itself spends only 1 ms of agent time on the call.

**Quality.** Bespoke q-error percentiles fall **9.9× at the median and
334× at p99** over the 10 iterations:

| pct  | iter 0 | iter 9 | PG baseline |
|------|--------|--------|-------------|
| p50  | 167.22 | 16.94  | 530.41 |
| p90  | 58 884 | 938    | 83 057 |
| p95  | 318 091| 3 032  | 590 032 |
| p99  | 7.58e6 | 22 690 | 3.06e7 |

The bespoke estimator overtakes Postgres at every percentile after the
first iteration and never gives ground back. The total regression rate
(`q_bespoke / q_pg > 1.10`) drops from 36.0 % to 15.3 %.

**Curriculum effect.** The two largest single-iteration improvements
arrive in `join_rounds`: iter 0 → 3 cuts p99 from 7.58e6 to 2.24e5
(34×), and iter 5 → 8 cuts p99 from 2.24e5 to 2.27e4 (10×). Four
iterations leave the archived q-error unchanged:

- **Iter 5, 7, 9** are the trailing rounds of `join_rounds`,
  `filter_rounds`, and `final_rounds` respectively. Each has exactly one
  LLM turn and zero tool calls — the coder read the prompt ("you can
  check the feedback and revert if required, otherwise don't update or
  run evaluate") and chose not to act. These are the loop's structural
  no-ops.
- **Iter 4** is the only iteration that *did* patch (1 `apply_patch`,
  1 `evaluate_estimator`) but left the global q-error percentiles
  identical to iter 3 to two decimals — its change had no effect on the
  full-eval slice.

**Best-of-run.** The planner's final `identify_best` call selected
iteration 8 (p50 = 16.94), which is also the last iteration that changed
the estimator. Estimator size grew from 17.9 MB (iter 0) to 26.0 MB
(iter 8), well under the 1 000 MB hard cap.

## Notes / caveats

- `estimator_size` is only logged when `evaluate.py` runs `setup()`, i.e.
  on full eval. Iterations primed by `--skip_setup --no_filters` or
  `--no_joins` re-use the prior `outputs/job_subplans.json` and so leave
  the size field empty in their archived `feedback.json`. This is why
  only iter 0, 8, and 9 report a size in `per_iteration.csv`.
- The "PG baseline" row in `q_error_progression.pdf` is constant across
  iterations because Postgres-side annotations are computed once in
  `annotate_subplans_with_pg_cards` before the loop starts.
- Phase tagging is inferred from `ResourceTracker.request_type` at archive
  time; the recorded run has no `initial_rounds` group because the
  scheduler entered `join_rounds` directly after `implement_estimator`.
