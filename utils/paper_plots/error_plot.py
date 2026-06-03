import json
from pathlib import Path

import pandas as pd


def filter_subplan_duplicates(subplans_dict: dict) -> list[dict]:
    subplans = []
    for _, values in subplans_dict.items():
        for _, subplan in values.items():
            subplans.append(subplan)
    seen = set()
    filtered = []
    for sp in subplans:
        key = (
            sp["q_error_pg"],
            sp["true_card"],
            len(sp["query_raw_info"][0])
            + len(sp["query_raw_info"][1])
            + len(sp["query_raw_info"][2]),
            "".join(sorted([t["name"] for t in sp["query_raw_info"][0]])),
        )
        if key not in seen:
            seen.add(key)
            filtered.append(sp)
    return filtered


def load_error_df(
    est_path: Path, workload: str | None = None, deduplicate: bool = True
) -> pd.DataFrame:
    subplans_dict = json.loads(est_path.read_text())

    if deduplicate:
        subplans = filter_subplan_duplicates(subplans_dict)
    else:
        subplans = [sp for v in subplans_dict.values() for sp in v.values()]

    rows = []
    for sp in subplans:
        num_tables = len(sp["query_raw_info"][0])
        base = dict(
            num_tables=num_tables,
            true_card=sp["true_card"],
            over_estimate=None,
            q_error=None,
            estimator=None,
        )

        bespoke_row = base.copy()
        bespoke_row["estimator"] = "bespoke"
        bespoke_row["q_error"] = sp["q_error_bespoke"]
        bespoke_row["over_estimate"] = sp["bespoke_over"]
        rows.append(bespoke_row)

        pg_row = base.copy()
        pg_row["estimator"] = "postgres"
        pg_row["q_error"] = sp["q_error_pg"]
        pg_row["over_estimate"] = sp["pg_over"]
        rows.append(pg_row)

    df = pd.DataFrame(rows)
    df["num_tables"] = df["num_tables"].astype(int)
    df["q_error"] = df["q_error"].astype(float)
    if workload is not None:
        df["workload"] = workload
    return df


def plot(paths: list[tuple[str, Path]]):
    import sys

    sys.path.append(str(Path(__file__).parent.parent))

    from utils.paper_plots.error_boxplot_by_tables import plot_error_boxplot_by_tables
    from utils.paper_plots.q_error_distribution import (
        plot_q_error_distribution_all,
        plot_q_error_distribution_base,
    )
    from utils.paper_plots.q_error_table_by_tables import (
        plot_q_error_table_all,
        plot_q_error_table_by_tables,
    )
    from utils.paper_plots.utils import setup_plotting

    setup_plotting()

    dfs = [load_error_df(path, workload=workload) for workload, path in paths]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

    out_folder = Path(__file__).parent / "output"
    out_folder.mkdir(exist_ok=True)
    plot_error_boxplot_by_tables(df, out_folder)
    plot_q_error_table_by_tables(df, out_folder)
    plot_q_error_table_all(df, out_folder)
    plot_q_error_distribution_base(df, out_folder)
    plot_q_error_distribution_all(df, out_folder)


if __name__ == "__main__":
    paths = [
        ("JOB", Path(__file__).parent.parent / "results/job/job_subplans.json"),
        (
            "JOB-Complex",
            Path(__file__).parent.parent
            / "results/job-complex/jobcomplex_subplans.json",
        ),
    ]
    plot(paths)
