import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from utils.paper_plots.qo_runtimes import (
    plot_per_query_exec_times,
    plot_total_runtime_by_machine,
)
from utils.paper_plots.qo_runtimes_distribution import plot_slowdown_distribution
from utils.paper_plots.utils import setup_plotting

setup_plotting()


def extract_query_times(runtime_files: list[tuple[str, str, Path]]) -> pd.DataFrame:
    data = []
    for entry in runtime_files:
        workload, host, file = entry

        assert file.exists(), f"Runtime file {file} does not exist."

        with open(file, "r") as f:
            results = json.load(f)
        for query, times in results.items():
            data.append({**times, "query": query, "workload": workload, "host": host})

    return pd.DataFrame(data)


def plot(paths: list[tuple[str, str, Path]]):
    out_folder = Path(__file__).parent / "output"
    out_folder.mkdir(exist_ok=True)

    df = extract_query_times(paths)
    plot_per_query_exec_times(df, timeout_threshold=None, out_folder=out_folder)
    plot_total_runtime_by_machine(df, out_folder=out_folder)
    plot_slowdown_distribution(df, out_folder=out_folder)


if __name__ == "__main__":
    paths = [
        (
            "JOB",
            "c08",
            Path(__file__).parent.parent
            / "results/job/job_end_to_end_results_c08.json",
        ),
        (
            "JOB",
            "mac",
            Path(__file__).parent.parent
            / "results/job/job_end_to_end_results_mac.json",
        ),
        (
            "JOB-Complex",
            "c08",
            Path(__file__).parent.parent
            / "results/job-complex/jobcomplex_end_to_end_results_c08.json",
        ),
        (
            "JOB-Complex",
            "mac",
            Path(__file__).parent.parent
            / "results/job-complex/jobcomplex_end_to_end_results_mac.json",
        ),
    ]

    plot(paths)
