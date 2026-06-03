import json
import os
import statistics
import random
from typing import Any, Dict, List, Set, Tuple

import psycopg2
from duckdb import cursor
from tqdm import tqdm
import dotenv
import logging

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class QueryExecutor:
    def __init__(self):
        self.connection = psycopg2.connect(database="imdb", host="localhost", port=5432)
        self.connection.autocommit = True
        self.cursor = self.connection.cursor()
        self.cursor.execute(
            f"set statement_timeout to {os.getenv('end2end_timeout')}000;"
        )

    def execute(self, query: str):
        """Executes any SQL statement, returns rows only for SELECT."""
        try:
            self.cursor.execute(query)
            return self.cursor.fetchall()
        except psycopg2.Error as e:
            logger.error(f"Query execution timed out. {e}")
            return None


def run_and_measure_query(
    sql: str,
    hint: str,
    repetitions: int,
    executor: QueryExecutor,
) -> Dict:
    explain_flags = "EXPLAIN (ANALYZE) "
    sql = explain_flags + sql
    # assemble query with hint
    if hint != "":
        sql_with_hint = f"/*=pg_lab=  {hint} */ {sql}"
    else:
        sql_with_hint = sql

    times = []
    results = []
    for _ in range(repetitions):
        result = executor.execute(sql_with_hint)
        if result is None:  # timeout
            return None, int(
                os.getenv("end2end_timeout")
            )  # return timeout value as time
        else:
            execution_time = result[-2][0].split("Execution Time: ")[1].split(" ms")[0]
            times.append(float(execution_time) / 1000)  # convert to seconds
            results.append("\n".join([r[0] for r in result]))

    # return median time and corresponding result
    median_time = round(statistics.median(times), 4)
    median_result = results[times.index(min(times, key=lambda x: abs(x - median_time)))]

    return median_result, median_time


def execute_annotated_queries(
    approaches_to_execute: List[str] = ["true", "pg", "bespoke"],
    warmup: bool = False,
    repetitions: int = 1,
) -> List[Dict[str, Any]]:
    with open("outputs/job_subplans.json", "r") as f:
        subplans_dict = json.load(f)
    if os.path.exists("outputs/end_to_end_results.json"):
        with open("outputs/end_to_end_results.json", "r") as f:
            end_to_end_results = json.load(f)
    else:
        end_to_end_results = {}
        for sql in subplans_dict.keys():
            end_to_end_results[sql] = {}
            for item in ["time", "hint", "output"]:
                for approach in ["pg", "bespoke", "true"]:
                    end_to_end_results[sql][f"{approach}_{item}"] = 0

    total_pg = 0.0
    total_bespoke = 0.0
    total_true = 0.0

    executor = QueryExecutor()

    if warmup:
        logger.info("Warming up the database...")
        sample_queries = random.sample(list(subplans_dict.keys()), 25)
        for query in tqdm(sample_queries):
            executor.execute(query)
        logger.info("Warmup completed.")

    for sql, subplans in tqdm(subplans_dict.items()):
        bespoke_hint = ""
        true_hint = ""
        pg_hint = ""
        for subplan, items in subplans.items():
            hint = f"Card({' '.join(subplan.strip('()').replace("'", '').split(','))} #CARD) "
            bespoke_hint += hint.replace("CARD", str(items["bespoke_card"]))
            true_hint += hint.replace("CARD", str(items["true_card"]))
            pg_hint += hint.replace("CARD", str(items["pg_card"]))

        entry = end_to_end_results[sql]

        if "pg" in approaches_to_execute:
            out_pg, t_pg = run_and_measure_query(
                sql=sql, hint="", repetitions=repetitions, executor=executor
            )
            entry["pg_time"] = t_pg
            entry["pg_hint"] = pg_hint
            entry["pg_output"] = out_pg
        if "bespoke" in approaches_to_execute:
            out_bespoke, t_bespoke = run_and_measure_query(
                sql=sql, hint=bespoke_hint, repetitions=repetitions, executor=executor
            )
            entry["bespoke_time"] = t_bespoke
            entry["bespoke_hint"] = bespoke_hint
            entry["bespoke_output"] = out_bespoke
        if "true" in approaches_to_execute:
            out_true, t_true = run_and_measure_query(
                sql=sql, hint=true_hint, repetitions=repetitions, executor=executor
            )
            entry["true_time"] = t_true
            entry["true_hint"] = true_hint
            entry["true_output"] = out_true

        total_pg += entry.get("pg_time", 0)
        total_bespoke += entry.get("bespoke_time", 0)
        total_true += entry.get("true_time", 0)
        logger.info(
            f"PG={entry.get('pg_time', 0):.2f}, Bespoke={entry.get('bespoke_time', 0):.2f}, True={entry.get('true_time', 0):.2f}\t total PG={total_pg:.2f}, total Bespoke={total_bespoke:.2f}, total True={total_true:.2f}"
        )

    with open("outputs/end_to_end_results.json", "w") as f:
        json.dump(end_to_end_results, f, indent=2)
