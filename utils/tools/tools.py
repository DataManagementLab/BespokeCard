import duckdb
from typing import Any, Dict
import subprocess
import shlex
import subprocess
from pathlib import Path
import logging
from agents import function_tool
import dotenv
import os

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

DB_PATH = "data/imdb.duckdb"


def run_duckdb_query(sql: str) -> Dict[str, Any]:
    """
    Run a SQL query against the database using DuckDB.

    Args:
        sql: A string containing the SQL query to execute.

    Returns:
        A dictionary containing:
            columns: A list of column names returned by the query.
            rows: A list of tuples representing the rows returned by the query.
            row_count: The number of rows returned by the query.
    """
    logger.debug(f"Executing SQL query: {sql}")
    con = duckdb.connect(DB_PATH, read_only=True, config={"threads": 1})
    try:
        cursor = con.execute(sql)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        logger.debug(f"Query returned {columns}, {rows}")
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
        }
    finally:
        con.close()


@function_tool
def query_db(sql: str) -> dict:
    """
    Run a SQL query against the database using DuckDB.

    Args:
        sql: A string containing the SQL query to execute.

    Returns:
        A dictionary containing:
            columns: A list of column names returned by the query.
            rows: A list of tuples representing the rows returned by the query.
            row_count: The number of rows returned by the query.
    """
    return run_duckdb_query(sql)


@function_tool
def evaluate() -> str:
    """
    Evaluates the card_estimator code by a separate evaluation script.

    Returns:
        A string indicating the evaluation result:
            - "success" if the script exits with return code 0.
            - The captured stderr output if the script exits with a non-zero return code.
            - "Execution timed out" if the process exceeds the time limit of 2000 seconds.
    """
    try:
        venv_path = os.getenv("VENV_PATH")
        result = subprocess.run(
            [venv_path, "evaluate.py"],
            # ["conda", "run", "-n", "msc", "python", "evaluate.py"],
            capture_output=True,
            text=True,
            timeout=2000,
        )
        logger.info(
            f"Code execution stdout:\n{result.stdout}\nCode execution stderr:\n{result.stderr}"
        )
        if result.returncode == 0:
            return "success"
        else:
            return result.stderr
    except subprocess.TimeoutExpired:
        logger.info("Code execution timed out after 2000 seconds.")
        return "Execution timed out. Your code is taking too long to run. Please optimize it to run faster while maintaining or improving accuracy. If necessary, discuss with the planner what statistics can be adapted to be more efficient."


@function_tool
def shell(
    command: str,
    cwd: str = ".",
) -> dict:
    """
    Run a restricted shell command for project exploration. Some files are blocked from being read due to their large size.
    Allowed commands: ls, cat, sed, head, tail, wc. Please use 'ls' without flags like ["-la", "-l", "-al", "-R", "-all"] as this might break caching.

    Args:
        command: The full shell command to execute.
        cwd: The working directory in which to execute the command.

    Returns:
        A dictionary containing:
            ok (bool): True if the command exited with return code 0.
            stdout (str): Captured standard output (if executed).
            stderr (str): Captured standard error (if executed).
            error (str): Error message if execution was blocked or failed.
    """

    logger.info(f"Running shell command: {command} in {cwd}")

    ALLOWED = {"ls", "cat", "sed", "head", "tail", "wc"}
    forbidden_files = {
        "job",
        "env",
        "sql",
        "evaluate",
        "llm_cache",
        "job_subplans",
        "end_to_end_results",
        "outliers",
    }

    workspace = Path(".").resolve()
    resolved_cwd = (workspace / cwd).resolve()

    if not str(resolved_cwd).startswith(str(workspace)):
        return {
            "ok": False,
            "stdout": "",
            "stderr": "",
            "error": "Invalid working directory.",
        }

    tokens = shlex.split(command)
    if not tokens or tokens[0] not in ALLOWED:
        logger.debug(f"Command '{command}' is not allowed.")
        return {
            "ok": False,
            "stdout": "",
            "stderr": "",
            "error": f"Command '{tokens[0]}' is not allowed.",
        }

    if any(f in command for f in forbidden_files):
        logger.debug(f"Command '{command}' contains forbidden file access.")
        return {
            "ok": False,
            "stdout": "",
            "stderr": "",
            "error": "Access to certain files is forbidden.",
        }

    if "ls" in tokens and any(f in command for f in ["-la", "-l", "-al", "-R", "-all"]):
        logger.debug(
            f"Command '{command}' with flags is not allowed due to detailed listing. Please use 'ls' without flags."
        )
        return {
            "ok": False,
            "stdout": "",
            "stderr": "",
            "error": "'ls' with flags is not allowed as it breaks caching.",
        }
    try:
        result = subprocess.run(
            tokens,
            cwd=Path(cwd),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": "",
        }
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": "", "error": str(e)}
