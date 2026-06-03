import duckdb
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class QueryTool:
    """
    Run a SQL query against the imdb database.

    Args:
        sql: A string containing the SQL query to execute.

    Returns:
        A dictionary containing:
            columns: A list of column names returned by the query.
            rows: A list of tuples representing the rows returned by the query.
            row_count: The number of rows returned by the query.
    """

    def __init__(self):
        self.count = 0
        pass

    def run_duckdb_query(self, query: str) -> Dict[str, Any]:
        """
        Run a SQL query against the database using DuckDB.

        Args:
            query: A string containing the SQL query to execute.

        Returns:
            A dictionary containing:
                columns: A list of column names returned by the query.
                rows: A list of tuples representing the rows returned by the query.
                row_count: The number of rows returned by the query.
        """
        self.count += 1
        if self.count >= 150:
            logger.warning(
                f"Query count has reached {self.count}. Stopping to cap cost."
            )
            return "Maximum query count reached. Stopping to limit execution cost."
        DB_PATH = "data/imdb.duckdb"
        logger.debug(f"Executing SQL query: {query}")
        con = duckdb.connect(DB_PATH, read_only=True, config={"threads": 1})
        try:
            cursor = con.execute(query)
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
            logger.debug(f"Query returned {columns}, {rows}")
            return {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
            }
        except duckdb.Error as e:
            logger.error(f"Query execution failed. {e}")
            return {
                "error": str(e),
            }
        finally:
            con.close()

    def __call__(self, query: str) -> dict:
        return self.run_duckdb_query(query)
