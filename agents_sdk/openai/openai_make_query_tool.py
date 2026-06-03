from typing import Any

from agents.run_context import RunContextWrapper
from agents.tool import FunctionTool
from pydantic import BaseModel, Field

from utils.tools.query_tool import QueryTool


class QueryArgs(BaseModel):
    query: str = Field(..., description="SQL query to run against the database")


def make_openai_query_tool(
    query_tool: QueryTool,
    defer_loading: bool = False,
) -> FunctionTool:
    async def on_invoke(ctx: RunContextWrapper[Any], args_json: str) -> str:
        args = QueryArgs.model_validate_json(args_json)
        return query_tool(query=args.query)

    return FunctionTool(
        name="query_db",
        description="Run a SQL query against the imdb database. Takes query as string argument. Returns a dictionary containing: columns: A list of column names returned by the query. rows: A list of tuples representing the rows returned by the query. row_count: The number of rows returned by the query.",
        params_json_schema=QueryArgs.model_json_schema(),
        on_invoke_tool=on_invoke,
        defer_loading=defer_loading,  # loaded when needed
    )
