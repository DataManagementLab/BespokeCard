from typing import Any

from agents.run_context import RunContextWrapper
from agents.tool import FunctionTool

from utils.tools.evaluate_tool import EvaluateTool


def make_openai_evaluate_tool(
    evaluate_tool: EvaluateTool,
    defer_loading: bool = False,
) -> FunctionTool:
    async def on_invoke(ctx: RunContextWrapper[Any], args_json: str) -> str:
        return evaluate_tool()

    return FunctionTool(
        name="evaluate_estimator",
        description="Evaluates the card_estimator code by a separate evaluation script. Returns a string indicating the evaluation result: 'success' if the script exits with return code 0. The captured stderr output if the script exits with a non-zero return code. 'Execution timed out' if the estimator takes longer than 600 seconds to setup.",
        params_json_schema={},
        on_invoke_tool=on_invoke,
        defer_loading=defer_loading,  # loaded when needed
    )
