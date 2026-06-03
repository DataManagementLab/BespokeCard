from typing import Any

from agents.run_context import RunContextWrapper
from agents.tool import FunctionTool
from pydantic import BaseModel, Field

from utils.tools.shell_tool import CustomShellTool


class ShellArgs(BaseModel):
    command: str = Field(..., description="Shell command to execute")


def make_openai_shell_tool(
    shell_tool: CustomShellTool,
    defer_loading: bool = False,
) -> FunctionTool:
    async def on_invoke(ctx: RunContextWrapper[Any], args_json: str) -> str:
        args = ShellArgs.model_validate_json(args_json)
        return shell_tool(command=args.command)

    return FunctionTool(
        name="shell",
        description="Run a restricted shell command for project exploration. Some files are blocked from being read due to their large size or privacy. Allowed commands: ls, cat, sed, head, tail, wc. Please use 'ls' without flags like ['-la', '-l', '-al', '-R', '-all'] as this might break caching.",
        params_json_schema=ShellArgs.model_json_schema(),
        on_invoke_tool=on_invoke,
        defer_loading=defer_loading,  # loaded when needed
    )
