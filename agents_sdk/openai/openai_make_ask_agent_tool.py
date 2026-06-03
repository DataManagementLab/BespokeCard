from typing import Any

from agents.run_context import RunContextWrapper
from agents.tool import FunctionTool
from pydantic import BaseModel, Field

from utils.tools.ask_agent_tool import AskAgentTool


class QuestionArgs(BaseModel):
    question: str = Field(..., description="Question to ask the other agent")


def make_openai_ask_agent_tool(
    ask_agent_tool: AskAgentTool,
    defer_loading: bool = False,
) -> FunctionTool:
    async def on_invoke(ctx: RunContextWrapper[Any], args_json: str) -> str:
        args = QuestionArgs.model_validate_json(args_json)
        return await ask_agent_tool(message=args.question)

    return FunctionTool(
        name="ask_agent",
        description="Ask the planning agent a question. Takes a string argument containing the question to ask. Returns the planning agent's response as a string.",
        params_json_schema=QuestionArgs.model_json_schema(),
        on_invoke_tool=on_invoke,
        defer_loading=defer_loading,  # loaded when needed
    )
