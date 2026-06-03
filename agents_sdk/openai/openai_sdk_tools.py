import logging
from typing import Any, Dict
from agents.run_context import RunContextWrapper
from agents.tool import FunctionTool
from pydantic import BaseModel, Field

from utils.tools.custom_apply_patch import CustomApplyPatchTool
from utils.tools.workspace_editor import WorkspaceEditor

logger = logging.getLogger(__name__)


class CustomApplyPatchArgs(BaseModel):
    type: str = Field(..., description="create_file, update_file, or delete_file")
    path: str = Field(..., description="Path relative to workspace root")
    diff: str | None = Field(None, description="Unified diff for create/update")


def make_custom_openai_apply_patch_tool(editor: WorkspaceEditor) -> FunctionTool:
    impl = CustomApplyPatchTool(editor=editor)

    async def on_invoke(ctx: RunContextWrapper[Any], args_json: str) -> str:
        args = CustomApplyPatchArgs.model_validate_json(args_json)
        return await impl(args.type, args.path, args.diff)

    return FunctionTool(
        name="apply_patch",
        description="Applies a unified diff to create/update/delete a file",
        params_json_schema=CustomApplyPatchArgs.model_json_schema(),
        on_invoke_tool=on_invoke,
        defer_loading=False,  # always shown to the model
    )
