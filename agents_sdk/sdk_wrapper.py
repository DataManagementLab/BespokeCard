from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable

# from pipeline.runtime_tracker import RuntimeTracker
from utils.tools.workspace_editor import WorkspaceEditor
from utils.tools.shell_tool import CustomShellTool
from utils.tools.evaluate_tool import EvaluateTool
from utils.tools.query_tool import QueryTool

# from utils.logging_and_reporting.run_stats_collector import RunStatsCollector


class SDKWrapper:
    def __init__(
        self,
        sdk: str,
        model: str,
        agent_name: str,
        conv_name: str,
        editor: WorkspaceEditor,
        shell_tool: CustomShellTool,
        evaluate_tool: EvaluateTool,
        query_tool: QueryTool,
        cache_path: Path,
        workspace_path: str,
        workspace_path_absolute: Path,
    ):
        self._sdk = sdk
        self.model = model
        self.agent_name = agent_name
        self.conv_name = conv_name
        self.editor = editor
        self.shell = shell_tool
        self.evaluate = evaluate_tool
        self.query = query_tool
        self.cache_path = cache_path
        self.workspace_path = workspace_path
        self.workspace_path_absolute = workspace_path_absolute

        assert not Path(self.workspace_path).is_absolute(), (
            "workspace_path must be a relative path - otherwise caches across different machines/users would not be portable at all"
        )
        assert self.workspace_path_absolute.is_absolute(), (
            "workspace_path_absolute must be an absolute path - it is used for security checks to ensure that the agent does not access files outside of the working directory, so it needs to be an absolute path to do proper checks"
        )

        # has to exist
        assert workspace_path_absolute.exists(), (
            f"workspace_path_absolute {workspace_path_absolute} does not exist - it needs to exist for security checks to work properly"
        )

    def __getattr__(self, item):
        return getattr(self._sdk, item)

    @abstractmethod
    def get_total_saved_by_llm_cache(self) -> float:
        pass

    @abstractmethod
    async def run_agent(
        self,
        prompt: str,
        max_turns: int,
        short_desc: str | None = None,
    ) -> str:
        pass

    @abstractmethod
    async def get_conversation_turns(self) -> int:
        pass

    @abstractmethod
    async def switch_to_conversation_branch(self, branch_name: str):
        pass

    @abstractmethod
    async def create_conversation_branch_from_turn(
        self, branch_name: str, turn_nr: int
    ) -> str:
        pass

    @abstractmethod
    def last_llm_call_was_cached(self) -> bool:
        pass
