import inspect
import logging
from typing import Any, Callable
from pathlib import Path
from agents import (
    Agent,
    ApplyPatchTool,
    ModelBehaviorError,
    ModelSettings,
    Runner,
    ShellTool,
    Tool,
    ToolSearchTool,
    SQLiteSession,
    trace,
)
from agents.extensions.memory import AdvancedSQLiteSession

from agents_sdk.llm.cached_litellm import CachedLitellmModel
from agents_sdk.llm.cached_openai import CachedOpenAIResponsesModel
from agents_sdk.openai.openai_make_evaluate_tool import make_openai_evaluate_tool
from agents_sdk.openai.openai_make_query_tool import make_openai_query_tool
from agents_sdk.openai.openai_make_shell_tool import make_openai_shell_tool
from agents_sdk.openai.openai_make_ask_agent_tool import make_openai_ask_agent_tool
from agents_sdk.openai.openai_sdk_tools import (
    make_custom_openai_apply_patch_tool,
)
from agents_sdk.openai.openai_token_usage import (
    openai_get_tokens_context_and_dollar_info,
)
from agents_sdk.sdk_wrapper import SDKWrapper

from utils.model_setup import setup_model_config
from utils.tools.workspace_editor import WorkspaceEditor
from utils.tools.shell_tool import CustomShellTool
from utils.tools.evaluate_tool import EvaluateTool
from utils.tools.query_tool import QueryTool
from utils.tools.ask_agent_tool import AskAgentTool

logger = logging.getLogger(__name__)


class OpenAIAgentsSDKWrapper(SDKWrapper):
    def __init__(
        self,
        model: str,
        agent_name: str,
        conv_name: str,
        editor: WorkspaceEditor,
        shell_tool: CustomShellTool,
        evaluate_tool: EvaluateTool,
        query_tool: QueryTool,
        cache_path: str | Path,
        workspace_path: str,
        workspace_path_absolute: Path,
        tool_search_tool: bool = False,
        do_not_cache: bool = False,
        stop_on_cache_miss: bool = False,
        ask_agent_tool: AskAgentTool | None = None,
    ):
        super().__init__(
            sdk="OpenAIAgentsSDK",
            model=model,
            agent_name=agent_name,
            conv_name=conv_name,
            editor=editor,
            shell_tool=shell_tool,
            evaluate_tool=evaluate_tool,
            query_tool=query_tool,
            cache_path=Path(cache_path),
            workspace_path=workspace_path,
            workspace_path_absolute=workspace_path_absolute,
        )

        use_litellm, model_name, api_key, openai_client, api_base = setup_model_config(
            self.model
        )

        openai_evaluate_tool = make_openai_evaluate_tool(
            evaluate_tool=self.evaluate,
            defer_loading=tool_search_tool,  # if tool search tool is included, we want to load the evaluate tool in deferred loading mode, so that it is not loaded at the beginning of the conversation and does not take up context space and resources before it is actually needed. The tool search tool will load it when needed.
        )

        openai_query_tool = make_openai_query_tool(
            query_tool=self.query,
            defer_loading=tool_search_tool,  # if tool search tool is included, we want to load the run tool in deferred loading mode, so that it is not loaded at the beginning of the conversation and does not take up context space and resources before it is actually needed. The tool search tool will load it when needed.
        )

        openai_shell_tool = make_openai_shell_tool(
            shell_tool=self.shell,
            defer_loading=tool_search_tool,  # if tool search tool is included, we want to load the shell tool in deferred loading mode, so that it is not loaded at the beginning
        )

        if ask_agent_tool is not None:
            openai_ask_agent_tool = make_openai_ask_agent_tool(
                ask_agent_tool=ask_agent_tool,
                defer_loading=tool_search_tool,  # if tool search tool is included, we want to load the ask agent tool in deferred loading mode, so that it is not loaded at the beginning of the conversation and does not take up context space and resources before it is actually needed. The tool search tool will load it when needed.
            )
        else:
            openai_ask_agent_tool = None

        # assemble tools
        if not use_litellm:
            apply_patch = ApplyPatchTool(editor=self.editor)
        else:
            apply_patch = make_custom_openai_apply_patch_tool(editor=self.editor)

        if openai_ask_agent_tool is not None:  # coding agents tools
            self.tools: list[Tool] = [
                apply_patch,
                openai_shell_tool,
                openai_evaluate_tool,
                openai_ask_agent_tool,
            ]
        else:
            self.tools: list[Tool] = [openai_query_tool]  # planning agent tools

        if tool_search_tool:
            logger.info("Utilizing tool search tool.")
            self.tools.append(ToolSearchTool())

        #########################
        # Prepare Model and Agent
        #########################

        self.underlying_session = AdvancedSQLiteSession(
            session_id=self.conv_name, create_tables=True
        )

        # assemble session
        self.session = SQLiteSession(f"{self.conv_name}")

        if openai_ask_agent_tool is not None:
            instructions = [
                "You can edit card_estimator.py using the apply_patch tool. ",
                "You can run read-only shell commands using the shell tool. ",
                "You can test your implementation using the evaluate tool. ",
                "You can ask the planning agent for help using the ask_agent tool. ",
            ]
            cache_dir = self.cache_path / "coding_agent_cache"
        else:
            instructions = [
                "You can run queries on the database using the query tool. ",
            ]
            cache_dir = self.cache_path / "planning_agent_cache"
        if use_litellm:
            self.model = CachedLitellmModel(
                model=model_name,
                api_key=api_key,
                **({"base_url": api_base} if api_base else {}),
                llm_cache_dir=cache_dir,
                do_not_cache=do_not_cache,
                stop_on_cache_miss=stop_on_cache_miss,
                tools_loaded_deferred=tool_search_tool,  # if tool search tool is included, we want to load the litellm wrapper in deferred loading mode, so that it is not loaded at the beginning of the conversation and does not take up context space and resources before it is actually needed. The tool search tool will load it when needed.
                working_dir=self.workspace_path_absolute,
            )

        else:
            self.model = CachedOpenAIResponsesModel(
                model=model_name,
                openai_client=openai_client,
                llm_cache_dir=cache_dir,
                do_not_cache=do_not_cache,
                stop_on_cache_miss=stop_on_cache_miss,
                tools_loaded_deferred=tool_search_tool,  # add this info to llm cache
                working_dir=self.workspace_path_absolute,
            )
            instructions = [
                "You are an autonomous agent. Run independently. Don't ask the user questions - try to figure unclear things out by your own. If you encounter errors or negative feedback from tools, fix them immediately without user confirmation. ",
            ] + instructions

        model_settings = ModelSettings(tool_choice="auto", parallel_tool_calls=False)
        if use_litellm:
            model_settings = ModelSettings(
                tool_choice="auto", include_usage=True, parallel_tool_calls=False
            )

        self.agent = Agent(
            name=self.agent_name,
            model=self.model,
            instructions="".join(instructions),
            tools=self.tools,
            model_settings=model_settings,
        )

        logger.info(
            f"Using model: {self.model} {'(via litellm wrapper)' if use_litellm else ''}"
        )

    def get_total_saved_by_llm_cache(self) -> float:
        return self.model.total_saved

    async def run_agent(
        self,
        prompt: str,
        max_turns: int,
        short_desc: str | None = None,
    ) -> str:
        ## 2026/03/14: Experimentation with prompt_cache_key for openai models - no increased caching ratio observed.
        # # extract model name
        # if isinstance(model, str):
        #     model_name = model
        # else:
        #     model_name = str(model)

        # # assemble prompt-cache-key for openai
        # agent_model_settings = agent.model_settings
        # if model_name.startswith("openai/") or "gpt-" in model_name.lower():
        #     assert agent_model_settings is not None, (
        #         "Model settings must be provided for OpenAI models to use prompt caching"
        #     )

        #     # add prompt cache key
        #     extra_args = (
        #         agent_model_settings.extra_args
        #         if agent_model_settings.extra_args is not None
        #         else {}
        #     )
        #     extra_args = extra_args.copy()  # make a copy to avoid mutating original

        #     extra_args["prompt_cache_key"] = f"{model_name}:{idx}"

        #     model_settings = copy(agent_model_settings)
        #     model_settings.extra_args = extra_args
        # else:
        #     model_settings = agent_model_settings

        # Rename the agent for each stage based on the short description - this makes it easier to analyze the tracing logs and see which stage is producing which output, without having to rely on the prompt content which might be very long. The name will be reset to default_agent_name if short_desc is None, which is the case for normal prompts that are not associated with a specific stage.
        # We rewrite it to hack a different header for each stage into the tracing log.
        # THIS IS RISKY: if openai somehow refers to agent.name this is a problem, since it will be not an identifier anymore.
        if short_desc is None:
            workflow_name = self.conv_name
        else:
            workflow_name = f"{self.conv_name} ({short_desc})"
        self.agent.name = workflow_name

        try:
            result = await Runner.run(
                self.agent,
                input=prompt,
                session=self.session,
                max_turns=max_turns,
                # run_config=RunConfig(model_settings=model_settings),
            )
        except ModelBehaviorError as e:
            logger.error(f"Error runing agent: {prompt}\n{str(e)}")
            raise e

        # Log cost summary
        openai_get_tokens_context_and_dollar_info(
            result.context_wrapper.usage,
            str(self.model),
            last_entry_only=False,
            log=True,
        )

        return result.final_output

    async def get_conversation_turns(self) -> int:
        turns = await self.underlying_session.get_conversation_turns()
        if len(turns) == 0:
            return 0

        return turns[-1]["turn"]

    async def switch_to_conversation_branch(self, branch_name: str):
        # switch branch in underlying session

        branches = await self.underlying_session.list_branches()
        if len(branches) == 0:
            return  # no branches to switch to, likely the case at the beginning of the conversation - do nothing

        branch_names = [b["branch_id"] for b in branches]
        if branch_name not in branch_names:
            logger.error(
                f"Branch {branch_name} not found in underlying session. Available branches: {branches}"
            )
            raise Exception(f"Branch {branch_name} not found in underlying session.")

        await self.underlying_session.switch_to_branch(branch_name)

    async def list_conversation_branches(self) -> list[str]:
        branches = await self.underlying_session.list_branches()
        branch_names = [b["branch_id"] for b in branches]
        return branch_names

    async def create_conversation_branch_from_turn(
        self, branch_name: str, turn_nr: int
    ) -> str:
        # create branch from turn in underlying session
        return await self.underlying_session.create_branch_from_turn(
            turn_nr, branch_name=branch_name
        )

    def last_llm_call_was_cached(self) -> bool:
        return self.model.llm_was_cached
