import json
import os
import dotenv
from pathlib import Path
import asyncio
import logging
import subprocess

from utils.tools.evaluate_tool import EvaluateTool
from utils.tools.query_tool import QueryTool
from utils.tools.shell_tool import CustomShellTool
from utils.tools.ask_agent_tool import AskAgentTool
from utils.tools.workspace_editor import WorkspaceEditor
from agents_sdk.openai.openai_sdk import OpenAIAgentsSDKWrapper
from utils.archiver import Archiver
from utils.resource_tracker import ResourceTracker

dotenv.load_dotenv()


logger = logging.getLogger(__name__)


def run_eval(flags: list[str] | None = None):
    venv_path = os.getenv("VENV_PATH")
    try:
        result = subprocess.run(
            [venv_path, "evaluate.py"] + (flags or []),
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        logging.error("Evaluation timed out")


def get_planner_prompt():
    with open("prompts/prompt_planner.txt", "r") as f:
        prompt = f.read()
    with open("data/all_queries.sql", "r") as f:
        queries = f.read()
    with open("data/schema.sql", "r") as schema:
        schema = schema.read()
    with open("data/row_counts.json", "r") as stats:
        n_rows = json.load(stats)
    with open("data/unique_vals.json", "r") as stats:
        n_unique = json.load(stats)
    return prompt.format(
        schema=schema, row_counts=n_rows, unique_vals=n_unique, queries=queries
    )


def get_coder_prompt():
    with open("prompts/prompt_coder.txt", "r") as f:
        prompt = f.read()
    with open("outputs/statistics_plan.txt", "r") as stats:
        stats = stats.read()
    return prompt.replace("{statistics}", stats)


def coding_loop():
    archiver = Archiver()
    editor = WorkspaceEditor(root=Path("."))
    tracker_planner = ResourceTracker("Planning Agent")
    tracker_coder = ResourceTracker("Coding Agent")
    tracker_planner.request_type = "create_plan"
    tracker_coder.request_type = "implement_estimator"
    os.environ["no_tqdm"] = (
        "true"  # disable tqdm for agent runs as timing is not consistent and breaks caching.
    )
    rounds = 1  # to keep track of total rounds
    initial_rounds = 2
    join_rounds = 5
    filter_rounds = 5
    final_rounds = 2

    planner = OpenAIAgentsSDKWrapper(
        model="gpt-5.4",
        agent_name="Planning Agent",
        conv_name="test_conversation",
        editor=editor,
        shell_tool=CustomShellTool("."),
        evaluate_tool=EvaluateTool(),
        query_tool=QueryTool(),
        cache_path=Path("cache"),
        workspace_path=".",
        workspace_path_absolute=Path(".").resolve(),
        tool_search_tool=False,
        do_not_cache=False,
        stop_on_cache_miss=False,
    )

    coder = OpenAIAgentsSDKWrapper(
        model="gpt-5.4",  # anthropic/claude-sonnet-4-6
        agent_name="Coding Agent",
        conv_name="test_conversation",
        editor=editor,
        shell_tool=CustomShellTool("."),
        evaluate_tool=EvaluateTool(),
        query_tool=QueryTool(),
        cache_path=Path("cache"),
        workspace_path=".",
        workspace_path_absolute=Path(".").resolve(),
        tool_search_tool=False,
        do_not_cache=False,
        stop_on_cache_miss=False,
        ask_agent_tool=AskAgentTool(target_agent=planner),
    )

    plan = asyncio.run(
        planner.run_agent(
            get_planner_prompt(),
            40,
            "",
        )
    )
    with open("outputs/statistics_plan.txt", "w") as f:
        f.write(plan)

    archiver.log_planner("", get_planner_prompt(), plan)
    response = asyncio.run(
        coder.run_agent(
            get_coder_prompt(),
            30,
            "",
        )
    )
    archiver.log_coder("", get_coder_prompt(), response)
    logging.info(f"Coding done, response: {response}")

    tracker_planner.request_type = "coder_question"
    tracker_coder.request_type = "initial_rounds"
    for round in range(initial_rounds):
        with open("outputs/feedback.json", "r") as f:
            feedback = json.load(f)
        if feedback == {}:
            prompt = f"There is no feedback yet. This means you did not successfully run the evaluate tool. Either you did not call it, your implementation produced an error or it timed out. Please fix this with the apply_patch tool and run evaluate again."
        elif (
            feedback.get("estimator_size") is not None
            and float(feedback["estimator_size"].split(" ")[0]) > 1000.0
        ):
            prompt = f"The evaluate tool ran but the estimator consumes too much memory. The current size is {feedback['estimator_size']} while it should be below 1000 MB. Please adapt the estimator and make sure you are only storing the requested statistics per column. Afterward, run evaluate again."
        else:
            break
        # add requirements for computation time (setup and estimate separately)

        response = asyncio.run(
            coder.run_agent(
                prompt,
                20,
                "",
            )
        )
        logging.info(
            f"Initial optimization round {round} completed, response: {response}"
        )
        archiver.log_optim_loop(rounds, prompt, response)
        rounds += 1

    tracker_coder.request_type = "join_rounds"
    # join eval
    for round in range(join_rounds):
        run_eval(["--skip_setup", "--no_filters"])
        with open("outputs/feedback.json", "r") as f:
            feedback = json.load(f)
        if round == 0:
            # prompt = f"Now we'll focus on the join logic by isolating the queries that contain no filters but multiple tables. As previously, analyze the feedback to guide your next steps. Update the estimator with the apply_patch tool and run evaluate.\n {feedback}"
            prompt = f"Great, the implementation is running! Now we want to improve the estimation accuracy with targeted improvements. We'll start by evaluating the join logic by isolating the queries that contain no filters but multiple tables. Analyze the feedback to guide your next steps. Update the estimator with the apply_patch tool and run evaluate.\n {feedback}"
            # prompt += "If your estimates are worse than Postgres, the following strategies can help improve accuracy:\n1. Implement a Join Graph: Don't just multiply values. Process joins one by one.\n2. Prioritize PK-FK: Use the self.PK_FK dictionary to detect when a join shouldn't reduce the cardinality significantly (beyond the selectivity of the dimension table)."
        elif round == join_rounds - 1:
            prompt = f"This was the last round focusing on joins. Before moving on you can check the feedback and revert if required, otherwise don't update or run evaluate.\n {feedback}"
        else:
            prompt = f"Here is the feedback after the last change, analyze it and decide on the next step. In case the last change made things worse, revert it before continuing and don't forget to run evaluate again.\n {feedback}"

        response = asyncio.run(
            coder.run_agent(
                prompt,
                20,
                "",
            )
        )
        logging.info(f"Join optimization round {round} completed, response: {response}")
        run_eval(["--skip_setup"])
        archiver.log_optim_loop(rounds, prompt, response)
        rounds += 1

    tracker_coder.request_type = "filter_rounds"
    # single table eval
    for round in range(filter_rounds):
        run_eval(["--skip_setup", "--no_joins"])
        with open("outputs/feedback.json", "r") as f:
            feedback = json.load(f)
        if round == 0:
            # prompt = f"Great, the implementation is running! Now we want to improve the estimation accuracy with targeted improvements. We'll start by evaluating the filter performance by looking at single table queries only. Analyze the following feedback to guide your next steps. Update the estimator with the apply_patch tool and run evaluate.\n {feedback}"
            prompt = f"Now we'll focus on the filter performance by looking at single table queries only. Analyze the following feedback to guide your next steps. Validate every relevant column has a basic statistic (e.g. for single column queries). Update the estimator with the apply_patch tool and run evaluate.\n {feedback}"
            # prompt = f"Now we'll focus on the filter performance by looking at single filter queries only. Analyze the following feedback to guide your next steps. Update the estimator with the apply_patch tool and run evaluate.\n {feedback}"
        elif round == filter_rounds - 1:
            prompt = f"This was the last round focusing on filters. Before moving on you can check the feedback and revert if required, otherwise don't update or run evaluate.\n {feedback}"
        else:
            prompt = f"Here is the feedback after the last change, analyze it and decide on the next step. In case the last change made things worse, revert it before continuing and don't forget to run evaluate again.\n {feedback}"

        response = asyncio.run(
            coder.run_agent(
                prompt,
                20,
                "",
            )
        )
        logging.info(
            f"Filter optimization round {round} completed, response: {response}"
        )
        run_eval(["--skip_setup"])
        archiver.log_optim_loop(rounds, prompt, response)
        rounds += 1

    tracker_coder.request_type = "final_rounds"
    for round in range(final_rounds):
        with open("outputs/feedback.json", "r") as f:
            feedback = json.load(f)
        if round == 0:
            prompt = f"Great. Lastly we will look at all plans to see how filters + joins perform together. As previously, analyze the feedback and pay special attention to outliers to guide your next steps. Update the estimator with the apply_patch tool and run evaluate.\n {feedback}"
        elif round == final_rounds - 1:
            prompt = f"This was the last round focusing on the final optimization. Before moving on you can check the feedback and revert if required, otherwise don't update or run evaluate.\n {feedback}"
        else:
            prompt = f"Here is the feedback after the last change, analyze it and pay special attention to outliers to decide on the next step. In case the last change made things worse, revert it before continuing and don't forget to run evaluate again.\n {feedback}"

        response = asyncio.run(
            coder.run_agent(
                prompt,
                20,
                "",
            )
        )
        logging.info(
            f"Final optimization round {round} completed, response: {response}"
        )
        archiver.log_optim_loop(rounds, prompt, response)
        rounds += 1

    # identify best implementation by analyzing feedback in logs
    prompt = "The coding agent implemented the estimator and refined it in multiple iterations. Now you have to identify the best implementation by looking at the global performance (q_error_percentiles) of each iteration provided below. Return nothing but the index of the best implementation.\n\n"
    prompt += str(archiver.load_feedbacks())
    tracker_planner.request_type = "identify_best"

    idx_best = asyncio.run(
        planner.run_agent(
            prompt,
            20,
            "",
        )
    )
    logger.info(f"Best implementation identified, index: {idx_best}")
    archiver.load_implementation(int(idx_best))

    # run end 2 end evaluation after optimization rounds
    # run_eval(["--end2end"])
    archiver.log_optimization_plot()
    tracker_coder.dump(f"{archiver.log_path}/coder_usage.json")
    tracker_planner.dump(f"{archiver.log_path}/planner_usage.json")
    return
