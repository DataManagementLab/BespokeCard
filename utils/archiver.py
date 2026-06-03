"""Request archiver for logging LLM interactions to markdown files."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import os
from utils.plots import plot_optimization_progress, optimization_plot_helper

import logging

logger = logging.getLogger(__name__)


class Archiver:
    """Archives inputs and results for debugging and analysis.

    Args:
        log_dir: Directory where log files will be stored
    """

    def __init__(self, log_dir: str = ".logs"):
        self.log_dir = Path(log_dir)
        self.log_path: Path

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True, mode=0o777)

        # Initialize session
        self.renew_session()

    def renew_session(self) -> None:
        """Create a new log file with a timestamp."""
        date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_folder = f"log_{date_time_str}"
        self.log_path = self.log_dir / log_folder
        self.log_path.mkdir(parents=True, exist_ok=True, mode=0o777)

        logger.info(f"Started new logging session: {self.log_path}")

    def log_planner(
        self, system_prompt: str, planner_prompt: str, planner_response: str
    ) -> None:
        """Log the planner prompt and response to a markdown file."""
        md_lines = [
            "# Planner Interaction\n",
            "## System Prompt\n",
            "```text",
            system_prompt,
            "```\n",
            "## Prompt\n",
            "```text",
            planner_prompt,
            "```\n",
            "## Response\n",
            "```text",
            planner_response,
            "```\n",
        ]
        self._write_to_file(md_lines, file_name="planner_interaction.md")

    def log_coder(
        self, system_prompt: str, coder_prompt: str, coder_response: str
    ) -> None:
        """Log the coder prompt and response to a markdown file."""
        target_dir = self.log_path / "iteration_0"
        target_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
        md_lines = [
            "# Coder Interaction\n",
            "## System Prompt\n",
            "```text",
            system_prompt,
            "```\n",
            "## Prompt\n",
            "```text",
            coder_prompt,
            "```\n",
            "## Response\n",
            "```text",
            coder_response,
            "```\n",
        ]
        self._write_to_file(md_lines, file_name="iteration_0/coder_interaction.md")

        # copy plots folder into log path
        plots_dir = Path("plots")
        if plots_dir.exists() and plots_dir.is_dir():
            try:
                os.system(f"cp -r {plots_dir} {target_dir / 'plots'}")
                logger.info(
                    f"Copied plots directory to {target_dir / 'plots'}",
                )
            except Exception as e:
                logger.error(f"Failed to copy plots directory: {e}")
        # copy outputs folder into log path
        # outputs_dir = Path("outputs")
        # if outputs_dir.exists() and outputs_dir.is_dir():
        #    try:
        #        os.system(f"cp -r {outputs_dir} {self.log_path / 'outputs'}")
        #        logger.info(f"Copied outputs directory to {self.log_path / 'outputs'}")
        #    except Exception as e:
        #        logger.error(f"Failed to copy outputs directory: {e}")
        # copy card_estimator.py into log path
        estimator_file = Path("card_estimator.py")
        if estimator_file.exists() and estimator_file.is_file():
            try:
                os.system(f"cp {estimator_file} {target_dir / 'card_estimator.py'}")
                os.system(f"cp 'outputs/feedback.json' {target_dir / 'feedback.json'}")
                logger.info(
                    f"Copied card_estimator.py and feedback to {target_dir / 'card_estimator.py'}",
                )
            except Exception as e:
                logger.error(f"Failed to copy card_estimator.py and feedback: {e}")

    def log_optim_loop(self, iteration: int, prompt: str, response: str) -> None:
        """Log the optimization loop prompt and response to a markdown file."""
        (self.log_path / f"iteration_{iteration}").mkdir(
            parents=True, exist_ok=True, mode=0o777
        )

        md_lines = [
            f"# Optimization Loop - Iteration {iteration}\n",
            "## Prompt\n",
            "```text",
            prompt,
            "```\n",
            "## Response\n",
            "```text",
            response,
            "```\n",
        ]
        self._write_to_file(md_lines, file_name=f"iteration_{iteration}/update.md")
        # copy plots folder into log path
        plots_dir = Path("plots")
        if plots_dir.exists() and plots_dir.is_dir():
            try:
                os.system(
                    f"cp -r {plots_dir} {self.log_path / f'iteration_{iteration}' / 'plots'}"
                )
                logger.info(
                    f"Copied plots directory to {self.log_path / f'iteration_{iteration}' / 'plots'}",
                )
            except Exception as e:
                logger.error(f"Failed to copy plots directory: {e}")
        # copy outputs folder into log path
        # outputs_dir = Path("outputs")
        # if outputs_dir.exists() and outputs_dir.is_dir():
        #    try:
        #        os.system(
        #            f"cp -r {outputs_dir} {self.log_path / f'iteration_{iteration}' / 'outputs'}"
        #        )
        #        logger.info(
        #            f"Copied outputs directory to {self.log_path / f'iteration_{iteration}' / 'outputs'}"
        #        )
        #    except Exception as e:
        #        logger.error(f"Failed to copy outputs directory: {e}")
        # copy card_estimator.py into log path
        estimator_file = Path("card_estimator.py")
        if estimator_file.exists() and estimator_file.is_file():
            try:
                os.system(
                    f"cp {estimator_file} {self.log_path / f'iteration_{iteration}' / 'card_estimator.py'}"
                )
                os.system(
                    f"cp 'outputs/feedback.json' {self.log_path / f'iteration_{iteration}' / 'feedback.json'}"
                )
                logger.info(
                    f"Copied card_estimator.py and feedback to {self.log_path / f'iteration_{iteration}' / 'card_estimator.py'}",
                )
            except Exception as e:
                logger.error(f"Failed to copy card_estimator.py and feedback: {e}")

    def log_optimization_plot(self) -> None:
        """Generate and save optimization progress plot."""
        try:
            plot_optimization_progress(*optimization_plot_helper(str(self.log_path)))
            os.system(
                f"cp plots/optimization/optimization_progress.pdf {self.log_path / 'optimization_progress.pdf'}"
            )
            if Path("plots/e2e_execution_times.pdf").exists():
                os.system(
                    f"cp plots/e2e_execution_times.pdf {self.log_path / 'e2e_times.pdf'}"
                )
            logger.info("Optimization progress plot generated successfully.")
        except Exception as e:
            logger.error(f"Failed to generate optimization progress plot: {e}")

    def _write_to_file(self, md_lines: list[str], file_name: str) -> None:
        """Write markdown lines to the log file."""
        try:
            with open(self.log_path / file_name, "a", encoding="utf-8") as f:
                f.write("\n".join(md_lines))
            os.chmod(self.log_path / file_name, 0o777)
        except IOError as e:
            logger.error(
                f"Failed to write to log file {self.log_path / file_name}: {e}"
            )
            raise

    def load_feedbacks(self) -> Optional[str]:
        num_iterations = len(list(self.log_path.glob("iteration_*")))
        iterations = {}
        for it in range(num_iterations):
            iteration_dir = self.log_path / f"iteration_{it}"
            feedback_file = iteration_dir / "feedback.json"
            if feedback_file.exists():
                try:
                    with open(feedback_file, "r") as f:
                        feedback = json.load(f)
                        iterations[it] = feedback["q_error_percentiles"]["bespoke"]
                except Exception as e:
                    logger.error(f"Failed to read feedback file {feedback_file}: {e}")
        return iterations

    def load_implementation(self, iteration: int):
        implementation_file = (
            self.log_path / f"iteration_{iteration}" / "card_estimator.py"
        )
        if implementation_file.exists():
            try:
                with open(implementation_file, "r") as f:
                    implementation = f.read()
                with open("card_estimator.py", "w") as f:
                    f.write(implementation)
                logger.info(f"Loaded implementation from {implementation_file}")
            except Exception as e:
                logger.error(
                    f"Failed to read implementation file {implementation_file}: {e}"
                )
