import os
import subprocess
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class EvaluateTool:
    """
    Evaluates the card_estimator code by a separate evaluation script.

    Returns:
        A string indicating the evaluation result:
            - "success" if the script exits with return code 0.
            - The captured stderr output if the script exits with a non-zero return code.
            - "Execution timed out" if the process exceeds the time limit of 600 seconds.
    """

    def __init__(self):
        pass

    def __call__(self) -> str:
        logger.info("Running evaluation tool.")
        try:
            venv_path = os.getenv("VENV_PATH")
            result = subprocess.run(
                [venv_path, "evaluate.py"],
                capture_output=True,
                text=True,
                timeout=600,
            )
            logger.info(
                f"Code execution stdout:\n{result.stdout}\nCode execution stderr:\n{result.stderr}"
            )
            if result.returncode == 0:
                return "success"
            else:
                return result.stderr
        except subprocess.TimeoutExpired:
            logger.info("Code execution timed out after 600 seconds.")
            return "Execution timed out. Your code is taking too long to run. Please optimize it to run faster while maintaining or improving accuracy. If necessary, discuss with the planner what statistics can be adapted to be more efficient."
