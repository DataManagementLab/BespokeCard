import logging
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Any


logger = logging.getLogger(__name__)


class CustomShellTool:
    def __init__(
        self,
        cwd,
    ):
        self.cwd = cwd
        self.allowed_commands = {"ls", "cat", "sed", "head", "tail", "wc"}
        self.forbidden_files = {
            "job",
            "env",
            "sql",
            "evaluate",
            "llm_cache",
            "outputs",
        }

    def __call__(self, command: str) -> str:
        logger.info(f"Running shell command: {command}")

        tokens = shlex.split(command)
        if not tokens or tokens[0] not in self.allowed_commands:
            logger.debug(f"Command '{command}' is not allowed.")
            return {
                "ok": False,
                "stdout": "",
                "stderr": "",
                "error": f"Command '{tokens[0]}' is not allowed.",
            }

        if any(f in command for f in self.forbidden_files):
            logger.debug(f"Command '{command}' contains forbidden file access.")
            return {
                "ok": False,
                "stdout": "",
                "stderr": "",
                "error": "Access to certain files is forbidden.",
            }

        if "ls" in tokens and any(
            f in command for f in ["-la", "-l", "-al", "-R", "-all"]
        ):
            logger.debug(
                f"Command '{command}' with flags is not allowed due to detailed listing. Please use 'ls' without flags."
            )
            return {
                "ok": False,
                "stdout": "",
                "stderr": "",
                "error": "'ls' with flags is not allowed as it breaks caching.",
            }
        try:
            result = subprocess.run(
                tokens,
                cwd=Path(self.cwd),
                capture_output=True,
                text=True,
                timeout=5,
            )
            return {
                "ok": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": "",
            }
        except Exception as e:
            return {"ok": False, "stdout": "", "stderr": "", "error": str(e)}
