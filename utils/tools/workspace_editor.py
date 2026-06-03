import logging
from pathlib import Path

from agents import apply_diff, custom_span
from agents.editor import ApplyPatchOperation, ApplyPatchResult


logger = logging.getLogger(__name__)


class WorkspaceEditor:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def create_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        with custom_span(
            f"create file ({operation.path})",
            {
                "path": operation.path,
                "diff": operation.diff[:2000] if operation.diff else None,
            },
        ):
            relative = self._relative_path(operation.path)
            target = self._resolve(operation.path, ensure_parent=True)
            logger.info(f"Creating: {target}")
            diff = operation.diff or ""
            content = apply_diff("", diff, mode="create")
            target.write_text(content, encoding="utf-8")

            return ApplyPatchResult(output=f"Created {relative}")

    def update_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        with custom_span(
            f"update file ({operation.path})",
            {
                "file": operation.path,
                "diff": operation.diff[:2000] if operation.diff else None,
            },
        ):
            relative = self._relative_path(operation.path)
            target = self._resolve(operation.path)
            logger.info(f"Updating: {target}")
            original = target.read_text(encoding="utf-8")
            diff = operation.diff or ""
            patched = apply_diff(original, diff)
            target.write_text(patched, encoding="utf-8")
            return ApplyPatchResult(output=f"Updated {relative}")

    def delete_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        with custom_span(
            f"delete file ({operation.path})",
            {
                "file": operation.path,
                "diff": operation.diff[:2000] if operation.diff else None,
            },
        ):
            relative = self._relative_path(operation.path)
            target = self._resolve(operation.path)
            logger.info(f"Deleting: {target}")
            original = target.read_text(encoding="utf-8")
            target.unlink(missing_ok=True)
            return ApplyPatchResult(output=f"Deleted {relative}")

    def _relative_path(self, value: str) -> str:
        resolved = self._resolve(value)
        return resolved.relative_to(self._root).as_posix()

    def _resolve(self, relative: str, ensure_parent: bool = False) -> Path:
        candidate = Path(relative)
        target = candidate if candidate.is_absolute() else (self._root / candidate)
        target = target.resolve()
        # Only allow files directly in the root directory (no subdirectories)
        if target.parent != self._root:
            raise RuntimeError(
                f"Operation outside allowed root dir (no subdirs): {relative}"
            )
        if str(candidate) != "card_estimator.py":
            raise RuntimeError(
                f"Operation only allowed on card_estimator.py: {relative}"
            )
        try:
            target.relative_to(self._root)
        except ValueError:
            raise RuntimeError(f"Operation outside workspace: {relative}") from None
        if ensure_parent:
            target.parent.mkdir(parents=True, exist_ok=True)
        return target
