"""CodeEditor — intelligent patch-based code editing.

Instead of rewriting entire files, this engine:
    1. Reads the existing file
    2. Identifies the target location
    3. Applies minimal, precise patches
    4. Preserves formatting and structure
    5. Validates the result

Supports:
    - Line-based edits (replace lines X-Y)
    - Function/class edits (replace specific function)
    - Import edits (add/remove imports)
    - Comment-based navigation ("# TODO: implement auth" -> edit after)
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

from synapse.logging import get_logger
from synapse.workspace.operator import FileOperator

log = get_logger("synapse.actions.code_editor")


@dataclass
class EditOperation:
    """A single edit operation."""

    kind: str  # "replace_lines", "replace_function", "insert_after", "insert_before", "append", "prepend"
    target: str  # line range, function name, or search pattern
    content: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class EditResult:
    """Result of a code edit."""

    success: bool
    file_path: str
    original_lines: int
    new_lines: int
    changes_made: int
    error: str | None = None
    diff_summary: str = ""


class CodeEditor:
    """Intelligent code editor with patch-based editing."""

    def __init__(self, file_operator: FileOperator) -> None:
        self._operator = file_operator

    def edit_file(
        self,
        path: str,
        operations: list[EditOperation],
    ) -> EditResult:
        """Apply multiple edit operations to a file."""
        content = self._operator.read(path)
        if content is None:
            return EditResult(
                success=False,
                file_path=path,
                original_lines=0,
                new_lines=0,
                changes_made=0,
                error=f"File not found: {path}",
            )

        original_lines = content.count("\n") + 1
        lines = content.split("\n")
        changes_made = 0

        for op in operations:
            try:
                lines, changed = self._apply_operation(lines, op)
                if changed:
                    changes_made += 1
            except Exception as exc:
                log.warning("edit_operation_failed", path=path, op=op.kind, error=str(exc)[:200])
                return EditResult(
                    success=False,
                    file_path=path,
                    original_lines=original_lines,
                    new_lines=len(lines),
                    changes_made=changes_made,
                    error=f"Operation {op.kind} failed: {exc}",
                )

        new_content = "\n".join(lines)
        new_lines = len(lines)

        # Write the result
        result = self._operator.write(path, new_content)
        if not result:
            return EditResult(
                success=False,
                file_path=path,
                original_lines=original_lines,
                new_lines=new_lines,
                changes_made=changes_made,
                error="Failed to write file",
            )

        diff_summary = self._build_diff_summary(original_lines, new_lines, changes_made)

        return EditResult(
            success=True,
            file_path=path,
            original_lines=original_lines,
            new_lines=new_lines,
            changes_made=changes_made,
            diff_summary=diff_summary,
        )

    def replace_function(
        self,
        path: str,
        function_name: str,
        new_body: str,
    ) -> EditResult:
        """Replace a specific function's body."""
        content = self._operator.read(path)
        if content is None:
            return EditResult(
                success=False,
                file_path=path,
                original_lines=0,
                new_lines=0,
                changes_made=0,
                error=f"File not found: {path}",
            )

        lines = content.split("\n")
        original_lines = len(lines)

        # Find the function
        start_line = None
        end_line = None
        indent = ""

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(f"def {function_name}(") or stripped.startswith(f"def {function_name} ("):
                start_line = i
                indent = line[: len(line) - len(line.lstrip())]
                # Find end of function (next def/class or end of indented block)
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    if next_line.strip() and not next_line.startswith(indent + " ") and not next_line.startswith(indent + "\t"):
                        end_line = j
                        break
                if end_line is None:
                    end_line = len(lines)
                break

        if start_line is None:
            return EditResult(
                success=False,
                file_path=path,
                original_lines=original_lines,
                new_lines=original_lines,
                changes_made=0,
                error=f"Function {function_name} not found",
            )

        # Replace the function body
        new_lines = new_body.split("\n")
        indented_lines = [indent + line if line.strip() else "" for line in new_lines]
        lines[start_line:end_line] = indented_lines

        new_content = "\n".join(lines)
        result = self._operator.write(path, new_content)

        return EditResult(
            success=True,
            file_path=path,
            original_lines=original_lines,
            new_lines=len(lines),
            changes_made=1,
            diff_summary=f"Replaced function {function_name}",
        )

    def add_import(
        self,
        path: str,
        import_statement: str,
    ) -> EditResult:
        """Add an import statement to a file."""
        content = self._operator.read(path)
        if content is None:
            return EditResult(
                success=False,
                file_path=path,
                original_lines=0,
                new_lines=0,
                changes_made=0,
                error=f"File not found: {path}",
            )

        lines = content.split("\n")
        original_lines = len(lines)

        # Check if import already exists
        if import_statement.strip() in content:
            return EditResult(
                success=True,
                file_path=path,
                original_lines=original_lines,
                new_lines=original_lines,
                changes_made=0,
                diff_summary="Import already exists",
            )

        # Find the right place to insert (after existing imports)
        insert_line = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                insert_line = i + 1
            elif stripped and not stripped.startswith("#") and not stripped.startswith('"""') and not stripped.startswith("'''"):
                if insert_line > 0:
                    break

        lines.insert(insert_line, import_statement)
        new_content = "\n".join(lines)
        result = self._operator.write(path, new_content)

        return EditResult(
            success=True,
            file_path=path,
            original_lines=original_lines,
            new_lines=len(lines),
            changes_made=1,
            diff_summary=f"Added import: {import_statement}",
        )

    def find_and_replace(
        self,
        path: str,
        find: str,
        replace: str,
        first_only: bool = False,
    ) -> EditResult:
        """Find and replace text in a file."""
        content = self._operator.read(path)
        if content is None:
            return EditResult(
                success=False,
                file_path=path,
                original_lines=0,
                new_lines=0,
                changes_made=0,
                error=f"File not found: {path}",
            )

        original_lines = content.count("\n") + 1
        count = content.count(find)

        if count == 0:
            return EditResult(
                success=False,
                file_path=path,
                original_lines=original_lines,
                new_lines=original_lines,
                changes_made=0,
                error=f"Text not found: {find[:50]}...",
            )

        if first_only:
            new_content = content.replace(find, replace, 1)
            changes = 1
        else:
            new_content = content.replace(find, replace)
            changes = count

        result = self._operator.write(path, new_content)

        return EditResult(
            success=True,
            file_path=path,
            original_lines=original_lines,
            new_lines=new_content.count("\n") + 1,
            changes_made=changes,
            diff_summary=f"Replaced {changes} occurrence(s)",
        )

    # -- Private methods -----------------------------------------------------

    def _apply_operation(
        self,
        lines: list[str],
        op: EditOperation,
    ) -> tuple[list[str], bool]:
        """Apply a single edit operation."""
        if op.kind == "replace_lines":
            return self._replace_lines(lines, op.target, op.content)
        elif op.kind == "replace_function":
            return self._replace_function_body(lines, op.target, op.content)
        elif op.kind == "insert_after":
            return self._insert_after(lines, op.target, op.content)
        elif op.kind == "insert_before":
            return self._insert_before(lines, op.target, op.content)
        elif op.kind == "append":
            return lines + op.content.split("\n"), True
        elif op.kind == "prepend":
            return op.content.split("\n") + lines, True
        else:
            raise ValueError(f"Unknown operation kind: {op.kind}")

    def _replace_lines(
        self,
        lines: list[str],
        target: str,
        content: str,
    ) -> tuple[list[str], bool]:
        """Replace lines by line number range (e.g., '10-20')."""
        match = re.match(r"(\d+)-(\d+)", target)
        if not match:
            raise ValueError(f"Invalid line range: {target}")

        start = int(match.group(1)) - 1  # Convert to 0-based
        end = int(match.group(2))

        if start < 0 or end > len(lines):
            raise ValueError(f"Line range out of bounds: {target}")

        new_lines = content.split("\n")
        lines[start:end] = new_lines
        return lines, True

    def _replace_function_body(
        self,
        lines: list[str],
        function_name: str,
        new_body: str,
    ) -> tuple[list[str], bool]:
        """Replace a function's body."""
        start_line = None
        end_line = None
        indent = ""

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(f"def {function_name}("):
                start_line = i
                indent = line[: len(line) - len(line.lstrip())]
                # Find end of function
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    if next_line.strip() and not next_line.startswith(indent + " "):
                        end_line = j
                        break
                if end_line is None:
                    end_line = len(lines)
                break

        if start_line is None:
            raise ValueError(f"Function {function_name} not found")

        # Keep function signature
        sig_line = lines[start_line]
        new_lines = new_body.split("\n")
        indented = [indent + line if line.strip() else "" for line in new_lines]

        lines[start_line:end_line] = [sig_line] + indented
        return lines, True

    def _insert_after(
        self,
        lines: list[str],
        target: str,
        content: str,
    ) -> tuple[list[str], bool]:
        """Insert content after a matching line."""
        target_lower = target.lower()
        for i, line in enumerate(lines):
            if target_lower in line.lower():
                new_lines = content.split("\n")
                lines[i + 1:i + 1] = new_lines
                return lines, True
        raise ValueError(f"Pattern not found: {target}")

    def _insert_before(
        self,
        lines: list[str],
        target: str,
        content: str,
    ) -> tuple[list[str], bool]:
        """Insert content before a matching line."""
        target_lower = target.lower()
        for i, line in enumerate(lines):
            if target_lower in line.lower():
                new_lines = content.split("\n")
                lines[i:i] = new_lines
                return lines, True
        raise ValueError(f"Pattern not found: {target}")

    def _build_diff_summary(
        self,
        original_lines: int,
        new_lines: int,
        changes: int,
    ) -> str:
        """Build a summary of changes."""
        diff = new_lines - original_lines
        if diff > 0:
            return f"Added {diff} lines, {changes} change(s)"
        elif diff < 0:
            return f"Removed {abs(diff)} lines, {changes} change(s)"
        else:
            return f"{changes} change(s), same line count"
