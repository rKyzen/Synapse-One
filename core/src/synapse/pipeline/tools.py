"""Tool Registry — defines tools that models can call during pipeline execution.

Tools are the bridge between model output and real actions. Models output
structured tool calls, and the framework executes them safely through the
ActionEngine.

Supported tools:
    - read_file: read file contents
    - write_file: create or overwrite a file
    - edit_file: targeted edit (old -> new)
    - create_folder: create a directory
    - delete_file: remove a file
    - search_files: regex search across workspace
    - list_files: list workspace files
    - run_command: execute a shell command (sandboxed)
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable

from synapse.logging import get_logger
from synapse.workspace.operator import FileOperator

log = get_logger("synapse.pipeline.tools")


@dataclass
class ToolDefinition:
    """Definition of a callable tool."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema-like
    handler: Callable[..., Any] | None = None
    requires_confirmation: bool = False
    dangerous: bool = False


@dataclass
class ToolCall:
    """A model's request to call a tool."""

    id: str
    tool_name: str
    arguments: dict[str, Any]
    status: str = "pending"
    result: Any = None
    error: str | None = None


@dataclass
class ToolResult:
    """Result of a tool execution."""

    tool_call_id: str
    success: bool
    output: Any = None
    error: str | None = None
    side_effects: list[str] = field(default_factory=list)


class ToolRegistry:
    """Registry of available tools and their handlers."""

    def __init__(self, file_operator: FileOperator | None = None) -> None:
        self._operator = file_operator
        self._tools: dict[str, ToolDefinition] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register the built-in tools."""
        self.register(ToolDefinition(
            name="read_file",
            description="Read the contents of a file in the workspace",
            parameters={
                "path": {"type": "string", "description": "Relative path to the file"},
            },
            handler=self._read_file,
        ))
        self.register(ToolDefinition(
            name="write_file",
            description="Create or overwrite a file in the workspace",
            parameters={
                "path": {"type": "string", "description": "Relative path to the file"},
                "content": {"type": "string", "description": "File content"},
            },
            handler=self._write_file,
        ))
        self.register(ToolDefinition(
            name="edit_file",
            description="Apply a targeted edit to a file (find and replace)",
            parameters={
                "path": {"type": "string", "description": "Relative path to the file"},
                "old_text": {"type": "string", "description": "Text to find"},
                "new_text": {"type": "string", "description": "Replacement text"},
            },
            handler=self._edit_file,
        ))
        self.register(ToolDefinition(
            name="create_folder",
            description="Create a directory in the workspace",
            parameters={
                "path": {"type": "string", "description": "Relative path for the folder"},
            },
            handler=self._create_folder,
        ))
        self.register(ToolDefinition(
            name="delete_file",
            description="Delete a file from the workspace",
            parameters={
                "path": {"type": "string", "description": "Relative path to the file"},
            },
            handler=self._delete_file,
            requires_confirmation=True,
            dangerous=True,
        ))
        self.register(ToolDefinition(
            name="search_files",
            description="Search for a pattern across workspace files",
            parameters={
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "file_pattern": {"type": "string", "description": "File glob pattern (optional)"},
            },
            handler=self._search_files,
        ))
        self.register(ToolDefinition(
            name="list_files",
            description="List all files in the workspace",
            parameters={},
            handler=self._list_files,
        ))

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def get_schemas(self) -> list[dict]:
        """Get JSON schemas for all tools (for model consumption)."""
        schemas = []
        for tool in self._tools.values():
            schema = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            schemas.append(schema)
        return schemas

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        tool = self._tools.get(tool_call.tool_name)
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                error=f"Unknown tool: {tool_call.tool_name}",
            )
        if tool.handler is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                error=f"Tool {tool_call.tool_name} has no handler",
            )
        try:
            result = tool.handler(**tool_call.arguments)
            return ToolResult(
                tool_call_id=tool_call.id,
                success=True,
                output=result,
            )
        except Exception as exc:
            log.warning("tool_execution_failed", tool=tool_call.tool_name, error=str(exc)[:200])
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                error=str(exc),
            )

    # -- Tool handlers -------------------------------------------------------

    def _read_file(self, path: str) -> str:
        if self._operator is None:
            raise RuntimeError("No file operator available")
        content = self._operator.read(path)
        if content is None:
            raise FileNotFoundError(f"File not found: {path}")
        return content

    def _write_file(self, path: str, content: str) -> dict:
        if self._operator is None:
            raise RuntimeError("No file operator available")
        return self._operator.write(path, content)

    def _edit_file(self, path: str, old_text: str, new_text: str) -> dict:
        if self._operator is None:
            raise RuntimeError("No file operator available")
        content = self._operator.read(path)
        if content is None:
            raise FileNotFoundError(f"File not found: {path}")
        if old_text not in content:
            raise ValueError(f"Text not found in {path}")
        updated = content.replace(old_text, new_text, 1)
        return self._operator.write(path, updated)

    def _create_folder(self, path: str) -> dict:
        if self._operator is None:
            raise RuntimeError("No file operator available")
        folder = self._operator.path_for(path)
        folder.mkdir(parents=True, exist_ok=True)
        return {"path": path, "action": "created_folder"}

    def _delete_file(self, path: str) -> bool:
        if self._operator is None:
            raise RuntimeError("No file operator available")
        return self._operator.delete(path)

    def _search_files(self, pattern: str, file_pattern: str = "") -> list[dict]:
        if self._operator is None:
            raise RuntimeError("No file operator available")
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE)
        hits = []
        for entry in self._operator.list_tree():
            if file_pattern and not re.match(file_pattern.replace("*", ".*"), entry["name"]):
                continue
            content = self._operator.read(entry["path"]) or ""
            for lineno, line in enumerate(content.splitlines(), 1):
                if rx.search(line):
                    hits.append({
                        "path": entry["path"],
                        "line": lineno,
                        "text": line.strip()[:200],
                    })
                    if len(hits) >= 50:
                        return hits
        return hits

    def _list_files(self) -> list[dict]:
        if self._operator is None:
            raise RuntimeError("No file operator available")
        return self._operator.list_tree()


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract tool calls from model output.

    Supports multiple formats:
        - JSON blocks: ```json\n{"tool": "...", "args": {...}}\n```
        - Inline: TOOL:tool_name(arg1="val1", arg2="val2")
        - XML-style: <tool name="..."><arg>val</arg></tool>
    """
    calls = []

    # Format 1: JSON blocks
    json_pattern = r'```json\s*(\{[^`]+\})\s*```'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            data = json.loads(match.group(1))
            if "tool" in data or "tool_name" in data:
                calls.append(ToolCall(
                    id=f"tc_{len(calls)}",
                    tool_name=data.get("tool") or data.get("tool_name", ""),
                    arguments=data.get("args") or data.get("arguments", {}),
                ))
        except json.JSONDecodeError:
            continue

    # Format 2: TOOL:tool_name(args) format
    tool_pattern = r'TOOL:(\w+)\(([^)]*)\)'
    for match in re.finditer(tool_pattern, text):
        tool_name = match.group(1)
        args_str = match.group(2)
        args = _parse_inline_args(args_str)
        calls.append(ToolCall(
            id=f"tc_{len(calls)}",
            tool_name=tool_name,
            arguments=args,
        ))

    return calls


def _parse_inline_args(args_str: str) -> dict[str, str]:
    """Parse inline tool arguments like: path="file.py", content="..." """
    args = {}
    # Simple key="value" parsing
    pattern = r'(\w+)="([^"]*)"'
    for match in re.finditer(pattern, args_str):
        args[match.group(1)] = match.group(2)
    # Also handle key='value'
    pattern2 = r"(\w+)='([^']*)'"
    for match in re.finditer(pattern2, args_str):
        args[match.group(1)] = match.group(2)
    return args
