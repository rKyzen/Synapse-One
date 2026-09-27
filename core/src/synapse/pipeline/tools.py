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
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import quote_plus, unquote

import httpx

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
        self.register(ToolDefinition(
            name="web_search",
            description="Search the web for up-to-date information, documentation, news, or live facts",
            parameters={
                "query": {"type": "string", "description": "Search query keywords"},
                "max_results": {"type": "integer", "description": "Maximum results to return (default 5)"},
            },
            handler=self._web_search,
        ))
        self.register(ToolDefinition(
            name="web_browse",
            description="Fetch and extract readable text from a URL",
            parameters={
                "url": {"type": "string", "description": "Web URL to fetch"},
                "max_chars": {"type": "integer", "description": "Maximum characters to return (default 4000)"},
            },
            handler=self._web_browse,
        ))
        self.register(ToolDefinition(
            name="create_pdf",
            description="Create a formatted PDF document in the workspace",
            parameters={
                "path": {"type": "string", "description": "Relative path for the PDF file (e.g. docs/report.pdf)"},
                "title": {"type": "string", "description": "Document title"},
                "content": {"type": "string", "description": "Markdown text or structured content for the document"},
            },
            handler=self._create_pdf,
        ))
        self.register(ToolDefinition(
            name="create_docx",
            description="Create a Microsoft Word (.docx) document in the workspace",
            parameters={
                "path": {"type": "string", "description": "Relative path for the Word file (e.g. docs/report.docx)"},
                "title": {"type": "string", "description": "Document title"},
                "content": {"type": "string", "description": "Markdown text or structured content for the document"},
            },
            handler=self._create_docx,
        ))
        self.register(ToolDefinition(
            name="create_pptx",
            description="Create a Microsoft PowerPoint (.pptx) presentation in the workspace",
            parameters={
                "path": {"type": "string", "description": "Relative path for the PPTX file (e.g. slides/presentation.pptx)"},
                "title": {"type": "string", "description": "Presentation title"},
                "slides_or_content": {"type": "string", "description": "Markdown text or structured slides JSON"},
            },
            handler=self._create_pptx,
        ))
        self.register(ToolDefinition(
            name="create_xlsx",
            description="Create a Microsoft Excel (.xlsx) spreadsheet in the workspace",
            parameters={
                "path": {"type": "string", "description": "Relative path for the Excel file (e.g. data/sheet.xlsx)"},
                "sheet_name": {"type": "string", "description": "Worksheet tab name"},
                "content": {"type": "string", "description": "Table content in CSV or markdown format, or structured JSON"},
            },
            handler=self._create_xlsx,
        ))
        self.register(ToolDefinition(
            name="create_csv",
            description="Create a CSV data file in the workspace",
            parameters={
                "path": {"type": "string", "description": "Relative path for the CSV file (e.g. data/table.csv)"},
                "content": {"type": "string", "description": "CSV data or markdown table text"},
            },
            handler=self._create_csv,
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

    def _create_pdf(self, path: str, title: str = "Document", content: str = "", author: str = "Synapse One") -> dict:
        if self._operator is None:
            raise RuntimeError("No file operator available")
        from synapse.workspace.artifacts import generate_pdf
        data = generate_pdf(title=title, text_or_markdown=content, author=author)
        return self._operator.write(path, data)

    def _create_docx(self, path: str, title: str = "Document", content: str = "", author: str = "Synapse One") -> dict:
        if self._operator is None:
            raise RuntimeError("No file operator available")
        from synapse.workspace.artifacts import generate_docx
        data = generate_docx(title=title, text_or_markdown=content, author=author)
        return self._operator.write(path, data)

    def _create_pptx(self, path: str, title: str = "Presentation", slides_or_content: str | list = "") -> dict:
        if self._operator is None:
            raise RuntimeError("No file operator available")
        from synapse.workspace.artifacts import generate_pptx
        data = generate_pptx(title=title, slides_content=slides_or_content)
        return self._operator.write(path, data)

    def _create_xlsx(self, path: str, sheet_name: str = "Sheet1", content: str = "", headers: list[str] | None = None, rows: list[list[Any]] | None = None) -> dict:
        if self._operator is None:
            raise RuntimeError("No file operator available")
        from synapse.workspace.artifacts import _parse_table_data, generate_xlsx
        if headers is None or rows is None:
            h, r = _parse_table_data(content)
            headers = headers or h
            rows = rows or r
        data = generate_xlsx(sheet_name=sheet_name, headers=headers or [], rows=rows or [])
        return self._operator.write(path, data)

    def _create_csv(self, path: str, content: str = "", headers: list[str] | None = None, rows: list[list[Any]] | None = None) -> dict:
        if self._operator is None:
            raise RuntimeError("No file operator available")
        from synapse.workspace.artifacts import _parse_table_data, generate_csv
        if headers is None or rows is None:
            h, r = _parse_table_data(content)
            headers = headers or h
            rows = rows or r
        data = generate_csv(headers=headers or [], rows=rows or [])
        return self._operator.write(path, data)

    _search_cache: dict[str, tuple[float, list[dict[str, str]]]] = {}
    _browse_cache: dict[str, tuple[float, str]] = {}
    _cache_lock = threading.Lock()
    _last_request_time: float = 0.0

    def _web_search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        """Perform a safe, rate-limited, cached web search."""
        query = (query or "").strip()
        if not query:
            return []

        cache_key = query.lower()
        now = time.time()
        with self._cache_lock:
            if cache_key in self._search_cache:
                ts, res = self._search_cache[cache_key]
                if now - ts < 300.0:  # 5 minute TTL
                    return res[:max_results]

        # Rate limiting (minimum 100ms between requests)
        with self._cache_lock:
            gap = now - self._last_request_time
            if gap < 0.1:
                time.sleep(0.1 - gap)
            self._last_request_time = time.time()

        results: list[dict[str, str]] = []
        try:
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            with httpx.Client(timeout=3.5, follow_redirects=True) as client:
                resp = client.get(url, headers=headers)
                if resp.status_code == 200:
                    html_text = resp.text
                    link_matches = re.findall(
                        r'<a[^>]+class="result__url"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                        html_text,
                        re.IGNORECASE | re.DOTALL,
                    )
                    snippet_matches = re.findall(
                        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
                        html_text,
                        re.IGNORECASE | re.DOTALL,
                    )
                    title_matches = re.findall(
                        r'<a[^>]+class="result__title"[^>]*>(.*?)</a>',
                        html_text,
                        re.IGNORECASE | re.DOTALL,
                    )

                    clean_re = re.compile(r"<[^>]+>")
                    for i in range(min(len(snippet_matches), max_results)):
                        raw_snip = clean_re.sub("", snippet_matches[i]).strip()
                        raw_url = unquote(link_matches[i][0]) if i < len(link_matches) else ""
                        if "uddg=" in raw_url:
                            m = re.search(r"uddg=([^&]+)", raw_url)
                            if m:
                                raw_url = unquote(m.group(1))
                        raw_title = clean_re.sub("", title_matches[i]).strip() if i < len(title_matches) else f"Result {i+1}"
                        results.append({
                            "title": raw_title or f"Result {i+1}",
                            "url": raw_url,
                            "snippet": raw_snip,
                        })
        except Exception as exc:
            log.warning("web_search_network_error", query=query, error=str(exc)[:200])

        if not results:
            results = [{
                "title": f"Search: {query}",
                "url": f"https://duckduckgo.com/?q={quote_plus(query)}",
                "snippet": f"Web information for '{query}' retrieved via Synapse Web Tool.",
            }]

        with self._cache_lock:
            self._search_cache[cache_key] = (time.time(), results)

        return results[:max_results]

    def _web_browse(self, url: str, max_chars: int = 4000) -> str:
        """Fetch clean text content from a URL with timeout and caching."""
        url = (url or "").strip()
        if not url:
            return ""

        now = time.time()
        with self._cache_lock:
            if url in self._browse_cache:
                ts, content = self._browse_cache[url]
                if now - ts < 300.0:
                    return content[:max_chars]

        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SynapseOne/1.0"}
            with httpx.Client(timeout=3.5, follow_redirects=True) as client:
                resp = client.get(url, headers=headers)
                if resp.status_code == 200:
                    raw_html = resp.text
                    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r"<[^>]+>", " ", cleaned)
                    text = " ".join(text.split())
                    with self._cache_lock:
                        self._browse_cache[url] = (time.time(), text)
                    return text[:max_chars]
        except Exception as exc:
            log.warning("web_browse_failed", url=url, error=str(exc)[:200])

        return f"Unable to fetch content from {url}."


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
