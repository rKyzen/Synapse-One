"""SmartContextManager — intelligent context assembly for model prompts.

Instead of sending the entire workspace, this system:
    1. Analyzes the request to understand what's needed
    2. Uses the project index to find relevant files
    3. Retrieves only the most relevant context
    4. Manages token budgets across multiple sources
    5. Preserves context across multi-stage pipelines

Context sources (in priority order):
    - Direct user instructions
    - Project memory (goals, conventions)
    - Recent chat history
    - Relevant code files (from index)
    - Relevant documentation
    - Workspace structure overview
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from synapse.logging import get_logger
from synapse.workspace.project_index import FileIndex, ProjectIndex, SymbolInfo

log = get_logger("synapse.context.smart")


@dataclass
class ContextBudget:
    """Token budget allocation for different context sources."""

    total: int = 8000
    instructions: int = 500
    project_memory: int = 500
    chat_history: int = 1000
    code_files: int = 4000
    documentation: int = 1000
    structure: int = 500
    reserved: int = 500  # For response generation

    @property
    def available_for_code(self) -> int:
        return self.total - self.instructions - self.project_memory - self.reserved


@dataclass
class ContextPiece:
    """A piece of context with metadata."""

    source: str  # "memory", "chat", "code", "docs", "structure"
    content: str
    priority: int  # Higher = more important
    token_estimate: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssembledContext:
    """The final assembled context for a model prompt."""

    pieces: list[ContextPiece]
    total_tokens: int
    truncated: bool
    summary: str

    def to_prompt_prefix(self) -> str:
        """Convert to a string to prepend to the user prompt."""
        parts = []
        for piece in sorted(self.pieces, key=lambda p: -p.priority):
            parts.append(f"[{piece.source.upper()}]\n{piece.content}")
        return "\n\n".join(parts)


class SmartContextManager:
    """Manages intelligent context assembly for model prompts."""

    def __init__(
        self,
        project_index: ProjectIndex | None = None,
        budget: ContextBudget | None = None,
    ) -> None:
        self._index = project_index
        self._budget = budget or ContextBudget()

    def assemble_context(
        self,
        request: str,
        *,
        project_memory: str = "",
        chat_history: list[dict[str, str]] | None = None,
        workspace_brief: str = "",
        additional_files: list[str] | None = None,
    ) -> AssembledContext:
        """Assemble optimal context for a request."""
        pieces: list[ContextPiece] = []
        used_tokens = 0

        # 1. Project memory (high priority)
        if project_memory:
            mem_tokens = self._estimate_tokens(project_memory)
            if used_tokens + mem_tokens <= self._budget.project_memory:
                pieces.append(ContextPiece(
                    source="memory",
                    content=project_memory,
                    priority=90,
                    token_estimate=mem_tokens,
                ))
                used_tokens += mem_tokens

        # 2. Chat history (medium priority)
        if chat_history:
            history_text = self._format_chat_history(chat_history)
            hist_tokens = self._estimate_tokens(history_text)
            budget = self._budget.chat_history
            if hist_tokens > budget:
                history_text = self._truncate_to_budget(history_text, budget)
                hist_tokens = budget
            if used_tokens + hist_tokens <= self._budget.total - self._budget.reserved:
                pieces.append(ContextPiece(
                    source="chat",
                    content=history_text,
                    priority=70,
                    token_estimate=hist_tokens,
                ))
                used_tokens += hist_tokens

        # 3. Relevant code files (high priority for coding tasks)
        if self._index:
            relevant_files = self._index.get_context_files(request, max_files=5)
            code_context = self._assemble_code_context(relevant_files, additional_files)
            code_tokens = self._estimate_tokens(code_context)
            code_budget = self._budget.available_for_code - used_tokens
            if code_tokens > code_budget:
                code_context = self._truncate_to_budget(code_context, code_budget)
                code_tokens = code_budget
            if code_context:
                pieces.append(ContextPiece(
                    source="code",
                    content=code_context,
                    priority=80,
                    token_estimate=code_tokens,
                ))
                used_tokens += code_tokens

        # 4. Workspace structure (low priority, always include if space)
        if workspace_brief:
            struct_tokens = self._estimate_tokens(workspace_brief)
            if used_tokens + struct_tokens <= self._budget.total - self._budget.reserved:
                pieces.append(ContextPiece(
                    source="structure",
                    content=workspace_brief,
                    priority=30,
                    token_estimate=struct_tokens,
                ))
                used_tokens += struct_tokens

        truncated = used_tokens >= self._budget.total - self._budget.reserved
        summary = self._build_summary(pieces, used_tokens, truncated)

        return AssembledContext(
            pieces=pieces,
            total_tokens=used_tokens,
            truncated=truncated,
            summary=summary,
        )

    def get_file_context(
        self,
        file_path: str,
        max_lines: int = 200,
    ) -> str | None:
        """Get context for a specific file."""
        if not self._index:
            return None
        file_idx = self._index.get_file(file_path)
        if not file_idx:
            return None

        # Build file context with symbols
        parts = [f"File: {file_path}"]
        if file_idx.symbols:
            parts.append("Symbols:")
            for sym in file_idx.symbols[:20]:
                parts.append(f"  - {sym.kind} {sym.name} (line {sym.line})")
        if file_idx.imports:
            parts.append(f"Imports: {', '.join(file_idx.imports[:10])}")
        return "\n".join(parts)

    def get_symbol_context(
        self,
        symbol_name: str,
    ) -> list[dict[str, Any]]:
        """Find all references to a symbol across the project."""
        if not self._index:
            return []
        results = self._index.search_symbols(symbol_name)
        return [
            {
                "file": file_idx.path,
                "symbol": sym.name,
                "kind": sym.kind,
                "line": sym.line,
                "parent": sym.parent,
            }
            for file_idx, sym in results
        ]

    # -- Private methods -----------------------------------------------------

    def _assemble_code_context(
        self,
        files: list[FileIndex],
        additional_files: list[str] | None = None,
    ) -> str:
        """Assemble code context from relevant files."""
        parts = ["Relevant files:"]
        for file_idx in files[:5]:
            parts.append(f"\n--- {file_idx.path} ({file_idx.language}) ---")
            if file_idx.symbols:
                symbols = ", ".join(f"{s.kind} {s.name}" for s in file_idx.symbols[:10])
                parts.append(f"Symbols: {symbols}")
            if file_idx.imports:
                parts.append(f"Imports: {', '.join(file_idx.imports[:5])}")

        # Add additional files if specified
        if additional_files and self._index:
            for path in additional_files[:3]:
                file_idx = self._index.get_file(path)
                if file_idx:
                    parts.append(f"\n--- {path} (requested) ---")
                    if file_idx.symbols:
                        symbols = ", ".join(f"{s.kind} {s.name}" for s in file_idx.symbols[:10])
                        parts.append(f"Symbols: {symbols}")

        return "\n".join(parts)

    def _format_chat_history(self, history: list[dict[str, str]]) -> str:
        """Format chat history for context."""
        parts = ["Recent conversation:"]
        for msg in history[-5:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:200]
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(text) // 4

    def _truncate_to_budget(self, text: str, budget: int) -> str:
        """Truncate text to fit within token budget."""
        max_chars = budget * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 50] + "\n... [truncated]"

    def _build_summary(self, pieces: list[ContextPiece], tokens: int, truncated: bool) -> str:
        """Build a summary of the assembled context."""
        sources = [p.source for p in pieces]
        return (
            f"Context: {len(pieces)} pieces from {', '.join(sources)} "
            f"({tokens} tokens{', truncated' if truncated else ''})"
        )
