"""PipelineContext — shared state passed between pipeline stages.

Each stage receives the full context and can read/write to it.
The context preserves:
    - Original user prompt
    - Workspace state
    - Previous stage outputs
    - Tool call results
    - Accumulated file changes
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageResult:
    """Output from a single pipeline stage."""

    stage_id: str
    stage_type: str
    model_id: str
    provider_id: str
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """Mutable context shared across all pipeline stages."""

    # Original request
    prompt: str
    project_id: str | None = None
    conversation_id: str | None = None
    workspace_path: str | None = None

    # Workspace state (read at pipeline start)
    workspace_files: list[dict] = field(default_factory=list)
    workspace_brief: str = ""
    relevant_files: list[str] = field(default_factory=list)
    file_contents: dict[str, str] = field(default_factory=dict)

    # Memory state
    project_memory: str = ""
    chat_history: list[dict] = field(default_factory=list)

    # Stage tracking
    stage_results: list[StageResult] = field(default_factory=list)
    current_stage: int = 0

    # File changes (accumulated across stages)
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)

    # Tool results
    tool_results: list[dict] = field(default_factory=list)

    # Final output
    final_response: str = ""

    def add_stage_result(self, result: StageResult) -> None:
        """Record the output of a completed stage."""
        self.stage_results.append(result)
        self.files_created.extend(result.files_written)
        self.files_read.extend(result.files_read)
        self.tool_results.extend(result.tool_calls)

    def get_previous_outputs(self, max_stages: int = 3) -> str:
        """Get formatted outputs from recent stages for context."""
        recent = self.stage_results[-max_stages:]
        if not recent:
            return ""
        parts = []
        for r in recent:
            parts.append(f"[{r.stage_type} via {r.model_id}]:\n{r.content[:2000]}")
        return "\n\n".join(parts)

    def get_accumulated_files(self) -> dict[str, str]:
        """Get all file contents read or written during pipeline."""
        return dict(self.file_contents)

    def get_change_summary(self) -> dict[str, list[str]]:
        """Summary of all file changes (order-preserving, deduplicated)."""
        return {
            "created": list(dict.fromkeys(self.files_created)),
            "modified": list(dict.fromkeys(self.files_modified)),
            "deleted": list(dict.fromkeys(self.files_deleted)),
        }

    def to_dict(self) -> dict:
        """Serialize context for passing to models."""
        return {
            "prompt": self.prompt,
            "project_id": self.project_id,
            "workspace_brief": self.workspace_brief,
            "relevant_files": self.relevant_files[:10],
            "previous_outputs": self.get_previous_outputs(),
            "files_created": self.files_created,
            "files_modified": self.files_modified,
        }
