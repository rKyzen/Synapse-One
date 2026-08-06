"""Phase 4 workspace domain entities — files, retrieval, vision, outcome.

Pure data models shared between the Workspace subsystem, the Master Agent,
and the API layer. No behavior, no vendor logic.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class WorkspaceFileInfo(BaseModel):
    """One uploaded file in the workspace catalog."""

    id: str
    name: str
    extension: str
    pipeline: str = "document"  # image | document | code
    size_bytes: int = 0
    sha256: str = ""
    mime_type: str = ""
    status: str = "uploaded"    # uploaded | indexing | indexed | failed
    indexed_chunks: int = 0
    vision_description: str = ""
    created_at: str = ""
    error: str | None = None


class RetrievedChunk(BaseModel):
    """One retrieval hit — a chunk of a workspace file closest to the query."""

    file_id: str
    file_name: str
    chunk_index: int = 0
    text: str = ""
    score: float = 0.0
    page: int | None = None
    lines_start: int | None = None
    lines_end: int | None = None


class CodeMatch(BaseModel):
    """A deterministic pattern match (e.g. TODO/FIXME) inside a code file."""

    file_id: str
    file_name: str
    line: int
    text: str


class WorkspaceOutcome(BaseModel):
    """What the workspace contributed to one Master Agent request."""

    files_attached: list[str] = Field(default_factory=list)
    files_used: list[str] = Field(default_factory=list)
    pipelines: dict[str, list[str]] = Field(default_factory=dict)
    retrieval: list[RetrievedChunk] = Field(default_factory=list)
    code_matches: list[CodeMatch] = Field(default_factory=list)
    vision_descriptions: list[str] = Field(default_factory=list)
    context_chars: int = 0
    local_only: bool = True
    used_memory: bool = False