"""Phase 5/7 domain models — projects, chats, and chat messages.

Projects are the top-level isolation unit of the Workspace System. Phase 7
split: a project's **workspace** is a plain user-chosen folder on disk that
holds ONLY the project's source/generated files and Git data; all Synapse
metadata (chats, memory, vectors, logs) lives in internal storage keyed by
the unique project id. These are pure data models — the storage logic lives
in ``synapse.projects``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectInfo(BaseModel):
    """A project's public face: identity + aggregate counts.

    ``workspace_path`` is the only on-disk location Synapse remembers; the
    registry never stores anything inside the user's folder.
    """

    id: str
    name: str
    #: Absolute path of the user's workspace folder (source + generated
    #: files + Git). Synapse metadata NEVER lives here.
    workspace_path: str
    created_at: str
    updated_at: str
    archived: bool = False
    #: False when the workspace folder was moved/deleted on disk — the UI
    #: then offers a "reconnect" flow instead of failing silently.
    exists: bool = True
    #: Project-scoped settings (temperature, max_tokens, system_prompt, ...).
    settings: dict = Field(default_factory=dict)
    chats_count: int = 0
    files_count: int = 0
    #: Detected dominant language/framework of the workspace (Phase X scan).
    language: str = ""
    framework: str = ""
    #: Number of workspace files currently indexed for retrieval.
    indexed_count: int = 0

    @property
    def root(self) -> str:
        """Backward-compatible alias for ``workspace_path``."""
        return self.workspace_path


class ChatInfo(BaseModel):
    """One independently addressable chat inside a project."""

    id: str
    project_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class StoredMessage(BaseModel):
    """A single durable chat turn persisted on the project's chat file."""

    role: str  # "user" | "assistant"
    content: str
    created_at: str = Field(default_factory=_now_iso)
    meta: list[str] = Field(default_factory=list)
    trace: dict | None = Field(default=None)
    #: file ids attached to this user message (restores attachment chips).
    files: list[str] = Field(default_factory=list)


class ChatRecord(BaseModel):
    """Full chat file contents — header + ordered messages."""

    id: str
    project_id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[StoredMessage] = Field(default_factory=list)