"""Note domain models (AI Operating Workspace redesign, Phase B).

Notes are lightweight markdown entries within a workspace — free-form,
goal-linkable, taggable. They are the workspace's scratch layer: research
snippets, meeting bullets, study cards, outline fragments. Pure data models —
storage lives in ``synapse.projects.notes``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Note(BaseModel):
    """A single workspace note (markdown content)."""

    id: str
    project_id: str
    title: str
    content: str = ""
    goal_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
