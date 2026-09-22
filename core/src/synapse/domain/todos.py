"""Todo domain models (AI Operating Workspace redesign, Phase B).

Todo items are the user's to-do list within a workspace — goal-linkable,
prioritized, with optional due dates. They are user-facing artifacts and are
distinct from the background TaskQueue that runs orchestrator work. Pure data
models — storage lives in ``synapse.projects.todos``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from synapse.domain.enums import TodoPriority, TodoStatus


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TodoTask(BaseModel):
    """A single workspace todo item."""

    id: str
    project_id: str
    title: str
    description: str = ""
    status: TodoStatus = TodoStatus.TODO
    priority: TodoPriority = TodoPriority.MEDIUM
    due_date: str | None = None
    goal_id: str | None = None
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
