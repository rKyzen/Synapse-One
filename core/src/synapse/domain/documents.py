"""Document domain models (AI Operating Workspace redesign, Phase B).

Documents are longer-form typed artifacts in a workspace (report, essay,
lesson plan, business plan, presentation outline, ...). They carry a
``doc_type`` so future templates and renderers can target them, and a status
(draft/final). Pure data models — storage lives in ``synapse.projects.documents``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Document(BaseModel):
    """A single typed workspace document."""

    id: str
    project_id: str
    title: str
    doc_type: str = "document"  # report | essay | lesson_plan | business_plan | ...
    content: str = ""
    status: str = "draft"  # draft | final
    goal_id: str | None = None
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
