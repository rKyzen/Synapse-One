"""Goal domain models (AI Operating Workspace redesign, Phase A).

The goal is the atomic unit of the product. The user states an outcome
("prepare my history presentation for Friday") and the orchestrator plans,
executes, and tracks work toward it. Goals persist across sessions and
accumulate linked artifacts (chats, files, notes, tasks, documents). These
are pure data models — storage lives in ``synapse.projects.goals``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from synapse.domain.enums import GoalStatus


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class GoalStep(BaseModel):
    """One planned/executed step toward a goal.

    Mirrors the orchestrator's task graph: a step has a capability focus
    (e.g. research, writing, coding) and a status, so the workspace can show
    goal progress without exposing model internals.
    """

    description: str
    capability: str = "general"  # planning/reasoning/research/writing/coding/...
    status: str = "pending"  # pending | running | completed | failed | skipped
    model: str | None = None  # populated when a model executes the step
    result: str | None = None  # populated with the step's deliverable text


class GoalInfo(BaseModel):
    """A goal's public face: identity, progress, and linked artifacts."""

    id: str
    project_id: str
    title: str
    outcome: str = ""  # the full outcome statement, available in the info view
    status: GoalStatus = GoalStatus.ACTIVE
    progress: int = 0  # 0..100
    created_at: str
    updated_at: str
    deadline: str | None = None
    steps: list[GoalStep] = Field(default_factory=list)
    linked_chats: list[str] = Field(default_factory=list)
    linked_files: list[str] = Field(default_factory=list)
    linked_notes: list[str] = Field(default_factory=list)
    linked_tasks: list[str] = Field(default_factory=list)

    @property
    def step_summary(self) -> dict:
        """Counts per step status for the workspace home view."""
        counts: dict[str, int] = {}
        for step in self.steps:
            counts[step.status] = counts.get(step.status, 0) + 1
        return counts


class GoalRecord(BaseModel):
    """Full goal file contents — header + planned steps."""

    id: str
    project_id: str
    outcome: str
    title: str
    status: GoalStatus = GoalStatus.ACTIVE
    progress: int = 0
    deadline: str | None = None
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
    steps: list[GoalStep] = Field(default_factory=list)
    linked_chats: list[str] = Field(default_factory=list)
    linked_files: list[str] = Field(default_factory=list)
    linked_notes: list[str] = Field(default_factory=list)
    linked_tasks: list[str] = Field(default_factory=list)

    def to_info(self) -> GoalInfo:
        return GoalInfo(
            id=self.id,
            project_id=self.project_id,
            title=self.title,
            outcome=self.outcome,
            status=self.status,
            progress=self.progress,
            created_at=self.created_at,
            updated_at=self.updated_at,
            deadline=self.deadline,
            steps=self.steps,
            linked_chats=self.linked_chats,
            linked_files=self.linked_files,
            linked_notes=self.linked_notes,
            linked_tasks=self.linked_tasks,
        )
