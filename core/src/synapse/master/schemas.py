"""Structured output schemas for the AI Master Agent (Task Divider).

Pydantic models enforcing the exact JSON shape the Master AI must return
before any deterministic routing happens. ``TaskDecompositionPlan`` is used
both as the validation target and as the JSON-schema passed to the provider
(Ollama ``format``) so the model cannot drift from the contract.

The schema is deliberately small: the Master AI is a router, not a worker —
it only splits the prompt, labels each part, and declares dependencies.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Self

from pydantic import BaseModel, Field, StringConstraints, model_validator

from synapse.domain import Capability, TaskKind
from synapse.domain.tasks import Task, TaskDAG

_NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SubTaskIntent(str, Enum):
    """Intent categories the Master AI may assign to a sub-task."""

    CODING = "CODING"
    REASONING = "REASONING"
    WRITING = "WRITING"
    VISION = "VISION"
    CHAT = "CHAT"
    TOOL = "TOOL"


class ExecutionStrategy(str, Enum):
    """How the decomposed tasks should be executed."""

    SEQUENTIAL = "SEQUENTIAL"
    PARALLEL = "PARALLEL"
    HYBRID = "HYBRID"


#: subtask intent -> (domain TaskKind, required capabilities, preferred)
_INTENT_PROFILES: dict[SubTaskIntent, tuple[TaskKind, list[Capability], list[Capability]]] = {
    SubTaskIntent.CODING: (
        TaskKind.CODING,
        [Capability.CODING, Capability.REASONING],
        [Capability.DEBUGGING, Capability.JSON, Capability.TERMINAL, Capability.CHAT],
    ),
    SubTaskIntent.REASONING: (
        TaskKind.REASONING,
        [Capability.REASONING],
        [Capability.CHAT, Capability.PLANNING],
    ),
    SubTaskIntent.WRITING: (
        TaskKind.WRITING,
        [Capability.WRITING],
        [Capability.CHAT, Capability.TRANSLATION],
    ),
    SubTaskIntent.VISION: (
        TaskKind.GENERAL,
        [Capability.VISION, Capability.CHAT],
        [],
    ),
    SubTaskIntent.CHAT: (
        TaskKind.GENERAL,
        [Capability.CHAT],
        [Capability.WRITING],
    ),
    SubTaskIntent.TOOL: (
        TaskKind.GENERAL,
        [Capability.TOOLS, Capability.CHAT],
        [Capability.JSON],
    ),
}


def intent_profile(intent: SubTaskIntent) -> tuple[TaskKind, list[Capability], list[Capability]]:
    """Map a schema intent to domain routing profiles (kind, required, preferred)."""
    kind, required, preferred = _INTENT_PROFILES[intent]
    return kind, list(required), list(preferred)


class SubTask(BaseModel):
    """One decomposed unit of work produced by the Master AI."""

    task_id: int = Field(ge=1, description="Unique numeric id of this sub-task.")
    intent: SubTaskIntent
    sub_prompt: _NonEmptyStr = Field(description="Exact instructions for this step only.")
    dependencies: list[int] = Field(
        default_factory=list,
        description="task_ids that must complete before this sub-task runs.",
    )
    #: Adopted from the artifact pipeline: when true the model for this task is
    #: expected to return a structured file manifest instead of plain prose.
    file_output: bool = False


class TaskDecompositionPlan(BaseModel):
    """The complete JSON graph the Master AI must return."""

    tasks: list[SubTask] = Field(min_length=1)
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL

    @model_validator(mode="after")
    def _validate_dependencies(self) -> Self:
        ids = {t.task_id for t in self.tasks}
        if len(ids) != len(self.tasks):
            raise ValueError("duplicate task_id in decomposition plan")
        for task in self.tasks:
            for dep in task.dependencies:
                if dep == task.task_id:
                    raise ValueError(f"task {task.task_id} depends on itself")
                if dep not in ids:
                    raise ValueError(
                        f"task {task.task_id} depends on unknown task {dep}"
                    )
        return self

    # -- conversion ----------------------------------------------------------

    def to_dag(self, *, synthesis: bool = True) -> TaskDAG:
        """Convert this plan into the domain TaskDAG the Master executes.

        Numeric ids become ``t<id>`` task ids; dependencies are remapped to
        string ids. A ``t-synthesis`` task is appended when the plan contains
        more than one task.
        """
        tasks: list[Task] = []
        for sub in self.tasks:
            kind, required, preferred = intent_profile(sub.intent)
            tasks.append(
                Task(
                    id=f"t{sub.task_id}",
                    kind=kind,
                    description=sub.sub_prompt,
                    required_capabilities=required,
                    preferred_capabilities=preferred,
                    depends_on=[f"t{dep}" for dep in sub.dependencies],
                    file_output=sub.file_output,
                )
            )
        if synthesis and len(tasks) > 1:
            tasks.append(
                Task(
                    id="t-synthesis",
                    kind=TaskKind.SYNTHESIS,
                    description=(
                        "Combine the results of the tasks above into one cohesive, "
                        "complete final answer for the user."
                    ),
                    required_capabilities=[Capability.CHAT],
                    preferred_capabilities=[Capability.WRITING, Capability.REASONING],
                    depends_on=[t.id for t in tasks],
                )
            )
        dag = TaskDAG(tasks=tasks)
        dag.topological_order()  # validate acyclicity up front
        return dag