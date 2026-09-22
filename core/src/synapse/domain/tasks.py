"""Phase 3 domain: task DAG, execution graph, and their node/edge types.

Pure data structures describing HOW a request is decomposed and executed:
which sub-tasks exist, what they depend on, which model served each one, and
the order of execution. No behavior, no vendor logic.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from synapse.domain.enums import Capability, TaskKind, TaskStatus


class Task(BaseModel):
    """One decomposable unit of work produced by the Task Planner."""

    id: str
    kind: TaskKind = TaskKind.GENERAL
    description: str = ""
    required_capabilities: list[Capability] = Field(default_factory=list)
    preferred_capabilities: list[Capability] = Field(default_factory=list)
    #: ids of tasks this task depends on (DAG edges: dep -> this task).
    depends_on: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    provider_id: str = ""
    model_id: str = ""
    preferred_model: str | None = None
    fallback_model: str | None = None
    fallback_provider: str | None = None
    required_tools: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)
    validation_state: str | None = None
    capability_score: float = 0.0
    reason: str = ""
    result: str | None = None
    error: str | None = None
    latency_ms: float = 0.0
    #: execution order index (set by the orchestrator, 1-based).
    order: int = 0
    #: Phase 4 — internal confidence score assigned after generation (0.0..1.0).
    confidence_score: float | None = None
    #: Phase 6 — the model result is expected to carry a structured file
    #: manifest which the orchestrator applies to the project's work directory.
    file_output: bool = False
    #: Phase 6 — filename hint for the task (used by the manifest fallback).
    file_hint: str = ""
    #: Phase 6 — preferred model id chosen by the planner; the orchestrator
    #: honors it when the model is available, else falls back to the router.
    model_hint: str | None = None


class TaskDAG(BaseModel):
    """A directed acyclic graph of tasks.

    The DAG enables future parallel execution: any task whose dependencies are
    complete can run concurrently with its siblings. Today the orchestrator
    executes in topological order (sequential), but the structure is parallel-ready.
    """

    tasks: list[Task] = Field(default_factory=list)

    def add(self, task: Task) -> Task:
        """Append a task (idempotent by id)."""
        if not any(t.id == task.id for t in self.tasks):
            self.tasks.append(task)
        return task

    def get(self, task_id: str) -> Task | None:
        return next((t for t in self.tasks if t.id == task_id), None)

    def roots(self) -> list[Task]:
        """Tasks with no dependencies — safe to start first."""
        ids = {t.id for t in self.tasks}
        return [t for t in self.tasks if not any(dep in ids for dep in t.depends_on)]

    def leaves(self) -> list[Task]:
        """Tasks nothing depends on — terminal nodes of the DAG."""
        dependents = {dep for t in self.tasks for dep in t.depends_on}
        return [t for t in self.tasks if t.id not in dependents]

    def dependents_of(self, task_id: str) -> list[Task]:
        """Tasks that depend directly on ``task_id``."""
        return [t for t in self.tasks if task_id in t.depends_on]

    def topological_order(self) -> list[Task]:
        """Deterministic Kahn's algorithm. Raises ValueError on cycles."""
        indegree = {t.id: 0 for t in self.tasks}
        dependents: dict[str, list[str]] = {t.id: [] for t in self.tasks}
        for t in self.tasks:
            for dep in t.depends_on:
                if dep not in indegree:
                    raise ValueError(f"task {t.id} depends on unknown task {dep}")
                indegree[t.id] += 1
                dependents[dep].append(t.id)

        ready = sorted([tid for tid, deg in indegree.items() if deg == 0])
        order: list[str] = []
        while ready:
            tid = ready.pop(0)
            order.append(tid)
            for child in dependents.get(tid, []):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort()
        if len(order) != len(self.tasks):
            cyclic = [t.id for t in self.tasks if t.id not in order]
            raise ValueError(f"cycle detected among tasks: {cyclic}")
        return [self.get(tid) for tid in order]  # type: ignore[return-value]

    def edges(self) -> list[tuple[str, str]]:
        """(dependency, dependent) pairs for graph serialization."""
        return [(dep, t.id) for t in self.tasks for dep in t.depends_on]


class GraphNode(BaseModel):
    """Serialized view of one task for the execution graph."""

    task_id: str
    kind: TaskKind
    description: str = ""
    status: TaskStatus
    depends_on: list[str] = Field(default_factory=list)
    provider_id: str = ""
    model_id: str = ""
    preferred_model: str | None = None
    fallback_model: str | None = None
    required_tools: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    validation_state: str | None = None
    capability_score: float = 0.0
    reason: str = ""
    latency_ms: float = 0.0
    order: int = 0
    memory_context_used: bool = False


class GraphEdge(BaseModel):
    """DAG edge between two tasks."""

    source: str
    target: str


class ExecutionGraph(BaseModel):
    """The execution graph exposed to the UI: tasks + models + order + reasoning."""

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    #: task ids in the order they executed (topological).
    execution_order: list[str] = Field(default_factory=list)
    synthesized: bool = False
    total_latency_ms: float = 0.0

    def node(self, task_id: str) -> GraphNode | None:
        return next((n for n in self.nodes if n.task_id == task_id), None)