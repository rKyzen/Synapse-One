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


class ReasoningComplexity(str, Enum):
    """Reasoning complexity levels for task difficulty evaluation."""

    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"


class ExecutionMode(str, Enum):
    """Execution mode determined by the Master Model."""

    DIRECT_ANSWER = "direct_answer"
    WORKSPACE_AGENT = "workspace_agent"
    TOOL_EXECUTION = "tool_execution"
    ARTIFACT_GENERATION = "artifact_generation"
    MULTI_STEP_AGENT = "multi_step_agent"


class MasterAnalysis(BaseModel):
    """The complete structured analysis produced by the Master Model on every request."""

    intent: str = Field(description="High-level intent category (e.g. artifact_generation, question_answering, coding, math_reasoning, direct_answer).")
    domain: str = Field(description="Problem domain (e.g. web_development, mathematics, python, general, logic).")
    goal: str = Field(description="Clear summary of what the user is trying to achieve.")
    workspace_needed: bool = Field(default=False, description="Whether existing workspace files or folder context must be inspected.")
    workspace_reason: str = Field(default="", description="Why workspace access is or is not needed.")
    files_needed: bool = Field(default=False, description="Whether files must be created, read, or modified on disk.")
    memory_needed: bool = Field(default=False, description="Whether long-term project memory is needed.")
    tools_needed: bool = Field(default=False, description="Whether tools (filesystem, terminal, test runner) are required.")
    coding_needed: bool = Field(default=False, description="Whether code generation, editing, or execution is needed.")
    vision_needed: bool = Field(default=False, description="Whether image understanding is needed.")
    document_processing_needed: bool = False
    web_needed: bool = False
    artifact_required: bool = Field(default=False, description="Whether an artifact (files, website, code project) must be produced.")
    reasoning_complexity: ReasoningComplexity = Field(default=ReasoningComplexity.EASY, description="trivial, easy, medium, hard, very_hard.")
    required_capabilities: list[str] = Field(default_factory=list, description="Target capabilities e.g. ['html', 'css', 'code_generation', 'file_creation', 'math', 'reasoning'].")
    recommended_model_role: str = Field(default="", description="Recommended specialist model role from the registry.")
    execution_mode: ExecutionMode = Field(default=ExecutionMode.DIRECT_ANSWER, description="direct_answer, workspace_agent, tool_execution, artifact_generation, multi_step_agent.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in this analysis (0.0 - 1.0).")


class SubTaskIntent(str, Enum):
    """Intent categories the Master AI may assign to a sub-task."""

    CODING = "CODING"
    REASONING = "REASONING"
    DEEP_REASONING = "DEEP_REASONING"
    MATH = "MATH"
    PLANNING = "PLANNING"
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
    SubTaskIntent.DEEP_REASONING: (
        TaskKind.REASONING,
        [Capability.REASONING],
        [Capability.CHAT, Capability.PLANNING],
    ),
    SubTaskIntent.MATH: (
        TaskKind.MATH,
        [Capability.MATH, Capability.REASONING],
        [Capability.CHAT],
    ),
    SubTaskIntent.PLANNING: (
        TaskKind.PLANNING,
        [Capability.PLANNING, Capability.REASONING],
        [Capability.CHAT],
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
    intent: SubTaskIntent = Field(description="Subtask intent category.")
    capability: str = Field(default="", description="Target capability, e.g. coding, math, reasoning, vision, tool.")
    sub_prompt: _NonEmptyStr = Field(description="Exact instructions for this step only.")
    assigned_model: str = Field(default="", description="Specialist model or tool id selected from the Model Registry.")
    tool: str | None = Field(default=None, description="Tool name if this subtask uses a deterministic tool (e.g. filesystem_tool).")
    dependencies: list[int] = Field(
        default_factory=list,
        description="task_ids that must complete before this sub-task runs.",
    )
    #: Adopted from the artifact pipeline: when true the model for this task is
    #: expected to return a structured file manifest instead of plain prose.
    file_output: bool = False
    reasoning: str = Field(default="", description="Why this specialist model or tool was chosen.")


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
            if sub.capability:
                cap_lower = sub.capability.lower()
                if "cod" in cap_lower:
                    kind = TaskKind.CODING
                    required = [Capability.CODING, Capability.REASONING]
                elif "math" in cap_lower:
                    kind = TaskKind.MATH
                    required = [Capability.MATH, Capability.REASONING]
                elif "deep" in cap_lower:
                    kind = TaskKind.REASONING
                    required = [Capability.REASONING]
                elif "plan" in cap_lower:
                    kind = TaskKind.PLANNING
                    required = [Capability.PLANNING, Capability.REASONING]
                elif "vis" in cap_lower:
                    kind = TaskKind.GENERAL
                    required = [Capability.VISION, Capability.CHAT]

            model_hint = sub.assigned_model.strip() if sub.assigned_model else None
            req_tools = [sub.tool] if sub.tool else []
            if model_hint and model_hint.endswith("_tool") and not req_tools:
                req_tools = [model_hint]

            # Inferred file_hint and file_output
            file_hint = ""
            file_out = sub.file_output
            import re
            m = re.search(r"\b([A-Za-z0-9_\-\.\/]+\.(?:py|js|ts|html|css|json|md|sh|toml|yaml|yml|sql))\b", sub.sub_prompt)
            if m:
                file_hint = m.group(1).strip()
                file_out = True
            elif sub.intent in (SubTaskIntent.CODING, SubTaskIntent.WRITING) or (sub.capability and any(c in sub.capability.lower() for c in ("code", "file", "test", "doc"))):
                lowered_prompt = sub.sub_prompt.lower()
                if any(w in lowered_prompt for w in ("html", "css", "landing page", "website")):
                    file_hint = "index.html"
                    file_out = True
                elif "test" in lowered_prompt:
                    file_hint = "test/test_app.py"
                    file_out = True
                elif any(w in lowered_prompt for w in ("doc", "readme", "architecture")):
                    file_hint = "docs/README.md"
                    file_out = True
                elif any(w in lowered_prompt for w in ("python", "script", "app", "main")) or sub.intent == SubTaskIntent.CODING:
                    file_hint = "src/main.py"
                    file_out = True

            tasks.append(
                Task(
                    id=f"t{sub.task_id}",
                    kind=kind,
                    description=sub.sub_prompt,
                    required_capabilities=required,
                    preferred_capabilities=preferred,
                    depends_on=[f"t{dep}" for dep in sub.dependencies],
                    file_output=file_out,
                    file_hint=file_hint,
                    model_hint=model_hint if model_hint and not model_hint.endswith("_tool") else None,
                    preferred_model=model_hint if model_hint and not model_hint.endswith("_tool") else None,
                    required_tools=req_tools,
                    reason=sub.reasoning or "",
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