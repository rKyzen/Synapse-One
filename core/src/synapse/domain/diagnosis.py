"""Phase 2 domain entities: analysis results, decision, plan, trace, response.

These types flow through the Master Agent pipeline and are serialized by the
API layer. They are pure data — no behavior, no vendor logic.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from synapse.domain.enums import (
    Capability,
    ExecutionStrategy,
    IntentType,
    LatencyTier,
    PrivacyMode,
    ProviderKind,
    WorkspaceKind,
)
from synapse.domain.fileops import FileAction
from synapse.domain.tasks import ExecutionGraph
from synapse.domain.workspace import WorkspaceOutcome


# ---------------------------------------------------------------------------
# Analyzer results
# ---------------------------------------------------------------------------


class IntentResult(BaseModel):
    primary: IntentType = IntentType.GENERAL
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    secondary: IntentType | None = None
    reasoning: list[str] = Field(default_factory=list)
    #: Prompt-derived modality flags (Phase 2.5). Vision requires image
    #: understanding; embeddings flags semantic-retrieval-style tasks.
    vision_required: bool = False
    embeddings_required: bool = False


class ComplexityResult(BaseModel):
    score: int = Field(default=0, ge=0, le=100)
    reasoning: list[str] = Field(default_factory=list)


class PrivacyResult(BaseModel):
    mode: PrivacyMode = PrivacyMode.BALANCED
    internet_required: bool = False
    sensitivity: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Decision engine output
# ---------------------------------------------------------------------------


class Decision(BaseModel):
    """Structured answer of the Decision Engine — the heart of Phase 2."""

    can_stay_local: bool = True
    internet_required: bool = False
    use_cloud_reasoning: bool = False
    use_memory: bool = False  # placeholder — Memory Engine is a later phase
    privacy: PrivacyMode = PrivacyMode.BALANCED
    required_capabilities: list[Capability] = Field(default_factory=list)
    #: Soft capability preferences — boost matching models but never exclude.
    preferred_capabilities: list[Capability] = Field(default_factory=list)
    preferred_kind: ProviderKind = ProviderKind.LOCAL
    workspace: WorkspaceKind = WorkspaceKind.GENERAL
    reasoning: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Execution plan
# ---------------------------------------------------------------------------


class ExecutionPlan(BaseModel):
    """Reusable plan object for future phases (memory, multi-agent, UI)."""

    intent: IntentType = IntentType.GENERAL
    complexity: int = 0
    privacy: PrivacyMode = PrivacyMode.BALANCED
    required_capabilities: list[Capability] = Field(default_factory=list)
    preferred_kind: ProviderKind = ProviderKind.LOCAL
    workspace: WorkspaceKind = WorkspaceKind.GENERAL
    strategy: ExecutionStrategy = ExecutionStrategy.LOCAL
    steps: list[str] = Field(default_factory=list)
    estimated_latency_tier: LatencyTier = LatencyTier.MEDIUM
    estimated_cost_per_1k: float = 0.0
    estimated_latency_s: float | None = None
    expected_output_tokens: int | None = None


# ---------------------------------------------------------------------------
# Routing output
# ---------------------------------------------------------------------------


class RoutingDecision(BaseModel):
    provider_id: str = ""
    model_id: str = ""
    kind: ProviderKind | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    estimated_cost_per_1k: float = 0.0
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    # Phase 2.5 — explainable routing.
    capability_score: float = 0.0
    excluded_models: list[dict[str, str]] = Field(default_factory=list)
    estimated_latency_s: float | None = None
    expected_output_tokens: int | None = None
    historical_performance: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Trace + final response
# ---------------------------------------------------------------------------


class DecisionTrace(BaseModel):
    """Developer-mode trace of one request through the pipeline.

    Phase 2.5 makes routing explainable: the trace explains WHY each model
    was excluded, how strong the chosen model's capability match is, what
    latency is expected, and which historical performance data was used.
    """

    intent: IntentType
    intent_confidence: float
    complexity: int
    privacy: PrivacyMode
    internet_required: bool
    hardware: dict[str, Any] = Field(default_factory=dict)
    provider: str = ""
    model: str = ""
    reason: str = ""
    execution_time_ms: float = 0.0
    candidate_count: int = 0
    capability_score: float = 0.0
    excluded_models: list[dict[str, str]] = Field(default_factory=list)
    estimated_latency_s: float | None = None
    expected_output_tokens: int | None = None
    historical_performance: dict[str, Any] | None = None
    intent_reasoning: list[str] = Field(default_factory=list)
    complexity_reasoning: list[str] = Field(default_factory=list)
    privacy_reasoning: list[str] = Field(default_factory=list)
    hardware_reasoning: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """Normalized response returned to the UI layer."""

    response: str
    intent: IntentType
    complexity: int
    privacy: PrivacyMode
    provider: str
    model: str
    execution_plan: ExecutionPlan = Field(default_factory=ExecutionPlan)
    decision_trace: DecisionTrace | None = None
    latency_ms: float = 0.0
    #: Phase 3 — the decomposed task DAG and how each task executed.
    execution_graph: "ExecutionGraph | None" = None
    #: Phase 4 — what the workspace contributed to this request.
    workspace: WorkspaceOutcome | None = None
    #: Phase 6 — filesystem mutations applied on the project's work directory.
    actions: list[FileAction] = Field(default_factory=list)


class AgentRequest(BaseModel):
    """Input accepted by the Master Agent."""

    prompt: str = Field(min_length=1)
    #: Phase 4 — workspace file ids to attach (images → vision, docs → RAG).
    files: list[str] = Field(default_factory=list)
    #: Optional model override (skips the router when set).
    model: str | None = None
    #: Temperature override (0.0–2.0).
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    #: Max tokens override.
    max_tokens: int | None = None
    #: Phase 5 — project scope (defaults to the active session project).
    project_id: str | None = None
    #: Phase 5 — chat scope (defaults to the project's most recent chat).
    chat_id: str | None = None
