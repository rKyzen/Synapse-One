"""Domain layer: the shared vocabulary of Synapse One.

Entities here cross every subsystem boundary (providers, registry, hardware,
router, memory). They are plain typed data — no behavior, no vendor logic.
"""

from synapse.domain import diagnosis, enums, hardware, memory, models, projects, requests, tasks
from synapse.domain.diagnosis import (
    AgentRequest,
    AgentResponse,
    ComplexityResult,
    Decision,
    DecisionTrace,
    ExecutionPlan,
    IntentResult,
    PrivacyResult,
    RoutingDecision,
)
from synapse.domain.projects import ChatInfo, ChatRecord, ProjectInfo, StoredMessage
from synapse.domain.enums import (
    Capability,
    ExecutionStrategy,
    IntentType,
    LatencyTier,
    MemoryScope,
    PrivacyMode,
    ProviderKind,
    ProviderState,
    TaskKind,
    TaskStatus,
    WorkspaceKind,
)
from synapse.domain.hardware import HardwareProfile
from synapse.domain.memory import MemoryEntry
from synapse.domain.models import ModelCapabilities, ModelDescriptor, ModelMetadata
from synapse.domain.requests import ChatMessage, ChatRequest, ChatResponse, ProviderMetrics, Usage
from synapse.domain.tasks import (
    ExecutionGraph,
    GraphEdge,
    GraphNode,
    Task,
    TaskDAG,
)
from synapse.domain.workspace import (
    CodeMatch,
    RetrievedChunk,
    WorkspaceFileInfo,
    WorkspaceOutcome,
)

__all__ = [
    "Capability",
    "ExecutionStrategy",
    "IntentType",
    "LatencyTier",
    "PrivacyMode",
    "ProviderKind",
    "ProviderState",
    "TaskKind",
    "TaskStatus",
    "MemoryScope",
    "WorkspaceKind",
    "HardwareProfile",
    "ModelCapabilities",
    "ModelDescriptor",
    "ModelMetadata",
    "MemoryEntry",
    "Task",
    "TaskDAG",
    "GraphNode",
    "GraphEdge",
    "ExecutionGraph",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "Usage",
    "ProviderMetrics",
    "WorkspaceFileInfo",
    "RetrievedChunk",
    "CodeMatch",
    "WorkspaceOutcome",
    "ProjectInfo",
    "ChatInfo",
    "StoredMessage",
    "ChatRecord",
    "AgentRequest",
    "AgentResponse",
    "ComplexityResult",
    "Decision",
    "DecisionTrace",
    "ExecutionPlan",
    "IntentResult",
    "PrivacyResult",
    "RoutingDecision",
    "diagnosis",
    "enums",
    "hardware",
    "memory",
    "models",
    "projects",
    "requests",
    "tasks",
    "workspace",
]
