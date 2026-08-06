"""Enumerations shared across the domain."""

from __future__ import annotations

from enum import Enum


class ProviderKind(str, Enum):
    """Where a provider's compute lives."""

    LOCAL = "local"
    CLOUD = "cloud"


class ProviderState(str, Enum):
    """Lifecycle state of a provider instance."""

    DISCOVERED = "discovered"
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"
    SHUTDOWN = "shutdown"


class Capability(str, Enum):
    """Capabilities the router consults when assigning work.

    Scored capabilities (0.0..1.0 in a model's profile) are consulted with
    weights: reasoning, coding, writing, math, chat, translation, planning,
    debugging, architecture, ocr, pdf, json, terminal, embeddings.
    Boolean capabilities (either a model has them or not): vision, tools,
    long_context, languages.
    """

    REASONING = "reasoning"
    CODING = "coding"
    WRITING = "writing"
    VISION = "vision"
    MATH = "math"
    LANGUAGES = "languages"
    TOOLS = "tools"
    LONG_CONTEXT = "long_context"
    CHAT = "chat"
    TRANSLATION = "translation"
    PLANNING = "planning"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    OCR = "ocr"
    PDF = "pdf"
    EMBEDDINGS = "embeddings"
    JSON = "json"
    TERMINAL = "terminal"


class WorkspaceKind(str, Enum):
    """Adaptive workspace kinds (Phase 2+: Workspace Engine)."""

    GENERAL = "general"
    CODING = "coding"
    WRITING = "writing"


class LatencyTier(str, Enum):
    """Coarse latency classification, hardware-independent."""

    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"


class IntentType(str, Enum):
    """Intent categories the intent analyzer can detect."""

    GENERAL = "general"
    WRITING = "writing"
    CODING = "coding"
    RESEARCH = "research"
    EDUCATION = "education"
    BUSINESS = "business"
    CREATIVE = "creative"
    PLANNING = "planning"
    CONVERSATION = "conversation"


class PrivacyMode(str, Enum):
    """Privacy decision levels (Phase 2+ privacy analyzer / router)."""

    LOCAL_ONLY = "LOCAL_ONLY"
    PREFER_LOCAL = "PREFER_LOCAL"
    BALANCED = "BALANCED"
    PREFER_CLOUD = "PREFER_CLOUD"
    CLOUD_REQUIRED = "CLOUD_REQUIRED"


class ExecutionStrategy(str, Enum):
    """High-level execution strategy produced by the planner."""

    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"
    NO_ROUTE = "no_route"


class TaskKind(str, Enum):
    """Role of a decomposed sub-task (Phase 3 task orchestration)."""

    PLANNING = "planning"
    REASONING = "reasoning"
    CODING = "coding"
    WRITING = "writing"
    MATH = "math"
    RESEARCH = "research"
    GENERAL = "general"
    SYNTHESIS = "synthesis"
    REVIEW = "review"  # Phase 6 — validate/merge generated file outputs


class TaskStatus(str, Enum):
    """Lifecycle of one task in the execution graph."""

    PENDING = "pending"
    ROUTED = "routed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class MemoryScope(str, Enum):
    """Workspace memory scopes (Phase 3)."""

    CONVERSATION = "conversation"
    PROJECT = "project"
    GLOBAL = "global"


class RequestKind(str, Enum):
    """Phase 7 — how a request should be executed (Action-Based Engine).

    Every incoming prompt is classified into exactly one of these kinds. File
    kinds route through the Action Engine: the LLM only generates content while
    the backend performs every filesystem operation and returns a summary.
    """

    CHAT_RESPONSE = "chat_response"
    FILE_CREATION = "file_creation"
    FILE_MODIFICATION = "file_modification"
    PROJECT_GENERATION = "project_generation"
    PROJECT_ANALYSIS = "project_analysis"
    DOCUMENTATION = "documentation"
    WORKSPACE_OPERATION = "workspace_operation"
