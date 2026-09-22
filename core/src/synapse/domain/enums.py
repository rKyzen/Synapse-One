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
    weights. All 31 generalized workspace capabilities are supported.
    """

    # Core conversation & cognition
    CONVERSATION = "conversation"
    CHAT = "chat"
    REASONING = "reasoning"
    PLANNING = "planning"
    RESEARCH = "research"
    SUMMARIZATION = "summarization"
    WRITING = "writing"
    MATH = "math"
    TRANSLATION = "translation"
    LANGUAGES = "languages"

    # Coding & technical execution
    CODING = "coding"
    CODE_ANALYSIS = "code_analysis"
    FILE_READING = "file_reading"
    FILE_CREATION = "file_creation"
    FILE_EDITING = "file_editing"
    TERMINAL = "terminal"
    TESTING = "testing"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    JSON = "json"

    # Multimodal & document processing
    VISION = "vision"
    OCR = "ocr"
    IMAGE_UNDERSTANDING = "image_understanding"
    PDF = "pdf"
    PDF_READING = "pdf_reading"
    PDF_CREATION = "pdf_creation"
    DOCX_CREATION = "docx_creation"
    DOCUMENT_ANALYSIS = "document_analysis"
    PPT_CREATION = "ppt_creation"
    SPREADSHEET_CREATION = "spreadsheet_creation"
    DATA_ANALYSIS = "data_analysis"

    # Workspace & Knowledge
    PROJECT_CREATION = "project_creation"
    PROJECT_ANALYSIS = "project_analysis"
    TASK_MANAGEMENT = "task_management"
    MEMORY = "memory"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    EMBEDDINGS = "embeddings"
    CITATIONS = "citations"
    AUTOMATION = "automation"
    TOOLS = "tools"
    LONG_CONTEXT = "long_context"


class WorkspaceKind(str, Enum):
    """Adaptive workspace kinds (Phase 2+: Workspace Engine).

    Superseded by :class:`WorkspaceType` (Phase A of the AI Operating
    Workspace redesign) — kept for backward compatibility with the decision
    engine's ``_INTENT_WORKSPACE`` table.
    """

    GENERAL = "general"
    CODING = "coding"
    WRITING = "writing"


class WorkspaceType(str, Enum):
    """Workspace types (AI Operating Workspace redesign, Phase A).

    Each type loads a different set of tools/capabilities while sharing the
    same core infrastructure (workspace, memory, knowledge, model engines).
    Coding is one capability of the ``developer`` type — never the identity
    of the product.
    """

    GENERAL = "general"
    STUDENT = "student"
    RESEARCH = "research"
    BUSINESS = "business"
    DEVELOPER = "developer"
    WRITER = "writer"
    DESIGNER = "designer"
    TEACHER = "teacher"
    CUSTOM = "custom"


class GoalStatus(str, Enum):
    """Lifecycle of a workspace goal (AI Operating Workspace redesign).

    A goal is the atomic unit of the product: the user states an outcome, and
    the orchestrator plans/executes work toward it. Goals persist across
    sessions and accumulate linked artifacts (chats, files, notes, tasks).
    """

    ACTIVE = "active"
    PAUSED = "paused"
    DONE = "done"
    ARCHIVED = "archived"


class TodoStatus(str, Enum):
    """Lifecycle of a workspace todo item (Phase B).

    Todo items are user-facing tasks within a workspace — distinct from the
    background TaskQueue that runs orchestrator work. Both are goal-linkable.
    """

    TODO = "todo"
    DOING = "doing"
    DONE = "done"


class TodoPriority(str, Enum):
    """Priority of a workspace todo item (Phase B)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DocumentStatus(str, Enum):
    """Lifecycle of a workspace document (Phase B)."""

    DRAFT = "draft"
    FINAL = "final"


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
    WAITING = "waiting"
    ROUTED = "routed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class MemoryScope(str, Enum):
    """Workspace memory scopes (Phase 3).

    Phase A (AI Operating Workspace redesign) added ``USER``: user-level
    preferences and facts (writing style, favorite frameworks, defaults)
    that persist across projects — the vision's "User Preferences" layer.
    """

    CONVERSATION = "conversation"
    PROJECT = "project"
    GLOBAL = "global"
    USER = "user"


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


class IntentKind(str, Enum):
    """Intent Router intents (AI Operating Workspace redesign).

    The Intent Router runs BEFORE planning and gates whether a request may
    enter the workspace/artifact pipeline at all:

    - ``CONVERSATION`` / ``QUESTION_ANSWERING`` — pure chat: answered directly
      by the selected language model. Never plan, never touch the workspace,
      never create folders/files/manifests, never emit JSON.
    - ``WORKSPACE_MANAGEMENT`` — deterministic backend operations (list /
      search / read / rename / move / delete / folders): no model consulted.
    - ``TOOL_EXECUTION`` — explicit run/execute/automate commands.
    - ``GOAL_PLANNING`` — goals are the atomic unit: explicit goal statements.
    - ``FILE_GENERATION`` / ``FILE_EDITING`` — the only intents allowed to
      enter the artifact pipeline; file creation always requires explicit
      intent (build/create/write/generate/edit/modify/... targeting a file).

    Conversation and tool execution are completely separate surfaces: a
    greeting or factual question can never produce workspace artifacts.
    """

    CONVERSATION = "conversation"
    QUESTION_ANSWERING = "question_answering"
    WORKSPACE_MANAGEMENT = "workspace_management"
    TOOL_EXECUTION = "tool_execution"
    GOAL_PLANNING = "goal_planning"
    FILE_GENERATION = "file_generation"
    FILE_EDITING = "file_editing"

    @property
    def is_chat_only(self) -> bool:
        """True for intents that must never enter the workspace pipeline."""
        return self in (IntentKind.CONVERSATION, IntentKind.QUESTION_ANSWERING)
