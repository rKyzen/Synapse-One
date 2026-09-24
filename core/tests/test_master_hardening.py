"""Tests for Master AI Hardening, Exact Model Matrix, Fast Path Expansion, and TaskDAG Validation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from synapse.bootstrap import Boot, create_container
from synapse.config.paths import SynapsePaths
from synapse.contracts import ModelProvider
from synapse.domain import (
    Capability,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ComplexityResult,
    Decision,
    IntentResult,
    IntentType,
    LatencyTier,
    ModelDescriptor,
    ModelMetadata,
    PrivacyMode,
    PrivacyResult,
    ProviderKind,
    ProviderState,
    TaskKind,
)
from synapse.hardware.model_matrix import (
    EXACT_MODEL_MATRIX,
    ModelRole,
    capability_to_role,
    get_exact_model,
)
from synapse.hardware.tier_resolver import HardwareTier
from synapse.master.agent import MasterAgent
from synapse.master.fast_path import (
    FastPathResult,
    FastPathType,
    check_fast_path,
    evaluate_arithmetic,
)
from synapse.master.orchestrator import (
    AIMasterOrchestrator,
    _DIVIDER_SYSTEM_PROMPT,
    extract_json_object,
    validate_plan_dag,
)
from synapse.master.schemas import (
    ExecutionStrategy,
    SubTask,
    SubTaskIntent,
    TaskDecompositionPlan,
)
from synapse.router.router import Router


# ============================================================================
# 1. Exact Model Matrix Tests
# ============================================================================

def test_exact_model_matrix_tier1():
    """Verify Tier 1 model resolution across all roles."""
    assert get_exact_model(HardwareTier.TIER1, ModelRole.CONVERSATION) == "gemma3:1b"
    assert get_exact_model(HardwareTier.TIER1, ModelRole.MATH_REASONING) == "gemma3:1b"
    assert get_exact_model(HardwareTier.TIER1, ModelRole.CODING) == "qwen2.5-coder:1.5b"
    assert get_exact_model(HardwareTier.TIER1, ModelRole.PLANNING) == "qwen3:1.7b"
    assert get_exact_model(HardwareTier.TIER1, ModelRole.VISION) == "moondream"
    assert get_exact_model(HardwareTier.TIER1, ModelRole.EMBEDDINGS) == "all-minilm"


def test_exact_model_matrix_tier2():
    """Verify Tier 2 model resolution across all roles."""
    assert get_exact_model(HardwareTier.TIER2, ModelRole.CONVERSATION) == "gemma3:4b"
    assert get_exact_model(HardwareTier.TIER2, ModelRole.MATH_REASONING) == "gemma3:4b"
    assert get_exact_model(HardwareTier.TIER2, ModelRole.CODING) == "qwen2.5-coder:7b"
    assert get_exact_model(HardwareTier.TIER2, ModelRole.PLANNING) == "qwen3:4b"
    assert get_exact_model(HardwareTier.TIER2, ModelRole.COMPLEX_TECHNICAL) == "qwen2.5:7b"
    assert get_exact_model(HardwareTier.TIER2, ModelRole.VISION) == "qwen2.5vl:7b"
    assert get_exact_model(HardwareTier.TIER2, ModelRole.EMBEDDINGS) == "nomic-embed-text"


def test_exact_model_matrix_tier3():
    """Verify Tier 3 model resolution across all roles."""
    assert get_exact_model(HardwareTier.TIER3, ModelRole.CONVERSATION) == "gemma3:12b"
    assert get_exact_model(HardwareTier.TIER3, ModelRole.MATH_REASONING) == "gemma3:12b"
    assert get_exact_model(HardwareTier.TIER3, ModelRole.CODING) == "qwen2.5-coder:14b"
    assert get_exact_model(HardwareTier.TIER3, ModelRole.PLANNING) == "qwen2.5:14b"
    assert get_exact_model(HardwareTier.TIER3, ModelRole.COMPLEX_TECHNICAL) == "qwen2.5:14b"
    assert get_exact_model(HardwareTier.TIER3, ModelRole.VISION) == "qwen2.5vl:7b"
    assert get_exact_model(HardwareTier.TIER3, ModelRole.EMBEDDINGS) == "nomic-embed-text"


def test_exact_model_matrix_tier3_plus_guardrail():
    """Verify 32B model is ONLY selected for deep reasoning on Tier 3+, never for standard tasks."""
    # Deep reasoning on Tier 3+ gets 32B
    assert get_exact_model(HardwareTier.TIER3_PLUS, ModelRole.DEEP_REASONING, is_deep_reasoning=True) == "qwen2.5:32b"

    # Standard coding / conversation on Tier 3+ gets standard Tier 3 models
    assert get_exact_model(HardwareTier.TIER3_PLUS, ModelRole.CODING) == "qwen2.5-coder:14b"
    assert get_exact_model(HardwareTier.TIER3_PLUS, ModelRole.CONVERSATION) == "gemma3:12b"

    # Deep reasoning on Tier 2 falls back to Tier 2 math/reasoning model (gemma3:4b)
    assert get_exact_model(HardwareTier.TIER2, ModelRole.DEEP_REASONING, is_deep_reasoning=True) == "gemma3:4b"


def test_capability_to_role_mapping():
    """Verify Capability enum maps cleanly to ModelRole."""
    assert capability_to_role(Capability.CODING) == ModelRole.CODING
    assert capability_to_role(Capability.MATH) == ModelRole.MATH_REASONING
    assert capability_to_role(Capability.REASONING) == ModelRole.MATH_REASONING
    assert capability_to_role(Capability.PLANNING) == ModelRole.PLANNING
    assert capability_to_role(Capability.VISION) == ModelRole.VISION
    assert capability_to_role(Capability.EMBEDDINGS) == ModelRole.EMBEDDINGS
    assert capability_to_role(Capability.CHAT) == ModelRole.CONVERSATION


# ============================================================================
# 2. Fast Path Expansion Tests
# ============================================================================

def test_fast_path_workspace_queries():
    """Verify workspace / project structure queries are intercepted by fast path."""
    for query in ["list files", "show project structure", "show workspace files", "tree", "list directory", "show files"]:
        result = check_fast_path(query)
        assert result is not None, f"Failed to match fast path for: {query}"
        assert result.path_type == FastPathType.WORKSPACE_QUERY
        assert result.is_fast_path is True


def test_fast_path_loaded_models():
    """Verify loaded models queries are intercepted by fast path."""
    for query in ["what models are loaded", "loaded models", "which models are running", "show loaded models"]:
        result = check_fast_path(query)
        assert result is not None, f"Failed to match fast path for: {query}"
        assert result.path_type == FastPathType.LOADED_MODELS
        assert result.is_fast_path is True


def test_fast_path_system_status():
    """Verify system status / health queries are intercepted by fast path."""
    for query in ["system status", "health check", "is system healthy", "system health"]:
        result = check_fast_path(query)
        assert result is not None, f"Failed to match fast path for: {query}"
        assert result.path_type == FastPathType.SYSTEM_STATUS
        assert result.is_fast_path is True


def test_fast_path_arithmetic():
    """Verify arithmetic queries compute instant deterministic results."""
    res = check_fast_path("45 * 2 + 10")
    assert res is not None
    assert res.path_type == FastPathType.DIRECT_ARITHMETIC
    assert res.direct_response == "The answer is 100."

    assert evaluate_arithmetic("2 ** 8") == "256"
    assert evaluate_arithmetic("(100 - 25) / 5") == "15"


def test_fast_path_greetings_and_ping():
    """Verify greetings and ping instant responses."""
    assert check_fast_path("hello").path_type == FastPathType.FAST_CHAT
    assert check_fast_path("ping").path_type == FastPathType.DIRECT_IDENTITY
    assert check_fast_path("ping").direct_response == "pong"


# ============================================================================
# 3. TaskDAG Validation Tests
# ============================================================================

def test_validate_plan_dag_valid():
    """A well-formed sequential or DAG plan passes validation."""
    plan = TaskDecompositionPlan(
        execution_strategy=ExecutionStrategy.SEQUENTIAL,
        tasks=[
            SubTask(
                task_id=1,
                intent=SubTaskIntent.PLANNING,
                sub_prompt="Plan database schema",
                assigned_model="qwen3:4b",
                dependencies=[],
            ),
            SubTask(
                task_id=2,
                intent=SubTaskIntent.CODING,
                sub_prompt="Implement database migrations",
                assigned_model="qwen2.5-coder:7b",
                dependencies=[1],
            ),
        ],
    )
    validate_plan_dag(plan)
    dag = plan.to_dag()
    assert len(dag.tasks) == 3  # 2 tasks + 1 synthesis


def test_validate_plan_dag_rejects_self_dependency():
    """Plan where task depends on itself must fail validation."""
    with pytest.raises(ValueError, match="depends on itself"):
        TaskDecompositionPlan(
            tasks=[
                SubTask(
                    task_id=1,
                    intent=SubTaskIntent.CODING,
                    sub_prompt="Self depending task",
                    assigned_model="qwen2.5-coder:7b",
                    dependencies=[1],
                ),
            ],
        )


def test_validate_plan_dag_rejects_unknown_dependency():
    """Plan where task depends on non-existent task must fail validation."""
    with pytest.raises(ValueError, match="unknown task"):
        TaskDecompositionPlan(
            tasks=[
                SubTask(
                    task_id=1,
                    intent=SubTaskIntent.CODING,
                    sub_prompt="First task",
                    assigned_model="qwen2.5-coder:7b",
                    dependencies=[99],
                ),
            ],
        )


def test_validate_plan_dag_rejects_cycle():
    """Plan with cyclic dependencies must fail validation."""
    with pytest.raises(ValueError, match="[Cc]ycle"):
        plan = TaskDecompositionPlan.model_construct(
            execution_strategy=ExecutionStrategy.SEQUENTIAL,
            tasks=[
                SubTask(
                    task_id=1,
                    intent=SubTaskIntent.CODING,
                    sub_prompt="Task 1",
                    assigned_model="qwen2.5-coder:7b",
                    dependencies=[2],
                ),
                SubTask(
                    task_id=2,
                    intent=SubTaskIntent.CODING,
                    sub_prompt="Task 2",
                    assigned_model="qwen2.5-coder:7b",
                    dependencies=[1],
                ),
            ],
        )
        validate_plan_dag(plan)


# ============================================================================
# 4. Orchestrator Few-Shot Prompt and Token Limit Tests
# ============================================================================

def test_divider_system_prompt_contains_few_shots():
    """Verify Master Orchestrator prompt contains strict few-shot examples."""
    assert "CRITICAL NON-NEGOTIABLE RULES" in _DIVIDER_SYSTEM_PROMPT
    assert "Example 1: Compound Coding Project with Tests and Docs" in _DIVIDER_SYSTEM_PROMPT
    assert "Example 2: Mathematical Proof / Deep Reasoning" in _DIVIDER_SYSTEM_PROMPT
    assert "Example 3: Bug Fix in Workspace" in _DIVIDER_SYSTEM_PROMPT
    assert "Example 4: Image Analysis" in _DIVIDER_SYSTEM_PROMPT


class MockRecordingProvider(ModelProvider):
    """Provider that records chat requests for assertion inspection."""

    provider_id = "mock_rec"
    kind = ProviderKind.LOCAL

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.recorded_requests: list[ChatRequest] = []

    def initialize(self) -> None:
        pass

    def list_models(self) -> list[ModelDescriptor]:
        return [ModelDescriptor(id="qwen3:1.7b", provider_id=self.provider_id)]

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.recorded_requests.append(request)
        content = self.responses.pop(0) if self.responses else "{}"
        return ChatResponse(
            provider_id=self.provider_id,
            model_id=request.model or "qwen3:1.7b",
            kind=self.kind,
            content=content,
        )

    def health(self) -> bool:
        return True

    def supports(self, capability: Capability) -> bool:
        return True

    def shutdown(self) -> None:
        pass

    def to_metadata(self, descriptor: ModelDescriptor) -> ModelMetadata | None:
        return None


def test_orchestrator_enforces_max_tokens_and_retry(temp_paths: SynapsePaths, monkeypatch):
    """Verify orchestrator sets max_tokens=768 and sends repair prompt on retry."""
    monkeypatch.setenv("SYNAPSE_HOME", str(temp_paths.home))
    boot = Boot(create_container(paths=temp_paths))

    valid_json = json.dumps({
        "execution_strategy": "SEQUENTIAL",
        "tasks": [
            {
                "task_id": 1,
                "intent": "CODING",
                "sub_prompt": "Write a python calculator script",
                "assigned_model": "qwen2.5-coder:1.5b",
                "capability": "coding",
                "reasoning": "coding specialist",
                "dependencies": [],
            }
        ],
    })

    # Response 1: invalid json
    # Response 2: valid json
    mock = MockRecordingProvider(responses=["invalid non-json output", valid_json])
    boot.providers._providers["mock_rec"] = mock
    boot.providers._states["mock_rec"] = ProviderState.READY

    boot.registry._models["qwen3:1.7b"] = ModelMetadata(
        id="qwen3:1.7b",
        provider_id="mock_rec",
        kind=ProviderKind.LOCAL,
        privacy_score=1.0,
        capabilities={"reasoning": 0.9, "planning": 0.9, "chat": 0.9},
    )

    from synapse.contracts import ConfigProvider as ConfigProviderProtocol

    config = boot.container.resolve(ConfigProviderProtocol)
    orchestrator = AIMasterOrchestrator(
        providers=boot.providers,
        registry=boot.registry,
        hardware=boot.hardware,
        config=config,
        max_retries=1,
    )

    intent = IntentResult(intent=IntentType.CODING, confidence=1.0)
    complexity = ComplexityResult(score=30, tier=LatencyTier.FAST, is_complex=False, reasons=[])
    privacy = PrivacyResult(mode=PrivacyMode.LOCAL_ONLY, reasons=[])
    decision = Decision(required_capabilities=[Capability.CODING])

    dag = orchestrator.plan("Create calculator", intent, complexity, privacy, decision)

    assert orchestrator.used_ai is True
    assert len(dag.tasks) == 1
    assert len(mock.recorded_requests) == 2

    # Verify max_tokens=768 on both requests
    assert mock.recorded_requests[0].max_tokens == 768
    assert mock.recorded_requests[1].max_tokens == 768

    # Verify second request includes repair prompt
    second_user_msg = mock.recorded_requests[1].messages[1].content
    assert "CRITICAL FIX" in second_user_msg


# ============================================================================
# 5. File Manifest Tolerant Extraction & File Hint Tests
# ============================================================================

def test_manifest_extracts_raw_code_with_hint():
    """When a file task has a hint, raw code without markdown fences is extracted."""
    from synapse.workspace.manifest import parse_file_manifest

    raw_python = "import os\n\ndef add(a, b):\n    return a + b\n"
    ops = parse_file_manifest(raw_python, hint="output.py")
    assert len(ops) == 1
    assert ops[0]["action"] == "write"
    assert ops[0]["path"] == "output.py"
    assert "def add(a, b):" in ops[0]["content"]


def test_manifest_extracts_unclosed_code_fence():
    """When a model generation ends without closing ```, content is still extracted."""
    from synapse.workspace.manifest import parse_file_manifest

    unclosed = "```python\n# file: src/app.py\nprint('hello world')\n"
    ops = parse_file_manifest(unclosed, hint="src/app.py")
    assert len(ops) == 1
    assert ops[0]["action"] == "write"
    assert ops[0]["path"] == "src/app.py"
    assert "print('hello world')" in ops[0]["content"]


def test_decomposition_plan_infers_file_hint_and_output():
    """TaskDecompositionPlan to_dag correctly extracts file_hint and sets file_output."""
    plan = TaskDecompositionPlan(
        execution_strategy=ExecutionStrategy.SEQUENTIAL,
        tasks=[
            SubTask(
                task_id=1,
                intent=SubTaskIntent.PLANNING,
                sub_prompt="Plan architecture for expense tracker in docs/architecture.md",
                assigned_model="qwen3:4b",
                dependencies=[],
            ),
            SubTask(
                task_id=2,
                intent=SubTaskIntent.CODING,
                sub_prompt="Implement core logic in src/expense_tracker.py",
                assigned_model="qwen2.5-coder:7b",
                dependencies=[1],
            ),
            SubTask(
                task_id=3,
                intent=SubTaskIntent.CODING,
                sub_prompt="Write unit tests in test/test_expense_tracker.py",
                assigned_model="qwen2.5-coder:7b",
                dependencies=[2],
            ),
        ],
    )
    dag = plan.to_dag()
    assert dag.get("t1").file_output is True
    assert dag.get("t1").file_hint == "docs/architecture.md"

    assert dag.get("t2").file_output is True
    assert dag.get("t2").file_hint == "src/expense_tracker.py"

    assert dag.get("t3").file_output is True
    assert dag.get("t3").file_hint == "test/test_expense_tracker.py"


def test_escalation_and_verification_accepts_list_context():
    """Verify escalation and verification handle list[RetrievedChunk] context without str+list TypeError."""
    from synapse.contracts.escalation import EscalationDecision
    from synapse.domain.workspace import RetrievedChunk
    from synapse.escalation.impl import OllamaEscalationEngine
    from synapse.verification.impl import CompositeVerifier

    class MockEscalationProvider(ModelProvider):
        provider_id = "mock_esc"
        kind = ProviderKind.LOCAL
        def initialize(self): pass
        def list_models(self): return []
        def chat(self, req: ChatRequest) -> ChatResponse:
            return ChatResponse(provider_id="mock_esc", model_id="target", kind=self.kind, content="Corrected answer")
        def health(self): return True
        def supports(self, cap): return True
        def shutdown(self): pass
        def to_metadata(self, d): return None

    class MockProviders:
        def get(self, pid): return MockEscalationProvider()

    engine = OllamaEscalationEngine(MockProviders())
    decision = EscalationDecision(
        should_escalate=True,
        target_model=ModelMetadata(id="target", provider_id="mock_esc", kind=ProviderKind.LOCAL),
        reason="low confidence",
        current_score=0.4,
        expected_improvement=0.3,
    )

    chunks = [
        RetrievedChunk(file_id="f1", text="Fact from document", file_name="doc.txt", score=0.9),
        RetrievedChunk(file_id="f2", text="Another fact", file_name="notes.md", score=0.8),
    ]

    # Must not raise TypeError: can only concatenate str (not "list") to str
    new_resp, new_conf = engine.escalate(
        "Explain X",
        "Wrong answer",
        decision,
        context=chunks,
    )
    assert new_resp == "Corrected answer"
    assert new_conf > 0.4

    # Verifier must also accept list context
    verifier = CompositeVerifier()
    res = verifier.verify("Explain X", "Fact from document is true.", context=chunks)
    assert res is not None

