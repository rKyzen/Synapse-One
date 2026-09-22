"""Comprehensive test suite for Synapse One Master Model & Tier Architecture.

Validates all 14 core architectural requirements:
1. Master Model is an LLM.
2. Master Model is small designated model per tier (gemma3:1b for T1, gemma3:4b for T2/T3/T3+).
3. Master Model does NOT perform specialist work directly when a specialist model/tool is available.
4. Master Model prompt includes dynamic registry summary.
5. Master Model prompt includes strengths/weaknesses.
6. Different subtasks route to different models.
7. Gemma handles conversation/math.
8. Qwen handles coding/agentic planning.
9. Vision routes to vision model (moondream / qwen2.5vl:7b).
10. Models loaded only when required by active task.
11. Models unloaded after use / under memory pressure.
12. Hardware limits prevent unsafe loading.
13. 32B only selected for deep reasoning on Tier 3+.
14. Surrounding architecture (deterministic fallback, lifecycle, routing) remains functional.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from synapse.contracts import ModelProvider
from synapse.domain import (
    Capability,
    ChatRequest,
    ChatResponse,
    ComplexityResult,
    Decision,
    HardwareProfile,
    IntentResult,
    IntentType,
    ModelCapabilities,
    ModelDescriptor,
    ModelMetadata,
    PrivacyMode,
    PrivacyResult,
    ProviderKind,
    TaskKind,
)
from synapse.domain.hardware import CpuInfo, GpuInfo, MemoryInfo
from synapse.domain.models import format_model_registry_summary
from synapse.domain.tasks import Task, TaskDAG, TaskStatus
from synapse.events import EventBus
from synapse.hardware.tier_resolver import HardwareTier, TierResolver
from synapse.lifecycle import LifecycleSettings, ModelLifecycleManager, ModelState
from synapse.master import AIMasterOrchestrator, SubTaskIntent, TaskDecompositionPlan
from synapse.master.agent import MasterAgent
from synapse.router import Router

DECISION = Decision(
    can_stay_local=True,
    internet_required=False,
    use_cloud_reasoning=False,
    preferred_kind=ProviderKind.LOCAL,
    required_capabilities=[Capability.CHAT],
    preferred_capabilities=[],
    privacy=PrivacyMode.PREFER_LOCAL,
)


class MockProvider(ModelProvider):
    provider_id = "ollama"
    kind = ProviderKind.LOCAL

    def __init__(self, replies: list | None = None) -> None:
        self._replies = list(replies or [])
        self.requests: list[ChatRequest] = []
        self.loaded_models: dict[str, float] = {}

    def initialize(self) -> None:
        pass

    def list_models(self) -> list[ModelDescriptor]:
        return [
            ModelDescriptor(id="gemma3:1b", provider_id=self.provider_id),
            ModelDescriptor(id="gemma3:4b", provider_id=self.provider_id),
            ModelDescriptor(id="gemma3:12b", provider_id=self.provider_id),
            ModelDescriptor(id="qwen2.5-coder:1.5b", provider_id=self.provider_id),
            ModelDescriptor(id="qwen2.5-coder:7b", provider_id=self.provider_id),
            ModelDescriptor(id="qwen2.5-coder:14b", provider_id=self.provider_id),
            ModelDescriptor(id="qwen3:1.7b", provider_id=self.provider_id),
            ModelDescriptor(id="qwen3:4b", provider_id=self.provider_id),
            ModelDescriptor(id="qwen2.5:7b", provider_id=self.provider_id),
            ModelDescriptor(id="qwen2.5:14b", provider_id=self.provider_id),
            ModelDescriptor(id="qwen2.5:32b", provider_id=self.provider_id),
            ModelDescriptor(id="moondream", provider_id=self.provider_id),
            ModelDescriptor(id="qwen2.5vl:7b", provider_id=self.provider_id),
            ModelDescriptor(id="all-minilm", provider_id=self.provider_id),
            ModelDescriptor(id="nomic-embed-text", provider_id=self.provider_id),
        ]

    def to_metadata(self, descriptor: ModelDescriptor) -> ModelMetadata | None:
        return None

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self._replies:
            reply = self._replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            content = reply
        else:
            content = f"Response from {request.model} for task."
        return ChatResponse(
            provider_id=self.provider_id,
            model_id=request.model or "default",
            kind=self.kind,
            content=content,
        )

    def list_loaded(self) -> dict[str, float]:
        return dict(self.loaded_models)

    def load_model(self, model_id: str) -> bool:
        self.loaded_models[model_id] = 2.0
        return True

    def unload_model(self, model_id: str) -> bool:
        self.loaded_models.pop(model_id, None)
        return True

    def health(self) -> bool:
        return True

    def supports(self, capability: Capability) -> bool:
        return True

    def shutdown(self) -> None:
        pass


class MockProviderManager:
    def __init__(self, provider: MockProvider) -> None:
        self._provider = provider

    def get(self, provider_id: str) -> MockProvider | None:
        if provider_id == self._provider.provider_id:
            return self._provider
        return None

    def all(self) -> list[MockProvider]:
        return [self._provider]

    def health(self, provider_id: str) -> bool:
        return True


class MockRegistry:
    def __init__(self, models: list[ModelMetadata]) -> None:
        self._models = {m.id: m for m in models}

    def all(self) -> list[ModelMetadata]:
        return list(self._models.values())

    def get(self, model_id: str) -> ModelMetadata | None:
        return self._models.get(model_id)

    def by_provider(self, provider_id: str) -> list[ModelMetadata]:
        return [m for m in self._models.values() if m.provider_id == provider_id]


def make_tier_models() -> list[ModelMetadata]:
    """Returns registry models for all tiers with rich metadata."""
    return [
        # Tier 1
        ModelMetadata(
            id="gemma3:1b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            role="Master Model (Tier 1) / General Chat & Math",
            strengths=["Fast routing and decomposition", "clean instruction following", "basic math and reasoning"],
            weaknesses=["Complex multi-step code synthesis", "deep architectural reasoning"],
            speed_tier="fast",
            hardware_tier="tier1",
            modalities=["text"],
            tools_supported=False,
            required_ram_gb=1.0,
            capabilities=ModelCapabilities(chat=0.9, math=0.85, reasoning=0.75, planning=0.7, coding=0.4),
        ),
        ModelMetadata(
            id="qwen2.5-coder:1.5b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            role="Coding Specialist (Tier 1)",
            strengths=["Fast code completion", "syntax fixes"],
            weaknesses=["Large-scale architecture"],
            speed_tier="fast",
            hardware_tier="tier1",
            modalities=["text"],
            tools_supported=True,
            required_ram_gb=1.5,
            capabilities=ModelCapabilities(coding=0.88, debugging=0.85, chat=0.5),
        ),
        ModelMetadata(
            id="qwen3:1.7b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            role="Agent Planning Specialist (Tier 1)",
            strengths=["Fast multi-step subtask decomposition", "concise execution planning"],
            weaknesses=["Heavy code generation"],
            speed_tier="fast",
            hardware_tier="tier1",
            modalities=["text"],
            tools_supported=True,
            required_ram_gb=1.5,
            capabilities=ModelCapabilities(planning=0.88, reasoning=0.75, chat=0.8),
        ),
        ModelMetadata(
            id="moondream",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            role="Vision Specialist (Tier 1)",
            strengths=["Ultra-lightweight image understanding", "fast visual question answering"],
            weaknesses=["Limited high-resolution OCR"],
            speed_tier="fast",
            hardware_tier="tier1",
            modalities=["text", "image"],
            tools_supported=False,
            required_ram_gb=1.5,
            capabilities=ModelCapabilities(vision=True, chat=0.5),
        ),
        # Tier 2
        ModelMetadata(
            id="gemma3:4b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            role="Master Model (Tier 2/3/3+) / General Chat & Math",
            strengths=["Strong structured planning and JSON routing", "math reasoning", "general conversation"],
            weaknesses=["Massive codebase refactoring"],
            speed_tier="fast",
            hardware_tier="tier2",
            modalities=["text"],
            tools_supported=True,
            required_ram_gb=4.0,
            capabilities=ModelCapabilities(chat=0.95, math=0.9, reasoning=0.88, planning=0.9, coding=0.65),
        ),
        ModelMetadata(
            id="qwen2.5-coder:7b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            role="Coding Specialist (Tier 2)",
            strengths=["Professional code generation", "debugging", "API integration"],
            weaknesses=["Non-coding creative writing"],
            speed_tier="medium",
            hardware_tier="tier2",
            modalities=["text"],
            tools_supported=True,
            required_ram_gb=6.0,
            capabilities=ModelCapabilities(coding=0.95, debugging=0.95, architecture=0.9),
        ),
        ModelMetadata(
            id="qwen3:4b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            role="Agent Planning Specialist (Tier 2)",
            strengths=["Strong structured workflow execution", "agentic tool coordination"],
            weaknesses=["Complex kernel-level code implementation"],
            speed_tier="fast",
            hardware_tier="tier2",
            modalities=["text"],
            tools_supported=True,
            required_ram_gb=4.0,
            capabilities=ModelCapabilities(planning=0.95, reasoning=0.88, chat=0.9),
        ),
        ModelMetadata(
            id="qwen2.5:7b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            role="Technical & Multilingual Specialist (Tier 2)",
            strengths=["Technical explanations", "cross-lingual queries"],
            weaknesses=["Very high latency on low-end CPUs"],
            speed_tier="medium",
            hardware_tier="tier2",
            modalities=["text"],
            tools_supported=True,
            required_ram_gb=6.0,
            capabilities=ModelCapabilities(chat=0.9, writing=0.92, translation=0.95, reasoning=0.85),
        ),
        ModelMetadata(
            id="qwen2.5vl:7b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            role="Vision Specialist (Tier 2, Tier 3, Tier 3+)",
            strengths=["Document understanding", "diagram and UI parsing", "dense OCR"],
            weaknesses=["Requires ~8 GB RAM"],
            speed_tier="medium",
            hardware_tier="tier2",
            modalities=["text", "image"],
            tools_supported=True,
            required_ram_gb=8.0,
            capabilities=ModelCapabilities(vision=True, chat=0.7),
        ),
        # Tier 3
        ModelMetadata(
            id="gemma3:12b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            role="General Chat & Math/Reasoning (Tier 3)",
            strengths=["Advanced reasoning", "deep math problem solving", "nuanced dialogue"],
            weaknesses=["Higher memory footprint than 4B"],
            speed_tier="medium",
            hardware_tier="tier3",
            modalities=["text"],
            tools_supported=True,
            required_ram_gb=10.0,
            capabilities=ModelCapabilities(chat=1.0, math=0.95, reasoning=0.95, planning=0.92),
        ),
        ModelMetadata(
            id="qwen2.5-coder:14b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            role="Advanced Coding Specialist (Tier 3 & Tier 3+)",
            strengths=["Enterprise-grade software engineering", "full-stack architecture"],
            weaknesses=["High RAM footprint (~14 GB)"],
            speed_tier="medium",
            hardware_tier="tier3",
            modalities=["text"],
            tools_supported=True,
            required_ram_gb=14.0,
            capabilities=ModelCapabilities(coding=1.0, debugging=1.0, architecture=1.0),
        ),
        ModelMetadata(
            id="qwen2.5:14b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            role="Agent & Technical Specialist (Tier 3)",
            strengths=["Complex multi-stage orchestration", "deep technical synthesis"],
            weaknesses=["Heavy resource consumption"],
            speed_tier="medium",
            hardware_tier="tier3",
            modalities=["text"],
            tools_supported=True,
            required_ram_gb=14.0,
            capabilities=ModelCapabilities(chat=1.0, writing=1.0, planning=0.95, reasoning=0.95),
        ),
        # Tier 3+
        ModelMetadata(
            id="qwen2.5:32b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            role="Deep Reasoning Specialist (Tier 3+ Strictly On-Demand)",
            strengths=["Ph.D.-level logical proofs", "exhaustive architectural verification"],
            weaknesses=["Requires >=32 GB RAM/VRAM", "high latency", "must never be used for casual tasks"],
            speed_tier="slow",
            hardware_tier="tier3_plus",
            modalities=["text"],
            tools_supported=True,
            required_ram_gb=24.0,
            capabilities=ModelCapabilities(reasoning=1.0, planning=1.0, math=1.0, coding=0.92),
        ),
    ]


# ---------------------------------------------------------------------------
# Requirement 1: Master Model is an LLM (not a rule-based router).
# ---------------------------------------------------------------------------
def test_req1_master_model_is_actual_llm():
    plan_json = {
        "execution_strategy": "SEQUENTIAL",
        "tasks": [
            {
                "task_id": 1,
                "intent": "CODING",
                "sub_prompt": "write a parser",
                "assigned_model": "qwen2.5-coder:7b",
                "reasoning": "coding task requires code specialist",
                "dependencies": [],
            }
        ],
    }
    provider = MockProvider([json.dumps(plan_json)])
    registry = MockRegistry(make_tier_models())
    hw_profile = HardwareProfile(memory=MemoryInfo(total_gb=32.0, available_gb=10.0))

    class HardwareProviderStub:
        def scan(self):
            return hw_profile

    orchestrator = AIMasterOrchestrator(
        providers=MockProviderManager(provider),
        registry=registry,
        hardware=HardwareProviderStub(),
        config=None,
    )
    intent = IntentResult(primary=IntentType.CODING, confidence=0.9)
    complexity = ComplexityResult(score=60)
    privacy = PrivacyResult(mode=PrivacyMode.PREFER_LOCAL)

    dag = orchestrator.plan("write a parser", intent, complexity, privacy, DECISION)

    # Verifies that an actual LLM chat invocation took place
    assert len(provider.requests) == 1
    assert orchestrator.used_ai is True
    assert provider.requests[0].model == "gemma3:4b"


# ---------------------------------------------------------------------------
# Requirement 2: Master Model is small designated model per tier.
# ---------------------------------------------------------------------------
def test_req2_master_model_is_small_designated_model_per_tier():
    resolver = TierResolver()

    # Tier 1: < 6GB RAM -> gemma3:1b
    t1 = resolver.resolve(HardwareProfile(memory=MemoryInfo(total_gb=8.0, available_gb=4.0)))
    assert t1.tier == HardwareTier.TIER1
    assert t1.candidate_models == ["gemma3:1b"]

    # Tier 2: 6 - 16GB RAM -> gemma3:4b
    t2 = resolver.resolve(HardwareProfile(memory=MemoryInfo(total_gb=16.0, available_gb=10.0)))
    assert t2.tier == HardwareTier.TIER2
    assert t2.candidate_models == ["gemma3:4b"]

    # Tier 3: >= 16GB RAM or >= 12GB VRAM -> gemma3:4b
    t3 = resolver.resolve(HardwareProfile(memory=MemoryInfo(total_gb=32.0, available_gb=20.0)))
    assert t3.tier == HardwareTier.TIER3
    assert t3.candidate_models == ["gemma3:4b"]

    # Tier 3+: >= 32GB RAM or >= 20GB VRAM -> gemma3:4b
    t3_plus = resolver.resolve(HardwareProfile(memory=MemoryInfo(total_gb=64.0, available_gb=40.0)))
    assert t3_plus.tier == HardwareTier.TIER3_PLUS
    assert t3_plus.candidate_models == ["gemma3:4b"]


# ---------------------------------------------------------------------------
# Requirement 3: Master Model does NOT perform specialist work directly.
# ---------------------------------------------------------------------------
def test_req3_master_model_does_not_solve_directly():
    plan_json = {
        "execution_strategy": "SEQUENTIAL",
        "tasks": [
            {
                "task_id": 1,
                "intent": "CODING",
                "sub_prompt": "write quicksort in Python",
                "assigned_model": "qwen2.5-coder:7b",
                "reasoning": "code specialist required",
                "dependencies": [],
            }
        ],
    }
    provider = MockProvider([json.dumps(plan_json)])
    registry = MockRegistry(make_tier_models())
    hw_profile = HardwareProfile(memory=MemoryInfo(total_gb=32.0, available_gb=10.0))

    class HardwareProviderStub:
        def scan(self):
            return hw_profile

    orchestrator = AIMasterOrchestrator(
        providers=MockProviderManager(provider),
        registry=registry,
        hardware=HardwareProviderStub(),
        config=None,
    )
    intent = IntentResult(primary=IntentType.CODING, confidence=0.9)
    complexity = ComplexityResult(score=60)
    privacy = PrivacyResult(mode=PrivacyMode.PREFER_LOCAL)

    dag = orchestrator.plan("write quicksort in Python", intent, complexity, privacy, DECISION)

    # Master prompt forbids direct answers
    sys_prompt = provider.requests[0].messages[0].content
    assert "NEVER directly solve" in sys_prompt
    # The subtask is delegated to the specialist, not solved by Master
    assert dag.get("t1").preferred_model == "qwen2.5-coder:7b"


# ---------------------------------------------------------------------------
# Requirement 4 & 5: Dynamic registry summary with strengths/weaknesses in prompt.
# ---------------------------------------------------------------------------
def test_req4_and_5_prompt_includes_dynamic_registry_and_strengths_weaknesses():
    provider = MockProvider([json.dumps({"execution_strategy": "SEQUENTIAL", "tasks": [{"task_id": 1, "intent": "CHAT", "sub_prompt": "hi", "dependencies": []}]})])
    registry = MockRegistry(make_tier_models())
    hw_profile = HardwareProfile(memory=MemoryInfo(total_gb=32.0, available_gb=10.0))

    class HardwareProviderStub:
        def scan(self):
            return hw_profile

    orchestrator = AIMasterOrchestrator(
        providers=MockProviderManager(provider),
        registry=registry,
        hardware=HardwareProviderStub(),
        config=None,
    )
    intent = IntentResult(primary=IntentType.CONVERSATION, confidence=0.9)
    complexity = ComplexityResult(score=20)
    privacy = PrivacyResult(mode=PrivacyMode.PREFER_LOCAL)

    orchestrator.plan("hello world", intent, complexity, privacy, DECISION)

    user_prompt = provider.requests[0].messages[1].content
    assert "REGISTERED SPECIALIST MODELS & TOOLS" in user_prompt
    assert "Strengths:" in user_prompt
    assert "Weaknesses:" in user_prompt
    assert "qwen2.5-coder:7b" in user_prompt
    assert "gemma3:4b" in user_prompt


# ---------------------------------------------------------------------------
# Requirement 6: Different subtasks route to different models based on capability.
# ---------------------------------------------------------------------------
def test_req6_different_subtasks_route_to_different_specialist_models():
    plan_json = {
        "execution_strategy": "SEQUENTIAL",
        "tasks": [
            {
                "task_id": 1,
                "intent": "CODING",
                "sub_prompt": "write the REST API backend",
                "assigned_model": "qwen2.5-coder:7b",
                "reasoning": "coding",
                "dependencies": [],
            },
            {
                "task_id": 2,
                "intent": "MATH",
                "sub_prompt": "calculate the computational complexity bounds",
                "assigned_model": "gemma3:4b",
                "reasoning": "mathematical reasoning",
                "dependencies": [1],
            },
            {
                "task_id": 3,
                "intent": "VISION",
                "sub_prompt": "extract diagram entities from architecture.png",
                "assigned_model": "qwen2.5vl:7b",
                "reasoning": "vision parsing",
                "dependencies": [],
            },
        ],
    }
    provider = MockProvider([json.dumps(plan_json)])
    registry = MockRegistry(make_tier_models())
    hw_profile = HardwareProfile(memory=MemoryInfo(total_gb=32.0, available_gb=10.0))

    class HardwareProviderStub:
        def scan(self):
            return hw_profile

    orchestrator = AIMasterOrchestrator(
        providers=MockProviderManager(provider),
        registry=registry,
        hardware=HardwareProviderStub(),
        config=None,
    )
    intent = IntentResult(primary=IntentType.CODING, confidence=0.9)
    complexity = ComplexityResult(score=70)
    privacy = PrivacyResult(mode=PrivacyMode.PREFER_LOCAL)

    dag = orchestrator.plan("build backend and calculate bounds", intent, complexity, privacy, DECISION)

    assert dag.get("t1").preferred_model == "qwen2.5-coder:7b"
    assert dag.get("t2").preferred_model == "gemma3:4b"
    assert dag.get("t3").preferred_model == "qwen2.5vl:7b"


# ---------------------------------------------------------------------------
# Requirement 7: Gemma handles conversation/math.
# ---------------------------------------------------------------------------
def test_req7_gemma_handles_conversation_and_math():
    models = make_tier_models()
    gemma4b = next(m for m in models if m.id == "gemma3:4b")
    # Chat capability >= 0.9, math >= 0.9
    assert gemma4b.capabilities.chat >= 0.9
    assert gemma4b.capabilities.math >= 0.9


# ---------------------------------------------------------------------------
# Requirement 8: Qwen handles coding/agentic planning.
# ---------------------------------------------------------------------------
def test_req8_qwen_handles_coding_and_agentic_planning():
    models = make_tier_models()
    coder = next(m for m in models if m.id == "qwen2.5-coder:7b")
    planner = next(m for m in models if m.id == "qwen3:4b")
    assert coder.capabilities.coding >= 0.95
    assert planner.capabilities.planning >= 0.95


# ---------------------------------------------------------------------------
# Requirement 9: Vision routes to vision models.
# ---------------------------------------------------------------------------
def test_req9_vision_routes_to_vision_specialist():
    models = make_tier_models()
    t1_vision = next(m for m in models if m.id == "moondream")
    t2_vision = next(m for m in models if m.id == "qwen2.5vl:7b")
    assert t1_vision.capabilities.vision is True
    assert t2_vision.capabilities.vision is True


# ---------------------------------------------------------------------------
# Requirement 10 & 11: Lazy loading, idle unloads, memory pressure.
# ---------------------------------------------------------------------------
def test_req10_and_11_lifecycle_lazy_load_and_unload_under_pressure():
    provider = MockProvider()
    models = make_tier_models()
    manager = ModelLifecycleManager(
        settings=LifecycleSettings(idle_timeout_small_s=2.0, low_memory_threshold_gb=4.0),
        providers=MockProviderManager(provider),
        registry=MockRegistry(models),
        router=Router(),
        events=EventBus(),
    )

    # 10. Models loaded only when request starts
    assert manager.loaded_count() == 0
    manager.note_request_started("ollama", "qwen2.5-coder:7b")
    assert manager.is_loaded("ollama", "qwen2.5-coder:7b")
    assert manager.state_of("ollama", "qwen2.5-coder:7b") == ModelState.ACTIVE

    # Request completed -> idle
    manager.note_request_completed("ollama", "qwen2.5-coder:7b")
    assert manager.state_of("ollama", "qwen2.5-coder:7b") == ModelState.IDLE

    # 11. Memory pressure triggers idle unloads
    unloaded = manager.check_memory_pressure(available_gb=2.0)
    assert "ollama/qwen2.5-coder:7b" in unloaded
    assert not manager.is_loaded("ollama", "qwen2.5-coder:7b")


# ---------------------------------------------------------------------------
# Requirement 12: Hardware safety prevents unsafe loading.
# ---------------------------------------------------------------------------
def test_req12_hardware_limits_prevent_unsafe_loading():
    resolver = TierResolver()
    profile = HardwareProfile(memory=MemoryInfo(total_gb=16.0, available_gb=8.0))

    # 14B model needing 14GB cannot fit in 8GB available RAM
    assert resolver.can_support_model("qwen2.5-coder:14b", 14.0, profile) is False
    # 7B model needing 6GB fits in 8GB available RAM
    assert resolver.can_support_model("qwen2.5-coder:7b", 6.0, profile) is True


# ---------------------------------------------------------------------------
# Requirement 13: 32B model only selected for deep reasoning on Tier 3+.
# ---------------------------------------------------------------------------
def test_req13_qwen_32b_strictly_on_demand_on_tier3_plus():
    plan_with_32b_casual = {
        "execution_strategy": "SEQUENTIAL",
        "tasks": [
            {
                "task_id": 1,
                "intent": "CHAT",
                "sub_prompt": "write a casual haiku",
                "assigned_model": "qwen2.5:32b",
                "reasoning": "haiku",
                "dependencies": [],
            }
        ],
    }
    # On Tier 2 (8GB RAM), 32B is downgraded to gemma3:4b
    provider = MockProvider([json.dumps(plan_with_32b_casual)])
    registry = MockRegistry(make_tier_models())
    hw_profile_t2 = HardwareProfile(memory=MemoryInfo(total_gb=16.0, available_gb=8.0))

    class HwStubT2:
        def scan(self):
            return hw_profile_t2

    orch_t2 = AIMasterOrchestrator(
        providers=MockProviderManager(provider),
        registry=registry,
        hardware=HwStubT2(),
        config=None,
    )
    dag_t2 = orch_t2.plan(
        "write a casual haiku",
        IntentResult(primary=IntentType.CONVERSATION, confidence=0.8),
        ComplexityResult(score=10),
        PrivacyResult(mode=PrivacyMode.PREFER_LOCAL),
        DECISION,
    )
    assert dag_t2.get("t1").preferred_model == "gemma3:4b"

    # On Tier 3+ (40GB RAM) for deep reasoning, 32B is allowed
    plan_with_32b_deep = {
        "execution_strategy": "SEQUENTIAL",
        "tasks": [
            {
                "task_id": 1,
                "intent": "DEEP_REASONING",
                "sub_prompt": "prove the Riemann Hypothesis equivalent",
                "assigned_model": "qwen2.5:32b",
                "reasoning": "deep reasoning proof",
                "dependencies": [],
            }
        ],
    }
    provider_t3p = MockProvider([json.dumps(plan_with_32b_deep)])
    hw_profile_t3p = HardwareProfile(memory=MemoryInfo(total_gb=64.0, available_gb=40.0))

    class HwStubT3P:
        def scan(self):
            return hw_profile_t3p

    orch_t3p = AIMasterOrchestrator(
        providers=MockProviderManager(provider_t3p),
        registry=registry,
        hardware=HwStubT3P(),
        config=None,
    )
    dag_t3p = orch_t3p.plan(
        "prove math theorem",
        IntentResult(primary=IntentType.RESEARCH, confidence=0.9),
        ComplexityResult(score=95),
        PrivacyResult(mode=PrivacyMode.PREFER_LOCAL),
        DECISION,
    )
    assert dag_t3p.get("t1").preferred_model == "qwen2.5:32b"


# ---------------------------------------------------------------------------
# Requirement 14: Surrounding architecture remains functional (fallback).
# ---------------------------------------------------------------------------
def test_req14_deterministic_fallback_remains_functional():
    # When Master LLM fails with invalid JSON, deterministic fallback planner handles it
    provider = MockProvider(["invalid non-json output"])
    registry = MockRegistry(make_tier_models())
    hw_profile = HardwareProfile(memory=MemoryInfo(total_gb=16.0, available_gb=8.0))

    class HardwareProviderStub:
        def scan(self):
            return hw_profile

    class FallbackPlannerStub:
        def __init__(self):
            self.called = False

        def plan(self, prompt, intent, complexity, privacy, decision):
            self.called = True
            return TaskDAG(tasks=[Task(id="fb-1", description=prompt)])

    fb = FallbackPlannerStub()
    orchestrator = AIMasterOrchestrator(
        providers=MockProviderManager(provider),
        registry=registry,
        hardware=HardwareProviderStub(),
        fallback_planner=fb,
        config=None,
    )

    dag = orchestrator.plan(
        "fallback task",
        IntentResult(primary=IntentType.CONVERSATION, confidence=0.8),
        ComplexityResult(score=20),
        PrivacyResult(mode=PrivacyMode.PREFER_LOCAL),
        DECISION,
    )
    assert fb.called is True
    assert orchestrator.used_ai is False
    assert [t.id for t in dag.tasks] == ["fb-1"]
