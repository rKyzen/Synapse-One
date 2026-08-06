"""Decision Engine tests."""

from __future__ import annotations

from synapse.decision import DecisionEngine
from synapse.domain import (
    Capability,
    ComplexityResult,
    HardwareProfile,
    IntentResult,
    IntentType,
    PrivacyMode,
    PrivacyResult,
    ProviderKind,
    WorkspaceKind,
)
from synapse.domain.hardware import RecommendedModelLimits
from synapse.domain.models import ModelCapabilities, ModelMetadata

LOCAL_MODEL = ModelMetadata(
    id="llama3.2",
    provider_id="ollama",
    kind=ProviderKind.LOCAL,
    capabilities=ModelCapabilities(reasoning=0.6, coding=0.5, writing=0.6),
)
CLOUD_MODEL = ModelMetadata(
    id="gpt-4o-mini",
    provider_id="openai",
    kind=ProviderKind.CLOUD,
    context_window=128_000,
    capabilities=ModelCapabilities(reasoning=0.9, coding=0.8, writing=0.9, vision=True),
)


def _hw() -> HardwareProfile:
    return HardwareProfile(
        memory={"total_gb": 16, "available_gb": 8, "used_percent": 50.0},
        recommendations=RecommendedModelLimits(can_run_local_llm=True),
    )


def _intent(t: IntentType) -> IntentResult:
    return IntentResult(primary=t, confidence=0.9)


def test_local_balanced_stays_local():
    engine = DecisionEngine()
    decision = engine.decide(
        _intent(IntentType.CONVERSATION),
        ComplexityResult(score=10),
        PrivacyResult(mode=PrivacyMode.BALANCED),
        _hw(),
        [LOCAL_MODEL, CLOUD_MODEL],
    )
    assert decision.can_stay_local is True
    assert decision.preferred_kind == ProviderKind.LOCAL
    assert decision.use_cloud_reasoning is False
    assert decision.workspace == WorkspaceKind.GENERAL


def test_cloud_required_forces_cloud():
    engine = DecisionEngine()
    decision = engine.decide(
        _intent(IntentType.CONVERSATION),
        _complexity(90),
        PrivacyResult(mode=PrivacyMode.CLOUD_REQUIRED),
        _hw(),
        [LOCAL_MODEL, CLOUD_MODEL],
    )
    assert decision.preferred_kind == ProviderKind.CLOUD
    assert decision.use_cloud_reasoning is True


def test_coding_intent_requires_coding_capability():
    engine = DecisionEngine()
    decision = engine.decide(
        _intent(IntentType.CODING),
        _complexity(60),
        PrivacyResult(mode=PrivacyMode.BALANCED),
        _hw(),
        [LOCAL_MODEL, CLOUD_MODEL],
    )
    assert Capability.CODING in decision.required_capabilities
    assert decision.workspace == WorkspaceKind.CODING


def test_use_memory_is_placeholder():
    engine = DecisionEngine()
    decision = engine.decide(
        _intent(IntentType.GENERAL),
        _complexity(5),
        PrivacyResult(mode=PrivacyMode.BALANCED),
        _hw(),
        [LOCAL_MODEL],
    )
    assert decision.use_memory is False


def _complexity(score: int) -> ComplexityResult:
    return ComplexityResult(score=score)