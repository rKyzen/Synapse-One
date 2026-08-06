"""Router V1 tests — vendor-agnostic selection."""

from __future__ import annotations

from synapse.domain import (
    Capability,
    ComplexityResult,
    Decision,
    HardwareProfile,
    IntentResult,
    IntentType,
    PrivacyMode,
    PrivacyResult,
    ProviderKind,
)
from synapse.domain.hardware import RecommendedModelLimits
from synapse.domain.models import ModelCapabilities, ModelMetadata
from synapse.router import Router

LOCAL_STRONG = ModelMetadata(
    id="qwen3:8b",
    provider_id="ollama",
    kind=ProviderKind.LOCAL,
    latency="fast",
    required_ram_gb=8,
    privacy_score=1.0,
    estimated_cost_per_1k=0.0,
    capabilities=ModelCapabilities(reasoning=0.7, coding=0.8, writing=0.6),
)
LOCAL_WEAK = ModelMetadata(
    id="llama3.2",
    provider_id="ollama",
    kind=ProviderKind.LOCAL,
    latency="fast",
    required_ram_gb=4,
    privacy_score=1.0,
    estimated_cost_per_1k=0.0,
    capabilities=ModelCapabilities(reasoning=0.4, coding=0.4, writing=0.5),
)
CLOUD_MODEL = ModelMetadata(
    id="gpt-4o-mini",
    provider_id="openai",
    kind=ProviderKind.CLOUD,
    context_window=128_000,
    latency="fast",
    required_ram_gb=0,
    privacy_score=0.0,
    estimated_cost_per_1k=0.15,
    capabilities=ModelCapabilities(reasoning=0.9, coding=0.9, writing=0.9),
)


def _decision(kind: ProviderKind, privacy: PrivacyMode, caps: list[Capability] | None = None, preferred: list[Capability] | None = None) -> Decision:
    return Decision(
        can_stay_local=True,
        internet_required=False,
        use_cloud_reasoning=False,
        preferred_kind=kind,
        required_capabilities=caps or [],
        preferred_capabilities=preferred or [],
        privacy=privacy,
    )


def _hw(available_gb: float = 8.0) -> HardwareProfile:
    return HardwareProfile(
        memory={"total_gb": 16, "available_gb": available_gb, "used_percent": 50.0},
        recommendations=RecommendedModelLimits(can_run_local_llm=True),
    )


def _installed(*ids: str) -> dict[str, set[str]]:
    return {"ollama": set(ids), "openai": set(ids)}


def test_router_picks_best_local_for_coding():
    router = Router()
    routing = router.route(
        _decision(ProviderKind.LOCAL, PrivacyMode.PREFER_LOCAL, [Capability.CODING]),
        _hw(),
        [LOCAL_STRONG, LOCAL_WEAK],
        {"ollama": True},
        _installed("qwen3:8b", "llama3.2"),
    )
    assert routing.model_id == "qwen3:8b"
    assert routing.provider_id == "ollama"
    assert routing.confidence > 0.5
    assert routing.reason


def test_router_skips_unhealthy_provider():
    router = Router()
    routing = router.route(
        _decision(ProviderKind.LOCAL, PrivacyMode.PREFER_LOCAL),
        _hw(),
        [LOCAL_STRONG],
        {"ollama": False},
        _installed("qwen3:8b"),
    )
    assert routing.model_id == ""


def test_router_skips_model_not_installed():
    router = Router()
    routing = router.route(
        _decision(ProviderKind.LOCAL, PrivacyMode.PREFER_LOCAL, [Capability.CODING]),
        _hw(),
        [LOCAL_STRONG, LOCAL_WEAK],
        {"ollama": True},
        _installed("llama3.2"),  # qwen3:8b not pulled -> must be excluded
    )
    assert routing.model_id == "llama3.2"


def test_router_empty_installation_routes_nothing():
    router = Router()
    routing = router.route(
        _decision(ProviderKind.LOCAL, PrivacyMode.PREFER_LOCAL),
        _hw(),
        [LOCAL_STRONG],
        {"ollama": True},
        _installed(),
    )
    assert routing.model_id == ""


def test_router_cloud_only_when_required():
    router = Router()
    routing = router.route(
        _decision(ProviderKind.CLOUD, PrivacyMode.CLOUD_REQUIRED),
        _hw(),
        [LOCAL_STRONG, CLOUD_MODEL],
        {"ollama": True, "openai": True},
        {"ollama": {"qwen3:8b"}, "openai": {"gpt-4o-mini"}},
    )
    assert routing.kind == ProviderKind.CLOUD
    assert routing.model_id == "gpt-4o-mini"


def test_router_hardware_gate():
    router = Router()
    routing = router.route(
        _decision(ProviderKind.LOCAL, PrivacyMode.PREFER_LOCAL),
        _hw(available_gb=5.0),  # qwen needs 8GB -> excluded, llama needs 4GB -> ok
        [LOCAL_STRONG, LOCAL_WEAK],
        {"ollama": True},
        _installed("qwen3:8b", "llama3.2"),
    )
    assert routing.model_id == "llama3.2"


def test_router_returns_candidates():
    router = Router()
    routing = router.route(
        _decision(ProviderKind.LOCAL, PrivacyMode.PREFER_LOCAL),
        _hw(),
        [LOCAL_STRONG, LOCAL_WEAK],
        {"ollama": True},
        _installed("qwen3:8b", "llama3.2"),
    )
    assert len(routing.candidates) == 2
    assert routing.candidates[0]["model"] == "qwen3:8b"


# ---------------------------------------------------------------------------
# Phase 2.5 — intelligent routing
# ---------------------------------------------------------------------------

VISION_MODEL = ModelMetadata(
    id="qwen2.5-vl:7b",
    provider_id="ollama",
    kind=ProviderKind.LOCAL,
    latency="medium",
    required_ram_gb=8,
    privacy_score=1.0,
    capabilities=ModelCapabilities(chat=0.6, coding=0.25, vision=True, ocr=1.0, pdf=1.0),
)
CODER_MODEL = ModelMetadata(
    id="qwen2.5-coder:7b",
    provider_id="ollama",
    kind=ProviderKind.LOCAL,
    latency="medium",
    required_ram_gb=8,
    privacy_score=1.0,
    capabilities=ModelCapabilities(
        chat=0.5, coding=1.0, debugging=1.0, architecture=0.95, reasoning=0.8, writing=0.4, json=1.0, terminal=0.7
    ),
)
REASONING_MODEL = ModelMetadata(
    id="llama3.1:8b",
    provider_id="ollama",
    kind=ProviderKind.LOCAL,
    latency="medium",
    required_ram_gb=8,
    privacy_score=1.0,
    capabilities=ModelCapabilities(reasoning=1.0, planning=1.0, math=0.9, coding=0.7, writing=0.75, chat=0.6),
)


def test_vision_model_never_wins_text_only_task():
    router = Router()
    routing = router.route(
        _decision(ProviderKind.LOCAL, PrivacyMode.PREFER_LOCAL),
        _hw(),
        [VISION_MODEL, LOCAL_STRONG],
        {"ollama": True},
        _installed("qwen2.5-vl:7b", "qwen3:8b"),
    )
    assert routing.model_id == "qwen3:8b"
    for c in routing.candidates:
        if c["model"] == "qwen2.5-vl:7b":
            assert c["score"] == 0.0


def test_coding_task_routes_to_coding_specialist():
    router = Router()
    routing = router.route(
        _decision(
            ProviderKind.LOCAL,
            PrivacyMode.PREFER_LOCAL,
            [Capability.CODING, Capability.REASONING],
            preferred=[Capability.CHAT, Capability.DEBUGGING, Capability.ARCHITECTURE, Capability.JSON, Capability.TERMINAL],
        ),
        _hw(),
        [CODER_MODEL, LOCAL_STRONG, VISION_MODEL],
        {"ollama": True},
        _installed("qwen2.5-coder:7b", "qwen3:8b", "qwen2.5-vl:7b"),
    )
    assert routing.model_id == "qwen2.5-coder:7b"


def test_writing_task_routes_to_writing_general_before_reasoning():
    router = Router()
    generalist = ModelMetadata(
        id="qwen2.5:3b",
        provider_id="ollama",
        kind=ProviderKind.LOCAL,
        latency="fast",
        required_ram_gb=4,
        privacy_score=1.0,
        capabilities=ModelCapabilities(
            chat=1.0, writing=0.95, translation=0.9, planning=0.75, reasoning=0.7, coding=0.4
        ),
    )
    routing = router.route(
        _decision(
            ProviderKind.LOCAL,
            PrivacyMode.PREFER_LOCAL,
            [Capability.WRITING],
            preferred=[Capability.CHAT, Capability.TRANSLATION, Capability.PLANNING],
        ),
        _hw(),
        [generalist, REASONING_MODEL, CODER_MODEL, VISION_MODEL],
        {"ollama": True},
        _installed("qwen2.5:3b", "llama3.1:8b", "qwen2.5-coder:7b", "qwen2.5-vl:7b"),
    )
    assert routing.model_id == "qwen2.5:3b"  # writing+chat generalist beats specialists


def test_vision_task_requires_vision_model():
    router = Router()
    routing = router.route(
        _decision(
            ProviderKind.LOCAL,
            PrivacyMode.PREFER_LOCAL,
            [Capability.VISION, Capability.OCR, Capability.PDF],
        ),
        _hw(),
        [VISION_MODEL, LOCAL_STRONG, CODER_MODEL],
        {"ollama": True},
        _installed("qwen2.5-vl:7b", "qwen3:8b", "qwen2.5-coder:7b"),
    )
    assert routing.model_id == "qwen2.5-vl:7b"
    excluded = {e["model"] for e in routing.excluded_models}
    assert "qwen3:8b" in excluded
    assert any("vision" in e["reason"] for e in routing.excluded_models)


def test_deep_reasoning_favors_reasoning_specialist():
    router = Router()
    routing = router.route(
        _decision(
            ProviderKind.LOCAL,
            PrivacyMode.PREFER_LOCAL,
            [Capability.REASONING],
            preferred=[Capability.CHAT, Capability.PLANNING],
        ),
        _hw(),
        [REASONING_MODEL, LOCAL_STRONG],
        {"ollama": True},
        _installed("llama3.1:8b", "qwen3:8b"),
        complexity=95,
    )
    assert routing.model_id == "llama3.1:8b"


def test_embeddings_only_model_excluded_from_chat():
    router = Router()
    embed = ModelMetadata(
        id="nomic-embed-text",
        provider_id="ollama",
        kind=ProviderKind.LOCAL,
        capabilities=ModelCapabilities(embeddings=1.0, chat=0.0),
    )
    routing = router.route(
        _decision(ProviderKind.LOCAL, PrivacyMode.PREFER_LOCAL),
        _hw(),
        [embed, LOCAL_STRONG],
        {"ollama": True},
        _installed("nomic-embed-text", "qwen3:8b"),
    )
    assert routing.model_id == "qwen3:8b"
    assert any("embeddings-only" in e["reason"] for e in routing.excluded_models)


def test_latency_prediction_and_output_tokens():
    router = Router()
    routing = router.route(
        _decision(ProviderKind.LOCAL, PrivacyMode.PREFER_LOCAL),
        _hw(),
        [LOCAL_STRONG],
        {"ollama": True},
        _installed("qwen3:8b"),
        complexity=60,
        prompt="Write a detailed technical design document for a microservice platform",
    )
    assert routing.estimated_latency_s is not None and routing.estimated_latency_s > 0
    assert routing.expected_output_tokens == 800


def test_historical_performance_blends_latency():
    router = Router()
    history = {
        "ollama/qwen3:8b": {
            "samples": 10,
            "avg_latency_s": 4.0,
            "avg_tokens_per_second": 40.0,
            "success_rate": 0.9,
            "timeout_rate": 0.0,
        }
    }
    cold = router.route(
        _decision(ProviderKind.LOCAL, PrivacyMode.PREFER_LOCAL),
        _hw(),
        [LOCAL_STRONG],
        {"ollama": True},
        _installed("qwen3:8b"),
        complexity=30,
        prompt="hi",
    )
    warm = router.route(
        _decision(ProviderKind.LOCAL, PrivacyMode.PREFER_LOCAL),
        _hw(),
        [LOCAL_STRONG],
        {"ollama": True},
        _installed("qwen3:8b"),
        complexity=30,
        prompt="hi",
        performance=history,
    )
    assert warm.estimated_latency_s < cold.estimated_latency_s
    assert warm.historical_performance == history["ollama/qwen3:8b"]


def test_availability_matching_ignores_latest_tag():
    router = Router()
    # Config key is untagged; Ollama reports "llama3.2:latest" — must match.
    config_model = ModelMetadata(
        id="llama3.2",
        provider_id="ollama",
        kind=ProviderKind.LOCAL,
        latency="fast",
        required_ram_gb=4,
        privacy_score=1.0,
        capabilities=ModelCapabilities(chat=0.8, reasoning=0.5, coding=0.5, writing=0.6, math=0.5),
    )
    routing = router.route(
        _decision(ProviderKind.LOCAL, PrivacyMode.PREFER_LOCAL),
        _hw(),
        [config_model],
        {"ollama": True},
        {"ollama": {"llama3.2:latest"}},
    )
    assert routing.model_id == "llama3.2"


def test_embeddings_model_matched_via_latest_tag_still_excluded():
    router = Router()
    embed = ModelMetadata(
        id="nomic-embed-text",
        provider_id="ollama",
        kind=ProviderKind.LOCAL,
        capabilities=ModelCapabilities(embeddings=1.0, chat=0.0),
    )
    routing = router.route(
        _decision(ProviderKind.LOCAL, PrivacyMode.PREFER_LOCAL),
        _hw(),
        [embed, LOCAL_STRONG],
        {"ollama": True},
        {"ollama": {"nomic-embed-text:latest", "qwen3:8b"}},
    )
    assert routing.model_id == "qwen3:8b"
    assert any(e["model"] == "nomic-embed-text" and "embeddings-only" in e["reason"] for e in routing.excluded_models)