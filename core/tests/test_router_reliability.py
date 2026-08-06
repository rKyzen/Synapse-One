"""Router Phase 3 reliability tests — historical performance demotes flaky models."""

from __future__ import annotations

import pytest

from synapse.domain import (
    Capability,
    Decision,
    HardwareProfile,
    PrivacyMode,
    ProviderKind,
)
from synapse.domain.hardware import RecommendedModelLimits
from synapse.domain.models import ModelCapabilities, ModelMetadata
from synapse.router import Router

#: Two identical models so only reliability can break the tie.
MODEL_A = ModelMetadata(
    id="model-a",
    provider_id="ollama",
    kind=ProviderKind.LOCAL,
    latency="fast",
    required_ram_gb=4,
    privacy_score=1.0,
    estimated_cost_per_1k=0.0,
    capabilities=ModelCapabilities(reasoning=0.8, coding=0.7, writing=0.6),
)
MODEL_B = ModelMetadata(
    id="model-b",
    provider_id="ollama",
    kind=ProviderKind.LOCAL,
    latency="fast",
    required_ram_gb=4,
    privacy_score=1.0,
    estimated_cost_per_1k=0.0,
    capabilities=ModelCapabilities(reasoning=0.8, coding=0.7, writing=0.6),
)


def _decision() -> Decision:
    return Decision(
        can_stay_local=True,
        internet_required=False,
        use_cloud_reasoning=False,
        preferred_kind=ProviderKind.LOCAL,
        required_capabilities=[Capability.CODING],
        preferred_capabilities=[],
        privacy=PrivacyMode.PREFER_LOCAL,
    )


def _hw() -> HardwareProfile:
    return HardwareProfile(
        memory={"total_gb": 16, "available_gb": 8, "used_percent": 50.0},
        recommendations=RecommendedModelLimits(can_run_local_llm=True),
    )


def _history(samples: int, success_rate: float, failures: float = 0.0) -> dict:
    return {
        "samples": samples,
        "success_rate": success_rate,
        "timeout_rate": 0.0,
        "failure_rate": failures,
    }


def test_reliability_untouched_below_min_samples():
    router = Router()
    score = router._reliability_adjust(
        10.0, MODEL_A, {"ollama/model-a": _history(2, 0.0, failures=1.0)}
    )
    assert score == 10.0


def test_reliability_untouched_without_history():
    assert Router()._reliability_adjust(10.0, MODEL_A, None) == 10.0


def test_reliable_model_keeps_score():
    router = Router()
    score = router._reliability_adjust(
        10.0, MODEL_A, {"ollama/model-a": _history(10, 1.0)}
    )
    assert score == pytest.approx(10.0, abs=0.001)


def test_flaky_model_is_demoted():
    router = Router()
    score = router._reliability_adjust(
        10.0, MODEL_A, {"ollama/model-a": _history(10, 0.5)}
    )
    assert score < 10.0


def test_failures_add_extra_penalty():
    router = Router()
    score = router._reliability_adjust(
        10.0, MODEL_A, {"ollama/model-a": _history(10, 0.8, failures=0.2)}
    )
    assert score < router._reliability_adjust(
        10.0, MODEL_A, {"ollama/model-a": _history(10, 0.8)}
    )


def test_route_prefers_reliable_over_flaky_tie():
    router = Router()
    performance = {
        "ollama/model-a": _history(10, 1.0),
        "ollama/model-b": _history(10, 0.4),
    }
    routing = router.route(
        _decision(),
        _hw(),
        [MODEL_A, MODEL_B],
        {"ollama": True},
        {"ollama": {"model-a", "model-b"}},
        performance=performance,
    )
    assert routing.model_id == "model-a"


def test_route_includes_reliability_in_candidates_view():
    router = Router()
    performance = {"ollama/model-a": _history(10, 0.9)}
    routing = router.route(
        _decision(),
        _hw(),
        [MODEL_A],
        {"ollama": True},
        {"ollama": {"model-a"}},
        performance=performance,
    )
    assert routing.candidates
    assert routing.candidates[0]["reliability"]["samples"] == 10
    assert routing.candidates[0]["reliability"]["success_rate"] == 0.9


def test_reliability_note_in_reason():
    router = Router()
    performance = {"ollama/model-a": _history(10, 0.9)}
    routing = router.route(
        _decision(),
        _hw(),
        [MODEL_A],
        {"ollama": True},
        {"ollama": {"model-a"}},
        performance=performance,
    )
    assert "reliability 90%" in routing.reason
    assert "10 samples" in routing.reason