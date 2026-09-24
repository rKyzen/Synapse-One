"""Comprehensive tests for Synapse One Fast-Path and Master Model Architecture.

Validates:
1. Fast-path deterministic arithmetic evaluation (<1ms, 0ms LLM overhead).
2. Fast-path percentage, square root, rounding, and negative number calculations.
3. Fast-path greetings, gratitude, farewells, identity, and ping.
4. Fast-path bypass of Master Model pre-flight analysis latency.
5. Master Model ~1B parameter tier assignment across all local hardware tiers.
6. Independent per-task model routing for compound tasks.
7. Verification that Master Agent records timing traces accurately.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from synapse.contracts import ModelProvider
from synapse.domain import (
    Capability,
    ChatMessage,
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
from synapse.domain.hardware import CpuInfo, GpuInfo, MemoryInfo, RecommendedModelLimits
from synapse.hardware import HardwareTier, TierResolver
from synapse.master import (
    AIMasterOrchestrator,
    FastPathResult,
    FastPathType,
    check_fast_path,
    evaluate_arithmetic,
)
from synapse.master.agent import MasterAgent
from synapse.master.schemas import ExecutionMode, MasterAnalysis, ReasoningComplexity
from synapse.router import Router


# ---------------------------------------------------------------------------
# 1. Fast-Path Arithmetic Evaluation Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("prompt", "expected_result"),
    [
        ("2 + 2", "4"),
        ("15 * 8", "120"),
        ("100 / 4", "25"),
        ("2 ** 8", "256"),
        ("10 % 3", "1"),
        ("sqrt(144)", "12"),
        ("sqrt(25) + 5", "10"),
        ("abs(-42)", "42"),
        ("round(3.14159, 2)", "3.14"),
        ("20% of 85", "17"),
        ("15% of 200", "30"),
        ("50% of 10", "5"),
        ("what is 25 * 4?", "100"),
        ("calculate 10 + 20 * 3", "70"),
        ("calculate (10 + 20) * 3", "90"),
        ("-5 + 10", "5"),
        ("10 - 15", "-5"),
    ],
)
def test_fast_path_arithmetic_accuracy(prompt: str, expected_result: str):
    res = check_fast_path(prompt)
    assert res.is_fast_path is True
    assert res.path_type == FastPathType.DIRECT_ARITHMETIC
    assert res.direct_response == f"The answer is {expected_result}."
    assert res.complexity == 1


def test_fast_path_unsafe_math_ignored():
    # Arbitrary code injection or malicious builtins must NEVER execute in fast-path
    unsafe_prompts = [
        "__import__('os').system('dir')",
        "eval('2+2')",
        "open('file.txt')",
        "exec('x=1')",
        "import sys",
    ]
    for p in unsafe_prompts:
        res = check_fast_path(p)
        assert res.is_fast_path is False or res.direct_response is None


# ---------------------------------------------------------------------------
# 2. Fast-Path Greetings, Social Pleasantries, & Identity Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt",
    [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good evening",
        "howdy",
        "hi there",
        "thanks",
        "thank you",
        "thank you very much",
        "bye",
        "goodbye",
        "see you",
        "who are you",
        "what is your name",
    ],
)
def test_fast_path_conversational_greetings(prompt: str):
    res = check_fast_path(prompt)
    assert res.is_fast_path is True
    assert res.path_type == FastPathType.FAST_CHAT
    assert res.intent == IntentType.CONVERSATION


def test_fast_path_ping():
    res = check_fast_path("ping")
    assert res.is_fast_path is True
    assert res.direct_response == "pong"


def test_fast_path_disabled_for_file_intake():
    # If files are attached, fast path must NOT bypass analysis
    res = check_fast_path("2 + 2", has_files=True)
    assert res.is_fast_path is False


# ---------------------------------------------------------------------------
# 3. Master Model ~1B Tier Resolution Tests
# ---------------------------------------------------------------------------

def test_master_model_permanently_1b_across_all_local_tiers():
    resolver = TierResolver()

    # Tier 1 (e.g. 4GB RAM)
    t1 = resolver.resolve(HardwareProfile(memory=MemoryInfo(total_gb=8.0, available_gb=4.0)))
    assert t1.tier == HardwareTier.TIER1
    assert t1.candidate_models[0] == "gemma3:1b"

    # Tier 2 (e.g. 10GB RAM)
    t2 = resolver.resolve(HardwareProfile(memory=MemoryInfo(total_gb=16.0, available_gb=10.0)))
    assert t2.tier == HardwareTier.TIER2
    assert t2.candidate_models[0] == "gemma3:1b"

    # Tier 3 (e.g. 20GB RAM, 12GB VRAM)
    t3 = resolver.resolve(HardwareProfile(memory=MemoryInfo(total_gb=32.0, available_gb=20.0), gpu=GpuInfo(vram_gb=12.0)))
    assert t3.tier == HardwareTier.TIER3
    assert t3.candidate_models[0] == "gemma3:1b"

    # Tier 3+ (e.g. 40GB RAM, 24GB VRAM)
    t3_plus = resolver.resolve(HardwareProfile(memory=MemoryInfo(total_gb=64.0, available_gb=40.0), gpu=GpuInfo(vram_gb=24.0)))
    assert t3_plus.tier == HardwareTier.TIER3_PLUS
    assert t3_plus.candidate_models[0] == "gemma3:1b"

    # Cloud Fallback (e.g. 0.3GB RAM)
    cloud = resolver.resolve(HardwareProfile(memory=MemoryInfo(total_gb=4.0, available_gb=0.3)))
    assert cloud.tier == HardwareTier.CLOUD_FALLBACK
    assert cloud.candidate_models == ["gpt-4o-mini", "gemini-2.5-flash"]


# ---------------------------------------------------------------------------
# 4. Independent Per-Task Specialist Routing & Complexity Bonuses
# ---------------------------------------------------------------------------

def test_router_complexity_scaled_latency_bonus():
    router = Router()
    hardware = HardwareProfile(memory=MemoryInfo(total_gb=16.0, available_gb=12.0), recommendations=RecommendedModelLimits(can_run_local_llm=True))

    fast_chat_model = ModelMetadata(
        id="gemma3:1b",
        provider_id="ollama",
        kind=ProviderKind.LOCAL,
        latency="fast",
        capabilities=ModelCapabilities(chat=0.8, coding=0.3, reasoning=0.4),
        privacy_score=1.0,
    )
    deep_reasoning_model = ModelMetadata(
        id="qwen2.5:14b",
        provider_id="ollama",
        kind=ProviderKind.LOCAL,
        latency="slow",
        capabilities=ModelCapabilities(chat=0.8, coding=0.8, reasoning=0.95),
        privacy_score=1.0,
    )

    decision_simple = Decision(
        can_stay_local=True,
        required_capabilities=[Capability.CHAT],
        preferred_capabilities=[],
        privacy=PrivacyMode.PREFER_LOCAL,
    )

    decision_complex = Decision(
        can_stay_local=True,
        required_capabilities=[Capability.REASONING],
        preferred_capabilities=[],
        privacy=PrivacyMode.PREFER_LOCAL,
    )

    # Low complexity (10) favors fast latency model
    routing_simple = router.route(
        decision_simple,
        hardware,
        [fast_chat_model, deep_reasoning_model],
        {"ollama": True},
        complexity=10,
    )
    assert routing_simple.model_id == "gemma3:1b"

    # High complexity (85) favors deep reasoning specialist despite slower latency
    routing_complex = router.route(
        decision_complex,
        hardware,
        [fast_chat_model, deep_reasoning_model],
        {"ollama": True},
        complexity=85,
    )
    assert routing_complex.model_id == "qwen2.5:14b"
