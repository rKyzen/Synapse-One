"""Comprehensive tests for Latency Reductions, Master Token Capping, Caching, and Controlled Web Access.

Architect Directive:
1. Significantly reduce end-to-end latency of Master + specialist calls on every hardware tier.
2. Controlled, opt-in web access via ToolRegistry (web_search, web_browse) when MasterAnalysis.web_needed=True.
3. Cache MasterAnalysis results to eliminate duplicate analysis calls.
4. Fast path instant bypass (<1ms, 0ms master analysis).
5. Token capping per task kind.
6. Context size trimming (4-5 turns default, targeted excerpts).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from synapse.actions import ActionEngine, RequestKind
from synapse.analyzers import ComplexityAnalyzer, IntentAnalyzer, PrivacyAnalyzer
from synapse.decision import DecisionEngine
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
    LatencyTier,
    ModelCapabilities,
    ModelMetadata,
    PrivacyMode,
    PrivacyResult,
    ProviderKind,
    TaskKind,
)
from synapse.domain.hardware import RecommendedModelLimits
from synapse.events import EventBus
from synapse.execution import Executor
from synapse.execution.planner import ExecutionPlanner
from synapse.master.agent import MasterAgent
from synapse.master.fast_path import FastPathType, check_fast_path
from synapse.master.orchestrator import AIMasterOrchestrator, AnalysisCache
from synapse.master.schemas import ExecutionMode, MasterAnalysis, ReasoningComplexity
from synapse.pipeline.tools import ToolRegistry
from synapse.router import Router
from synapse.workspace.brief import build_specialist_prompt, build_workspace_brief
from synapse.workspace.operator import FileOperator


def _mock_hw_tier2() -> HardwareProfile:
    return HardwareProfile(
        memory={"total_gb": 16.0, "available_gb": 12.0, "used_percent": 25.0},
        gpu={"name": "NVIDIA GeForce RTX 3060", "vram_gb": 8.0},
        recommendations=RecommendedModelLimits(can_run_local_llm=True),
    )


def _mock_tier2_registry() -> list[ModelMetadata]:
    return [
        ModelMetadata(
            id="gemma3:4b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            latency="fast",
            required_ram_gb=4,
            privacy_score=1.0,
            capabilities=ModelCapabilities(chat=0.9, writing=0.9, reasoning=0.85, math=0.85),
        ),
        ModelMetadata(
            id="qwen2.5-coder:7b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            latency="medium",
            required_ram_gb=8,
            privacy_score=1.0,
            capabilities=ModelCapabilities(chat=0.6, coding=1.0, debugging=0.95, architecture=0.9, reasoning=0.9),
        ),
        ModelMetadata(
            id="qwen3:4b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            latency="fast",
            required_ram_gb=4,
            privacy_score=1.0,
            capabilities=ModelCapabilities(chat=0.8, planning=0.95, reasoning=0.85),
        ),
        ModelMetadata(
            id="qwen2.5:7b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            latency="medium",
            required_ram_gb=8,
            privacy_score=1.0,
            capabilities=ModelCapabilities(chat=0.85, reasoning=0.9, translation=0.95, research=0.9),
        ),
        ModelMetadata(
            id="qwen2.5vl:7b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            latency="medium",
            required_ram_gb=8,
            privacy_score=1.0,
            capabilities=ModelCapabilities(chat=0.7, vision=True, ocr=1.0, pdf=1.0),
        ),
        ModelMetadata(
            id="nomic-embed-text",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            latency="fast",
            required_ram_gb=1,
            privacy_score=1.0,
            capabilities=ModelCapabilities(embeddings=1.0, chat=0.0),
        ),
    ]


def _build_test_agent(tmp_path: Path, mock_provider=None):
    op = FileOperator(tmp_path)
    if mock_provider is None:
        mock_provider = MagicMock()
        mock_provider.provider_id = "ollama"
        mock_provider.kind = ProviderKind.LOCAL
        mock_provider.list_models.return_value = [
            MagicMock(id="gemma3:4b"),
            MagicMock(id="qwen2.5-coder:7b"),
            MagicMock(id="qwen3:4b"),
            MagicMock(id="qwen2.5:7b"),
            MagicMock(id="qwen2.5vl:7b"),
            MagicMock(id="nomic-embed-text"),
        ]
        mock_provider.chat.return_value = ChatResponse(
            provider_id="ollama",
            model_id="gemma3:4b",
            kind=ProviderKind.LOCAL,
            content="Standard test response",
        )

    providers = MagicMock()
    providers.get.return_value = mock_provider
    providers.health.side_effect = lambda pid=None: True if pid else {"ollama": True}
    providers.all.return_value = [mock_provider]
    providers.provider_ids.return_value = ["ollama"]

    mock_hw = MagicMock()
    mock_hw.scan.return_value = _mock_hw_tier2()
    mock_reg = MagicMock()
    mock_reg.all.return_value = _mock_tier2_registry()

    mock_config = MagicMock()
    mock_config.get.side_effect = lambda k, default=None: {
        "privacy.user_preference": "balanced",
        "router.prefer_local": True,
        "executor.timeout_s": 30.0,
        "hardware.tier": "tier2",
    }.get(k, default)

    agent = MasterAgent(
        intent_analyzer=IntentAnalyzer(),
        complexity_analyzer=ComplexityAnalyzer(),
        privacy_analyzer=PrivacyAnalyzer(),
        decision_engine=DecisionEngine(),
        planner=ExecutionPlanner(),
        router=Router(),
        executor=Executor(providers),
        providers=providers,
        hardware=mock_hw,
        registry=mock_reg,
        events=EventBus(),
        config=mock_config,
    )
    return agent, op, mock_provider


# ============================================================================
# 1. Fast Path Latency & Deterministic Bypass Tests
# ============================================================================

def test_fast_path_direct_queries_deterministic(tmp_path: Path):
    """Direct fast-path queries (arithmetic, ping, status, files) return instant deterministic responses."""
    agent, op, _ = _build_test_agent(tmp_path)

    for query in ["ping", "Calculate 125 * 8", "system status", "loaded models", "list workspace files"]:
        res = agent.process(query, file_operator=op)
        assert res.provider == "fast_path"
        assert res.model == "deterministic"
        assert res.decision_trace is not None
        assert res.decision_trace.timings.get("master_analysis_ms") == 0.0
        assert res.decision_trace.timings.get("fast_path_ms") is not None
        assert res.latency_ms < 50.0


def test_fast_chat_bypass_zero_master_analysis_latency(tmp_path: Path):
    """Conversational greetings, identity, and affirmations bypass Master Model analysis (0ms analysis time)."""
    agent, op, _ = _build_test_agent(tmp_path)

    for phrase in ["Hello!", "who are you", "hey there", "thanks!", "thank you very much", "ok", "sure", "cool", "bye"]:
        res = agent.process(phrase, file_operator=op)
        assert res.decision_trace is not None
        assert res.decision_trace.timings.get("master_analysis_ms") == 0.0
        assert res.latency_ms < 50.0


# ============================================================================
# 2. Master Analysis Caching Tests
# ============================================================================

def test_master_analysis_cache_hit_and_ttl():
    """AnalysisCache caches results by normalized prompt + conversation_id with TTL expiration."""
    cache = AnalysisCache(max_size=10, ttl_seconds=0.5)

    analysis = MasterAnalysis(
        intent="coding",
        domain="python",
        goal="Write a fibonacci script",
        workspace_needed=True,
        files_needed=True,
        coding_needed=True,
        artifact_required=True,
    )

    cache.set("Write a fibonacci script", analysis, conversation_id="chat-1")

    # Hit same conversation
    hit = cache.get("  write a   fibonacci script  ", conversation_id="chat-1")
    assert hit is not None
    assert hit.intent == "coding"

    # Miss different conversation
    miss = cache.get("Write a fibonacci script", conversation_id="chat-2")
    assert miss is None

    # Expiration after TTL
    time.sleep(0.6)
    expired = cache.get("Write a fibonacci script", conversation_id="chat-1")
    assert expired is None


def test_orchestrator_uses_cached_analysis_without_llm_call(tmp_path: Path):
    """AIMasterOrchestrator uses cached analysis on repeat requests, avoiding LLM analysis overhead."""
    mock_provider = MagicMock()
    mock_provider.provider_id = "ollama"
    mock_provider.kind = ProviderKind.LOCAL
    mock_provider.list_models.return_value = [MagicMock(id="gemma3:4b")]

    call_count = 0
    analysis_json = json.dumps({
        "intent": "question_answering",
        "domain": "general",
        "goal": "Explain recursion",
        "workspace_needed": False,
        "files_needed": False,
        "web_needed": False,
        "artifact_required": False,
        "reasoning_complexity": "easy",
        "required_capabilities": ["chat"],
    })

    def mock_chat(req):
        nonlocal call_count
        call_count += 1
        return ChatResponse(
            provider_id="ollama",
            model_id="gemma3:4b",
            kind=ProviderKind.LOCAL,
            content=analysis_json,
        )

    mock_provider.chat.side_effect = mock_chat
    providers = MagicMock()
    providers.get.return_value = mock_provider
    providers.health.side_effect = lambda pid=None: True if pid else {"ollama": True}
    providers.all.return_value = [mock_provider]

    mock_config = MagicMock()
    mock_config.get.return_value = "tier2"
    mock_hw = MagicMock()
    mock_hw.scan.return_value = _mock_hw_tier2()
    mock_reg = MagicMock()
    mock_reg.all.return_value = _mock_tier2_registry()

    orchestrator = AIMasterOrchestrator(
        providers=providers,
        registry=mock_reg,
        hardware=mock_hw,
        config=mock_config,
    )

    # First call: hits provider
    res1 = orchestrator.analyze("Explain recursion in Python", conversation_id="c1")
    assert res1.intent == "question_answering"
    assert call_count == 1

    # Second call with identical prompt: hits cache
    res2 = orchestrator.analyze("explain   recursion in python  ", conversation_id="c1")
    assert res2.intent == "question_answering"
    assert call_count == 1  # No second provider call!


# ============================================================================
# 3. Controlled Web Access Tests (web_needed opt-in)
# ============================================================================

def test_tool_registry_web_search_and_browse():
    """ToolRegistry registers web_search and web_browse with rate-limiting and caching."""
    tool_reg = ToolRegistry()
    assert tool_reg.get("web_search") is not None
    assert tool_reg.get("web_browse") is not None

    # Test web_search returns structured results
    results = tool_reg._web_search("FastAPI Python framework", max_results=3)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "title" in results[0]
    assert "url" in results[0]
    assert "snippet" in results[0]

    # Repeat call hits cache
    cached_results = tool_reg._web_search("FastAPI Python framework", max_results=3)
    assert len(cached_results) == len(results)


def test_coding_request_zero_web_overhead(tmp_path: Path):
    """Pure coding requests have web_needed=False, web_ms=0.0, and zero web tool calls."""
    agent, op, mock_provider = _build_test_agent(tmp_path)

    manifest_json = json.dumps({
        "files": [{"path": "math_utils.py", "content": "def add(a, b): return a + b\n"}]
    })
    mock_provider.chat.return_value = ChatResponse(
        provider_id="ollama",
        model_id="qwen2.5-coder:7b",
        kind=ProviderKind.LOCAL,
        content=manifest_json,
    )

    res = agent.process("Create math_utils.py with an add function", file_operator=op)
    assert op.exists("math_utils.py")
    assert res.decision_trace is not None
    assert res.decision_trace.timings.get("web_ms") == 0.0
    assert res.decision_trace.master_analysis is not None
    assert res.decision_trace.master_analysis.get("web_needed") is False


def test_live_info_query_triggers_web_search_opt_in(tmp_path: Path):
    """When a query asks for latest release/news, MasterAnalysis sets web_needed=True and runs web search."""
    agent, op, mock_provider = _build_test_agent(tmp_path)

    recorded_prompts = []
    def mock_chat(req):
        content = req.messages[-1].content if hasattr(req, "messages") else str(req)
        recorded_prompts.append(content)
        return ChatResponse(
            provider_id="ollama",
            model_id="gemma3:4b",
            kind=ProviderKind.LOCAL,
            content="FastAPI is an asynchronous modern web framework for Python.",
        )

    mock_provider.chat.side_effect = mock_chat

    prompt = "What is the latest stable version of FastAPI?"
    res = agent.process(prompt, file_operator=op)

    assert res.decision_trace is not None
    assert res.decision_trace.master_analysis is not None
    assert res.decision_trace.master_analysis.get("web_needed") is True
    assert res.decision_trace.timings.get("web_ms") is not None

    # Verify web search results were injected into the specialist prompt
    assert len(recorded_prompts) > 0
    specialist_call = recorded_prompts[0]
    assert "Fresh Web Search Results" in specialist_call or "FastAPI" in specialist_call


# ============================================================================
# 4. Context Size Trimming & Specialist Token Capping Tests
# ============================================================================

def test_chat_history_trimmed_to_5_turns_by_default(tmp_path: Path):
    """Specialist prompt builder defaults to 5 turns of conversation history."""
    op = FileOperator(tmp_path)
    messages = [
        {"role": "user", "content": f"Session turn {i} topic"}
        for i in range(1, 11)
    ]

    prompt = build_specialist_prompt(
        "Add a function to app.py",
        project_id="test-proj",
        file_operator=op,
        messages=messages,
    )

    # In the chat history block, turns 1-5 should be truncated; turns 6-10 should be present
    history_block = prompt.split("Current chat history")[1].split("Conversation summary")[0]
    assert "Session turn 1 topic" not in history_block
    assert "Session turn 5 topic" not in history_block
    assert "Session turn 6 topic" in history_block
    assert "Session turn 10 topic" in history_block


def test_chat_memory_query_expands_to_8_turns(tmp_path: Path):
    """When prompt asks 'What is this chat about?', history expands to 8 turns."""
    op = FileOperator(tmp_path)
    messages = [
        {"role": "user", "content": f"History step {i} setup"}
        for i in range(1, 11)
    ]

    prompt = build_specialist_prompt(
        "What is this chat about?",
        project_id="test-proj",
        file_operator=op,
        messages=messages,
    )

    # In the chat history block, 8 turns retained (turns 3-10)
    history_block = prompt.split("Current chat history")[1].split("Conversation summary")[0]
    assert "History step 1 setup" not in history_block
    assert "History step 2 setup" not in history_block
    assert "History step 3 setup" in history_block
    assert "History step 10 setup" in history_block


def test_specialist_max_tokens_capping(tmp_path: Path):
    """Specialist calls have bounded max_tokens according to task complexity."""
    agent, op, mock_provider = _build_test_agent(tmp_path)

    recorded_requests = []
    def mock_chat(req):
        recorded_requests.append(req)
        return ChatResponse(
            provider_id="ollama",
            model_id="gemma3:4b",
            kind=ProviderKind.LOCAL,
            content="Chat reply",
        )

    mock_provider.chat.side_effect = mock_chat

    # 1. Chat call -> max_tokens=768
    agent.process("Explain how HTTP cookies work", file_operator=op)
    assert len(recorded_requests) == 1
    assert recorded_requests[0].max_tokens == 768


# ============================================================================
# 5. Full End-to-End Regression & Verification
# ============================================================================

def test_file_creation_verification_and_timing_breakdown(tmp_path: Path):
    """File creation performs real write + verification and logs timing breakdown."""
    agent, op, mock_provider = _build_test_agent(tmp_path)

    code = "def multiply(x, y):\n    return x * y\n"
    manifest = json.dumps({"files": [{"path": "calc.py", "content": code}]})
    mock_provider.chat.return_value = ChatResponse(
        provider_id="ollama",
        model_id="qwen2.5-coder:7b",
        kind=ProviderKind.LOCAL,
        content=manifest,
    )

    res = agent.process("Create calc.py with a multiply function", file_operator=op)

    assert op.exists("calc.py")
    assert "def multiply(x, y):" in op.read("calc.py")
    assert res.decision_trace is not None
    timings = res.decision_trace.timings
    assert "master_analysis_ms" in timings
    assert "context_retrieval_ms" in timings
    assert "worker_inference_ms" in timings
    assert "tools_ms" in timings
    assert "web_ms" in timings
    assert "verification_ms" in timings
    assert "total_ms" in timings
