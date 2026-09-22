"""Regression tests for Synapse One Master Model Analysis and Routing Architecture.

Covers:
- Test 1: "Generate Me a Simple Html Css Landing PAge for A Product" -> artifact_generation, creates files on disk.
- Test 2: "Explain how HTML and CSS work." -> direct_answer, no files created.
- Test 3: "Create a landing page in my existing project." -> workspace_agent, workspace_needed=True, creates files in workspace.
- Test 4: Father/Son age word problem -> standalone math, workspace_needed=False, zero workspace scan, math/chat worker.
- Test 5: Chessboard dominoes proof -> hard difficulty, capable reasoning model selected.
- Test 6: Observability timeline & timing metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from synapse.bootstrap import Boot, create_container
from synapse.config.paths import SynapsePaths
from synapse.domain import (
    Capability,
    ChatMessage,
    ChatResponse,
    Decision,
    IntentResult,
    IntentType,
    ModelDescriptor,
    ModelMetadata,
    ProviderKind,
    ProviderMetrics,
)
from synapse.domain.enums import ProviderState, RequestKind
from synapse.hardware.tier_resolver import TierResolver
from synapse.master import ExecutionMode, MasterAnalysis, ReasoningComplexity
from synapse.master.orchestrator import AIMasterOrchestrator
from synapse.providers.base import ModelProvider
from synapse.router.router import Router
from synapse.workspace.operator import FileOperator


class FakeModelProvider(ModelProvider):
    provider_id = "fake"
    kind = ProviderKind.LOCAL

    def __init__(self) -> None:
        self._models: list[str] = [
            "gemma3:1b",
            "gemma3:4b",
            "gemma3:12b",
            "qwen2.5-coder:1.5b",
            "qwen2.5-coder:7b",
            "qwen3:1.7b",
        ]
        self.calls: list[dict] = []

    def initialize(self) -> None:
        pass

    def health(self) -> bool:
        return True

    def supports(self, capability) -> bool:
        return True

    def shutdown(self) -> None:
        pass

    def list_models(self) -> list[ModelDescriptor]:
        return [ModelDescriptor(id=m, provider_id="fake") for m in self._models]

    def to_metadata(self, descriptor: ModelDescriptor) -> ModelMetadata | None:
        m = descriptor.id
        if "coder" in m:
            return ModelMetadata(
                id=m,
                provider_id="fake",
                kind=self.kind,
                privacy_score=1.0,
                capabilities={"coding": 0.95, "reasoning": 0.85, "chat": 0.6},
                strengths=["clean code output", "web applications", "html css generation"],
                required_ram_gb=4.0 if "7b" in m else 2.0,
            )
        if m == "qwen3:1.7b":
            return ModelMetadata(
                id=m,
                provider_id="fake",
                kind=self.kind,
                privacy_score=1.0,
                capabilities={"chat": 0.7, "reasoning": 0.5, "math": 0.5},
                weaknesses=["deep mathematical proofs", "complex combinatorial logic"],
                required_ram_gb=2.0,
            )
        if m == "gemma3:12b":
            return ModelMetadata(
                id=m,
                provider_id="fake",
                kind=self.kind,
                privacy_score=1.0,
                capabilities={"chat": 0.9, "reasoning": 0.95, "math": 0.9},
                strengths=["deep reasoning", "mathematical proofs", "logic puzzles"],
                required_ram_gb=8.0,
            )
        if m == "gemma3:4b":
            return ModelMetadata(
                id=m,
                provider_id="fake",
                kind=self.kind,
                privacy_score=1.0,
                capabilities={"chat": 0.85, "reasoning": 0.8, "math": 0.8},
                strengths=["math reasoning", "logic puzzles"],
                required_ram_gb=4.0,
            )
        return ModelMetadata(
            id=m,
            provider_id="fake",
            kind=self.kind,
            privacy_score=1.0,
            capabilities={"chat": 0.8, "reasoning": 0.5},
            required_ram_gb=2.0,
        )

    def chat(self, request) -> ChatResponse:
        self.calls.append({"model": request.model, "messages": request.messages})
        user_msg = next((m.content for m in request.messages if m.role == "user"), "")
        system_msg = next((m.content for m in request.messages if m.role == "system"), "")

        # 1. Master Model Analysis call
        if request.format == MasterAnalysis.model_json_schema() or "Master AI Orchestrator" in system_msg:
            analysis = AIMasterOrchestrator._fallback_analysis(user_msg, "fake model analysis")
            return ChatResponse(
                content=json.dumps(analysis.model_dump()),
                model_id=request.model or "gemma3:1b",
                provider_id="fake",
                kind=ProviderKind.LOCAL,
                metrics=ProviderMetrics(total_latency_s=0.01),
            )

        # 2. Worker code generation call with MANIFEST_INSTRUCTION
        if "FILE MANIFEST INSTRUCTION" in user_msg or "landing page" in user_msg.lower():
            content = json.dumps({
                "files": [
                    {
                        "path": "index.html",
                        "content": "<!DOCTYPE html>\n<html>\n<head><link rel='stylesheet' href='style.css'></head>\n<body><h1>Product Landing Page</h1></body>\n</html>",
                    },
                    {
                        "path": "style.css",
                        "content": "body { font-family: sans-serif; background: #f0f0f0; }\nh1 { color: #333; }",
                    },
                ]
            })
            return ChatResponse(
                content=content,
                model_id=request.model or "qwen2.5-coder:7b",
                provider_id="fake",
                kind=ProviderKind.LOCAL,
                metrics=ProviderMetrics(total_latency_s=0.05),
            )

        # 3. Regular chat / explanation / math solution
        return ChatResponse(
            content="Here is the detailed solution and proof.",
            model_id=request.model or "gemma3:4b",
            provider_id="fake",
            kind=ProviderKind.LOCAL,
            metrics=ProviderMetrics(total_latency_s=0.02),
        )


@pytest.fixture
def test_env(temp_paths: SynapsePaths, monkeypatch):
    monkeypatch.setenv("SYNAPSE_HOME", str(temp_paths.home))
    boot = Boot(create_container(paths=temp_paths))
    fake = FakeModelProvider()
    boot.providers._providers["fake"] = fake
    boot.providers._states["fake"] = ProviderState.READY

    from synapse.domain.hardware import CpuInfo, GpuInfo, HardwareProfile, HardwareRecommendations, MemoryInfo
    monkeypatch.setattr(
        boot.hardware,
        "scan",
        lambda: HardwareProfile(
            cpu=CpuInfo(cores=8, threads=16, model="Test CPU"),
            memory=MemoryInfo(total_gb=32.0, available_gb=16.0, used_percent=50.0),
            gpu=GpuInfo(available=True, name="Test GPU", vram_gb=12.0),
            recommendations=HardwareRecommendations(can_run_local_llm=True, max_parameters_b=14.0),
        ),
    )

    models = [
        ("gemma3:1b", {"chat": 0.8, "reasoning": 0.5}, [], [], 2.0),
        ("gemma3:4b", {"chat": 0.85, "reasoning": 0.8, "math": 0.8}, ["math reasoning", "logic puzzles"], [], 4.0),
        ("gemma3:12b", {"chat": 0.9, "reasoning": 0.95, "math": 0.9}, ["deep reasoning", "mathematical proofs", "logic puzzles"], [], 8.0),
        ("qwen2.5-coder:1.5b", {"coding": 0.9, "reasoning": 0.7, "chat": 0.6}, ["clean code output"], [], 2.0),
        ("qwen2.5-coder:7b", {"coding": 0.95, "reasoning": 0.85, "chat": 0.6}, ["clean code output", "web applications", "html css generation"], [], 4.0),
        ("qwen3:1.7b", {"chat": 0.7, "reasoning": 0.5, "math": 0.5}, [], ["deep mathematical proofs", "complex combinatorial logic"], 2.0),
    ]
    for mid, caps, strengths, weaknesses, ram in models:
        boot.registry._models[mid] = ModelMetadata(
            id=mid,
            provider_id="fake",
            kind=ProviderKind.LOCAL,
            privacy_score=1.0,
            capabilities=caps,
            strengths=strengths,
            weaknesses=weaknesses,
            required_ram_gb=ram,
        )

    workspace_dir = temp_paths.home / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    file_op = FileOperator(root=workspace_dir)
    return boot, fake, workspace_dir, file_op


def test_test1_generate_landing_page_creates_files_on_disk(test_env):
    boot, fake, ws_dir, file_op = test_env
    prompt = "Generate Me a Simple Html Css Landing PAge for A Product"

    response = boot.master.process(prompt, file_operator=file_op)

    # 1. Master Analysis verification
    trace = response.decision_trace
    assert trace is not None
    assert trace.master_analysis is not None
    ma = trace.master_analysis
    assert ma["intent"] == "artifact_generation"
    assert ma["artifact_required"] is True
    assert ma["files_needed"] is True
    assert ma["coding_needed"] is True
    assert ma["domain"] == "web_development"
    assert ma["execution_mode"] == "artifact_generation"

    # 2. Worker model routing (must route to coding specialist)
    assert "coder" in response.model.lower()

    # 3. Physical file creation on disk
    index_file = ws_dir / "index.html"
    css_file = ws_dir / "style.css"
    assert index_file.exists(), "index.html must be created on disk"
    assert css_file.exists(), "style.css must be created on disk"
    assert "Product Landing Page" in index_file.read_text(encoding="utf-8")
    assert "font-family" in css_file.read_text(encoding="utf-8")

    # 4. Actions reported in response
    created_paths = [a.path for a in response.actions if a.status == "ok"]
    assert "index.html" in created_paths
    assert "style.css" in created_paths

    # 5. Observability timing metrics
    assert "master_analysis_ms" in trace.timings
    assert "worker_inference_ms" in trace.timings
    assert "tools_ms" in trace.timings
    assert "total_ms" in trace.timings
    assert trace.timings["total_ms"] > 0


def test_test2_explain_html_css_direct_answer(test_env):
    boot, fake, ws_dir, file_op = test_env
    prompt = "Explain how HTML and CSS work."

    response = boot.master.process(prompt, file_operator=file_op)

    trace = response.decision_trace
    assert trace is not None
    ma = trace.master_analysis
    assert ma is not None
    assert ma["intent"] == "question_answering"
    assert ma["artifact_required"] is False
    assert ma["files_needed"] is False
    assert ma["workspace_needed"] is False
    assert ma["execution_mode"] == "direct_answer"

    # No files created
    assert len(list(ws_dir.iterdir())) == 0
    assert len(response.actions) == 0


def test_test3_create_landing_page_in_existing_project(test_env):
    boot, fake, ws_dir, file_op = test_env
    (ws_dir / "README.md").write_text("# Existing Project", encoding="utf-8")
    prompt = "Create a landing page in my existing project."

    response = boot.master.process(prompt, file_operator=file_op)

    trace = response.decision_trace
    assert trace is not None
    ma = trace.master_analysis
    assert ma is not None
    assert ma["intent"] == "artifact_generation"
    assert ma["artifact_required"] is True
    assert ma["workspace_needed"] is True
    assert ma["files_needed"] is True
    assert ma["execution_mode"] == "workspace_agent"

    assert (ws_dir / "index.html").exists()


def test_test4_father_son_math_puzzle_workspace_opt_in(test_env):
    boot, fake, ws_dir, file_op = test_env
    # Workspace has files, but request is standalone math
    (ws_dir / "app.py").write_text("print('hello')", encoding="utf-8")
    prompt = "Father is 4 times older than son. In 20 years, he will be 2 times older. How old are they?"

    response = boot.master.process(prompt, file_operator=file_op)

    trace = response.decision_trace
    assert trace is not None
    ma = trace.master_analysis
    assert ma is not None
    assert ma["domain"] == "mathematics"
    assert ma["workspace_needed"] is False
    assert ma["artifact_required"] is False
    assert ma["files_needed"] is False
    assert ma["coding_needed"] is False
    assert trace.timings["context_retrieval_ms"] == 0.0

    # Routed to general/math model, not coder
    assert "coder" not in response.model.lower()


def test_test5_chessboard_dominoes_hard_reasoning(test_env):
    boot, fake, ws_dir, file_op = test_env
    prompt = "Can you cover a standard 8x8 chessboard with two opposite corners removed using 31 2x1 dominoes? Prove your answer."

    analysis = AIMasterOrchestrator._fallback_analysis(prompt, "test")
    assert analysis.reasoning_complexity == ReasoningComplexity.HARD
    assert analysis.domain == "mathematics"
    assert analysis.workspace_needed is False
    assert analysis.artifact_required is False

    response = boot.master.process(prompt, file_operator=file_op)
    trace = response.decision_trace
    assert trace.reasoning_complexity == "hard"
    # Router must NOT select weak model qwen3:1.7b (which has weakness 'deep mathematical proofs')
    assert response.model != "qwen3:1.7b"
    assert response.model in ("gemma3:4b", "gemma3:12b")
