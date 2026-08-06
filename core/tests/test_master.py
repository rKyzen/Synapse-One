"""Master Agent tests — end-to-end pipeline with a mock provider."""

from __future__ import annotations

import json

import pytest

from synapse.bootstrap import Boot, create_container
from synapse.config.paths import SynapsePaths
from synapse.contracts import ModelProvider
from synapse.domain import Capability, ChatRequest, ChatResponse, ModelDescriptor, ModelMetadata, ProviderKind
from synapse.domain.enums import TaskKind


class MockProvider(ModelProvider):
    """Fake provider: no network, deterministic replies."""

    provider_id = "mock"
    kind = ProviderKind.LOCAL

    def __init__(self) -> None:
        self.calls: list[str] = []

    def initialize(self) -> None:
        pass

    def list_models(self) -> list[ModelDescriptor]:
        return [ModelDescriptor(id="mock-1", provider_id="mock")]

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request.messages[-1].content)
        return ChatResponse(
            provider_id="mock",
            model_id="mock-1",
            kind=self.kind,
            content="mock reply",
            raw={},
        )

    def health(self) -> bool:
        return True

    def supports(self, capability: Capability) -> bool:
        return True

    def shutdown(self) -> None:
        pass

    def to_metadata(self, descriptor: ModelDescriptor) -> ModelMetadata | None:
        return None


@pytest.fixture
def mock_boot(temp_paths: SynapsePaths, monkeypatch) -> tuple[Boot, MockProvider]:
    """A boot whose ProviderManager holds the mock provider, with a registry
    entry so routing can pick it."""
    monkeypatch.setenv("SYNAPSE_HOME", str(temp_paths.home))
    boot = Boot(create_container(paths=temp_paths))

    # Register mock into the manager directly (bypasses factory).
    manager = boot.providers
    mock = MockProvider()
    manager._providers["mock"] = mock
    manager._states["mock"] = __import__("synapse.domain", fromlist=["ProviderState"]).ProviderState.READY

    # Inject mock model into registry.
    registry = boot.registry
    registry._models["mock-1"] = ModelMetadata(
        id="mock-1",
        provider_id="mock",
        kind=ProviderKind.LOCAL,
        privacy_score=1.0,
        capabilities={"reasoning": 0.9, "coding": 0.9, "writing": 0.9, "math": 0.9},
    )
    return boot, mock


def test_master_process_returns_normalized_response(mock_boot):
    boot, mock = mock_boot
    response = boot.master.process("Build a python script to fix this bug")
    assert response.response == "mock reply"
    assert response.intent.value == "coding"
    assert response.provider == "mock"
    assert response.model == "mock-1"
    assert 0 <= response.complexity <= 100
    assert response.latency_ms >= 0
    assert response.execution_plan.steps
    assert response.decision_trace is not None
    assert mock.calls  # provider was actually invoked


def test_master_no_route_returns_guidance(mock_boot):
    boot, _ = mock_boot
    # Remove the local mock model; registry holds only models from providers
    # that are NOT registered/healthy (ollama from default config).
    boot.registry._models.pop("mock-1", None)
    response = boot.master.process("hello there")
    assert response.response  # guidance text, non-empty
    assert response.model == ""


def test_master_always_returns_trace(mock_boot):
    boot, _ = mock_boot
    response = boot.master.process("explain the weather today with latest news")
    assert response.decision_trace is not None
    trace = response.decision_trace
    assert trace.hardware  # hardware snapshot present
    assert trace.execution_time_ms >= 0


def test_master_trace_explains_routing(mock_boot):
    boot, _ = mock_boot
    response = boot.master.process("Build a python script to fix this bug")
    trace = response.decision_trace
    assert trace is not None
    assert trace.capability_score > 0
    assert isinstance(trace.excluded_models, list)
    assert trace.estimated_latency_s is None or trace.estimated_latency_s > 0
    assert trace.expected_output_tokens is None or trace.expected_output_tokens > 0
    assert trace.intent_reasoning
    assert trace.complexity_reasoning
    assert trace.privacy_reasoning
    assert trace.hardware_reasoning


def test_master_performance_loop_records_outcome(mock_boot):
    boot, _ = mock_boot
    boot.master.process("Build a python script to fix this bug")
    perf = boot.container.resolve(__import__("synapse.contracts", fromlist=["PerformanceStore"]).PerformanceStore)
    stats = perf.stats()
    key = "mock/mock-1"
    assert key in stats
    # Phase 6: file-producing task + its review both reuse the mock model.
    assert stats[key]["samples"] == 2
    assert stats[key]["success_rate"] == 1.0


# -- Phase 3: task orchestration --------------------------------------------

MULTI_TASK_PROMPT = (
    "1. Design a website for our store.\n"
    "2. Write a poem for the launch.\n"
    "3. Explain the pricing to customers."
)


def test_master_decomposes_prompt_and_builds_execution_graph(mock_boot):
    boot, mock = mock_boot
    response = boot.master.process(MULTI_TASK_PROMPT)
    graph = response.execution_graph
    assert graph is not None
    assert len(graph.nodes) >= 4  # t1..t3 + synthesis
    ids = {n.task_id for n in graph.nodes}
    assert {"t1", "t2", "t3", "t-synthesis"} <= ids
    assert graph.synthesized is True
    assert graph.total_latency_ms >= 0
    # every execution task completed through the mock provider
    executed = [n for n in graph.nodes if n.task_id in {"t1", "t2", "t3"}]
    assert all(n.status.value == "completed" for n in executed)
    assert all(n.model_id == "mock-1" for n in executed)
    # Phase 6: t1 (a website -> file task) plus its review task both execute.
    assert len(mock.calls) == 4


def test_master_synthesis_merges_task_outputs(mock_boot):
    boot, _ = mock_boot
    response = boot.master.process(MULTI_TASK_PROMPT)
    # Phase 6: the file-producing task (t1) is excluded from synthesis parts,
    # leaving t2 + t3 + t-review to be merged.
    assert "consolidated result combining 3 sub-tasks" in response.response
    assert response.provider == "mock"
    assert response.model == "mock-1"


def test_master_single_task_passes_through_untouched(mock_boot):
    boot, _ = mock_boot
    response = boot.master.process("Build a python script to fix this bug")
    assert response.execution_graph is not None
    # Phase 6: the single file-producing task gains a review task.
    assert [n.task_id for n in response.execution_graph.nodes] == ["t1", "t-review"]
    assert response.execution_graph.nodes[0].task_id == "t1"
    assert response.response == "mock reply"


def test_master_writes_conversation_memory(mock_boot, temp_paths):
    boot, _ = mock_boot
    boot.master.process(MULTI_TASK_PROMPT)
    path = temp_paths.data_dir / "memory" / "conversation.json"
    assert path.exists()
    entries = json.loads(path.read_text(encoding="utf-8"))
    assert len(entries) == 2  # user prompt + assistant response
    assert entries[0]["text"] == MULTI_TASK_PROMPT


def test_master_task_failure_still_returns_response(mock_boot):
    boot, mock = mock_boot

    def boom(request):
        raise RuntimeError("provider exploded")

    mock.chat = boom
    response = boot.master.process(MULTI_TASK_PROMPT)
    graph = response.execution_graph
    assert graph is not None
    failed = [n for n in graph.nodes if n.task_id in {"t1", "t2", "t3"}]
    assert all(n.status.value == "failed" for n in failed)
    assert response.response  # guidance text, non-empty


def test_master_no_route_graph_has_failed_nodes(mock_boot):
    boot, _ = mock_boot
    boot.registry._models.pop("mock-1", None)
    response = boot.master.process(MULTI_TASK_PROMPT)
    graph = response.execution_graph
    assert graph is not None
    executed = [n for n in graph.nodes if n.task_id in {"t1", "t2", "t3"}]
    assert all(n.status.value == "failed" for n in executed)
    assert response.model == ""
