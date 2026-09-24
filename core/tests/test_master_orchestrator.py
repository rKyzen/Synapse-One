"""AIMasterOrchestrator tests — AI task divider with hardware-tier model role."""

from __future__ import annotations

import json

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
from synapse.domain.hardware import MemoryInfo
from synapse.hardware import HardwareTier, TierResolver
from synapse.master import AIMasterOrchestrator, TaskDecompositionPlan, extract_json_object
from synapse.master.orchestrator import MasterModelSelection

DECISION = Decision(
    can_stay_local=True,
    internet_required=False,
    use_cloud_reasoning=False,
    preferred_kind=ProviderKind.LOCAL,
    required_capabilities=[Capability.CHAT],
    preferred_capabilities=[],
    privacy=PrivacyMode.PREFER_LOCAL,
)

_GOOD_PLAN = {
    "execution_strategy": "SEQUENTIAL",
    "tasks": [
        {"task_id": 1, "intent": "CODING", "sub_prompt": "fix the bug in main.py", "dependencies": []},
        {"task_id": 2, "intent": "WRITING", "sub_prompt": "write an email to the team", "dependencies": [1]},
    ],
}

_FENCED_PLAN = "```json\n" + json.dumps(_GOOD_PLAN) + "\n```"


def _results() -> tuple[IntentResult, ComplexityResult, PrivacyResult]:
    return (
        IntentResult(primary=IntentType.CODING, confidence=0.9),
        ComplexityResult(score=60),
        PrivacyResult(mode=PrivacyMode.PREFER_LOCAL),
    )


class ScriptedProvider(ModelProvider):
    provider_id = "fake"
    kind = ProviderKind.LOCAL

    def __init__(self, replies: list):
        self._replies = list(replies)
        self.requests: list[ChatRequest] = []

    def initialize(self):
        pass

    def list_models(self) -> list[ModelDescriptor]:
        return [ModelDescriptor(id="gemma3:1b", provider_id=self.provider_id)]

    def to_metadata(self, descriptor: ModelDescriptor) -> ModelMetadata | None:
        return None

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResponse(
            provider_id=self.provider_id,
            model_id=request.model or self.provider_id,
            kind=self.kind,
            content=reply,
        )

    def health(self) -> bool:
        return True

    def supports(self, capability: Capability) -> bool:
        return True

    def shutdown(self):
        pass


class FakeManager:
    def __init__(self, provider: ScriptedProvider | None = None):
        self._provider = provider

    def get(self, provider_id: str):
        if self._provider and provider_id == self._provider.provider_id:
            return self._provider
        return None

    def health(self, provider_id: str) -> bool:
        return bool(self._provider and provider_id == self._provider.provider_id)


class FakeRegistry:
    def __init__(self, models: list[ModelMetadata]):
        self._models = models

    def all(self) -> list[ModelMetadata]:
        return self._models


class FakeHardware:
    def __init__(self, available_gb: float):
        self._profile = HardwareProfile(memory=MemoryInfo(total_gb=32.0, available_gb=available_gb))

    def scan(self) -> HardwareProfile:
        return self._profile


class FallbackStub:
    def __init__(self):
        self.called_with: str | None = None

    def plan(self, prompt, intent, complexity, privacy, decision):
        self.called_with = prompt
        from synapse.domain.tasks import Task, TaskDAG

        return TaskDAG(tasks=[Task(id="fallback", description=prompt)])


def _model(model_id: str, kind: ProviderKind = ProviderKind.LOCAL) -> ModelMetadata:
    return ModelMetadata(
        id=model_id,
        provider_id="fake",
        kind=kind,
        capabilities=ModelCapabilities(chat=0.8, coding=0.7, reasoning=0.6),
    )


def _make_orchestrator(
    provider: ScriptedProvider,
    *,
    models: list[ModelMetadata] | None = None,
    available_gb: float = 8.0,
    fallback: FallbackStub | None = None,
    resolver: TierResolver | None = None,
    **kwargs,
):
    return AIMasterOrchestrator(
        providers=FakeManager(provider),
        registry=FakeRegistry(models if models is not None else [_model("gemma3:1b")]),
        hardware=FakeHardware(available_gb),
        config=None,  # type: ignore[arg-type] - defaults suffice
        resolver=resolver,
        fallback_planner=fallback,
        **kwargs,
    )


# -- happy path -------------------------------------------------------------


def test_valid_plan_produces_dag_and_picks_tier_model():
    provider = ScriptedProvider([json.dumps(_GOOD_PLAN)])
    orch = _make_orchestrator(provider, available_gb=8.0)
    intent, complexity, privacy = _results()
    dag = orch.plan("fix this bug in main.py and write an email", intent, complexity, privacy, DECISION)

    assert orch.used_ai is True
    assert orch.last_tier.value == "tier2"
    assert orch.last_model == "gemma3:1b"
    ids = [t.id for t in dag.tasks]
    assert ids == ["t1", "t2", "t-synthesis"]
    assert dag.get("t2").depends_on == ["t1"]
    assert dag.get("t2").kind is TaskKind.WRITING


def test_schema_passed_as_structured_output_and_system_prompt_used():
    provider = ScriptedProvider([_FENCED_PLAN])
    orch = _make_orchestrator(provider)
    intent, complexity, privacy = _results()
    orch.plan("fix and email", intent, complexity, privacy, DECISION)

    request = provider.requests[0]
    assert request.format == TaskDecompositionPlan.model_json_schema()
    system = request.messages[0]
    assert system.role == "system"
    assert "Master AI Orchestrator" in system.content
    assert "NEVER directly solve" in system.content
    assert "USER PROMPT:\nfix and email" in request.messages[1].content
    assert "fix and email" in request.messages[1].content


def test_fenced_output_is_repaired():
    provider = ScriptedProvider([_FENCED_PLAN])
    orch = _make_orchestrator(provider)
    intent, complexity, privacy = _results()
    dag = orch.plan("fix and email", intent, complexity, privacy, DECISION)
    assert orch.used_ai and dag.get("t1").kind is TaskKind.CODING


def test_retry_after_provider_exception():
    provider = ScriptedProvider([RuntimeError("boom"), json.dumps(_GOOD_PLAN)])
    orch = _make_orchestrator(provider)
    intent, complexity, privacy = _results()
    dag = orch.plan("fix and email", intent, complexity, privacy, DECISION)
    assert orch.used_ai is True
    assert len(provider.requests) == 2
    assert [t.id for t in dag.tasks] == ["t1", "t2", "t-synthesis"]


# -- fallbacks -------------------------------------------------------------


def test_garbage_output_falls_back_to_planner():
    provider = ScriptedProvider(["I am a router" * 5])
    fallback = FallbackStub()
    orch = _make_orchestrator(provider, fallback=fallback)
    intent, complexity, privacy = _results()
    dag = orch.plan("fix and email", intent, complexity, privacy, DECISION)
    assert orch.used_ai is False
    assert fallback.called_with == "fix and email"
    assert [t.id for t in dag.tasks] == ["fallback"]


def test_invalid_schema_output_falls_back():
    provider = ScriptedProvider([json.dumps({"tasks": [{"task_id": 1, "intent": "CODING", "sub_prompt": "x", "dependencies": [9]}]})])
    fallback = FallbackStub()
    orch = _make_orchestrator(provider, fallback=fallback)
    intent, complexity, privacy = _results()
    orch.plan("fix and email", intent, complexity, privacy, DECISION)
    assert fallback.called_with == "fix and email"


def test_no_candidate_model_falls_back():
    provider = ScriptedProvider([])
    fallback = FallbackStub()
    orch = _make_orchestrator(provider, models=[], fallback=fallback)
    intent, complexity, privacy = _results()
    orch.plan("fix and email", intent, complexity, privacy, DECISION)
    assert orch.last_model is None
    assert fallback.called_with is not None
    assert provider.requests == []


def test_unhealthy_provider_falls_back():
    provider = ScriptedProvider([json.dumps(_GOOD_PLAN)])
    provider._healthy = False

    class UnhealthyManager(FakeManager):
        def health(self, provider_id):
            return False

    orch = AIMasterOrchestrator(
        providers=UnhealthyManager(provider),
        registry=FakeRegistry([_model("gemma3:4b")]),
        hardware=FakeHardware(8.0),
        config=None,  # type: ignore[arg-type]
        fallback_planner=FallbackStub(),
    )
    intent, complexity, privacy = _results()
    dag = orch.plan("fix and email", intent, complexity, privacy, DECISION)
    assert [t.id for t in dag.tasks] == ["fallback"]


def test_disabled_uses_fallback_directly():
    provider = ScriptedProvider([])
    fallback = FallbackStub()
    orch = _make_orchestrator(provider, fallback=fallback, enabled=False)
    intent, complexity, privacy = _results()
    orch.plan("fix and email", intent, complexity, privacy, DECISION)
    assert fallback.called_with is not None
    assert provider.requests == []


def test_tier_escalation_when_assigned_tier_missing():
    # tier2 assigned (8GB) with gemma3:4b configured, but only tier1 model (gemma3:1b) installed -> tier1 selected.
    provider = ScriptedProvider([json.dumps(_GOOD_PLAN)])
    custom_resolver = TierResolver(candidates={HardwareTier.TIER1: ["gemma3:1b"], HardwareTier.TIER2: ["gemma3:4b"]})
    orch = _make_orchestrator(
        provider,
        models=[_model("gemma3:1b")],
        available_gb=8.0,
        resolver=custom_resolver,
    )
    intent, complexity, privacy = _results()
    orch.plan("fix and email", intent, complexity, privacy, DECISION)
    assert orch.last_tier.value == "tier1"
    assert orch.last_model == "gemma3:1b"


def test_cloud_fallback_tier_used_when_local_ram_tiny():
    provider = ScriptedProvider([json.dumps(_GOOD_PLAN)])
    orch = _make_orchestrator(
        provider,
        models=[_model("gpt-4o-mini", kind=ProviderKind.CLOUD)],
        available_gb=0.3,
    )
    intent, complexity, privacy = _results()
    orch.plan("fix and email", intent, complexity, privacy, DECISION)
    assert orch.last_tier.value == "cloud_fallback"
    assert orch.last_model == "gpt-4o-mini"
    assert provider.requests[0].model == "gpt-4o-mini"


def test_max_tasks_caps_plan_and_removes_stale_deps():
    plan = {
        "execution_strategy": "SEQUENTIAL",
        "tasks": [
            {"task_id": i, "intent": "CHAT", "sub_prompt": f"step {i}", "dependencies": [] if i == 1 else [i - 1]}
            for i in range(1, 13)
        ],
    }
    provider = ScriptedProvider([json.dumps(plan)])
    orch = _make_orchestrator(provider, available_gb=16.0, max_tasks=4)
    intent, complexity, privacy = _results()
    dag = orch.plan("long multi-step", intent, complexity, privacy, DECISION)
    assert orch.used_ai is True
    assert [t.id for t in dag.tasks] == ["t1", "t2", "t3", "t4", "t-synthesis"]
    assert dag.get("t4").depends_on == ["t3"]  # dependency on dropped t5 removed


def test_extract_json_object_repairs_noise():
    assert extract_json_object('{"a": 1}') == {"a": 1}
    assert extract_json_object("sure, here it is:\n```json\n{\"a\": 1}\n```") == {"a": 1}
    assert extract_json_object("prefix {\"a\": [1,2]} suffix") == {"a": [1, 2]}
    with pytest.raises(ValueError):
        extract_json_object("no json here")


def test_prompt_includes_dynamic_registry_and_hardware_context():
    provider = ScriptedProvider([json.dumps(_GOOD_PLAN)])
    meta = ModelMetadata(
        id="qwen2.5-coder:7b",
        provider_id="fake",
        kind=ProviderKind.LOCAL,
        role="Coding Specialist",
        strengths=["Fast code", "clean syntax"],
        weaknesses=["No vision"],
        speed_tier="fast",
        hardware_tier="tier2",
        capabilities=ModelCapabilities(coding=0.9),
    )
    orch = _make_orchestrator(provider, models=[_model("gemma3:4b"), meta], available_gb=8.0)
    intent, complexity, privacy = _results()
    orch.plan("build an app", intent, complexity, privacy, DECISION)
    prompt = provider.requests[0].messages[1].content
    assert "SYSTEM HARDWARE CONTEXT:" in prompt
    assert "REGISTERED SPECIALIST MODELS & TOOLS" in prompt
    assert "qwen2.5-coder:7b" in prompt
    assert "Strengths: Fast code; clean syntax" in prompt
    assert "Weaknesses: No vision" in prompt


def test_qwen_32b_guardrail_downgrades_on_tier2():
    plan_with_32b = {
        "execution_strategy": "SEQUENTIAL",
        "tasks": [
            {
                "task_id": 1,
                "intent": "CHAT",
                "sub_prompt": "hello",
                "assigned_model": "qwen2.5:32b",
                "reasoning": "simple chat",
                "dependencies": [],
            }
        ],
    }
    provider = ScriptedProvider([json.dumps(plan_with_32b)])
    orch = _make_orchestrator(provider, available_gb=8.0)
    intent, complexity, privacy = _results()
    dag = orch.plan("hello", intent, complexity, privacy, DECISION)
    assert dag.get("t1").preferred_model != "qwen2.5:32b"
    assert dag.get("t1").preferred_model == "gemma3:4b"


def test_qwen_32b_allowed_on_tier3_plus_for_deep_reasoning():
    plan_with_32b = {
        "execution_strategy": "SEQUENTIAL",
        "tasks": [
            {
                "task_id": 1,
                "intent": "DEEP_REASONING",
                "sub_prompt": "prove this mathematical theorem",
                "assigned_model": "qwen2.5:32b",
                "reasoning": "deep mathematical theorem proof required",
                "dependencies": [],
            }
        ],
    }
    provider = ScriptedProvider([json.dumps(plan_with_32b)])
    orch = _make_orchestrator(provider, available_gb=36.0)
    intent, complexity, privacy = _results()
    dag = orch.plan("prove theorem", intent, complexity, privacy, DECISION)
    assert dag.get("t1").preferred_model == "qwen2.5:32b"