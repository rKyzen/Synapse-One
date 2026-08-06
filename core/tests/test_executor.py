"""Executor + planner tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from synapse.config.paths import SynapsePaths
from synapse.config.provider import ConfigProvider
from synapse.domain import (
    ComplexityResult,
    ExecutionStrategy,
    IntentResult,
    IntentType,
    PrivacyMode,
    PrivacyResult,
    ProviderKind,
    RoutingDecision,
)
from synapse.events import EventBus
from synapse.execution import Executor, ExecutionPlanner, ProviderUnavailable
from synapse.providers.factory import ProviderFactory
from synapse.providers.manager import ProviderManager
from test_master import MockProvider


def _manager(mock: MockProvider) -> ProviderManager:
    config = ConfigProvider(paths=SynapsePaths.discover(), env={}).load()
    events = EventBus()
    manager = ProviderManager(config, ProviderFactory(config, events), events)
    manager._providers["mock"] = mock
    return manager


def _decision(**overrides) -> SimpleNamespace:
    base = dict(
        can_stay_local=True,
        use_cloud_reasoning=False,
        internet_required=False,
        preferred_kind=ProviderKind.LOCAL,
        required_capabilities=[],
        workspace="general",
        use_memory=False,
        privacy=PrivacyMode.PREFER_LOCAL,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_executor_routes_through_provider():
    mock = MockProvider()
    executor = Executor(_manager(mock))
    routing = RoutingDecision(provider_id="mock", model_id="mock-1", kind=ProviderKind.LOCAL)
    resp = executor.execute(routing, "hello")
    assert resp.content == "mock reply"
    assert mock.calls == ["hello"]
    assert resp.model_id == "mock-1"


def test_executor_missing_provider_raises():
    executor = Executor(_manager(MockProvider()))
    with pytest.raises(ProviderUnavailable):
        executor.execute(RoutingDecision(provider_id="ghost", model_id="x"), "hi")


def test_planner_local_strategy():
    plan = ExecutionPlanner().build_plan(
        _decision(),
        IntentResult(primary=IntentType.CONVERSATION),
        ComplexityResult(score=10),
        PrivacyResult(mode=PrivacyMode.PREFER_LOCAL),
    )
    assert plan.strategy == ExecutionStrategy.LOCAL
    assert plan.steps
    assert plan.intent == IntentType.CONVERSATION
    assert plan.complexity == 10


def test_planner_cloud_strategy():
    plan = ExecutionPlanner().build_plan(
        _decision(can_stay_local=False, use_cloud_reasoning=True),
        IntentResult(primary=IntentType.RESEARCH),
        ComplexityResult(score=95),
        PrivacyResult(mode=PrivacyMode.CLOUD_REQUIRED),
    )
    assert plan.strategy == ExecutionStrategy.CLOUD


def test_planner_hybrid_strategy():
    plan = ExecutionPlanner().build_plan(
        _decision(use_cloud_reasoning=True, internet_required=True),
        IntentResult(primary=IntentType.RESEARCH),
        ComplexityResult(score=90),
        PrivacyResult(mode=PrivacyMode.CLOUD_REQUIRED, internet_required=True),
    )
    assert plan.strategy == ExecutionStrategy.HYBRID
