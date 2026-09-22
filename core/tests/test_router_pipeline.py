"""TaskExecutionPipeline tests — plan-driven routing with dependency forwarding."""

from __future__ import annotations

import pytest

from synapse.domain import (
    Capability,
    ChatResponse,
    Decision,
    HardwareProfile,
    PrivacyMode,
    ProviderKind,
    RoutingDecision,
)
from synapse.master import ExecutionStrategy, SubTask, SubTaskIntent, TaskDecompositionPlan
from synapse.router.pipeline import ExecutionReport, PipelineOutcome, TaskExecutionPipeline

HARDWARE = HardwareProfile()

BASE_DECISION = Decision(
    can_stay_local=True,
    internet_required=False,
    use_cloud_reasoning=False,
    preferred_kind=ProviderKind.LOCAL,
    required_capabilities=[Capability.CHAT],
    preferred_capabilities=[],
    privacy=PrivacyMode.PREFER_LOCAL,
)


class FakeRouter:
    """Routes by prompt markers; records decisions it received."""

    def __init__(self, coder_model: str = "coder"):
        self.coder_model = coder_model
        self.prompts: list[str] = []
        self.decisions: list[Decision] = []
        self.capability_checks: list[list[Capability]] = []

    def route(
        self,
        decision,
        hardware,
        registry,
        provider_health,
        available_models=None,
        *,
        complexity=None,
        prompt=None,
        performance=None,
    ):
        self.prompts.append(prompt or "")
        self.decisions.append(decision)
        self.capability_checks.append(list(decision.required_capabilities))
        if "unroutable" in (prompt or ""):
            return RoutingDecision(reason="no model could satisfy the task")
        return RoutingDecision(
            provider_id="fake",
            model_id=self.coder_model,
            kind=ProviderKind.LOCAL,
            reason="fake ok",
            confidence=0.9,
        )


class FakeExecutor:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def execute(self, routing, prompt, *, temperature=0.7, max_tokens=None):
        self.calls.append((routing.model_id, prompt))
        return ChatResponse(
            provider_id=routing.provider_id,
            model_id=routing.model_id,
            kind=ProviderKind.LOCAL,
            content=f"OUTPUT-{routing.model_id}",
        )


def _plan(*subs: tuple[int, str, str, list[int]]) -> TaskDecompositionPlan:
    return TaskDecompositionPlan(
        tasks=[
            SubTask(
                task_id=tid,
                intent=SubTaskIntent(intent),
                sub_prompt=prompt,
                dependencies=deps,
            )
            for tid, intent, prompt, deps in subs
        ],
        execution_strategy=ExecutionStrategy.SEQUENTIAL,
    )


def test_executes_in_dependency_order():
    plan = _plan(
        (1, "CODING", "fix the bug", []),
        (2, "CHAT", "summarize", [1]),
        (3, "WRITING", "document it", [1]),
    )
    router = FakeRouter()
    executor = FakeExecutor()
    report = TaskExecutionPipeline(router, executor, HARDWARE, [], {}).execute(plan, BASE_DECISION)

    assert report.execution_order == [1, 2, 3]
    assert [o.task_id for o in report.outcomes] == [1, 2, 3]
    assert not report.failed_task_ids
    # execution order enforced by the router receiving prompts in that order
    assert "fix the bug" in router.prompts[0]
    assert "summarize" in router.prompts[1]
    assert "document it" in router.prompts[2]


def test_dependency_outputs_forwarded_into_downstream_prompts():
    plan = _plan(
        (1, "CODING", "generate the config", []),
        (2, "CHAT", "explain the config", [1]),
    )
    router = FakeRouter()
    executor = FakeExecutor()
    TaskExecutionPipeline(router, executor, HARDWARE, [], {}).execute(plan, BASE_DECISION)

    second_prompt = router.prompts[1]
    assert "## Result of task 1" in second_prompt
    assert "OUTPUT-coder" in second_prompt
    # the original sub-prompt is preserved at the front
    assert second_prompt.startswith("explain the config")


def test_independent_tasks_do_not_receive_each_others_output():
    plan = _plan(
        (1, "CHAT", "greet", []),
        (2, "CHAT", "calculate", []),
    )
    router = FakeRouter()
    executor = FakeExecutor()
    TaskExecutionPipeline(router, executor, HARDWARE, [], {}).execute(plan, BASE_DECISION)
    assert "## Result of task" not in router.prompts[0]
    assert "## Result of task" not in router.prompts[1]


def test_unroutable_task_reported_and_skipped():
    plan = _plan((1, "CHAT", "unroutable thing", []), (2, "CHAT", "fine task", [1]))
    router = FakeRouter()
    executor = FakeExecutor()
    report = TaskExecutionPipeline(router, executor, HARDWARE, [], {}).execute(plan, BASE_DECISION)

    assert report.failed_task_ids == [1]
    first = report.outcomes[0]
    assert first.failed and first.failure_reason == "no model could satisfy the task"
    assert len(executor.calls) == 1  # only the routable task executed
    assert report.outcomes[1].output == "OUTPUT-coder"


def test_executor_failure_recorded_without_aborting_run():
    plan = _plan((1, "CHAT", "a", []), (2, "CHAT", "b", []))
    router = FakeRouter()

    class ExplodingExecutor:
        def execute(self, routing, prompt, *, temperature=0.7, max_tokens=None):
            raise RuntimeError("provider exploded")

    report = TaskExecutionPipeline(router, ExplodingExecutor(), HARDWARE, [], {}).execute(plan, BASE_DECISION)
    assert report.failed_task_ids == [1, 2]
    assert all(o.failed for o in report.outcomes)
    assert "provider exploded" in report.outcomes[0].failure_reason


def test_per_task_decision_carries_intent_capabilities():
    plan = _plan((1, "CODING", "write code", []), (2, "VISION", "look at image", []))
    router = FakeRouter()
    executor = FakeExecutor()
    TaskExecutionPipeline(router, executor, HARDWARE, [], {}).execute(plan, BASE_DECISION)

    assert Capability.CODING in router.capability_checks[0]
    assert Capability.VISION in router.capability_checks[1]


def test_custom_decision_factory_honored():
    plan = _plan((1, "CHAT", "x", []))
    router = FakeRouter()
    executor = FakeExecutor()

    def factory(task, base):
        return base.model_copy(update={"use_cloud_reasoning": True})

    TaskExecutionPipeline(
        router, executor, HARDWARE, [], {}, decision_factory=factory
    ).execute(plan, BASE_DECISION)
    assert router.decisions[0].use_cloud_reasoning is True


def test_cycle_degrades_to_insertion_order():
    plan = _plan((1, "CHAT", "a", [2]), (2, "CHAT", "b", [1]))
    router = FakeRouter()
    executor = FakeExecutor()
    report = TaskExecutionPipeline(router, executor, HARDWARE, [], {}).execute(plan, BASE_DECISION)
    # schema-level validation already forbids unknown deps, but a cycle
    # constructed directly must not hang the pipeline.
    assert report.execution_order == [1, 2]
    assert len(executor.calls) == 2


def test_report_shape():
    report = ExecutionReport(execution_strategy=ExecutionStrategy.HYBRID, execution_order=[1], outcomes=[], failed_task_ids=[])
    assert report.execution_strategy is ExecutionStrategy.HYBRID
    outcome = PipelineOutcome(task_id=1, intent=SubTaskIntent.CHAT, sub_prompt="x", output="y")
    assert outcome.model_dump()["intent"] == "CHAT"