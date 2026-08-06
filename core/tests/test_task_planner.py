"""Task Planner tests — deterministic decomposition into a task DAG."""

from __future__ import annotations

import pytest

from synapse.domain import (
    Capability,
    ComplexityResult,
    Decision,
    IntentResult,
    IntentType,
    PrivacyMode,
    PrivacyResult,
    ProviderKind,
)
from synapse.domain.enums import TaskKind
from synapse.planner import HeuristicTaskPlanner

DECISION = Decision(
    can_stay_local=True,
    internet_required=False,
    use_cloud_reasoning=False,
    preferred_kind=ProviderKind.LOCAL,
    required_capabilities=[Capability.CHAT],
    preferred_capabilities=[],
    privacy=PrivacyMode.PREFER_LOCAL,
)


def _intent() -> IntentResult:
    return IntentResult(primary=IntentType.CODING, confidence=0.9)


def _complexity(score: int) -> ComplexityResult:
    return ComplexityResult(score=score)


def _privacy() -> PrivacyResult:
    return PrivacyResult(mode=PrivacyMode.PREFER_LOCAL)


def _plan(prompt: str, complexity: int = 60, planner: HeuristicTaskPlanner | None = None):
    planner = planner or HeuristicTaskPlanner()
    return planner.plan(prompt, _intent(), _complexity(complexity), _privacy(), DECISION)


def test_low_complexity_prompt_is_single_task():
    dag = _plan("hello there", complexity=10)
    assert [t.id for t in dag.tasks] == ["t1"]
    assert len(dag.edges()) == 0


def test_numbered_list_splits_into_tasks_with_synthesis():
    dag = _plan("1. Build a python script to fix this bug.\n2. Write a poem about the sea.\n3. Explain the pricing.")
    ids = [t.id for t in dag.tasks]
    # Phase 6: t1 produces files -> a review task is appended before synthesis.
    assert ids == ["t1", "t2", "t3", "t-review", "t-synthesis"]
    kinds = {t.id: t.kind for t in dag.tasks}
    assert kinds["t1"] == TaskKind.CODING
    assert kinds["t2"] == TaskKind.WRITING
    assert kinds["t-review"] == TaskKind.REVIEW
    assert kinds["t-synthesis"] == TaskKind.SYNTHESIS
    review = dag.get("t-review")
    assert sorted(review.depends_on) == ["t1"]  # only the file-producing task
    synthesis = dag.get("t-synthesis")
    assert set(synthesis.depends_on) == {"t1", "t2", "t3", "t-review"}


def test_synthesis_depends_on_all_tasks():
    dag = _plan("1. Research the market.\n2. Design the roadmap.")
    synthesis = dag.get("t-synthesis")
    assert sorted(synthesis.depends_on) == ["t1", "t2"]
    assert all(dep in {t.id for t in dag.tasks} for dep in synthesis.depends_on)


def test_plan_is_acyclic_and_orders_tasks():
    dag = _plan("1. Build a script.\n2. Debug it.\n3. Document it.")
    order = [t.id for t in dag.topological_order()]
    # Phase 6: t1 produces a file -> review is scheduled right after t1.
    assert order[:4] == ["t1", "t-review", "t2", "t3"]
    assert order[-1] == "t-synthesis"
    assert all(
        dep in order and order.index(dep) < order.index(t.id)
        for t in dag.tasks for dep in t.depends_on
    )


def test_disabled_planner_never_splits():
    planner = HeuristicTaskPlanner(enabled=False)
    dag = planner.plan("1. First task. 2. Second task.", _intent(), _complexity(90), _privacy(), DECISION)
    assert [t.id for t in dag.tasks] == ["t1"]


def test_max_tasks_ceiling():
    planner = HeuristicTaskPlanner(max_tasks=2)
    dag = _plan(
        "1. Build the login page.\n2. Build the cart page.\n3. Build the checkout page.\n4. Build the admin panel.",
        planner=planner,
    )
    exec_ids = [t.id for t in dag.tasks if t.id != "t-synthesis"]
    assert len(exec_ids) == 2


def test_unknown_parts_classify_as_general():
    dag = _plan("1. zebra quirk.\n2. llama glance.")
    kinds = {t.id: t.kind for t in dag.tasks}
    assert kinds["t1"] == TaskKind.GENERAL
    assert kinds["t2"] == TaskKind.GENERAL


def test_single_task_keeps_prompt_verbatim():
    dag = _plan("Build a python script to fix this bug", complexity=10)
    task = dag.get("t1")
    assert task.description == "Build a python script to fix this bug"
    assert task.required_capabilities  # profile applied


def test_synthesis_task_has_instruction_description():
    dag = _plan("1. Write a summary.\n2. Write an essay.")
    synthesis = dag.get("t-synthesis")
    assert "combine" in synthesis.description.lower()


def test_tiny_parts_are_merged_into_previous():
    dag = _plan("1. Write an essay about trees.\n2. ok")
    ids = [t.id for t in dag.tasks]
    assert "t2" not in ids  # the tiny "ok" fragment was merged into t1