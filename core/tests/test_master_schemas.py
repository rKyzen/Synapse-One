"""Master AI schemas — structured TaskDecompositionPlan validation + DAG mapping."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synapse.domain import Capability, TaskKind
from synapse.master import ExecutionStrategy, SubTask, SubTaskIntent, TaskDecompositionPlan, intent_profile


def _sub(task_id: int, intent: str, prompt: str, deps: list[int] | None = None) -> dict:
    return {
        "task_id": task_id,
        "intent": intent,
        "sub_prompt": prompt,
        "dependencies": deps or [],
    }


def test_valid_plan_validates():
    plan = TaskDecompositionPlan(
        tasks=[
            SubTask(task_id=1, intent=SubTaskIntent.CODING, sub_prompt="fix the bug"),
            SubTask(task_id=2, intent=SubTaskIntent.WRITING, sub_prompt="write the email", dependencies=[1]),
        ],
        execution_strategy=ExecutionStrategy.SEQUENTIAL,
    )
    assert len(plan.tasks) == 2
    assert plan.execution_strategy is ExecutionStrategy.SEQUENTIAL


def test_multi_intent_prompt_like_fix_bug_and_email():
    data = {
        "execution_strategy": "HYBRID",
        "tasks": [
            _sub(1, "CODING", "Fix the bug in main.py"),
            _sub(2, "WRITING", "Write an email to the team", [1]),
            _sub(3, "CHAT", "Summarize the changes", [1, 2]),
        ],
    }
    plan = TaskDecompositionPlan.model_validate(data)
    intents = [t.intent for t in plan.tasks]
    assert intents == [SubTaskIntent.CODING, SubTaskIntent.WRITING, SubTaskIntent.CHAT]
    assert plan.tasks[2].dependencies == [1, 2]


def test_duplicate_task_ids_rejected():
    with pytest.raises(ValidationError):
        TaskDecompositionPlan.model_validate(
            {"tasks": [_sub(1, "CHAT", "a"), _sub(1, "CODING", "b")]}
        )


def test_unknown_dependency_rejected():
    with pytest.raises(ValidationError, match="unknown task 9"):
        TaskDecompositionPlan.model_validate(
            {"tasks": [_sub(1, "CHAT", "a", [9])]}
        )


def test_self_dependency_rejected():
    with pytest.raises(ValidationError, match="depends on itself"):
        TaskDecompositionPlan.model_validate(
            {"tasks": [_sub(1, "CHAT", "a", [1])]}
        )


def test_empty_tasks_rejected():
    with pytest.raises(ValidationError):
        TaskDecompositionPlan.model_validate({"tasks": []})


def test_unknown_intent_rejected():
    with pytest.raises(ValidationError):
        TaskDecompositionPlan.model_validate({"tasks": [_sub(1, "INVENTING", "a")]})


def test_empty_sub_prompt_rejected():
    with pytest.raises(ValidationError):
        TaskDecompositionPlan.model_validate({"tasks": [_sub(1, "CHAT", "  ")]})


def test_intent_profiles_map_to_routing_capabilities():
    kind, required, preferred = intent_profile(SubTaskIntent.CODING)
    assert kind is TaskKind.CODING
    assert Capability.CODING in required
    kind, required, preferred = intent_profile(SubTaskIntent.VISION)
    assert Capability.VISION in required
    kind, required, preferred = intent_profile(SubTaskIntent.TOOL)
    assert Capability.TOOLS in required


def test_to_dag_maps_ids_deps_and_appends_synthesis():
    plan = TaskDecompositionPlan.model_validate(
        {
            "execution_strategy": "SEQUENTIAL",
            "tasks": [
                _sub(1, "CODING", "build the script"),
                _sub(2, "WRITING", "document it", [1]),
            ],
        }
    )
    dag = plan.to_dag()
    ids = [t.id for t in dag.tasks]
    assert ids == ["t1", "t2", "t-synthesis"]
    t2 = dag.get("t2")
    assert t2 is not None
    assert t2.depends_on == ["t1"]
    assert t2.kind is TaskKind.WRITING
    assert dag.get("t-synthesis").depends_on == ["t1", "t2"]
    assert [t.id for t in dag.topological_order()] == ["t1", "t2", "t-synthesis"]


def test_to_dag_single_task_no_synthesis():
    plan = TaskDecompositionPlan.model_validate({"tasks": [_sub(1, "CHAT", "hello")]})
    dag = plan.to_dag()
    assert [t.id for t in dag.tasks] == ["t1"]
    assert dag.get("t1").kind is TaskKind.GENERAL
    assert Capability.CHAT in dag.get("t1").required_capabilities


def test_file_output_flag_flows_through():
    data = {"tasks": [_sub(1, "CODING", "build a website")]}
    data["tasks"][0]["file_output"] = True
    plan = TaskDecompositionPlan.model_validate(data)
    dag = plan.to_dag()
    assert dag.get("t1").file_output is True


def test_schema_is_strict_json_schema():
    schema = TaskDecompositionPlan.model_json_schema()
    assert schema["type"] == "object"
    assert "properties" in schema and "tasks" in schema["properties"]
    assert schema["required"] == ["tasks"]