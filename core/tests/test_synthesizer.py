"""TemplateSynthesizer tests — single passthrough and multi-task merging."""

from __future__ import annotations

from synapse.domain.enums import TaskKind
from synapse.domain.tasks import Task
from synapse.synthesis import TemplateSynthesizer


def _task(tid: str, kind: TaskKind, result: str | None, model: str | None = "mock-1") -> Task:
    return Task(
        id=tid,
        kind=kind,
        description=f"{tid} description",
        required_capabilities=[],
        preferred_capabilities=[],
        result=result,
        model_id=model,
    )


def test_single_task_passes_through():
    synth = TemplateSynthesizer()
    result = synth.synthesize("prompt", [_task("t1", TaskKind.CODING, "the answer")])
    assert result.response == "the answer"
    assert result.task_ids == ["t1"]
    assert result.models_used == ["mock-1"]


def test_empty_tasks_return_empty():
    result = TemplateSynthesizer().synthesize("prompt", [])
    assert result.response == ""
    assert result.task_ids == []


def test_unfinished_tasks_are_excluded():
    synth = TemplateSynthesizer()
    result = synth.synthesize(
        "prompt",
        [_task("t1", TaskKind.CODING, "done"), _task("t2", TaskKind.WRITING, None)],
    )
    assert result.task_ids == ["t1"]
    assert result.response == "done"


def test_multi_task_merges_with_headers_and_attribution():
    synth = TemplateSynthesizer()
    result = synth.synthesize(
        "prompt",
        [
            _task("t1", TaskKind.CODING, "script output", "mock-1"),
            _task("t2", TaskKind.WRITING, "poem output", "mock-2"),
        ],
    )
    assert result.task_ids == ["t1", "t2"]
    assert result.models_used == ["mock-1", "mock-2"]
    assert "consolidated result combining 2 sub-tasks (2 model(s))" in result.response
    assert "## Coding" in result.response
    assert "## Writing" in result.response
    assert "*via mock-1*" in result.response
    assert "script output" in result.response
    assert "poem output" in result.response


def test_models_used_deduplicated_and_sorted():
    synth = TemplateSynthesizer()
    result = synth.synthesize(
        "prompt",
        [
            _task("t1", TaskKind.CODING, "a", "mock-2"),
            _task("t2", TaskKind.WRITING, "b", "mock-1"),
            _task("t3", TaskKind.MATH, "c", "mock-2"),
        ],
    )
    assert result.models_used == ["mock-1", "mock-2"]