"""TemplateSynthesizer — merges per-task outputs into one final response.

Deterministic, rule-based merging: each completed task becomes a headed
section attributing its model, then the parts are joined. A single task
passes through untouched. A model-backed synthesizer can replace this behind
the Synthesizer contract later.
"""

from __future__ import annotations

import structlog

from synapse.contracts.synthesizer import SynthesisResult, Synthesizer
from synapse.domain.tasks import Task

log = structlog.get_logger("synapse.synthesis")


class TemplateSynthesizer(Synthesizer):
    """Rule-based multi-model result merger."""

    def synthesize(self, prompt: str, tasks: list[Task]) -> SynthesisResult:
        completed = [t for t in tasks if t.result]
        if not completed:
            return SynthesisResult(response="", task_ids=[], models_used=[])

        models_used = sorted({t.model_id for t in completed if t.model_id})

        if len(completed) == 1:
            task = completed[0]
            log.info("synthesis_single", task_id=task.id, model=task.model_id)
            return SynthesisResult(
                response=task.result or "",
                task_ids=[task.id],
                models_used=models_used,
            )

        parts = []
        for task in completed:
            header = f"## {task.kind.value.title()} — {task.description.strip()[:80]}"
            attribution = f"*via {task.model_id}*" if task.model_id else ""
            parts.append(f"{header} {attribution}\n\n{task.result.strip()}")
        body = "\n\n".join(parts)
        final = (
            f"Here is the consolidated result combining {len(completed)} "
            f"sub-tasks ({len(models_used)} model(s)).\n\n{body}"
        )
        log.info("synthesis_merged", parts=len(completed), models=len(models_used))
        return SynthesisResult(
            response=final,
            task_ids=[t.id for t in completed],
            models_used=models_used,
        )