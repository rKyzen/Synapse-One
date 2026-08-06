"""Result Synthesizer contract — merges per-task outputs into one response."""

from __future__ import annotations

from abc import ABC, abstractmethod

from synapse.domain.tasks import Task


class SynthesisResult:
    """Output of the synthesizer: final text + provenance."""

    def __init__(self, response: str, task_ids: list[str], models_used: list[str]) -> None:
        self.response = response
        self.task_ids = task_ids
        self.models_used = models_used


class Synthesizer(ABC):
    @abstractmethod
    def synthesize(
        self,
        prompt: str,
        tasks: list[Task],
    ) -> SynthesisResult:
        """Merge completed task outputs into a single final response.

        ``tasks`` are the executed tasks (status COMPLETED) in dependency
        order; each carries its ``result`` text and ``model_id``. The
        synthesizer must handle 1..N tasks and never raise.
        """
