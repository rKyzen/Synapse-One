"""Task Planner contract — decomposes a prompt into an executable task DAG."""

from __future__ import annotations

from abc import ABC, abstractmethod

from synapse.domain import (
    ComplexityResult,
    Decision,
    IntentResult,
    PrivacyResult,
)
from synapse.domain.tasks import TaskDAG


class TaskPlanner(ABC):
    @abstractmethod
    def plan(
        self,
        prompt: str,
        intent: IntentResult,
        complexity: ComplexityResult,
        privacy: PrivacyResult,
        decision: Decision,
    ) -> TaskDAG:
        """Break a prompt into smaller tasks as a dependency DAG.

        The planner is pure: prompt + analysis in, TaskDAG out. It never
        touches providers or the router; each task is routed independently
        later. A single-task DAG (the whole prompt) is a valid result for
        simple requests.
        """
