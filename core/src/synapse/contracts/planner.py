"""Execution Planner contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from synapse.domain import (
    ComplexityResult,
    Decision,
    ExecutionPlan,
    IntentResult,
    PrivacyResult,
    RoutingDecision,
)


class ExecutionPlanner(ABC):
    @abstractmethod
    def build_plan(
        self,
        decision: Decision,
        intent: IntentResult,
        complexity: ComplexityResult,
        privacy: PrivacyResult,
        routing: RoutingDecision | None = None,
    ) -> ExecutionPlan:
        """Assemble the reusable execution plan from analysis + routing.

        Routing may be absent at plan time (planned before routing); the plan
        then carries strategy + steps only.
        """