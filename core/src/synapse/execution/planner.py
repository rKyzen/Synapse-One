"""Execution planning — turns analysis results into a reusable ExecutionPlan."""

from __future__ import annotations

from synapse.contracts import ExecutionPlanner
from synapse.domain import (
    ComplexityResult,
    Decision,
    ExecutionPlan,
    ExecutionStrategy,
    IntentResult,
    LatencyTier,
    PrivacyResult,
    RoutingDecision,
)


class ExecutionPlanner(ExecutionPlanner):
    """Deterministic plan assembly. Reusable by future phases (memory,
    multi-agent, workspace engine)."""

    def build_plan(
        self,
        decision: Decision,
        intent: IntentResult,
        complexity: ComplexityResult,
        privacy: PrivacyResult,
        routing: RoutingDecision | None = None,
    ) -> ExecutionPlan:
        if decision.internet_required and decision.use_cloud_reasoning:
            strategy = ExecutionStrategy.HYBRID
        elif decision.use_cloud_reasoning:
            strategy = ExecutionStrategy.CLOUD
        elif decision.can_stay_local:
            strategy = ExecutionStrategy.LOCAL
        else:
            strategy = ExecutionStrategy.NO_ROUTE

        steps = self._steps(decision, routing, strategy)
        latency = self._latency_tier(strategy, routing)
        return ExecutionPlan(
            intent=intent.primary,
            complexity=complexity.score,
            privacy=privacy.mode,
            required_capabilities=decision.required_capabilities,
            preferred_kind=decision.preferred_kind,
            workspace=decision.workspace,
            strategy=strategy,
            steps=steps,
            estimated_latency_tier=latency,
            estimated_cost_per_1k=routing.estimated_cost_per_1k if routing else 0.0,
            estimated_latency_s=routing.estimated_latency_s if routing else None,
            expected_output_tokens=routing.expected_output_tokens if routing else None,
        )

    @staticmethod
    def _steps(decision: Decision, routing: RoutingDecision | None, strategy: ExecutionStrategy) -> list[str]:
        steps = [
            "analyze: intent → complexity → privacy",
            f"route: prefer {decision.preferred_kind.value} execution",
        ]
        if strategy == ExecutionStrategy.LOCAL:
            steps.append("execute locally (privacy-first)")
        elif strategy == ExecutionStrategy.CLOUD:
            steps.append("execute on cloud provider")
        elif strategy == ExecutionStrategy.HYBRID:
            steps.append("hybrid: cloud for reasoning, local for cheap parts")
        else:
            steps.append("no route available — return guidance")
        if routing and routing.model_id:
            steps.append(f"selected {routing.provider_id}/{routing.model_id}")
        return steps

    @staticmethod
    def _latency_tier(strategy: ExecutionStrategy, routing: RoutingDecision | None) -> LatencyTier:
        if routing and routing.kind and routing.kind.value == "local":
            return LatencyTier.FAST
        if strategy in (ExecutionStrategy.LOCAL, ExecutionStrategy.HYBRID):
            return LatencyTier.MEDIUM
        return LatencyTier.SLOW