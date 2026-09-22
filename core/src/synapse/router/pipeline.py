"""Execution loop that consumes a TaskDecompositionPlan (AI Master output).

The deterministic Router routes each independent sub-task, and the outputs of
completed dependency tasks are forwarded into the context of downstream tasks
so later steps build on earlier results. This is the reusable, replaceable
loop for plan-driven execution; the Master Agent uses the equivalent loop over
its TaskDAG (with the full quality pipeline), sharing the same contract types.
"""

from __future__ import annotations

from typing import Callable

import structlog
from pydantic import BaseModel, Field

from synapse.contracts import Executor, Router
from synapse.domain.diagnosis import Decision, RoutingDecision
from synapse.domain.hardware import HardwareProfile
from synapse.master.schemas import ExecutionStrategy, SubTask, SubTaskIntent, TaskDecompositionPlan, intent_profile

log = structlog.get_logger("synapse.router.pipeline")


class PipelineOutcome(BaseModel):
    """Result of one sub-task in a pipeline execution."""

    task_id: int
    intent: SubTaskIntent
    sub_prompt: str
    provider_id: str = ""
    model_id: str = ""
    output: str = ""
    failed: bool = False
    failure_reason: str = ""


class ExecutionReport(BaseModel):
    """Full run: outcomes in execution order + overall strategy."""

    execution_strategy: ExecutionStrategy
    execution_order: list[int] = Field(default_factory=list)
    outcomes: list[PipelineOutcome] = Field(default_factory=list)
    failed_task_ids: list[int] = Field(default_factory=list)


class TaskExecutionPipeline:
    """Runs a decomposition plan: route -> execute -> forward outputs.

    Dependencies are honored (a task's prompt receives the outputs of every
    completed prerequisite), and each task is routed independently through the
    deterministic capabilities router — the plan decides WHAT runs, the router
    decides WHO runs it.
    """

    def __init__(
        self,
        router: Router,
        executor: Executor,
        hardware: HardwareProfile,
        registry: list,
        provider_health: dict[str, bool],
        *,
        performance: dict[str, dict] | None = None,
        decision_factory: Callable[[SubTask, Decision], Decision] | None = None,
    ) -> None:
        self._router = router
        self._executor = executor
        self._hardware = hardware
        self._registry = registry
        self._provider_health = provider_health
        self._performance = performance
        self._decision_factory = decision_factory or TaskExecutionPipeline._default_decision

    # -- public -------------------------------------------------------------

    def execute(
        self,
        plan: TaskDecompositionPlan,
        decision: Decision,
        *,
        complexity: int = 0,
        prompt: str = "",
    ) -> ExecutionReport:
        """Execute every supported task of the plan in dependency order.

        Tasks that can never route (no gates survive) are reported as failed
        and skipped — their dependents still run with whatever context exists.
        """
        order = self._topological(plan)
        outputs: dict[int, str] = {}
        outcomes: list[PipelineOutcome] = []
        failed: list[int] = []

        for task in order:
            sub_prompt = self._with_prior_context(task, outputs)
            task_decision = self._decision_factory(task, decision)
            routing = self._route(task_decision, sub_prompt, complexity)
            if routing is None or not routing.model_id:
                reason = routing.reason if routing else "no model could satisfy the task"
                outcomes.append(
                    PipelineOutcome(
                        task_id=task.task_id,
                        intent=task.intent,
                        sub_prompt=task.sub_prompt,
                        failed=True,
                        failure_reason=reason,
                    )
                )
                failed.append(task.task_id)
                log.warning("pipeline_task_unrouted", task_id=task.task_id, reason=reason)
                continue

            try:
                response = self._executor.execute(
                    routing, sub_prompt, temperature=0.7
                )
                content = response.content or ""
            except Exception as exc:  # noqa: BLE001 - one task's failure is not a run failure
                outcomes.append(
                    PipelineOutcome(
                        task_id=task.task_id,
                        intent=task.intent,
                        sub_prompt=task.sub_prompt,
                        failed=True,
                        failure_reason=str(exc)[:200],
                    )
                )
                failed.append(task.task_id)
                log.warning("pipeline_task_failed", task_id=task.task_id, error=str(exc)[:200])
                continue

            outputs[task.task_id] = content
            outcomes.append(
                PipelineOutcome(
                    task_id=task.task_id,
                    intent=task.intent,
                    sub_prompt=task.sub_prompt,
                    provider_id=routing.provider_id,
                    model_id=routing.model_id,
                    output=content,
                )
            )
            log.info(
                "pipeline_task_completed",
                task_id=task.task_id,
                provider=routing.provider_id,
                model=routing.model_id,
                chars=len(content),
            )

        return ExecutionReport(
            execution_strategy=plan.execution_strategy,
            execution_order=[t.task_id for t in order],
            outcomes=outcomes,
            failed_task_ids=failed,
        )

    # -- internals ----------------------------------------------------------

    def _route(
        self, decision: Decision, sub_prompt: str, complexity: int
    ) -> RoutingDecision | None:
        try:
            return self._router.route(
                decision,
                self._hardware,
                self._registry,
                self._provider_health,
                None,
                complexity=complexity,
                prompt=sub_prompt,
                performance=self._performance,
            )
        except Exception as exc:  # noqa: BLE001 - router failures degrade, never crash
            log.warning("pipeline_route_failed", error=str(exc)[:200])
            return None

    @staticmethod
    def _default_decision(task: SubTask, base: Decision) -> Decision:
        """Derive a per-task decision from the base one + intent capabilities."""
        _, required, preferred = intent_profile(task.intent)
        return base.model_copy(
            update={
                "required_capabilities": required,
                "preferred_capabilities": preferred,
            }
        )

    @staticmethod
    def _with_prior_context(task: SubTask, outputs: dict[int, str]) -> str:
        """Append the outputs of completed dependencies to the task prompt."""
        prior: list[str] = []
        for dep in task.dependencies:
            out = outputs.get(dep)
            if out:
                prior.append(f"## Result of task {dep}\n{out}")
        if not prior:
            return task.sub_prompt
        return f"{task.sub_prompt}\n\nContext from completed dependency tasks:\n\n" + "\n\n".join(prior)

    @staticmethod
    def _topological(plan: TaskDecompositionPlan) -> list[SubTask]:
        """Deterministic Kahn's sort over int task ids. Cycle-safe fallback."""
        by_id = {t.task_id: t for t in plan.tasks}
        indegree = {t.task_id: len([d for d in t.dependencies if d in by_id]) for t in plan.tasks}
        dependents: dict[int, list[int]] = {t.task_id: [] for t in plan.tasks}
        for t in plan.tasks:
            for dep in t.dependencies:
                if dep in dependents:
                    dependents[dep].append(t.task_id)

        ready = sorted(tid for tid, deg in indegree.items() if deg == 0)
        ordered: list[int] = []
        while ready:
            tid = ready.pop(0)
            ordered.append(tid)
            for child in dependents.get(tid, []):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort()
        if len(ordered) != len(plan.tasks):
            cyclic = [t.task_id for t in plan.tasks if t.task_id not in ordered]
            log.warning("pipeline_cycle_fallback", cyclic=cyclic)
            ordered = [t.task_id for t in plan.tasks]  # degrade: insertion order
        return [by_id[tid] for tid in ordered]