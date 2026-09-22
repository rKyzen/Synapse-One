"""GoalExecutor — capability-first execution of goal plans (Phase B).

The AI Operating Workspace runs the Goal → Plan → Execute → Update pipeline:
after the GoalPlanner produces capability-labeled steps, this executor picks a
model per step via the SAME contract chain as the legacy pipeline (Router with
performance history + lifecycle model reuse) and executes through the Executor.
The orchestrator, not the agent, owns the goal record: step status/progress are
persisted through the GoalStore, results are attached to steps, and every phase
publishes GOAL_* events so the workspace timeline stays live.

Phase B scope: step results are text deliverables, returned to the caller and
recorded on the goal. A step that cannot be routed fails the step (not the
goal) — the plan stays visible and re-runnable.
"""

from __future__ import annotations

from typing import Any

import structlog

from synapse.contracts import (
    MemoryStore,
    ModelRegistry,
    PerformanceStore,
    Router,
)
from synapse.domain import (
    Capability,
    ChatResponse,
    Decision,
    MemoryScope,
    PrivacyMode,
    RoutingDecision,
    WorkspaceKind,
)
from synapse.domain.enums import WorkspaceKind
from synapse.events import Events
from synapse.projects.system import WorkspaceSystem
from synapse.providers.exceptions import ProviderUnavailable

log = structlog.get_logger("synapse.planner.goal_executor")


def _capability(value: str) -> Capability:
    try:
        return Capability(value)
    except ValueError:
        return Capability.REASONING


class GoalExecutor:
    """Runs one goal's plan, step by step, through the routing contract chain.

    Mirrors MasterAgent's step-level routing (``_decision_for_task`` /
    ``_route_task``) without touching the agent: the goal record is the only
    state, and the workspace system is the only entry point for persistence.
    """

    def __init__(
        self,
        system: WorkspaceSystem,
        router: Router,
        providers: Any,
        registry: ModelRegistry,
        hardware: Any,
        executor: Any,
        events: Any,
        performance: PerformanceStore | None = None,
        lifecycle: Any = None,
        memory: MemoryStore | None = None,
    ) -> None:
        self._system = system
        self._router = router
        self._providers = providers
        self._registry = registry
        self._hardware = hardware
        self._executor = executor
        self._events = events
        self._performance = performance
        self._lifecycle = lifecycle
        self._memory = memory

    # -- public --------------------------------------------------------------

    def execute(self, project_id: str, goal_id: str) -> dict:
        """Execute all pending steps of a goal. Returns a summary dict.

        Already-completed steps are skipped (idempotent re-run); failed steps
        are retried. The goal's progress is updated after every step.
        """
        store = self._system.goal_store(project_id)
        info = store.get(goal_id)
        if info is None:
            return {"goal_id": goal_id, "error": "goal not found", "completed": 0, "failed": 0}
        outcome = info.outcome or info.title
        workspace_type = self._system.workspace_type(project_id)

        completed = failed = 0
        results: list[dict] = []
        for index, step in enumerate(info.steps):
            if step.status == "completed":
                results.append(self._step_summary(index, step.description, "completed", step.model))
                continue
            store.update_step(goal_id, index, status="running")
            summary = self._run_step(project_id, goal_id, index, step.description, step.capability, outcome, workspace_type)
            results.append(summary)
            if summary["status"] == "completed":
                completed += 1
            else:
                failed += 1

        status = "completed" if completed and not failed else "active"
        final = store.get(goal_id)
        return {
            "goal_id": goal_id,
            "project_id": project_id,
            "status": status,
            "progress": final.progress if final else 0,
            "completed": completed,
            "failed": failed,
            "steps": results,
        }

    # -- internals -----------------------------------------------------------

    def _run_step(
        self,
        project_id: str,
        goal_id: str,
        index: int,
        description: str,
        capability: str,
        outcome: str,
        workspace_type: str,
    ) -> dict:
        step_prompt = self._step_prompt(outcome, workspace_type, description, index + 1)
        decision = self._decision_for_step(_capability(capability), step_prompt)
        try:
            routing = self._route_step(decision, step_prompt)
            if routing is None:
                self._mark(project_id, goal_id, index, status="failed")
                self._publish_goal("goal.step.failed", project_id, goal_id, index, description, reason="no route")
                return {"index": index, "description": description, "status": "failed", "reason": "no route"}
        except Exception as exc:  # noqa: BLE001 - routing must never crash the executor
            self._mark(project_id, goal_id, index, status="failed")
            self._publish_goal("goal.step.failed", project_id, goal_id, index, description, reason=str(exc)[:200])
            return {"index": index, "description": description, "status": "failed", "reason": str(exc)[:200]}

        try:
            self._publish_goal("goal.step.running", project_id, goal_id, index, description, model=routing.model_id)
            response = self._execute(routing, step_prompt)
        except ProviderUnavailable as exc:
            self._mark(project_id, goal_id, index, status="failed")
            self._publish_goal("goal.step.failed", project_id, goal_id, index, description, reason=str(exc)[:200])
            return {"index": index, "description": description, "status": "failed", "reason": str(exc)[:200]}

        self._mark(project_id, goal_id, index, status="completed", model=routing.model_id, result=response.content)
        self._save_memory(goal_id, description, response.content)
        self._publish_goal(
            "goal.step.completed", project_id, goal_id, index, description,
            model=routing.model_id, result_length=len(response.content or ""),
        )
        return self._step_summary(index, description, "completed", routing.model_id)

    def _decision_for_step(self, capability: Capability, prompt: str) -> Decision:
        # The plan says the step's focus; the router picks the model.
        # TODO(Phase D): capabilities from GoalStep.capability when the plan
        # carries structured capability hints (goal plans are text today).
        return Decision(
            can_stay_local=True,
            internet_required=False,
            privacy=PrivacyMode.BALANCED,
            required_capabilities=[capability],
            preferred_capabilities=[Capability.CHAT],
            preferred_kind=None,
            workspace=WorkspaceKind.GENERAL,
            reasoning=[f"goal step routed by capability focus: {capability.value}"],
        )

    def _route_step(self, decision: Decision, prompt: str) -> RoutingDecision | None:
        registry_models = self._registry.all()
        health = self._provider_availability()
        available = self._provider_available_models()
        complexity = min(100, max(1, len(prompt) // 50))
        if self._lifecycle is not None:
            reuse = self._lifecycle.find_reuse(
                decision, self._hardware, health, available, complexity=complexity
            )
            if reuse is not None:
                return reuse
        return self._router.route(
            decision,
            self._hardware,
            registry_models,
            health,
            available,
            complexity=complexity,
            prompt=prompt,
            performance=self._performance,
        )

    def _execute(self, routing: RoutingDecision, prompt: str) -> ChatResponse:
        if self._lifecycle is not None:
            self._lifecycle.note_request_started(routing.provider_id, routing.model_id)
        try:
            response = self._executor.execute(routing, prompt, temperature=0.7)
            if self._performance is not None:
                self._performance.record(
                    routing.provider_id, routing.model_id,
                    latency_s=response.metrics.total_latency_s or 0.0,
                    success=True,
                )
            return response
        except ProviderUnavailable:
            if self._performance is not None:
                self._performance.record(routing.provider_id, routing.model_id, latency_s=0.0, success=False)
            raise
        finally:
            if self._lifecycle is not None:
                self._lifecycle.note_request_completed(routing.provider_id, routing.model_id)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _step_prompt(outcome: str, workspace_type: str, description: str, position: int) -> str:
        return (
            f"Goal outcome: {outcome}\n"
            f"Workspace type: {workspace_type}\n"
            f"Plan step {position}: {description}\n\n"
            "Produce the deliverable for this step only. Be concrete and "
            "self-contained; do not refer to steps that have not run yet."
        )

    def _mark(self, project_id: str, goal_id: str, index: int, **kwargs) -> None:
        try:
            self._system.goal_store(project_id).update_step(goal_id, index, **kwargs)
        except Exception:  # noqa: BLE001 - persistence must never crash execution
            log.warning("goal_step_persist_failed", goal_id=goal_id, index=index)

    def _save_memory(self, goal_id: str, description: str, result: str) -> None:
        if self._memory is None:
            return
        try:
            self._memory.save(MemoryScope.PROJECT, f"[goal {goal_id}] {description}", source="goal")
            self._memory.save(MemoryScope.PROJECT, result[:400], source="goal_result", conversation=goal_id)
        except Exception:  # noqa: BLE001 - memory must never break execution
            log.debug("goal_memory_save_failed")

    def _publish_goal(self, kind: str, project_id: str, goal_id: str, index: int, description: str, **extra) -> None:
        try:
            self._events.publish(
                Events.TIMELINE,
                {"kind": kind, "project_id": project_id, "goal_id": goal_id,
                 "step_index": index, "description": description, **extra},
            )
        except Exception:  # noqa: BLE001 - a timeline must never break execution
            log.debug("goal_timeline_publish_failed")

    @staticmethod
    def _step_summary(index: int, description: str, status: str, model: str | None) -> dict:
        return {"index": index, "description": description, "status": status, "model": model}

    def _provider_availability(self) -> dict[str, bool]:
        try:
            kinds = {pid: self._providers.get(pid).kind.value for pid in self._providers.provider_ids()}
            return {"local": "local" in kinds.values(), "cloud": "cloud" in kinds.values()}
        except Exception:  # noqa: BLE001 - unavailable providers never block goal execution
            return {"local": False, "cloud": False}

    def _provider_available_models(self) -> dict[str, set[str] | None]:
        available: dict[str, set[str] | None] = {}
        for provider in self._providers.all():
            try:
                available[provider.provider_id] = {d.id for d in provider.list_models()}
            except Exception:  # noqa: BLE001 - a failing provider never blocks routing
                available[provider.provider_id] = None
        return available
