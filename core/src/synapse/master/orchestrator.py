"""AI Master Orchestrator — a model-backed Task Planner (Task Divider).

Replaces (behind the same :class:`TaskPlanner` contract) the deterministic
decomposition with a small LLM acting as the Master Agent: it reads the user
prompt, understands the intent, and emits a structured
:class:`TaskDecompositionPlan` (JSON) before any deterministic
``Router.route()`` call happens.

Flow per request:

1. Resolve the hardware tier (``TierResolver``) -> candidate model ids.
2. Pick the first candidate that is registered AND its provider is healthy —
   the "Master AI" role is thus assigned dynamically per machine, so
   commercial distribution never risks an OOM crash.
3. Call the Master AI through the provider abstraction with the JSON schema
   of ``TaskDecompositionPlan`` attached (``ChatRequest.format``) plus a
   system prompt that forbids solving the prompt.
4. Parse + validate with Pydantic; repair obvious fence/extra-noise damage;
   retry once on hallucinated/invalid/timed-out output.
5. Any failure -> graceful fallback (single pass-through task with the raw
   prompt; when an injected fallback planner exists, it is used instead so
   behavior degrades to the current deterministic planner exactly).

No vendor code lives here: everything goes through ProviderManager and the
ModelProvider interface.
"""

from __future__ import annotations

import json
import re
import structlog
from typing import Any

from pydantic import ValidationError

from synapse.config import ConfigProvider
from synapse.contracts import HardwareProvider, ModelRegistry, TaskPlanner
from synapse.domain import (
    Capability,
    ChatMessage,
    ChatRequest,
    ComplexityResult,
    Decision,
    IntentResult,
    PrivacyResult,
)
from synapse.domain.enums import TaskKind
from synapse.domain.tasks import Task, TaskDAG
from synapse.events import EventBus
from synapse.hardware.tier_resolver import HardwareTier, TierAssignment, TierResolver
from synapse.master.schemas import TaskDecompositionPlan
from synapse.providers.manager import ProviderManager

log = structlog.get_logger("synapse.master.orchestrator")

#: system instruction — the Master AI is a router, never a worker.
_DIVIDER_SYSTEM_PROMPT = (
    "You are the Synapse One Master AI Orchestrator. Your ONLY job is to analyze "
    "the user prompt and decompose it into clear, minimal sub-tasks. Assign the "
    "required intent category and specify task dependencies. You are a router, "
    "not a worker. Do not solve the prompt. Output ONLY valid JSON matching the "
    "provided schema."
)

#: fallback search order when the assigned tier has no usable model installed:
#: remaining local tiers (ascending size), then the cloud tier.
_LOCAL_TIER_ORDER = (HardwareTier.TIER1, HardwareTier.TIER2, HardwareTier.TIER3)

#: strip ```json ... ``` fences and any prose around the JSON object.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort repair: pull the first balanced JSON object out of ``text``.

    Handles fences, leading prose, trailing prose and stray backticks — the
    common failure modes of instruction-following models. Raises ValueError
    when no JSON object can be found.
    """
    cleaned = _FENCE_RE.sub("", text or "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(cleaned[start : end + 1])


class MasterModelSelection:
    """Which provider/model serves the Master AI role for one request."""

    __slots__ = ("provider_id", "model_id", "tier", "reason")

    def __init__(self, provider_id: str, model_id: str, tier: HardwareTier, reason: str) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        self.tier = tier
        self.reason = reason


class AIMasterOrchestrator(TaskPlanner):
    """Model-backed front planner; deterministic fallback behind it.

    Composes with any other TaskPlanner: the injected ``fallback_planner``
    receives the whole prompt untouched whenever the Master AI cannot produce
    a valid plan (and when disabled).
    """

    def __init__(
        self,
        *,
        providers: ProviderManager,
        registry: ModelRegistry,
        hardware: HardwareProvider,
        config: ConfigProvider,
        resolver: TierResolver | None = None,
        fallback_planner: TaskPlanner | None = None,
        enabled: bool = True,
        temperature: float = 0.1,
        max_retries: int = 1,
        max_tasks: int = 8,
        events: EventBus | None = None,
    ) -> None:
        self._providers = providers
        self._registry = registry
        self._hardware = hardware
        self._config = config
        self._resolver = resolver or TierResolver(config)
        self._fallback = fallback_planner
        self._enabled = enabled
        self._temperature = temperature
        self._max_retries = max(0, int(max_retries))
        self._max_tasks = max(1, int(max_tasks))
        self._events = events

        # observability for the last plan() call.
        self.last_tier: HardwareTier | None = None
        self.last_model: str | None = None
        self.used_ai: bool = False

    # -- contract -----------------------------------------------------------

    def plan(
        self,
        prompt: str,
        intent: IntentResult,
        complexity: ComplexityResult,
        privacy: PrivacyResult,
        decision: Decision,
    ) -> TaskDAG:
        self.used_ai = False
        self.last_tier = None
        self.last_model = None
        if not self._enabled:
            return self._fallback_plan(prompt, intent, complexity, privacy, decision, "disabled")

        selection = self._select_master_model()
        if selection is None:
            return self._fallback_plan(
                prompt, intent, complexity, privacy, decision,
                "no master model available on this machine",
            )
        self.last_tier = selection.tier
        self.last_model = selection.model_id
        log.info(
            "ai_master_model_selected",
            tier=selection.tier.value,
            provider=selection.provider_id,
            model=selection.model_id,
            reason=selection.reason,
        )

        plan: TaskDecompositionPlan | None = None
        failures: list[str] = []
        for attempt in range(self._max_retries + 1):
            try:
                raw = self._call_master(selection, prompt)
                plan = self._to_plan(raw, prompt)
                break
            except Exception as exc:  # noqa: BLE001 - any failure falls back
                failures.append(str(exc)[:200])
                log.warning(
                    "ai_master_plan_failed",
                    attempt=attempt,
                    provider=selection.provider_id,
                    model=selection.model_id,
                    error=str(exc)[:200],
                )

        if plan is None:
            log.warning("ai_master_fallback", prompt_length=len(prompt), failures=failures)
            return self._fallback_plan(
                prompt, intent, complexity, privacy, decision,
                "master ai failed to produce a valid plan",
            )

        dag = plan.to_dag()
        self.used_ai = True
        log.info(
            "ai_master_plan_ok",
            tasks=len([t for t in dag.tasks if t.kind.value != "synthesis"]),
            strategy=plan.execution_strategy.value,
            provider=selection.provider_id,
            model=selection.model_id,
        )
        return dag

    # -- the AI call --------------------------------------------------------

    def _call_master(self, selection: MasterModelSelection, prompt: str) -> str:
        provider = self._providers.get(selection.provider_id)
        if provider is None:
            raise RuntimeError(f"master provider '{selection.provider_id}' not registered")
        request = ChatRequest(
            messages=[
                ChatMessage(role="system", content=_DIVIDER_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=(
                        "Decompose the following user prompt into a JSON task graph. "
                        f"Schema: {json.dumps(TaskDecompositionPlan.model_json_schema())}\n\n"
                        f"USER PROMPT:\n{prompt}"
                    ),
                ),
            ],
            temperature=self._temperature,
            model=selection.model_id,
            #: Ollama structured outputs: the provider enforces the schema on
            #: the emitted tokens. Providers without format support ignore it
            #: and the Pydantic validation below still guards the output.
            format=TaskDecompositionPlan.model_json_schema(),
        )
        response = provider.chat(request)
        content = (response.content or "").strip()
        if not content:
            raise RuntimeError("master provider returned empty output")
        return content

    def _to_plan(self, raw: str, prompt: str) -> TaskDecompositionPlan:
        """Parse + validate raw model output into a TaskDecompositionPlan."""
        obj = extract_json_object(raw)
        tasks = obj.get("tasks")
        if isinstance(tasks, list) and len(tasks) > self._max_tasks:
            kept = tasks[: self._max_tasks]
            kept_ids = {t.get("task_id") for t in kept if isinstance(t, dict)}
            for t in kept:
                if isinstance(t, dict):
                    deps = t.get("dependencies")
                    if isinstance(deps, list):
                        #: dependencies on dropped tasks are removed so the
                        #: capped DAG stays valid.
                        t["dependencies"] = [d for d in deps if d in kept_ids]
            obj = {**obj, "tasks": kept}
        plan = TaskDecompositionPlan.model_validate(obj)
        if not plan.tasks:
            raise ValueError("decomposition plan contains no tasks")
        return plan

    # -- model selection -----------------------------------------------------

    def _select_master_model(self) -> MasterModelSelection | None:
        """Pick the first usable candidate: assigned tier, then escalation."""
        try:
            hardware = self._hardware.scan()
        except Exception:  # noqa: BLE001 - a broken scanner must not break planning
            log.warning("ai_master_hardware_scan_failed")
            return None
        assignment = self._resolver.resolve(hardware)
        registry = self._registry.all()

        tiers = self._tier_search_order(assignment)
        health_cache: dict[str, bool] = {}
        for tier in tiers:
            for model_id in self._resolver.candidates_for(tier):
                meta = next((m for m in registry if m.id == model_id), None)
                if meta is None:
                    continue
                pid = meta.provider_id
                if pid not in health_cache:
                    health_cache[pid] = self._providers.health(pid)
                if not health_cache[pid]:
                    continue
                return MasterModelSelection(
                    provider_id=pid,
                    model_id=model_id,
                    tier=tier,
                    reason=f"hardware tier {assignment.tier.value}: {assignment.reason}",
                )
        log.warning(
            "ai_master_no_candidate",
            assigned_tier=assignment.tier.value,
            registry_count=len(registry),
        )
        return None

    @staticmethod
    def _tier_search_order(assignment: TierAssignment) -> tuple[HardwareTier, ...]:
        """Assigned tier first (excl. cloud), then ascending local, then cloud."""
        assigned = assignment.tier
        if assigned is HardwareTier.CLOUD_FALLBACK:
            return (HardwareTier.CLOUD_FALLBACK,)
        return (
            assigned,
            *(t for t in _LOCAL_TIER_ORDER if t is not assigned),
            HardwareTier.CLOUD_FALLBACK,
        )

    # -- fallback -------------------------------------------------------------

    def _fallback_plan(
        self,
        prompt: str,
        intent: IntentResult,
        complexity: ComplexityResult,
        privacy: PrivacyResult,
        decision: Decision,
        reason: str,
    ) -> TaskDAG:
        if self._fallback is not None:
            log.info("ai_master_fallback_planner", reason=reason)
            return self._fallback.plan(prompt, intent, complexity, privacy, decision)
        log.info("ai_master_fallback_single", reason=reason)
        return TaskDAG(
            tasks=[
                Task(
                    id="t1",
                    kind=self._fallback_kind(prompt),
                    description=prompt,
                    required_capabilities=list(decision.required_capabilities or [Capability.CHAT]),
                    preferred_capabilities=list(decision.preferred_capabilities),
                )
            ]
        )

    @staticmethod
    def _fallback_kind(prompt: str) -> TaskKind:
        """Smallest deterministic kind guess for the pass-through fallback."""
        from synapse.planner.heuristic import _KIND_HINTS  # noqa: PLC0415

        lowered = prompt.lower()
        for kind, hints in _KIND_HINTS:
            if any(hint in lowered for hint in hints):
                return kind
        return TaskKind.GENERAL