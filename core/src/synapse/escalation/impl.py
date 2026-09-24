"""Adaptive Model Escalation — Phase 4.

When a small model's confidence is low, the escalation engine re-asks a
stronger local model before answering. The policy picks the target model;
the engine executes the retry through the provider abstraction so no vendor
knowledge leaks here.
"""

from __future__ import annotations

import structlog

from synapse.contracts.escalation import EscalationDecision, EscalationEngine, EscalationPolicy
from synapse.domain.enums import Capability, ProviderKind
from synapse.domain.models import ModelMetadata
from synapse.domain.requests import ChatMessage, ChatRequest
from synapse.providers.manager import ProviderManager

log = structlog.get_logger("synapse.escalation")

#: Capability strength required before a model is a valid escalation target.
_MIN_CAPABILITY = 0.55


class CapabilityBasedEscalationPolicy(EscalationPolicy):
    """Escalates to the strongest local model that covers the task and fits
    the hardware budget — but only when the current model was clearly weak."""

    def decide(
        self,
        current_model: ModelMetadata,
        confidence_score: float,
        available_models: list[ModelMetadata],
        task_capabilities: list[str],
        hardware_constraints: dict | None = None,
    ) -> EscalationDecision:
        if confidence_score >= 0.6:
            return EscalationDecision(
                should_escalate=False,
                target_model=None,
                reason=f"confidence {confidence_score:.2f} is acceptable",
                current_score=confidence_score,
                expected_improvement=0.0,
            )

        ram_budget = (hardware_constraints or {}).get("ram_gb", 64.0)
        capability_enum = {c.value: c for c in Capability}
        required: list[Capability] = [
            capability_enum[c] for c in task_capabilities if c in capability_enum
        ] or [Capability.REASONING]

        candidates: list[ModelMetadata] = []
        for model in available_models:
            if model.id == current_model.id:
                continue
            if model.kind != ProviderKind.LOCAL:
                continue
            if model.required_ram_gb > ram_budget:
                continue
            if not all(model.capabilities.score_for(cap) >= _MIN_CAPABILITY for cap in required):
                continue
            candidates.append(model)

        if not candidates:
            return EscalationDecision(
                should_escalate=False,
                target_model=None,
                reason="no stronger local model available",
                current_score=confidence_score,
                expected_improvement=0.0,
            )

        target = max(candidates, key=self._strength)
        current_strength = self._strength(current_model)
        improvement = max(0.0, (self._strength(target) - current_strength))
        if improvement < 0.05:
            return EscalationDecision(
                should_escalate=False,
                target_model=None,
                reason=f"{target.id} is not meaningfully stronger",
                current_score=confidence_score,
                expected_improvement=0.0,
            )

        return EscalationDecision(
            should_escalate=True,
            target_model=target,
            reason=f"low confidence {confidence_score:.2f}; escalating {current_model.id} -> {target.id}",
            current_score=confidence_score,
            expected_improvement=round(improvement, 3),
        )

    @staticmethod
    def _strength(model: ModelMetadata) -> float:
        caps = model.capabilities
        return (
            caps.reasoning
            + caps.coding
            + caps.math
            + caps.chat
            + caps.planning
            + caps.debugging
            + caps.architecture
        ) / 7.0


class OllamaEscalationEngine(EscalationEngine):
    """Re-runs the request on a stronger local model via ProviderManager."""

    def __init__(self, providers: ProviderManager, policy: EscalationPolicy | None = None) -> None:
        self._providers = providers
        self._policy = policy or CapabilityBasedEscalationPolicy()

    def escalate(
        self,
        prompt: str,
        current_response: str,
        decision: EscalationDecision,
        *,
        context: str | None = None,
        temperature: float = 0.3,
    ) -> tuple[str, float]:
        if not decision.should_escalate or decision.target_model is None:
            return current_response, decision.current_score

        target = decision.target_model
        provider = self._providers.get(target.provider_id)
        if provider is None:
            log.warning("escalation_provider_missing", provider_id=target.provider_id)
            return current_response, decision.current_score

        ctx_str = ""
        if context:
            if isinstance(context, list):
                ctx_str = "\n".join(
                    c.text if hasattr(c, "text") else (c.get("text", "") if isinstance(c, dict) else str(c))
                    for c in context
                )
            else:
                ctx_str = str(context)

        messages = [
            ChatMessage(role="user", content=prompt),
            ChatMessage(
                role="assistant",
                content=current_response,
            ),
            ChatMessage(
                role="user",
                content=(
                    "Your colleague gave the answer above with low confidence. "
                    + (f"Context:\n{ctx_str}\n\n" if ctx_str else "")
                    + "Please provide a corrected, more accurate answer. "
                    "Start your answer directly, without meta-commentary."
                ),
            ),
        ]
        try:
            response = provider.chat(
                ChatRequest(messages=messages, temperature=temperature, model=target.id)
            )
            new_text = response.content.strip()
            if not new_text:
                log.warning("escalation_empty_response", model=target.id)
                return current_response, decision.current_score
            # Escalated answers are treated as strongly validated.
            return new_text, max(0.85, decision.current_score + 0.2)
        except Exception as exc:  # noqa: BLE001 - escalation must never crash the pipeline
            log.warning("escalation_failed", model=target.id, error=str(exc)[:200])
            return current_response, decision.current_score
