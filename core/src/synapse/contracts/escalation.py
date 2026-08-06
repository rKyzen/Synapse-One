"""Phase 4: Adaptive Model Escalation contract.

If a smaller model has low confidence, automatically retry using a stronger
local model before responding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from synapse.domain.models import ModelMetadata


@dataclass(frozen=True)
class EscalationDecision:
    """Decision about whether to escalate and to which model."""

    should_escalate: bool
    target_model: ModelMetadata | None
    reason: str
    current_score: float
    expected_improvement: float


class EscalationPolicy(ABC):
    """Policy for when and how to escalate to stronger models."""

    @abstractmethod
    def decide(
        self,
        current_model: ModelMetadata,
        confidence_score: float,
        available_models: list[ModelMetadata],
        task_capabilities: list[str],
        hardware_constraints: dict | None = None,
    ) -> EscalationDecision:
        """Decide whether to escalate and to which model.

        Args:
            current_model: The model that produced the low-confidence response
            confidence_score: The confidence score (0.0 - 1.0)
            available_models: All available models sorted by capability
            task_capabilities: Required capabilities for this task
            hardware_constraints: RAM/VRAM limits

        Returns:
            EscalationDecision with target model and reasoning
        """


class EscalationEngine(ABC):
    """Executes model escalation when confidence is low."""

    @abstractmethod
    def escalate(
        self,
        prompt: str,
        current_response: str,
        decision: EscalationDecision,
        *,
        context: str | None = None,
        temperature: float = 0.3,
    ) -> tuple[str, float]:
        """Execute escalation to a stronger model.

        Args:
            prompt: Original prompt
            current_response: The low-confidence response
            decision: Escalation decision with target model
            context: Context to provide to the stronger model
            temperature: Temperature for the escalation call

        Returns:
            Tuple of (new_response, new_confidence)
        """