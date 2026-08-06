"""Phase 4: Confidence Engine contract.

Every response gets an internal confidence score. Low-confidence responses
trigger additional reasoning or model verification. The assistant can answer
"I don't know" instead of fabricating information.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from synapse.domain.enums import Capability


@dataclass(frozen=True)
class ConfidenceScore:
    """A confidence score with reasoning."""

    score: float  # 0.0 - 1.0
    reason: str
    # Capability-specific breakdown
    breakdown: dict[Capability, float] | None = None


class ConfidenceEngine(ABC):
    """Evaluates and scores response confidence."""

    @abstractmethod
    def score(
        self,
        response: str,
        prompt: str,
        *,
        context: str | None = None,
        retrieved_chunks: list[dict] | None = None,
        model_id: str | None = None,
        capability_requirements: list[Capability] | None = None,
    ) -> ConfidenceScore:
        """Score a response's confidence.

        Args:
            response: The generated response text
            prompt: The original user prompt
            context: Optional context that was provided (RAG, memory, etc.)
            retrieved_chunks: List of retrieved document chunks with metadata
            model_id: The model that generated the response
            capability_requirements: Required capabilities for this task

        Returns:
            ConfidenceScore with score, reason, and capability breakdown
        """

    @abstractmethod
    def should_escalate(self, score: ConfidenceScore, threshold: float = 0.6) -> bool:
        """Whether this score warrants model escalation."""

    @abstractmethod
    def should_verify(self, score: ConfidenceScore, threshold: float = 0.7) -> bool:
        """Whether this score warrants multi-step verification."""

    @abstractmethod
    def should_refuse(self, score: ConfidenceScore, threshold: float = 0.3) -> bool:
        """Whether confidence is so low we should refuse to answer."""