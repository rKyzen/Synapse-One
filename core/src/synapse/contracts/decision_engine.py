"""Decision Engine contract — the heart of the intelligent pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

from synapse.domain.diagnosis import ComplexityResult, Decision, IntentResult, PrivacyResult
from synapse.domain.hardware import HardwareProfile
from synapse.domain.models import ModelMetadata


class DecisionEngine(ABC):
    @abstractmethod
    def decide(
        self,
        intent: IntentResult,
        complexity: ComplexityResult,
        privacy: PrivacyResult,
        hardware: HardwareProfile,
        registry: list[ModelMetadata],
    ) -> Decision:
        """Answer: can this stay local? should cloud be used? which
        capabilities are required? which workspace? which provider kind?"""