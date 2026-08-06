"""Router contract — vendor-agnostic model selection."""

from __future__ import annotations

from abc import ABC, abstractmethod

from synapse.domain.diagnosis import Decision, RoutingDecision
from synapse.domain.hardware import HardwareProfile


class Router(ABC):
    @abstractmethod
    def route(
        self,
        decision: Decision,
        hardware: HardwareProfile,
        registry: list[object],
        provider_health: dict[str, bool],
        available_models: dict[str, set[str]] | None = None,
        *,
        complexity: int | None = None,
        prompt: str | None = None,
        performance: dict[str, dict] | None = None,
    ) -> RoutingDecision:
        """Choose a provider+model, with confidence and a human-readable reason.

        ``registry`` is a list of ModelMetadata; ``provider_health`` maps
        provider_id -> healthy bool; ``available_models`` maps provider_id ->
        set of model ids the provider actually reports as installed (None =
        no availability info, gate skipped). Must remain vendor-agnostic.

        Phase 2.5 additions (all optional, keyword-only):
            ``complexity``   — 0..100 task difficulty (drives expected output
                                size and deep-reasoning bonuses).
            ``prompt``       — raw prompt (prompt length feeds latency estimates).
            ``performance``  — ``provider/model -> stats`` history produced by a
                                PerformanceStore, used for latency prediction.
        """
