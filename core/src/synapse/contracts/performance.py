"""Performance history contract — the Master Agent's learning loop.

The router becomes explainable AND adaptive: per-model execution statistics
(latency, throughput, success/timeout rates) are recorded after every request
and fed back into the next routing decision. Replaceable behind this contract
(JSON file today, SQLite/remote later).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class PerformanceStore(ABC):
    @abstractmethod
    def record(
        self,
        provider_id: str,
        model_id: str,
        *,
        latency_s: float,
        tokens_per_s: float | None = None,
        success: bool = True,
        interrupted: bool = False,
    ) -> None:
        """Record one execution outcome for a provider/model pair.

        ``interrupted`` counts timeouts/early-stops (e.g. generation budget).
        A failed call is recorded with ``success=False`` and latency 0.
        """

    @abstractmethod
    def stats(self) -> dict[str, dict]:
        """Return per-model statistics keyed by ``"provider/model"``.

        Each entry carries: samples, avg_latency_s, avg_tokens_per_second,
        success_rate, timeout_rate.
        """
