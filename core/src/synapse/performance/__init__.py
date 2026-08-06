"""Performance history — the Master Agent's learning loop.

Per-model execution statistics are recorded after every request and fed back
into routing (latency prediction, timeout awareness). JSON-file backed,
replaceable behind the PerformanceStore contract.
"""

from synapse.performance.store import FilePerformanceStore

__all__ = ["FilePerformanceStore"]
