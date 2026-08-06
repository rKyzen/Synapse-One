"""Aggregate model-lifecycle metrics for reporting and introspection.

Reads per-model state from :class:`LoadedModel` records and folds counters
into system-wide aggregates. Pure Python; no I/O.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


class ModelLifecycleMetrics:
    """Collects and reports aggregate lifecycle metrics."""

    def __init__(self) -> None:
        self._model_metrics: dict[str, dict[str, Any]] = {}
        self._system: dict[str, Any] = {
            "total_loads": 0,
            "total_unloads": 0,
            "total_reuses": 0,
            "total_failures": 0,
            "peak_simultaneous_models": 0,
            "current_loaded_models": 0,
            "cleanup_cycles": 0,
            "memory_pressure_events": 0,
            "average_load_time_s": 0.0,
            "average_unload_time_s": 0.0,
            "average_idle_lifetime_s": 0.0,
            "average_ram_reclaimed_gb": 0.0,
            "system_ram_gb": 0.0,
        }
        self._cleanup_times: list[float] = []
        self._start = datetime.now()

    # -- per-model updates (called by LoadedModel / manager) -------------------

    def update_from_model(self, model) -> None:
        """Refresh the aggregate view for one model from its LoadedModel record."""
        key = f"{model.provider_id}/{model.model_id}"
        self._model_metrics[key] = {
            "model_id": model.model_id,
            "provider_id": model.provider_id,
            "state": model.state.value,
            "load_count": model.load_count,
            "unload_count": model.unload_count,
            "reuse_count": model.reuse_count,
            "failure_count": model.failure_count,
            "total_idle_s": round(model.total_idle_s, 3),
            "total_load_s": round(model.total_load_s, 3),
            "total_unload_s": round(model.total_unload_s, 3),
            "total_ram_reclaimed_gb": round(model.total_ram_reclaimed_gb, 3),
            "peak_ram_gb": round(model.ram_gb, 3) if model.ram_gb is not None else None,
            "loaded": model.is_loaded,
            "idle_s": round(model.idle_duration_s(), 1),
        }

    # -- system-wide counters --------------------------------------------------

    def record_load(self) -> None:
        self._system["total_loads"] += 1

    def record_unload(self) -> None:
        self._system["total_unloads"] += 1

    def record_reuse(self) -> None:
        self._system["total_reuses"] += 1

    def record_failure(self) -> None:
        self._system["total_failures"] += 1

    def record_memory_pressure(self) -> None:
        self._system["memory_pressure_events"] += 1

    def record_cleanup(self, duration_s: float, current_loaded: int) -> None:
        self._system["cleanup_cycles"] += 1
        self._system["current_loaded_models"] = current_loaded
        self._system["peak_simultaneous_models"] = max(
            self._system["peak_simultaneous_models"], current_loaded
        )
        self._cleanup_times.append(duration_s)

    def update_system_ram(self, ram_gb: float) -> None:
        self._system["system_ram_gb"] = round(ram_gb, 2)

    def _recompute_averages(self) -> None:
        loads = self._system["total_loads"]
        unloads = self._system["total_unloads"]

        if loads:
            total = sum(m["total_load_s"] for m in self._model_metrics.values())
            self._system["average_load_time_s"] = round(total / loads, 3)

        if unloads:
            total = sum(m["total_unload_s"] for m in self._model_metrics.values())
            self._system["average_unload_time_s"] = round(total / unloads, 3)

        if loads:
            total = sum(m["total_idle_s"] for m in self._model_metrics.values())
            self._system["average_idle_lifetime_s"] = round(total / loads, 3)

        if unloads:
            total = sum(m["total_ram_reclaimed_gb"] for m in self._model_metrics.values())
            self._system["average_ram_reclaimed_gb"] = round(total / unloads, 3)

    # -- queries --------------------------------------------------------------

    def get_model_metrics(self, model_id: str, provider_id: str) -> dict[str, Any] | None:
        return self._model_metrics.get(f"{provider_id}/{model_id}")

    def get_all_model_metrics(self) -> list[dict[str, Any]]:
        return list(self._model_metrics.values())

    def get_system_metrics(self) -> dict[str, Any]:
        self._recompute_averages()
        out = dict(self._system)
        return out

    def get_cleanup_stats(self) -> dict[str, Any]:
        return {
            "total_cleanup_cycles": self._system["cleanup_cycles"],
            "total_memory_pressure_events": self._system["memory_pressure_events"],
            "average_cleanup_time_s": round(sum(self._cleanup_times) / len(self._cleanup_times), 3)
            if self._cleanup_times else 0.0,
            "peak_cleanup_time_s": round(max(self._cleanup_times), 3) if self._cleanup_times else 0.0,
        }

    def get_loaded_model_keys(self) -> list[str]:
        return [key for key, m in self._model_metrics.items() if m["loaded"]]

    def reset(self) -> None:
        self._model_metrics.clear()
        self._cleanup_times.clear()
        for k in self._system:
            self._system[k] = 0
        self._system["average_load_time_s"] = 0.0
        self._system["average_unload_time_s"] = 0.0
        self._system["average_idle_lifetime_s"] = 0.0
        self._system["average_ram_reclaimed_gb"] = 0.0
        self._start = datetime.now()

    def get_session_duration_s(self) -> float:
        return (datetime.now() - self._start).total_seconds()
