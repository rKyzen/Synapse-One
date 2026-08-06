"""JSON-file-backed PerformanceStore.

Statistics are persisted under ``<home>/data/performance.json`` and survive
restarts. All writes are serialized by a lock; the store never raises on I/O
errors (a broken/absent history file must never break routing).
"""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path

import structlog

from synapse.config.paths import SynapsePaths
from synapse.contracts import PerformanceStore
from synapse.events import EventBus, Events

log = structlog.get_logger("synapse.performance")


class FilePerformanceStore(PerformanceStore):
    """Persistent per-model execution statistics."""

    def __init__(
        self,
        paths: SynapsePaths,
        *,
        enabled: bool = True,
        max_entries: int = 500,
        events: EventBus | None = None,
    ) -> None:
        self._path = paths.data_dir / "performance.json"
        self._enabled = enabled
        self._max_entries = max_entries
        self._events = events
        self._data: dict[str, dict] = {}
        self._lock = threading.Lock()
        if enabled:
            self._load()

    # -- contract -----------------------------------------------------------

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
        if not self._enabled:
            return
        with self._lock:
            key = f"{provider_id}/{model_id}"
            entry = self._data.setdefault(
                key,
                {
                    "samples": 0,
                    "successes": 0,
                    "failures": 0,
                    "timeouts": 0,
                    "avg_latency_s": 0.0,
                    "avg_tokens_per_second": None,
                },
            )
            n = entry["samples"] = entry["samples"] + 1
            entry["avg_latency_s"] = round(
                (entry.get("avg_latency_s", 0.0) * (n - 1) + max(0.0, latency_s)) / n,
                3,
            )
            if tokens_per_s:
                prev = entry.get("avg_tokens_per_second") or 0.0
                entry["avg_tokens_per_second"] = round((prev * (n - 1) + tokens_per_s) / n, 3)
            if success:
                entry["successes"] = entry.get("successes", 0) + 1
            else:
                entry["failures"] = entry.get("failures", 0) + 1
            if interrupted:
                entry["timeouts"] = entry.get("timeouts", 0) + 1
            entry["success_rate"] = round(entry.get("successes", 0) / n, 3)
            entry["timeout_rate"] = round(entry.get("timeouts", 0) / n, 3)
            entry["failure_rate"] = round(entry.get("failures", 0) / n, 3)
            if self._events:
                self._events.publish(
                    Events.PERFORMANCE_RECORDED,
                    {"key": key, "success": success, "latency_s": round(latency_s, 3)},
                )
            self._trim()
            self._save()

    def stats(self) -> dict[str, dict]:
        with self._lock:
            return deepcopy(self._data)

    # -- internals ----------------------------------------------------------

    def _trim(self) -> None:
        if self._max_entries <= 0 or len(self._data) <= self._max_entries:
            return
        for key in list(self._data)[: len(self._data) - self._max_entries]:
            del self._data[key]

    def _load(self) -> None:
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._data = {}
                for key, entry in raw.items():
                    if not isinstance(entry, dict):
                        continue
                    samples = int(entry.get("samples", 0) or 0)
                    successes = int(entry.get("successes", 0) or 0)
                    failures = int(entry.get("failures", 0) or 0)
                    timeouts = int(entry.get("timeouts", 0) or 0)
                    n = max(samples, 1)
                    self._data[key] = {
                        "samples": samples,
                        "successes": successes,
                        "failures": failures,
                        "timeouts": timeouts,
                        "avg_latency_s": entry.get("avg_latency_s", 0.0),
                        "avg_tokens_per_second": entry.get("avg_tokens_per_second"),
                        "success_rate": round(successes / n, 3),
                        "timeout_rate": round(timeouts / n, 3),
                        "failure_rate": round(failures / n, 3),
                    }
        except Exception:  # noqa: BLE001 - broken history never blocks routing
            log.warning("performance_store_load_failed", path=str(self._path))
            self._data = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
            tmp.replace(self._path)
        except Exception:  # noqa: BLE001
            log.warning("performance_store_save_failed", path=str(self._path))

    @property
    def path(self) -> Path:
        return self._path
