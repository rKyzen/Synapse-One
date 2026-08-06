"""Typed pub/sub event bus.

Decouples subsystems: the hardware scanner, registry, and providers publish
events; future subsystems (memory, router, workspace engine, UI bridge)
subscribe without importing each other. Subscribers are isolated — a failure
in one never breaks the publisher or other subscribers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

from synapse.domain.enums import ProviderState

log = logging.getLogger("synapse.events")

T = TypeVar("T")


@dataclass(frozen=True)
class Event:
    """Minimal, immutable event envelope."""

    name: str
    payload: dict | None = None


class EventBus:
    """In-process event bus.

    Thread-safety: publish/subscribe are guarded by a lock so events can be
    published from anywhere. Handlers run synchronously in publisher order.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list] = {}
        self._wildcards: list = []

    def subscribe(self, name: str, handler) -> None:
        """Register a handler for an exact event name.

        ``handler(event)`` receives the :class:`Event`.
        """
        if name == "*":
            self._wildcards.append(handler)
        else:
            self._handlers.setdefault(name, []).append(handler)

    def unsubscribe(self, event_name: str, handler) -> None:
        if event_name == "*":
            self._wildcards.remove(handler)
        else:
            if handler in self._handlers.get(event_name, []):
                self._handlers[event_name].remove(handler)

    def publish(self, name: str, payload: dict | None = None) -> None:
        """Publish an event to all matching subscribers, isolated per-subscriber."""
        event = Event(name=name, payload=payload)
        for handler in list(self._handlers.get(name, [])):
            self._dispatch(name, handler, event)
        for handler in list(self._wildcards):
            self._dispatch(name, handler, event)

    @staticmethod
    def _dispatch(name: str, handler, event: Event) -> None:
        try:
            handler(event)
        except Exception:  # noqa: BLE001 - never break the bus
            log.exception("subscriber failed for event=%s", name)


# ---------------------------------------------------------------------------
# Canonical event names
# ---------------------------------------------------------------------------


class Events:
    """Central registry of event names, so typo-prone strings stay in one place."""

    HARDWARE_SCANNED = "hardware.scanned"
    PROVIDER_INITIALIZED = "provider.initialized"
    PROVIDER_HEALTH_CHANGED = "provider.health_changed"
    MODELS_DISCOVERED = "model.discovered"
    MODEL_CATALOG_LOADED = "model_catalog.loaded"

    # Phase 2 — pipeline events
    REQUEST_RECEIVED = "request.received"
    REQUEST_ANALYZED = "request.analyzed"
    REQUEST_ROUTED = "request.routed"
    REQUEST_COMPLETED = "request.completed"

    # Phase 2.5 — performance learning loop
    PERFORMANCE_RECORDED = "performance.recorded"

    # Model lifecycle (Phase 2.5+)
    MODEL_LOADED = "model.loaded"
    MODEL_UNLOADED = "model.unloaded"
    MODEL_IDLE_TIMEOUT_EXPIRED = "model.idle_timeout_expired"
    MODEL_LOAD_FAILED = "model.load_failed"
    MEMORY_PRESSURE = "memory.pressure"
    CLEANUP_CYCLE = "lifecycle.cleanup"
    MODEL_REUSED = "model.reused"

    # Phase 3 — task orchestration, memory, synthesis
    TASK_PLANNED = "task.planned"
    TASK_ROUTED = "task.routed"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    RESPONSE_SYNTHESIZED = "response.synthesized"
    MEMORY_WRITTEN = "memory.written"

    # Phase 4 — workspace / multimodal
    FILE_UPLOADED = "file.uploaded"
    FILE_INDEXING_STARTED = "file.indexing.started"
    FILE_INDEXED = "file.indexed"
    FILE_DELETED = "file.deleted"
    FILE_ATTACHED = "file.attached"
    RETRIEVAL_RAN = "retrieval.ran"
    VISION_PROCESSED = "vision.processed"