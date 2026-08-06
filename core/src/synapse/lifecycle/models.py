"""Model lifecycle domain: states, per-model tracking records, settings.

These are plain data structures. The manager operates on them, and the metrics
module aggregates them. Nothing here performs I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ModelState(str, Enum):
    """Lifecycle state machine for a model (provider-side load state)."""

    OFFLINE = "offline"  # not loaded (or not present)
    LOADING = "loading"  # load in progress
    ACTIVE = "active"  # currently generating a response
    IDLE = "idle"  # loaded, waiting, eligible for unload
    UNLOADING = "unloading"  # unload in progress
    FAILED = "failed"  # load/unload failed


@dataclass
class LifecycleSettings:
    """Configurable lifecycle policy (mirrors ``[lifecycle]`` in config.toml)."""

    idle_timeout_small_s: float = 60.0
    idle_timeout_large_s: float = 30.0
    idle_timeout_embedding_s: float = float("inf")  # never auto-unload
    cleanup_interval_s: float = 15.0
    keep_embedding_loaded: bool = True
    low_memory_threshold_gb: float = 4.0
    prefer_loaded_model_margin: float = 0.05  # relative capability tolerance
    large_model_min_ram_gb: float = 8.0
    max_loaded_models: int = 4
    enabled: bool = True

    @classmethod
    def from_config(cls, raw: dict | None) -> "LifecycleSettings":
        """Build from a raw config dict; missing keys use defaults."""
        raw = raw or {}
        try:
            return cls(
                idle_timeout_small_s=float(raw.get("idle_timeout_small", 60.0)),
                idle_timeout_large_s=float(raw.get("idle_timeout_large", 30.0)),
                idle_timeout_embedding_s=float(raw.get("idle_timeout_embedding", 1e18)),
                cleanup_interval_s=float(raw.get("cleanup_interval", 15.0)),
                keep_embedding_loaded=bool(raw.get("keep_embedding_loaded", True)),
                low_memory_threshold_gb=float(raw.get("low_memory_threshold", 4.0)),
                prefer_loaded_model_margin=float(raw.get("prefer_loaded_model_margin", 0.05)),
                large_model_min_ram_gb=float(raw.get("large_model_min_ram_gb", 8.0)),
                max_loaded_models=int(raw.get("max_loaded_models", 4)),
                enabled=bool(raw.get("enabled", True)),
            )
        except (TypeError, ValueError):
            return cls()


@dataclass
class LoadedModel:
    """Runtime tracking record for one model on one provider."""

    model_id: str
    provider_id: str
    display_name: str = ""
    # classification used to pick the right idle timeout
    is_embedding: bool = False
    is_large: bool = False  # high-RAM model -> shorter idle timeout
    required_ram_gb: float = 0.0

    state: ModelState = ModelState.OFFLINE
    loaded_at: datetime | None = None
    idle_since: datetime | None = None
    last_used_at: datetime | None = None
    unload_reason: str | None = None

    # runtime stats sampled from the provider (Ollama /api/ps)
    ram_gb: float | None = None

    # metrics accumulated over the model's lifetime on this manager
    load_count: int = 0
    unload_count: int = 0
    reuse_count: int = 0
    failure_count: int = 0
    total_idle_s: float = 0.0
    total_load_s: float = 0.0
    total_unload_s: float = 0.0
    total_ram_reclaimed_gb: float = 0.0

    def mark_loaded(self, ram_gb: float | None, now: datetime | None = None) -> None:
        self.loaded_at = now or datetime.now()
        self.state = ModelState.IDLE
        self.idle_since = self.loaded_at
        self.last_used_at = self.loaded_at
        self.ram_gb = ram_gb
        self.load_count += 1

    def mark_active(self, now: datetime | None = None) -> None:
        now = now or datetime.now()
        if self.state == ModelState.IDLE and self.idle_since is not None:
            self.total_idle_s += (now - self.idle_since).total_seconds()
        self.state = ModelState.ACTIVE
        self.last_used_at = now
        self.idle_since = None

    def mark_idle(self, now: datetime | None = None) -> None:
        now = now or datetime.now()
        if self.state == ModelState.ACTIVE:
            self.state = ModelState.IDLE
            self.idle_since = now
            self.last_used_at = now

    def mark_unloaded(self, ram_gb_reclaimed: float | None, duration_s: float, now: datetime | None = None) -> None:
        now = now or datetime.now()
        self.state = ModelState.OFFLINE
        self.unload_count += 1
        self.total_unload_s += duration_s
        if ram_gb_reclaimed is not None:
            self.total_ram_reclaimed_gb += ram_gb_reclaimed
        self.idle_since = None
        self.loaded_at = None

    def mark_unload_start(self, now: datetime | None = None) -> None:
        self.state = ModelState.UNLOADING

    def mark_load_start(self, now: datetime | None = None) -> None:
        self.state = ModelState.LOADING

    def mark_failed(self, now: datetime | None = None) -> None:
        self.state = ModelState.FAILED
        self.failure_count += 1

    def idle_duration_s(self, now: datetime | None = None) -> float:
        if self.state != ModelState.IDLE or self.idle_since is None:
            return 0.0
        now = now or datetime.now()
        return (now - self.idle_since).total_seconds()


    @property
    def is_loaded(self) -> bool:
        return self.state in (ModelState.ACTIVE, ModelState.IDLE, ModelState.LOADING, ModelState.UNLOADING)

    @property
    def is_generating(self) -> bool:
        return self.state == ModelState.ACTIVE

    def can_unload_now(self) -> bool:
        """Eligible for automated cleanup: idle and not in a protected state."""
        return self.state == ModelState.IDLE

    def capability_summary(self) -> dict:
        return {
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "state": self.state.value,
            "idle_s": round(self.idle_duration_s(), 1),
            "ram_gb": self.ram_gb,
        }
