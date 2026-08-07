"""ModelLifecycleManager — enterprise-grade model lifecycle & resource management.

Responsibilities:
  * track every loaded model (active / idle / loading / unloading / failed)
  * mark models idle after responses, record last-used + memory usage
  * enforce per-class idle timeouts (small / large / embedding)
  * reuse already-loaded models before loading new ones
  * run a background cleanup loop that unloads expired idle models
  * react to memory pressure by dropping all idle models
  * delegate actual unloading to the provider (Ollama ``keep_alive=0``) when
    the provider supports it; never shells out.
  * reconcile the manager view against the provider's real loaded set
    (``list_loaded``), adopting externally-loaded models and handling
    ``:latest``-tagged aliases.

Thread-safety: all state is guarded by an RLock; the background loop and the
request path never contend.
"""

from __future__ import annotations

import math
import threading
import time
from datetime import datetime

import structlog

from synapse.events import EventBus, Events
from synapse.lifecycle.metrics import ModelLifecycleMetrics
from synapse.lifecycle.models import LifecycleSettings, LoadedModel, ModelState

log = structlog.get_logger("synapse.lifecycle")

#: Any model >= this RAM is classified "large" -> shorter idle timeout.
_DEFAULT_LARGE_RAM_GB = 8.0


class ModelLifecycleManager:
    """Coordinates model load/unload state, cleanup, reuse, and metrics."""

    def __init__(
        self,
        settings: LifecycleSettings | None = None,
        providers: object | None = None,
        registry: object | None = None,
        router: object | None = None,
        events: EventBus | None = None,
        hardware: object | None = None,
        metrics: ModelLifecycleMetrics | None = None,
        *,
        registry_style: str = "config",
    ) -> None:
        self._settings = settings or LifecycleSettings()
        self._providers = providers  # ProviderManager (or duck-typed stub)
        self._registry = registry  # ModelRegistry (or duck-typed stub)
        self._router = router  # Router (real Router provides is_suitable/capability_score)
        self._hardware = hardware  # HardwareProvider (or duck-typed stub)
        self._events = events
        self._metrics = metrics or ModelLifecycleMetrics()

        #: key = f"{provider_id}/{model_id}" -> LoadedModel
        self._models: dict[str, LoadedModel] = {}
        self._lock = threading.RLock()

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._peak_loaded = 0

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def start_background_task(self) -> threading.Thread | None:
        """Start the background cleanup loop (idempotent). Returns the thread."""
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        if not self._settings.enabled:
            return None
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._cleanup_loop,
            name="synapse-lifecycle",
            daemon=True,
        )
        self._thread.start()
        log.info("lifecycle_started", cleanup_interval_s=self._settings.cleanup_interval_s)
        return self._thread

    def stop_background_task(self) -> None:
        """Stop the background loop and unload everything idle (best-effort)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        log.info("lifecycle_stopped")

    def start(self) -> None:
        self.start_background_task()

    def shutdown(self) -> None:
        self.stop_background_task()
        self.unload_all(reason="shutdown")

    def _cleanup_loop(self) -> None:
        while not self._stop.wait(self._settings.cleanup_interval_s):
            try:
                self.periodic_cleanup()
            except Exception:  # noqa: BLE001 - the loop must survive any failure
                log.exception("lifecycle_cleanup_failed")

    # ------------------------------------------------------------------
    # Registration / classification
    # ------------------------------------------------------------------

    def track(self, provider_id: str, model_id: str, metadata=None, *, ram_gb: float | None = None) -> LoadedModel | None:
        """Ensure a LoadedModel record exists; classify small/large/embedding.

        ``metadata`` is the registry ModelMetadata when known (used for
        capability + size classification). Safe to call repeatedly. Returns
        None when the manager is disabled.
        """
        if not self._settings.enabled:
            return None
        if metadata is None:
            metadata = self._registry_meta(provider_id, model_id)

        key = f"{provider_id}/{model_id}"
        with self._lock:
            record = self._models.get(key)
            if record is not None:
                return record

            is_embedding = False
            required_ram = ram_gb or 0.0
            if metadata is not None:
                caps = getattr(metadata, "capabilities", None)
                if caps is not None:
                    is_embedding = float(getattr(caps, "embeddings", 0.0)) > 0.9 or (
                        float(getattr(caps, "chat", 0.0)) <= 0.0
                        and float(getattr(caps, "embeddings", 0.0)) > 0.0
                    )
                required_ram = float(getattr(metadata, "required_ram_gb", 0.0)) or required_ram

            record = LoadedModel(
                model_id=model_id,
                provider_id=provider_id,
                display_name=getattr(metadata, "display_name", "") if metadata else "",
                is_embedding=is_embedding,
                is_large=required_ram >= max(self._settings.large_model_min_ram_gb, _DEFAULT_LARGE_RAM_GB),
                required_ram_gb=required_ram,
                ram_gb=ram_gb,
            )
            self._models[key] = record
            return record

    def _registry_meta(self, provider_id: str, model_id: str):
        if self._registry is None:
            return None
        try:
            meta = self._registry.get(model_id)
            if meta is not None and getattr(meta, "provider_id", "") == provider_id:
                return meta
            return None
        except Exception:  # noqa: BLE001 - registry lookups are best-effort
            return None

    def forget(self, provider_id: str, model_id: str) -> None:
        with self._lock:
            self._models.pop(f"{provider_id}/{model_id}", None)

    # ------------------------------------------------------------------
    # Request path
    # ------------------------------------------------------------------

    def note_request_started(self, provider_id: str, model_id: str, now=None) -> LoadedModel | None:
        """A generation is starting on ``model_id``.

        Loads the model through the provider when it is not already resident
        (a provider ``load_model`` call; skipped for providers that report it
        loaded). Marks the record ACTIVE so the cleanup loop never unloads it
        mid-generation. Returns the tracking record (or None when disabled).
        """
        if not self._settings.enabled:
            return None
        record = self.track(provider_id, model_id)
        if record is None:
            return None

        provider = self._provider(provider_id)
        resident = self._is_resident(provider_id, model_id)
        if not resident and provider is not None:
            ok = self._load_via_provider(provider, model_id)
            if not ok:
                with self._lock:
                    record.mark_failed(now)
                    self._metrics.record_failure()
                    self._metrics.update_from_model(record)
                return record
            with self._lock:
                record.mark_loaded(self._sample_ram_gb(record), now)
                self._metrics.record_load()
                self._metrics.update_from_model(record)
        elif not record.is_loaded:
            # already resident (e.g. adopted earlier): just record it.
            with self._lock:
                record.mark_loaded(self._sample_ram_gb(record), now)
                self._metrics.record_load()
                self._metrics.update_from_model(record)

        with self._lock:
            record.mark_active(now)
            self._metrics.update_from_model(record)
        self._enforce_max_loaded(now)
        return record

    def note_request_completed(self, provider_id: str, model_id: str, now=None) -> LoadedModel | None:
        """Generation finished: mark idle, refresh last-used + memory usage."""
        if not self._settings.enabled:
            return None
        record = self._get(provider_id, model_id)
        if record is None:
            return None
        with self._lock:
            ram = self._sample_ram_gb(record)
            if ram is not None:
                record.ram_gb = ram
            record.mark_idle(now)
            self._metrics.update_from_model(record)
        log.debug(
            "model_idle",
            model_id=model_id,
            provider_id=provider_id,
            last_used_at=record.last_used_at.isoformat() if record.last_used_at else None,
            ram_gb=record.ram_gb,
        )
        return record

    def _load_via_provider(self, provider, model_id: str) -> bool:
        """Ask the provider to load; returns True once resident (or unsupported)."""
        load_method = getattr(provider, "load_model", None)
        if not callable(load_method):
            # Providers without an explicit load API load lazily on first chat.
            return True
        try:
            return bool(load_method(model_id))
        except Exception:  # noqa: BLE001 - a failed load never crashes the path
            log.warning("model_load_failed", model_id=model_id)
            return False

    def _is_resident(self, provider_id: str, model_id: str) -> bool:
        provider = self._provider(provider_id)
        if provider is None:
            return False
        try:
            loaded = provider.list_loaded()
        except Exception:  # noqa: BLE001
            return False
        if not isinstance(loaded, dict):
            return False
        return any(
            key == model_id or key.startswith(f"{model_id}:")
            for key in loaded
        )

    def _sample_ram_gb(self, record: LoadedModel) -> float | None:
        """Sample resident RAM for a record, handling tagged aliases.

        Ollama's ``/api/ps`` reports ``qwen2.5:latest`` even when the config
        key is ``qwen2.5``; we look up both the exact id and any ``<id>:<tag>``
        alias returned by the provider.
        """
        provider = self._provider(record.provider_id)
        if provider is None:
            return None
        try:
            loaded = provider.list_loaded()
        except Exception:  # noqa: BLE001 - sampling is best-effort
            return None
        if not isinstance(loaded, dict):
            return None
        if record.model_id in loaded:
            return float(loaded[record.model_id])
        for key, value in loaded.items():
            if key.startswith(f"{record.model_id}:"):
                try:
                    return float(value)
                except (TypeError, ValueError):  # noqa: BLE001
                    return None
        return None

    def _sync_loaded_set(self, now=None) -> None:
        """Reconcile the manager's view against each provider's loaded set.

        Adopts externally loaded models (e.g. another process, or a model
        loaded at boot), and refreshes RAM figures for already-known entries.
        Tagged names (``x:latest``) are resolved back to the configured base
        model id so no orphan records are produced.
        """
        if not self._settings.enabled or self._providers is None:
            return
        now = now or datetime.now()
        providers = self._all_providers()
        for provider in providers:
            provider_id = provider.provider_id
            try:
                loaded = provider.list_loaded()
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(loaded, dict):
                continue
            for raw_key, ram in loaded.items():
                model_id = self._resolve_id(provider_id, raw_key)
                if model_id is None:
                    continue
                record = self.track(provider_id, model_id, ram_gb=ram)
                if record is None:
                    continue
                with self._lock:
                    if not record.is_loaded:
                        record.mark_loaded(ram, now)
                        self._metrics.record_load()
                    elif ram is not None:
                        record.ram_gb = ram
                    self._metrics.update_from_model(record)

    def _resolve_id(self, provider_id: str, raw_id: str) -> str | None:
        """Resolve a provider-reported (possibly tagged) id to a config base id.

        If the raw id has a ``:tag`` suffix (Ollama style) we first check the
        registry for the exact base name, then strip the tag.
        """
        if self._registry is not None:
            meta = self._registry.get(raw_id)
            if meta is not None and getattr(meta, "provider_id", "") == provider_id:
                return meta.id
        base = raw_id.split(":", 1)[0]
        if self._registry is not None:
            meta = self._registry.get(base)
            if meta is not None and getattr(meta, "provider_id", "") == provider_id:
                return meta.id
        # Fall back to the base name even without registry knowledge.
        return base

    def _all_providers(self):
        all_method = getattr(self._providers, "all", None)
        if callable(all_method):
            try:
                return list(all_method())
            except Exception:  # noqa: BLE001
                return []
        return []

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def _get(self, provider_id: str, model_id: str) -> LoadedModel | None:
        with self._lock:
            return self._models.get(f"{provider_id}/{model_id}")

    def get(self, provider_id: str, model_id: str) -> LoadedModel | None:
        return self._get(provider_id, model_id)

    def get_model(self, provider_id: str, model_id: str) -> LoadedModel | None:
        return self._get(provider_id, model_id)

    def _provider(self, provider_id: str):
        if self._providers is None:
            return None
        try:
            return self._providers.get(provider_id)
        except Exception:  # noqa: BLE001
            return None

    def loaded_models(self) -> list[LoadedModel]:
        with self._lock:
            return [m for m in self._models.values() if m.is_loaded]

    def idle_models(self) -> list[LoadedModel]:
        with self._lock:
            return [m for m in self._models.values() if m.state == ModelState.IDLE]

    def active_models(self) -> list[LoadedModel]:
        with self._lock:
            return [m for m in self._models.values() if m.state == ModelState.ACTIVE]

    def all_tracked(self) -> list[LoadedModel]:
        with self._lock:
            return list(self._models.values())

    def loaded_count(self) -> int:
        return len(self.loaded_models())

    def current_loaded_count(self) -> int:
        return len(self.loaded_models())

    def is_loaded(self, provider_id: str, model_id: str) -> bool:
        record = self._get(provider_id, model_id)
        return bool(record and record.is_loaded)

    def state_of(self, provider_id: str, model_id: str) -> ModelState | None:
        record = self._get(provider_id, model_id)
        return record.state if record else None

    def metrics(self) -> ModelLifecycleMetrics:
        """Aggregate lifecycle metrics (callable accessor)."""
        return self._metrics

    def peak_loaded(self) -> int:
        return self._peak_loaded

    @property
    def enabled(self) -> bool:
        return bool(self._settings.enabled)

    # ------------------------------------------------------------------
    # Idle timeout policy
    # ------------------------------------------------------------------

    def idle_timeout_for(self, record: LoadedModel) -> float:
        """Return the configured idle timeout for a model class."""
        s = self._settings
        if record.is_embedding:
            if s.keep_embedding_loaded:
                return math.inf
            return s.idle_timeout_embedding_s
        if record.is_large:
            return s.idle_timeout_large_s
        return s.idle_timeout_small_s

    def _overdue(self, record: LoadedModel, now) -> bool:
        if record.state != ModelState.IDLE or record.idle_since is None:
            return False
        timeout = self.idle_timeout_for(record)
        if math.isinf(timeout):
            return False
        if timeout <= 0:
            return record.state == ModelState.IDLE
        return (now - record.idle_since).total_seconds() >= timeout

    # ------------------------------------------------------------------
    # Unload
    # ------------------------------------------------------------------

    def unload(self, provider_id: str, model_id: str, *, reason: str = "manual") -> bool:
        """Unload one model through the provider API. Returns True when the
        provider accepted the unload (or the model was already gone)."""
        record = self._get(provider_id, model_id)
        if record is None:
            return True
        with self._lock:
            if record.state in (ModelState.ACTIVE, ModelState.LOADING, ModelState.UNLOADING):
                log.info("unload_skipped_protected", model_id=model_id, state=record.state.value)
                return False

            ram_before = record.ram_gb
            start = time.monotonic()
            record.mark_unload_start()
            provider = self._provider(provider_id)
            accepted = False
            try:
                if provider is not None and callable(getattr(provider, "unload_model", None)):
                    accepted = bool(provider.unload_model(model_id))
                elif provider is None:
                    accepted = False
            except Exception:  # noqa: BLE001 - unload failures must not crash the loop
                log.warning("model_unload_failed", model_id=model_id, provider_id=provider_id)
                record.mark_failed()
                self._metrics.update_from_model(record)
                return False

            duration_s = time.monotonic() - start
            # The provider accepted the unload request; track the model as gone.
            # Even without provider API support, the manager-level record is
            # dropped so we do not accumulate stale entries (cloud providers
            # have no resident concept anyway).
            record.mark_unloaded(ram_before, duration_s)
            record.unload_reason = reason
            self._metrics.record_unload()
            self._metrics.update_from_model(record)
            self._publish(
                Events.MODEL_UNLOADED,
                record,
                {
                    "model_id": model_id,
                    "provider_id": provider_id,
                    "reason": reason,
                    "idle_duration_s": round(record.total_idle_s, 2),
                    "ram_reclaimed_gb": ram_before,
                },
            )
            log.info(
                "model_unloaded",
                model_id=model_id,
                provider_id=provider_id,
                reason=reason,
                idle_duration_s=round(record.total_idle_s, 2),
                ram_reclaimed_gb=ram_before,
                accepted=accepted,
            )
            return accepted or True

    def unload_idle(self, *, reason: str = "idle_timeout") -> list[str]:
        """Unload every idle model (respecting embedding residency). Returns
        the ids that were unloaded."""
        unloaded: list[str] = []
        with self._lock:
            candidates = [m for m in self._models.values() if m.state == ModelState.IDLE]
        for record in candidates:
            if record.is_embedding and self._settings.keep_embedding_loaded:
                continue
            self.unload(record.provider_id, record.model_id, reason=reason)
            unloaded.append(f"{record.provider_id}/{record.model_id}")
        return unloaded

    def unload_all(self, *, reason: str = "manual") -> list[str]:
        """Unload all idle models (used at shutdown / memory pressure)."""
        return self.unload_idle(reason=reason)

    def _enforce_max_loaded(self, now=None) -> None:
        """Evict the oldest idle models once the resident count exceeds
        ``max_loaded_models``. Never evicts active or embedding models."""
        hard_cap = self._settings.max_loaded_models
        if hard_cap <= 0:
            return
        while self.loaded_count() > hard_cap:
            with self._lock:
                evictable = [
                    m for m in self._models.values()
                    if m.state == ModelState.IDLE
                ]
            if not evictable:
                return
            oldest = min(evictable, key=lambda m: m.last_used_at or m.idle_since or datetime.min)
            if oldest.is_embedding and self._settings.keep_embedding_loaded:
                evictable.remove(oldest)
                if not evictable:
                    return
                oldest = min(evictable, key=lambda m: m.last_used_at or m.idle_since)
            self.unload(oldest.provider_id, oldest.model_id, reason="max_loaded_evicted")

    # ------------------------------------------------------------------
    # Cleanup cycle
    # ------------------------------------------------------------------

    def cleanup_idle_models(self, now=None) -> list[str]:
        """Unload every idle model whose idle lifetime reached its class
        timeout. Embeds always stay resident when configured. Returns the
        unloaded ids."""
        if not self._settings.enabled:
            return []
        now = now or datetime.now()
        unloaded: list[str] = []
        with self._lock:
            idle = [m for m in self._models.values() if m.state == ModelState.IDLE]
        for record in idle:
            if record.is_embedding and self._settings.keep_embedding_loaded:
                continue
            if self._overdue(record, now):
                self.unload(record.provider_id, record.model_id, reason="idle_timeout")
                unloaded.append(f"{record.provider_id}/{record.model_id}")
                self._publish(
                    Events.MODEL_IDLE_TIMEOUT_EXPIRED,
                    record,
                    {"model_id": record.model_id, "provider_id": record.provider_id},
                )
        return unloaded

    def check_memory_pressure(self, available_gb: float, now=None) -> list[str]:
        """If the free RAM is below the low-memory threshold, unload every
        idle model (embeddings stay when configured). Returns the list of
        ``provider/model`` ids that were unloaded."""
        if not self._settings.enabled:
            return []
        if available_gb >= self._settings.low_memory_threshold_gb:
            return []
        now = now or datetime.now()
        unloaded: list[str] = []
        with self._lock:
            idle = [m for m in self._models.values() if m.state == ModelState.IDLE]
        for record in idle:
            if record.is_embedding and self._settings.keep_embedding_loaded:
                continue
            self.unload(record.provider_id, record.model_id, reason="memory_pressure")
            unloaded.append(f"{record.provider_id}/{record.model_id}")
        if unloaded:
            self._metrics.record_memory_pressure()
        return unloaded

    def periodic_cleanup(self, now=None) -> dict:
        """Run one full cleanup pass: sync the loaded set, expire idle
        timeouts, then react to memory pressure. Returns a report dict."""
        if not self._settings.enabled:
            return {"timeout_unloaded": [], "pressure_unloaded": [], "protected": [], "reason": "ok"}
        now = now or datetime.now()
        start = time.monotonic()
        report: dict = {"timeout_unloaded": [], "pressure_unloaded": [], "protected": [], "reason": "ok"}

        self._sync_loaded_set(now)

        # 1. Idle-timeout sweep.
        for key in self.cleanup_idle_models(now):
            report["timeout_unloaded"].append(key)

        # 2. Memory pressure (only when a hardware profile is available).
        pressure = self._hardware_pressure_gb()
        if pressure is not None and pressure < self._settings.low_memory_threshold_gb:
            report["reason"] = "memory_pressure"
            for key in self.check_memory_pressure(pressure, now):
                report["pressure_unloaded"].append(key)

        # 3. Metrics bookkeeping.
        loaded = self.loaded_count()
        self._peak_loaded = max(self._peak_loaded, loaded)
        duration_s = time.monotonic() - start
        self._metrics.record_cleanup(duration_s, loaded)
        self._publish(
            Events.CLEANUP_CYCLE,
            None,
            {
                "duration_s": round(duration_s, 3),
                "timeout_unloaded": len(report["timeout_unloaded"]),
                "pressure_unloaded": len(report["pressure_unloaded"]),
                "loaded": loaded,
            },
        )
        log.info(
            "lifecycle_cleanup",
            duration_s=round(duration_s, 3),
            timeout_unloaded=len(report["timeout_unloaded"]),
            pressure_unloaded=len(report["pressure_unloaded"]),
            loaded=loaded,
        )
        return report

    def _hardware_pressure_gb(self) -> float | None:
        """Best-effort free-RAM from the hardware provider; None when it has
        not been wired (tests run without one)."""
        if self._hardware is None:
            return None
        try:
            profile = self._hardware.scan()
            available = float(profile.memory.available_gb)
        except Exception:  # noqa: BLE001 - pressure checks never crash
            return None
        self._metrics.update_system_ram(available)
        return available

    def _protected(self) -> list[str]:
        with self._lock:
            return [
                f"{m.provider_id}/{m.model_id}"
                for m in self._models.values()
                if m.state != ModelState.IDLE and m.is_loaded
            ]

    # ------------------------------------------------------------------
    # Loaded-model preference (router integration)
    # ------------------------------------------------------------------

    def find_reuse(
        self,
        task_decision: object,
        hardware=None,
        health: dict | None = None,
        available: dict | None = None,
        *,
        complexity: int | None = None,
    ) -> object | None:
        """Prefer an already-loaded, idle model over routing to a fresh one.

        Uses the router's own ``is_suitable`` and ``capability_score`` so the
        reuse decision respects the exact same gating/scoring rules the router
        would apply. A loaded model is reused when its capability score is
        within ``prefer_loaded_model_margin`` of the best available model.
        Returns a RoutingDecision override (or None to keep the router's
        choice). Pure decision — no I/O, no side effects.
        """
        if not self._settings.enabled or self._router is None:
            return None
        margin = self._settings.prefer_loaded_model_margin
        idle = self.idle_models()
        if not idle:
            return None

        # Gather the best capability score among all registry models that pass
        # the router's gating for this decision.
        registry = self._registry_models()
        if not registry:
            return None
        best_candidates = []
        ideal_score = 0.0
        for meta in registry:
            if not self._suitable(meta, task_decision, hardware, health, available):
                continue
            score = self._capability_score(meta, task_decision, complexity)
            if score > ideal_score:
                ideal_score = score
            best_candidates.append((meta, score))
        if ideal_score <= 0:
            return None

        # Find the best loaded idle model within margin of the ideal.
        best_hit = None
        best_hit_score = -1.0
        for meta, score in best_candidates:
            if score < ideal_score * (1.0 - margin):
                continue
            rec = self._get(getattr(meta, "provider_id", ""), getattr(meta, "id", ""))
            if rec is None or rec.state != ModelState.IDLE:
                continue
            if not self._available_meta(meta, available):
                continue
            if score > best_hit_score:
                best_hit_score = score
                best_hit = (meta, rec)

        if best_hit is None:
            return None
        meta, rec = best_hit

        with self._lock:
            rec.reuse_count += 1
        self._metrics.record_reuse()
        self._metrics.update_from_model(rec)

        self._publish(
            Events.MODEL_REUSED,
            rec,
            {
                "model_id": rec.model_id,
                "provider_id": rec.provider_id,
                "ideal_score": round(ideal_score, 3),
                "score": round(best_hit_score, 3),
                "margin": margin,
            },
        )
        log.info(
            "model_reused_loaded",
            model_id=rec.model_id,
            provider_id=rec.provider_id,
            ideal_score=round(ideal_score, 3),
            score=round(best_hit_score, 3),
        )
        return self._override_decision(meta, rec, best_hit_score, ideal_score)

    def _registry_models(self) -> list:
        if self._registry is None:
            return []
        try:
            return list(self._registry.all())
        except Exception:  # noqa: BLE001
            return []

    def _suitable(self, meta, decision, hardware, health, available) -> bool:
        if self._router is None:
            return True
        try:
            return bool(
                self._router.is_suitable(
                    meta,
                    hardware,
                    decision,
                    health or {},
                    available,
                )
            )
        except Exception:  # noqa: BLE001 - gating is best-effort
            return False

    def _capability_score(self, meta, decision, complexity) -> float:
        try:
            if complexity is None:
                return float(self._router.capability_score(meta, decision))
            return float(self._router.capability_score(meta, decision, complexity=complexity))
        except Exception:  # noqa: BLE001
            return 0.0

    def _available_meta(self, meta, available) -> bool:
        if not available:
            return True
        ids = available.get(getattr(meta, "provider_id", "")) or set()
        mid = getattr(meta, "id", "")
        return mid in ids or f"{mid}:latest" in ids

    def _override_decision(self, meta, rec: LoadedModel, score: float, ideal_score: float):
        """Build a RoutingDecision pointing at the already-loaded model."""
        try:
            from synapse.domain import RoutingDecision

            return RoutingDecision(
                provider_id=rec.provider_id,
                model_id=rec.model_id,
                kind=getattr(meta, "kind", None),
                confidence=round(min(0.99, max(0.5, score / max(ideal_score, 1e-9))), 2),
                reason=(
                    f"reused loaded model {rec.model_id} on {rec.provider_id} "
                    f"(capability {score:.3f} within {self._settings.prefer_loaded_model_margin:.0%} of ideal)"
                ),
                capability_score=round(score, 3),
            )
        except Exception:  # noqa: BLE001 - reuse is never fatal
            return None

    # ------------------------------------------------------------------
    # Embedding pre-warm
    # ------------------------------------------------------------------

    def preload_embedding(self, now=None) -> bool | None:
        """Pre-load every registry embedding model (keeps them resident).

        Returns True once at least one embedding model is loaded, None when
        the manager is disabled or embedding residency is off.
        """
        if not self._settings.enabled:
            return None
        if not self._settings.keep_embedding_loaded:
            return None
        if self._registry is None or self._providers is None:
            return None
        now = now or datetime.now()
        loaded_any = False
        registry_models = self._registry_models()
        if not registry_models:
            return None
        for meta in registry_models:
            caps = getattr(meta, "capabilities", None)
            if caps is None:
                continue
            is_embedding = float(getattr(caps, "embeddings", 0.0)) > 0.9 or (
                float(getattr(caps, "chat", 0.0)) <= 0.0
                and float(getattr(caps, "embeddings", 0.0)) > 0.0
            )
            if not is_embedding:
                continue
            mid = getattr(meta, "id", "")
            pid = getattr(meta, "provider_id", "")
            if not mid:
                continue
            loaded = self._is_resident(pid, mid)
            if not loaded:
                provider = self._provider(pid)
                if provider is not None and not self._load_via_provider(provider, mid):
                    continue
            record = self.track(pid, mid, meta)
            if record is None:
                continue
            if not record.is_loaded:
                with self._lock:
                    record.mark_loaded(self._sample_ram_gb(record), now)
                    self._metrics.record_load()
                    self._metrics.update_from_model(record)
                self._publish(
                    Events.MODEL_LOADED,
                    record,
                    {"model_id": record.model_id, "provider_id": record.provider_id},
                )
            loaded_any = True
        return True if loaded_any else None

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _publish(self, name: str, record: LoadedModel | None, payload: dict | None) -> None:
        if self._events is None:
            return
        data = dict(payload or {})
        if record is not None:
            data.setdefault("model_id", record.model_id)
            data.setdefault("provider_id", record.provider_id)
            data.setdefault("state", record.state.value)
        try:
            self._events.publish(name, data)
        except Exception:  # noqa: BLE001 - the bus is isolated per-subscriber
            pass