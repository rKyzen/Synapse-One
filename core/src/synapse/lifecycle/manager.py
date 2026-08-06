"""ModelLifecycleManager — keeps loaded models useful and RAM low.

Responsibilities (provider-agnostic; the provider is only ever asked to
list/inspect/load/unload a model, never to prefer one policy or another):

* Track every model we know is loaded (or that we asked to be loaded).
* Mark models active while generating, idle once a response completes.
* Prefer to reuse an already-loaded model when its capability match is
  within ``prefer_loaded_model_margin`` of the best cold candidate.
* Auto-unload idle models after a configurable timeout (large models sooner).
* Never auto-unload the embedding model while ``keep_embedding_loaded``
  is on; optionally preload it at startup.
* Under memory pressure, immediately unload every idle model.
* Expose aggregate metrics and publish lifecycle events on the bus.

The manager is synchronous (providers are synchronous). A thin async wrapper
(``run_cleanup_loop``) is provided so the API layer can drive periodic
cleanup from an event-loop task without blocking.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

import structlog

from synapse.contracts import ModelRegistry, Router
from synapse.domain.diagnosis import Decision, RoutingDecision
from synapse.domain.hardware import HardwareProfile
from synapse.events import EventBus, Events
from synapse.lifecycle.metrics import ModelLifecycleMetrics
from synapse.lifecycle.models import LifecycleSettings, LoadedModel, ModelState

log = structlog.get_logger("synapse.lifecycle.manager")


class ModelLifecycleManager:
    """Coordinates load/unload hooks, idle cleanup, reuse, and metrics."""

    def __init__(
        self,
        *,
        settings: LifecycleSettings | None = None,
        providers,
        registry: ModelRegistry,
        router: Router,
        events: EventBus,
        hardware=None,
    ) -> None:
        self._settings = settings or LifecycleSettings()
        self._providers = providers
        self._registry = registry
        self._router = router
        self._events = events
        self._hardware = hardware
        self._metrics = ModelLifecycleMetrics()
        #: key = (provider_id, model_id) -> LoadedModel
        self._models: dict[tuple[str, str], LoadedModel] = {}
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # state queries
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    def get_model(self, provider_id: str, model_id: str) -> LoadedModel | None:
        return self._models.get((provider_id, model_id))

    def loaded_models(self) -> list[LoadedModel]:
        return [m for m in self._models.values() if m.is_loaded]

    def current_loaded_count(self) -> int:
        return sum(1 for m in self._models.values() if m.is_loaded)

    def metrics(self) -> ModelLifecycleMetrics:
        return self._metrics

    # ------------------------------------------------------------------
    # request hooks (called by the Master around execution)
    # ------------------------------------------------------------------

    def note_request_started(
        self,
        provider_id: str,
        model_id: str,
        now: datetime | None = None,
    ) -> None:
        """Mark a model active for a request, loading it first if needed."""
        if not self.enabled:
            return
        rec = self._ensure_tracked(provider_id, model_id, now=now)
        provider = self._providers.get(provider_id)

        if not rec.is_loaded:
            self._load(rec, provider, now)

        rec.mark_active(now)
        self._metrics.update_from_model(rec)

    def note_request_completed(
        self,
        provider_id: str,
        model_id: str,
        now: datetime | None = None,
    ) -> None:
        """Move a model back to idle after a response completes."""
        if not self.enabled:
            return
        rec = self._ensure_tracked(provider_id, model_id, now=now)
        rec.mark_idle(now)
        self._metrics.update_from_model(rec)

    # ------------------------------------------------------------------
    # reuse
    # ------------------------------------------------------------------

    def find_reuse(
        self,
        decision: Decision,
        hardware: HardwareProfile,
        provider_health: dict[str, bool],
        available_models: dict[str, set[str]] | None = None,
        *,
        complexity: int = 0,
    ) -> RoutingDecision | None:
        """Return a reuse RoutingDecision if a loaded model is close enough.

        Compares the best already-loaded suitable model against the best
        cold-candidate score; reuses only when the loaded model is within
        ``prefer_loaded_model_margin`` (relative) of the winner. Never picks
        a model still generating. Returns None to let the router decide.
        """
        if not self.enabled:
            return None

        suitable = self._candidate_metas(decision, hardware, provider_health, available_models)
        if not suitable:
            return None

        best_meta = max(suitable, key=lambda mb: self._router.capability_score(mb, decision, complexity=complexity))
        best_score = self._router.capability_score(best_meta, decision, complexity=complexity)

        loaded_idle: list[tuple[LoadedModel, Any, float]] = []
        for rec in self._models.values():
            if rec.state != ModelState.IDLE:
                continue
            meta = self._resolve_metadata_rec(rec)
            if meta is None:
                continue
            if not self._router.is_suitable(
                meta, hardware, decision, provider_health, available_models
            ):
                continue
            score = self._router.capability_score(meta, decision, complexity=complexity)
            loaded_idle.append((rec, meta, score))

        if not loaded_idle:
            return None

        rec, meta, score = max(loaded_idle, key=lambda x: x[2])
        margin = self._settings.prefer_loaded_model_margin
        if score <= 0.0 or score < best_score * (1.0 - margin):
            return None

        rec.reuse_count += 1
        self._metrics.record_reuse()
        self._metrics.update_from_model(rec)
        reason = (
            f"reused loaded model {rec.model_id} "
            f"(score {score:.2f} vs best-candidate {best_score:.2f})"
        )
        self._events.publish(
            Events.MODEL_REUSED,
            {
                "provider_id": rec.provider_id,
                "model_id": rec.model_id,
                "score": round(score, 3),
                "best_score": round(best_score, 3),
                "reason": reason,
            },
        )
        log.info(
            "model_reused",
            provider_id=rec.provider_id,
            model_id=rec.model_id,
            score=round(score, 3),
            best_score=round(best_score, 3),
        )
        return RoutingDecision(
            provider_id=rec.provider_id,
            model_id=rec.model_id,
            kind=meta.kind,
            confidence=min(1.0, score / best_score) if best_score > 0 else 0.5,
            reason=reason,
            capability_score=round(score, 4),
        )

    # ------------------------------------------------------------------
    # startup preload
    # ------------------------------------------------------------------

    def preload_embedding(self, now: datetime | None = None) -> bool | None:
        """Load and pin the embedding model at startup, best-effort.

        Returns True when loaded, False on failure, None when skipped (no
        embedding model configured/installed, or policy says keep going).
        """
        if not self.enabled or not self._settings.keep_embedding_loaded:
            return None

        for meta in self._registry.all():
            if meta.capabilities.embeddings <= 0.9:
                continue
            provider = self._providers.get(meta.provider_id)
            if provider is None:
                continue
            try:
                if not provider.health():
                    log.info("embedding_preload_skipped_unhealthy", provider_id=meta.provider_id)
                    continue
            except Exception:  # noqa: BLE001 - best effort
                continue
            rec = self._ensure_tracked(meta.provider_id, meta.id, now=now)
            if rec.is_loaded:
                return True
            self._load(rec, provider, now)
            return rec.is_loaded
        log.info("embedding_model_not_found", note="no installed embedding model")
        return None

    # ------------------------------------------------------------------
    # cleanup
    # ------------------------------------------------------------------

    def cleanup_idle_models(self, now: datetime | None = None) -> list[str]:
        """Unload idle models whose timeout elapsed. Returns unloaded keys."""
        now = now or datetime.now()
        unloaded: list[str] = []
        for rec in list(self._models.values()):
            if rec.state != ModelState.IDLE:
                continue
            timeout_s = self._idle_timeout(rec)
            if timeout_s == float("inf"):
                continue
            if rec.idle_duration_s(now) >= timeout_s:
                key = f"{rec.provider_id}/{rec.model_id}"
                self._events.publish(
                    Events.MODEL_IDLE_TIMEOUT_EXPIRED,
                    {"provider_id": rec.provider_id, "model_id": rec.model_id, "idle_s": round(rec.idle_duration_s(now), 2)},
                )
                self._unload(rec, reason="idle_timeout", now=now)
                unloaded.append(key)
        return unloaded

    def check_memory_pressure(self, available_gb: float | None, now: datetime | None = None) -> list[str]:
        """Under low RAM, immediately unload every idle (non-embedding) model."""
        if available_gb is None:
            return []
        if available_gb >= self._settings.low_memory_threshold_gb:
            return []
        if not self.enabled:
            return []

        now = now or datetime.now()
        unloaded: list[str] = []
        self._metrics.record_memory_pressure()
        self._events.publish(
            Events.MEMORY_PRESSURE,
            {"available_gb": round(available_gb, 2), "threshold_gb": self._settings.low_memory_threshold_gb},
        )
        log.warning(
            "memory_pressure",
            available_gb=round(available_gb, 2),
            threshold_gb=self._settings.low_memory_threshold_gb,
        )
        for rec in list(self._models.values()):
            if rec.state != ModelState.IDLE:
                continue
            if rec.is_embedding and self._settings.keep_embedding_loaded:
                continue
            key = f"{rec.provider_id}/{rec.model_id}"
            self._unload(rec, reason="memory_pressure", now=now)
            unloaded.append(key)
        return unloaded

    def periodic_cleanup(self, now: datetime | None = None) -> dict[str, Any]:
        """One full lifecycle pass: reconcile, sample RAM, then evict idle."""
        start = time.monotonic()
        self._sync_loaded_set(now)

        available_gb: float | None = None
        if self._hardware is not None:
            try:
                profile = self._hardware.scan()
                available_gb = profile.memory.available_gb
                self._metrics.update_system_ram(available_gb)
            except Exception:  # noqa: BLE001 - RAM sampling is best effort
                log.exception("lifecycle_hardware_scan_failed")

        unloaded: list[str] = []
        unloaded += self.check_memory_pressure(available_gb, now)
        unloaded += self.cleanup_idle_models(now)

        self._metrics.record_cleanup(time.monotonic() - start, self.current_loaded_count())
        summary = {
            "duration_s": round(time.monotonic() - start, 4),
            "unloaded": unloaded,
            "loaded_count": self.current_loaded_count(),
        }
        self._events.publish(Events.CLEANUP_CYCLE, summary)
        log.info("lifecycle_cleanup", **summary)
        return summary

    # ------------------------------------------------------------------
    # background driver (async)
    # ------------------------------------------------------------------

    async def run_cleanup_loop(self) -> None:
        """Periodic cleanup driver, meant to run as an asyncio task."""
        log.info("lifecycle_loop_started", interval_s=self._settings.cleanup_interval_s)
        while True:
            try:
                await asyncio.sleep(self._settings.cleanup_interval_s)
                if self.enabled:
                    await asyncio.to_thread(self.periodic_cleanup)
            except asyncio.CancelledError:
                break
            except Exception:  # noqa: BLE001 - never kill the loop
                log.exception("lifecycle_cleanup_error")

    def start_background_task(self) -> asyncio.Task | None:
        """Start the cleanup loop on the running event loop, if any."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return None
        self.stop_background_task()
        self._task = asyncio.create_task(self.run_cleanup_loop())
        return self._task

    def stop_background_task(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _ensure_tracked(
        self,
        provider_id: str,
        model_id: str,
        now: datetime | None = None,
    ) -> LoadedModel:
        key = (provider_id, model_id)
        rec = self._models.get(key)
        if rec is not None:
            return rec

        meta = self._resolve_metadata(provider_id, model_id)
        required_ram = 0.0
        if meta is not None:
            required_ram = meta.required_ram_gb or 0.0
            size_estimate = (meta.size_bytes or 0) / (1024**3)
            if not required_ram and size_estimate:
                required_ram = size_estimate * 1.1

        rec = LoadedModel(
            model_id=model_id,
            provider_id=provider_id,
            display_name=meta.display_name if meta else model_id,
            is_embedding=self._is_embedding(model_id, meta),
            is_large=bool(meta is not None and required_ram >= self._settings.large_model_min_ram_gb),
            required_ram_gb=round(required_ram, 3),
        )
        self._models[key] = rec
        return rec

    def _resolve_metadata(self, provider_id: str, model_id: str):
        try:
            meta = self._registry.get(model_id)
            if meta is not None:
                return meta
        except Exception:  # noqa: BLE001
            pass
        # fall back to a prefix match for tagged ids (e.g. "qwen3:4b" -> "qwen3")
        base = model_id.split(":", 1)[0]
        try:
            for m in self._registry.by_provider(provider_id):
                if m.id == base:
                    return m
        except Exception:  # noqa: BLE001
            pass
        return None

    def _resolve_metadata_rec(self, rec: LoadedModel):
        return self._resolve_metadata(rec.provider_id, rec.model_id)

    @staticmethod
    def _is_embedding(model_id: str, meta) -> bool:
        if meta is not None:
            try:
                if meta.capabilities.embeddings > 0.9:
                    return True
            except Exception:  # noqa: BLE001
                pass
        return "embed" in model_id.lower()

    def _idle_timeout(self, rec: LoadedModel) -> float:
        if rec.is_embedding:
            return self._settings.idle_timeout_embedding_s
        if rec.is_large:
            return self._settings.idle_timeout_large_s
        return self._settings.idle_timeout_small_s

    def _load(self, rec: LoadedModel, provider, now: datetime | None = None) -> bool:
        """Best-effort pin+load. Returns True when the model is resident."""
        start = time.monotonic()
        rec.mark_load_start(now)

        accepted = False
        if provider is not None:
            try:
                if self._provider_is_loaded(provider, rec.model_id):
                    accepted = True
                else:
                    accepted = bool(provider.load_model(rec.model_id))
            except Exception as exc:  # noqa: BLE001 - hard failure
                log.error("model_load_failed", provider_id=rec.provider_id, model_id=rec.model_id, error=str(exc))
                rec.mark_failed(now)
                record_time = time.monotonic() - start
                rec.total_load_s += record_time
                self._metrics.record_failure()
                self._metrics.update_from_model(rec)
                self._events.publish(
                    Events.MODEL_LOAD_FAILED,
                    {"provider_id": rec.provider_id, "model_id": rec.model_id, "error": str(exc)},
                )
                return False

        if accepted:
            rec.mark_loaded(None, now)
            rec.total_load_s += time.monotonic() - start
            self._metrics.record_load()
            self._metrics.update_from_model(rec)
            self._events.publish(
                Events.MODEL_LOADED,
                {"provider_id": rec.provider_id, "model_id": rec.model_id},
            )
            log.info("model_loaded", provider_id=rec.provider_id, model_id=rec.model_id)
            self._enforce_max_loaded()
            return True

        # Provider has no load support (or refused) — chat will lazy-load it.
        rec.mark_loaded(None, now)
        log.debug("model_no_pin", provider_id=rec.provider_id, model_id=rec.model_id)
        return True

    @staticmethod
    def _provider_is_loaded(provider, model_id: str) -> bool:
        try:
            return provider.is_loaded(model_id) is True
        except Exception:  # noqa: BLE001
            return False

    def _unload(self, rec: LoadedModel, reason: str, now: datetime | None = None) -> float | None:
        """Ask the provider to evict ``rec``; returns reclaimed RAM in GB."""
        if rec.ram_gb is None:
            rec.ram_gb = self._sample_ram_gb(rec)

        start = time.monotonic()
        rec.mark_unload_start(now)
        provider = self._providers.get(rec.provider_id)
        if provider is not None:
            try:
                provider.unload_model(rec.model_id)
            except Exception:  # noqa: BLE001
                log.exception("model_unload_error", provider_id=rec.provider_id, model_id=rec.model_id)

        duration = time.monotonic() - start
        reclaimed = rec.ram_gb
        rec.mark_unloaded(reclaimed, duration, now)
        self._metrics.record_unload()
        self._metrics.update_from_model(rec)
        self._events.publish(
            Events.MODEL_UNLOADED,
            {
                "provider_id": rec.provider_id,
                "model_id": rec.model_id,
                "reason": reason,
                "ram_reclaimed_gb": round(reclaimed, 3) if reclaimed is not None else None,
            },
        )
        log.info(
            "model_unloaded",
            provider_id=rec.provider_id,
            model_id=rec.model_id,
            reason=reason,
            ram_reclaimed_gb=round(reclaimed, 3) if reclaimed is not None else None,
        )
        return reclaimed

    def _sample_ram_gb(self, rec: LoadedModel) -> float | None:
        provider = self._providers.get(rec.provider_id)
        if provider is None:
            return None
        try:
            loaded = provider.list_loaded()
        except Exception:  # noqa: BLE001
            return None
        for lid, ram in loaded.items():
            if self._alias_match(lid, rec.model_id):
                return ram
        return None

    def _sync_loaded_set(self, now: datetime | None = None) -> None:
        """Reconcile tracked state with what providers report as resident.

        Models that became loaded outside Synapse (``ollama run``) are adopted;
        tracked models that vanished are marked unloaded.
        """
        if not self.enabled:
            return
        for provider in self._providers.all():
            try:
                loaded = provider.list_loaded()
            except Exception:  # noqa: BLE001
                continue
            if not loaded:
                # Empty result is ambiguous (nothing loaded vs API failure).
                continue

            for lid, ram in loaded.items():
                rec = self._find_alias_record(provider.provider_id, lid)
                if rec is None:
                    rec = self._ensure_tracked(provider.provider_id, lid, now=now)
                rec.ram_gb = ram
                if rec.state in (ModelState.OFFLINE, ModelState.FAILED):
                    rec.mark_loaded(ram, now)
                    self._metrics.record_load()
                    self._metrics.update_from_model(rec)
                else:
                    self._metrics.update_from_model(rec)

    def _alias_match(self, live_id: str, model_id: str) -> bool:
        if live_id == model_id:
            return True
        return live_id.split(":", 1)[0] == model_id.split(":", 1)[0]

    def _find_alias_record(self, provider_id: str, live_id: str) -> LoadedModel | None:
        """Return an existing tracked record for ``live_id`` (tag-aware).

        Ollama reports tagged ids (``qwen3:latest``) while Synapse routes by
        the untagged base name (``qwen3``). Prefer updating the untagged record
        over creating a duplicate tagged one.
        """
        exact = self._models.get((provider_id, live_id))
        if exact is not None:
            return exact
        base = live_id.split(":", 1)[0]
        return self._models.get((provider_id, base))

    # ------------------------------------------------------------------
    # max-loaded cap
    # ------------------------------------------------------------------

    def _enforce_max_loaded(self) -> None:
        max_loaded = self._settings.max_loaded_models
        loaded = [m for m in self._models.values() if m.is_loaded]
        if len(loaded) <= max_loaded:
            return
        # unload the least-recently-used idle models until under the cap
        idle = [m for m in loaded if m.state == ModelState.IDLE]
        idle.sort(key=lambda m: m.last_used_at or datetime.min)
        for rec in idle[: len(loaded) - max_loaded]:
            self._unload(rec, reason="max_loaded")

    # ------------------------------------------------------------------
    # builder helpers
    # ------------------------------------------------------------------

    def _candidate_metas(
        self,
        decision: Decision,
        hardware: HardwareProfile,
        provider_health: dict[str, bool],
        available_models: dict[str, set[str]] | None,
    ) -> list[Any]:
        out = []
        for meta in self._registry.all():
            try:
                if self._router.is_suitable(meta, hardware, decision, provider_health, available_models):
                    out.append(meta)
            except Exception:  # noqa: BLE001
                continue
        return out