"""Tests for the Model Lifecycle Manager.

Uses a controllable fake provider (load/unload reflected in a local dict, like
Ollama's ``/api/ps``) plus the real Router for reuse gating, so the gating and
margin logic is exercised against production rules.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from synapse.domain import Capability, Decision, HardwareProfile, ModelCapabilities, ModelMetadata, ProviderKind
from synapse.domain.hardware import CpuInfo, MemoryInfo
from synapse.events import EventBus
from synapse.lifecycle import LifecycleSettings, ModelLifecycleManager, ModelState
from synapse.router import Router

T0 = datetime(2026, 1, 1, 12, 0, 0)


def make_model(mid: str, *, chat: float = 0.0, embeddings: float = 0.0,
               coding: float = 0.0, ram_gb: float = 0.0,
               provider: str = "ollama") -> ModelMetadata:
    return ModelMetadata(
        id=mid,
        provider_id=provider,
        kind=ProviderKind.LOCAL,
        display_name=mid,
        required_ram_gb=ram_gb,
        capabilities=ModelCapabilities(chat=chat, embeddings=embeddings, coding=coding),
    )


class FakeProvider:
    """Controllable provider adapter mimicking Ollama's lifecycle endpoints."""

    def __init__(self, provider_id: str = "ollama", loaded: dict | None = None) -> None:
        self.provider_id = provider_id
        self.loaded: dict[str, float] = dict(loaded or {})
        self.load_calls: list[str] = []
        self.unload_calls: list[str] = []
        self.fail_load = False

    def list_loaded(self) -> dict[str, float]:
        return dict(self.loaded)

    def is_loaded(self, model_id: str) -> bool | None:
        if model_id in self.loaded:
            return True
        return False if self.loaded else None

    def load_model(self, model_id: str) -> bool:
        self.load_calls.append(model_id)
        if self.fail_load:
            return False
        self.loaded[model_id] = 2.0
        return True

    def unload_model(self, model_id: str) -> bool:
        self.unload_calls.append(model_id)
        self.loaded.pop(model_id, None)
        return True

    def health(self) -> bool:
        return True


class FakeManager:
    def __init__(self, providers: list[FakeProvider]) -> None:
        self._providers = {p.provider_id: p for p in providers}

    def get(self, provider_id: str) -> FakeProvider | None:
        return self._providers.get(provider_id)

    def all(self) -> list[FakeProvider]:
        return list(self._providers.values())


class FakeRegistry:
    def __init__(self, models: list[ModelMetadata]) -> None:
        self._models = {m.id: m for m in models}

    def all(self) -> list[ModelMetadata]:
        return list(self._models.values())

    def get(self, model_id: str) -> ModelMetadata | None:
        return self._models.get(model_id)

    def by_provider(self, provider_id: str) -> list[ModelMetadata]:
        return [m for m in self._models.values() if m.provider_id == provider_id]


def build_manager(
    models: list[ModelMetadata],
    providers: list[FakeProvider] | None = None,
    settings: LifecycleSettings | None = None,
) -> ModelLifecycleManager:
    providers = providers or [FakeProvider()]
    return ModelLifecycleManager(
        settings=settings,
        providers=FakeManager(providers),
        registry=FakeRegistry(models),
        router=Router(),
        events=EventBus(),
        hardware=None,
    )


def hardware(available_gb: float = 16.0) -> HardwareProfile:
    return HardwareProfile(
        cpu=CpuInfo(model="test", cores=4),
        memory=MemoryInfo(total_gb=32.0, available_gb=available_gb),
    )


def chat_decision() -> Decision:
    return Decision(
        required_capabilities=[Capability.CHAT],
        preferred_kind=ProviderKind.LOCAL,
    )


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------


def test_settings_from_config_defaults():
    s = LifecycleSettings.from_config(None)
    assert s.idle_timeout_small_s == 60.0
    assert s.keep_embedding_loaded is True
    assert s.enabled is True


def test_settings_from_config_overrides():
    s = LifecycleSettings.from_config({"idle_timeout_small": 5, "cleanup_interval": 3})
    assert s.idle_timeout_small_s == 5.0
    assert s.cleanup_interval_s == 3.0


# ---------------------------------------------------------------------------
# request hooks: loading, active, idle
# ---------------------------------------------------------------------------


def test_note_started_loads_and_marks_active():
    m = ModelLifecycleManager(
        settings=LifecycleSettings(),
        providers=FakeManager([FakeProvider()]),
        registry=FakeRegistry([make_model("qwen", chat=1.0)]),
        router=Router(),
        events=EventBus(),
    )
    prov = m._providers.get("ollama")
    assert not prov.load_calls

    m.note_request_started("ollama", "qwen", now=T0)
    rec = m.get_model("ollama", "qwen")
    assert rec.state == ModelState.ACTIVE
    assert rec.is_loaded
    assert prov.load_calls == ["qwen"]

    m.note_request_completed("ollama", "qwen", now=T0)
    assert rec.state == ModelState.IDLE
    assert rec.load_count == 1


def test_note_started_reuses_resident_model_without_unloading():
    prov = FakeProvider(loaded={"qwen": 2.0})
    m = build_manager([make_model("qwen", chat=1.0)], providers=[prov])
    m.note_request_started("ollama", "qwen", now=T0)
    assert prov.load_calls == []  # already resident, no load ping


def test_note_started_then_genuine_generation_state():
    m = build_manager([make_model("qwen", chat=1.0)])
    m.note_request_started("ollama", "qwen", now=T0)
    rec = m.get_model("ollama", "qwen")
    assert rec.is_generating
    m.note_request_completed("ollama", "qwen", now=T0)
    assert not rec.is_generating


# ---------------------------------------------------------------------------
# idle timeout cleanup
# ---------------------------------------------------------------------------


def test_idle_model_unloaded_after_timeout():
    m = build_manager(
        [make_model("qwen", chat=1.0)],
        settings=LifecycleSettings(idle_timeout_small_s=60.0, idle_timeout_large_s=30.0),
    )
    prov = m._providers.get("ollama")
    m.note_request_started("ollama", "qwen", now=T0)
    m.note_request_completed("ollama", "qwen", now=T0)

    m.cleanup_idle_models(now=T0 + timedelta(seconds=10))
    assert prov.unload_calls == []  # not yet expired

    m.cleanup_idle_models(now=T0 + timedelta(seconds=61))
    assert prov.unload_calls == ["qwen"]
    assert m.get_model("ollama", "qwen").state == ModelState.OFFLINE


def test_embedding_model_never_expires():
    m = build_manager(
        [make_model("embed", chat=0.0, embeddings=1.0)],
        settings=LifecycleSettings(idle_timeout_small_s=1.0),
    )
    prov = m._providers.get("ollama")
    m.note_request_started("ollama", "embed", now=T0)
    m.note_request_completed("ollama", "embed", now=T0)

    m.cleanup_idle_models(now=T0 + timedelta(hours=1))
    assert prov.unload_calls == []


def test_large_model_has_shorter_timeout():
    m = build_manager(
        [make_model("big", chat=1.0, ram_gb=10.0)],
        settings=LifecycleSettings(idle_timeout_small_s=60.0, idle_timeout_large_s=5.0, large_model_min_ram_gb=8.0),
    )
    prov = m._providers.get("ollama")
    m.note_request_started("ollama", "big", now=T0)
    m.note_request_completed("ollama", "big", now=T0)

    m.cleanup_idle_models(now=T0 + timedelta(seconds=6))
    assert prov.unload_calls == ["big"]


# ---------------------------------------------------------------------------
# memory pressure
# ---------------------------------------------------------------------------


def test_memory_pressure_unloads_idle_but_not_active():
    m = build_manager(
        [make_model("a", chat=1.0), make_model("b", chat=1.0)],
        settings=LifecycleSettings(low_memory_threshold_gb=4.0),
    )
    prov = m._providers.get("ollama")
    m.note_request_started("ollama", "a", now=T0)
    m.note_request_completed("ollama", "a", now=T0)
    m.note_request_started("ollama", "b", now=T0)  # stays active

    unloaded = m.check_memory_pressure(2.0, now=T0)
    assert "ollama/a" in unloaded
    assert prov.unload_calls == ["a"]
    assert m.get_model("ollama", "b").is_generating  # protected


def test_memory_pressure_keeps_embedding_when_requested():
    m = build_manager(
        [make_model("embed", chat=0.0, embeddings=1.0), make_model("qwen", chat=1.0)],
        settings=LifecycleSettings(low_memory_threshold_gb=4.0, keep_embedding_loaded=True),
    )
    prov = m._providers.get("ollama")
    m.note_request_started("ollama", "embed", now=T0)
    m.note_request_completed("ollama", "embed", now=T0)
    m.note_request_started("ollama", "qwen", now=T0)
    m.note_request_completed("ollama", "qwen", now=T0)

    m.check_memory_pressure(2.0, now=T0)
    assert "embed" not in prov.unload_calls
    assert "qwen" in prov.unload_calls


def test_no_pressure_above_threshold():
    m = build_manager([make_model("qwen", chat=1.0)])
    m.note_request_started("ollama", "qwen", now=T0)
    m.note_request_completed("ollama", "qwen", now=T0)
    assert m.check_memory_pressure(10.0, now=T0) == []


# ---------------------------------------------------------------------------
# reuse within capability margin
# ---------------------------------------------------------------------------


def test_find_reuse_returns_loaded_model_within_margin():
    models = [make_model("ideal", chat=1.0), make_model("good", chat=0.95)]
    m = build_manager(models)
    m.note_request_started("ollama", "good", now=T0)
    m.note_request_completed("ollama", "good", now=T0)

    reuse = m.find_reuse(chat_decision(), hardware(), {"ollama": True})
    assert reuse is not None
    assert reuse.model_id == "good"
    assert reuse.reason.startswith("reused loaded model")
    rec = m.get_model("ollama", "good")
    assert rec.reuse_count == 1


def test_find_reuse_skips_when_loaded_model_too_weak():
    models = [make_model("ideal", chat=1.0), make_model("weak", chat=0.1)]
    m = build_manager(models)
    m.note_request_started("ollama", "weak", now=T0)
    m.note_request_completed("ollama", "weak", now=T0)

    reuse = m.find_reuse(chat_decision(), hardware(), {"ollama": True})
    assert reuse is None


def test_find_reuse_none_with_no_idle_model():
    m = build_manager([make_model("qwen", chat=1.0)])
    m.note_request_started("ollama", "qwen", now=T0)  # active, not idle
    reuse = m.find_reuse(chat_decision(), hardware(), {"ollama": True})
    assert reuse is None


def test_reuse_respects_availability():
    models = [make_model("qwen", chat=1.0)]
    m = build_manager(models)
    m.note_request_started("ollama", "qwen", now=T0)
    m.note_request_completed("ollama", "qwen", now=T0)
    # provider reports the model as NOT installed via /api/ps path is per-model
    reuse = m.find_reuse(chat_decision(), hardware(), {"ollama": True}, {"ollama": {"other"}})
    assert reuse is None  # not the model being tested? acceptable


# ---------------------------------------------------------------------------
# preload
# ---------------------------------------------------------------------------


def test_preload_embedding():
    prov = FakeProvider()
    m = build_manager([make_model("nomic-embed", chat=0.0, embeddings=1.0)], providers=[prov])
    assert m.preload_embedding(now=T0) is True
    assert "nomic-embed" in prov.load_calls
    rec = m.get_model("ollama", "nomic-embed")
    assert rec.is_loaded


def test_preload_skipped_when_disabled():
    m = build_manager(
        [make_model("nomic-embed", chat=0.0, embeddings=1.0)],
        settings=LifecycleSettings(keep_embedding_loaded=False),
    )
    assert m.preload_embedding(now=T0) is None


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_metrics_count_load_unload_and_ram():
    m = build_manager(
        [make_model("qwen", chat=1.0)],
        settings=LifecycleSettings(idle_timeout_small_s=5.0),
    )
    m.note_request_started("ollama", "qwen", now=T0)
    m.note_request_completed("ollama", "qwen", now=T0)
    m.cleanup_idle_models(now=T0 + timedelta(seconds=10))

    sys_metrics = m.metrics().get_system_metrics()
    assert sys_metrics["total_loads"] == 1
    assert sys_metrics["total_unloads"] == 1
    assert sys_metrics["total_reuses"] == 0
    assert sys_metrics["average_ram_reclaimed_gb"] == pytest.approx(2.0)
    assert sys_metrics["current_loaded_models"] == 0

    model_metrics = m.metrics().get_model_metrics("qwen", "ollama")
    assert model_metrics["load_count"] == 1
    assert model_metrics["unload_count"] == 1


def test_metrics_record_reuse():
    models = [make_model("ideal", chat=1.0), make_model("good", chat=0.95)]
    m = build_manager(models)
    m.note_request_started("ollama", "good", now=T0)
    m.note_request_completed("ollama", "good", now=T0)
    m.find_reuse(chat_decision(), hardware(), {"ollama": True})
    assert m.metrics().get_system_metrics()["total_reuses"] == 1


def test_cleanup_cycle_records_metrics():
    m = build_manager([make_model("qwen", chat=1.0)])
    m.note_request_started("ollama", "qwen", now=T0)
    m.note_request_completed("ollama", "qwen", now=T0)
    m.periodic_cleanup(now=T0 + timedelta(seconds=61))
    stats = m.metrics().get_cleanup_stats()
    assert stats["total_cleanup_cycles"] >= 1


# ---------------------------------------------------------------------------
# provider reconciliation
# ---------------------------------------------------------------------------


def test_sync_adopts_externally_loaded_model():
    prov = FakeProvider(loaded={"qwen": 2.0})
    m = build_manager([make_model("qwen", chat=1.0)], providers=[prov])
    m._sync_loaded_set(now=T0)
    rec = m.get_model("ollama", "qwen")
    assert rec is not None
    assert rec.is_loaded
    assert rec.ram_gb == pytest.approx(2.0)


def test_tagged_alias_ram_sample():
    m = build_manager([make_model("qwen", chat=1.0)])
    m.note_request_started("ollama", "qwen", now=T0)
    rec = m.get_model("ollama", "qwen")
    # fake reports the tagged name, like real Ollama /api/ps
    m._providers.get("ollama").loaded = {"qwen:latest": 3.0}
    assert m._sample_ram_gb(rec) == pytest.approx(3.0)


def test_sync_updates_untagged_record_via_alias():
    prov = FakeProvider()
    m = build_manager([make_model("qwen", chat=1.0)], providers=[prov])
    m.note_request_started("ollama", "qwen", now=T0)
    m.note_request_completed("ollama", "qwen", now=T0)
    assert m.get_model("ollama", "qwen:latest") is None  # no orphan record

    # Ollama reports the tagged name, like /api/ps
    prov.loaded = {"qwen:latest": 2.5}
    m._sync_loaded_set(now=T0)
    rec = m.get_model("ollama", "qwen")
    assert rec is not None
    assert rec.ram_gb == pytest.approx(2.5)
    assert rec.is_loaded


# ---------------------------------------------------------------------------
# max-loaded cap
# ---------------------------------------------------------------------------


def test_max_loaded_evicts_oldest_idle():
    prov = FakeProvider()
    m = build_manager(
        [make_model("a", chat=1.0), make_model("b", chat=1.0)],
        providers=[prov],
        settings=LifecycleSettings(max_loaded_models=1),
    )
    m.note_request_started("ollama", "a", now=T0)
    m.note_request_completed("ollama", "a", now=T0)
    later = T0 + timedelta(seconds=5)
    m.note_request_started("ollama", "b", now=later)
    assert prov.unload_calls == ["a"]
    assert m.get_model("ollama", "a").state == ModelState.OFFLINE
    assert m.get_model("ollama", "b").is_loaded


# ---------------------------------------------------------------------------
# disabled manager is inert
# ---------------------------------------------------------------------------


def test_disabled_manager_does_not_track():
    m = build_manager(
        [make_model("qwen", chat=1.0)],
        settings=LifecycleSettings(enabled=False),
    )
    m.note_request_started("ollama", "qwen", now=T0)
    m.note_request_completed("ollama", "qwen", now=T0)
    assert m.get_model("ollama", "qwen") is None
    assert m.find_reuse(chat_decision(), hardware(), {"ollama": True}) is None


# ---------------------------------------------------------------------------
# background loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_loop_runs_cleanup_and_stops():
    settings = LifecycleSettings(cleanup_interval_s=0.01, idle_timeout_small_s=5.0)
    m = build_manager([make_model("qwen", chat=1.0)], settings=settings)
    m.note_request_started("ollama", "qwen", now=T0)
    m.note_request_completed("ollama", "qwen", now=T0)

    task = m.start_background_task()
    assert task is not None
    await asyncio.sleep(0.05)
    m.stop_background_task()
    assert m.metrics().get_system_metrics()["cleanup_cycles"] >= 1
    # the (now long-idle) model got unloaded by the loop
    rec = m.get_model("ollama", "qwen")
    assert rec.state == ModelState.OFFLINE