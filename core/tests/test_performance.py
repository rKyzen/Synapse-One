"""PerformanceStore tests — the Phase 2.5 learning loop."""

from __future__ import annotations

from synapse.config.paths import SynapsePaths
from synapse.performance import FilePerformanceStore


def _store(tmp_path, **kwargs) -> FilePerformanceStore:
    paths = SynapsePaths.discover(home=tmp_path / "home", config_dir=tmp_path / "config")
    return FilePerformanceStore(paths, **kwargs)


def test_record_aggregates_averages(tmp_path):
    store = _store(tmp_path)
    store.record("ollama", "qwen2.5:3b", latency_s=10.0, tokens_per_s=12.0)
    store.record("ollama", "qwen2.5:3b", latency_s=20.0, tokens_per_s=18.0)
    stats = store.stats()
    entry = stats["ollama/qwen2.5:3b"]
    assert entry["samples"] == 2
    assert entry["avg_latency_s"] == 15.0
    assert entry["avg_tokens_per_second"] == 15.0
    assert entry["success_rate"] == 1.0
    assert entry["timeout_rate"] == 0.0


def test_record_tracks_failures_and_timeouts(tmp_path):
    store = _store(tmp_path)
    store.record("ollama", "qwen2.5:3b", latency_s=5.0, success=False)
    store.record("ollama", "qwen2.5:3b", latency_s=5.0, interrupted=True)
    entry = store.stats()["ollama/qwen2.5:3b"]
    assert entry["success_rate"] == 0.5
    assert entry["timeout_rate"] == 0.5


def test_persists_across_instances(tmp_path):
    _store(tmp_path).record("ollama", "llama3.1:8b", latency_s=33.3, tokens_per_s=7.0)
    reloaded = _store(tmp_path)
    assert reloaded.stats()["ollama/llama3.1:8b"]["avg_latency_s"] == 33.3
    assert reloaded.stats()["ollama/llama3.1:8b"]["avg_tokens_per_second"] == 7.0


def test_disabled_store_records_nothing(tmp_path):
    store = _store(tmp_path, enabled=False)
    store.record("ollama", "qwen2.5:3b", latency_s=1.0)
    assert store.stats() == {}


def test_stats_are_defensive_copies(tmp_path):
    store = _store(tmp_path)
    store.record("ollama", "qwen2.5:3b", latency_s=1.0)
    store.stats()["ollama/qwen2.5:3b"]["samples"] = 999
    assert store.stats()["ollama/qwen2.5:3b"]["samples"] == 1


def test_corrupt_history_file_is_tolerated(tmp_path):
    store = _store(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{ not json", encoding="utf-8")
    store.record("ollama", "qwen2.5:3b", latency_s=2.0)
    assert store.stats()["ollama/qwen2.5:3b"]["samples"] == 1


def test_pre_phase3_history_is_migrated_on_load(tmp_path):
    """Stale files written before the `failures` counter existed must not
    crash record() nor the router's reliability summary (regression)."""
    store = _store(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        '{"ollama/qwen2.5:3b": {"samples": 4, "successes": 3, "timeouts": 1, '
        '"avg_latency_s": 12.0, "avg_tokens_per_second": 9.0, '
        '"success_rate": 0.75, "timeout_rate": 0.25}}',
        encoding="utf-8",
    )
    reloaded = _store(tmp_path)
    entry = reloaded.stats()["ollama/qwen2.5:3b"]
    assert entry["failures"] == 0
    assert entry["failure_rate"] == 0.0
    assert entry["success_rate"] == 0.75
    reloaded.record("ollama", "qwen2.5:3b", latency_s=3.0, success=False)
    assert reloaded.stats()["ollama/qwen2.5:3b"]["failures"] == 1
