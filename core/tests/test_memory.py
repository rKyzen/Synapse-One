"""WorkspaceMemory tests — scoped save/search/persistence with and without embeddings."""

from __future__ import annotations

import json

import pytest

from synapse.domain import ModelDescriptor
from synapse.domain.enums import MemoryScope
from synapse.memory import WorkspaceMemory


class EmbedProvider:
    """Fake provider that exposes embed() and an embedding-capable model."""

    provider_id = "ollama"

    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}

    def embed(self, texts: list[str], model=None) -> list[list[float]]:
        return [self.vectors.get(t, [0.0, 0.0, 0.0, 0.0]) for t in texts]

    def list_models(self):
        return [ModelDescriptor(id="nomic-embed-text", provider_id="ollama")]


class NonEmbedProvider:
    """Fake provider with no embedding support (forces keyword fallback)."""

    provider_id = "ollama"
    embed = None

    def list_models(self):
        return [ModelDescriptor(id="llama3.2", provider_id="ollama")]


class FakeProviders:
    def __init__(self, *providers) -> None:
        self._providers = list(providers)

    def all(self):
        return self._providers


@pytest.fixture
def memory(temp_paths) -> WorkspaceMemory:
    return WorkspaceMemory(
        temp_paths, None, FakeProviders(EmbedProvider()), embedding_model="nomic-embed-text"
    )


@pytest.fixture
def memory_keyword(temp_paths) -> WorkspaceMemory:
    return WorkspaceMemory(
        temp_paths, None, FakeProviders(NonEmbedProvider()), embedding_model=None
    )


def test_save_returns_entry_and_persists(memory, temp_paths):
    entry = memory.save(MemoryScope.CONVERSATION, "the user prefers fast answers")
    assert entry.id and entry.scope == MemoryScope.CONVERSATION
    assert memory.recent(MemoryScope.CONVERSATION) == [entry]
    path = temp_paths.data_dir / "memory" / "conversation.json"
    assert path.exists()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw[0]["text"] == "the user prefers fast answers"


def test_search_uses_embeddings_when_available(memory):
    memory._providers._providers[0].vectors["deploy the api server"] = [1.0, 1.0, 0.0, 0.0]
    memory.save(MemoryScope.CONVERSATION, "deploy the api server")
    hits = memory.search(MemoryScope.CONVERSATION, "deploy the api server")
    assert len(hits) == 1
    assert hits[0].similarity == pytest.approx(1.0, abs=0.01)


def test_search_filters_below_min_similarity(memory):
    vectors = memory._providers._providers[0].vectors
    vectors["unrelated shopping list"] = [0.0, 0.0, 1.0, 1.0]
    vectors["deploy the api server"] = [1.0, 1.0, 0.0, 0.0]
    memory.save(MemoryScope.CONVERSATION, "unrelated shopping list")
    memory.save(MemoryScope.CONVERSATION, "deploy the api server")
    hits = memory.search(
        MemoryScope.CONVERSATION, "deploy the api server", min_similarity=0.5
    )
    assert len(hits) == 1  # only the semantically matching entry survives
    assert hits[0].text == "deploy the api server"


def test_search_keyword_fallback_without_embedding(memory_keyword):
    memory_keyword.save(MemoryScope.CONVERSATION, "deploy the api server safely")
    hits = memory_keyword.search(MemoryScope.CONVERSATION, "deploy the server")
    assert len(hits) == 1
    assert hits[0].similarity >= 0.3
    assert hits[0].text == "deploy the api server safely"


def test_scopes_are_isolated(memory):
    vectors = memory._providers._providers[0].vectors
    vectors["conversation note"] = [1.0, 0.0, 0.0, 0.0]
    vectors["project note"] = [0.0, 1.0, 0.0, 0.0]
    vectors["global note"] = [0.0, 0.0, 1.0, 0.0]
    memory.save(MemoryScope.CONVERSATION, "conversation note")
    memory.save(MemoryScope.PROJECT, "project note")
    memory.save(MemoryScope.GLOBAL, "global note")
    assert [e.text for e in memory.search(MemoryScope.CONVERSATION, "conversation note")] == ["conversation note"]
    assert memory.search(MemoryScope.PROJECT, "conversation note") == []


def test_recent_returns_newest_first(memory):
    memory.save(MemoryScope.CONVERSATION, "first")
    memory.save(MemoryScope.CONVERSATION, "second")
    recent = memory.recent(MemoryScope.CONVERSATION, limit=1)
    assert [e.text for e in recent] == ["second"]


def test_clear_removes_entries_and_persists(memory, temp_paths):
    memory.save(MemoryScope.CONVERSATION, "first")
    cleared = memory.clear(MemoryScope.CONVERSATION)
    assert cleared == 1
    assert memory.recent(MemoryScope.CONVERSATION) == []
    raw = json.loads((temp_paths.data_dir / "memory" / "conversation.json").read_text(encoding="utf-8"))
    assert raw == []


def test_entries_survive_reload(temp_paths):
    first = WorkspaceMemory(temp_paths, None, FakeProviders(EmbedProvider()))
    first.save(MemoryScope.PROJECT, "persisted note")
    second = WorkspaceMemory(temp_paths, None, FakeProviders(EmbedProvider()))
    assert [e.text for e in second.recent(MemoryScope.PROJECT)] == ["persisted note"]


def test_corrupt_memory_file_does_not_break_boot(temp_paths):
    path = temp_paths.data_dir / "memory" / "conversation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    memory = WorkspaceMemory(temp_paths, None, FakeProviders(EmbedProvider()))
    assert memory.recent(MemoryScope.CONVERSATION) == []


def test_embedding_failure_degrades_to_keyword(memory):
    memory.save(MemoryScope.CONVERSATION, "deploy the api server")

    def boom(texts, model=None):
        raise RuntimeError("embedding service down")

    memory._providers._providers[0].embed = boom
    hits = memory.search(MemoryScope.CONVERSATION, "deploy the server")
    assert len(hits) == 1  # degraded search still returns the entry