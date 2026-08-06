"""Workspace Memory contract — scoped semantic memory for the Master.

Memory is scoped (conversation / project / global) and embedding-based:
entries are stored with vectors so queries can retrieve semantically related
history. Providers that cannot embed (or have no embedding model installed)
degrade to keyword matching — memory must never break the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from synapse.domain.enums import MemoryScope
from synapse.domain.memory import MemoryEntry


class MemoryStore(ABC):
    @abstractmethod
    def save(
        self,
        scope: MemoryScope,
        text: str,
        *,
        embedding: list[float] | None = None,
        source: str = "",
        metadata: dict | None = None,
    ) -> MemoryEntry:
        """Store an entry and return it (with id + timestamp)."""

    @abstractmethod
    def search(
        self,
        scope: MemoryScope,
        query: str,
        *,
        k: int = 3,
        min_similarity: float = 0.35,
    ) -> list[MemoryEntry]:
        """Return up to ``k`` entries most similar to ``query``, best first.

        Entries carry a ``similarity`` field. Degrades to keyword match when
        embeddings are unavailable. Never raises.
        """

    @abstractmethod
    def recent(self, scope: MemoryScope, limit: int = 10) -> list[MemoryEntry]:
        """Newest entries first — cheap chronological view."""

    @abstractmethod
    def clear(self, scope: MemoryScope | None = None) -> int:
        """Remove all entries in ``scope`` (or every scope when None)."""
