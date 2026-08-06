"""Phase 3 domain: workspace memory entries."""

from __future__ import annotations

from pydantic import BaseModel, Field

from synapse.domain.enums import MemoryScope


class MemoryEntry(BaseModel):
    """One stored memory item with optional embedding for similarity search."""

    id: str
    scope: MemoryScope
    text: str
    embedding: list[float] | None = None
    created_at: str = ""
    source: str = ""
    metadata: dict = Field(default_factory=dict)
    #: similarity of this entry to the query that returned it (0..1).
    similarity: float | None = None