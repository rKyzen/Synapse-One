"""Phase 4: Citation Support contract — Grounded Responses.

When answers come from workspace files or retrieved knowledge, indicate
which source(s) were used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Citation:
    """A citation to a source."""

    source_id: str
    source_type: str  # "file" | "memory" | "web" | "model"
    file_name: str | None = None
    chunk_index: int | None = None
    page: int | None = None
    lines_start: int | None = None
    lines_end: int | None = None
    text_snippet: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class GroundedResponse:
    """A response with citations."""

    response: str
    citations: list[Citation]
    # Provenance: which sources contributed to which parts
    claim_sources: dict[str, list[str]]  # claim_text -> [citation_ids]


class CitationEngine(ABC):
    """Attaches citations to grounded responses."""

    @abstractmethod
    def attach_citations(
        self,
        response: str,
        retrieved_chunks: list[dict],
        memory_entries: list[dict] | None = None,
        *,
        citation_format: str = "inline",  # "inline" | "footnote" | "bracket"
    ) -> GroundedResponse:
        """Attach citations to a response based on retrieved sources.

        Args:
            response: The generated response
            retrieved_chunks: Chunks retrieved from workspace
            memory_entries: Memory entries used
            citation_format: How to format citations in the response

        Returns:
            GroundedResponse with citations attached
        """


class GroundingValidator(ABC):
    """Validates that responses are properly grounded in sources."""

    @abstractmethod
    def validate(
        self,
        response: str,
        sources: list[dict],
        *,
        require_citation_for_claims: bool = True,
    ) -> tuple[bool, list[str]]:
        """Check if response claims are supported by sources.

        Args:
            response: The response to validate
            sources: Available sources (chunks, memory entries)
            require_citation_for_claims: Whether factual claims need citations

        Returns:
            Tuple of (is_grounded, list_of_ungrounded_claims)
        """