"""Citation & Grounding — Phase 4.

When an answer draws on workspace files or memory, we say which sources were
used. ``InlineCitationEngine`` attaches numbered citations; the
``GroundingValidatorImpl`` verifies the response actually reflects the
sources instead of making ungrounded claims.
"""

from __future__ import annotations

import re

from synapse.contracts.citations import Citation, CitationEngine, GroundingValidator, GroundedResponse


#: Words that never count toward grounding overlap.
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "was", "are", "were",
    "you", "your", "has", "have", "its", "his", "her", "but", "not", "all",
    "can", "could", "would", "should", "will", "shall", "may", "might", "about",
}  # fmt: skip


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOPWORDS}


class InlineCitationEngine(CitationEngine):
    """Attaches numbered citations to a response.

    Formats:
        inline   — "[1]" / "([1], [2])" after each sentence
        bracket  — "[1]" at the end of the response
        footnote — trailing "Sources: 1. … 2. …" block
    """

    def attach_citations(
        self,
        response: str,
        retrieved_chunks: list[dict],
        memory_entries: list[dict] | None = None,
        *,
        citation_format: str = "inline",
    ) -> GroundedResponse:
        citations: list[Citation] = []
        used_indices: list[int] = []

        for chunk in retrieved_chunks:
            source_id = str(chunk.get("chunk_index") or chunk.get("id") or f"chunk-{len(citations)}")
            text = str(chunk.get("text", ""))[:200]
            if not text or not self._overlaps(response, text):
                continue
            used_indices.append(len(citations))
            citations.append(
                Citation(
                    source_id=source_id,
                    source_type="file",
                    file_name=chunk.get("file_name") or chunk.get("filename"),
                    chunk_index=chunk.get("chunk_index"),
                    page=chunk.get("page"),
                    lines_start=chunk.get("lines_start"),
                    lines_end=chunk.get("lines_end"),
                    text_snippet=text[:160],
                    score=chunk.get("score"),
                )
            )

        for entry in memory_entries or []:
            text = str(entry.get("text", ""))[:200]
            if not text or not self._overlaps(response, text):
                continue
            used_indices.append(len(citations))
            citations.append(
                Citation(
                    source_id=str(entry.get("id") or f"memory-{len(citations)}"),
                    source_type="memory",
                    file_name=entry.get("source"),
                    text_snippet=text[:160],
                )
            )

        if not citations:
            return GroundedResponse(response=response, citations=[], claim_sources={})

        if citation_format == "bracket":
            refs = ", ".join(f"[{i + 1}]" for i in used_indices)
            annotated = f"{response}\n\n[{refs}]"
        elif citation_format == "footnote":
            lines = [f"{i + 1}. {c.file_name or c.source_id}" for i, c in enumerate(citations) if i in used_indices]
            annotated = f"{response}\n\nSources: {'; '.join(lines)}"
        else:  # inline — mark sentences that quote a source verbatim
            annotated = response
            for i in used_indices:
                snippet = citations[i].text_snippet or ""
                if snippet and snippet in response:
                    marked = f"{snippet} [{i + 1}]"
                    annotated = annotated.replace(snippet, marked, 1)
            refs = ", ".join(f"[{i + 1}]" for i in used_indices)
            if "[" in annotated and "]" in annotated:
                annotated = f"{annotated}\n\nCited sources: {refs}"
            else:
                annotated = f"{response}\n\nCited sources: {refs}"

        claim_sources = {response[:120]: [str(c.source_id) for c in citations]}
        return GroundedResponse(response=annotated, citations=citations, claim_sources=claim_sources)

    @staticmethod
    def _overlaps(response: str, chunk_text: str) -> bool:
        """True when the response and a source share meaningful content words."""
        resp_words = _content_words(response)
        chunk_words = _content_words(chunk_text)
        if not chunk_words:
            return False
        return len(resp_words & chunk_words) / len(chunk_words) >= 0.15


class GroundingValidatorImpl(GroundingValidator):
    """Checks that the response's claims are supported by the sources."""

    def validate(
        self,
        response: str,
        sources: list[dict],
        *,
        require_citation_for_claims: bool = True,
    ) -> tuple[bool, list[str]]:
        del require_citation_for_claims
        source_text = " ".join(str(s.get("text", "")) for s in sources)
        source_words = _content_words(source_text)
        if not source_words:
            return True, []

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response)]
        ungrounded: list[str] = []
        for sentence in sentences:
            words = _content_words(sentence)
            if len(words) < 3:
                continue
            if sentence.startswith(("Sources:", "1.")):
                continue
            if "(not found in workspace)" in sentence:
                continue
            overlap = len(words & source_words) / len(words)
            if overlap < 0.25:
                ungrounded.append(sentence[:120])

        return len(ungrounded) == 0, ungrounded
