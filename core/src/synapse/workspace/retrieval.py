"""Retrieval — semantic + keyword lookup over indexed workspace files.

Combines the Embedding Engine + Vector Store for RAG, deterministic pattern
scanning for source code (TODO/FIXME-style), and a context-block builder that
formats hits for prompt injection. Never sends file contents anywhere: this
component only reads the local index and the local file store.
"""

from __future__ import annotations

import logging
import re

from synapse.domain.workspace import CodeMatch, RetrievedChunk
from synapse.workspace.contracts import Retrieval as RetrievalContract

log = logging.getLogger("synapse.workspace.retrieval")

_WORD = re.compile(r"[a-z0-9_]{4,}")


class Retrieval(RetrievalContract):
    def __init__(self, embeddings, vector_store, files, settings) -> None:
        self._embeddings = embeddings
        self._store = vector_store
        self._files = files
        self._settings = settings

    # -- semantic retrieval -------------------------------------------------

    def retrieve(self, query: str, k: int = 4, file_ids: list[str] | None = None) -> list[RetrievedChunk]:
        if not query:
            return []
        k = max(1, k)
        vectors = self._embeddings.embed([query]) if self._embeddings else None
        if vectors and vectors[0]:
            hits = self._store.search(vectors[0], k=max(k, 32), file_ids=file_ids)
            threshold = self._settings.min_similarity
            return [
                RetrievedChunk(
                    file_id=h.metadata.get("file_id", ""),
                    file_name=h.metadata.get("file_name", ""),
                    chunk_index=int(h.metadata.get("chunk_index", 0) or 0),
                    text=h.text[:2000],
                    score=h.score,
                    page=h.metadata.get("page"),
                    lines_start=h.metadata.get("lines_start"),
                    lines_end=h.metadata.get("lines_end"),
                )
                for h in hits
                if h.score >= threshold
            ][:k]
        return self._keyword_search(query, k, file_ids)

    def _keyword_search(self, query: str, k: int, file_ids: list[str] | None) -> list[RetrievedChunk]:
        terms = {t for t in _WORD.findall(query.lower())}
        scored: list[tuple[float, RetrievedChunk]] = []
        for hit in self._store.all():
            meta = hit.metadata
            if file_ids and meta.get("file_id") not in file_ids:
                continue
            text = hit.text.lower()
            matches = sum(1 for term in terms if term in text)
            if not matches or not terms:
                continue
            score = min(1.0, matches / len(terms)) * 0.9
            scored.append((
                score,
                RetrievedChunk(
                    file_id=meta.get("file_id", ""),
                    file_name=meta.get("file_name", ""),
                    chunk_index=int(meta.get("chunk_index", 0) or 0),
                    text=hit.text[:2000],
                    score=round(score, 3),
                    page=meta.get("page"),
                    lines_start=meta.get("lines_start"),
                    lines_end=meta.get("lines_end"),
                ),
            ))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in scored[:k]]

    # -- deterministic code scan --------------------------------------------

    def scan_code(self, patterns: list[str], file_ids: list[str] | None = None) -> list[CodeMatch]:
        compiled = [re.compile(p, re.IGNORECASE) for p in patterns if p]
        if not compiled:
            return []
        matches: list[CodeMatch] = []
        for info in self._files.list():
            if info.pipeline != "code":
                continue
            if file_ids and info.id not in file_ids:
                continue
            data = self._files.read(info.id)
            if not data:
                continue
            try:
                text = data.decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if any(pattern.search(line) for pattern in compiled):
                    matches.append(
                        CodeMatch(file_id=info.id, file_name=info.name, line=lineno, text=line.strip()[:300])
                    )
        return matches

    # -- context building ---------------------------------------------------

    @staticmethod
    def render(chunks: list[RetrievedChunk]) -> str:
        parts: list[str] = []
        for chunk in chunks:
            location = []
            if chunk.page is not None:
                location.append(f"page {chunk.page}")
            if chunk.lines_start is not None:
                location.append(f"lines {chunk.lines_start}-{chunk.lines_end}")
            suffix = f" ({', '.join(location)})" if location else ""
            parts.append(f"[{chunk.file_name}{suffix}]\n{chunk.text}")
        return "\n\n".join(parts)

    def context(self, query: str, k: int = 4, file_ids: list[str] | None = None) -> str:
        return self.render(self.retrieve(query, k, file_ids))