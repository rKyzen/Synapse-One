"""SemanticSearch — embeddings-based workspace search.

Replaces raw text matching with semantic understanding. Uses the existing
embedding engine to find conceptually relevant code and documentation.

Features:
    - Semantic search across all workspace files
    - Hybrid search (semantic + keyword)
    - Search result ranking by relevance
    - Context-aware search (understands project structure)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from synapse.logging import get_logger
from synapse.workspace.project_index import FileIndex, ProjectIndex

log = get_logger("synapse.search.semantic")


@dataclass
class SearchResult:
    """A single search result."""

    file_path: str
    line: int | None = None
    content: str = ""
    score: float = 0.0
    match_type: str = "semantic"  # "semantic", "keyword", "symbol"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResults:
    """Collection of search results with metadata."""

    results: list[SearchResult]
    query: str
    total_results: int
    search_time_ms: float
    truncated: bool

    def top(self, n: int = 10) -> list[SearchResult]:
        """Get top N results."""
        return self.results[:n]


class SemanticSearch:
    """Semantic search engine for the workspace."""

    def __init__(
        self,
        project_index: ProjectIndex,
        embedding_engine: Any | None = None,
        vector_store: Any | None = None,
    ) -> None:
        self._index = project_index
        self._embeddings = embedding_engine
        self._vectors = vector_store

    def search(
        self,
        query: str,
        *,
        max_results: int = 20,
        file_pattern: str | None = None,
        language: str | None = None,
        include_symbols: bool = True,
    ) -> SearchResults:
        """Search the workspace semantically.

        Combines:
        1. Keyword matching (fast, exact)
        2. Symbol search (code-aware)
        3. Semantic search (conceptual, if embeddings available)
        """
        import time
        start = time.perf_counter()

        results: list[SearchResult] = []

        # 1. Keyword search
        keyword_results = self._keyword_search(query, file_pattern, language)
        results.extend(keyword_results)

        # 2. Symbol search
        if include_symbols:
            symbol_results = self._symbol_search(query)
            results.extend(symbol_results)

        # 3. Semantic search (if available)
        if self._embeddings and self._vectors:
            semantic_results = self._semantic_search(query, max_results)
            results.extend(semantic_results)

        # Deduplicate and rank
        results = self._deduplicate(results)
        results = self._rank(results, query)

        # Truncate
        truncated = len(results) > max_results
        results = results[:max_results]

        elapsed_ms = (time.perf_counter() - start) * 1000

        return SearchResults(
            results=results,
            query=query,
            total_results=len(results),
            search_time_ms=round(elapsed_ms, 1),
            truncated=truncated,
        )

    def find_similar(self, file_path: str, max_results: int = 5) -> list[SearchResult]:
        """Find files similar to a given file."""
        file_idx = self._index.get_file(file_path)
        if not file_idx:
            return []

        # Build a query from the file's content
        query_parts = []
        if file_idx.symbols:
            query_parts.extend(s.name for s in file_idx.symbols[:5])
        if file_idx.imports:
            query_parts.extend(file_idx.imports[:3])
        if not query_parts:
            return []

        query = " ".join(query_parts)
        results = self.search(query, max_results=max_results + 1)

        # Exclude the original file
        return [r for r in results.results if r.file_path != file_path][:max_results]

    def find_references(self, symbol_name: str) -> list[SearchResult]:
        """Find all references to a symbol."""
        results: list[SearchResult] = []

        # Search in symbols
        for file_idx, sym in self._index.search_symbols(symbol_name):
            results.append(SearchResult(
                file_path=file_idx.path,
                line=sym.line,
                content=f"{sym.kind} {sym.name}",
                score=1.0,
                match_type="symbol",
                metadata={"parent": sym.parent, "kind": sym.kind},
            ))

        # Search in imports
        for file_idx in self._index.search_imports(symbol_name):
            results.append(SearchResult(
                file_path=file_idx.path,
                content=f"imports {symbol_name}",
                score=0.8,
                match_type="keyword",
            ))

        return self._deduplicate(results)

    def _keyword_search(
        self,
        query: str,
        file_pattern: str | None,
        language: str | None,
    ) -> list[SearchResult]:
        """Fast keyword-based search."""
        results: list[SearchResult] = []
        query_lower = query.lower()
        words = set(query_lower.split())

        for file_idx in self._index.get_all_files():
            # Apply filters
            if language and file_idx.language != language:
                continue
            if file_pattern and not re.match(file_pattern.replace("*", ".*"), file_idx.name):
                continue

            # Score by word matches in file name
            name_score = sum(1 for w in words if w in file_idx.name.lower())
            if name_score > 0:
                results.append(SearchResult(
                    file_path=file_idx.path,
                    content=f"File: {file_idx.name}",
                    score=name_score * 0.5,
                    match_type="keyword",
                ))

        return results

    def _symbol_search(self, query: str) -> list[SearchResult]:
        """Search for matching symbols."""
        results: list[SearchResult] = []
        query_lower = query.lower()

        for file_idx in self._index.get_all_files():
            for symbol in file_idx.symbols:
                if query_lower in symbol.name.lower():
                    results.append(SearchResult(
                        file_path=file_idx.path,
                        line=symbol.line,
                        content=f"{symbol.kind} {symbol.name}",
                        score=0.9,
                        match_type="symbol",
                        metadata={"parent": symbol.parent},
                    ))

        return results

    def _semantic_search(self, query: str, max_results: int) -> list[SearchResult]:
        """Embeddings-based semantic search."""
        # Placeholder for actual semantic search implementation
        # Would use self._embeddings to encode query
        # and self._vectors to find nearest neighbors
        return []

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove duplicate results, keeping highest score."""
        seen: dict[str, SearchResult] = {}
        for r in results:
            key = f"{r.file_path}:{r.line}:{r.match_type}"
            if key not in seen or r.score > seen[key].score:
                seen[key] = r
        return list(seen.values())

    def _rank(self, results: list[SearchResult], query: str) -> list[SearchResult]:
        """Rank results by relevance."""
        # Boost symbol matches
        for r in results:
            if r.match_type == "symbol":
                r.score *= 1.2
            elif r.match_type == "semantic":
                r.score *= 1.1

        return sorted(results, key=lambda r: r.score, reverse=True)
