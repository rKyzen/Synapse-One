"""Phase 4 workspace contracts — replaceable interfaces for every component.

Each subsystem (File Manager, Document Parser, Vision Pipeline, Embedding
Engine, Vector Store, Retrieval) is pluggable behind a small protocol. The
diagram of modules/workspace.py shows how they compose at runtime.
"""

from __future__ import annotations

from typing import Protocol

from synapse.domain.workspace import CodeMatch, RetrievedChunk, WorkspaceFileInfo


class FileStore(Protocol):
    """File Manager — ingestion, dedupe, catalog, storage layout."""

    def upload(self, name: str, data: bytes) -> WorkspaceFileInfo: ...
    def get(self, file_id: str) -> WorkspaceFileInfo | None: ...
    def list(self) -> list[WorkspaceFileInfo]: ...
    def read(self, file_id: str) -> bytes | None: ...
    def delete(self, file_id: str) -> bool: ...


class ParsedDocument:
    """Output of a Document Parser: full text plus optional page segments."""

    def __init__(self, text: str, pages: list[str] | None = None, metadata: dict | None = None) -> None:
        self.text = text
        self.pages = pages or ([text] if text else [])
        self.metadata = metadata or {}


class DocumentParser(Protocol):
    """Document Parser — extracts text from one file kind."""

    def parse(self, data: bytes, extension: str) -> ParsedDocument: ...


class VisionPipeline(Protocol):
    """Vision Pipeline — answers questions about an image via a vision model."""

    def analyze(self, data: bytes, prompt: str) -> str: ...


class EmbeddingEngine(Protocol):
    """Embedding Engine — text vectors via a local embedding model."""

    def embed(self, texts: list[str]) -> list[list[float]] | None: ...


class VectorHit:
    """One vector-store hit with its metadata."""

    __slots__ = ("id", "score", "metadata", "text")

    def __init__(self, id: str, score: float, metadata: dict | None = None, text: str = "") -> None:
        self.id = id
        self.score = score
        self.metadata = metadata or {}
        self.text = text


class VectorStore(Protocol):
    """Vector Store — local embedding index with persistence."""

    def add(self, vectors: list[list[float]], metadatas: list[dict]) -> None: ...
    def search(self, vector: list[float], k: int = 4, file_ids: list[str] | None = None) -> list[VectorHit]: ...
    def delete(self, file_id: str) -> None: ...
    def all(self) -> list[VectorHit]: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...


class Retrieval(Protocol):
    """Retrieval — semantic + keyword lookup over indexed workspace files."""

    def retrieve(self, query: str, k: int = 4, file_ids: list[str] | None = None) -> list[RetrievedChunk]: ...
    def scan_code(self, patterns: list[str], file_ids: list[str] | None = None) -> list[CodeMatch]: ...
    def context(self, query: str, k: int = 4, file_ids: list[str] | None = None) -> str: ...