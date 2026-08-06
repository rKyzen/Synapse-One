"""synapse.workspace — Phase 4 multimodal workspace subsystem.

Components (each replaceable behind a contract):
- File Manager:          files.WorkspaceFileManager
- Document Parser:       parsers.parse
- Chunking:              chunking.chunk_text, chunking.chunk_code
- Vision Pipeline:       vision.VisionPipeline
- Embedding Engine:      embeddings.EmbeddingEngine
- Vector Store:          vectors.FaissVectorStore, vectors.LocalVectorStore, vectors.create_vector_store
- Retrieval:             retrieval.Retrieval
- Indexing:              indexer.WorkspaceIndexer, jobs.IndexingJob, jobs.JobRegistry
- Facade:                workspace.Workspace
"""

from __future__ import annotations

from .config import VectorStoreSettings, WorkspaceSettings
from .embeddings import EmbeddingEngine
from .files import WorkspaceFileManager
from .indexer import WorkspaceIndexer
from .jobs import IndexingJob, JobRegistry
from .retrieval import Retrieval
from .vision import VisionPipeline
from .vectors import FaissVectorStore, LocalVectorStore, create_vector_store
from .workspace import Workspace

__all__ = [
    "Workspace",
    "WorkspaceSettings",
    "VectorStoreSettings",
    "WorkspaceFileManager",
    "EmbeddingEngine",
    "VisionPipeline",
    "Retrieval",
    "WorkspaceIndexer",
    "IndexingJob",
    "JobRegistry",
    "create_vector_store",
    "FaissVectorStore",
    "LocalVectorStore",
    "parse", "chunk_text", "chunk_code",
]

from .parsers import parse
from .chunking import chunk_text, chunk_code