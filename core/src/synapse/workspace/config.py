"""Workspace + vector-store settings — parsed from the raw config dict.

Everything is config-driven: file-type → pipeline mapping, chunk sizes,
retrieval thresholds, the embedding model, and the vision model all come
from ``config.toml``. Values below are only shipped defaults; nothing about a
specific model or file type is hardcoded as a runtime behavior.
"""

from __future__ import annotations

import copy

from pydantic import BaseModel, Field

#: Default file-type → pipeline mapping (overridable in [workspace.pipelines]).
_DEFAULT_PIPELINES: dict[str, list[str]] = {
    "image": ["png", "jpg", "jpeg", "webp", "gif", "bmp"],
    "document": ["pdf", "docx", "txt", "md", "markdown", "rtf"],
    "code": [
        "py", "js", "ts", "dart", "rs", "go", "java", "c", "cpp", "h", "hpp",
        "cs", "rb", "php", "sh", "bash", "ps1", "sql", "html", "css", "scss",
        "json", "jsonc", "yaml", "yml", "toml", "xml", "proto", "kt", "swift",
        "m", "mm", "lua", "r", "jl", "clj", "ex", "exs", "groovy",
    ],
}


class WorkspaceSettings(BaseModel):
    enabled: bool = True
    storage_dir: str = "workspace"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    code_chunk_lines: int = 200
    code_chunk_overlap: int = 20
    embedding_model: str = "nomic-embed-text"
    vision_model: str = "qwen2.5vl:7b"
    top_k: int = 4
    min_similarity: float = 0.25
    embed_batch_size: int = 16
    allow_cloud_forwarding: bool = False
    auto_retrieve: bool = True
    code_scan_patterns: list[str] = Field(default_factory=lambda: ["TODO", "FIXME", "HACK", "XXX"])
    code_scan_triggers: list[str] = Field(default_factory=lambda: ["todo", "fixme", "hack"])
    #: file-type → pipeline (image | document | code). Empty falls back to defaults.
    pipelines: dict[str, list[str]] = Field(default_factory=dict)

    @classmethod
    def from_config(cls, raw: dict | None) -> "WorkspaceSettings":
        raw = dict(raw or {})
        pipelines = copy.deepcopy(_DEFAULT_PIPELINES)
        configured = raw.pop("pipelines", {})
        if isinstance(configured, dict):
            for kind, exts in configured.items():
                if isinstance(exts, list):
                    pipelines[kind] = [str(e).lower().lstrip(".") for e in exts]
        known = set(cls.model_fields)
        fields = {k: v for k, v in raw.items() if k in known}
        return cls(**fields, pipelines=pipelines)

    def pipeline_for(self, extension: str) -> str:
        """Map a file extension to its pipeline kind without case sensitivity."""
        ext = extension.lower().lstrip(".")
        for kind, exts in self.pipelines.items():
            if ext in exts:
                return kind
        return "document"


class VectorStoreSettings(BaseModel):
    provider: str = "faiss"  # faiss | local

    @classmethod
    def from_config(cls, raw: dict | None) -> "VectorStoreSettings":
        raw = dict(raw or {})
        return cls(**{k: v for k, v in raw.items() if k in cls.model_fields})