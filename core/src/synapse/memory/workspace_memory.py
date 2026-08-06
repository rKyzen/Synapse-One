"""WorkspaceMemory — scoped, embedding-backed memory for the Master.

Stores entries per scope (conversation/project/global) with optional
embeddings produced by an embedding-capable provider. Search uses cosine
similarity when embeddings are available and falls back to keyword matching
otherwise. Persistence is a simple JSON file per scope under the data dir;
memory must never raise or break the pipeline.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog

from synapse.config.paths import SynapsePaths
from synapse.contracts import ConfigProvider, MemoryStore
from synapse.domain.enums import MemoryScope
from synapse.domain.memory import MemoryEntry

log = structlog.get_logger("synapse.memory")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkspaceMemory(MemoryStore):
    """File-backed scoped memory with optional semantic search."""

    def __init__(
        self,
        paths: SynapsePaths,
        config: ConfigProvider,
        providers,
        *,
        enabled: bool = True,
        top_k: int = 3,
        min_similarity: float = 0.35,
        embedding_model: str | None = None,
        root: Path | None = None,
    ) -> None:
        self._paths = paths
        self._config = config
        self._providers = providers
        self._enabled = enabled
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._embedding_model = embedding_model
        self._root = root
        self._entries: dict[MemoryScope, list[MemoryEntry]] = {s: [] for s in MemoryScope}
        self._lock = threading.Lock()
        if enabled:
            self._load_all()

    # -- contract -----------------------------------------------------------

    def save(
        self,
        scope: MemoryScope,
        text: str,
        *,
        embedding: list[float] | None = None,
        source: str = "",
        metadata: dict | None = None,
    ) -> MemoryEntry:
        if embedding is None:
            vectors = self._embed([text])
            embedding = vectors[0] if vectors else None
        entry = MemoryEntry(
            id=uuid.uuid4().hex[:12],
            scope=scope,
            text=text,
            embedding=embedding if embedding is not None else self._embed([text])[0] if self._embed([text]) else None,
            created_at=_now_iso(),
            source=source,
            metadata=metadata or {},
        )
        with self._lock:
            self._entries[scope].append(entry)
            self._save_scope(scope)
        log.info("memory_saved", scope=scope.value, source=source or "unknown", chars=len(text))
        return entry

    def search(
        self,
        scope: MemoryScope,
        query: str,
        *,
        k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[MemoryEntry]:
        k = k or self._top_k
        threshold = min_similarity if min_similarity is not None else self._min_similarity
        if not self._enabled or not self._entries[scope]:
            return []

        query_embedding = self._embed([query])
        with self._lock:
            entries = list(self._entries[scope])

        if query_embedding and query_embedding[0]:
            scored = [
                (self._cosine(query_embedding[0], e.embedding), e)
                for e in entries
                if e.embedding
            ]
            scored.sort(key=lambda pair: pair[0], reverse=True)
            results = [
                e.model_copy(update={"similarity": round(sim, 3)})
                for sim, e in scored
                if sim >= threshold
            ][:k]
        else:
            # Degraded: keyword overlap.
            terms = [t for t in re.findall(r"[a-z0-9]{4,}", query.lower())]
            scored = []
            for e in entries:
                text = e.text.lower()
                hits = sum(1 for term in terms if term in text)
                if hits:
                    sim = min(1.0, hits / max(1, len(terms))) * 0.9
                    scored.append((sim, e))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            results = [
                e.model_copy(update={"similarity": round(sim, 3)})
                for sim, e in scored
                if sim >= threshold
            ][:k]
        if results:
            log.info("memory_search", scope=scope.value, hits=len(results), query_len=len(query))
        return results

    def recent(self, scope: MemoryScope, limit: int = 10) -> list[MemoryEntry]:
        with self._lock:
            entries = list(self._entries[scope])
        return entries[-limit:][::-1]

    def clear(self, scope: MemoryScope | None = None) -> int:
        cleared = 0
        scopes = [scope] if scope else list(MemoryScope)
        with self._lock:
            for s in scopes:
                cleared += len(self._entries[s])
                self._entries[s] = []
                self._save_scope(s)
        return cleared

    # -- embeddings ---------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]] | None:
        provider = self._embedding_provider()
        if provider is None:
            return None
        try:
            return provider.embed(texts, model=self._embedding_model)
        except Exception:  # noqa: BLE001 - memory never breaks the pipeline
            log.debug("memory_embed_failed")
            return None

    def _embedding_provider(self):
        """First provider that exposes an embedding-capable model, or None."""
        for provider in self._providers.all():
            try:
                if provider.embed is not None and self._provider_has_embedding(provider):
                    return provider
            except Exception:  # noqa: BLE001
                continue
        return None

    def _provider_has_embedding(self, provider) -> bool:
        # Prefer the configured model if installed; else any embedding-capable
        # model the provider reports.
        installed = {d.id for d in provider.list_models()}
        if self._embedding_model and self._embedding_model in installed:
            return True
        try:
            for desc in provider.list_models():
                meta = provider.to_metadata(desc)
                if meta and meta.capabilities.embeddings > 0.9:
                    return True
        except Exception:  # noqa: BLE001
            return False
        return False

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # -- persistence --------------------------------------------------------

    def _file(self, scope: MemoryScope) -> Path:
        base = self._root if self._root is not None else self._paths.data_dir / "memory"
        return base / f"{scope.value}.json"

    def _load_all(self) -> None:
        for scope in MemoryScope:
            path = self._file(scope)
            try:
                if path.exists():
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    self._entries[scope] = [MemoryEntry(**item) for item in raw]
            except Exception:  # noqa: BLE001 - corrupt memory never breaks boot
                log.warning("memory_load_failed", scope=scope.value, path=str(path))

    def _save_scope(self, scope: MemoryScope) -> None:
        try:
            path = self._file(scope)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps([e.model_dump() for e in self._entries[scope]], indent=2),
                encoding="utf-8",
            )
            tmp.replace(path)
        except Exception:  # noqa: BLE001
            log.warning("memory_save_failed", scope=scope.value)