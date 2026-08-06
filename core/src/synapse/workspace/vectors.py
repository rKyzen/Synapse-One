"""Vector Store — local embedding index with two interchangeable backends.

- ``faiss``  : FAISS flat inner-product index over L2-normalized vectors
               (cosine similarity), persisted as a faiss index + JSON metadata.
- ``local``  : dependency-free numpy fallback with identical semantics, used
               automatically when FAISS is not importable.

Both backends share one contract (``synapse.workspace.contracts.VectorStore``)
and one on-disk layout under ``<data>/workspace/index/``, so swapping the
engine is a config change, not a code change.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from synapse.config.paths import SynapsePaths
from synapse.logging import get_logger
from synapse.workspace.contracts import VectorHit
from synapse.workspace.config import VectorStoreSettings

log = get_logger("synapse.workspace.vectors")

try:  # faiss is a soft dependency: the local backend covers missing installs
    import faiss  # type: ignore
    import numpy as np
except Exception:  # pragma: no cover - exercised on faiss-less machines
    faiss = None  # type: ignore
    np = None  # type: ignore


def _normalize(vector: list[float]) -> list[float]:
    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0:
        return list(vector)
    return [x / norm for x in vector]


class BaseVectorStore:
    """Shared persistence + metadata bookkeeping for both backends."""

    name = "base"

    def __init__(self, paths: SynapsePaths, settings: VectorStoreSettings, *, root: Path | None = None) -> None:
        base = root if root is not None else paths.data_dir / "workspace"
        self._dir = base / "index"
        self._meta_path = self._dir / f"{self.name}.meta.json"
        self._metas: list[dict] = []
        self._load_metas()

    def _load_metas(self) -> None:
        try:
            if self._meta_path.exists():
                raw = json.loads(self._meta_path.read_text(encoding="utf-8"))
                self._metas = [m for m in raw if isinstance(m, dict)]
        except Exception:  # noqa: BLE001
            log.warning("vector_meta_load_failed", path=str(self._meta_path))
            self._metas = []

    def _save_metas(self) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp = self._meta_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._metas, indent=2), encoding="utf-8")
            tmp.replace(self._meta_path)
        except Exception:  # noqa: BLE001
            log.warning("vector_meta_save_failed", path=str(self._meta_path))

    def _hit(self, index: int, score: float) -> VectorHit:
        meta = self._metas[index] if index < len(self._metas) else {}
        return VectorHit(
            id=meta.get("id", f"vec-{index}"),
            score=round(float(score), 4),
            metadata=meta,
            text=meta.get("text", ""),
        )

    def all(self) -> list[VectorHit]:
        return [self._hit(i, 1.0) for i in range(len(self._metas))]

    def count(self) -> int:
        return len(self._metas)

    def clear(self) -> None:
        self._metas = []
        self._save_metas()
        self._drop_index()

    def delete(self, file_id: str) -> None:
        keep = [i for i, m in enumerate(self._metas) if m.get("file_id") != file_id]
        self._metas = [self._metas[i] for i in keep]
        self._save_metas()
        self._rebuild(keep)

    def _drop_index(self) -> None:  # overridden by backends
        pass

    def _rebuild(self, keep_indices: list[int]) -> None:  # overridden
        pass


class FaissVectorStore(BaseVectorStore):
    """FAISS flat-IP backend (cosine) with numpy persistence."""

    name = "faiss"

    def __init__(self, paths: SynapsePaths, settings: VectorStoreSettings, *, root: Path | None = None) -> None:
        super().__init__(paths, settings, root=root)
        self._index_path = self._dir / "faiss.index"
        self._vectors_path = self._dir / "faiss.vectors.npy"
        self._vectors = self._load_vectors()
        self._index = self._build_index()

    def _load_vectors(self) -> list[list[float]]:
        if np is None:
            return []
        try:
            if self._vectors_path.exists():
                arr = np.load(self._vectors_path, allow_pickle=False)
                return [list(map(float, row)) for row in arr]
        except Exception:  # noqa: BLE001
            log.warning("vector_vectors_load_failed", path=str(self._vectors_path))
        return []

    def _build_index(self):
        if faiss is None or np is None or not self._vectors:
            return None
        index = faiss.IndexFlatIP(len(self._vectors[0]))
        if self._vectors:
            index.add(np.asarray(self._vectors, dtype="float32"))
        return index

    def add(self, vectors: list[list[float]], metadatas: list[dict]) -> None:
        if not vectors:
            return
        start = len(self._metas)
        for i, (vector, meta) in enumerate(zip(vectors, metadatas)):
            norm = _normalize(vector)
            meta = dict(meta)
            meta.setdefault("id", f"v-{uuid.uuid4().hex[:10]}")
            self._vectors.append(norm)
            self._metas.append(meta)
        if faiss is not None and np is not None:
            if self._index is None:
                self._index = faiss.IndexFlatIP(len(vectors[0]))
            self._index.add(np.asarray([_normalize(v) for v in vectors], dtype="float32"))
        self._save_metas()
        self._save_vectors()

    def search(self, vector: list[float], k: int = 4, file_ids: list[str] | None = None) -> list[VectorHit]:
        if not self._metas or not vector:
            return []
        if faiss is not None and np is not None and self._index is not None:
            query = np.asarray([_normalize(vector)], dtype="float32")
            scores, indices = self._index.search(query, min(k, self._index.ntotal))
            hits = [
                (float(scores[0][i]), int(indices[0][i]))
                for i in range(len(indices[0]))
                if int(indices[0][i]) >= 0
            ]
        else:
            hits = []
            for i, stored in enumerate(self._vectors):
                sim = sum(a * b for a, b in zip(_normalize(vector), stored))
                hits.append((float(sim), i))
            hits.sort(key=lambda pair: pair[0], reverse=True)
            hits = hits[:k]
        return self._filtered_hits(hits, file_ids)

    def _filtered_hits(self, hits, file_ids: list[str] | None) -> list[VectorHit]:
        out: list[VectorHit] = []
        for score, index in hits:
            if index >= len(self._metas):
                continue
            meta = self._metas[index]
            if file_ids and meta.get("file_id") not in file_ids:
                continue
            out.append(self._hit(index, score))
        return out[: len(self._metas)]

    def _save_vectors(self) -> None:
        if np is None or not self._vectors:
            return
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            # np.save appends ".npy" itself — give tmp a name that ends in .npy.
            tmp = self._vectors_path.with_name(self._vectors_path.name + ".tmp.npy")
            np.save(tmp, np.asarray(self._vectors, dtype="float32"))
            tmp.replace(self._vectors_path)
        except Exception:  # noqa: BLE001
            log.warning("vector_vectors_save_failed", path=str(self._vectors_path))

    def _drop_index(self) -> None:
        self._vectors = []
        self._index = None
        try:
            self._index_path.unlink(missing_ok=True)
            self._vectors_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _rebuild(self, keep_indices: list[int]) -> None:
        self._vectors = [self._vectors[i] for i in keep_indices if i < len(self._vectors)]
        self._index = self._build_index()
        self._save_vectors()


class LocalVectorStore(BaseVectorStore):
    """Numpy-only fallback with the same contract (no FAISS required)."""

    name = "local"

    def __init__(self, paths: SynapsePaths, settings: VectorStoreSettings, *, root: Path | None = None) -> None:
        super().__init__(paths, settings, root=root)
        self._vectors_path = self._dir / "local.vectors.json"
        self._vectors = self._load_vectors()

    def _load_vectors(self) -> list[list[float]]:
        try:
            if self._vectors_path.exists():
                raw = json.loads(self._vectors_path.read_text(encoding="utf-8"))
                return [list(map(float, row)) for row in raw]
        except Exception:  # noqa: BLE001
            log.warning("local_vectors_load_failed", path=str(self._vectors_path))
        return []

    def add(self, vectors: list[list[float]], metadatas: list[dict]) -> None:
        if not vectors:
            return
        for vector, meta in zip(vectors, metadatas):
            meta = dict(meta)
            meta.setdefault("id", f"v-{uuid.uuid4().hex[:10]}")
            self._vectors.append(_normalize(vector))
            self._metas.append(meta)
        self._save_vectors()
        self._save_metas()

    def search(self, vector: list[float], k: int = 4, file_ids: list[str] | None = None) -> list[VectorHit]:
        if not self._metas or not vector:
            return []
        query = _normalize(vector)
        scored = []
        for i, stored in enumerate(self._vectors):
            sim = sum(a * b for a, b in zip(query, stored))
            scored.append((sim, i))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return self._filtered_hits(scored, file_ids)

    def _filtered_hits(self, scored, file_ids: list[str] | None) -> list[VectorHit]:
        out: list[VectorHit] = []
        for score, index in scored:
            if index >= len(self._metas):
                continue
            meta = self._metas[index]
            if file_ids and meta.get("file_id") not in file_ids:
                continue
            out.append(self._hit(index, score))
        return out

    def _save_vectors(self) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp = self._vectors_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._vectors), encoding="utf-8")
            tmp.replace(self._vectors_path)
        except Exception:  # noqa: BLE001
            log.warning("local_vectors_save_failed", path=str(self._vectors_path))

    def _drop_index(self) -> None:
        self._vectors = []
        try:
            self._vectors_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _rebuild(self, keep_indices: list[int]) -> None:
        self._vectors = [self._vectors[i] for i in keep_indices if i < len(self._vectors)]
        self._save_vectors()


def create_vector_store(paths: SynapsePaths, settings: VectorStoreSettings, *, root: Path | None = None):
    """Instantiate the configured backend, falling back to ``local``."""
    provider = (settings.provider or "faiss").lower()
    if provider == "faiss" and faiss is not None:
        return FaissVectorStore(paths, settings, root=root)
    if provider == "faiss" and faiss is None:
        log.info("vector_store_faiss_unavailable_using_local")
    return LocalVectorStore(paths, settings, root=root)