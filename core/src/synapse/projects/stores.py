"""Shared JSON entity store — durable, atomic per-entity persistence.

Phase B introduces the workspace entity stores (notes, documents, todos).
They share one persistence pattern with ChatStore/GoalStore: one JSON file
per entity under ``<internal>/<bucket>/<project_id>/<entity_id>.json``,
written atomically (tmp + replace) so every mutation is durable immediately.
This base class captures that pattern for the NEW entity stores only —
ChatStore and GoalStore keep their standalone implementations untouched.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel

from synapse.logging import get_logger

log = get_logger("synapse.projects.stores")

T = TypeVar("T", bound=BaseModel)


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


class JsonEntityStore(Generic[T]):
    """File-backed storage for one entity type, scoped to a project."""

    def __init__(
        self,
        project_id: str,
        root: Path,
        entity_cls: type[T],
        *,
        bucket_name: str = "entity",
    ) -> None:
        self._project_id = project_id
        self._dir = root
        self._entity_cls = entity_cls
        self._bucket = bucket_name
        self._lock = threading.RLock()
        self._dir.mkdir(parents=True, exist_ok=True)

    # -- paths / persistence -------------------------------------------------

    def _path(self, entity_id: str) -> Path:
        return self._dir / f"{entity_id}.json"

    def _read(self, entity_id: str) -> T | None:
        path = self._path(entity_id)
        try:
            if not path.exists():
                return None
            entity = self._entity_cls(**json.loads(path.read_text(encoding="utf-8")))
            entity.project_id = self._project_id
            return entity
        except Exception:  # noqa: BLE001 - a corrupt entity never breaks a project
            log.warning(
                f"{self._bucket}_load_failed", entity_id=entity_id, path=str(path)
            )
            return None

    def _write(self, entity: T) -> None:
        try:
            tmp = self._path(entity.id).with_suffix(".json.tmp")
            tmp.write_text(json.dumps(entity.model_dump(), indent=2), encoding="utf-8")
            tmp.replace(self._path(entity.id))
        except Exception:  # noqa: BLE001
            log.warning(f"{self._bucket}_save_failed", entity_id=entity.id)

    # -- API -----------------------------------------------------------------

    def list(self) -> list[T]:
        with self._lock:
            out = []
            for path in sorted(self._dir.glob("*.json")):
                if path.name.endswith(".tmp"):
                    continue
                entity = self._read(path.stem)
                if entity is not None:
                    out.append(entity)
            out.sort(key=lambda e: e.updated_at, reverse=True)
            return out

    def get(self, entity_id: str) -> T | None:
        with self._lock:
            return self._read(entity_id)

    def new_id(self) -> str:
        return _uuid()

    def put(self, entity: T) -> T:
        """Persist a fully-formed entity (used by the typed subclasses)."""
        with self._lock:
            self._write(entity)
        return entity

    def delete(self, entity_id: str) -> bool:
        with self._lock:
            path = self._path(entity_id)
            try:
                if path.exists():
                    path.unlink()
                    log.info(f"{self._bucket}_deleted", entity_id=entity_id,
                             project_id=self._project_id)
                    return True
            except OSError:
                log.warning(f"{self._bucket}_unlink_failed", entity_id=entity_id)
            return False
