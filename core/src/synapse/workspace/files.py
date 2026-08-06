"""Workspace File Manager — ingestion, sha256 dedupe, catalog, storage.

Files land under ``<data>/workspace/files/<file_id>.<ext>`` with a
``catalog.json`` index. Nothing here talks to models or vendors: it is pure
storage bookkeeping with atomic persistence and never raises on I/O errors.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from synapse.config.paths import SynapsePaths
from synapse.domain.workspace import WorkspaceFileInfo
from synapse.logging import get_logger

log = get_logger("synapse.workspace.files")

MIME_BY_EXT: dict[str, str] = {
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp",
    "gif": "image/gif", "bmp": "image/bmp",
    "pdf": "application/pdf", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain", "md": "text/markdown", "markdown": "text/markdown", "rtf": "application/rtf",
    "py": "text/x-python", "js": "text/javascript", "ts": "text/typescript", "html": "text/html",
    "css": "text/css", "json": "application/json", "yaml": "text/yaml", "yml": "text/yaml",
    "toml": "text/toml", "xml": "text/xml", "sh": "text/x-shellscript", "ps1": "text/x-powershell",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkspaceFileManager:
    """Catalog + blob storage for uploaded workspace files."""

    def __init__(self, paths: SynapsePaths, settings, *, root: Path | None = None) -> None:
        self._paths = paths
        self._settings = settings
        self._root = root if root is not None else paths.data_dir / "workspace"
        self._files_dir = self._root / "files"
        self._catalog_path = self._root / "catalog.json"
        self._catalog: dict[str, WorkspaceFileInfo] = {}
        self._load()

    # -- paths --------------------------------------------------------------

    def _blob_path(self, file_id: str, extension: str) -> Path:
        return self._files_dir / f"{file_id}.{extension}"

    # -- catalog persistence -----------------------------------------------

    def _load(self) -> None:
        try:
            if self._catalog_path.exists():
                raw = json.loads(self._catalog_path.read_text(encoding="utf-8"))
                self._catalog = {e["id"]: WorkspaceFileInfo(**e) for e in raw}
        except Exception:  # noqa: BLE001
            log.warning("catalog_load_failed", path=str(self._catalog_path))
            self._catalog = {}

    def _save_catalog(self) -> None:
        try:
            self._catalog_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._catalog_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps([e.model_dump() for e in self._catalog.values()], indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._catalog_path)
        except Exception:  # noqa: BLE001
            log.warning("catalog_save_failed", path=str(self._catalog_path))

    # -- FileStore contract -------------------------------------------------

    def upload(self, name: str, data: bytes) -> WorkspaceFileInfo:
        import hashlib

        digest = hashlib.sha256(data).hexdigest()
        existing = next((e for e in self._catalog.values() if e.sha256 == digest), None)
        if existing is not None:
            log.info("file_deduped", file_id=existing.id, name=name)
            return existing

        ext = Path(name).suffix.lstrip(".").lower()
        file_id = uuid.uuid4().hex[:12]
        blob = self._blob_path(file_id, ext or "bin")
        try:
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(data)
        except OSError:
            log.warning("file_write_failed", name=name)
            raise

        entry = WorkspaceFileInfo(
            id=file_id,
            name=name,
            extension=ext,
            pipeline=self._settings.pipeline_for(ext),
            size_bytes=len(data),
            sha256=digest,
            mime_type=MIME_BY_EXT.get(ext, "application/octet-stream"),
            status="uploaded",
            created_at=_now(),
        )
        self._catalog[file_id] = entry
        self._save_catalog()
        log.info("file_uploaded", file_id=file_id, name=name, pipeline=entry.pipeline, bytes=len(data))
        return entry

    def get(self, file_id: str) -> WorkspaceFileInfo | None:
        return self._catalog.get(file_id)

    def list(self) -> list[WorkspaceFileInfo]:
        return [e.model_copy() for e in self._catalog.values()]

    def read(self, file_id: str) -> bytes | None:
        entry = self._catalog.get(file_id)
        if entry is None:
            return None
        blob = self._blob_path(file_id, entry.extension or "bin")
        try:
            return blob.read_bytes()
        except OSError:
            return None

    def update(self, entry: WorkspaceFileInfo) -> None:
        if entry.id not in self._catalog:
            log.debug("file_update_skipped_deleted", file_id=entry.id)
            return
        self._catalog[entry.id] = entry
        self._save_catalog()

    def delete(self, file_id: str) -> bool:
        entry = self._catalog.pop(file_id, None)
        if entry is None:
            return False
        blob = self._blob_path(file_id, entry.extension or "bin")
        try:
            if blob.exists():
                blob.unlink()
        except OSError:
            log.warning("file_unlink_failed", file_id=file_id)
        self._save_catalog()
        log.info("file_deleted", file_id=file_id, name=entry.name)
        return True