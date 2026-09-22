"""FileOperator — Phase 6 controlled filesystem access for the AI.

The operator is the ONLY way the Master Agent touches the user's disk. It is
rooted at a project's ``work/`` directory and every operation resolves the
requested relative path against that root, refusing anything that escapes:

- absolute paths (``C:\\x``, ``/etc``) — rejected
- parent traversal (``..``, ``a/../../b``) — rejected
- symlink escapes — rejected after ``resolve()``

Operations are atomic (temp file + rename) and create parent directories on
demand. ``WorkspaceSafetyError`` is raised on any violation; the orchestrator
turns that into a failed FileAction instead of letting it crash the request.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from synapse.logging import get_logger

log = get_logger("synapse.workspace.operator")


class WorkspaceSafetyError(PermissionError):
    """Raised when a filesystem operation would leave the workspace root."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_safe_relative_path(rel: str) -> bool:
    """True when ``rel`` stays inside a single root: relative, no ``..``,
    no absolute prefix, no drive letters, no empty segments issues."""
    if not isinstance(rel, str) or not rel or rel in (".", "/", "\\"):
        return False
    rel = rel.replace("\\", "/")
    if rel.startswith("/"):
        return False
    if ":" in rel:
        return False  # drive letters / scheme
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts or any(p in ("..",) for p in parts):
        return False
    if any(p in (".", "..") for p in rel.split("/")):
        return False
    return True


class FileOperator:
    """Safe CRUD inside one workspace folder. Never writes outside it.

    Phase 7: the root is the project's **workspace folder** (a plain user
    directory, possibly a Git repo). Git internals (``.git/``) are read-only
    for the operator: writes, renames, and deletes inside them are refused,
    and listings skip them.
    """

    def __init__(self, root: str | Path, *, create_dir: bool = True) -> None:
        self._root = Path(root).resolve()
        self._lock = threading.RLock()
        if create_dir:
            self._root.mkdir(parents=True, exist_ok=True)
        log.info("file_operator_ready", root=str(self._root))

    # -- paths ---------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, rel: str) -> Path:
        """Resolve ``rel`` to an absolute path, enforcing containment.

        Raises WorkspaceSafetyError when the path escapes the root or targets
        the workspace's Git internals.
        """
        if not is_safe_relative_path(rel):
            raise WorkspaceSafetyError(f"unsafe path rejected: {rel!r}")
        rel_norm = rel.replace("\\", "/").strip("/")
        if rel_norm.split("/", 1)[0] == ".git":
            raise WorkspaceSafetyError(f"git internals are read-only: {rel!r}")
        candidate = (self._root / rel_norm).resolve()
        if not candidate.is_relative_to(self._root):
            raise WorkspaceSafetyError(f"path escapes workspace root: {rel!r}")
        return candidate

    def exists(self, rel: str) -> bool:
        return self.path_for(rel).exists()

    # -- write ---------------------------------------------------------------

    def write(self, rel: str, content: str | bytes) -> dict:
        """Create or overwrite a file atomically. Returns metadata."""
        if not isinstance(content, (str, bytes)):
            raise TypeError("content must be str or bytes")
        path = self.path_for(rel)
        with self._lock:
            existed = path.exists()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f".{path.name}.tmp")
                if isinstance(content, bytes):
                    tmp.write_bytes(content)
                else:
                    tmp.write_text(content, encoding="utf-8")
                tmp.replace(path)
            except OSError as exc:
                log.warning("work_write_failed", path=rel, error=str(exc)[:160])
                raise
            size = path.stat().st_size
        action = "modified" if existed else "created"
        log.info("work_file_written", path=rel, action=action, bytes=size)
        return {"path": rel, "action": action, "bytes": size}

    def write_bytes(self, rel: str, data: bytes) -> dict:
        """Write binary data atomically to a file."""
        return self.write(rel, data)

    def read(self, rel: str) -> str | None:
        path = self.path_for(rel)
        if not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            data = path.read_bytes()
            for enc in ("utf-16", "latin-1"):
                try:
                    return data.decode(enc)
                except (UnicodeDecodeError, UnicodeError):
                    continue
            return data.decode("utf-8", errors="replace")

    def rename(self, src: str, dst: str) -> dict:
        """Move a file (or empty directory) inside the root."""
        src_path = self.path_for(src)
        dst_path = self.path_for(dst)
        with self._lock:
            if not src_path.exists():
                raise FileNotFoundError(f"source not found: {src}")
            if dst_path.exists():
                raise FileExistsError(f"destination exists: {dst}")
            try:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                src_path.rename(dst_path)
            except OSError as exc:
                log.warning("work_rename_failed", src=src, dst=dst, error=str(exc)[:160])
                raise
        log.info("work_file_renamed", src=src, dst=dst)
        return {"path": dst, "from": src, "action": "renamed"}

    def delete(self, rel: str) -> bool:
        path = self.path_for(rel)
        with self._lock:
            if not path.exists():
                return False
            try:
                if path.is_dir():
                    path.rmdir()
                else:
                    path.unlink()
            except OSError as exc:
                log.warning("work_delete_failed", path=rel, error=str(exc)[:160])
                raise
        log.info("work_file_deleted", path=rel)
        return True

    # -- listing ---------------------------------------------------------------

    def list_tree(self, rel: str = "") -> list[dict]:
        """Recursively list files under ``rel`` (default: whole work dir).

        Each entry: {path, name, dir, size, modified_at}.
        """
        base = self.path_for(rel) if rel else self._root
        if not base.is_dir():
            return []
        entries: list[dict] = []
        for child in sorted(base.rglob("*")):
            if not child.is_file():
                continue
            try:
                rel_path = str(child.relative_to(self._root)).replace("\\", "/")
            except ValueError:
                continue
            if rel_path.split("/", 1)[0] == ".git":
                continue
            stat = child.stat()
            entries.append(
                {
                    "path": rel_path,
                    "name": child.name,
                    "dir": str(child.parent.relative_to(self._root)).replace("\\", "/") if child.parent != self._root else "",
                    "size": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                }
            )
        return entries

    # -- structured export -----------------------------------------------------

    def export_manifest(self) -> dict:
        """JSON snapshot of the whole work tree (for backups/tests)."""
        tree: dict[str, str] = {}
        for entry in self.list_tree():
            content = self.read(entry["path"])
            if content is not None:
                tree[entry["path"]] = content
        return tree

    def import_manifest(self, tree: dict[str, str]) -> list[dict]:
        """Write many files from {relative_path: content}. Returns write results."""
        results = []
        for rel, content in tree.items():
            results.append(self.write(rel, content))
        return results
