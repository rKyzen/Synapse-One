"""ProjectManager — SQLite project registry + workspace folder lifecycle.

Phase 7 architecture:

- **Registry** — ``<internal>/data/projects.db`` (SQLite) stores ONLY the
  absolute workspace path per project. No Synapse metadata, chats, memory, or
  indexes are ever written into the user's folder.
- **Workspace folder** — a plain directory the user chose (or, when no parent
  is given, an internal scratch location). It contains only the project's
  files: a ``README.md`` seed file and a Git repository (best-effort ``git
  init``). The agent's generated files land here too, via FileOperator.
- **Unique ids** — slugs stay human-friendly and unique (collisions get a
  ``-2`` suffix); all internal per-project storage (chats, memory, vectors,
  logs) is keyed by this id.
- **Lifecycle** — ``create`` makes the folder + seeds it, ``delete`` can
  either unregister only (keeping the workspace intact) or erase the folder,
  ``reconnect`` repoints a registry entry to a moved/relocated workspace,
  ``verify`` reports workspaces that are missing on disk.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from synapse.domain.projects import ProjectInfo
from synapse.logging import get_logger

log = get_logger("synapse.projects.manager")

DEFAULT_PROJECT_ID = "general"
DEFAULT_PROJECT_NAME = "General"
DB_FILE = "projects.db"
INTERNAL_WORKSPACES_DIR = "workspaces"
SEED_README = "# {name}\n\nWorkspace created by Synapse.\n"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sqlite_rows(row: sqlite3.Row) -> dict:
    return dict(row)


def safe_folder_name(name: str) -> str:
    """Filesystem-safe folder name derived from a project name.

    Unlike ``slugify`` this keeps spaces and dashes — ``"Portfolio Website"``
    becomes a folder called ``Portfolio Website``.
    """
    safe = "".join(ch for ch in name if ch.isalnum() or ch in (" ", "-", "_", ".")).strip()
    safe = safe.rstrip(" .")  # Windows forbids trailing dots/spaces
    return safe or "project"


class ProjectManager:
    """Owns the registry database and the workspace folder lifecycle."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._db_path = self._root / DB_FILE
        self._workspaces_root = self._root / INTERNAL_WORKSPACES_DIR
        self._lock = threading.RLock()
        self._init_db()

    # -- database ------------------------------------------------------------

    def _init_db(self) -> None:
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            with self._conn() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS projects (
                        id            TEXT PRIMARY KEY,
                        name          TEXT NOT NULL,
                        workspace_path TEXT NOT NULL,
                        created_at    TEXT NOT NULL,
                        updated_at    TEXT NOT NULL,
                        archived      INTEGER NOT NULL DEFAULT 0,
                        settings      TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
        except Exception:  # noqa: BLE001
            log.warning("project_db_init_failed", path=str(self._db_path))
            raise

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # -- paths ---------------------------------------------------------------

    def workspace_path(self, project_id: str) -> Path:
        """Absolute workspace folder for a project ('' when unknown)."""
        row = self._fetch_one("SELECT workspace_path FROM projects WHERE id = ?", (project_id,))
        if row is None:
            return Path()
        return Path(row["workspace_path"])

    def default_workspace_path(self, name: str, project_id: str | None = None) -> Path:
        """Where a project created without a user-chosen parent lives."""
        pid = project_id or slugify(name)
        return self._workspaces_root / pid

    # -- CRUD ----------------------------------------------------------------

    def create(self, name: str, parent_dir: str | Path | None = None) -> ProjectInfo:
        """Create a workspace folder and register it.

        ``parent_dir`` — the folder the user picked with the native dialog;
        Synapse creates ``<parent_dir>/<Name>`` inside it. When omitted the
        workspace lives under the internal workspaces root.
        """
        name = (name or "").strip() or "Untitled"
        with self._lock:
            project_id = self._unique_slug(slugify(name))
            if parent_dir:
                base = Path(parent_dir).expanduser().resolve()
                workspace = base / safe_folder_name(name)
            else:
                base = None
                workspace = self.default_workspace_path(name, project_id)
            self._prepare_workspace(workspace, name)
            self._upsert(
                project_id,
                name,
                str(workspace),
                archived=False,
                settings={},
                created_at=_now_iso(),
            )
            log.info(
                "project_created",
                project_id=project_id,
                name=name,
                workspace=str(workspace),
                user_chosen=bool(base),
            )
            return self.get(project_id)

    def register(self, name: str, workspace_path: str | Path) -> ProjectInfo:
        """Register an EXISTING user folder as a project workspace."""
        name = (name or "").strip() or "Untitled"
        workspace = Path(workspace_path).expanduser().resolve()
        if not workspace.is_dir():
            raise ValueError(f"workspace folder does not exist: {workspace}")
        with self._lock:
            project_id = self._unique_slug(slugify(name))
            self._upsert(
                project_id,
                name,
                str(workspace),
                archived=False,
                settings={},
                created_at=_now_iso(),
            )
            log.info("project_registered", project_id=project_id, workspace=str(workspace))
            return self.get(project_id)

    def reconnect(self, project_id: str, workspace_path: str | Path) -> ProjectInfo | None:
        """Repoint a project to its workspace after it was moved/relocated."""
        workspace = Path(workspace_path).expanduser().resolve()
        if not workspace.is_dir():
            raise ValueError(f"workspace folder does not exist: {workspace}")
        with self._lock:
            row = self._fetch_one("SELECT id FROM projects WHERE id = ?", (project_id,))
            if row is None:
                return None
            self._update(
                project_id,
                workspace_path=str(workspace),
                updated_at=_now_iso(),
            )
            log.info("project_reconnected", project_id=project_id, workspace=str(workspace))
            return self.get(project_id)

    def rename(self, project_id: str, name: str) -> ProjectInfo | None:
        """Rename the display name. The workspace folder name is untouched —
        it is the user's folder, not Synapse's to move."""
        name = (name or "").strip()
        if not name:
            return None
        with self._lock:
            row = self._fetch_one("SELECT id FROM projects WHERE id = ?", (project_id,))
            if row is None:
                return None
            self._update(project_id, name=name, updated_at=_now_iso())
            log.info("project_renamed", project_id=project_id, name=name)
            return self.get(project_id)

    def update_settings(self, project_id: str, settings: dict) -> ProjectInfo | None:
        """Merge project settings (e.g. detected stack) into the registry."""
        with self._lock:
            row = self._fetch_one("SELECT settings FROM projects WHERE id = ?", (project_id,))
            if row is None:
                return None
            try:
                current = json.loads(row["settings"] or "{}")
            except Exception:  # noqa: BLE001
                current = {}
            current.update(settings or {})
            self._update(
                project_id,
                settings=json.dumps(current),
                updated_at=_now_iso(),
            )
            return self.get(project_id)

    def touch(self, project_id: str) -> ProjectInfo | None:
        """Bump ``updated_at`` (e.g. after a workspace scan)."""
        with self._lock:
            row = self._fetch_one("SELECT id FROM projects WHERE id = ?", (project_id,))
            if row is None:
                return None
            self._update(project_id, updated_at=_now_iso())
            return self.get(project_id)

    def archive(self, project_id: str) -> ProjectInfo | None:
        with self._lock:
            row = self._fetch_one("SELECT id FROM projects WHERE id = ?", (project_id,))
            if row is None:
                return None
            self._update(project_id, archived=True, updated_at=_now_iso())
            log.info("project_archived", project_id=project_id)
            return self.get(project_id)

    def unarchive(self, project_id: str) -> ProjectInfo | None:
        with self._lock:
            row = self._fetch_one("SELECT id FROM projects WHERE id = ?", (project_id,))
            if row is None:
                return None
            self._update(project_id, archived=False, updated_at=_now_iso())
            return self.get(project_id)

    def delete(self, project_id: str, *, delete_workspace: bool = False) -> bool:
        """Remove a project from the registry.

        ``delete_workspace=False`` keeps the user's folder on disk (the
        "remove only from Synapse" mode); ``True`` permanently erases it.
        """
        with self._lock:
            row = self._fetch_one("SELECT workspace_path FROM projects WHERE id = ?", (project_id,))
            if row is None:
                return False
            workspace = Path(row["workspace_path"])
            self._exec("DELETE FROM projects WHERE id = ?", (project_id,))
            if delete_workspace:
                try:
                    shutil.rmtree(workspace, ignore_errors=True)
                    log.info("workspace_erased", project_id=project_id, path=str(workspace))
                except OSError:
                    log.warning("workspace_erase_failed", project_id=project_id)
            log.info("project_deleted", project_id=project_id, delete_workspace=delete_workspace)
            return True

    # -- reads ---------------------------------------------------------------

    def list_ids(self, *, include_archived: bool = False) -> list[str]:
        if include_archived:
            rows = self._fetch_all("SELECT id FROM projects ORDER BY name COLLATE NOCASE")
        else:
            rows = self._fetch_all(
                "SELECT id FROM projects WHERE archived = 0 ORDER BY name COLLATE NOCASE"
            )
        return [r["id"] for r in rows]

    def get(self, project_id: str) -> ProjectInfo | None:
        row = self._fetch_one("SELECT * FROM projects WHERE id = ?", (project_id,))
        return self.to_info(row) if row else None

    # -- health ---------------------------------------------------------------

    def verify(self) -> dict[str, bool]:
        """Map of project id -> workspace-folder-exists. Used to surface
        moved/missing workspaces so the UI can offer reconnect."""
        result: dict[str, bool] = {}
        with self._lock:
            rows = self._fetch_all("SELECT id, workspace_path FROM projects")
            for row in rows:
                result[row["id"]] = Path(row["workspace_path"]).is_dir()
        return result

    # -- persistence helpers ---------------------------------------------------

    def _prepare_workspace(self, workspace: Path, name: str) -> None:
        """Create the folder, seed a README, and best-effort ``git init``."""
        if workspace.exists():
            if not workspace.is_dir():
                raise ValueError(f"cannot create project folder: {workspace} is not a directory")
            if any(workspace.iterdir()):
                log.warning("workspace_not_empty", path=str(workspace))
        else:
            workspace.mkdir(parents=True, exist_ok=True)
        readme = workspace / "README.md"
        if not readme.exists():
            try:
                readme.write_text(SEED_README.format(name=name), encoding="utf-8")
            except OSError:
                log.warning("readme_seed_failed", path=str(workspace))
        if not (workspace / ".git").exists() and shutil.which("git"):
            try:
                import subprocess

                subprocess.run(
                    ["git", "init", "-q", str(workspace)],
                    check=False,
                    capture_output=True,
                    timeout=30,
                )
            except Exception:  # noqa: BLE001 - git is optional
                log.debug("git_init_skipped", path=str(workspace))

    def _upsert(
        self,
        project_id: str,
        name: str,
        workspace_path: str,
        *,
        archived: bool,
        settings: dict,
        created_at: str,
    ) -> None:
        now = created_at
        self._exec(
            """
            INSERT OR REPLACE INTO projects
                (id, name, workspace_path, created_at, updated_at, archived, settings)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                name,
                workspace_path,
                now,
                now,
                int(archived),
                json.dumps(settings),
            ),
        )

    def _update(self, project_id: str, **fields) -> None:
        if not fields:
            return
        columns = ", ".join(f"{key} = ?" for key in fields)
        values = [int(v) if k == "archived" else v for k, v in fields.items()]
        self._exec(f"UPDATE projects SET {columns} WHERE id = ?", (*values, project_id))

    def _exec(self, sql: str, params: tuple = ()) -> None:
        try:
            with self._conn() as conn:
                conn.execute(sql, params)
        except Exception:  # noqa: BLE001
            log.warning("project_db_write_failed", sql=sql[:80])
            raise

    def _fetch_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        try:
            with self._conn() as conn:
                return conn.execute(sql, params).fetchone()
        except Exception:  # noqa: BLE001
            log.warning("project_db_read_failed", sql=sql[:80])
            return None

    def _fetch_all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        try:
            with self._conn() as conn:
                return list(conn.execute(sql, params).fetchall())
        except Exception:  # noqa: BLE001
            log.warning("project_db_read_failed", sql=sql[:80])
            return []

    # -- helpers --------------------------------------------------------------

    def _unique_slug(self, slug: str, exclude: str | None = None) -> str:
        slug = slug or "project"
        candidate = slug
        n = 2
        while candidate != exclude and self._slug_taken(candidate):
            candidate = f"{slug}-{n}"
            n += 1
        return candidate

    def _slug_taken(self, slug: str) -> bool:
        return self._fetch_one("SELECT id FROM projects WHERE id = ?", (slug,)) is not None

    def to_info(self, row: sqlite3.Row) -> ProjectInfo:
        workspace = Path(row["workspace_path"])
        try:
            settings = json.loads(row["settings"] or "{}")
        except Exception:  # noqa: BLE001
            settings = {}
        return ProjectInfo(
            id=row["id"],
            name=row["name"],
            workspace_path=str(workspace),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            archived=bool(row["archived"]),
            exists=workspace.is_dir(),
            settings=settings,
            chats_count=0,
            files_count=0,
        )


def slugify(name: str) -> str:
    """Safe directory slug derived from a project name (ASCII)."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "project"
