"""WorkspaceSystem — Phase 5/7 facade over projects, chats, and isolation.

Phase 7 responsibilities (professional IDE architecture):

1. **Workspace folders** — a project's workspace is a plain user-chosen
   folder holding ONLY source/generated files and Git data. Synapse never
   writes chats, memory, indexes, or logs into it.
2. **Internal storage** — everything Synapse owns lives under the OS appdata
   tree (``SynapsePaths``) keyed by the unique project id: chats, memory,
   embeddings/vector data, action logs, session state.
3. **Native folder picker** — ``create_project(name, parent_dir=...)`` makes
   ``<parent_dir>/<Name>``; the parent is usually chosen via the OS dialog
   (``synapse.projects.picker``).
4. **Registry** — ``ProjectManager`` (SQLite) stores only workspace paths;
   ``verify`` reports workspaces missing on disk so the UI can offer
   reconnect; ``delete`` can detach (keep files) or erase the folder.
5. **Lifecycle** — per-project chats/memory/workspace are cached per id and
   dropped on delete; ``cleanup()`` sweeps stale temp folders, orphaned
   internal state, and empty leftover workspaces from failed creations.
6. **Session recovery** — last active project + chat restored on boot.
"""

from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

from synapse.domain.projects import ChatInfo, ProjectInfo, StoredMessage
from synapse.logging import get_logger
from synapse.projects.chats import ChatStore
from synapse.projects.manager import DEFAULT_PROJECT_ID, DEFAULT_PROJECT_NAME, ProjectManager

log = get_logger("synapse.projects.system")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkspaceSystem:
    """Top-level project/chat/workspace coordinator."""

    def __init__(
        self,
        root: Path,
        paths,
        config,
        providers,
        registry,
        events,
        lifecycle=None,
        *,
        memory_config: dict | None = None,
        workspace_settings=None,
        vector_store_settings=None,
    ) -> None:
        self._root = Path(root)
        self._paths = paths
        self._config = config
        self._providers = providers
        self._registry = registry
        self._events = events
        self._lifecycle = lifecycle
        self._session_path = paths.session_file

        self.manager = ProjectManager(self._root)
        self._workspaces: dict[str, object] = {}
        self._memories: dict[str, object] = {}
        self._chat_stores: dict[str, ChatStore] = {}
        self._operators: dict[str, object] = {}
        self._action_logs: dict[str, object] = {}
        self._projects: dict[str, ProjectInfo] = {}
        self._lock = threading.RLock()

        self._memory_cfg = dict(memory_config or {})
        self._workspace_settings = workspace_settings
        self._vector_store_settings = vector_store_settings

        self._active_project_id: str | None = None
        self._active_chat_id: str | None = None
        self.recover()

    # -- projects -----------------------------------------------------------

    def _reload_projects(self, *, include_archived: bool = False) -> None:
        with self._lock:
            self._projects = {}
            for pid in self.manager.list_ids(include_archived=include_archived):
                info = self.manager.get(pid)
                if info is not None:
                    self._projects[pid] = self._refresh_info(info)

    def _refresh_info(self, info: ProjectInfo) -> ProjectInfo:
        """Patch aggregate counts + disk-existence flag onto a ProjectInfo."""
        chats_dir = self._paths.chats_dir / info.id
        chats_count = 0
        if chats_dir.is_dir():
            chats_count = len([p for p in chats_dir.glob("*.json") if not p.name.endswith(".tmp")])
        workspace_dir = self._paths.embeddings_dir / info.id
        files_count = 0
        indexed_count = 0
        catalog = workspace_dir / "catalog.json"
        if catalog.is_file():
            try:
                entries = json.loads(catalog.read_text(encoding="utf-8"))
                files_count = len(entries)
                indexed_count = len([e for e in entries if e.get("status") == "indexed"])
            except Exception:  # noqa: BLE001
                files_count = 0
                indexed_count = 0
        stack = (info.settings or {}).get("stack") or {}
        return info.model_copy(
            update={
                "exists": Path(info.workspace_path).is_dir(),
                "chats_count": chats_count,
                "files_count": files_count,
                "indexed_count": indexed_count,
                "language": stack.get("language", ""),
                "framework": stack.get("framework", ""),
            }
        )

    def _ensure_default_project(self) -> None:
        if self.manager.get(DEFAULT_PROJECT_ID) is None:
            self.create_project(DEFAULT_PROJECT_NAME)

    def projects(self, *, include_archived: bool = False) -> list[ProjectInfo]:
        self._reload_projects(include_archived=include_archived)
        return sorted(self._projects.values(), key=lambda p: (p.archived, p.name.lower()))

    def get_project(self, project_id: str) -> ProjectInfo | None:
        info = self.manager.get(project_id)
        return self._refresh_info(info) if info else None

    def create_project(self, name: str, parent_dir: str | None = None) -> ProjectInfo:
        info = self.manager.create(name, parent_dir=parent_dir)
        self._reload_projects(include_archived=True)
        log.info("project_ready", project_id=info.id, workspace=info.workspace_path)
        return info

    def register_existing(self, name: str, workspace_path: str) -> ProjectInfo:
        """Adopt an existing folder on disk as a project workspace."""
        info = self.manager.register(name, workspace_path)
        self._reload_projects(include_archived=True)
        return info

    def scan_project(self, project_id: str) -> dict:
        """Import an existing workspace folder into the project's index.

        Walks the folder (honouring ignore rules), uploads each file into the
        project's internal workspace (sha256 dedupe), kicks off background
        indexing jobs for anything not yet indexed, detects the language /
        framework stack, and persists it on the registry.
        """
        info = self.manager.get(project_id)
        if info is None:
            return {"ok": False, "error": "project not found"}
        folder = Path(info.workspace_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"workspace folder missing: {folder}")

        from synapse.projects.scanner import detect_stack, iter_project_files

        ws = self.workspace_for(project_id)
        paths = iter_project_files(folder)
        before = {f.id for f in ws.files.list()}
        jobs: list[str] = []
        imported = 0
        skipped = 0
        for path in paths:
            try:
                data = path.read_bytes()
            except OSError:
                skipped += 1
                continue
            try:
                rel = path.relative_to(folder).as_posix()
            except ValueError:
                rel = path.name
            entry = ws.upload(rel, data)
            if entry.status != "indexed":
                jobs.append(ws.index(entry.id))
        after = ws.files.list()
        imported = len({f.id for f in after} - before)

        stack = detect_stack(folder)
        if any(stack.values()):
            self.manager.update_settings(project_id, {"stack": stack})
        else:
            self.manager.touch(project_id)
        self._reload_projects(include_archived=True)
        log.info(
            "project_scanned",
            project_id=project_id,
            files=len(paths),
            imported=imported,
            jobs=len(jobs),
            language=stack.get("language"),
            framework=stack.get("framework"),
        )
        return {
            "ok": True,
            "project_id": project_id,
            "files": len(paths),
            "imported": imported,
            "index_jobs": len(jobs),
            "skipped": skipped,
            "language": stack.get("language", ""),
            "framework": stack.get("framework", ""),
        }

    def reconnect_project(self, project_id: str, workspace_path: str) -> ProjectInfo | None:
        """Repoint a project to its workspace after it moved on disk."""
        info = self.manager.reconnect(project_id, workspace_path)
        self._reload_projects(include_archived=True)
        return info

    def verify_projects(self) -> dict[str, bool]:
        """id -> workspace-folder-exists; drives the reconnect UI."""
        return self.manager.verify()

    def rename_project(self, project_id: str, name: str) -> ProjectInfo | None:
        with self._lock:
            info = self.manager.rename(project_id, name)
            self._reload_projects(include_archived=True)
        return info

    def archive_project(self, project_id: str) -> ProjectInfo | None:
        info = self.manager.archive(project_id)
        self._reload_projects(include_archived=True)
        if info and self._active_project_id == project_id:
            self.switch_project(DEFAULT_PROJECT_ID)
        return info

    def unarchive_project(self, project_id: str) -> ProjectInfo | None:
        info = self.manager.unarchive(project_id)
        self._reload_projects(include_archived=True)
        return info

    def delete_project(self, project_id: str, *, delete_workspace: bool = False) -> bool:
        with self._lock:
            ok = self.manager.delete(project_id, delete_workspace=delete_workspace)
            if ok:
                self._drop_cached(project_id)
                self._purge_internal_state(project_id)
            self._reload_projects(include_archived=True)
        if ok and self._active_project_id == project_id:
            self.switch_project(DEFAULT_PROJECT_ID)
        return ok

    def _purge_internal_state(self, project_id: str) -> None:
        """Remove a deleted project's chats/memory/embeddings/action logs."""
        for bucket in (
            self._paths.chats_dir / project_id,
            self._paths.memory_dir / project_id,
            self._paths.embeddings_dir / project_id,
            self._paths.actions_dir / project_id,
        ):
            try:
                if bucket.is_dir():
                    shutil.rmtree(bucket, ignore_errors=True)
            except OSError:
                log.warning("internal_purge_failed", path=str(bucket))

    def _drop_cached(self, project_id: str) -> None:
        for bucket in (self._workspaces, self._memories, self._chat_stores, self._operators, self._action_logs):
            bucket.pop(project_id, None)

    # -- per-project internal state ------------------------------------------

    def chats_dir(self, project_id: str) -> Path:
        return self._paths.chats_dir / project_id

    def workspace_path(self, project_id: str) -> Path:
        """The user's workspace folder (agent files land here)."""
        return self.manager.workspace_path(project_id)

    def chat_store(self, project_id: str) -> ChatStore:
        with self._lock:
            if project_id not in self._chat_stores:
                self._chat_stores[project_id] = ChatStore(project_id, self.chats_dir(project_id))
            return self._chat_stores[project_id]

    def file_operator(self, project_id: str):
        """Safe filesystem tool rooted at the project's WORKSPACE FOLDER.

        The agent's generated files are real user files: they persist in the
        workspace (and its Git repo), not in a temp cache.
        """
        with self._lock:
            if project_id not in self._operators:
                from synapse.workspace.operator import FileOperator

                self._operators[project_id] = FileOperator(self.workspace_path(project_id))
            return self._operators[project_id]

    def action_log(self, project_id: str):
        """Per-project JSONL audit trail, stored internally."""
        with self._lock:
            if project_id not in self._action_logs:
                from synapse.projects.actionlog import ActionLog

                self._action_logs[project_id] = ActionLog(
                    project_id, self._paths.actions_dir / project_id
                )
            return self._action_logs[project_id]

    def memory_for(self, project_id: str):
        """Per-project WorkspaceMemory, stored internally under ``memory/``."""
        with self._lock:
            if project_id in self._memories:
                return self._memories[project_id]
            from synapse.memory import WorkspaceMemory

            memory = WorkspaceMemory(
                self._paths,
                self._config,
                self._providers,
                enabled=bool(self._memory_cfg.get("enabled", True)),
                top_k=int(self._memory_cfg.get("top_k", 3)),
                min_similarity=float(self._memory_cfg.get("min_similarity", 0.35)),
                embedding_model=self._memory_cfg.get("embedding_model"),
                root=self._paths.memory_dir / project_id,
            )
            self._memories[project_id] = memory
            return memory

    def workspace_for(self, project_id: str):
        """Per-project Workspace (files, vector index, retrieval), cached."""
        with self._lock:
            if project_id in self._workspaces:
                return self._workspaces[project_id]
            from synapse.workspace import Workspace

            internal_root = self._paths.embeddings_dir / project_id
            workspace = Workspace(
                self._paths,
                self._config,
                self._providers,
                self._registry,
                self._events,
                memory=self.memory_for(project_id),
                lifecycle=self._lifecycle,
                root=internal_root,
            )
            self._workspaces[project_id] = workspace
            log.info("workspace_ready_for_project", project_id=project_id, root=str(internal_root))
            return workspace

    # -- chats ---------------------------------------------------------------

    def chats(self, project_id: str) -> list[ChatInfo]:
        return self.chat_store(project_id).list()

    def ensure_chat(self, project_id: str, chat_id: str | None = None) -> ChatInfo:
        """Return an existing chat, or create the project's default chat."""
        if chat_id:
            info = self.chat_store(project_id).get(chat_id)
            if info:
                return info
        chats = self.chat_store(project_id).list()
        if chats:
            return chats[0]
        return self.chat_store(project_id).create("General Discussion")

    def append_message(
        self,
        project_id: str,
        chat_id: str,
        *,
        role: str,
        content: str,
        meta: list[str] | None = None,
        trace: dict | None = None,
        files: list[str] | None = None,
    ) -> StoredMessage | None:
        message = self.chat_store(project_id).append(
            chat_id, role=role, content=content, meta=meta, trace=trace, files=files,
        )
        if message:
            self._save_session(project_id, chat_id)
        return message

    # -- cleanup (Phase 7: no orphaned temp/empty folders) ---------------------

    def cleanup(self) -> None:
        """Sweep transient + orphaned state. Safe to run on every boot.

        - wipes the internal temp dir (``cache/tmp``)
        - removes internal per-project state (chats/memory/embeddings/action
          logs) whose project id is no longer registered
        - removes leftover empty workspace folders from failed creations
          (only inside Synapse's internal workspaces root — never in a
          user-chosen parent)
        """
        with self._lock:
            known = set(self.manager.list_ids(include_archived=True))
            for bucket_name in (
                self._paths.chats_dir,
                self._paths.memory_dir,
                self._paths.embeddings_dir,
                self._paths.actions_dir,
            ):
                if not bucket_name.is_dir():
                    continue
                for child in bucket_name.iterdir():
                    if child.name in known:
                        continue
                    try:
                        if child.is_dir():
                            shutil.rmtree(child, ignore_errors=True)
                        elif child.is_file():
                            child.unlink(missing_ok=True)
                    except OSError:
                        log.warning("cleanup_orphan_failed", path=str(child))
            try:
                if self._paths.temp_dir.is_dir():
                    shutil.rmtree(self._paths.temp_dir, ignore_errors=True)
                self._paths.temp_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                log.warning("cleanup_temp_failed")
            self._cleanup_stale_workspaces(known)

    def _cleanup_stale_workspaces(self, known: set[str]) -> None:
        """Remove empty leftover workspaces under the internal root."""
        workspaces_root = self._root / "workspaces"
        if not workspaces_root.is_dir():
            return
        for child in workspaces_root.iterdir():
            if not child.is_dir() or child.name in known:
                continue
            try:
                contents = list(child.iterdir())
                if not contents or all(p.name == "README.md" for p in contents):
                    shutil.rmtree(child, ignore_errors=True)
                    log.info("cleanup_stale_workspace", path=str(child))
            except OSError:
                log.warning("cleanup_stale_failed", path=str(child))

    # -- session recovery ------------------------------------------------------

    def recover(self) -> None:
        """Boot-time recovery: cleanup, default project, last session."""
        self.cleanup()
        self._ensure_default_project()
        self._reload_projects(include_archived=True)
        session = self._load_session()
        if session:
            pid = session.get("project_id")
            if pid in self._projects and not self._projects[pid].archived:
                self._active_project_id = pid
                cid = session.get("chat_id")
                if cid and self.chat_store(pid).get(cid):
                    self._active_chat_id = cid
        if self._active_project_id is None:
            self._active_project_id = DEFAULT_PROJECT_ID
        if self._active_project_id not in self._projects:
            self._active_project_id = DEFAULT_PROJECT_ID
        self._active_chat_id = self.ensure_chat(self._active_project_id, self._active_chat_id).id
        log.info(
            "session_recovered",
            project_id=self._active_project_id,
            chat_id=self._active_chat_id,
        )

    def switch_project(self, project_id: str) -> ProjectInfo | None:
        info = self.get_project(project_id)
        if info is None or info.archived:
            return None
        self._active_project_id = project_id
        self._active_chat_id = self.ensure_chat(project_id).id
        self._save_session(project_id, self._active_chat_id)
        self._reload_projects(include_archived=True)
        return info

    def switch_chat(self, project_id: str, chat_id: str) -> ChatInfo | None:
        info = self.chat_store(project_id).get(chat_id)
        if info is None:
            return None
        self._active_project_id = project_id
        self._active_chat_id = chat_id
        self._save_session(project_id, chat_id)
        return info

    def session(self) -> dict:
        return {
            "project_id": self._active_project_id,
            "chat_id": self._active_chat_id,
        }

    def save_session(self, project_id: str, chat_id: str | None = None) -> None:
        self._save_session(project_id, chat_id or self._active_chat_id)

    def _save_session(self, project_id: str, chat_id: str | None) -> None:
        self._active_project_id = project_id
        if chat_id:
            self._active_chat_id = chat_id
        try:
            data = {
                "project_id": self._active_project_id,
                "chat_id": self._active_chat_id,
                "updated_at": _now_iso(),
            }
            self._session_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._session_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self._session_path)
        except Exception:  # noqa: BLE001
            log.warning("session_save_failed", project_id=project_id)

    def _load_session(self) -> dict:
        try:
            if self._session_path.exists():
                return json.loads(self._session_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            log.warning("session_load_failed")
        return {}
