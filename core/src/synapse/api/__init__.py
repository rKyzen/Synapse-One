"""FastAPI surface — status, chat, workspace, projects, and lifecycle.

Phase 5 adds per-project scoping: every workspace endpoint accepts an
optional ``project_id`` query param; when absent, the legacy global workspace
is used (backward-compatible with tests).  ``/request`` records user and
assistant messages to the active project's chat store.  New ``/projects``
and ``/chats`` endpoints manage project CRUD and chat persistence.

The API layer is thin: it validates input, delegates, and serialises.
No business logic lives here.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

import synapse
from synapse.api.timeline import TimelineHub
from synapse.bootstrap import Boot, create_container
from synapse.domain import AgentRequest, AgentResponse
from synapse.domain.workspace import WorkspaceFileInfo
from synapse.execution import ProviderUnavailable
from synapse.logging import get_logger
from synapse.projects.system import WorkspaceSystem
from synapse.tasks.queue import TaskQueue


#: Per-project background task queues (Phase XV).
_project_queues: dict[str, TaskQueue] = {}


class ProviderStatus(BaseModel):
    provider_id: str
    kind: str
    state: str
    ready: bool


class HardwareField(BaseModel):
    key: str
    value: str | int | float | None


class AppStatus(BaseModel):
    name: str
    version: str
    providers: list[ProviderStatus]
    hardware: list[HardwareField]
    registry_count: int
    models: list[str]


def build_app(boot: Boot | None = None) -> FastAPI:
    """Create the FastAPI app. Pass a Boot to override DI for tests."""
    boot = boot or Boot(create_container())
    boot.start()
    log = get_logger("synapse.api")
    system: WorkspaceSystem = boot.projects  # workspace system facade
    timeline = TimelineHub(boot.events)

    app = FastAPI(
        title="Synapse One — AI Core",
        version=synapse.__version__,
        description="Status + intelligence surface for the Synapse One AI Core.",
    )

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "name": "synapse-one",
            "version": synapse.__version__,
            "docs": "/docs",
            "status": "/status",
        }

    @app.get("/ui", include_in_schema=False)
    def web_ui() -> HTMLResponse:
        """Clean agent chat UI over the workspace (Phase 5)."""
        from synapse.api.web_ui import render_page

        resp = HTMLResponse(render_page())
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    @app.get("/status", response_model=AppStatus)
    def status() -> AppStatus:
        profile = boot.hardware.scan()
        registry = boot.registry
        providers = [
            ProviderStatus(
                provider_id=pid,
                kind=boot.providers.get(pid).kind.value,
                state=boot.providers.state(pid).value,
                ready=boot.providers.health(pid),
            )
            for pid in boot.providers.provider_ids()
        ]
        return AppStatus(
            name="synapse-one",
            version=synapse.__version__,
            providers=providers,
            hardware=[
                HardwareField(key="os", value=profile.os_name),
                HardwareField(key="os_version", value=profile.os_version),
                HardwareField(key="cpu", value=f"{profile.cpu.model} ({profile.cpu.cores}c/{profile.cpu.threads}t)"),
                HardwareField(key="ram_gb", value=profile.memory.total_gb),
                HardwareField(key="ram_available_gb", value=profile.memory.available_gb),
                HardwareField(key="gpu", value=profile.gpu.name if profile.gpu else None),
                HardwareField(key="gpu_vram_gb", value=profile.gpu.vram_gb if profile.gpu else None),
                HardwareField(key="can_run_local_llm", value=profile.recommendations.can_run_local_llm),
                HardwareField(key="max_params_billions", value=profile.recommendations.max_quantized_params_billions),
            ],
            registry_count=len(registry.all()),
            models=[m.id for m in registry.all()],
        )

    # -- /request (chat recording + per-project scoping) --------------------
    # A shared runner powers both the plain POST (JSON in, JSON out) and the
    # SSE variant that streams live execution-timeline steps while the agent
    # works, ending with the same AgentResponse payload.

    def _perform_request(body: AgentRequest) -> tuple[AgentResponse, list[str]]:
        pid = body.project_id or system.session().get("project_id") or "general"
        info = system.get_project(pid)
        chat = system.ensure_chat(pid, body.chat_id)
        chat_store = system.chat_store(pid)
        chat_store.append(
            chat.id,
            role="user",
            content=body.prompt,
            files=body.files,
            project_name=info.name if info else None,
        )
        # Phase A — a chat turn is a conversation INSIDE a goal: link the chat
        # to the workspace's most recent active goal when one exists.
        try:
            for goal in system.goals(pid):
                if goal.status.value == "active":
                    system.goal_store(pid).link(goal.id, chat_id=chat.id)
                    break
        except Exception:  # noqa: BLE001 - goal linking never breaks a request
            log.debug("goal_link_failed", project_id=pid)
        resp = boot.master.process(
            body.prompt,
            files=body.files or None,
            model=body.model,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            workspace=system.workspace_for(pid),
            memory=system.memory_for(pid),
            # Phase 6 — per-project safe filesystem tool + audit trail.
            file_operator=system.file_operator(pid),
            action_log=system.action_log(pid),
            project_id=pid,
            conversation_id=chat.id,
            # Phase XIII — surface the opened project's identity (name, folder)
            # inside every model prompt via the workspace brief.
            project_name=info.name if info else None,
            project_path=info.workspace_path if info else None,
            # Phase XV — per-project indexing, change tracking, diagnostics.
            project_index=system.project_index(pid),
            change_panel=system.change_panel(pid),
            diagnostics=system.diagnostics(pid),
        )
        meta = []
        if resp.provider:
            meta.append(resp.provider)
        if resp.model:
            meta.append(resp.model)
        if resp.latency_ms:
            meta.append(f"{resp.latency_ms/1000:.1f}s")
        if resp.intent:
            meta.append(str(resp.intent))
        if resp.actions:
            created = [a.path for a in resp.actions if a.action == "created" and a.status == "ok"]
            if created:
                meta.append(f"created {len(created)} file(s)")
        ws = resp.workspace
        if ws:
            parts = []
            if ws.files_used:
                parts.append(f"{len(ws.files_used)} files used")
            if ws.retrieval:
                parts.append(f"{len(ws.retrieval)} chunks")
            if ws.local_only:
                parts.append("local")
            if parts:
                meta.append(" · ".join(parts))
        chat_store.append(
            chat.id,
            role="assistant",
            content=resp.response,
            meta=meta,
            trace=resp.decision_trace.model_dump() if resp.decision_trace else None,
        )
        system.save_session(pid, chat.id)
        return resp, meta

    @app.post("/request", response_model=AgentResponse)
    def request(body: AgentRequest) -> AgentResponse:
        """The Master Agent's public entry point.

        When ``project_id`` is supplied (or the active session project is used),
        user and assistant messages are recorded to the project's chat store.
        """
        try:
            resp, _ = _perform_request(body)
        except ProviderUnavailable as exc:
            pid = body.project_id or system.session().get("project_id") or "general"
            chat = system.ensure_chat(pid, body.chat_id)
            chat_store = system.chat_store(pid)
            chat_store.append(chat.id, role="assistant", content=f"Error: {exc}")
            log.warning("request_failed", error=str(exc)[:200])
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return resp

    @app.post("/request/stream")
    def request_stream(body: AgentRequest) -> StreamingResponse:
        """SSE variant of ``/request``: streams live execution-timeline steps,
        then a final ``done`` event carrying the same AgentResponse payload."""
        return timeline.stream(lambda: _perform_request(body))

    # -- /projects (CRUD) ----------------------------------------------------
    #
    # Phase 7: a project is a workspace path in the registry + internal
    # per-project storage. Creating asks for a parent dir (usually chosen via
    # the native picker at /projects/pick); deleting can detach (keep files)
    # or permanently erase the workspace folder; reconnect repoints a project
    # whose workspace moved.

    @app.get("/projects")
    def list_projects(include_archived: bool = False) -> list[dict]:
        return [p.model_dump() for p in system.projects(include_archived=include_archived)]

    @app.post("/projects")
    def create_project(body: dict) -> dict:
        name = (body.get("name") or "").strip()
        # Phase 7 — the parent folder (typically the result of /projects/pick)
        # or an explicit full workspace path for adopting an existing folder.
        parent = (body.get("parent_dir") or body.get("location") or "").strip() or None
        workspace = (body.get("workspace_path") or "").strip() or None
        # A location-only create (the UI only asks *where*) derives the project
        # name from the chosen folder — the user is never asked to name it.
        if not name:
            from pathlib import Path

            base = workspace or parent
            if base:
                name = Path(base).name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="name required")
        try:
            if workspace:
                info = system.register_existing(name, workspace)
            else:
                info = system.create_project(name, parent_dir=parent)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return info.model_dump()

    @app.post("/projects/pick")
    def pick_parent_folder() -> dict:
        """Open the OS-native folder picker; returns the chosen FOLDER path.

        The chosen folder becomes the workspace. The UI POSTs /projects with
        ``parent_dir`` = the folder's parent and ``name`` = the folder's name.
        Returns 503 when the dialog is unavailable (then the UI falls back to
        a manual path).
        """
        from synapse.projects.picker import pick_folder

        path = pick_folder()
        if not path:
            raise HTTPException(status_code=503, detail="folder picker unavailable or cancelled")
        return {"path": path}

    @app.post("/projects/{project_id}/reconnect")
    def reconnect_project(project_id: str, body: dict) -> dict:
        """Repoint a project to its workspace after it was moved/relocated."""
        workspace = (body.get("workspace_path") or "").strip()
        if not workspace:
            raise HTTPException(status_code=400, detail="workspace_path required")
        try:
            info = system.reconnect_project(project_id, workspace)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if info is None:
            raise HTTPException(status_code=404, detail="project not found")
        return info.model_dump()

    @app.post("/projects/{project_id}/verify")
    def verify_project(project_id: str) -> dict:
        """Report whether the project's workspace folder exists on disk."""
        info = system.get_project(project_id)
        if info is None:
            raise HTTPException(status_code=404, detail="project not found")
        return {"id": project_id, "exists": info.exists, "workspace_path": info.workspace_path}

    @app.patch("/projects/{project_id}")
    def rename_project(project_id: str, body: dict) -> dict:
        name = (body.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="name required")
        info = system.rename_project(project_id, name)
        if info is None:
            raise HTTPException(status_code=404, detail="project not found")
        return info.model_dump()

    @app.post("/projects/{project_id}/archive")
    def archive_project(project_id: str) -> dict:
        info = system.archive_project(project_id)
        if info is None:
            raise HTTPException(status_code=404, detail="project not found")
        return info.model_dump()

    @app.post("/projects/{project_id}/unarchive")
    def unarchive_project(project_id: str) -> dict:
        info = system.unarchive_project(project_id)
        if info is None:
            raise HTTPException(status_code=404, detail="project not found")
        return info.model_dump()

    @app.delete("/projects/{project_id}")
    def delete_project(
        project_id: str,
        delete_workspace: bool | None = Query(None),
        body: dict | None = None,
    ) -> dict:
        """Delete a project. ``delete_workspace=true`` also erases the folder."""
        if delete_workspace is None:
            delete_workspace = bool((body or {}).get("delete_workspace"))
        ok = system.delete_project(project_id, delete_workspace=delete_workspace)
        if not ok:
            raise HTTPException(status_code=404, detail="project not found")
        return {"deleted": project_id, "files_deleted": delete_workspace}

    @app.get("/projects/{project_id}/chats")
    def list_chats(project_id: str) -> list[dict]:
        return [c.model_dump() for c in system.chats(project_id)]

    @app.post("/projects/{project_id}/chats")
    def create_chat(project_id: str, body: dict) -> dict:
        info = system.chat_store(project_id).create(body.get("title") or "New chat")
        return info.model_dump()

    @app.patch("/projects/{project_id}/chats/{chat_id}")
    def rename_chat(project_id: str, chat_id: str, body: dict) -> dict:
        info = system.chat_store(project_id).rename(chat_id, body.get("title", ""))
        if info is None:
            raise HTTPException(status_code=404, detail="chat not found")
        return info.model_dump()

    @app.delete("/projects/{project_id}/chats/{chat_id}")
    def delete_chat(project_id: str, chat_id: str) -> dict:
        ok = system.chat_store(project_id).delete(chat_id)
        if not ok:
            raise HTTPException(status_code=404, detail="chat not found")
        return {"deleted": chat_id}

    @app.get("/projects/{project_id}/chats/{chat_id}/messages")
    def get_messages(project_id: str, chat_id: str) -> list[dict]:
        messages = system.chat_store(project_id).messages(chat_id)
        return [m.model_dump() for m in messages]

    # -- Phase X: workspace introspection + opened-folder indexing ----------

    @app.get("/projects/{project_id}/dashboard")
    def project_dashboard(project_id: str) -> dict:
        """Project dashboard: detected stack, index health, recent files."""
        info = system.get_project(project_id)
        if info is None:
            raise HTTPException(status_code=404, detail="project not found")
        ws = system.workspace_for(project_id)
        files = sorted(ws.list_files(), key=lambda f: f.created_at, reverse=True)
        return {
            "project_id": project_id,
            "name": info.name,
            "workspace_path": info.workspace_path,
            "exists": info.exists,
            "language": info.language,
            "framework": info.framework,
            "files_count": info.files_count,
            "indexed_count": info.indexed_count,
            "chunks": ws.vector_store.count(),
            "chats_count": info.chats_count,
            "created_at": info.created_at,
            "updated_at": info.updated_at,
            "recent_files": [
                {
                    "id": f.id,
                    "name": f.name,
                    "extension": f.extension,
                    "status": f.status,
                    "indexed_chunks": f.indexed_chunks,
                    "created_at": f.created_at,
                }
                for f in files[:5]
            ],
            "settings": info.settings,
        }

    @app.post("/projects/{project_id}/scan")
    def scan_project_workspace(project_id: str) -> dict:
        """Import + index every existing file in the project's workspace."""
        if system.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        try:
            return system.scan_project(project_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # -- Phase 6 work directory (agent-generated project files) ------------

    @app.get("/projects/{project_id}/work")
    def list_work_files(project_id: str) -> list[dict]:
        """List the project's generated files (relative to ``work/``)."""
        return system.file_operator(project_id).list_tree()

    @app.get("/projects/{project_id}/work/file")
    def read_work_file(project_id: str, path: str) -> dict:
        content = system.file_operator(project_id).read(path)
        if content is None:
            raise HTTPException(status_code=404, detail="file not found")
        return {"path": path, "content": content}

    @app.post("/projects/{project_id}/work/file")
    def write_work_file(project_id: str, body: dict) -> dict:
        path = (body.get("path") or "").strip()
        content = body.get("content")
        if not path or content is None:
            raise HTTPException(status_code=400, detail="path and content required")
        operator = system.file_operator(project_id)
        try:
            return operator.write(path, content)
        except PermissionError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/projects/{project_id}/work/file/rename")
    def rename_work_file(project_id: str, body: dict) -> dict:
        src = (body.get("from_path") or body.get("path") or "").strip()
        dst = (body.get("to_path") or "").strip()
        if not src or not dst:
            raise HTTPException(status_code=400, detail="from_path and to_path required")
        operator = system.file_operator(project_id)
        try:
            return operator.rename(src, dst)
        except (PermissionError, FileNotFoundError, FileExistsError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.delete("/projects/{project_id}/work/file")
    def delete_work_file(project_id: str, path: str) -> dict:
        operator = system.file_operator(project_id)
        try:
            ok = operator.delete(path)
        except PermissionError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if not ok:
            raise HTTPException(status_code=404, detail="file not found")
        return {"deleted": path}

    # -- Phase XV: project index, search, changes, diagnostics, terminal, tasks --

    @app.get("/projects/{project_id}/index")
    def get_project_index(project_id: str) -> dict:
        """Project code index: files, symbols, imports, structure."""
        if system.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        index = system.project_index(project_id)
        structure = index.scan()  # always fresh on read
        return {
            "project_id": project_id,
            "total_files": structure.total_files,
            "total_folders": structure.total_folders,
            "languages": structure.languages,
            "main_language": structure.main_language,
            "has_tests": structure.has_tests,
            "has_docs": structure.has_docs,
            "entry_points": structure.entry_points,
            "config_files": structure.config_files,
            "dependencies_file": structure.dependencies_file,
            "files": {
                f.path: {
                    "name": f.name,
                    "language": f.language,
                    "size": f.size,
                    "symbols": [s.name for s in f.symbols[:10]],
                    "imports": f.imports[:10],
                }
                for f in index.get_all_files()
            },
        }

    @app.post("/projects/{project_id}/index/refresh")
    def refresh_project_index(project_id: str) -> dict:
        """Force a full re-scan of the workspace into the code index."""
        if system.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        return system.refresh_project_index(project_id)

    @app.get("/projects/{project_id}/search")
    def search_workspace(project_id: str, q: str, max_results: int = 20) -> dict:
        """Semantic + keyword + symbol search across project files."""
        if system.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        return system.search_workspace(project_id, q, max_results=max_results)

    @app.get("/projects/{project_id}/changes")
    def get_project_changes(project_id: str) -> dict:
        """File change panel: created/modified/deleted/renamed this session."""
        if system.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        return system.change_panel(project_id).to_dict()
    @app.delete("/projects/{project_id}/changes")
    def clear_project_changes(project_id: str) -> dict:
        """Reset the change panel."""
        if system.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        system.change_panel(project_id).clear()
        return {"cleared": True}

    @app.post("/projects/{project_id}/terminal")
    def run_terminal_command(project_id: str, body: dict) -> dict:
        """Run a command in the project's workspace. Returns the result."""
        if system.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        command = (body.get("command") or "").strip()
        if not command:
            raise HTTPException(status_code=400, detail="command required")
        timeout = body.get("timeout") or None
        result = system.terminal(project_id).run(command, timeout=timeout)
        return result.to_dict()

    @app.post("/projects/{project_id}/terminal/async")
    def run_terminal_command_async(project_id: str, body: dict) -> dict:
        """Run a command in the background; returns the command id."""
        if system.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        command = (body.get("command") or "").strip()
        if not command:
            raise HTTPException(status_code=400, detail="command required")
        entry_id = system.terminal(project_id).run_async(command)
        return {"command_id": entry_id}

    @app.get("/projects/{project_id}/terminal/history")
    def terminal_history(project_id: str, limit: int = 50) -> list[dict]:
        """Recent command history for the project."""
        if system.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        return [
            {
                "id": e.id,
                "command": e.command,
                "status": e.status,
                "started_at": e.started_at,
                "completed_at": e.completed_at,
                "result": e.result.to_dict() if e.result else None,
            }
            for e in system.terminal(project_id).get_history(limit=limit)
        ]

    @app.get("/projects/{project_id}/diagnostics")
    def project_diagnostics(project_id: str) -> dict:
        """Diagnostics report: syntax errors, unused imports, style warnings."""
        if system.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        report = system.diagnostics(project_id).analyze_all()
        return {
            "files_analyzed": report.files_analyzed,
            "errors": report.errors,
            "warnings": report.warnings,
            "info": report.info,
            "diagnostics": [
                {
                    "file": d.file,
                    "line": d.line,
                    "severity": d.severity,
                    "code": d.code,
                    "message": d.message,
                    "fixable": d.fixable,
                    "fix_suggestion": d.fix_suggestion,
                }
                for d in report.diagnostics
            ],
            "summary": report.summary(),
        }

    @app.get("/projects/{project_id}/tasks")
    def list_project_tasks(project_id: str, limit: int = 50) -> list[dict]:
        """Recent background tasks for the project."""
        if system.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        queue = _project_queues.get(project_id)
        if queue is None:
            return []
        return [t.__dict__ for t in queue.list_tasks(limit=limit)]

    @app.post("/projects/{project_id}/tasks")
    def create_project_task(project_id: str, body: dict) -> dict:
        """Submit a background task (e.g. terminal command)."""
        if system.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        command = (body.get("command") or "").strip()
        if not command:
            raise HTTPException(status_code=400, detail="command required")
        queue = _project_queues.setdefault(project_id, TaskQueue())
        task_id = queue.submit(
            f"terminal: {command[:60]}",
            system.terminal(project_id).run,
            command,
        )
        return {"task_id": task_id}

    @app.get("/projects/{project_id}/tasks/{task_id}")
    def get_project_task(project_id: str, task_id: str) -> dict:
        queue = _project_queues.get(project_id)
        if queue is None:
            raise HTTPException(status_code=404, detail="task not found")
        info = queue.get_task(task_id)
        if info is None:
            raise HTTPException(status_code=404, detail="task not found")
        return info.__dict__

    @app.post("/projects/{project_id}/tasks/{task_id}/cancel")
    def cancel_project_task(project_id: str, task_id: str) -> dict:
        queue = _project_queues.get(project_id)
        if queue is None or not queue.cancel(task_id):
            raise HTTPException(status_code=404, detail="task not found")
        return {"cancelled": task_id}

    @app.post("/projects/{project_id}/tasks/{task_id}/retry")
    def retry_project_task(project_id: str, task_id: str) -> dict:
        queue = _project_queues.get(project_id)
        if queue is None or not queue.retry(task_id):
            raise HTTPException(status_code=404, detail="task not found")
        return {"retried": task_id}

    @app.get("/projects/{project_id}/actions")
    def list_actions(project_id: str, limit: int = 50) -> list[dict]:
        """Phase 6 audit trail: plans, models, tools, files, failures."""
        return system.action_log(project_id).recent(limit=limit)

    # -- /session -----------------------------------------------------------

    @app.get("/session")
    def get_session() -> dict:
        return system.session()

    @app.post("/session")
    def set_session(body: dict) -> dict:
        project_id = body.get("project_id")
        chat_id = body.get("chat_id")
        view = body.get("view")
        if project_id:
            system.switch_project(project_id)
        if chat_id:
            current = system.session()
            system.switch_chat(current.get("project_id") or "general", chat_id)
        if view:
            system.switch_view(str(view))
        return system.session()

    @app.post("/session/view")
    def set_session_view(body: dict) -> dict:
        """Switch the workspace UI view (home/goals/files/notes/tasks/chats/...)."""
        view = str(body.get("view") or "home").strip()
        return system.switch_view(view)

    # -- goals + workspace home (AI Operating Workspace redesign, Phase A) ----

    @app.post("/projects/{project_id}/goals")
    def create_goal(project_id: str, body: dict) -> dict:
        """Primary intake point: create a goal from an outcome statement.

        The user states what they want to accomplish; the workspace tracks
        progress, steps, and linked artifacts around it. ``/request`` remains
        for chat-scoped turns and links to the active goal.
        """
        if system.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        outcome = (body.get("outcome") or body.get("prompt") or "").strip()
        if not outcome:
            raise HTTPException(status_code=400, detail="outcome required")
        info = system.goal_store(project_id).create(
            outcome, deadline=body.get("deadline") or None
        )
        return info.model_dump()

    @app.get("/projects/{project_id}/goals")
    def list_goals(project_id: str, include_archived: bool = False) -> list[dict]:
        if system.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        return [g.model_dump() for g in system.goals(project_id, include_archived=include_archived)]

    @app.get("/projects/{project_id}/goals/{goal_id}")
    def get_goal(project_id: str, goal_id: str) -> dict:
        info = system.goal_store(project_id).get(goal_id)
        if info is None:
            raise HTTPException(status_code=404, detail="goal not found")
        return info.model_dump()

    @app.patch("/projects/{project_id}/goals/{goal_id}")
    def update_goal(project_id: str, goal_id: str, body: dict) -> dict:
        """Update a goal: rename, change status, or set progress."""
        store = system.goal_store(project_id)
        info = store.get(goal_id)
        if info is None:
            raise HTTPException(status_code=404, detail="goal not found")
        if body.get("title"):
            store.rename(goal_id, str(body["title"]))
        if body.get("status"):
            from synapse.domain.enums import GoalStatus

            try:
                store.update_status(goal_id, GoalStatus(body["status"]))
            except ValueError:
                raise HTTPException(status_code=422, detail="invalid goal status")
        if body.get("progress") is not None:
            store.set_progress(goal_id, int(body["progress"]))
        return store.get(goal_id).model_dump()

    @app.post("/projects/{project_id}/goals/{goal_id}/link")
    def link_goal_artifact(project_id: str, goal_id: str, body: dict) -> dict:
        """Link an artifact (chat/file/note/task) to a goal. Idempotent."""
        info = system.goal_store(project_id).link(
            goal_id,
            chat_id=body.get("chat_id"),
            file_id=body.get("file_id"),
            note_id=body.get("note_id"),
            task_id=body.get("task_id"),
        )
        if info is None:
            raise HTTPException(status_code=404, detail="goal not found")
        return info.model_dump()

    @app.delete("/projects/{project_id}/goals/{goal_id}")
    def delete_goal(project_id: str, goal_id: str) -> dict:
        if not system.goal_store(project_id).delete(goal_id):
            raise HTTPException(status_code=404, detail="goal not found")
        return {"deleted": goal_id}

    @app.post("/projects/{project_id}/goals/{goal_id}/plan")
    def plan_goal(project_id: str, goal_id: str) -> dict:
        """Plan a goal: decompose the outcome into capability-labeled steps.

        The planner never names a model — model selection happens at
        execution time by the Router (capabilities first, models second).
        """
        if system.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        store = system.goal_store(project_id)
        record = store.get(goal_id)
        if record is None:
            raise HTTPException(status_code=404, detail="goal not found")
        from synapse.planner.goal_planner import GoalPlanner

        steps = GoalPlanner().plan(record.outcome, system.workspace_type(project_id))
        info = store.set_steps(goal_id, steps)
        return info.model_dump()

    @app.post("/projects/{project_id}/goals/{goal_id}/execute")
    def execute_goal(project_id: str, goal_id: str) -> dict:
        """Execute the planned steps of a goal, one model per step.

        Each step is routed by capability through the same Router contract
        chain as the legacy pipeline; results are recorded onto the goal.
        """
        if system.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        from synapse.planner.goal_executor import GoalExecutor

        executor = GoalExecutor(
            system=system,
            router=boot.master._router,
            providers=boot.master._providers,
            registry=boot.master._registry,
            hardware=boot.master._hardware,
            executor=boot.master._executor,
            events=boot.events,
            performance=boot.master._performance,
            lifecycle=boot.master._lifecycle,
            memory=boot.master._memory,
        )
        return executor.execute(project_id, goal_id)

    # -- notes / documents / todos (AI Operating Workspace redesign, Phase B) --

    def _require_project(project_id: str) -> None:
        if system.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")

    @app.get("/projects/{project_id}/notes")
    def list_notes(project_id: str) -> list[dict]:
        _require_project(project_id)
        return [n.model_dump() for n in system.notes(project_id)]

    @app.post("/projects/{project_id}/notes")
    def create_note(project_id: str, body: dict) -> dict:
        """Create a scratch note; optionally link it to a goal."""
        _require_project(project_id)
        note = system.note_store(project_id).create(
            str(body.get("title") or ""),
            content=str(body.get("content") or ""),
            goal_id=body.get("goal_id") or None,
            tags=body.get("tags"),
        )
        if note.goal_id:
            system.goal_store(project_id).link(note.goal_id, note_id=note.id)
        return note.model_dump()

    @app.get("/projects/{project_id}/notes/{note_id}")
    def get_note(project_id: str, note_id: str) -> dict:
        note = system.note_store(project_id).get(note_id)
        if note is None:
            raise HTTPException(status_code=404, detail="note not found")
        return note.model_dump()

    @app.patch("/projects/{project_id}/notes/{note_id}")
    def update_note(project_id: str, note_id: str, body: dict) -> dict:
        note = system.note_store(project_id).update(
            note_id,
            title=body.get("title"),
            content=body.get("content"),
            tags=body.get("tags"),
            goal_id=body.get("goal_id"),
        )
        if note is None:
            raise HTTPException(status_code=404, detail="note not found")
        return note.model_dump()

    @app.delete("/projects/{project_id}/notes/{note_id}")
    def delete_note(project_id: str, note_id: str) -> dict:
        if not system.note_store(project_id).delete(note_id):
            raise HTTPException(status_code=404, detail="note not found")
        return {"deleted": note_id}

    @app.get("/projects/{project_id}/documents")
    def list_documents(project_id: str) -> list[dict]:
        _require_project(project_id)
        return [d.model_dump() for d in system.documents(project_id)]

    @app.post("/projects/{project_id}/documents")
    def create_document(project_id: str, body: dict) -> dict:
        """Create a typed document (report/essay/plan/...); goal-linkable."""
        _require_project(project_id)
        doc = system.document_store(project_id).create(
            str(body.get("title") or ""),
            doc_type=str(body.get("doc_type") or "document"),
            content=str(body.get("content") or ""),
            goal_id=body.get("goal_id") or None,
        )
        if doc.goal_id:
            system.goal_store(project_id).link(doc.goal_id, note_id=doc.id)
        return doc.model_dump()

    @app.get("/projects/{project_id}/documents/{document_id}")
    def get_document(project_id: str, document_id: str) -> dict:
        doc = system.document_store(project_id).get(document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="document not found")
        return doc.model_dump()

    @app.patch("/projects/{project_id}/documents/{document_id}")
    def update_document(project_id: str, document_id: str, body: dict) -> dict:
        doc = system.document_store(project_id).update(
            document_id,
            title=body.get("title"),
            doc_type=body.get("doc_type"),
            content=body.get("content"),
            status=body.get("status"),
            goal_id=body.get("goal_id"),
        )
        if doc is None:
            raise HTTPException(status_code=404, detail="document not found")
        return doc.model_dump()

    @app.delete("/projects/{project_id}/documents/{document_id}")
    def delete_document(project_id: str, document_id: str) -> dict:
        if not system.document_store(project_id).delete(document_id):
            raise HTTPException(status_code=404, detail="document not found")
        return {"deleted": document_id}

    @app.get("/projects/{project_id}/todos")
    def list_todos(project_id: str) -> list[dict]:
        _require_project(project_id)
        return [t.model_dump() for t in system.todos(project_id)]

    @app.post("/projects/{project_id}/todos")
    def create_todo(project_id: str, body: dict) -> dict:
        """Create a workspace todo item; goal-linkable, prioritized."""
        _require_project(project_id)
        todo = system.todo_store(project_id).create(
            str(body.get("title") or ""),
            description=str(body.get("description") or ""),
            status=body.get("status"),
            priority=body.get("priority"),
            due_date=body.get("due_date"),
            goal_id=body.get("goal_id") or None,
        )
        if todo.goal_id:
            system.goal_store(project_id).link(todo.goal_id, note_id=todo.id)
        return todo.model_dump()

    @app.get("/projects/{project_id}/todos/{todo_id}")
    def get_todo(project_id: str, todo_id: str) -> dict:
        todo = system.todo_store(project_id).get(todo_id)
        if todo is None:
            raise HTTPException(status_code=404, detail="todo not found")
        return todo.model_dump()

    @app.patch("/projects/{project_id}/todos/{todo_id}")
    def update_todo(project_id: str, todo_id: str, body: dict) -> dict:
        todo = system.todo_store(project_id).update(
            todo_id,
            title=body.get("title"),
            description=body.get("description"),
            status=body.get("status"),
            priority=body.get("priority"),
            due_date=body.get("due_date"),
            goal_id=body.get("goal_id"),
        )
        if todo is None:
            raise HTTPException(status_code=404, detail="todo not found")
        return todo.model_dump()

    @app.delete("/projects/{project_id}/todos/{todo_id}")
    def delete_todo(project_id: str, todo_id: str) -> dict:
        if not system.todo_store(project_id).delete(todo_id):
            raise HTTPException(status_code=404, detail="todo not found")
        return {"deleted": todo_id}

    @app.post("/projects/{project_id}/type")
    def set_workspace_type(project_id: str, body: dict) -> dict:
        """Change the workspace type (general/student/research/business/...).

        The type loads different tools/capabilities over the same core
        infrastructure; coding is only one capability of ``developer``.
        """
        if system.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        workspace_type = (body.get("workspace_type") or "").strip().lower()
        if not system.set_workspace_type(project_id, workspace_type):
            raise HTTPException(status_code=422, detail="invalid workspace type")
        return {"project_id": project_id, "workspace_type": workspace_type}

    @app.get("/projects/{project_id}/home")
    def workspace_home(project_id: str) -> dict:
        """Workspace landing aggregate (goal-centric, not chat-centric).

        Active goals, orchestrator progress, open state, and workspace stats —
        the data a workspace home view renders. Chat is deliberately absent:
        it is one view inside the workspace, not the product surface.
        """
        info = system.get_project(project_id)
        if info is None:
            raise HTTPException(status_code=404, detail="project not found")
        goals = system.goals(project_id)
        store = system.goal_store(project_id)
        active = [g for g in goals if g.status.value == "active"]
        active_steps = sum(len(g.steps) for g in active)
        done_steps = sum(
            1 for g in active for s in g.steps if s.status in ("completed", "skipped")
        )
        ws = system.workspace_for(project_id)
        files = ws.list_files()
        recent = sorted(files, key=lambda f: f.created_at, reverse=True)[:8]
        return {
            "project_id": project_id,
            "name": info.name,
            "workspace_type": system.workspace_type(project_id),
            "exists": info.exists,
            "files_count": len(files),
            "indexed_count": info.indexed_count,
            "chunks": ws.vector_store.count(),
            "chats_count": info.chats_count,
            "notes_count": len(system.notes(project_id)),
            "documents_count": len(system.documents(project_id)),
            "todos_count": len(system.todos(project_id)),
            "goals_count": len(goals),
            "active_goals_count": len(active),
            "goal_progress": {
                "steps_total": active_steps,
                "steps_done": done_steps,
            },
            "active_goals": [
                {
                    "id": g.id,
                    "title": g.title,
                    "progress": g.progress,
                    "deadline": g.deadline,
                    "steps": [s.model_dump() for s in g.steps],
                }
                for g in active
            ],
            "recent_files": [
                {
                    "id": f.id,
                    "name": f.name,
                    "extension": f.extension,
                    "status": f.status,
                    "indexed_chunks": f.indexed_chunks,
                    "created_at": f.created_at,
                }
                for f in recent
            ],
        }

    # -- workspace endpoints (project-scoped via query param) ---------------

    def _ws(project_id: str | None = None) -> object:
        """Resolve a workspace instance: per-project when project_id is given."""
        if project_id:
            return system.workspace_for(project_id)
        return boot.workspace

    @app.post("/files", response_model=dict)
    async def upload_file(file: UploadFile, project_id: str | None = Query(None)):
        """Upload a workspace file. Indexing starts in the background."""
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty file")
        ws = _ws(project_id)
        info = ws.upload(file.filename or "unnamed", data)
        job_id = ws.index(info.id)
        return {"file": info.model_dump(), "job_id": job_id}

    @app.get("/files", response_model=list[WorkspaceFileInfo])
    def list_files(project_id: str | None = Query(None)) -> list[WorkspaceFileInfo]:
        return _ws(project_id).list_files()

    @app.get("/files/{file_id}/content")
    def file_content(file_id: str, project_id: str | None = Query(None)) -> dict:
        """Read a workspace file's text content (for clickable file panels)."""
        ws = _ws(project_id)
        entry = ws.get_file(file_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="file not found")
        data = ws.files.read(file_id)
        if not data:
            raise HTTPException(status_code=404, detail="file not found")
        if entry.pipeline == "image":
            return {"id": file_id, "name": entry.name, "kind": "image", "size_bytes": len(data)}
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return {"id": file_id, "name": entry.name, "kind": "binary", "size_bytes": len(data)}
        truncated = len(text) > 100_000
        return {
            "id": file_id,
            "name": entry.name,
            "kind": "text",
            "size_bytes": len(data),
            "content": text[:100_000],
            "truncated": truncated,
        }

    @app.delete("/files/{file_id}")
    def delete_file(file_id: str, project_id: str | None = Query(None)) -> dict:
        removed = _ws(project_id).delete(file_id)
        if not removed:
            raise HTTPException(status_code=404, detail="file not found")
        return {"deleted": file_id}

    @app.get("/jobs/{job_id}", response_model=dict)
    def job_status(job_id: str, project_id: str | None = Query(None)) -> dict:
        job = _ws(project_id).job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.get("/models")
    def list_models() -> list[dict]:
        """All registered models with their provider, capabilities, and status."""
        registry = boot.registry
        available = {}
        for pid in boot.providers.provider_ids():
            try:
                for m in boot.providers.get(pid).list_models():
                    available.setdefault(pid, set()).add(m.id)
            except Exception:  # noqa: BLE001
                pass
        result = []
        for m in registry.all():
            result.append({
                "id": m.id,
                "provider_id": m.provider_id,
                "kind": m.kind.value,
                "available": m.id in available.get(m.provider_id, set()),
            })
        return result

    @app.get("/workspace", response_model=dict)
    def workspace_status(project_id: str | None = Query(None)) -> dict:
        ws = _ws(project_id)
        files = ws.list_files()
        indexed = [f for f in files if f.status == "indexed"]
        return {
            "enabled": ws.enabled,
            "vector_store": ws._vs_settings.provider,
            "files_count": len(files),
            "indexed_count": len(indexed),
            "chunks": ws.vector_store.count(),
            "files": [f.model_dump() for f in files],
        }

    @app.on_event("startup")
    def startup() -> None:
        boot.lifecycle.start_background_task()

    @app.get("/lifecycle", response_model=dict)
    def lifecycle_status() -> dict:
        """Model lifecycle state + aggregate metrics for introspection."""
        manager = boot.lifecycle
        return {
            "enabled": manager.enabled,
            "loaded_models": [m.capability_summary() for m in manager.loaded_models()],
            "loaded_count": manager.current_loaded_count(),
            "settings": {
                "idle_timeout_small_s": manager._settings.idle_timeout_small_s,
                "idle_timeout_large_s": manager._settings.idle_timeout_large_s,
                "idle_timeout_embedding_s": manager._settings.idle_timeout_embedding_s,
                "cleanup_interval_s": manager._settings.cleanup_interval_s,
                "low_memory_threshold_gb": manager._settings.low_memory_threshold_gb,
                "max_loaded_models": manager._settings.max_loaded_models,
                "keep_embedding_loaded": manager._settings.keep_embedding_loaded,
            },
            "system_metrics": manager.metrics().get_system_metrics(),
            "model_metrics": manager.metrics().get_all_model_metrics(),
            "cleanup_stats": manager.metrics().get_cleanup_stats(),
        }

    @app.on_event("shutdown")
    def shutdown() -> None:
        boot.shutdown()

    return app


#: Module-level instance convenient for `uvicorn synapse.api:app`.
#: Tests set SYNAPSE_TEST=1 so importing this module never boots real services.
app = build_app() if os.environ.get("SYNAPSE_TEST") != "1" else None