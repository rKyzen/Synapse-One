"""Phase X tests — open existing projects: scanner, detection, scan API.

Covers:
- Folder walking with ignore rules + size caps
- Language/framework stack detection (manifests + extension fallback)
- Adopting an existing folder, scanning/importing/indexing it, and the
  project dashboard + file content endpoints
- ``/request/stream`` — SSE live execution timeline, ending with the same
  AgentResponse payload as ``/request``
- Per-chat isolation (conversation memory, history) + auto-rename from
  conversation context
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from synapse.bootstrap import Boot, create_container  # noqa: E402
from synapse.config.paths import SynapsePaths  # noqa: E402
from synapse.contracts import ModelProvider  # noqa: E402
from synapse.domain import ModelDescriptor, ModelMetadata, ProviderKind  # noqa: E402
from synapse.domain.enums import ProviderState  # noqa: E402
from synapse.projects.scanner import detect_stack, iter_project_files  # noqa: E402


class FakeProvider(ModelProvider):
    provider_id = "fake"
    kind = ProviderKind.LOCAL

    def initialize(self) -> None:
        pass

    def list_models(self) -> list[ModelDescriptor]:
        return [ModelDescriptor(id="test-model", provider_id="fake")]

    def to_metadata(self, descriptor: ModelDescriptor) -> ModelMetadata | None:
        return ModelMetadata(
            id=descriptor.id,
            provider_id="fake",
            kind=self.kind,
            privacy_score=1.0,
            capabilities={"vision": 1.0, "embeddings": 1.0},
        )

    def embed(self, texts, *, model=None):  # noqa: ANN001
        return [[1.0] if t else [0.0] for t in texts]

    def chat(self, request):  # noqa: ANN001
        from synapse.domain import ChatResponse

        lowered = request.messages[0].content.lower()
        content = "ok"
        if "analysis.md" in lowered:
            content = '{"files": [{"path": "analysis.md", "content": "# Project Analysis\\nStack: Python\\n"}]}'
        elif "output.md" in lowered:
            content = '{"files": [{"path": "output.md", "content": "# Overview\\nA tiny project.\\n"}]}'
        elif "findings.md" in lowered:
            content = '{"files": [{"path": "findings.md", "content": "No blockers.\\n"}]}'
        return ChatResponse(
            provider_id="fake",
            model_id="test-model",
            kind=self.kind,
            content=content,
            raw={},
        )

    def health(self) -> bool:
        return True

    def supports(self, capability) -> bool:  # noqa: ANN001
        return True

    def shutdown(self) -> None:
        pass


@pytest.fixture
def client(temp_paths: SynapsePaths, monkeypatch):
    monkeypatch.setenv("SYNAPSE_HOME", str(temp_paths.home))
    boot = Boot(create_container(paths=temp_paths))
    monkeypatch.setattr(type(boot.providers), "load_all", lambda self: None)
    fake = FakeProvider()
    boot.providers._providers["fake"] = fake
    boot.providers._states["fake"] = ProviderState.READY
    boot.registry._models["test-model"] = ModelMetadata(
        id="test-model",
        provider_id="fake",
        kind=ProviderKind.LOCAL,
        privacy_score=1.0,
        capabilities={
            "vision": 1.0,
            "embeddings": 1.0,
            "coding": 1.0,
            "reasoning": 1.0,
            "chat": 1.0,
            "writing": 1.0,
            "tools": 1.0,
            "json": 1.0,
            "planning": 1.0,
            "translation": 1.0,
        },
    )

    from synapse.api import build_app

    app = build_app(boot)
    with TestClient(app) as c:
        yield c


def _make_project_folder(base: Path) -> Path:
    folder = base / "existing"
    (folder / "app").mkdir(parents=True)
    (folder / "app" / "main.py").write_text("def main():\n    print('hi')\n", encoding="utf-8")
    (folder / "app" / "utils.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (folder / "index.html").write_text("<html><body>hello</body></html>", encoding="utf-8")
    (folder / ".git" / "objects").mkdir(parents=True)
    (folder / "node_modules" / "pkg").mkdir(parents=True)
    (folder / "node_modules" / "pkg" / "index.js").write_text("ignored", encoding="utf-8")
    return folder


def _wait_indexed(client, project_id, expected: int, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/projects/{project_id}/dashboard").json()
        if last["indexed_count"] >= expected:
            return last
        time.sleep(0.1)
    return last


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


class TestScanner:
    def test_iter_skips_vcs_and_dependency_dirs(self, tmp_path):
        folder = _make_project_folder(tmp_path)
        files = iter_project_files(folder)
        names = [p.name for p in files]
        assert "main.py" in names
        assert "utils.py" in names
        assert "index.html" in names
        assert "index.js" not in names
        assert all(".git" not in p.parts and "node_modules" not in p.parts for p in files)

    def test_iter_empty_folder(self, tmp_path):
        assert iter_project_files(tmp_path) == []

    def test_detect_python_fastapi(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            "[project]\ndependencies = [\"fastapi\", \"uvicorn\"]\n", encoding="utf-8"
        )
        (tmp_path / "app.py").write_text("x = 1", encoding="utf-8")
        stack = detect_stack(tmp_path)
        assert stack["language"] == "Python"
        assert stack["framework"] == "FastAPI"

    def test_detect_node_react_typescript(self, tmp_path):
        (tmp_path / "package.json").write_text(
            '{"dependencies": {"react": "^18.0.0"}, "devDependencies": {"typescript": "^5.0.0"}}',
            encoding="utf-8",
        )
        (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "App.tsx").write_text("export const App = () => null", encoding="utf-8")
        stack = detect_stack(tmp_path)
        assert stack["language"] == "TypeScript"
        assert stack["framework"] == "React"

    def test_detect_static_website_fallback(self, tmp_path):
        (tmp_path / "index.html").write_text("<html></html>", encoding="utf-8")
        (tmp_path / "style.css").write_text("body {}", encoding="utf-8")
        (tmp_path / "app.js").write_text("console.log(1)", encoding="utf-8")
        stack = detect_stack(tmp_path)
        assert "HTML" in stack["language"]
        assert stack["framework"] == ""

    def test_detect_empty_folder(self, tmp_path):
        assert detect_stack(tmp_path) == {"language": "", "framework": ""}

    def test_detect_go_manifest(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example\n", encoding="utf-8")
        (tmp_path / "main.go").write_text("package main", encoding="utf-8")
        stack = detect_stack(tmp_path)
        assert stack["language"] == "Go"


# ---------------------------------------------------------------------------
# Open Existing Project flow (API)
# ---------------------------------------------------------------------------


class TestOpenExistingProject:
    def test_adopt_scan_dashboard_content(self, client, temp_paths: SynapsePaths):
        folder = _make_project_folder(temp_paths.home)

        adopted = client.post("/projects", json={"name": "Existing", "workspace_path": str(folder)})
        assert adopted.status_code == 200
        pid = adopted.json()["id"]

        scan = client.post(f"/projects/{pid}/scan")
        assert scan.status_code == 200
        body = scan.json()
        assert body["ok"] is True
        assert body["imported"] == 3
        assert body["index_jobs"] >= 3
        assert body["skipped"] == 0
        assert body["language"] == "Python"

        dash = _wait_indexed(client, pid, expected=3)
        assert dash["files_count"] == 3
        assert dash["indexed_count"] == 3
        assert dash["language"] == "Python"
        assert dash["chunks"] >= 3
        assert dash["chats_count"] >= 0
        recent = {f["name"] for f in dash["recent_files"]}
        assert "app/main.py" in recent

        files = client.get(f"/files?project_id={pid}").json()
        main = next(f for f in files if f["name"] == "app/main.py")
        content = client.get(f"/files/{main['id']}/content?project_id={pid}").json()
        assert content["kind"] == "text"
        assert "def main()" in content["content"]

    def test_project_list_reports_detected_stack(self, client, temp_paths: SynapsePaths):
        folder = _make_project_folder(temp_paths.home)
        pid = client.post("/projects", json={"name": "Stacked", "workspace_path": str(folder)}).json()["id"]
        client.post(f"/projects/{pid}/scan")
        project = next(p for p in client.get("/projects").json() if p["id"] == pid)
        assert project["language"] == "Python"
        assert project["files_count"] == 3
        assert project["indexed_count"] == 3

    def test_rescan_is_idempotent(self, client, temp_paths: SynapsePaths):
        folder = _make_project_folder(temp_paths.home)
        pid = client.post("/projects", json={"name": "Again", "workspace_path": str(folder)}).json()["id"]
        first = client.post(f"/projects/{pid}/scan").json()
        second = client.post(f"/projects/{pid}/scan").json()
        assert second["imported"] == 0
        assert second["files"] == first["files"]

    def test_scan_missing_folder_404(self, client, temp_paths: SynapsePaths):
        folder = temp_paths.home / "gone"
        folder.mkdir()
        pid = client.post("/projects", json={"name": "Gone", "workspace_path": str(folder)}).json()["id"]
        folder.rmdir()
        resp = client.post(f"/projects/{pid}/scan")
        assert resp.status_code == 404

    def test_scan_unknown_project_404(self, client):
        assert client.post("/projects/nope/scan").status_code == 404

    def test_dashboard_unknown_project_404(self, client):
        assert client.get("/projects/nope/dashboard").status_code == 404

    def test_file_content_unknown_404(self, client):
        assert client.get("/files/nope/content").status_code == 404

    def test_opened_project_dashboard_fields(self, client, temp_paths: SynapsePaths):
        folder = _make_project_folder(temp_paths.home)
        pid = client.post("/projects", json={"name": "Dash", "workspace_path": str(folder)}).json()["id"]
        dash = client.get(f"/projects/{pid}/dashboard").json()
        assert dash["name"] == "Dash"
        assert dash["workspace_path"] == str(folder)
        assert dash["exists"] is True
        assert dash["framework"] == ""
        assert "recent_files" in dash
        assert "updated_at" in dash

    def test_read_file_via_workspace_command(self, client, temp_paths: SynapsePaths):
        folder = _make_project_folder(temp_paths.home)
        pid = client.post("/projects", json={"name": "Reader", "workspace_path": str(folder)}).json()["id"]
        client.post(f"/projects/{pid}/scan")
        resp = client.post("/request", json={"prompt": "read app/main.py", "project_id": pid})
        assert resp.status_code == 200
        body = resp.json()
        assert "### app/main.py" in body["response"]
        assert "def main()" in body["response"]

    def test_read_missing_file_reports_not_found(self, client, temp_paths: SynapsePaths):
        folder = _make_project_folder(temp_paths.home)
        pid = client.post("/projects", json={"name": "Miss", "workspace_path": str(folder)}).json()["id"]
        client.post(f"/projects/{pid}/scan")
        resp = client.post("/request", json={"prompt": "read app/nope.py", "project_id": pid})
        assert resp.status_code == 200
        assert "not found" in resp.json()["response"]


# ---------------------------------------------------------------------------
# Phase X — strict manifest parsing (universal workspace tool)
# ---------------------------------------------------------------------------


class TestStrictManifest:
    def test_strict_requires_whole_answer_to_be_json(self, tmp_path):
        from synapse.workspace.manifest import parse_file_manifest

        assert parse_file_manifest("Some prose about the project.", strict=True) == []
        assert parse_file_manifest("", strict=True) == []
        assert parse_file_manifest(None, strict=True) == []

    def test_strict_ignores_json_embedded_in_prose(self, tmp_path):
        from synapse.workspace.manifest import parse_file_manifest

        assert parse_file_manifest('Here are the notes {"files": [{"path": "a.md", "content": "x"}]} done.', strict=True) == []

    def test_strict_parses_pure_manifest_and_fenced_json(self, tmp_path):
        from synapse.workspace.manifest import parse_file_manifest

        ops = parse_file_manifest(
            '{"folders": ["docs"], "files": [{"path": "docs/a.md", "content": "x"}]}',
            strict=True,
        )
        assert {"action": "create_folder", "path": "docs"} in ops
        assert {"action": "write", "path": "docs/a.md", "content": "x"} in ops

        ops = parse_file_manifest('```json\n{"files": [{"path": "a.py", "content": "x"}]}\n```', strict=True)
        assert ops == [{"action": "write", "path": "a.py", "content": "x"}]

    def test_non_strict_keeps_tolerant_fences_fallback(self, tmp_path):
        from synapse.workspace.manifest import parse_file_manifest

        ops = parse_file_manifest(
            "Sure:\n```python\n# file: a.py\nprint(1)\n```\n", strict=False
        )
        assert ops == [{"action": "write", "path": "a.py", "content": "print(1)"}]


# ---------------------------------------------------------------------------
# Phase X — universal workspace tool (all agents can read AND write)
# ---------------------------------------------------------------------------


class TestUniversalWorkspaceTool:
    def _adopted(self, client, temp_paths: SynapsePaths, name: str) -> tuple[str, Path]:
        folder = _make_project_folder(temp_paths.home)
        pid = client.post(
            "/projects", json={"name": name, "workspace_path": str(folder)}
        ).json()["id"]
        client.post(f"/projects/{pid}/scan")
        return pid, folder

    def test_analysis_agent_writes_report_file(self, client, temp_paths: SynapsePaths):
        pi, folder = self._adopted(client, temp_paths, "AnalysisWriter")
        resp = client.post(
            "/request",
            json={"prompt": "Analyze the project and save the analysis to analysis.md", "project_id": pi},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        paths = {a["path"] for a in body["actions"]}
        assert "analysis.md" in paths
        assert "Project Analysis" in (folder / "analysis.md").read_text(encoding="utf-8")

    def test_chat_agent_writes_named_file(self, client, temp_paths: SynapsePaths):
        # A chat-classified request still gets the full workspace tool: context
        # injection AND the write path — the file lands in the folder.
        pi, folder = self._adopted(client, temp_paths, "ChatWriter")
        resp = client.post(
            "/request",
            json={"prompt": "Just a quick chat — write the project overview to output.md", "project_id": pi},
        )
        assert resp.status_code == 200, resp.text
        assert "output.md" in {a["path"] for a in resp.json()["actions"]}
        assert "# Overview" in (folder / "output.md").read_text(encoding="utf-8")

    def test_plain_analysis_without_destination_writes_nothing(self, client, temp_paths: SynapsePaths):
        pi, folder = self._adopted(client, temp_paths, "ReadOnly")
        before = self._work_files(folder)
        resp = client.post(
            "/request",
            json={"prompt": "Analyze this project and tell me what the issues are", "project_id": pi},
        )
        assert resp.status_code == 200, resp.text
        assert self._work_files(folder) == before
        assert resp.json()["actions"] == []

    @staticmethod
    def _work_files(folder: Path) -> list[str]:
        return sorted(
            p.relative_to(folder).as_posix()
            for p in folder.rglob("*")
            if p.is_file() and ".git" not in p.parts and "node_modules" not in p.parts
        )


# ---------------------------------------------------------------------------
# Phase XI — live execution timeline (/request/stream, SSE)
# ---------------------------------------------------------------------------


class TestRequestStream:
    def _events(self, client, **body) -> list[dict]:
        events = []
        with client.stream(
            "POST", "/request/stream", json={"prompt": "hello", **body}
        ) as resp:
            assert resp.status_code == 200, resp.text
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                events.append(json.loads(line[len("data: ") :]))
        return events

    def test_stream_emits_timeline_then_done(self, client, temp_paths: SynapsePaths):
        events = self._events(client)
        kinds = [e["kind"] for e in events]
        assert "understand" in kinds
        assert "finished" in kinds
        assert kinds[-1] == "done"
        assert events[-1]["response"]["response"]

    def test_stream_shows_workspace_reads_and_finished(self, client, temp_paths: SynapsePaths):
        pi, _ = self._adopted(client, temp_paths, "StreamProject")
        events = self._events(client, project_id=pi)
        kinds = [e["kind"] for e in events]
        assert "read_workspace" in kinds
        assert "finished" in kinds
        assert kinds[-1] == "done"
        assert events[-1]["meta"]

    @staticmethod
    def _adopted(client, temp_paths: SynapsePaths, name: str) -> tuple[str, Path]:
        folder = _make_project_folder(temp_paths.home)
        pid = client.post(
            "/projects", json={"name": name, "workspace_path": str(folder)}
        ).json()["id"]
        client.post(f"/projects/{pid}/scan")
        return pid, folder


# ---------------------------------------------------------------------------
# Phase XII — per-chat isolation + auto-rename
# ---------------------------------------------------------------------------


class TestChatIsolation:
    def test_conversation_memory_is_isolated_per_chat(self, client, temp_paths: SynapsePaths):
        pi, _ = TestRequestStream._adopted(client, temp_paths, "IsolatedMem")
        chat_a = client.post(f"/projects/{pi}/chats", json={}).json()
        chat_b = client.post(f"/projects/{pi}/chats", json={}).json()
        assert chat_a["id"] != chat_b["id"]

        r1 = client.post(
            "/request", json={"prompt": "alpha topic", "project_id": pi, "chat_id": chat_a["id"]}
        )
        r2 = client.post(
            "/request", json={"prompt": "beta topic", "project_id": pi, "chat_id": chat_b["id"]}
        )
        assert r1.status_code == 200, r1.text
        assert r2.status_code == 200, r2.text

        convo_file = temp_paths.memory_dir / pi / "conversation.json"
        data = json.loads(convo_file.read_text(encoding="utf-8"))
        assert len(data) >= 4
        assert {e["conversation"] for e in data} == {chat_a["id"], chat_b["id"]}
        a_conv = [e for e in data if e["conversation"] == chat_a["id"]]
        b_conv = [e for e in data if e["conversation"] == chat_b["id"]]
        assert all(e["text"] != "beta topic" for e in a_conv)
        assert all(e["text"] != "alpha topic" for e in b_conv)

    def test_chats_have_own_messages_and_titles(self, client, temp_paths: SynapsePaths):
        pi, _ = TestRequestStream._adopted(client, temp_paths, "OwnState")
        chat_a = client.post(f"/projects/{pi}/chats", json={}).json()
        chat_b = client.post(f"/projects/{pi}/chats", json={}).json()
        client.post(
            "/request", json={"prompt": "history only for a", "project_id": pi, "chat_id": chat_a["id"]}
        )
        msgs_a = client.get(f"/projects/{pi}/chats/{chat_a['id']}/messages").json()
        msgs_b = client.get(f"/projects/{pi}/chats/{chat_b['id']}/messages").json()
        assert len(msgs_a) == 2
        assert msgs_b == []
        assert msgs_a[0]["content"] == "history only for a"


class TestAutoRename:
    def test_first_user_message_renames_default_chat(self, tmp_path):
        from synapse.projects.chats import ChatStore

        store = ChatStore("p", tmp_path / "chats")
        chat = store.create("General Discussion")
        store.append(chat.id, role="user", content="Analyze\n  our codebase   for bugs")
        assert store.get(chat.id).title == "Analyze our codebase for bugs"

    def test_explicit_rename_is_never_overwritten(self, tmp_path):
        from synapse.projects.chats import ChatStore

        store = ChatStore("p", tmp_path / "chats")
        chat = store.create("New chat")
        store.rename(chat.id, "My custom title")
        store.append(chat.id, role="user", content="something else")
        assert store.get(chat.id).title == "My custom title"

    def test_blank_message_falls_back(self, tmp_path):
        from synapse.projects.chats import ChatStore

        store = ChatStore("p", tmp_path / "chats")
        chat = store.create("New chat")
        store.append(chat.id, role="user", content="   \n\t ")
        assert store.get(chat.id).title == "Chat"


# ---------------------------------------------------------------------------
# Phase XIII — workspace-first / tool-first redesign
# ---------------------------------------------------------------------------


class TestWorkspaceAccessGate:
    def test_build_verbs_touch_the_workspace(self):
        from synapse.actions import requires_workspace_access

        for prompt in (
            "build a calculator app",
            "create a python script that prints hello",
            "refactor the login function",
            "fix the bug in main.py",
            "design a landing page",
        ):
            assert requires_workspace_access(prompt), prompt

    def test_explicit_destination_touches_the_workspace(self):
        from synapse.actions import requires_workspace_access

        assert requires_workspace_access("save the summary to analysis.md")

    def test_pure_greeting_stays_chat(self):
        from synapse.actions import requires_workspace_access

        for prompt in ("hello", "hi there", "thanks", "who are you", "ok goodbye"):
            assert requires_workspace_access(prompt) is False, prompt

    def test_chat_only_opinion_stays_chat(self):
        from synapse.actions import requires_workspace_access

        assert requires_workspace_access("sum two numbers and tell me") is False


class TestWorkspaceBrief:
    class StubOperator:
        root = r"C:\user\project"

        @staticmethod
        def list_tree():
            return [
                {"path": "src/main.py", "size": 12, "modified_at": "2026-08-06T10:00:00+00:00"},
                {"path": "README.md", "size": 4, "modified_at": "2026-08-06T11:00:00+00:00"},
            ]

    class StubMemory:
        def __init__(self, convo_id="chat-1"):
            self._convo = convo_id

        def recent(self, scope, limit=10, conversation=None):
            from synapse.domain.memory import MemoryEntry

            def entry(i, text, source):
                return MemoryEntry(
                    id=f"e{i}", scope=scope, text=text, source=source, conversation=conversation
                )

            return [entry(2, "agent: wrote the fix", "assistant"), entry(1, "user: please fix the bug", "user")]

    def test_brief_names_project_path_tree_tools_and_recent(self):
        from synapse.workspace.brief import build_workspace_brief

        brief = build_workspace_brief(
            project_id="p123",
            project_name="MyApp",
            project_path=None,
            file_operator=self.StubOperator(),
            memory=self.StubMemory(),
            conversation_id="chat-1",
        )
        assert "Current workspace: MyApp" in brief
        assert "p123" in brief
        assert r"C:\user\project" in brief
        assert "Read Files" in brief
        assert "Write Files" in brief
        assert "src/main.py" in brief
        assert "README.md" in brief
        assert "Recently modified" in brief
        # Newest first.
        assert brief.index("README.md") < brief.index("src/main.py")

    def test_brief_summarizes_the_current_chat(self):
        from synapse.workspace.brief import build_workspace_brief

        brief = build_workspace_brief(
            project_id="p1", file_operator=self.StubOperator(),
            memory=self.StubMemory(), conversation_id="chat-1",
        )
        assert "fix the bug" in brief
        assert "- user:" in brief

    def test_brief_degrades_gracefully(self):
        from synapse.workspace.brief import build_workspace_brief

        brief = build_workspace_brief(project_id="pFresh")
        assert "Current workspace: pFresh" in brief
        assert "Available workspace tools" in brief
        assert "(none yet" in brief


class TestWorkspaceFirstRequest:
    @staticmethod
    def _adopted(client, temp_paths: SynapsePaths, name: str) -> tuple[str, Path]:
        folder = _make_project_folder(temp_paths.home)
        pid = client.post(
            "/projects", json={"name": name, "workspace_path": str(folder)}
        ).json()["id"]
        client.post(f"/projects/{pid}/scan")
        return pid, folder

    def test_written_file_is_verified_indexed_and_remembered(self, client, temp_paths: SynapsePaths):
        pi, folder = self._adopted(client, temp_paths, "Redesign")
        resp = client.post(
            "/request",
            json={"prompt": "Just a chat — write the project overview to output.md", "project_id": pi},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "output.md" in {a["path"] for a in body["actions"]}

        # Verified on disk.
        assert (folder / "output.md").exists()

        # Index refreshed: the written file is now part of the project index.
        dashboard = client.get(f"/projects/{pi}/dashboard").json()
        assert "output.md" in {f["name"] for f in dashboard["recent_files"]}

        # Remembered at project scope: a later request can recall the write
        # without re-reading the folder.
        project_memory = json.loads(
            (temp_paths.memory_dir / pi / "project.json").read_text(encoding="utf-8")
        )
        assert "wrote to the workspace: created output.md" in " ".join(
            e["text"] for e in project_memory
        )

    def test_stream_shows_verify_and_index_steps(self, client, temp_paths: SynapsePaths):
        pi, _ = self._adopted(client, temp_paths, "RedesignStream")
        events = []
        with client.stream(
            "POST", "/request/stream",
            json={"prompt": "write the project overview to output.md", "project_id": pi},
        ) as stream:
            for line in stream.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[len("data: ") :]))
        kinds = [e["kind"] for e in events]
        assert "verified" in kinds
        assert "index" in kinds
        assert kinds[-1] == "done"
