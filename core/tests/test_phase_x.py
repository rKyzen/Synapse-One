"""Phase X tests — open existing projects: scanner, detection, scan API.

Covers:
- Folder walking with ignore rules + size caps
- Language/framework stack detection (manifests + extension fallback)
- Adopting an existing folder, scanning/importing/indexing it, and the
  project dashboard + file content endpoints
"""

from __future__ import annotations

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
