"""Phase 6 tests — AI Workspace: folders, filesystem safety, multi-model
execution, file generation, planner correctness, and recovery.

Covers:
- Project creation at a user-chosen location (registry + directory layout)
- FileOperator safety (traversal/absolute/symlink escapes rejected)
- FileOperator CRUD (write/read/rename/delete/list, atomic writes)
- Manifest parsing (JSON shapes, fences, unsafe paths, fallbacks)
- Planner correctness (file-task detection, review task, model hints)
- OutputReviewer (python/json/js/html validation)
- End-to-end multi-model execution through the API (coder + reasoner models)
- Recovery after interruption (failed tasks, session + work files persist)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from synapse.bootstrap import Boot, create_container  # noqa: E402
from synapse.config.paths import SynapsePaths  # noqa: E402
from synapse.contracts import ModelProvider  # noqa: E402
from synapse.domain import ChatResponse, ModelCapabilities, ModelDescriptor, ModelMetadata, ProviderKind  # noqa: E402
from synapse.domain.enums import ProviderState  # noqa: E402
from synapse.planner.heuristic import HeuristicTaskPlanner  # noqa: E402
from synapse.projects.manager import ProjectManager  # noqa: E402
from synapse.projects.system import WorkspaceSystem  # noqa: E402
from synapse.workspace.manifest import parse_file_manifest  # noqa: E402
from synapse.workspace.operator import FileOperator, WorkspaceSafetyError  # noqa: E402
from synapse.workspace.review import summarize, validate_file  # noqa: E402


# ---------------------------------------------------------------------------
# Fake multi-model provider (coder + reasoner)
# ---------------------------------------------------------------------------

def _reply_for(prompt: str) -> str:
    lowered = prompt.lower()
    if "explode" in lowered:
        raise RuntimeError("injected failure")
    if "review" in lowered:
        return "All generated files are consistent and well-formed."
    if "create a python app" in lowered:
        return json.dumps({
            "files": [
                {"path": "hello.py", "content": "print('hello')\n"},
                {"path": "README.md", "content": "# Hello App\n"},
            ]
        })
    if "website" in lowered:
        return json.dumps({
            "files": [
                {"path": "index.html", "content": "<!doctype html>\n<html><body><h1>Home</h1></body></html>\n"},
                {"path": "style.css", "content": "body { color: #333; }\n"},
            ]
        })
    if "readme" in lowered or "documentation" in lowered:
        return json.dumps({"files": [{"path": "README.md", "content": "# Project\n\nDocs.\n"}]})
    return "ok"


class FakeProvider(ModelProvider):
    provider_id = "fake"
    kind = ProviderKind.LOCAL

    _CAPS = {
        "coder": ModelCapabilities(coding=0.95, reasoning=0.3, chat=0.6),
        "reasoner": ModelCapabilities(coding=0.1, reasoning=0.95, chat=0.4),
    }

    def initialize(self) -> None:
        pass

    def list_models(self) -> list[ModelDescriptor]:
        return [ModelDescriptor(id=mid, provider_id="fake") for mid in self._CAPS]

    def to_metadata(self, descriptor: ModelDescriptor) -> ModelMetadata | None:
        caps = self._CAPS.get(descriptor.id)
        if caps is None:
            return None
        return ModelMetadata(
            id=descriptor.id,
            provider_id="fake",
            kind=self.kind,
            privacy_score=1.0,
            capabilities=caps,
        )

    def embed(self, texts, *, model=None):  # noqa: ANN001
        return [[1.0] if t else [0.0] for t in texts]

    def chat(self, request):  # noqa: ANN001
        content = _reply_for(request.messages[0].content)
        return ChatResponse(
            provider_id="fake",
            model_id=request.model,
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
    fake = FakeProvider()
    boot.providers._providers["fake"] = fake
    boot.providers._states["fake"] = ProviderState.READY
    # Hermetic routing: stop load_all from registering the configured (real)
    # providers — e.g. a local Ollama — so only the fake models are routable.
    monkeypatch.setattr(type(boot.providers), "load_all", lambda self: None)
    from synapse.api import build_app

    app = build_app(boot)
    with TestClient(app) as c:
        yield c, boot


@pytest.fixture
def project_manager(temp_paths: SynapsePaths) -> ProjectManager:
    projects_dir = temp_paths.data_dir / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    return ProjectManager(projects_dir)


@pytest.fixture
def operator(temp_paths: SynapsePaths) -> FileOperator:
    return FileOperator(temp_paths.data_dir / "work")


# ===========================================================================
# 1. Workspace creation — project at a user-chosen location
# ===========================================================================

class TestWorkspaceCreation:
    def test_create_project_at_location_creates_workspace(self, temp_paths: SynapsePaths):
        manager = ProjectManager(temp_paths.data_dir / "projects")
        base = temp_paths.home / "dev"
        info = manager.create("My Website", parent_dir=base)
        workspace = base / "My Website"
        assert workspace.is_dir()
        # The workspace folder contains ONLY project files — no Synapse data.
        assert (workspace / "README.md").is_file()
        assert not (workspace / "chats").exists()
        assert not (workspace / "memory").exists()
        assert not (workspace / "work").exists()
        assert not (workspace / "config.json").exists()
        assert info.workspace_path == str(workspace)
        assert info.root == str(workspace)  # backward-compat alias

    def test_external_project_registered_and_discoverable(self, temp_paths: SynapsePaths):
        projects_dir = temp_paths.data_dir / "projects"
        manager = ProjectManager(projects_dir)
        base = temp_paths.home / "sites"
        manager.create("Discover Me", parent_dir=base)
        fresh = ProjectManager(projects_dir)
        ids = fresh.list_ids()
        assert "discover-me" in ids
        info = fresh.get("discover-me")
        assert info is not None
        assert info.workspace_path == str(base / "Discover Me")

    def test_delete_external_project_detaches_keeps_files(self, temp_paths: SynapsePaths):
        projects_dir = temp_paths.data_dir / "projects"
        manager = ProjectManager(projects_dir)
        base = temp_paths.home / "sites"
        info = manager.create("Delete Me", parent_dir=base)
        assert manager.delete(info.id)
        assert (base / "Delete Me").exists()  # folder untouched
        fresh = ProjectManager(projects_dir)
        assert fresh.get(info.id) is None

    def test_delete_external_project_erase_workspace(self, temp_paths: SynapsePaths):
        projects_dir = temp_paths.data_dir / "projects"
        manager = ProjectManager(projects_dir)
        base = temp_paths.home / "sites"
        info = manager.create("Delete Me", parent_dir=base)
        assert manager.delete(info.id, delete_workspace=True)
        assert not (base / "Delete Me").exists()

    def test_default_project_workspace_is_internal(self, temp_paths: SynapsePaths):
        projects_dir = temp_paths.data_dir / "projects"
        manager = ProjectManager(projects_dir)
        info = manager.create("Plain")
        workspace = temp_paths.data_dir / "projects" / "workspaces" / "plain"
        assert info.workspace_path == str(workspace)
        assert (workspace / "README.md").is_file()
        assert not (workspace / "chats").exists()

    def test_register_existing_folder(self, temp_paths: SynapsePaths):
        projects_dir = temp_paths.data_dir / "projects"
        manager = ProjectManager(projects_dir)
        base = temp_paths.home / "existing-project"
        base.mkdir(parents=True)
        (base / "main.py").write_text("print('hi')", encoding="utf-8")
        info = manager.register("Existing", base)
        assert info.workspace_path == str(base)
        assert (Path(info.workspace_path) / "main.py").is_file()

    def test_reconnect_after_move(self, temp_paths: SynapsePaths):
        projects_dir = temp_paths.data_dir / "projects"
        manager = ProjectManager(projects_dir)
        old = temp_paths.home / "old"
        info = manager.create("Moved", parent_dir=old)
        moved = temp_paths.home / "new"
        moved.mkdir(parents=True)
        refreshed = manager.reconnect(info.id, moved)
        assert refreshed is not None
        assert refreshed.workspace_path == str(moved)

    def test_verify_reports_missing_workspace(self, temp_paths: SynapsePaths):
        projects_dir = temp_paths.data_dir / "projects"
        manager = ProjectManager(projects_dir)
        info = manager.create("Vanishing", parent_dir=temp_paths.home / "v")
        import shutil

        shutil.rmtree(Path(info.workspace_path))
        assert manager.verify()[info.id] is False

    def test_api_create_project_with_parent_dir(self, client, temp_paths: SynapsePaths):
        c, _ = client
        base = str(temp_paths.home / "api-sites")
        resp = c.post("/projects", json={"name": "API Site", "parent_dir": base})
        assert resp.status_code == 200
        body = resp.json()
        workspace = Path(base) / "API Site"
        assert body["workspace_path"] == str(workspace)
        assert workspace.is_dir()
        assert (workspace / "README.md").is_file()
        assert not (workspace / "work").is_dir()

    def test_api_pick_folder_endpoint(self, client, monkeypatch, temp_paths: SynapsePaths):
        c, _ = client
        base = temp_paths.home / "pick-target"
        base.mkdir(parents=True)
        monkeypatch.setattr(
            "synapse.projects.picker.pick_folder",
            lambda initial_dir=None: str(base),
        )
        resp = c.post("/projects/pick")
        assert resp.status_code == 200
        assert resp.json() == {"path": str(base)}

    def test_api_pick_folder_unavailable(self, client, monkeypatch):
        c, _ = client
        monkeypatch.setattr(
            "synapse.projects.picker.pick_folder",
            lambda initial_dir=None: None,
        )
        resp = c.post("/projects/pick")
        assert resp.status_code == 503

    def test_api_work_dir_endpoints(self, client):
        c, _ = client
        pid = c.post("/projects", json={"name": "Work Endpoint"}).json()["id"]
        assert [f["path"] for f in c.get(f"/projects/{pid}/work").json()] == ["README.md"]
        assert c.post(
            f"/projects/{pid}/work/file",
            json={"path": "app.py", "content": "x = 1\n"},
        ).status_code == 200
        listing = c.get(f"/projects/{pid}/work").json()
        assert [f["path"] for f in listing] == ["app.py", "README.md"]
        assert c.get(f"/projects/{pid}/work/file", params={"path": "app.py"}).json()["content"] == "x = 1\n"
        assert c.post(
            f"/projects/{pid}/work/file/rename",
            json={"from_path": "app.py", "to_path": "main.py"},
        ).status_code == 200
        assert c.delete(f"/projects/{pid}/work/file", params={"path": "main.py"}).status_code == 200
        assert [f["path"] for f in c.get(f"/projects/{pid}/work").json()] == ["README.md"]

    def test_api_work_file_safety(self, client):
        c, _ = client
        pid = c.post("/projects", json={"name": "Safe"}).json()["id"]
        resp = c.post(f"/projects/{pid}/work/file", json={"path": "../escape.txt", "content": "x"})
        assert resp.status_code == 422
        assert [f["path"] for f in c.get(f"/projects/{pid}/work").json()] == ["README.md"]

    def test_actions_endpoint(self, client):
        c, _ = client
        pid = c.post("/projects", json={"name": "Actions"}).json()["id"]
        resp = c.get(f"/projects/{pid}/actions")
        assert resp.status_code == 200
        assert resp.json() == []


# ===========================================================================
# 2. Filesystem safety
# ===========================================================================

class TestFilesystemSafety:
    @pytest.mark.parametrize("bad", [
        "../escape.txt",
        "a/../../b.txt",
        "../../../etc/passwd",
        "/etc/passwd",
        "C:/Windows/win.ini",
        "C:\\Windows\\win.ini",
        "a/..",
        "..",
        "",
        ".",
    ])
    def test_unsafe_paths_rejected(self, operator: FileOperator, bad: str):
        with pytest.raises(WorkspaceSafetyError):
            operator.path_for(bad)

    def test_nested_safe_paths_allowed(self, operator: FileOperator):
        p = operator.path_for("src/app/main.py")
        assert p == operator.root / "src" / "app" / "main.py"

    def test_write_never_escapes_root(self, operator: FileOperator, temp_paths: SynapsePaths):
        operator.write("nested/dir/file.txt", "content")
        outside = temp_paths.data_dir / "file.txt"
        assert not outside.exists()
        assert (operator.root / "nested" / "dir" / "file.txt").exists()

    def test_operator_isolation_between_projects(self, temp_paths: SynapsePaths):
        a = FileOperator(temp_paths.data_dir / "p1" / "work")
        b = FileOperator(temp_paths.data_dir / "p2" / "work")
        a.write("x.txt", "from a")
        b.write("x.txt", "from b")
        assert a.read("x.txt") == "from a"
        assert b.read("x.txt") == "from b"

    def test_git_internals_are_read_only(self, temp_paths: SynapsePaths):
        op = FileOperator(temp_paths.data_dir / "ws-git")
        with pytest.raises(WorkspaceSafetyError):
            op.write(".git/config", "x")
        with pytest.raises(WorkspaceSafetyError):
            op.path_for(".git/HEAD")
        op.write("a.txt", "1")
        with pytest.raises(WorkspaceSafetyError):
            op.rename("a.txt", ".git/a.txt")
        with pytest.raises(WorkspaceSafetyError):
            op.delete(".git/objects/aa/bb")

    def test_list_tree_skips_git_internals(self, temp_paths: SynapsePaths):
        op = FileOperator(temp_paths.data_dir / "ws-git-listing")
        op.write("src/main.py", "x")
        (op.root / ".git" / "objects").mkdir(parents=True)
        (op.root / ".git" / "objects" / "aa").write_text("blob", encoding="utf-8")
        paths = [e["path"] for e in op.list_tree()]
        assert paths == ["src/main.py"]
        assert all(not p.startswith(".git") for p in paths)


# ===========================================================================
# 3. FileOperator CRUD
# ===========================================================================

class TestFileOperator:
    def test_write_creates_file(self, operator: FileOperator):
        result = operator.write("hello.txt", "hi")
        assert result["action"] == "created"
        assert (operator.root / "hello.txt").read_text(encoding="utf-8") == "hi"

    def test_write_overwrite_is_modified(self, operator: FileOperator):
        operator.write("f.txt", "v1")
        result = operator.write("f.txt", "v2")
        assert result["action"] == "modified"

    def test_write_creates_parent_dirs(self, operator: FileOperator):
        operator.write("a/b/c.txt", "deep")
        assert operator.read("a/b/c.txt") == "deep"

    def test_atomic_write_no_tmp_leftovers(self, operator: FileOperator):
        operator.write("f.txt", "v")
        leftovers = list(operator.root.rglob("*.tmp"))
        assert leftovers == []

    def test_read_missing(self, operator: FileOperator):
        assert operator.read("nope.txt") is None

    def test_rename(self, operator: FileOperator):
        operator.write("old.txt", "data")
        result = operator.rename("old.txt", "new.txt")
        assert result["action"] == "renamed"
        assert not (operator.root / "old.txt").exists()
        assert operator.read("new.txt") == "data"

    def test_rename_missing_source(self, operator: FileOperator):
        with pytest.raises(FileNotFoundError):
            operator.rename("nope.txt", "x.txt")

    def test_rename_existing_dest(self, operator: FileOperator):
        operator.write("a.txt", "1")
        operator.write("b.txt", "2")
        with pytest.raises(FileExistsError):
            operator.rename("a.txt", "b.txt")

    def test_delete(self, operator: FileOperator):
        operator.write("gone.txt", "x")
        assert operator.delete("gone.txt")
        assert not operator.exists("gone.txt")
        assert not operator.delete("gone.txt")

    def test_list_tree(self, operator: FileOperator):
        operator.write("a.txt", "1")
        operator.write("sub/b.txt", "2")
        paths = sorted(e["path"] for e in operator.list_tree())
        assert paths == ["a.txt", "sub/b.txt"]

    def test_export_import_roundtrip(self, operator: FileOperator, temp_paths: SynapsePaths):
        operator.write("a.txt", "one")
        operator.write("deep/b.py", "print('x')")
        other = FileOperator(temp_paths.data_dir / "work2")
        other.import_manifest(operator.export_manifest())
        assert other.read("deep/b.py") == "print('x')"


# ===========================================================================
# 4. Manifest parsing
# ===========================================================================

class TestManifestParsing:
    def test_full_manifest(self):
        ops = parse_file_manifest(json.dumps({
            "files": [
                {"path": "a.py", "content": "x=1"},
                {"path": "b.md", "content": "# B"},
            ]
        }))
        assert [(o["action"], o["path"]) for o in ops] == [("write", "a.py"), ("write", "b.md")]

    def test_fenced_json(self):
        text = "```json\n" + json.dumps({"files": [{"path": "x.py", "content": "y"}]}) + "\n```"
        ops = parse_file_manifest(text)
        assert ops[0]["path"] == "x.py"

    def test_flat_mapping(self):
        ops = parse_file_manifest(json.dumps({"a.txt": "one", "sub/b.txt": "two"}))
        assert [(o["path"], o["content"]) for o in ops] == [("a.txt", "one"), ("sub/b.txt", "two")]

    def test_single_file_key(self):
        ops = parse_file_manifest(json.dumps({"file": {"path": "solo.py", "content": "c"}}))
        assert ops == [{"action": "write", "path": "solo.py", "content": "c"}]

    def test_rename_and_delete_ops(self):
        ops = parse_file_manifest(json.dumps({
            "files": [
                {"action": "rename", "from": "old.py", "to": "new.py"},
                {"action": "delete", "path": "stale.txt"},
            ]
        }))
        assert ops[0] == {"action": "rename", "path": "old.py", "to": "new.py"}
        assert ops[1] == {"action": "delete", "path": "stale.txt"}

    def test_unsafe_paths_flagged_not_dropped(self):
        ops = parse_file_manifest(json.dumps({"files": [{"path": "../evil.txt", "content": "x"}]}))
        assert len(ops) == 1
        assert "error" in ops[0]

    def test_no_manifest_but_single_fence(self):
        ops = parse_file_manifest("Here is the code:\n```python\nprint('hi')\n```")
        assert ops[0]["action"] == "write"
        assert ops[0]["path"] == "main.py"  # language-default name
        assert ops[0]["content"] == "print('hi')"

    def test_fence_with_path_in_hint(self):
        ops = parse_file_manifest("```javascript\nlet x = 1;\n```", hint="app.js")
        assert ops[0]["path"] == "app.js"

    def test_prose_only(self):
        assert parse_file_manifest("I don't know how to do that.") == []

    def test_empty(self):
        assert parse_file_manifest(None) == []
        assert parse_file_manifest("") == []


# ===========================================================================
# 5. Planner correctness
# ===========================================================================

def _plan(prompt: str, complexity: int = 100):
    from synapse.domain import ComplexityResult, Decision, IntentResult, PrivacyResult
    from synapse.domain.enums import IntentType, PrivacyMode

    planner = HeuristicTaskPlanner()
    return planner.plan(
        prompt,
        IntentResult(primary=IntentType.CODING, confidence=0.9),
        ComplexityResult(score=complexity),
        PrivacyResult(mode=PrivacyMode.BALANCED),
        Decision(),
    )


class TestPlannerPhase6:
    def test_create_website_is_file_task(self):
        dag = _plan("Create a website with a homepage")
        task = dag.get("t1")
        assert task is not None
        assert task.file_output is True
        assert any(t.kind.value == "review" for t in dag.tasks)

    def test_write_python_app_is_file_task(self):
        dag = _plan("Write a python app that does math")
        assert dag.get("t1").file_output is True
        assert dag.get("t1").kind.value == "coding"

    def test_generate_docs_is_file_task(self):
        dag = _plan("Generate documentation for my project")
        assert dag.get("t1").file_output is True

    def test_refactor_is_file_task(self):
        dag = _plan("Refactor my project")
        assert dag.get("t1").file_output is True

    def test_plain_chat_no_file_output(self):
        dag = _plan("What is the meaning of life?")
        assert dag.get("t1").file_output is False
        assert dag.get("t-review") is None

    def test_review_depends_only_on_file_tasks(self):
        dag = _plan("Create a website. Then explain the design.")
        review = dag.get("t-review")
        assert review is not None
        file_tasks = [t.id for t in dag.tasks if t.file_output]
        assert set(review.depends_on) == set(file_tasks)

    def test_file_hint_detected(self):
        dag = _plan("Create a script that uses requests.py")
        assert dag.get("t1").file_hint == "requests.py"

    def test_model_hint_from_config(self):
        planner = HeuristicTaskPlanner(model_hints={"coding": "qwen2.5-coder"})
        from synapse.domain import ComplexityResult, Decision, IntentResult, PrivacyResult
        from synapse.domain.enums import IntentType, PrivacyMode

        dag = planner.plan(
            "Create a python app",
            IntentResult(primary=IntentType.CODING, confidence=0.9),
            ComplexityResult(score=10),
            PrivacyResult(mode=PrivacyMode.BALANCED),
            Decision(),
        )
        assert dag.get("t1").model_hint == "qwen2.5-coder"

    def test_disabled_file_output(self):
        planner = HeuristicTaskPlanner(file_output_enabled=False)
        from synapse.domain import ComplexityResult, Decision, IntentResult, PrivacyResult
        from synapse.domain.enums import IntentType, PrivacyMode

        dag = planner.plan(
            "Create a website",
            IntentResult(primary=IntentType.CODING, confidence=0.9),
            ComplexityResult(score=10),
            PrivacyResult(mode=PrivacyMode.BALANCED),
            Decision(),
        )
        assert dag.get("t1").file_output is False


# ===========================================================================
# 6. OutputReviewer
# ===========================================================================

class TestOutputReviewer:
    def test_python_ok(self):
        result = validate_file("app.py", "def f():\n    return 1\n")
        assert result.ok

    def test_python_syntax_error(self):
        result = validate_file("app.py", "def f(:\n")
        assert not result.ok
        assert "python" in result.error

    def test_json_ok(self):
        assert validate_file("data.json", '{"a": 1}').ok

    def test_json_invalid(self):
        assert not validate_file("data.json", "{oops").ok

    def test_js_balance(self):
        assert validate_file("app.js", "function f() { return [1, 2]; }").ok
        assert not validate_file("app.js", "function f() { return [1, 2];").ok

    def test_html_document_root(self):
        assert validate_file("index.html", "<!doctype html><html><body>hi</body></html>").ok
        assert not validate_file("index.html", "<div>hi</div>").ok

    def test_text_non_empty(self):
        assert validate_file("notes.txt", "hello").ok
        assert not validate_file("notes.txt", "   ").ok

    def test_summarize(self):
        text = summarize([validate_file("app.py", "x = 1")])
        assert "app.py" in text
        assert "ok" in text


# ===========================================================================
# 7. End-to-end multi-model execution (via API)
# ===========================================================================

class TestEndToEnd:
    def test_file_generation_pipeline(self, client):
        c, _ = client
        pid = c.post("/projects", json={"name": "E2E"}).json()["id"]
        resp = c.post("/request", json={"prompt": "Create a python app that prints hello", "project_id": pid})
        assert resp.status_code == 200, resp.text
        body = resp.json()

        # files written to the project work dir
        work = c.get(f"/projects/{pid}/work").json()
        paths = {f["path"] for f in work}
        assert "hello.py" in paths
        assert "README.md" in paths
        content = c.get(f"/projects/{pid}/work/file", params={"path": "hello.py"}).json()["content"]
        assert content == "print('hello')\n"

        # response carries actions + summary
        actions = body["actions"]
        assert any(a["action"] == "created" and a["path"] == "hello.py" and a["validated"] for a in actions)
        assert "Actions (" in body["response"]

        # multi-model: coding routed to coder, review routed to reasoner
        nodes = {n["task_id"]: n for n in body["execution_graph"]["nodes"]}
        assert nodes["t1"]["model_id"] == "coder"
        assert nodes["t-review"]["model_id"] == "reasoner"

        # action log entry
        entries = c.get(f"/projects/{pid}/actions").json()
        assert entries, "action log should have an entry"
        entry = entries[0]
        assert "hello.py" in entry["files_created"]
        assert "coder" in entry["models_used"]
        assert "reasoner" in entry["models_used"]
        assert entry["files_failed"] == []
        assert entry["failures"] == []

    def test_multifile_website(self, client):
        c, _ = client
        pid = c.post("/projects", json={"name": "Site"}).json()["id"]
        resp = c.post("/request", json={"prompt": "Create a website with a homepage", "project_id": pid})
        assert resp.status_code == 200, resp.text
        paths = {f["path"] for f in c.get(f"/projects/{pid}/work").json()}
        assert {"index.html", "style.css"} <= paths  # README.md is the seed file
        assert len(paths) == 3

    def test_chat_agent_can_write_when_model_emits_manifest(self, client):
        # Phase X universal workspace tool — chat requests now carry the
        # workspace context, and any agent that answers with a manifest gets
        # those files written. (The fake reacts to the injected README.md
        # listing with a manifest, so the file lands in the workspace.)
        c, _ = client
        pid = c.post("/projects", json={"name": "Prose"}).json()["id"]
        resp = c.post("/request", json={"prompt": "explain the weather", "project_id": pid})
        assert resp.status_code == 200
        body = resp.json()
        assert any(a["path"] == "README.md" for a in body["actions"])
        paths = {f["path"] for f in c.get(f"/projects/{pid}/work").json()}
        assert "README.md" in paths

    def test_interruption_recovers_partial_work(self, client):
        """A failing sub-task must not lose the files the successful one wrote."""
        c, _ = client
        pid = c.post("/projects", json={"name": "Interrupt"}).json()["id"]
        prompt = "Create a python app that prints hello. then explode on purpose."
        resp = c.post("/request", json={"prompt": prompt, "project_id": pid})
        assert resp.status_code == 200, resp.text
        body = resp.json()

        paths = {f["path"] for f in c.get(f"/projects/{pid}/work").json()}
        assert "hello.py" in paths

        entries = c.get(f"/projects/{pid}/actions").json()
        assert entries
        entry = entries[0]
        assert entry["failures"], "the failing task should be recorded"
        assert any(f["task_id"] for f in entry["failures"])
        assert "hello.py" in entry["files_created"]

    def test_session_and_work_survive_reboot(self, client, temp_paths: SynapsePaths):
        """Recovery after interruption: session + generated files persist."""
        c, boot = client
        pid = c.post("/projects", json={"name": "Reboot"}).json()["id"]
        c.post("/request", json={"prompt": "Create a python app that prints hello", "project_id": pid})
        c.post("/session", json={"project_id": pid})

        from synapse.projects.system import WorkspaceSystem

        system2 = WorkspaceSystem(
            root=temp_paths.data_dir / "projects",
            paths=temp_paths,
            config=boot.projects._config,
            providers=boot.projects._providers,
            registry=boot.projects._registry,
            events=boot.projects._events,
            lifecycle=boot.projects._lifecycle,
        )
        assert system2.session()["project_id"] == pid
        paths = {f["path"] for f in system2.file_operator(pid).list_tree()}
        assert "hello.py" in paths


class TestPlannerUnit:
    """Planner unit tests without the analyzer pipeline."""

    def test_split_prompt_produces_dag(self):
        from synapse.domain import ComplexityResult, Decision, IntentResult, PrivacyResult
        from synapse.domain.enums import IntentType, PrivacyMode

        planner = HeuristicTaskPlanner()
        dag = planner.plan(
            "Create a python app. then write a readme for it.",
            IntentResult(primary=IntentType.CODING, confidence=0.9),
            ComplexityResult(score=90),
            PrivacyResult(mode=PrivacyMode.BALANCED),
            Decision(),
        )
        ids = [t.id for t in dag.tasks]
        assert "t1" in ids and "t2" in ids
        assert "t-review" in ids
        assert "t-synthesis" in ids