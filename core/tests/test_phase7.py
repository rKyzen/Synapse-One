"""Phase 7 tests — Action-Based Execution Engine.

Covers:
- Request classification (the seven kinds)
- Explicit workspace-op extraction (list / search / rename / delete / folders)
- ActionEngine: folder creation, edits, rename/move, delete confirmation gate,
  existence verification, search, model-free workspace ops, summaries
- Manifest additions: ``folders`` and ``edit`` ops
- Planner: modification prompts become file tasks
- End-to-end: project generation writes real folders + files into the
  workspace and the chat only ever shows a summary (never the code)
- End-to-end: workspace operations are answered by the backend alone
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from synapse.actions import ActionEngine, RequestKind, classify_request, extract_workspace_ops  # noqa: E402
from synapse.bootstrap import Boot, create_container  # noqa: E402
from synapse.config.paths import SynapsePaths  # noqa: E402
from synapse.contracts import ModelProvider  # noqa: E402
from synapse.domain import Capability, ChatResponse, ModelCapabilities, ModelDescriptor, ModelMetadata, ProviderKind  # noqa: E402
from synapse.domain.enums import ProviderState  # noqa: E402
from synapse.domain.fileops import FileAction, ValidationResult  # noqa: E402
from synapse.planner.heuristic import HeuristicTaskPlanner  # noqa: E402
from synapse.workspace.manifest import parse_file_manifest  # noqa: E402
from synapse.workspace.operator import FileOperator  # noqa: E402


# ---------------------------------------------------------------------------
# Fake multi-model provider with Phase 7 manifests (folders, edits, deletes)
# ---------------------------------------------------------------------------

def _reply_for(prompt: str) -> str:
    lowered = prompt.lower()
    if "portfolio" in lowered:
        return json.dumps({
            "folders": ["assets", "css", "js"],
            "files": [
                {"path": "index.html", "content": "<!doctype html>\n<html><body><h1>Portfolio</h1></body></html>\n"},
                {"path": "style.css", "content": "body { color: #333; }\n"},
                {"path": "script.js", "content": "console.log('hi');\n"},
            ],
        })
    if "cleanup" in lowered:
        return json.dumps({
            "files": [
                {"action": "delete", "path": "stale.txt"},
                {"path": "hello.py", "content": "print('hello')\n"},
            ]
        })
    if "blue" in lowered:
        return json.dumps({
            "files": [
                {"path": "index.html", "content": "<!doctype html>\n<html><body style=\"color:blue\">Home</body></html>\n"},
            ]
        })
    if "python app" in lowered:
        return json.dumps({
            "files": [
                {"path": "hello.py", "content": "print('hello')\n"},
                {"path": "README.md", "content": "# Hello App\n"},
            ]
        })
    if "review" in lowered:
        return "All generated files are consistent and well-formed."
    if "plain site" in lowered:
        # realistic model output: prose + fenced code, NO JSON manifest
        return (
            "Sure, here is your website!\n\n"
            "```html\n# file: index.html\n<h1>Plain site</h1>\n```\n\n"
            "```css\n/* style.css */\nbody { font-family: sans-serif; }\n```\n"
        )
    if "novel" in lowered:
        # prose with no manifest and no fences at all
        return "Here is my outline: it should start at dawn and end after the hero wins the vote."
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
    monkeypatch.setattr(type(boot.providers), "load_all", lambda self: None)
    from synapse.api import build_app

    app = build_app(boot)
    with TestClient(app) as c:
        yield c, boot


@pytest.fixture
def engine(temp_paths: SynapsePaths) -> ActionEngine:
    return ActionEngine(FileOperator(temp_paths.data_dir / "work"))


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


# ===========================================================================
# 1. Request classification — every prompt maps to exactly one of 7 kinds
# ===========================================================================

class TestClassifier:
    @pytest.mark.parametrize("prompt", [
        "Create a portfolio website",
        "Build a dashboard app for my data",
        "Generate a landing page project",
        "Make a game",
    ])
    def test_project_generation(self, prompt):
        assert classify_request(prompt) is RequestKind.PROJECT_GENERATION

    @pytest.mark.parametrize("prompt", [
        "Write a python script that parses csv",
        "Create a function that sums a list",
        "Generate a unit test for the calculator",
    ])
    def test_file_creation(self, prompt):
        assert classify_request(prompt) is RequestKind.FILE_CREATION

    @pytest.mark.parametrize("prompt", [
        "make my website blue",
        "Change the color of my website",
        "Edit the style.css file",
        "Fix the bug in my script",
        "Add a dark mode feature",
    ])
    def test_file_modification(self, prompt):
        assert classify_request(prompt) is RequestKind.FILE_MODIFICATION

    @pytest.mark.parametrize("prompt", [
        "Generate documentation for my project",
        "Write a readme for the repo",
        "Create a user guide",
        "Update the changelog",
    ])
    def test_documentation(self, prompt):
        assert classify_request(prompt) is RequestKind.DOCUMENTATION

    @pytest.mark.parametrize("prompt", [
        "Analyze my codebase",
        "Explain this project to me",
        "What does this code do?",
        "Review the files in my workspace",
    ])
    def test_project_analysis(self, prompt):
        assert classify_request(prompt) is RequestKind.PROJECT_ANALYSIS

    @pytest.mark.parametrize("prompt", [
        "List files in my project",
        "Show the structure of my workspace",
        "Read main.py",
        "Open index.html",
        "Show the contents of src/app.js",
        "Rename app.py to main.py",
        "Move README.md to docs/",
        "Delete old.txt",
        "Search for print statements",
        "Create a folder called assets",
    ])
    def test_workspace_operation(self, prompt):
        assert classify_request(prompt) is RequestKind.WORKSPACE_OPERATION

    def test_read_whole_project_stays_analysis(self):
        assert classify_request("read the project") is RequestKind.PROJECT_ANALYSIS
        assert classify_request("read my files") is RequestKind.PROJECT_ANALYSIS

    @pytest.mark.parametrize("prompt", [
        "What is the meaning of life?",
        "Tell me a joke",
        "How do I cook pasta?",
        "",
    ])
    def test_chat_response(self, prompt):
        assert classify_request(prompt) is RequestKind.CHAT_RESPONSE

    def test_creation_wins_over_chat(self):
        assert classify_request("I want to chat about websites") is RequestKind.CHAT_RESPONSE


# ===========================================================================
# 2. Explicit workspace-op extraction
# ===========================================================================

class TestExtractWorkspaceOps:
    def test_list_op(self):
        ops = extract_workspace_ops("List files in my project")
        assert ops == [{"action": "list"}]

    def test_search_op(self):
        ops = extract_workspace_ops("Search for 'print' in my files")
        assert ops == [{"action": "search", "pattern": "print"}]

    def test_read_op(self):
        ops = extract_workspace_ops("Read src/main.py")
        assert ops == [{"action": "read", "path": "src/main.py"}]

    def test_read_whole_project_has_no_op(self):
        assert extract_workspace_ops("read the project") == []

    def test_rename_op(self):
        ops = extract_workspace_ops("Rename app.py to main.py")
        assert ops == [{"action": "rename", "path": "app.py", "to": "main.py"}]

    def test_move_is_rename(self):
        ops = extract_workspace_ops("Move README.md to docs/readme.md")
        assert ops == [{"action": "rename", "path": "README.md", "to": "docs/readme.md"}]

    def test_delete_is_confirmed(self):
        ops = extract_workspace_ops("Delete stale.txt from the project")
        assert ops == [{"action": "delete", "path": "stale.txt", "confirmed": True}]

    def test_create_folder_op(self):
        ops = extract_workspace_ops("Create a folder called assets")
        assert ops == [{"action": "create_folder", "path": "assets"}]

    def test_multiple_ops_deduplicated(self):
        ops = extract_workspace_ops("List files. then list files again. and rename a.txt to b.txt")
        assert ops == [{"action": "list"}, {"action": "rename", "path": "a.txt", "to": "b.txt"}]

    def test_chat_has_no_ops(self):
        assert extract_workspace_ops("What is the meaning of life?") == []


# ===========================================================================
# 3. ActionEngine — execution, gates, verification, summary
# ===========================================================================

class TestActionEngine:
    def test_apply_writes_and_creates_folders(self, engine: ActionEngine):
        actions, validations = engine.apply([
            {"action": "create_folder", "path": "assets"},
            {"action": "create_folder", "path": "css"},
            {"action": "write", "path": "index.html", "content": "<!doctype html>\n<html></html>\n"},
            {"action": "write", "path": "css/style.css", "content": "body {}\n"},
        ])
        assert (engine._op.root / "assets").is_dir()
        assert (engine._op.root / "css").is_dir()
        assert engine.read("index.html").startswith("<!doctype html>")
        kinds = {a.action for a in actions}
        assert {"created", "created_folder"} <= kinds
        assert all(a.status == "ok" for a in actions)
        assert all(v.ok for v in validations)

    def test_delete_requires_confirmation(self, engine: ActionEngine):
        engine.write("stale.txt", "x")
        actions, _ = engine.apply([{"action": "delete", "path": "stale.txt"}])
        assert actions[0].status == "failed"
        assert "confirmation" in actions[0].error
        assert engine.exists("stale.txt")

    def test_delete_with_confirmation(self, engine: ActionEngine):
        engine.write("stale.txt", "x")
        actions, _ = engine.apply([{"action": "delete", "path": "stale.txt"}], confirm_delete=True)
        assert actions[0].status == "ok"
        assert not engine.exists("stale.txt")

    def test_edit_op(self, engine: ActionEngine):
        engine.write("app.js", "let color = 'red';\n")
        actions, validations = engine.apply([
            {"action": "edit", "path": "app.js", "old": "red", "new": "blue"},
        ])
        assert engine.read("app.js") == "let color = 'blue';\n"
        assert actions[0].action == "modified"
        assert actions[0].validated

    def test_edit_missing_old_text_fails(self, engine: ActionEngine):
        engine.write("app.js", "let color = 'red';\n")
        actions, _ = engine.apply([
            {"action": "edit", "path": "app.js", "old": "zzz", "new": "blue"},
        ])
        assert actions[0].status == "failed"
        assert "old text" in actions[0].error

    def test_rename_and_move(self, engine: ActionEngine):
        engine.write("old.py", "x = 1\n")
        actions, _ = engine.apply([{"action": "rename", "path": "old.py", "to": "new.py"}])
        assert actions[0].action == "renamed"
        assert not engine.exists("old.py")
        assert engine.read("new.py") == "x = 1\n"

    def test_verification_flags_missing_file(self, engine: ActionEngine):
        action = FileAction(path="ghost.txt", action="created", status="ok")
        engine.verify([action])
        assert action.status == "failed"
        assert "verification" in action.error

    def test_search(self, engine: ActionEngine):
        engine.write("README.md", "# Demo\n\nWorkspace created by Synapse.\n")
        engine.write("main.py", "print('hello')\n")
        hits = engine.search("print")
        assert hits and hits[0]["path"] == "main.py"
        hits = engine.search("synapse")
        assert any(h["path"] == "README.md" for h in hits)
        hits = engine.search("[" * 5)  # invalid regex -> escaped fallback
        assert hits == []

    def test_plan_project_ops_ensures_assets(self):
        ops = ActionEngine.plan_project_ops(RequestKind.PROJECT_GENERATION, [
            {"action": "write", "path": "index.html", "content": ""},
        ])
        assert ops[0] == {"action": "create_folder", "path": "assets"}

    def test_plan_project_ops_respects_explicit_folders(self):
        ops = ActionEngine.plan_project_ops(RequestKind.PROJECT_GENERATION, [
            {"action": "create_folder", "path": "img"},
            {"action": "write", "path": "index.html", "content": ""},
        ])
        assert ops[0] == {"action": "create_folder", "path": "img"}

    def test_plan_project_ops_only_for_generation(self):
        ops = ActionEngine.plan_project_ops(RequestKind.FILE_CREATION, [
            {"action": "write", "path": "a.py", "content": ""},
        ])
        assert ops == [{"action": "write", "path": "a.py", "content": ""}]

    def test_build_context_snapshot(self, engine: ActionEngine):
        assert engine.build_context() is None
        engine.write("main.py", "print('x')\n")
        context = engine.build_context()
        assert "main.py" in context
        assert "print('x')" in context

    def test_summarize_never_contains_content(self, engine: ActionEngine):
        actions, validations = engine.apply([
            {"action": "write", "path": "a.py", "content": "secret = 42\n"},
        ])
        summary = engine.summarize(actions, validations)
        assert "Actions (1):" in summary
        assert "- created a.py [ok]" in summary
        assert "secret" not in summary

    def test_run_workspace_ops_list_and_delete(self, engine: ActionEngine):
        engine.write("README.md", "# X\n")
        engine.write("stale.txt", "y\n")
        lines, actions = engine.run_workspace_ops([
            {"action": "list"},
            {"action": "delete", "path": "stale.txt", "confirmed": True},
            {"action": "create_folder", "path": "assets"},
        ])
        assert "README.md" in "\n".join(lines)
        assert "Deleted stale.txt" in "\n".join(lines)
        assert (engine._op.root / "assets").is_dir()
        assert any(a.action == "listed" for a in actions)

    def test_run_workspace_ops_read(self, engine: ActionEngine):
        engine.write("notes.md", "alpha\nbeta\n")
        lines, actions = engine.run_workspace_ops([{"action": "read", "path": "notes.md"}])
        body = "\n".join(lines)
        assert "### notes.md" in body
        assert "alpha" in body
        assert "beta" in body
        assert actions[0].action == "read"
        assert actions[0].status == "ok"

    def test_run_workspace_ops_read_missing(self, engine: ActionEngine):
        lines, actions = engine.run_workspace_ops([{"action": "read", "path": "nope.txt"}])
        assert "not found" in lines[0]
        assert actions[0].action == "read"
        assert actions[0].status == "failed"


# ===========================================================================
# 4. Manifest additions — folders and edit ops
# ===========================================================================

class TestManifestPhase7:
    def test_folders_ops(self):
        ops = parse_file_manifest(json.dumps({"folders": ["assets", "css/js"]}))
        assert ops == [
            {"action": "create_folder", "path": "assets"},
            {"action": "create_folder", "path": "css/js"},
        ]

    def test_folders_and_files(self):
        ops = parse_file_manifest(json.dumps({
            "dirs": ["assets"],
            "files": [{"path": "a.txt", "content": "x"}],
        }))
        assert ops[0] == {"action": "create_folder", "path": "assets"}
        assert ops[1] == {"action": "write", "path": "a.txt", "content": "x"}

    def test_unsafe_folder_flagged(self):
        ops = parse_file_manifest(json.dumps({"folders": ["../escape"]}))
        assert len(ops) == 1
        assert "error" in ops[0]

    def test_edit_op(self):
        ops = parse_file_manifest(json.dumps({
            "files": [{"action": "edit", "path": "a.py", "old": "red", "new": "blue"}],
        }))
        assert ops[0] == {"action": "edit", "path": "a.py", "old": "red", "new": "blue"}


# ===========================================================================
# 4b. Prose / fenced-code fallback (real-model output without a JSON manifest)
# ===========================================================================

class TestManifestFallbackPhase7:
    def test_multi_fence_with_file_headers(self):
        text = (
            "Sure! Here is your site:\n\n"
            "```html\n# file: index.html\n<h1>Hi</h1>\n```\n\n"
            "```css\n/* style.css */\nbody { color: red; }\n```\n\n"
            "```javascript\n// script.js\nconsole.log('x');\n```\n"
        )
        assert parse_file_manifest(text) == [
            {"action": "write", "path": "index.html", "content": "<h1>Hi</h1>"},
            {"action": "write", "path": "style.css", "content": "body { color: red; }"},
            {"action": "write", "path": "script.js", "content": "console.log('x');"},
        ]

    def test_fence_bare_path_first_line(self):
        text = "```python\nmain.py\nprint('hello')\n```"
        ops = parse_file_manifest(text)
        assert ops[0]["path"] == "main.py"
        assert ops[0]["content"] == "print('hello')"

    def test_fence_language_default_names(self):
        text = "```html\n<p>a</p>\n```\n```css\np{color:teal}\n```\n```js\nx=1\n```"
        paths = [o["path"] for o in parse_file_manifest(text)]
        assert paths == ["index.html", "style.css", "script.js"]

    def test_fence_uses_planner_hint_over_lang_default(self):
        ops = parse_file_manifest("```python\nprint(1)\n```", hint="main.py")
        assert ops == [{"action": "write", "path": "main.py", "content": "print(1)"}]

    def test_colliding_names_get_suffix_not_lost(self):
        text = "```html\n# file: index.html\nA\n```\n```html\n# file: index.html\nB\n```"
        ops = parse_file_manifest(text)
        assert [o["path"] for o in ops] == ["index.html", "index_2.html"]
        assert ops[1]["content"] == "B"

    def test_empty_fences_skipped(self):
        text = "```python\n\n```\n```css\n/* app.css */\nbody{}\n```"
        assert parse_file_manifest(text) == [
            {"action": "write", "path": "app.css", "content": "body{}"}
        ]

    def test_pure_prose_without_fences_is_empty(self):
        assert parse_file_manifest("Here is the code:\n\nprint('hi')") == []

    def test_instruction_guides_models_to_parseable_shapes(self):
        from synapse.workspace.manifest import MANIFEST_INSTRUCTION

        # the constant every file-output task prompt ends with: it must tell
        # the model to emit exactly the shapes this parser understands.
        assert '"files"' in MANIFEST_INSTRUCTION
        assert '"folders"' in MANIFEST_INSTRUCTION
        assert "JSON" in MANIFEST_INSTRUCTION
        # and a model reply that follows it parses cleanly
        reply = json.dumps({
            "folders": ["assets"],
            "files": [{"path": "a.py", "content": "x=1"}],
        })
        assert parse_file_manifest(reply) == [
            {"action": "create_folder", "path": "assets"},
            {"action": "write", "path": "a.py", "content": "x=1"},
        ]


# ===========================================================================
# 5. Planner — modifications & generation become file tasks
# ===========================================================================

class TestPlannerPhase7:
    @pytest.mark.parametrize("prompt", [
        "Edit the style.css file to use a new palette",
        "Change the color of my website to blue",
        "Fix the bug in my script",
        "Add a dark mode feature to the app",
        "make my website blue",
    ])
    def test_modification_is_file_task(self, prompt):
        dag = _plan(prompt)
        assert dag.get("t1").file_output is True

    @pytest.mark.parametrize("prompt", [
        "Create a portfolio website",
        "create a blog with a dark theme",
        "make a todo list app",
        "build an ecommerce store",
        "create a chat application",
        "create a plain site",
    ])
    def test_generation_is_file_task(self, prompt):
        dag = _plan(prompt)
        assert dag.get("t1").file_output is True
        assert dag.get("t1").kind.value in ("coding", "writing")

    def test_generation_stays_single_task(self):
        # "create a website with html, css and js" must not be shredded into
        # per-word tasks that leak code into the chat synthesis.
        dag = _plan("Create a portfolio website with index.html, style.css and script.js")
        assert dag.get("t1").file_output is True
        assert len([t for t in dag.tasks if t.kind.value != "review"]) == 1

    def test_analysis_is_not_a_file_task(self):
        dag = _plan("Analyze my codebase and report issues")
        assert dag.get("t1").file_output is False

    @pytest.mark.parametrize("prompt", [
        "Analyze the codebase and save the results to findings.md",
        "Research the topic and write the summary to notes.txt",
        "Review the design and output a changelog to docs/changelog.md",
    ])
    def test_named_destination_is_file_task_any_kind(self, prompt):
        # Phase X — "save/write ... to <file.ext>" makes ANY kind of agent a
        # file-writing agent; analysis/research prompts keep their kind (no
        # forced coding upgrade) but gain tools- and JSON-capable routing.
        dag = _plan(prompt)
        tasks = [t for t in dag.tasks if t.kind.value != "review"]
        assert tasks, "at least one real task"
        assert any(t.file_output for t in tasks)
        file_task = next(t for t in tasks if t.file_output)
        assert Capability.TOOLS in file_task.preferred_capabilities
        assert Capability.JSON in file_task.preferred_capabilities

    def test_pure_chat_stays_general_routing(self):
        dag = _plan("Tell me a joke about robots")
        task = dag.get("t1")
        assert task.file_output is False
        assert Capability.TOOLS not in task.preferred_capabilities
        assert Capability.JSON not in task.preferred_capabilities


# ===========================================================================
# 6. End-to-end — project generation writes files, chat shows only summary
# ===========================================================================

class TestEndToEndPhase7:
    def test_portfolio_project_written_to_workspace(self, client):
        c, _ = client
        pid = c.post("/projects", json={"name": "Portfolio"}).json()["id"]
        resp = c.post("/request", json={"prompt": "Create a portfolio website", "project_id": pid})
        assert resp.status_code == 200, resp.text
        body = resp.json()

        workspace = Path(next(p for p in c.get("/projects").json() if p["id"] == pid)["workspace_path"])
        for rel in ("index.html", "style.css", "script.js"):
            assert (workspace / rel).is_file(), rel
        assert (workspace / "assets").is_dir()
        assert (workspace / "css").is_dir()
        assert (workspace / "js").is_dir()

        # the chat response is a summary — never the generated code
        response = body["response"]
        assert "Actions (" in response
        assert "- created index.html [ok]" in response
        assert "<!doctype" not in response
        assert "style.css" in response

        # action log records folders + files
        entry = c.get(f"/projects/{pid}/actions").json()[0]
        tool_actions = {a["action"] for a in entry["tool_actions"]}
        assert "created_folder" in tool_actions
        assert "index.html" in entry["files_created"]

    def test_modification_writes_back_into_workspace(self, client):
        c, _ = client
        pid = c.post("/projects", json={"name": "Restyle"}).json()["id"]
        c.post("/request", json={"prompt": "Create a python app", "project_id": pid})
        resp = c.post("/request", json={"prompt": "make my website blue", "project_id": pid})
        assert resp.status_code == 200, resp.text
        content = c.get(f"/projects/{pid}/work/file", params={"path": "index.html"}).json()["content"]
        assert "color:blue" in content
        assert "<!doctype" not in resp.json()["response"]

    def test_manifest_delete_without_confirmation_fails_safely(self, client):
        c, _ = client
        pid = c.post("/projects", json={"name": "Cleanup"}).json()["id"]
        resp = c.post("/request", json={"prompt": "Create a python app with cleanup", "project_id": pid})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert any(
            a["action"] == "deleted" and a["status"] == "failed" and "confirmation" in a["error"]
            for a in body["actions"]
        )
        paths = {f["path"] for f in c.get(f"/projects/{pid}/work").json()}
        assert "hello.py" in paths

    # -- backend-only workspace operations ----------------------------------

    def test_list_files_answered_by_backend(self, client):
        c, _ = client
        pid = c.post("/projects", json={"name": "Listing"}).json()["id"]
        resp = c.post("/request", json={"prompt": "List files in my project", "project_id": pid})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        response = body["response"]
        assert "Files in workspace (1):" in response
        assert "- README.md" in response
        # no model was consulted
        assert body["execution_graph"]["nodes"] == []
        assert any(a["action"] == "listed" for a in body["actions"])

    def test_search_answered_by_backend(self, client):
        c, _ = client
        pid = c.post("/projects", json={"name": "Search"}).json()["id"]
        resp = c.post("/request", json={"prompt": "Search for synapse in my files", "project_id": pid})
        assert resp.status_code == 200, resp.text
        assert "README.md" in resp.json()["response"]

    def test_rename_answered_by_backend(self, client):
        c, _ = client
        pid = c.post("/projects", json={"name": "Rename"}).json()["id"]
        resp = c.post("/request", json={"prompt": "Rename README.md to index.md", "project_id": pid})
        assert resp.status_code == 200, resp.text
        assert "Renamed README.md -> index.md" in resp.json()["response"]
        paths = {f["path"] for f in c.get(f"/projects/{pid}/work").json()}
        assert "index.md" in paths and "README.md" not in paths
    def test_delete_answered_by_backend(self, client):
        c, _ = client
        pid = c.post("/projects", json={"name": "Delete"}).json()["id"]
        resp = c.post("/request", json={"prompt": "Delete README.md from the project", "project_id": pid})
        assert resp.status_code == 200, resp.text
        assert "Deleted README.md" in resp.json()["response"]
        assert c.get(f"/projects/{pid}/work").json() == []

    def test_create_folder_answered_by_backend(self, client):
        c, _ = client
        pid = c.post("/projects", json={"name": "Folders"}).json()["id"]
        resp = c.post("/request", json={"prompt": "Create a folder called assets", "project_id": pid})
        assert resp.status_code == 200, resp.text
        workspace = Path(next(p for p in c.get("/projects").json() if p["id"] == pid)["workspace_path"])
        assert (workspace / "assets").is_dir()

    def test_plain_text_model_output_still_creates_files(self, client):
        # A real LLM often answers in prose + fenced code, not a JSON manifest.
        # The backend must still write every block to the workspace and the chat
        # must show only the summary — the code lands on disk, not in the reply.
        c, _ = client
        pid = c.post("/projects", json={"name": "Plain"}).json()["id"]
        resp = c.post("/request", json={"prompt": "create a plain site", "project_id": pid})
        assert resp.status_code == 200, resp.text
        body = resp.json()

        workspace = Path(next(p for p in c.get("/projects").json() if p["id"] == pid)["workspace_path"])
        assert (workspace / "index.html").is_file()
        assert (workspace / "style.css").is_file()
        assert "<h1>Plain site</h1>" in (workspace / "index.html").read_text()

        response = body["response"]
        assert "Actions (" in response
        assert "- created index.html [ok]" in response
        assert "<h1>" not in response
        assert "```" not in response
        assert any(a["action"] == "created" and a["path"] == "index.html" for a in body["actions"])

    def test_unparsable_model_output_never_dumps_code_into_chat(self, client):
        c, _ = client
        pid = c.post("/projects", json={"name": "Novel"}).json()["id"]
        resp = c.post("/request", json={"prompt": "create a novel site", "project_id": pid})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # a summary with the failed action is shown — never the raw model text
        assert "FAILED" in body["response"]
        assert "dawn" not in body["response"]
        assert any(a["status"] == "failed" and "no file manifest" in a["error"] for a in body["actions"])
