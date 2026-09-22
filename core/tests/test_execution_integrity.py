"""Phase XVI — execution integrity.

The final response and the action log must reflect REAL filesystem execution,
never model claims:

1. every reported creation actually exists on disk;
2. an instruction-example echo (``docs/a.md`` with content ``...``) creates
   nothing — no phantom files, no phantom folders;
3. a failed filesystem op can never appear successful (verify + retry once);
4. the previous model's output is passed as context to the next model while
   per-task capability routing keeps multi-model chains working;
5. "Hello" never invokes workspace/coding/file tools;
6. a project-generation request routes reasoning -> coding -> filesystem ->
   validation end to end.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from synapse.actions import ActionEngine  # noqa: E402
from synapse.bootstrap import Boot, create_container  # noqa: E402
from synapse.config.paths import SynapsePaths  # noqa: E402
from synapse.contracts import ModelProvider  # noqa: E402
from synapse.domain import (  # noqa: E402
    ChatResponse,
    ModelCapabilities,
    ModelDescriptor,
    ModelMetadata,
    ProviderKind,
)
from synapse.domain.enums import ProviderState  # noqa: E402
from synapse.workspace.operator import FileOperator  # noqa: E402


class FakeProvider(ModelProvider):
    """Two-capability provider (coder / reasoner) that records every prompt.

    Replies are keyed off the FULL first message so tests can prove the
    previous model's output reached the next model's prompt.
    """

    provider_id = "fake"
    kind = ProviderKind.LOCAL

    _CAPS = {
        "coder": ModelCapabilities(coding=0.95, reasoning=0.3, chat=0.6),
        "reasoner": ModelCapabilities(coding=0.1, reasoning=0.95, chat=0.4),
    }

    def __init__(self) -> None:
        self.prompts: list[str] = []

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
        content = request.messages[0].content
        self.prompts.append(content)
        lowered = content.lower()
        if "review the generated files" in lowered:
            reply = "All generated files are consistent and well-formed."
        elif "CACHE_REASONING_NOTE_42" in content:
            reply = '{"files": [{"path": "cache.py", "content": "class Cache:\\n    pass\\n"}]}'
        elif "explain why" in lowered:
            reply = "CACHE_REASONING_NOTE_42"
        elif "phantom" in lowered:
            reply = '{"folders": ["docs"], "files": [{"path": "docs/a.md", "content": "..."}]}'
        elif "output.md" in lowered:
            reply = '{"files": [{"path": "output.md", "content": "# Overview\\nA tiny project.\\n"}]}'
        elif "python script that prints hello" in lowered:
            reply = '{"files": [{"path": "hello.py", "content": "print(\'hello\')\\n"}]}'
        elif "hello" in lowered:
            reply = "Hello! How can I help you today?"
        else:
            reply = "ok"
        return ChatResponse(
            provider_id="fake",
            model_id=request.model,
            kind=self.kind,
            content=reply,
            raw={},
        )

    def health(self) -> bool:
        return True

    def supports(self, capability) -> bool:  # noqa: ANN001
        return True

    def shutdown(self) -> None:
        pass


@pytest.fixture
def app_env(temp_paths: SynapsePaths, monkeypatch):
    monkeypatch.setenv("SYNAPSE_HOME", str(temp_paths.home))
    boot = Boot(create_container(paths=temp_paths))
    fake = FakeProvider()
    boot.providers._providers["fake"] = fake
    boot.providers._states["fake"] = ProviderState.READY
    monkeypatch.setattr(type(boot.providers), "load_all", lambda self: None)
    from synapse.api import build_app

    app = build_app(boot)
    with TestClient(app) as client:
        yield client, fake, temp_paths


def _adopt(client: TestClient, temp_paths: SynapsePaths, name: str) -> tuple[str, Path]:
    """Create a project with a real workspace folder and scan it in."""
    folder = temp_paths.home / f"ws-{name}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "README.md").write_text("# Project\n", encoding="utf-8")
    pid = client.post(
        "/projects", json={"name": name, "workspace_path": str(folder)}
    ).json()["id"]
    client.post(f"/projects/{pid}/scan")
    return pid, folder


def _work_paths(client: TestClient, project_id: str) -> set[str]:
    return {f["path"] for f in client.get(f"/projects/{project_id}/work").json()}


class TestReportedStateIsReal:
    def test_reported_created_file_exists_on_disk(self, app_env):
        client, _, temp_paths = app_env
        pid, folder = _adopt(client, temp_paths, "IntegrityReport")
        resp = client.post(
            "/request",
            json={"prompt": "write the project overview to output.md", "project_id": pid},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        created = [
            a for a in body["actions"] if a["action"] == "created" and a["status"] == "ok"
        ]
        assert any(a["path"] == "output.md" for a in created), body["actions"]
        assert (folder / "output.md").is_file()
        assert "# Overview" in (folder / "output.md").read_text(encoding="utf-8")

    def test_instruction_echo_creates_no_phantom_artifacts(self, app_env):
        """A model that echoes the workspace-tool example (``docs/a.md`` with
        content ``...``) must create NOTHING — not the file, not the folder."""
        client, _, temp_paths = app_env
        pid, folder = _adopt(client, temp_paths, "Echo")
        resp = client.post(
            "/request",
            json={"prompt": "phantom site — create a tiny website", "project_id": pid},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        ok_writes = [
            a for a in body["actions"]
            if a["action"] in ("created", "modified") and a["status"] == "ok"
        ]
        assert ok_writes == [], body["actions"]
        assert _work_paths(client, pid) <= {"README.md"}, _work_paths(client, pid)
        assert not (folder / "docs").exists()
        assert not (folder / "docs" / "a.md").exists()
        assert "FAILED" in body["response"]
        assert "no file manifest found" in body["response"]

    def test_strict_echo_creates_nothing(self, app_env):
        """Same guarantee for non-file pipeline tasks (strict parsing)."""
        client, _, temp_paths = app_env
        pid, folder = _adopt(client, temp_paths, "StrictEcho")
        resp = client.post(
            "/request",
            json={"prompt": "phantom analyze the project structure", "project_id": pid},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["actions"] == []
        assert _work_paths(client, pid) <= {"README.md"}, _work_paths(client, pid)
        assert not (folder / "docs").exists()
        assert not (folder / "docs" / "a.md").exists()


class TestFailedOpsNeverLookSuccessful:
    @pytest.fixture
    def engine(self, temp_paths: SynapsePaths) -> ActionEngine:
        operator = FileOperator(temp_paths.data_dir / "engine-root")
        return ActionEngine(operator)

    def test_happy_path_write_is_verified_ok(self, engine):
        actions, _ = engine.apply([{"path": "real.txt", "content": "x"}])
        assert actions[0].status == "ok"
        assert actions[0].action == "created"

    def test_silent_noop_write_is_flagged_failed(self, engine, monkeypatch):
        """A write that reports success but writes nothing must be failed."""

        def noop(rel, content):  # noqa: ANN001, ANN202
            return {"path": rel, "action": "created", "bytes": len(content.encode())}

        monkeypatch.setattr(engine._op, "write", noop)
        actions, _ = engine.apply([{"path": "ghost.txt", "content": "x"}])
        assert actions[0].status == "failed"
        assert "verification" in actions[0].error

    def test_content_mismatch_is_flagged_failed(self, engine, monkeypatch):
        """A write that stores different bytes than reported is a failure."""
        real_write = engine._op.write

        def corrupt(rel, content):  # noqa: ANN001, ANN202
            real_write(rel, "different")
            return {"path": rel, "action": "created", "bytes": 9}

        monkeypatch.setattr(engine._op, "write", corrupt)
        actions, _ = engine.apply([{"path": "a.txt", "content": "expected"}])
        assert actions[0].status == "failed"
        assert "verification" in actions[0].error

    def test_delete_that_leaves_the_file_is_flagged_failed(self, engine, monkeypatch):
        engine.write("stale.txt", "data")
        monkeypatch.setattr(engine._op, "delete", lambda rel: True)  # lies

        actions, _ = engine.apply(
            [{"action": "delete", "path": "stale.txt"}], confirm_delete=True
        )
        assert actions[0].status == "failed"
        assert "verification" in actions[0].error
        assert engine.exists("stale.txt")  # truth on disk, failed in the log


class TestMultiModelContextChain:
    def test_previous_model_result_passed_as_context(self, app_env):
        client, fake, temp_paths = app_env
        pid, folder = _adopt(client, temp_paths, "Chain")
        resp = client.post(
            "/request",
            json={
                "prompt": "explain why caching is hard with concurrent access then write a python cache class",
                "project_id": pid,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        nodes = {n["task_id"]: n for n in body["execution_graph"]["nodes"]}
        # per-task capability routing: reasoning step -> reasoner, coding step -> coder.
        assert nodes["t1"]["model_id"] == "reasoner", nodes
        assert nodes["t2"]["model_id"] == "coder", nodes
        # the previous model's output reached the next model's prompt.
        manifest_prompts = [p for p in fake.prompts if "OUTPUT FORMAT" in p]
        assert manifest_prompts, "no file task prompt seen"
        assert any("CACHE_REASONING_NOTE_42" in p for p in manifest_prompts), fake.prompts
        # the coding model's manifest became a real, verified file.
        assert (folder / "cache.py").is_file()
        assert "class Cache:" in (folder / "cache.py").read_text(encoding="utf-8")

    def test_review_sees_produced_files_as_paths_only(self, app_env):
        client, fake, temp_paths = app_env
        pid, _ = _adopt(client, temp_paths, "ChainReview")
        resp = client.post(
            "/request",
            json={
                "prompt": "explain why caching is hard with concurrent access then write a python cache class",
                "project_id": pid,
            },
        )
        assert resp.status_code == 200, resp.text
        assert any(
            "files produced in this step: cache.py" in p for p in fake.prompts
        ), fake.prompts


class TestChatNeverTouchesTools:
    def test_hello_never_invokes_file_tools(self, app_env):
        client, fake, temp_paths = app_env
        pid, folder = _adopt(client, temp_paths, "HelloOnly")
        resp = client.post(
            "/request", json={"prompt": "Hello", "project_id": pid}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["actions"] == []
        assert body["intent"] in ("conversation", "question_answering")
        assert len(fake.prompts) == 1, "chat must make exactly one model call"
        assert "OUTPUT FORMAT" not in fake.prompts[0]
        assert "manifest" not in fake.prompts[0]
        assert _work_paths(client, pid) <= {"README.md"}, _work_paths(client, pid)


class TestProjectGenerationChain:
    def test_routes_coding_filesystem_and_review(self, app_env):
        client, fake, temp_paths = app_env
        pid, folder = _adopt(client, temp_paths, "GenChain")
        resp = client.post(
            "/request",
            json={
                "prompt": "build a python script that prints hello",
                "project_id": pid,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        nodes = {n["task_id"]: n for n in body["execution_graph"]["nodes"]}
        assert nodes["t1"]["model_id"] == "coder", nodes
        assert nodes["t-review"]["model_id"] == "reasoner", nodes
        assert nodes["t-review"]["status"] == "completed"
        assert any(
            a["action"] == "created" and a["path"] == "hello.py" and a["status"] == "ok"
            for a in body["actions"]
        ), body["actions"]
        assert (folder / "hello.py").read_text(encoding="utf-8") == "print('hello')\n"
        assert "hello.py" in body["response"]
