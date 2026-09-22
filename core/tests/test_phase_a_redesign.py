"""Phase A tests — AI Operating Workspace redesign foundation.

Covers: Goal entity + GoalStore, WorkspaceType + workspace-type setting,
USER memory scope, engines/ namespace, ToolRegistry wiring, goals + home API
endpoints, and session view. All existing behavior is preserved — these are
additive, backward-compatible changes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from synapse.bootstrap import Boot, create_container  # noqa: E402
from synapse.config.paths import SynapsePaths  # noqa: E402
from synapse.contracts import ModelProvider  # noqa: E402
from synapse.domain import ModelDescriptor, ModelMetadata, ProviderKind  # noqa: E402
from synapse.domain.enums import GoalStatus, MemoryScope, WorkspaceType  # noqa: E402
from synapse.domain.enums import ProviderState  # noqa: E402


class FakeProvider(ModelProvider):
    provider_id = "fake"
    kind = ProviderKind.LOCAL

    def initialize(self) -> None:
        pass

    def list_models(self) -> list[ModelDescriptor]:
        return [ModelDescriptor(id="vision-test", provider_id="fake")]

    def to_metadata(self, descriptor: ModelDescriptor) -> ModelMetadata | None:
        return ModelMetadata(
            id=descriptor.id, provider_id="fake", kind=self.kind,
            privacy_score=1.0, capabilities={"vision": 1.0, "embeddings": 1.0},
        )

    def embed(self, texts, *, model=None):  # noqa: ANN001
        return [[1.0] if t else [0.0] for t in texts]

    def chat(self, request):  # noqa: ANN001
        from synapse.domain import ChatResponse
        return ChatResponse(provider_id="fake", model_id="vision-test",
                            kind=self.kind, content="ok", raw={})

    def health(self) -> bool:
        return True

    def supports(self, capability) -> bool:  # noqa: ANN001
        return True

    def shutdown(self) -> None:
        pass


@pytest.fixture
def boot(temp_paths: SynapsePaths, monkeypatch):
    monkeypatch.setenv("SYNAPSE_HOME", str(temp_paths.home))
    boot = Boot(create_container(paths=temp_paths))
    fake = FakeProvider()
    boot.providers._providers["fake"] = fake
    boot.providers._states["fake"] = ProviderState.READY
    boot.registry._models["vision-test"] = ModelMetadata(
        id="vision-test", provider_id="fake", kind=ProviderKind.LOCAL,
        privacy_score=1.0, capabilities={"vision": 1.0, "embeddings": 1.0},
    )
    yield boot
    boot.shutdown()


@pytest.fixture
def client(boot):
    from synapse.api import build_app

    app = build_app(boot)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def project(client, temp_paths):
    (temp_paths.home / "ws-a").mkdir(parents=True, exist_ok=True)
    resp = client.post("/projects", json={"name": "ws-a", "parent_dir": str(temp_paths.home)})
    assert resp.status_code == 200
    return resp.json()


# -- GoalStore ---------------------------------------------------------------


def test_goal_store_crud(boot, project):
    store = boot.projects.goal_store(project["id"])
    goal = store.create("Prepare the history presentation", deadline="2026-09-01")
    assert goal.status == GoalStatus.ACTIVE
    assert goal.progress == 0
    assert "history presentation" in goal.title
    assert goal.deadline == "2026-09-01"

    # list + get
    listed = store.list()
    assert any(g.id == goal.id for g in listed)

    # rename + status + progress
    store.rename(goal.id, "History deck")
    assert store.get(goal.id).title == "History deck"
    store.update_status(goal.id, GoalStatus.PAUSED)
    assert store.get(goal.id).status == GoalStatus.PAUSED
    store.set_progress(goal.id, 42)
    assert store.get(goal.id).progress == 42

    # steps
    from synapse.domain.goals import GoalStep

    store.set_steps(goal.id, [GoalStep(description="research"), GoalStep(description="outline")])
    updated = store.get(goal.id)
    assert len(updated.steps) == 2
    store.update_step(goal.id, 0, status="completed")
    updated = store.get(goal.id)
    assert updated.steps[0].status == "completed"
    assert updated.progress == 50  # 1 of 2 steps done

    # link artifacts (idempotent)
    store.link(goal.id, chat_id="c1", file_id="f1")
    store.link(goal.id, chat_id="c1")  # duplicate — no double entry
    assert store.get(goal.id).linked_chats == ["c1"]
    assert store.get(goal.id).linked_files == ["f1"]

    # archive hides from default list
    store.update_status(goal.id, GoalStatus.ARCHIVED)
    assert store.get(goal.id).status == GoalStatus.ARCHIVED
    assert not any(g.id == goal.id for g in store.list())
    assert any(g.id == goal.id for g in store.list(include_archived=True))

    # delete
    assert store.delete(goal.id)
    assert store.get(goal.id) is None


def test_goal_store_persistence_across_instances(boot, project):
    pid = project["id"]
    goal = boot.projects.goal_store(pid).create("Persist me")
    # a fresh store instance over the same dir reads the same data
    store2 = boot.projects.goal_store(pid)
    assert store2.get(goal.id) is not None
    assert store2.get(goal.id).title == "Persist me"


def test_goal_requires_outcome(client, project):
    resp = client.post(f"/projects/{project['id']}/goals", json={})
    assert resp.status_code == 400


def test_goal_api_flow(client, project):
    pid = project["id"]
    created = client.post(
        f"/projects/{pid}/goals", json={"outcome": "Draft a business plan", "deadline": "2026-10-01"}
    ).json()
    gid = created["id"]
    assert created["status"] == "active"

    listed = client.get(f"/projects/{pid}/goals").json()
    assert any(g["id"] == gid for g in listed)

    # patch status + progress
    patched = client.patch(f"/projects/{pid}/goals/{gid}", json={"status": "done", "progress": 100}).json()
    assert patched["status"] == "done"
    assert patched["progress"] == 100

    # invalid status rejected
    assert client.patch(f"/projects/{pid}/goals/{gid}", json={"status": "bogus"}).status_code == 422

    # link
    linked = client.post(f"/projects/{pid}/goals/{gid}/link", json={"chat_id": "c9"}).json()
    assert "c9" in linked["linked_chats"]

    # 404s
    assert client.get(f"/projects/{pid}/goals/nope").status_code == 404
    assert client.patch(f"/projects/{pid}/goals/nope", json={"title": "x"}).status_code == 404
    assert client.delete(f"/projects/{pid}/goals/nope").status_code == 404

    # delete
    assert client.delete(f"/projects/{pid}/goals/{gid}").status_code == 200
    assert client.get(f"/projects/{pid}/goals/{gid}").status_code == 404


# -- WorkspaceType ------------------------------------------------------------


def test_workspace_type_default_and_set(boot, project):
    pid = project["id"]
    assert boot.projects.workspace_type(pid) == "general"

    assert boot.projects.set_workspace_type(pid, "writer")
    assert boot.projects.workspace_type(pid) == "writer"

    # invalid type rejected, previous value kept
    assert not boot.projects.set_workspace_type(pid, "banana")
    assert boot.projects.workspace_type(pid) == "writer"

    # persisted in project settings
    info = boot.projects.get_project(pid)
    assert info.settings.get("workspace_type") == "writer"


def test_workspace_type_api(client, project):
    pid = project["id"]
    resp = client.post(f"/projects/{pid}/type", json={"workspace_type": "research"})
    assert resp.status_code == 200
    assert resp.json()["workspace_type"] == "research"

    assert client.post(f"/projects/{pid}/type", json={"workspace_type": "nope"}).status_code == 422
    assert client.post("/projects/nope/type", json={"workspace_type": "research"}).status_code == 404

    home = client.get(f"/projects/{pid}/home").json()
    assert home["workspace_type"] == "research"


def test_workspace_type_enum():
    assert {t.value for t in WorkspaceType} == {
        "general", "student", "research", "business", "developer",
        "writer", "designer", "teacher", "custom",
    }


# -- USER memory scope ---------------------------------------------------------


def test_user_memory_scope(boot, project):
    pid = project["id"]
    mem = boot.projects.memory_for(pid)
    entry = mem.save(MemoryScope.USER, "User prefers dark mode and concise answers")
    assert entry.scope == MemoryScope.USER

    hits = mem.search(MemoryScope.USER, "prefers dark mode")
    assert any(h.id == entry.id for h in hits)

    recent = mem.recent(MemoryScope.USER)
    assert any(e.id == entry.id for e in recent)


# -- engines namespace ----------------------------------------------------------


def test_engines_namespace():
    import synapse.engines as engines

    assert engines.MasterAgent is not None
    assert engines.WorkspaceSystem is not None
    assert engines.WorkspaceMemory is not None
    assert engines.ToolRegistry is not None
    assert engines.ProviderManager is not None
    assert engines.Router is not None
    assert engines.GoalStore is not None
    assert engines.TerminalRunner is not None


# -- ToolRegistry wiring --------------------------------------------------------


def test_tool_registry_wired(boot, project):
    pid = project["id"]
    registry = boot.projects.tool_registry(pid)
    names = {t.name for t in registry.list_tools()}
    assert {"read_file", "write_file", "edit_file", "list_files", "search_files",
            "create_folder", "delete_file", "run_terminal"} <= names

    # terminal tool executes in the workspace folder
    schemas = {t.name: t for t in registry.list_tools()}
    assert "command" in schemas["run_terminal"].parameters

    from synapse.pipeline.tools import ToolCall

    ws_path = Path(project["workspace_path"])
    (ws_path / "hello.txt").write_text("hi", encoding="utf-8")
    result = registry.execute(ToolCall(id="1", tool_name="read_file", arguments={"path": "hello.txt"}))
    assert result.success and result.output == "hi"


# -- workspace home + session view ----------------------------------------------


def test_home_endpoint(client, project):
    pid = project["id"]
    client.post(f"/projects/{pid}/goals", json={"outcome": "Study for finals"})

    home = client.get(f"/projects/{pid}/home").json()
    assert home["project_id"] == pid
    assert home["workspace_type"] == "general"
    assert home["active_goals_count"] == 1
    assert home["goals_count"] == 1
    assert "Study for finals" in home["active_goals"][0]["title"]
    assert client.get("/projects/nope/home").status_code == 404


def test_session_view(client):
    assert client.get("/session").json().get("view") in ("home", "chats", None)
    resp = client.post("/session/view", json={"view": "goals"}).json()
    assert resp["view"] == "goals"
    assert client.get("/session").json()["view"] == "goals"

    resp = client.post("/session", json={"view": "documents"}).json()
    assert resp["view"] == "documents"
