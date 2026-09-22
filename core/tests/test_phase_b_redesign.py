"""Phase B tests — AI Operating Workspace goal pipeline + workspace stores.

Covers: NoteStore/DocumentStore/TodoStore (CRUD, persistence, validation),
WorkspaceSystem factories, GoalPlanner per-workspace-type templates, the goal
plan API endpoint, and the enriched home aggregate. All additive and
backward-compatible with Phase A behavior."""

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
    (temp_paths.home / "ws-b").mkdir(parents=True, exist_ok=True)
    resp = client.post("/projects", json={"name": "ws-b", "parent_dir": str(temp_paths.home)})
    assert resp.status_code == 200
    return resp.json()


# -- notes --------------------------------------------------------------------


def test_note_store_crud_and_persistence(boot, project):
    pid = project["id"]
    store = boot.projects.note_store(pid)
    note = store.create("Ideas", content="first line\nmore", tags=["brainstorm"])
    assert note.title == "Ideas"
    assert note.tags == ["brainstorm"]

    assert any(n.id == note.id for n in store.list())

    updated = store.update(note.id, content="rewritten", tags=[])
    assert updated.content == "rewritten"
    assert updated.tags == []

    assert store.delete(note.id)
    assert store.get(note.id) is None

    # fresh store instance over the same dir reads the same data
    persisted = store.create("Persist me")
    store2 = boot.projects.note_store(pid)
    assert store2.get(persisted.id).title == "Persist me"


def test_note_title_derived_from_content(boot, project):
    note = boot.projects.note_store(project["id"]).create("", content="## Big idea\nrest")
    assert note.title == "Big idea"


# -- documents -----------------------------------------------------------------


def test_document_store_crud(boot, project):
    store = boot.projects.document_store(project["id"])
    doc = store.create("Quarterly report", doc_type="report", content="body")
    assert doc.doc_type == "report"
    assert doc.status == "draft"

    updated = store.update(doc.id, status="final", title="Q3 report")
    assert updated.status == "final"
    assert updated.title == "Q3 report"

    # invalid status falls back to draft
    updated = store.update(doc.id, status="bogus")
    assert updated.status == "draft"

    assert store.delete(doc.id)
    assert store.get(doc.id) is None


# -- todos ---------------------------------------------------------------------


def test_todo_store_crud_and_validation(boot, project):
    store = boot.projects.todo_store(project["id"])
    todo = store.create("Write intro", priority="high", due_date="2026-09-01")
    assert todo.status.value == "todo"
    assert todo.priority.value == "high"
    assert todo.due_date == "2026-09-01"

    updated = store.update(todo.id, status="doing")
    assert updated.status.value == "doing"

    # invalid status/priority are ignored
    updated = store.update(todo.id, status="banana", priority="banana")
    assert updated.status.value == "doing"
    assert updated.priority.value == "high"

    assert store.delete(todo.id)
    assert store.get(todo.id) is None


# -- WorkspaceSystem factories ---------------------------------------------------


def test_workspace_factories_and_caches(boot, project):
    pid = project["id"]
    assert boot.projects.note_store(pid) is boot.projects.note_store(pid)
    assert boot.projects.document_store(pid) is boot.projects.document_store(pid)
    assert boot.projects.todo_store(pid) is boot.projects.todo_store(pid)

    boot.projects.note_store(pid).create("A note")
    boot.projects.document_store(pid).create("A doc")
    boot.projects.todo_store(pid).create("A task")

    assert len(boot.projects.notes(pid)) == 1
    assert len(boot.projects.documents(pid)) == 1
    assert len(boot.projects.todos(pid)) == 1


# -- GoalPlanner ----------------------------------------------------------------


def test_goal_planner_templates():
    from synapse.planner.goal_planner import GoalPlanner

    planner = GoalPlanner()

    dev = planner.plan("Build a CLI tool", "developer")
    caps = [s.capability for s in dev]
    assert "coding" in caps
    assert "debugging" in caps
    assert all(s.status == "pending" for s in dev)

    writer = planner.plan("Write a short story", "writer")
    assert "writing" in {s.capability for s in writer}
    assert "planning" in {s.capability for s in writer}

    # unknown workspace types fall back to the general template
    general = planner.plan("Anything", "unknown_type")
    assert len(general) >= 3

    # every capability is a valid Capability value
    from synapse.domain import Capability

    for step in dev + writer + general:
        assert step.capability in {c.value for c in Capability}


# -- API ------------------------------------------------------------------------


def test_notes_api(client, project):
    pid = project["id"]
    created = client.post(f"/projects/{pid}/notes", json={"title": "Sketch", "content": "draft"}).json()
    assert created["title"] == "Sketch"

    listed = client.get(f"/projects/{pid}/notes").json()
    assert any(n["id"] == created["id"] for n in listed)

    patched = client.patch(f"/projects/{pid}/notes/{created['id']}", json={"content": "v2"}).json()
    assert patched["content"] == "v2"

    assert client.get(f"/projects/{pid}/notes/nope").status_code == 404
    assert client.delete(f"/projects/{pid}/notes/{created['id']}").status_code == 200
    assert client.get("/projects/nope/notes").status_code == 404


def test_documents_api(client, project):
    pid = project["id"]
    created = client.post(f"/projects/{pid}/documents", json={"title": "Plan", "doc_type": "business_plan"}).json()
    assert created["doc_type"] == "business_plan"
    assert created["status"] == "draft"

    patched = client.patch(f"/projects/{pid}/documents/{created['id']}", json={"status": "final"}).json()
    assert patched["status"] == "final"

    assert client.delete(f"/projects/{pid}/documents/{created['id']}").status_code == 200
    assert client.get(f"/projects/{pid}/documents/{created['id']}").status_code == 404


def test_todos_api(client, project):
    pid = project["id"]
    created = client.post(f"/projects/{pid}/todos", json={"title": "Draft intro", "priority": "high"}).json()
    assert created["priority"] == "high"

    patched = client.patch(f"/projects/{pid}/todos/{created['id']}", json={"status": "done"}).json()
    assert patched["status"] == "done"

    assert client.delete(f"/projects/{pid}/todos/{created['id']}").status_code == 200
    assert client.get(f"/projects/{pid}/todos/{created['id']}").status_code == 404


def test_goal_plan_endpoint(client, project):
    pid = project["id"]
    gid = client.post(f"/projects/{pid}/goals", json={"outcome": "Build a dashboard"}).json()["id"]

    planned = client.post(f"/projects/{pid}/goals/{gid}/plan").json()
    assert planned["steps"], "planner must produce steps"
    assert all(s["status"] == "pending" for s in planned["steps"])
    assert planned["progress"] == 0

    # 404s
    assert client.post(f"/projects/{pid}/goals/nope/plan").status_code == 404
    assert client.post("/projects/nope/goals/x/plan").status_code == 404


def test_home_counts_include_workspace_stores(client, project):
    pid = project["id"]
    client.post(f"/projects/{pid}/notes", json={"title": "n1"})
    client.post(f"/projects/{pid}/documents", json={"title": "d1"})
    client.post(f"/projects/{pid}/todos", json={"title": "t1"})

    home = client.get(f"/projects/{pid}/home").json()
    assert home["notes_count"] == 1
    assert home["documents_count"] == 1
    assert home["todos_count"] == 1


def test_goal_link_via_note_creation(client, project):
    pid = project["id"]
    gid = client.post(f"/projects/{pid}/goals", json={"outcome": "Track my thesis"}).json()["id"]

    note = client.post(f"/projects/{pid}/notes", json={"title": "Ch1 notes", "goal_id": gid}).json()
    assert note["goal_id"] == gid

    goal = client.get(f"/projects/{pid}/goals/{gid}").json()
    assert note["id"] in goal["linked_notes"]
