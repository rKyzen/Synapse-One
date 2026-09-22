"""Phase XV API tests — project index, search, changes, diagnostics, terminal.

SYNAPSE_TEST=1 (set in conftest) keeps the module from auto-booting, so we can
drive an injected test Boot through `build_app`."""

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
from synapse.domain.enums import MemoryScope, ProviderState  # noqa: E402


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
    parent = temp_paths.home
    (parent / "synapse-demo").mkdir(parents=True, exist_ok=True)
    resp = client.post("/projects", json={"name": "synapse-demo", "parent_dir": str(parent)})
    assert resp.status_code == 200
    return resp.json()


def tempfile_dir():
    import tempfile
    return tempfile.gettempdir()


def test_index_refresh_and_get(client, project):
    pid = project["id"]
    # seed a source file directly on disk
    ws_path = Path(project["workspace_path"])
    (ws_path / "app.py").write_text(
        "import os\nfrom pathlib import Path\n\ndef main():\n    return Path.cwd()\n",
        encoding="utf-8",
    )
    (ws_path / "README.md").write_text("# demo\n", encoding="utf-8")

    refreshed = client.post(f"/projects/{pid}/index/refresh").json()
    assert refreshed["total_files"] == 2
    assert refreshed["main_language"] == "python"

    body = client.get(f"/projects/{pid}/index").json()
    assert body["total_files"] == 2
    assert "app.py" in body["files"]
    app = body["files"]["app.py"]
    assert app["language"] == "python"
    assert any("main" in s for s in app["symbols"])
    assert any("os" in imp for imp in app["imports"])


def test_index_refresh_missing_project_404(client):
    assert client.post("/projects/nope/index/refresh").status_code == 404


def test_search_workspace(client, project):
    pid = project["id"]
    ws_path = Path(project["workspace_path"])
    (ws_path / "app.py").write_text(
        "import math\n\ndef fibonacci(n):\n    return n\n",
        encoding="utf-8",
    )
    client.post(f"/projects/{pid}/index/refresh")

    resp = client.get(f"/projects/{pid}/search", params={"q": "fibonacci"}).json()
    assert resp["total_results"] >= 1
    hit = resp["results"][0]
    assert hit["file_path"].endswith("app.py")
    assert "fibonacci" in hit["content"]


def test_search_missing_project_404(client):
    assert client.get("/projects/nope/search", params={"q": "x"}).status_code == 404


def test_changes_endpoints(client, boot, project):
    pid = project["id"]
    # record a change the same way the master agent does (post-write)
    boot.projects.change_panel(pid).record_created("main.py", size=12)

    body = client.get(f"/projects/{pid}/changes").json()
    assert any(c["path"] == "main.py" and c["type"] == "created" for c in body["files"])

    # clear resets the panel
    assert client.delete(f"/projects/{pid}/changes").status_code == 200
    body = client.get(f"/projects/{pid}/changes").json()
    assert body["files"] == []


def test_diagnostics_endpoints(client, project):
    pid = project["id"]
    ws_path = Path(project["workspace_path"])
    (ws_path / "broken.py").write_text("def broken(:\n    pass\n", encoding="utf-8")

    resp = client.get(f"/projects/{pid}/diagnostics").json()
    assert resp["files_analyzed"] >= 1
    assert any(d["file"].endswith("broken.py") and d["severity"] == "error"
               for d in resp["diagnostics"])


def test_terminal_endpoints(client, project):
    pid = project["id"]
    resp = client.post(f"/projects/{pid}/terminal", json={"command": "echo hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["return_code"] == 0
    assert "hello" in body["stdout"]

    # missing command -> 400
    assert client.post(f"/projects/{pid}/terminal", json={}).status_code == 400


def test_terminal_missing_project_404(client):
    assert client.post("/projects/nope/terminal", json={"command": "echo"}).status_code == 404


def test_task_submit_cancel_flow(client, project):
    pid = project["id"]
    resp = client.post(f"/projects/{pid}/tasks", json={"command": "echo task"})
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]
    assert task_id

    tasks = client.get(f"/projects/{pid}/tasks").json()
    assert any(t["id"] == task_id for t in tasks)

    detail = client.get(f"/projects/{pid}/tasks/{task_id}").json()
    assert detail["id"] == task_id

    assert client.post(f"/projects/{pid}/tasks/{task_id}/cancel").status_code == 200
    assert client.get(f"/projects/{pid}/tasks/nope").status_code == 404
