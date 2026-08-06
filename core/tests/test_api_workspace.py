"""Phase 4 API tests — /files, /jobs, /workspace, and /ui drag & drop page.

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
        return ModelMetadata(id=descriptor.id, provider_id="fake", kind=self.kind,
                             privacy_score=1.0, capabilities={"vision": 1.0, "embeddings": 1.0})

    def embed(self, texts, *, model=None):  # noqa: ANN001
        return [ [1.0] if t else [0.0] for t in texts ]

    def chat(self, request):  # noqa: ANN001
        from synapse.domain import ChatResponse
        return ChatResponse(provider_id="fake", model_id="vision-test", kind=self.kind,
                            content="ok", raw={})

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
    boot.registry._models["vision-test"] = ModelMetadata(
        id="vision-test", provider_id="fake", kind=ProviderKind.LOCAL,
        privacy_score=1.0, capabilities={"vision": 1.0, "embeddings": 1.0},
    )

    from synapse.api import build_app

    app = build_app(boot)
    with TestClient(app) as c:
        yield c


def test_ui_page_served(client):
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert "synapse" in resp.text.lower()
    assert "projSel" in resp.text
    assert "boot()" in resp.text


def test_upload_list_job_progress_delete(client):
    # upload
    resp = client.post("/files", files={"file": ("notes.md", b"# alpha database design")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"]
    assert body["file"]["pipeline"] == "document"

    # workspace summary reflects the upload
    ws = client.get("/workspace").json()
    assert ws["files_count"] == 1
    assert ws["enabled"] is True
    assert ws["vector_store"]

    # job endpoint reports progress
    job = client.get(f"/jobs/{body['job_id']}").json()
    assert job["status"] in ("running", "completed", "failed")
    assert 0 <= job["progress"] <= 100

    # list endpoint
    files = client.get("/files").json()
    assert len(files) == 1
    assert files[0]["id"] == body["file"]["id"]

    # delete
    assert client.delete(f"/files/{body['file']['id']}").status_code == 200
    assert client.delete(f"/files/{body['file']['id']}").status_code == 404
    assert client.get("/workspace").json()["files_count"] == 0


def test_upload_empty_rejected(client):
    resp = client.post("/files", files={"file": ("empty.txt", b"")})
    assert resp.status_code == 400


def test_unknown_job_404(client):
    assert client.get("/jobs/nope").status_code == 404


def test_models_endpoint(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert any(m["id"] == "vision-test" for m in data)