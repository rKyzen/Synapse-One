"""Phase 5/7 tests — workspace system, projects, chats, and API endpoints.

Covers:
- Domain models (ProjectInfo, ChatInfo, StoredMessage, ChatRecord)
- ProjectManager (SQLite registry, CRUD, slugify, workspace lifecycle)
- ChatStore (create, append, list, messages, delete)
- WorkspaceSystem (per-project internal storage, sessions, ensure_chat,
  cleanup of stale state, reconnect, delete modes)
- API endpoints (projects, picker, reconnect, delete, chats, messages,
  session, scoped files/workspace)
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
from synapse.domain import ModelDescriptor, ModelMetadata, ProviderKind  # noqa: E402
from synapse.domain.enums import ProviderState  # noqa: E402
from synapse.domain.projects import ChatInfo, ChatRecord, ProjectInfo, StoredMessage  # noqa: E402
from synapse.projects.chats import ChatStore  # noqa: E402
from synapse.projects.manager import (
    DEFAULT_PROJECT_ID,
    DEFAULT_PROJECT_NAME,
    ProjectManager,
    slugify,
)  # noqa: E402
from synapse.projects.system import WorkspaceSystem  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

        return ChatResponse(
            provider_id="fake",
            model_id="test-model",
            kind=self.kind,
            content="ok",
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
    boot.registry._models["test-model"] = ModelMetadata(
        id="test-model",
        provider_id="fake",
        kind=ProviderKind.LOCAL,
        privacy_score=1.0,
        capabilities={"vision": 1.0, "embeddings": 1.0},
    )

    from synapse.api import build_app

    app = build_app(boot)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def workspace_system(temp_paths: SynapsePaths, monkeypatch) -> WorkspaceSystem:
    monkeypatch.setenv("SYNAPSE_HOME", str(temp_paths.home))
    container = create_container(paths=temp_paths)
    boot = Boot(container)
    boot.start()
    return boot.projects


@pytest.fixture
def project_manager(temp_paths: SynapsePaths) -> ProjectManager:
    projects_dir = temp_paths.data_dir / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    return ProjectManager(projects_dir)


@pytest.fixture
def chat_store(temp_paths: SynapsePaths, project_manager: ProjectManager) -> ChatStore:
    info = project_manager.create("Test Chat Project")
    return ChatStore(info.id, temp_paths.chats_dir / info.id)


# ===========================================================================
# 1. Domain models
# ===========================================================================

class TestDomainModels:
    def test_project_info_fields(self):
        info = ProjectInfo(
            id="my-project",
            name="My Project",
            workspace_path="D:/Projects/My Project",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            archived=False,
            exists=True,
            settings={"theme": "dark"},
            chats_count=3,
            files_count=5,
        )
        assert info.id == "my-project"
        assert info.workspace_path == "D:/Projects/My Project"
        assert info.root == "D:/Projects/My Project"  # alias
        assert info.chats_count == 3
        assert info.settings["theme"] == "dark"
        assert not info.archived

    def test_project_info_archived(self):
        info = ProjectInfo(
            id="p1",
            name="P1",
            workspace_path="/p1",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            archived=True,
        )
        assert info.archived

    def test_project_info_exists_default_true(self):
        info = ProjectInfo(
            id="p1", name="P1", workspace_path="/p1",
            created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z",
        )
        assert info.exists is True

    def test_chat_info_fields(self):
        info = ChatInfo(
            id="chat-abc",
            project_id="my-project",
            title="Debug session",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T01:00:00Z",
            message_count=12,
        )
        assert info.id == "chat-abc"
        assert info.project_id == "my-project"
        assert info.message_count == 12

    def test_stored_message_fields(self):
        msg = StoredMessage(
            role="user",
            content="Hello world",
            created_at="2026-01-01T00:00:00Z",
            meta=["gpt-4", "2.1s"],
            trace={"intent": "greeting"},
            files=["file-a"],
        )
        assert msg.role == "user"
        assert msg.content == "Hello world"
        assert msg.meta == ["gpt-4", "2.1s"]
        assert msg.trace == {"intent": "greeting"}
        assert msg.files == ["file-a"]

    def test_stored_message_defaults(self):
        msg = StoredMessage(role="assistant", content="Hi there")
        assert msg.created_at != ""
        assert msg.meta == []
        assert msg.trace is None
        assert msg.files == []

    def test_chat_record_roundtrip(self):
        record = ChatRecord(
            id="rec-1",
            project_id="proj-1",
            title="Test record",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            messages=[
                StoredMessage(role="user", content="Hello"),
                StoredMessage(role="assistant", content="Hi there"),
            ],
        )
        data = record.model_dump()
        restored = ChatRecord(**data)
        assert restored.id == "rec-1"
        assert len(restored.messages) == 2
        assert restored.messages[1].role == "assistant"


# ===========================================================================
# 2. slugify
# ===========================================================================

class TestSlugify:
    def test_basic(self):
        assert slugify("My Project") == "my-project"

    def test_special_chars(self):
        assert slugify("Hello World! @#$%") == "hello-world"

    def test_empty(self):
        assert slugify("") == "project"

    def test_whitespace(self):
        assert slugify("   ") == "project"

    def test_unicode(self):
        result = slugify("Projet Français")
        assert result.isascii()
        assert result == "projet-fran-ais"


# ===========================================================================
# 3. ProjectManager
# ===========================================================================

class TestProjectManager:
    def test_create_project(self, project_manager: ProjectManager):
        info = project_manager.create("Alpha Project")
        assert info.id == "alpha-project"
        assert info.name == "Alpha Project"
        assert not info.archived
        assert info.exists

    def test_create_project_makes_workspace_only(self, project_manager: ProjectManager, temp_paths: SynapsePaths):
        info = project_manager.create("Dir Test")
        workspace = Path(info.workspace_path)
        assert workspace.is_dir()
        assert (workspace / "README.md").is_file()
        # workspace must not contain any Synapse metadata
        for banned in ("chats", "memory", "files", "index", "logs", "work", "config.json", "registry.json"):
            assert not (workspace / banned).exists(), banned

    def test_create_with_parent_keeps_name_as_folder(self, project_manager: ProjectManager, temp_paths: SynapsePaths):
        base = temp_paths.home / "dev"
        info = project_manager.create("Portfolio Website", parent_dir=base)
        assert Path(info.workspace_path) == base / "Portfolio Website"
        assert (base / "Portfolio Website" / "README.md").is_file()

    def test_create_default_project(self, project_manager: ProjectManager):
        info = project_manager.create(DEFAULT_PROJECT_NAME)
        assert info.id == DEFAULT_PROJECT_ID

    def test_registry_is_sqlite(self, project_manager: ProjectManager, temp_paths: SynapsePaths):
        assert (temp_paths.data_dir / "projects" / "projects.db").is_file()

    def test_rename_project(self, project_manager: ProjectManager):
        info = project_manager.create("Old Name")
        renamed = project_manager.rename(info.id, "New Name")
        assert renamed is not None
        assert renamed.name == "New Name"
        # the user's folder keeps its name — only the display name changes
        assert renamed.workspace_path == info.workspace_path

    def test_rename_project_empty_name(self, project_manager: ProjectManager):
        info = project_manager.create("Original")
        result = project_manager.rename(info.id, "")
        assert result is None

    def test_rename_nonexistent(self, project_manager: ProjectManager):
        assert project_manager.rename("nonexistent", "New") is None

    def test_archive_project(self, project_manager: ProjectManager):
        info = project_manager.create("Archivable")
        archived = project_manager.archive(info.id)
        assert archived is not None
        assert archived.archived

    def test_archive_nonexistent(self, project_manager: ProjectManager):
        assert project_manager.archive("nope") is None

    def test_unarchive(self, project_manager: ProjectManager):
        info = project_manager.create("Restorable")
        project_manager.archive(info.id)
        restored = project_manager.unarchive(info.id)
        assert restored is not None
        assert not restored.archived

    def test_delete_detaches_keeps_files(self, project_manager: ProjectManager):
        info = project_manager.create("Deletable")
        workspace = Path(info.workspace_path)
        assert project_manager.delete(info.id)
        assert workspace.exists()  # detach keeps the folder
        assert project_manager.get(info.id) is None

    def test_delete_with_workspace_erases_folder(self, project_manager: ProjectManager):
        info = project_manager.create("Deletable")
        workspace = Path(info.workspace_path)
        assert project_manager.delete(info.id, delete_workspace=True)
        assert not workspace.exists()

    def test_delete_nonexistent(self, project_manager: ProjectManager):
        assert not project_manager.delete("nope")

    def test_list_ids(self, project_manager: ProjectManager):
        project_manager.create("A")
        project_manager.create("B")
        ids = project_manager.list_ids()
        assert "a" in ids
        assert "b" in ids

    def test_list_ids_excludes_archived(self, project_manager: ProjectManager):
        project_manager.create("Keep")
        archived = project_manager.create("Gone")
        project_manager.archive(archived.id)
        ids = project_manager.list_ids(include_archived=False)
        assert archived.id not in ids

    def test_list_ids_includes_archived(self, project_manager: ProjectManager):
        project_manager.create("Keep")
        archived = project_manager.create("Gone")
        project_manager.archive(archived.id)
        ids = project_manager.list_ids(include_archived=True)
        assert archived.id in ids

    def test_get_project(self, project_manager: ProjectManager):
        project_manager.create("Gettable")
        info = project_manager.get("gettable")
        assert info is not None
        assert info.name == "Gettable"

    def test_get_nonexistent(self, project_manager: ProjectManager):
        assert project_manager.get("nope") is None

    def test_slug_collision(self, project_manager: ProjectManager):
        a = project_manager.create("Duplicate")
        b = project_manager.create("Duplicate")
        assert a.id != b.id
        assert b.id == "duplicate-2"
        assert Path(a.workspace_path) != Path(b.workspace_path)

    def test_sqlite_persistence_across_instances(self, project_manager: ProjectManager, temp_paths: SynapsePaths):
        info = project_manager.create("Persisted")
        fresh = ProjectManager(temp_paths.data_dir / "projects")
        assert fresh.get(info.id) is not None
        assert fresh.get(info.id).workspace_path == info.workspace_path

    def test_register_existing(self, project_manager: ProjectManager, temp_paths: SynapsePaths):
        base = temp_paths.home / "adopt"
        base.mkdir(parents=True)
        (base / "code.py").write_text("x = 1", encoding="utf-8")
        info = project_manager.register("Adopted", base)
        assert info.workspace_path == str(base)
        assert project_manager.get(info.id).workspace_path == str(base)

    def test_register_missing_folder_raises(self, project_manager: ProjectManager, temp_paths: SynapsePaths):
        with pytest.raises(ValueError):
            project_manager.register("Broken", temp_paths.home / "does-not-exist")

    def test_reconnect(self, project_manager: ProjectManager, temp_paths: SynapsePaths):
        info = project_manager.create("Relocatable", parent_dir=temp_paths.home / "a")
        new_home = temp_paths.home / "b"
        new_home.mkdir(parents=True)
        refreshed = project_manager.reconnect(info.id, new_home)
        assert refreshed is not None
        assert refreshed.workspace_path == str(new_home)
        assert refreshed.exists

    def test_reconnect_missing_raises(self, project_manager: ProjectManager, temp_paths: SynapsePaths):
        info = project_manager.create("Relocatable")
        with pytest.raises(ValueError):
            project_manager.reconnect(info.id, temp_paths.home / "missing")

    def test_reconnect_nonexistent(self, project_manager: ProjectManager, temp_paths: SynapsePaths):
        assert project_manager.reconnect("nope", temp_paths.home) is None

    def test_verify(self, project_manager: ProjectManager, temp_paths: SynapsePaths):
        info = project_manager.create("Verifiable", parent_dir=temp_paths.home / "v")
        assert project_manager.verify()[info.id] is True
        import shutil

        shutil.rmtree(Path(info.workspace_path))
        assert project_manager.verify()[info.id] is False


# ===========================================================================
# 4. ChatStore
# ===========================================================================

class TestChatStore:
    def test_create_chat(self, chat_store: ChatStore):
        info = chat_store.create("Test Chat")
        assert info.title == "Test Chat"
        assert info.message_count == 0

    def test_list_chats(self, chat_store: ChatStore):
        chat_store.create("First")
        chat_store.create("Second")
        chats = chat_store.list()
        assert len(chats) == 2

    def test_get_chat(self, chat_store: ChatStore):
        info = chat_store.create("Gettable")
        retrieved = chat_store.get(info.id)
        assert retrieved is not None
        assert retrieved.title == "Gettable"

    def test_get_nonexistent(self, chat_store: ChatStore):
        assert chat_store.get("nope") is None

    def test_rename_chat(self, chat_store: ChatStore):
        info = chat_store.create("Old Title")
        renamed = chat_store.rename(info.id, "New Title")
        assert renamed is not None
        assert renamed.title == "New Title"

    def test_rename_chat_empty_title(self, chat_store: ChatStore):
        info = chat_store.create("Original")
        renamed = chat_store.rename(info.id, "")
        assert renamed is not None
        assert renamed.title == "Original"

    def test_delete_chat(self, chat_store: ChatStore):
        info = chat_store.create("Deletable")
        assert chat_store.delete(info.id)
        assert chat_store.get(info.id) is None

    def test_delete_nonexistent(self, chat_store: ChatStore):
        assert not chat_store.delete("nope")

    def test_append_message(self, chat_store: ChatStore):
        info = chat_store.create("Chat")
        msg = chat_store.append(info.id, role="user", content="Hello")
        assert msg is not None
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_append_message_persists(self, chat_store: ChatStore):
        info = chat_store.create("Persist")
        chat_store.append(info.id, role="user", content="First")
        chat_store.append(info.id, role="assistant", content="Second")
        messages = chat_store.messages(info.id)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_append_to_nonexistent_chat(self, chat_store: ChatStore):
        result = chat_store.append("nope", role="user", content="Hi")
        assert result is None

    def test_messages_empty(self, chat_store: ChatStore):
        info = chat_store.create("Empty")
        messages = chat_store.messages(info.id)
        assert messages == []

    def test_messages_nonexistent_chat(self, chat_store: ChatStore):
        assert chat_store.messages("nope") == []

    def test_auto_title(self, chat_store: ChatStore):
        info = chat_store.create("New chat")
        chat_store.append(info.id, role="user", content="What is the meaning of life?")
        refreshed = chat_store.get(info.id)
        assert refreshed.title == "What is the meaning of life?"

    def test_auto_title_truncation(self, chat_store: ChatStore):
        info = chat_store.create("New chat")
        long_content = "x" * 100
        chat_store.append(info.id, role="user", content=long_content)
        refreshed = chat_store.get(info.id)
        assert len(refreshed.title) <= 63

    def test_append_with_meta(self, chat_store: ChatStore):
        info = chat_store.create("Meta")
        msg = chat_store.append(
            info.id,
            role="assistant",
            content="Here is the answer",
            meta=["gpt-4", "1.5s"],
            trace={"intent": "qa"},
        )
        assert msg.meta == ["gpt-4", "1.5s"]
        assert msg.trace == {"intent": "qa"}

    def test_append_with_files(self, chat_store: ChatStore):
        info = chat_store.create("Files")
        msg = chat_store.append(
            info.id,
            role="user",
            content="Analyze this",
            files=["file-a", "file-b"],
        )
        assert msg.files == ["file-a", "file-b"]

    def test_chat_store_isolation(self, temp_paths: SynapsePaths):
        p1_dir = temp_paths.chats_dir / "p1"
        p2_dir = temp_paths.chats_dir / "p2"
        store1 = ChatStore("p1", p1_dir)
        store2 = ChatStore("p2", p2_dir)
        info1 = store1.create("Chat in P1")
        store1.append(info1.id, role="user", content="Hello from P1")
        info2 = store2.create("Chat in P2")
        store2.append(info2.id, role="user", content="Hello from P2")
        assert len(store1.messages(info1.id)) == 1
        assert len(store2.messages(info2.id)) == 1
        assert store1.messages(info1.id)[0].content == "Hello from P1"
        assert store2.messages(info2.id)[0].content == "Hello from P2"


# ===========================================================================
# 5. WorkspaceSystem
# ===========================================================================

class TestWorkspaceSystem:
    def test_default_project_created(self, workspace_system: WorkspaceSystem):
        info = workspace_system.get_project(DEFAULT_PROJECT_ID)
        assert info is not None
        assert info.name == DEFAULT_PROJECT_NAME

    def test_create_project(self, workspace_system: WorkspaceSystem):
        info = workspace_system.create_project("AI Assistant")
        assert info.id == "ai-assistant"
        assert info.name == "AI Assistant"

    def test_projects_list(self, workspace_system: WorkspaceSystem):
        workspace_system.create_project("Alpha")
        workspace_system.create_project("Beta")
        projects = workspace_system.projects()
        ids = [p.id for p in projects]
        assert "general" in ids
        assert "alpha" in ids
        assert "beta" in ids

    def test_rename_project(self, workspace_system: WorkspaceSystem):
        info = workspace_system.create_project("Old Name")
        renamed = workspace_system.rename_project(info.id, "New Name")
        assert renamed is not None
        assert renamed.name == "New Name"

    def test_archive_project(self, workspace_system: WorkspaceSystem):
        info = workspace_system.create_project("Archivable")
        archived = workspace_system.archive_project(info.id)
        assert archived.archived

    def test_delete_project_detach(self, workspace_system: WorkspaceSystem):
        info = workspace_system.create_project("Deletable")
        assert workspace_system.delete_project(info.id)
        assert workspace_system.get_project(info.id) is None
        assert Path(info.workspace_path).exists()

    def test_delete_project_erase(self, workspace_system: WorkspaceSystem):
        info = workspace_system.create_project("Deletable")
        assert workspace_system.delete_project(info.id, delete_workspace=True)
        assert not Path(info.workspace_path).exists()

    def test_delete_project_purges_internal_state(self, workspace_system: WorkspaceSystem):
        info = workspace_system.create_project("Purgeable")
        workspace_system.chat_store(info.id).create("Chat")
        workspace_system.action_log(info.id).record({"prompt": "x"})
        workspace_system.delete_project(info.id)
        assert not (workspace_system._paths.chats_dir / info.id).exists()
        assert not (workspace_system._paths.actions_dir / info.id).exists()

    def test_reconnect_project(self, workspace_system: WorkspaceSystem, temp_paths: SynapsePaths):
        info = workspace_system.create_project("Moved")
        new_home = temp_paths.home / "relocated"
        new_home.mkdir(parents=True)
        refreshed = workspace_system.reconnect_project(info.id, str(new_home))
        assert refreshed is not None
        assert refreshed.workspace_path == str(new_home)

    def test_workspace_path(self, workspace_system: WorkspaceSystem):
        info = workspace_system.create_project("Located")
        assert workspace_system.workspace_path(info.id) == Path(info.workspace_path)

    def test_active_project_default(self, workspace_system: WorkspaceSystem):
        session = workspace_system.session()
        assert session["project_id"] == DEFAULT_PROJECT_ID

    def test_switch_project(self, workspace_system: WorkspaceSystem):
        info = workspace_system.create_project("New Project")
        result = workspace_system.switch_project(info.id)
        assert result is not None
        assert workspace_system.session()["project_id"] == info.id

    def test_switch_project_archived(self, workspace_system: WorkspaceSystem):
        info = workspace_system.create_project("Archived")
        workspace_system.archive_project(info.id)
        result = workspace_system.switch_project(info.id)
        assert result is None

    def test_switch_chat(self, workspace_system: WorkspaceSystem):
        chat = workspace_system.chat_store(DEFAULT_PROJECT_ID).create("Test Chat")
        result = workspace_system.switch_chat(DEFAULT_PROJECT_ID, chat.id)
        assert result is not None
        assert workspace_system.session()["chat_id"] == chat.id

    def test_ensure_chat_creates_default(self, workspace_system: WorkspaceSystem):
        chat = workspace_system.ensure_chat(DEFAULT_PROJECT_ID)
        assert chat.title == "General Discussion"

    def test_ensure_chat_returns_existing(self, workspace_system: WorkspaceSystem):
        custom = workspace_system.chat_store(DEFAULT_PROJECT_ID).create("Custom Chat")
        chat = workspace_system.ensure_chat(DEFAULT_PROJECT_ID, custom.id)
        assert chat.id == custom.id

    def test_chats(self, workspace_system: WorkspaceSystem):
        workspace_system.chat_store(DEFAULT_PROJECT_ID).create("A")
        workspace_system.chat_store(DEFAULT_PROJECT_ID).create("B")
        chats = workspace_system.chats(DEFAULT_PROJECT_ID)
        assert len(chats) == 3  # General Discussion + A + B

    def test_chats_stored_internally_not_in_workspace(self, workspace_system: WorkspaceSystem):
        info = workspace_system.create_project("Isolation")
        chat = workspace_system.chat_store(info.id).create("Chat")
        workspace_system.append_message(info.id, chat.id, role="user", content="x")
        workspace = Path(info.workspace_path)
        assert not (workspace / "chats").exists()
        assert not any(workspace.rglob("*.json"))
        assert (workspace_system._paths.chats_dir / info.id).is_dir()

    def test_action_log_stored_internally(self, workspace_system: WorkspaceSystem):
        info = workspace_system.create_project("Audited")
        workspace_system.action_log(info.id).record({"prompt": "hello"})
        assert (workspace_system._paths.actions_dir / info.id / "actions.jsonl").is_file()
        assert not (Path(info.workspace_path) / "logs").exists()

    def test_agent_files_land_in_workspace(self, workspace_system: WorkspaceSystem):
        info = workspace_system.create_project("Agent Files")
        op = workspace_system.file_operator(info.id)
        op.write("hello.py", "print('hi')")
        assert (Path(info.workspace_path) / "hello.py").is_file()
        assert op.read("hello.py") == "print('hi')"

    def test_per_project_workspace_isolation(self, workspace_system: WorkspaceSystem):
        ws_a = workspace_system.workspace_for("general")
        ws_b = workspace_system.workspace_for("general")
        assert ws_a is ws_b
        info = workspace_system.create_project("Other")
        ws_c = workspace_system.workspace_for(info.id)
        assert ws_c is not ws_a

    def test_per_project_memory_isolation(self, workspace_system: WorkspaceSystem):
        mem_a = workspace_system.memory_for("general")
        mem_b = workspace_system.memory_for("general")
        assert mem_a is mem_b
        info = workspace_system.create_project("Other")
        mem_c = workspace_system.memory_for(info.id)
        assert mem_c is not mem_a

    def test_memory_stored_internally(self, workspace_system: WorkspaceSystem):
        info = workspace_system.create_project("Memo")
        mem = workspace_system.memory_for(info.id)
        assert mem._root == workspace_system._paths.memory_dir / info.id
        assert not (Path(info.workspace_path) / "memory").exists()

    def test_session_persistence(self, workspace_system: WorkspaceSystem):
        info = workspace_system.create_project("Session Test")
        workspace_system.switch_project(info.id)
        chat = workspace_system.chat_store(info.id).create("Persisted Chat")
        workspace_system.switch_chat(info.id, chat.id)
        session_path = workspace_system._session_path
        data = json.loads(session_path.read_text(encoding="utf-8"))
        assert data["project_id"] == info.id
        assert data["chat_id"] == chat.id

    def test_session_recovery(self, workspace_system: WorkspaceSystem, temp_paths: SynapsePaths):
        info = workspace_system.create_project("Recovery Test")
        chat = workspace_system.chat_store(info.id).create("Recovery Chat")
        workspace_system.switch_project(info.id)
        workspace_system.switch_chat(info.id, chat.id)
        system2 = WorkspaceSystem(
            root=workspace_system._root,
            paths=temp_paths,
            config=workspace_system._config,
            providers=workspace_system._providers,
            registry=workspace_system._registry,
            events=workspace_system._events,
            lifecycle=workspace_system._lifecycle,
        )
        assert system2.session()["project_id"] == info.id
        assert system2.session()["chat_id"] == chat.id

    def test_session_recovery_missing_project(self, workspace_system: WorkspaceSystem, temp_paths: SynapsePaths):
        session_data = {"project_id": "deleted-project", "chat_id": "c1", "updated_at": "2026-01-01T00:00:00Z"}
        workspace_system._session_path.write_text(json.dumps(session_data), encoding="utf-8")
        system2 = WorkspaceSystem(
            root=workspace_system._root,
            paths=temp_paths,
            config=workspace_system._config,
            providers=workspace_system._providers,
            registry=workspace_system._registry,
            events=workspace_system._events,
            lifecycle=workspace_system._lifecycle,
        )
        assert system2.session()["project_id"] == DEFAULT_PROJECT_ID


# ===========================================================================
# 5b. Cleanup — no orphaned temp folders / stale internal state
# ===========================================================================

class TestCleanup:
    def test_temp_dir_cleared(self, workspace_system: WorkspaceSystem):
        (workspace_system._paths.temp_dir / "scratch").mkdir(parents=True)
        (workspace_system._paths.temp_dir / "scratch" / "x.tmp").write_text("x", encoding="utf-8")
        workspace_system.cleanup()
        assert not (workspace_system._paths.temp_dir / "scratch").exists()

    def test_orphaned_internal_state_removed(self, workspace_system: WorkspaceSystem):
        ghost = workspace_system._paths.chats_dir / "ghost-project"
        (ghost / "chat.json").parent.mkdir(parents=True)
        (ghost / "chat.json").write_text("{}", encoding="utf-8")
        workspace_system.cleanup()
        assert not ghost.exists()

    def test_registered_project_state_survives_cleanup(self, workspace_system: WorkspaceSystem):
        info = workspace_system.create_project("Kept")
        workspace_system.chat_store(info.id).create("Chat")
        workspace_system.cleanup()
        assert (workspace_system._paths.chats_dir / info.id).is_dir()

    def test_stale_empty_workspace_removed(self, workspace_system: WorkspaceSystem):
        stale = workspace_system.manager._workspaces_root / "abandoned"
        stale.mkdir(parents=True)
        (stale / "README.md").write_text("# abandoned", encoding="utf-8")
        workspace_system.cleanup()
        assert not stale.exists()

    def test_stale_workspace_with_files_kept(self, workspace_system: WorkspaceSystem):
        stale = workspace_system.manager._workspaces_root / "abandoned-2"
        stale.mkdir(parents=True)
        (stale / "real-file.txt").write_text("user data", encoding="utf-8")
        workspace_system.cleanup()
        assert stale.exists()

    def test_workspace_missing_reported_as_exists_false(self, workspace_system: WorkspaceSystem):
        info = workspace_system.create_project("Vanishing")
        import shutil

        shutil.rmtree(info.workspace_path)
        refreshed = workspace_system.get_project(info.id)
        assert refreshed is not None
        assert refreshed.exists is False


# ===========================================================================
# 6. API endpoints — projects
# ===========================================================================

class TestAPIProjects:
    def test_list_projects(self, client):
        resp = client.get("/projects")
        assert resp.status_code == 200
        projects = resp.json()
        assert any(p["id"] == "general" for p in projects)

    def test_create_project(self, client):
        resp = client.post("/projects", json={"name": "API Project"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "api-project"
        assert body["name"] == "API Project"
        assert "workspace_path" in body
        assert body["exists"] is True

    def test_create_project_empty_name(self, client):
        resp = client.post("/projects", json={"name": ""})
        assert resp.status_code == 400

    def test_create_project_with_parent_dir(self, client, temp_paths: SynapsePaths):
        base = temp_paths.home / "api-parent"
        resp = client.post("/projects", json={"name": "Parented", "parent_dir": str(base)})
        assert resp.status_code == 200
        body = resp.json()
        assert body["workspace_path"] == str(base / "Parented")
        assert (base / "Parented" / "README.md").is_file()

    def test_create_project_location_only_derives_name(self, client, temp_paths: SynapsePaths):
        # creating a project asks only *where*: the name comes from the folder
        base = temp_paths.home / "api-parent"
        folder = base / "my cool folder"
        folder.mkdir(parents=True)
        resp = client.post("/projects", json={"parent_dir": str(base), "name": ""})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["name"] == "api-parent"  # derived from the chosen folder's parent
        assert body["workspace_path"] == str(base / "api-parent")

    def test_create_project_existing_location_only_derives_name(self, client, temp_paths: SynapsePaths):
        base = temp_paths.home / "adopt-me-2"
        base.mkdir(parents=True)
        (base / "code.py").write_text("x=1", encoding="utf-8")
        resp = client.post("/projects", json={"workspace_path": str(base)})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["name"] == "adopt-me-2"
        assert body["workspace_path"] == str(base)

    def test_create_project_adopting_existing(self, client, temp_paths: SynapsePaths):
        base = temp_paths.home / "adopt-me"
        base.mkdir(parents=True)
        (base / "code.py").write_text("x=1", encoding="utf-8")
        resp = client.post("/projects", json={"name": "Adopted", "workspace_path": str(base)})
        assert resp.status_code == 200
        assert resp.json()["workspace_path"] == str(base)

    def test_create_project_adopting_missing_422(self, client, temp_paths: SynapsePaths):
        resp = client.post(
            "/projects",
            json={"name": "Broken", "workspace_path": str(temp_paths.home / "nope")},
        )
        assert resp.status_code == 422

    def test_pick_endpoint(self, client, monkeypatch, temp_paths: SynapsePaths):
        target = temp_paths.home / "picked"
        target.mkdir(parents=True)
        monkeypatch.setattr("synapse.projects.picker.pick_folder", lambda initial_dir=None: str(target))
        resp = client.post("/projects/pick")
        assert resp.status_code == 200
        assert resp.json() == {"path": str(target)}

    def test_pick_endpoint_unavailable(self, client, monkeypatch):
        monkeypatch.setattr("synapse.projects.picker.pick_folder", lambda initial_dir=None: None)
        resp = client.post("/projects/pick")
        assert resp.status_code == 503

    def test_reconnect_endpoint(self, client, temp_paths: SynapsePaths):
        pid = client.post("/projects", json={"name": "Relocate"}).json()["id"]
        new_home = temp_paths.home / "relocated"
        new_home.mkdir(parents=True)
        resp = client.post(f"/projects/{pid}/reconnect", json={"workspace_path": str(new_home)})
        assert resp.status_code == 200
        assert resp.json()["workspace_path"] == str(new_home)
        assert resp.json()["exists"] is True

    def test_reconnect_missing_422(self, client, temp_paths: SynapsePaths):
        pid = client.post("/projects", json={"name": "Relocate"}).json()["id"]
        resp = client.post(
            f"/projects/{pid}/reconnect",
            json={"workspace_path": str(temp_paths.home / "missing")},
        )
        assert resp.status_code == 422

    def test_reconnect_nonexistent(self, client, temp_paths: SynapsePaths):
        resp = client.post("/projects/nope/reconnect", json={"workspace_path": str(temp_paths.home)})
        assert resp.status_code == 404

    def test_verify_endpoint(self, client):
        pid = client.post("/projects", json={"name": "Verifiable"}).json()["id"]
        resp = client.post(f"/projects/{pid}/verify")
        assert resp.status_code == 200
        assert resp.json()["exists"] is True

    def test_rename_project(self, client):
        create = client.post("/projects", json={"name": "Rename Me"})
        pid = create.json()["id"]
        resp = client.patch(f"/projects/{pid}", json={"name": "Renamed"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Renamed"

    def test_rename_nonexistent(self, client):
        resp = client.patch("/projects/nope", json={"name": "X"})
        assert resp.status_code == 404

    def test_archive_project(self, client):
        create = client.post("/projects", json={"name": "Archive Me"})
        pid = create.json()["id"]
        resp = client.post(f"/projects/{pid}/archive")
        assert resp.status_code == 200
        assert resp.json()["archived"] is True

    def test_unarchive_project(self, client):
        pid = client.post("/projects", json={"name": "Restore"}).json()["id"]
        client.post(f"/projects/{pid}/archive")
        resp = client.post(f"/projects/{pid}/unarchive")
        assert resp.status_code == 200
        assert resp.json()["archived"] is False

    def test_delete_project_detach(self, client):
        create = client.post("/projects", json={"name": "Delete Me"})
        pid = create.json()["id"]
        resp = client.delete(f"/projects/{pid}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == pid
        assert resp.json()["files_deleted"] is False

    def test_delete_project_erase_workspace(self, client):
        create = client.post("/projects", json={"name": "Erase Me"})
        pid = create.json()["id"]
        workspace = Path(create.json()["workspace_path"])
        resp = client.delete(f"/projects/{pid}", params={"delete_workspace": "true"})
        assert resp.status_code == 200
        assert resp.json()["files_deleted"] is True
        assert not workspace.exists()

    def test_delete_nonexistent(self, client):
        resp = client.delete("/projects/nope")
        assert resp.status_code == 404

    def test_archive_nonexistent(self, client):
        resp = client.post("/projects/nope/archive")
        assert resp.status_code == 404


# ===========================================================================
# 7. API endpoints — chats
# ===========================================================================

class TestAPIChats:
    def test_list_chats(self, client):
        resp = client.get("/projects/general/chats")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_create_chat(self, client):
        resp = client.post("/projects/general/chats", json={"title": "API Chat"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["title"] == "API Chat"

    def test_rename_chat(self, client):
        create = client.post("/projects/general/chats", json={"title": "Old"})
        cid = create.json()["id"]
        resp = client.patch(f"/projects/general/chats/{cid}", json={"title": "New"})
        assert resp.status_code == 200
        assert resp.json()["title"] == "New"

    def test_delete_chat(self, client):
        create = client.post("/projects/general/chats", json={"title": "Delete Me"})
        cid = create.json()["id"]
        resp = client.delete(f"/projects/general/chats/{cid}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == cid

    def test_get_messages(self, client):
        create = client.post("/projects/general/chats", json={"title": "Msg Chat"})
        cid = create.json()["id"]
        resp = client.get(f"/projects/general/chats/{cid}/messages")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) == 0

    def test_rename_chat_nonexistent(self, client):
        resp = client.patch("/projects/general/chats/nope", json={"title": "X"})
        assert resp.status_code == 404

    def test_delete_chat_nonexistent(self, client):
        resp = client.delete("/projects/general/chats/nope")
        assert resp.status_code == 404


# ===========================================================================
# 8. API endpoints — session
# ===========================================================================

class TestAPISession:
    def test_get_session(self, client):
        resp = client.get("/session")
        assert resp.status_code == 200
        body = resp.json()
        assert body["project_id"] == "general"
        assert body["chat_id"]

    def test_set_session(self, client):
        resp = client.post("/session", json={"project_id": "general"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["project_id"] == "general"


# ===========================================================================
# 9. API endpoints — scoped files/workspace
# ===========================================================================

class TestAPIScopedWorkspace:
    def test_upload_file_scoped(self, client):
        resp = client.post(
            "/files",
            files={"file": ("doc.md", b"# Content")},
            params={"project_id": "general"},
        )
        assert resp.status_code == 200
        assert resp.json()["file"]

    def test_list_files_scoped(self, client):
        resp = client.get("/files", params={"project_id": "general"})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_workspace_status_scoped(self, client):
        resp = client.get("/workspace", params={"project_id": "general"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is True

    def test_upload_file_legacy(self, client):
        resp = client.post("/files", files={"file": ("legacy.txt", b"Legacy content")})
        assert resp.status_code == 200

    def test_list_files_legacy(self, client):
        resp = client.get("/files")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
