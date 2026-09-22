"""Phase XVII — follow-up context resolution.

Conversational wording with workspace intent must reach the planner and the
tools instead of the chat path. "fix the bugs", "continue", "change the UI",
"add search", "why isn't this working?" and "run the tests again" carry no
explicit artifact, yet inside an active project they continue the work: the
Master must resolve them against the actual workspace state (prior
conversation turns, prior recorded actions, existing files) and route them
through the planner and file tools — never answer them from chat alone.
Greetings and general questions stay pure chat even when a project is open.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from synapse.actions.followup import (  # noqa: E402
    has_workspace_context,
    is_follow_up,
)
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


import re  # noqa: E402

_FOLLOW_KEYS = (
    "review the generated files",
    "fix the bugs",
    "change the ui",
    "add search",
    "continue",
    "why isn",
    "python script that prints hello",
    "hello",
)


def _contains_follow_key(lowered: str, key: str) -> bool:
    return re.search(rf"\b{re.escape(key)}", lowered) is not None


class FakeProvider(ModelProvider):
    """Two-capability provider (coder / reasoner) recording every prompt.

    Follow-up keys are matched BEFORE the generic keys because the task
    prompt always repeats the brief (which repeats the conversation), so
    turn-2 prompts contain both the follow-up phrase and old fragments.
    Keys are matched at word boundaries so project/test names or paths
    that merely contain a keyword (e.g. "continue" inside
    "test_continue_after_build0") never hijack the reply.
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
        for _key in _FOLLOW_KEYS:
            if _contains_follow_key(lowered, _key):
                if _key == "review the generated files":
                    reply = "All generated files are consistent and well-formed."
                elif _key == "fix the bugs":
                    reply = (
                        '{"files": [{"path": "hello.py", "content": "print(\'fixed\')\\n"}, '
                        '{"path": "app.py", "content": "print(\'fixed\')\\n"}]}'
                    )
                elif _key == "change the ui":
                    reply = '{"files": [{"path": "style.css", "content": "body{color:red}\\n"}]}'
                elif _key == "add search":
                    reply = (
                        '{"files": [{"path": "search.py", '
                        '"content": "def search():\\n    return []\\n"}]}'
                    )
                elif _key == "continue":
                    reply = '{"files": [{"path": "hello.py", "content": "print(\'continued\')\\n"}]}'
                elif _key == "why isn":
                    reply = "The script fails because hello.py calls undefined main()."
                elif _key == "python script that prints hello":
                    reply = '{"files": [{"path": "hello.py", "content": "print(\'hello\')\\n"}]}'
                else:
                    reply = "Hello! How can I help you today?"
                break
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


def _turn_prompts(fake: FakeProvider, before: int) -> list[str]:
    return fake.prompts[before:]


# -- deterministic wording + state probes -----------------------------------


class TestFollowUpWording:
    @pytest.mark.parametrize(
        "prompt",
        [
            "fix the bugs",
            "continue",
            "keep going",
            "proceed",
            "do it",
            "redo it",
            "change the UI",
            "add search",
            "remove the button",
            "make it better",
            "improve it",
            "why isn't this working?",
            "what's wrong with it?",
            "the code doesn't work",
            "the tests are failing",
            "run the tests again",
            "it's broken",
            "the app crashes",
            "this is failing",
            "the UI",
            "same",
            "it",
            "this",
        ],
    )
    def test_follow_up_wording_detected(self, prompt: str):
        assert is_follow_up(prompt)

    @pytest.mark.parametrize(
        "prompt",
        [
            "Hello",
            "hi there",
            "good morning",
            "nice to meet you",
            "thanks!",
            "thank you",
            "lol",
            "haha",
            "ok",
            "yes",
            "sure",
            "who are you?",
            "what can you do?",
            "what is the capital of France?",
            "why is the sky blue?",
            "tell me a joke",
            "no problem",
            "no issue",
            "nothing wrong",
            "add 2 and 3",
            "add two numbers",
            "remove and replace",
        ],
    )
    def test_non_follow_up_wording_stays_chat(self, prompt: str):
        assert not is_follow_up(prompt)


class TestWorkspaceContextProbe:
    def test_no_context_signals(self):
        assert not has_workspace_context()

    def test_prior_conversation_turns(self):
        memory = SimpleNamespace(recent=lambda scope, limit, conversation: [object()])
        assert has_workspace_context(memory=memory, conversation_id="c1")

    def test_no_prior_turns_in_this_chat(self):
        memory = SimpleNamespace(recent=lambda scope, limit, conversation: [])
        assert not has_workspace_context(memory=memory, conversation_id="c1")

    def test_prior_workspace_actions(self):
        action_log = SimpleNamespace(recent=lambda limit: [{"project_id": "p"}])
        assert has_workspace_context(action_log=action_log)

    def test_existing_project_files(self):
        file_operator = SimpleNamespace(list_tree=lambda: [{"path": "a.py", "size": 3}])
        assert has_workspace_context(file_operator=file_operator)

    def test_empty_workspace_is_no_context(self):
        file_operator = SimpleNamespace(list_tree=lambda: [])
        assert not has_workspace_context(file_operator=file_operator)

    def test_failing_probes_are_tolerated(self):
        memory = SimpleNamespace(
            recent=lambda scope, limit, conversation: (_ for _ in ()).throw(
                RuntimeError("boom")
            )
        )
        assert not has_workspace_context(memory=memory, conversation_id="c1")


# -- multi-turn workflows: follow-ups reach planner + tools -----------------


class TestFollowUpsReachThePipeline:
    def test_fix_the_bugs_after_build(self, app_env):
        client, fake, temp_paths = app_env
        pid, folder = _adopt(client, temp_paths, "FollowUpFix")
        assert client.post(
            "/request", json={"prompt": "build a python script that prints hello", "project_id": pid}
        ).status_code == 200
        assert (folder / "hello.py").read_text(encoding="utf-8") == "print('hello')\n"
        before = len(fake.prompts)

        resp = client.post(
            "/request", json={"prompt": "fix the bugs", "project_id": pid}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        prompts = _turn_prompts(fake, before)
        # workspace pipeline, not chat: the model was routed per-task and
        # saw the real file contents before answering.
        assert prompts, "follow-up must call the model"
        assert any("fix the bugs" in p for p in prompts)
        assert any("print('hello')" in p for p in prompts), "file context not read"
        nodes = {n["task_id"]: n for n in body["execution_graph"]["nodes"]}
        assert nodes["t1"]["model_id"] == "coder", nodes
        ok = [a for a in body["actions"] if a["status"] == "ok"]
        assert any(a["path"] == "hello.py" for a in ok), body["actions"]
        assert (folder / "hello.py").read_text(encoding="utf-8") == "print('fixed')\n"
        assert "hello.py" in body["response"]

    def test_continue_after_build(self, app_env):
        client, fake, temp_paths = app_env
        pid, folder = _adopt(client, temp_paths, "FollowUpFlow")
        assert client.post(
            "/request", json={"prompt": "build a python script that prints hello", "project_id": pid}
        ).status_code == 200
        before = len(fake.prompts)

        resp = client.post("/request", json={"prompt": "continue", "project_id": pid})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        prompts = _turn_prompts(fake, before)
        assert any("continue" in p for p in prompts)
        assert any("print('hello')" in p for p in prompts), "file context not read"
        ok = [a for a in body["actions"] if a["status"] == "ok"]
        assert any(a["path"] == "hello.py" for a in ok), body["actions"]
        assert (folder / "hello.py").read_text(encoding="utf-8") == "print('continued')\n"

    def test_change_the_ui_after_build(self, app_env):
        client, fake, temp_paths = app_env
        pid, folder = _adopt(client, temp_paths, "FollowUpUI")
        assert client.post(
            "/request", json={"prompt": "build a python script that prints hello", "project_id": pid}
        ).status_code == 200
        before = len(fake.prompts)

        resp = client.post("/request", json={"prompt": "change the UI", "project_id": pid})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        ok = [a for a in body["actions"] if a["status"] == "ok"]
        assert any(a["path"] == "style.css" for a in ok), body["actions"]
        assert (folder / "style.css").is_file()
        assert "style.css" in body["response"]

    def test_add_search_after_build(self, app_env):
        client, fake, temp_paths = app_env
        pid, folder = _adopt(client, temp_paths, "FollowUpSearch")
        assert client.post(
            "/request", json={"prompt": "build a python script that prints hello", "project_id": pid}
        ).status_code == 200
        before = len(fake.prompts)

        resp = client.post("/request", json={"prompt": "add search", "project_id": pid})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        ok = [a for a in body["actions"] if a["status"] == "ok"]
        assert any(a["path"] == "search.py" for a in ok), body["actions"]
        assert (folder / "search.py").is_file()

    def test_why_isnt_this_working_after_build(self, app_env):
        client, fake, temp_paths = app_env
        pid, folder = _adopt(client, temp_paths, "FollowUpWhy")
        assert client.post(
            "/request", json={"prompt": "build a python script that prints hello", "project_id": pid}
        ).status_code == 200
        before = len(fake.prompts)

        resp = client.post(
            "/request", json={"prompt": "why isn't this working?", "project_id": pid}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        prompts = _turn_prompts(fake, before)
        # a diagnosis request still inspects the workspace first ...
        assert any("print('hello')" in p for p in prompts), "file context not read"
        # ... and the diagnosis answer is preserved (no files invented).
        assert "undefined main()" in body["response"]
        assert not body["actions"]

    def test_run_the_tests_again_after_build(self, app_env):
        client, fake, temp_paths = app_env
        pid, folder = _adopt(client, temp_paths, "FollowUpTests")
        assert client.post(
            "/request", json={"prompt": "build a python script that prints hello", "project_id": pid}
        ).status_code == 200
        before = len(fake.prompts)

        resp = client.post(
            "/request", json={"prompt": "run the tests again", "project_id": pid}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        prompts = _turn_prompts(fake, before)
        assert body["execution_graph"]["nodes"], "must go through the task pipeline"
        assert any("run the tests again" in p for p in prompts)
        assert any("print('hello')" in p for p in prompts), "file context not read"
        assert "Hello! How can I help" not in body["response"]

    def test_first_message_fix_with_existing_files(self, app_env):
        """Workspace state alone (files, no prior turns) resolves a follow-up."""
        client, fake, temp_paths = app_env
        pid, folder = _adopt(client, temp_paths, "FollowUpFresh")
        (folder / "app.py").write_text("print('buggy')\n", encoding="utf-8")
        client.post(f"/projects/{pid}/scan")

        resp = client.post("/request", json={"prompt": "fix the bugs", "project_id": pid})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        prompts = _turn_prompts(fake, 0)
        assert any("print('buggy')" in p for p in prompts), "file context not read"
        ok = [a for a in body["actions"] if a["status"] == "ok"]
        assert any(a["path"] == "app.py" for a in ok), body["actions"]
        assert (folder / "app.py").read_text(encoding="utf-8") == "print('fixed')\n"

    def test_hello_after_build_stays_chat(self, app_env):
        """A greeting in the middle of a project conversation stays pure chat."""
        client, fake, temp_paths = app_env
        pid, folder = _adopt(client, temp_paths, "FollowUpChat")
        assert client.post(
            "/request", json={"prompt": "build a python script that prints hello", "project_id": pid}
        ).status_code == 200
        before = len(fake.prompts)

        resp = client.post("/request", json={"prompt": "Hello", "project_id": pid})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["actions"] == []
        assert body["intent"] in ("conversation", "question_answering")
        assert len(fake.prompts) == before + 1, "chat must make exactly one model call"
        assert "Hello! How can I help" in body["response"]


# -- Phase XVIII: editing wording WITHOUT workspace context is conversation --


class TestEditingWordingWithoutContext:
    """Mirror rule of follow-ups: artifact wording with no workspace state and
    no explicit file target ("HI fix the issue") has nothing to edit — it must
    be answered as conversation, never pinned to a coding specialist."""

    def test_hi_fix_the_issue_without_project_stays_chat(self, app_env):
        client, fake, temp_paths = app_env
        before = len(fake.prompts)

        resp = client.post("/request", json={"prompt": "HI fix the issue"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["intent"] == "conversation", body["intent"]
        assert body["actions"] == []
        assert not body["execution_graph"]["nodes"], "chat must not enter the task pipeline"
        assert len(fake.prompts) == before + 1, "chat must make exactly one model call"
        assert "HI fix the issue" in fake.prompts[-1]

    def test_fix_the_bugs_without_project_stays_chat(self, app_env):
        """Follow-up wording as a FIRST message (no context, no target) is chat."""
        client, fake, temp_paths = app_env
        before = len(fake.prompts)

        resp = client.post("/request", json={"prompt": "fix the bugs"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["intent"] == "conversation", body["intent"]
        assert body["actions"] == []
        assert not body["execution_graph"]["nodes"]
        assert len(fake.prompts) == before + 1, "chat must make exactly one model call"
        assert "fix the bugs" in fake.prompts[-1]

    def test_explicit_file_target_without_context_stays_pipeline(self, app_env):
        """A real file operand ("main.py") keeps artifact wording in the pipeline."""
        client, fake, temp_paths = app_env

        resp = client.post("/request", json={"prompt": "fix the issue in main.py"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["intent"] != "conversation", body["intent"]
        assert body["execution_graph"]["nodes"], "explicit target must enter the task pipeline"
        assert fake.prompts, "pipeline must call the model"
        assert "main.py" in fake.prompts[-1]

    def test_hi_fix_the_issue_inside_project_stays_pipeline(self, app_env):
        """The same wording INSIDE an active project targets the workspace."""
        client, fake, temp_paths = app_env
        pid, folder = _adopt(client, temp_paths, "FollowUpPhase18")
        (folder / "app.py").write_text("print('buggy')\n", encoding="utf-8")
        client.post(f"/projects/{pid}/scan")

        resp = client.post("/request", json={"prompt": "HI fix the issue", "project_id": pid})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["execution_graph"]["nodes"], "in-project editing must enter the task pipeline"
        assert fake.prompts, "pipeline must call the model"
        assert "HI fix the issue" in fake.prompts[-1]
