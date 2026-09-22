"""Intent Router tests — chat and tool execution are completely separate.

The AI Operating Workspace routes every prompt to exactly one intent BEFORE
any planning. Conversation and question answering are answered directly by the
selected language model and must NEVER enter the artifact pipeline (no
folders, no manifests, no files, no JSON). Workspace changes require an
explicit work intent. These are permanent regression guards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synapse.actions.intent_router import IntentRouter  # noqa: E402
from synapse.domain.enums import IntentKind  # noqa: E402

_CHAT_ONLY = (IntentKind.CONVERSATION, IntentKind.QUESTION_ANSWERING)


def route(prompt: str) -> IntentKind:
    return IntentRouter().route(prompt)


def _expect(prompt: str, expected: IntentKind) -> None:
    got = route(prompt)
    assert got is expected, f"{prompt!r} -> {got.value} (expected {expected.value})"


class TestConversationIntent:
    @pytest.mark.parametrize(
        "prompt",
        [
            "Hello",
            "Hi there",
            "hey",
            "good morning",
            "nice to meet you",
            "thanks!",
            "lol",
        ],
    )
    def test_greetings_are_chat_only(self, prompt):
        _expect(prompt, IntentKind.CONVERSATION)
        assert route(prompt).is_chat_only is True


class TestQuestionAnsweringIntent:
    @pytest.mark.parametrize(
        "prompt",
        [
            "What is the capital of France?",
            "How do I boil an egg?",
            "who invented the telephone?",
            "When was the railroad invented?",
            "Explain how photosynthesis works",
            "Define recursion.",
            "who are you",
            "what can you do",
        ],
    )
    def test_factual_questions_are_chat_only(self, prompt):
        _expect(prompt, IntentKind.QUESTION_ANSWERING)
        assert route(prompt).is_chat_only is True


class TestExplicitWorkIntent:
    @pytest.mark.parametrize(
        "prompt",
        [
            "list files",
            "show the file tree",
            "read the README.md file",
            "rename main.txt to index.txt",
            "delete stale.txt",
            "create folder assets",
        ],
    )
    def test_workspace_management(self, prompt):
        _expect(prompt, IntentKind.WORKSPACE_MANAGEMENT)
        assert route(prompt).is_chat_only is False

    @pytest.mark.parametrize(
        "prompt",
        [
            "run the tests",
            "execute the test suite",
            "automate my backup",
            "launch the server",
        ],
    )
    def test_tool_execution(self, prompt):
        _expect(prompt, IntentKind.TOOL_EXECUTION)
        assert route(prompt).is_chat_only is False

    @pytest.mark.parametrize(
        "prompt",
        [
            "set a goal to learn python",
            "my goal is to finish the portfolio site",
            "make a roadmap to complete the app",
        ],
    )
    def test_goal_planning(self, prompt):
        _expect(prompt, IntentKind.GOAL_PLANNING)
        assert route(prompt).is_chat_only is False

    @pytest.mark.parametrize(
        "prompt",
        [
            "create a website",
            "build a python script that parses csv",
            "generate a dashboard app",
            "write a python function",
            "make a todo list",
            "scaffold a landing page",
        ],
    )
    def test_file_generation(self, prompt):
        _expect(prompt, IntentKind.FILE_GENERATION)
        assert route(prompt).is_chat_only is False

    @pytest.mark.parametrize(
        "prompt",
        [
            "edit the index.html file",
            "fix the bug in main.py",
            "modify the app.js file",
            "refactor the style.css",
            "update the README.md",
        ],
    )
    def test_file_editing(self, prompt):
        _expect(prompt, IntentKind.FILE_EDITING)
        assert route(prompt).is_chat_only is False


class TestRouterHardGuarantees:
    def test_only_conversation_and_qa_are_chat_only(self):
        for kind in IntentKind:
            expected = kind in (IntentKind.CONVERSATION, IntentKind.QUESTION_ANSWERING)
            assert kind.is_chat_only is expected, kind.value

    def test_chat_never_leaks_workspace_ops(self):
        ops = IntentRouter().route_with_ops("Hello")[1]
        assert ops == []

    def test_hello_is_never_a_file_intent(self):
        kinds = {
            IntentKind.FILE_GENERATION,
            IntentKind.FILE_EDITING,
            IntentKind.WORKSPACE_MANAGEMENT,
            IntentKind.TOOL_EXECUTION,
            IntentKind.GOAL_PLANNING,
        }
        assert route("Hello") not in kinds


class TestWorkspaceOpGuards:
    """Prose verbs inside bigger requests must never become backend ops."""

    def test_workspace_op_words_inside_prose_are_not_ops(self):
        # "Search and filter expenses" is a product feature sentence, not a
        # request to grep the workspace for the word "and".
        assert route("search and filter expenses") is IntentKind.CONVERSATION
        # "Delete expenses" has no real file target → not a delete operation.
        assert route("delete expenses") is IntentKind.CONVERSATION
        # Stopword-only search terms never trigger the Action Engine.
        assert route("find all files") is IntentKind.CONVERSATION
        assert IntentRouter().route_with_ops("search and filter expenses")[1] == []

    def test_real_backend_ops_still_route(self):
        assert route("search 'budget'") is IntentKind.WORKSPACE_MANAGEMENT
        assert route("delete stale.txt") is IntentKind.WORKSPACE_MANAGEMENT
        assert route("list files") is IntentKind.WORKSPACE_MANAGEMENT
        assert route("rename main.txt to index.txt") is IntentKind.WORKSPACE_MANAGEMENT
        assert route("create folder assets") is IntentKind.WORKSPACE_MANAGEMENT


LEDGER_PROMPT = (
    'Build a complete desktop expense tracker application called "LedgerLite". '
    "Requirements: Create the project from scratch with a clean folder structure. "
    "Use Python for the backend/business logic and a modern desktop UI. "
    "Users must be able to: Add, edit, and delete expenses. Categorize expenses "
    "(Food, Transport, Shopping, Bills, Other). Set a monthly budget. "
    "View total spending for the current month. See spending broken down by "
    "category. Search and filter expenses. Export expenses to CSV. "
    "Store data locally so the application works completely offline. "
    "Include input validation and graceful error handling. Create a README "
    "explaining setup, architecture, and usage. Create automated tests for the "
    "core expense calculations and data operations. Keep the code modular and "
    "maintainable."
)


class TestLedgerLiteRegression:
    """A full application-build request is FILE_GENERATION — never chat, never
    a workspace operation — even though its prose mentions search/delete/
    edit and 'expense tracker'."""

    def test_full_app_build_prompt_is_file_generation(self):
        assert route(LEDGER_PROMPT) is IntentKind.FILE_GENERATION

    def test_full_app_build_prompt_extracts_no_workspace_ops(self):
        ops = IntentRouter().route_with_ops(LEDGER_PROMPT)[1]
        assert ops == [], ops

    def test_app_build_is_never_chat_only(self):
        assert route(LEDGER_PROMPT).is_chat_only is False