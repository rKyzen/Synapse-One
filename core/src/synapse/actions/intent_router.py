"""Intent Router — deterministic intent gate before any planning.

The AI Operating Workspace keeps conversation and tool execution completely
separate. This router decides, BEFORE the workspace planner, whether a request
may enter the artifact pipeline at all:

- ``CONVERSATION`` / ``QUESTION_ANSWERING`` never do: they are answered
  directly by the selected language model — no planner, no brief, no workspace
  reads, no folders, no manifests, no JSON.
- ``WORKSPACE_MANAGEMENT`` is served by the backend alone (no model).
- ``TOOL_EXECUTION`` / ``GOAL_PLANNING`` route to their own surfaces.
- ``FILE_GENERATION`` / ``FILE_EDITING`` are the ONLY intents allowed into the
  artifact pipeline, and they require explicit work verbs targeting a file.

Rules are priority-ordered and offline: explicit work first (so "how do I
create a website?" is a file request, not a question), question patterns
second, and plain conversation is the fallback. Nothing here is a model call.
"""

from __future__ import annotations

import re

from synapse.actions.classifier import extract_workspace_ops
from synapse.domain.enums import IntentKind

#: artifact targets that make a work verb explicit file work.
_ARTIFACT_RE = re.compile(
    r"\b(website|web\s*app|web\s*page|site|portfolio|app|application|project|dashboard|"
    r"game|cli|tool|plugin|extension|module|package|api|landing\s*page|store|shop|"
    r"ecommerce|blog|forum|cms|chatbot|bot|calculator|todo\s*list|todo\s*app|wiki|timer|"
    r"catalog|script|program|function|class|method|unit\s*test|test|file|endpoint|"
    r"sql\s*query|python|javascript|typescript|html|css|bash|shell|go|rust|java|c\+\+|"
    r"ruby|php|config|readme|documentation|docs|doc\b|markdown|manual|guide|changelog|"
    r"wiki|tutorial|code|bug|error|issue|tracker|system|software|database|monitor|"
    r"server|library|component|interface|pipeline|automation|frontend|backend|"
    r"web\s*service|[a-z0-9_.\-/]+\.(py|js|ts|html|css|json|md|sh|toml|yaml|yml|sql))\b",
    re.IGNORECASE,
)

#: explicit file-creation verbs.
_CREATE_VERBS = r"(?:create|build|make|generate|scaffold|design|develop|write|compose)"

#: explicit file-editing verbs.
_EDIT_VERBS = (
    r"(?:edit|modify|update|change|rewrite|refactor|improve|adjust|tweak|patch|"
    r"redesign|restyle|fix|debug)"
)

#: priority-ordered (name, matcher) rules; first match wins.
_RULES: list[tuple[IntentKind, re.Pattern[str]]] = [
    # -- tool execution: explicit run/execute/automate ----------------------
    (
        IntentKind.TOOL_EXECUTION,
        re.compile(
            r"\b(run|execute|automate|launch)\s+(the\s+)?"
            r"(command|script|tests|test|test\s*suite|build|terminal|task|job|pipeline|"
            r"workflow|setup|server|install|migration)\b",
            re.IGNORECASE,
        ),
    ),
    (
        IntentKind.TOOL_EXECUTION,
        re.compile(r"\bautomate\b", re.IGNORECASE),
    ),
    # -- goal planning: explicit goal statements -----------------------------
    (
        IntentKind.GOAL_PLANNING,
        re.compile(
            r"\b(set|create|add|define|track)\s+(a\s+|an\s+|my\s+|the\s+)?goal\b|\bmy\s+"
            r"goal\s+is\b|\bgoals?\b.*\b(achieve|accomplish|finish|complete|reach)\b",
            re.IGNORECASE,
        ),
    ),
    (
        IntentKind.GOAL_PLANNING,
        re.compile(r"\b(roadmap|plan\s+to\s+achieve|steps\s+to\s+(achieve|complete|finish))\b", re.IGNORECASE),
    ),
    # -- file editing (before creation: no — creation and editing are siblings) --
    (
        IntentKind.FILE_EDITING,
        re.compile(
            rf"\b{_EDIT_VERBS}\s+(the\s+|my\s+)?{_ARTIFACT_RE.pattern}",
            re.IGNORECASE,
        ),
    ),
    # -- file generation: work verbs targeting an artifact -------------------
    # A bounded noun phrase may sit between the article and the artifact
    # ("build a complete desktop expense tracker application"), and a
    # "save … to <file>" destination is an explicit file request even when
    # the verb is analysis ("save the analysis to analysis.md").
    (
        IntentKind.FILE_GENERATION,
        re.compile(
            r"\b(save|write|export|dump)\s+(?:the\s+|this\s+|that\s+|your\s+)?"
            r"[\w\s\-]{1,48}?\bto\s+[\w\-/]+\.(?:py|js|ts|jsx|tsx|css|scss|html|htm|md|"
            r"markdown|json|yml|yaml|toml|sh|bat|ps1|sql|txt|env|ini|cfg|xml|svg|csv|"
            r"log|pdf)\b",
            re.IGNORECASE,
        ),
    ),
    (
        IntentKind.FILE_GENERATION,
        re.compile(
            rf"\b{_CREATE_VERBS}\s+(?:a\s+|an\s+|the\s+|my\s+)?"
            r"(?:[\w\-]+\s+){{0,4}}?" + _ARTIFACT_RE.pattern,
            re.IGNORECASE,
        ),
    ),
    # -- workspace introspection: analyze/summarize the opened project ------------
    # Read-only analysis of the workspace is work (context, index, retrieval),
    # not chat: it must never be swallowed into the conversation path.
    (
        IntentKind.WORKSPACE_MANAGEMENT,
        re.compile(
            r"\b(analyze|summarize|review|audit|inspect|examine|explore|profile|map\b|scan|"
            r"survey)\s+(the\s+|this\s+|my\s+)?(workspace|project|codebase|repository|repo|"
            r"folder|directory|files?|file\s*tree|structure|dependencies|architecture)\b",
            re.IGNORECASE,
        ),
    ),
    # -- question answering: factual/explanation queries ----------------------
    (
        IntentKind.QUESTION_ANSWERING,
        re.compile(
            r"^(what|what'?s|what\s+(is|are|was|were|does|do|did|can|should|could|would)|"
            r"who|who'?s|whom|whose|when|where|which|why|how|how\s+(is|are|was|were|"
            r"does|do|did|can|should|could|would|much|many|long|far|often|to)|explain|"
            r"define|describe|meaning\s+of|definition\s+of|tell\s+me\s+about|is\s+it\s+"
            r"true|does\s+\w+\s+have|can\s+you\s+explain)\b",
            re.IGNORECASE,
        ),
    ),
    (
        IntentKind.QUESTION_ANSWERING,
        re.compile(r"\b(?:what|who|when|where|which|why|how)\b[^?.!]{4,}\?", re.IGNORECASE),
    ),
    # -- explicit conversation (greetings / social) ---------------------------
    (
        IntentKind.CONVERSATION,
        re.compile(
            r"^(hi|hello|hey|yo|thanks|thank\s+you|thank\s+u|how\s+are\s+you|how\s+do\s+"
            r"you\s+do|good\s+(morning|afternoon|evening)|nice\s+to\s+meet|bye|goodbye|"
            r"what\s+can\s+you\s+do|who\s+are\s+you|what\s+is\s+this|help\s+me\s+with\s+"
            r"nothing|haha|lol)\b",
            re.IGNORECASE,
        ),
    ),
]


class IntentRouter:
    """Deterministic intent gate — conversation never reaches the workspace."""

    def route(self, prompt: str) -> IntentKind:
        """Classify ``prompt`` into exactly one :class:`IntentKind`."""
        text = (prompt or "").strip()
        lowered = text.lower()
        # Explicit work intent wins first: a build/create/analyze-style request
        # must never be captured by incidental workspace verbs in its prose
        # ("Search and filter expenses" inside a build-the-app prompt is not
        # a backend search operation).
        for kind, pattern in _RULES:
            if pattern.search(lowered):
                return kind
        # Backend operations (list/search/read/rename/move/delete/create folder)
        # are served by the Action Engine alone — the router never lets them
        # fall through to chat. Extractors are conservative: search needs a
        # quoted pattern (or explicit term) and delete needs a real file
        # target, so conversational prose never fires a filesystem op.
        if extract_workspace_ops(text):
            return IntentKind.WORKSPACE_MANAGEMENT
        # Fallback: a question mark after a question-word pattern or a plain
        # greeting is already covered above; everything else is conversation.
        if "?" in lowered and re.search(r"\b(what|who|when|where|which|why|how)\b", lowered):
            return IntentKind.QUESTION_ANSWERING
        return IntentKind.CONVERSATION

    def route_with_ops(self, prompt: str) -> tuple[IntentKind, list[dict]]:
        """Route, and additionally return extracted workspace ops (may be
        empty even for a WORKSPACE_MANAGEMENT intent — e.g. ``list files``
        has no operand)."""
        kind = self.route(prompt)
        ops = extract_workspace_ops(prompt) if kind == IntentKind.WORKSPACE_MANAGEMENT else []
        return kind, ops
