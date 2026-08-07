"""Request classifier — deterministic classification of every prompt.

Phase 7: every request is classified into exactly one ``RequestKind``:

- ``WORKSPACE_OPERATION``  — list / search / read / rename / move / delete / folders
- ``PROJECT_ANALYSIS``     — analyze / explain / review existing code
- ``DOCUMENTATION``        — write / generate readmes, guides, manuals
- ``FILE_MODIFICATION``    — edit / refactor / fix existing files
- ``PROJECT_GENERATION``   — scaffold a website / app / project
- ``FILE_CREATION``        — write one script / function / class / test
- ``CHAT_RESPONSE``        — everything else (plain conversation)

Classification is rule-based and offline on purpose: it must be deterministic,
fast, and testable before any model is consulted. Rules are evaluated in
priority order (specific operations win over broad chat).
"""

from __future__ import annotations

import re

from synapse.domain.enums import RequestKind

#: priority-ordered (pattern, kind) rules; first match wins.
_RULES: list[tuple[re.Pattern[str], RequestKind]] = [
    # -- workspace operations (explicit file verbs) ---------------------
    (
        re.compile(
            r"\b(list|show|display|print)\s+(the\s+)?"
            r"(files|file\s*tree|tree|structure|folders|contents|workspace)\b"
        ),
        RequestKind.WORKSPACE_OPERATION,
    ),
    (
        re.compile(
            r"\b(search|find|grep|locate)\s+(for\s+)?[\"']?[A-Za-z0-9_.*/?]+[\"']?"
        ),
        RequestKind.WORKSPACE_OPERATION,
    ),
    (
        re.compile(r"\b(rename|move)\s+[A-Za-z0-9_.\-/]+\s+to\s+[A-Za-z0-9_.\-/]+\b"),
        RequestKind.WORKSPACE_OPERATION,
    ),
    (
        re.compile(r"\b(delete|remove|erase)\s+[A-Za-z0-9_.\-/]+(?:\.[A-Za-z0-9]+)?\b"),
        RequestKind.WORKSPACE_OPERATION,
    ),
    (
        re.compile(r"\bcreate\s+(a\s+|an\s+)?(folder|directory|subfolder)s?\b"),
        RequestKind.WORKSPACE_OPERATION,
    ),
    # -- read an explicit file (must look like a filename: has an extension) --
    (
        re.compile(
            r"\b(?:read|cat|open|view|show|print)\s+(?:the\s+)?"
            r"(?:file\s+|contents?\s+of\s+)?"
            r"([A-Za-z0-9_\-/]*[A-Za-z0-9_\-]+\.[A-Za-z0-9]+)\b"
        ),
        RequestKind.WORKSPACE_OPERATION,
    ),
    # -- project analysis (reads, no writes) ------------------------------
    (
        re.compile(
            r"\b(analyze|explain|understand|describe|inspect|evaluate|audit|review|"
            r"summarize|summarise|read|go\s*through|walk\s*me\s*through)\s+(this\s+|the\s+|my\s+)?"
            r"(project|codebase|code|repository|app|application|workspace|script|files)\b"
        ),
        RequestKind.PROJECT_ANALYSIS,
    ),
    (
        re.compile(r"\b(what|how)\s+(does|is|are)\s+(this|the|my)\s+(project|code|codebase|app|script|file)\b"),
        RequestKind.PROJECT_ANALYSIS,
    ),
    # -- documentation ----------------------------------------------------
    (
        re.compile(
            r"\b(write|generate|create|update|improve)\s+(a\s+|an\s+|the\s+)?"
            r"(?:[\w\-]+\s+)?(readme|documentation|docs|doc\b|markdown|manual|guide|"
            r"changelog|wiki|tutorial)\b"
        ),
        RequestKind.DOCUMENTATION,
    ),
    # -- modification (existing artifacts) --------------------------------
    (
        re.compile(
            r"\b(edit|update|modify|change|rewrite|improve|adjust|tweak|patch|"
            r"redesign|restyle)\s+(the\s+)?(code|file|script|website|web\s*app|app|"
            r"function|class|test|readme|css|html|style|color|layout|design|theme|"
            r"[a-z0-9_.\-/]+\.[a-z0-9]+)\b"
        ),
        RequestKind.FILE_MODIFICATION,
    ),
    (
        re.compile(r"\bfix\s+(the\s+)?(bug|code|script|website|web\s*app|app|file|function|error)\b"),
        RequestKind.FILE_MODIFICATION,
    ),
    (
        re.compile(r"\bmake\s+my\s+(website|web\s*app|app|script|code|project|file|readme)\b"),
        RequestKind.FILE_MODIFICATION,
    ),
    (
        re.compile(r"\b(add|remove|change)\s+(a\s+)?(feature|button|color|style|dark\s*mode|light\s*mode|page|section|function|method|test|route|navbar)\b"),
        RequestKind.FILE_MODIFICATION,
    ),
    # -- project generation (new artifacts) -------------------------------
    (
        re.compile(
            r"\b(create|build|make|generate|scaffold|design|develop)\s+(a\s+|an\s+)?"
            r"(?:[\w\-]+\s+)?"
            r"(website|web\s*app|web\s*page|site|portfolio|app|application|project|"
            r"dashboard|game|cli|tool|plugin|extension|module|package|api|landing\s*page|"
            r"store|shop|ecommerce|blog|forum|cms|chatbot|bot|calculator|todo\s*list|"
            r"todo\s*app|wiki|timer|catalog)\b"
        ),
        RequestKind.PROJECT_GENERATION,
    ),
    # -- single file creation ----------------------------------------------
    (
        re.compile(
            r"\b(create|write|generate|add)\s+(a\s+|an\s+)?"
            r"(?:[\w\-]+\s+)?"
            r"(script|program|function|class|test|unit\s*test|file|endpoint|sql\s*query|"
            r"python|javascript|typescript|html|css|bash|shell|go|rust|java|c\+\+|ruby|php|config)\b"
        ),
        RequestKind.FILE_CREATION,
    ),
]


def classify_request(prompt: str) -> RequestKind:
    """Classify ``prompt`` into one of the seven request kinds."""
    lowered = (prompt or "").lower().strip()
    for pattern, kind in _RULES:
        if pattern.search(lowered):
            return kind
    return RequestKind.CHAT_RESPONSE


#: explicit-operation extractors: (action, regex) in priority order.
_OP_EXTRACTORS: list[tuple[str, re.Pattern[str]]] = [
    ("list", re.compile(r"\b(list|show|display|print)\s+(the\s+)?(files|file\s*tree|tree|structure|folders|contents|workspace)\b", re.IGNORECASE)),
    ("search", re.compile(r"\b(search|find|grep|locate)\s+(?:for\s+)?[\"']?([A-Za-z0-9_.*/?]+)[\"']?", re.IGNORECASE)),
    ("read", re.compile(r"\b(?:read|cat|open|view|show|print)\s+(?:the\s+)?(?:file\s+|contents?\s+of\s+)?([A-Za-z0-9_\-/]*[A-Za-z0-9_\-]+\.[A-Za-z0-9]+)\b", re.IGNORECASE)),
    ("rename", re.compile(r"\b(?:rename|move)\s+([A-Za-z0-9_.\-/]+)\s+to\s+([A-Za-z0-9_.\-/]+)\b", re.IGNORECASE)),
    ("delete", re.compile(r"\b(?:delete|remove|erase)\s+([A-Za-z0-9_.\-/]+(?:\.[A-Za-z0-9]+)?)\b", re.IGNORECASE)),
    ("create_folder", re.compile(r"\bcreate\s+(?:a\s+|an\s+)?(?:folder|directory|subfolder)s?\s+(?:called|named)?\s*([A-Za-z0-9_.\-/]+)\b", re.IGNORECASE)),
]


def extract_workspace_ops(prompt: str) -> list[dict]:
    """Deterministic backend-only operations from explicit user instructions.

    Returns ops like ``{"action": "rename", "path": ..., "to": ...}``.
    A ``delete`` here is considered user-confirmed: the user literally asked
    for the deletion. Returns ``[]`` when nothing is explicit enough.
    """
    text = prompt or ""
    ops: list[dict] = []
    for action, pattern in _OP_EXTRACTORS:
        for match in pattern.finditer(text):
            if action == "list":
                ops.append({"action": "list"})
            elif action == "search":
                ops.append({"action": "search", "pattern": match.group(2)})
            elif action == "read":
                ops.append({"action": "read", "path": match.group(1)})
            elif action == "rename":
                ops.append({"action": "rename", "path": match.group(1), "to": match.group(2)})
            elif action == "delete":
                ops.append({"action": "delete", "path": match.group(1), "confirmed": True})
            elif action == "create_folder":
                ops.append({"action": "create_folder", "path": match.group(1)})
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for op in ops:
        key = (op.get("action"), op.get("path", ""), op.get("to", ""), op.get("pattern", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(op)
    return deduped


# -- Phase XIII: the workspace-access gate ----------------------------------
# The planner instantiation decision pipeline starts with "does this request
# operate on the workspace?" before any model call. The gate is deterministic:
# explicit file verbs (build/create/modify/refactor/...), explicit destinations
# (``save ... to file.py``), explicit filesystem paths, or project/code coverage
# words all answer YES; pure greetings and chat-only opinions answer NO.

_WORKSPACE_GATE_RE = re.compile(
    r"\b(create|build|write|generate|scaffold|design|develop|make|modify|refactor|"
    r"rewrite|edit|update|change|fix|debug|delete|remove|rename|move|export|save|"
    r"dump|search|find|read|open|list)\b",
    re.IGNORECASE,
)
_FILE_OPERAND_RE = re.compile(r"(?:[\w\-/]+\.)[a-z0-9]+", re.IGNORECASE)
_PROJECT_WORDS_RE = re.compile(
    r"\b(file|files|folder|folders|directory|code|script|function|class|method|api|"
    r"endpoint|website|web\s*app|app|workspace|project|codebase|readme|documentation|"
    r"docs|test|tests|bug)\b",
    re.IGNORECASE,
)
_CHAT_ONLY_GREETING_RE = re.compile(
    r"^(hi|hello|hey|yo|thanks|thank\s+you|thank\s+u|how\s+are\s+you|"
    r"what\s+can\s+you\s+do|who\s+are\s+you|what\s+is\s+this)\b",
    re.IGNORECASE,
)


def requires_workspace_access(prompt: str) -> bool:
    """Deterministic planner gate — would this request operate on the opened
    workspace rather than only answer in chat?

    True for build/create/modify/refactor/generate/fix/design-type requests,
    requests naming real files, or requests about project code/artifacts. False
    only for pure greetings and chat-only opinions; everything else tends to
    touch the workspace and is flagged YES so context and file tools are used.
    """
    text = (prompt or "").strip().lower()
    if not text:
        return False
    if _CHAT_ONLY_GREETING_RE.match(text):
        return False
    if _WORKSPACE_GATE_RE.search(text):
        return True
    if _FILE_OPERAND_RE.search(prompt or ""):
        return True
    return bool(_PROJECT_WORDS_RE.search(text))
