"""Workspace-first awareness brief — injected before EVERY model call.

Phase XIII (workspace-first / tool-first): the agent is never allowed to
forget that it is operating inside the user's real project with a full set of
file tools. Every model invocation (task decomposition, tool execution,
review, final synthesis) is prefixed with a compact brief covering:

- what project it is operating inside (name, id, workspace path)
- which workspace tools exist and how it should default to using them
- a summary of the files in the workspace (+ the most recently modified)
- what the current conversation has already said

The brief is pure text: it never raises, and it degrades gracefully when the
memory store or the file operator are absent.
"""

from __future__ import annotations

from synapse.domain.enums import MemoryScope

#: how many most-recently-modified files to surface.
_MAX_RECENT = 6
#: cap on how many workspace paths are printed before collapsing the list.
_MAX_TREE_PATHS = 60

_WORKSPACE_HEADER = (
    "You are operating inside Synapse — an AI workspace connected to the user's "
    "real project (not an abstract chat)."
)

_AVAILABLE_TOOLS = (
    "Available workspace tools (all enabled, no permission needed):\n"
    "  ✓ Read Files\n"
    "  ✓ Write Files\n"
    "  ✓ Create Files & Folders\n"
    "  ✓ Rename / Move / Delete Files\n"
    "  ✓ Search the Workspace\n"
    "  ✓ Read Images & Documents"
)

_DEFAULT_OPERATION = (
    "Default behaviour — operate on the workspace:\n"
    "  - build / create / modify / refactor / generate / fix / design requests "
    "→ create and edit real files, then keep the chat reply short (the files "
    "are the deliverable).\n"
    "  - Read the workspace before changing it; never guess file contents.\n"
    "  - Docs and tests are project files too — write them when relevant."
)

_EMPTY_TREE = "Files in the workspace: (none yet — folder empty or not scanned)"


def _recent_modified(entries: list[dict], limit: int = _MAX_RECENT) -> list[str]:
    """Most recently modified file lines, newest first: ``- path (modified …)``."""
    with_stamp = [e for e in entries if e.get("modified_at")]
    ordered = sorted(with_stamp, key=lambda e: e["modified_at"], reverse=True)
    lines: list[str] = []
    for entry in ordered[:limit]:
        stamp = entry["modified_at"][:19].replace("T", " ")
        lines.append(f"- {entry['path']} (modified {stamp})")
    return lines


def _chat_lines(memory, conversation_id: str | None, limit: int = 3) -> list[str]:
    """Last few user/agent messages of THIS chat, newest first, one line each."""
    if memory is None or not conversation_id:
        return []
    try:
        entries = memory.recent(
            MemoryScope.CONVERSATION, limit + 2, conversation=conversation_id
        )
    except Exception:  # noqa: BLE001 - the brief never breaks the pipeline
        return []
    lines: list[str] = []
    for entry in entries:
        if not entry.text:
            continue
        who = "user" if entry.source == "user" else "agent"
        text = " ".join(entry.text.split())
        if len(text) > 140:
            text = f"{text[:139]}…"
        lines.append(f"- {who}: {text}")
        if len(lines) >= limit:
            break
    return lines


def build_workspace_brief(
    *,
    project_id: str | None = None,
    project_name: str | None = None,
    project_path: str | None = None,
    file_operator=None,
    memory=None,
    conversation_id: str | None = None,
) -> str:
    """Assemble the always-injected workspace awareness brief.

    ``file_operator`` may be any object exposing ``list_tree()`` (returns
    ``[{path, size, modified_at}, ...]``) and ``root``; ``memory`` any store
    exposing ``recent(scope, limit, conversation=)``. Everything is optional —
    the brief degrades to the identity + tools header when absent.
    """
    chunks: list[str] = [_WORKSPACE_HEADER]

    identity = project_name or project_id or "the opened project"
    line = f"Current workspace: {identity}"
    if project_id and project_id != identity:
        line += f" (id: {project_id})"
    chunks.append(line)

    root: str | None = None
    if project_path:
        root = str(project_path)
    elif file_operator is not None:
        try:
            root = str(file_operator.root)
        except Exception:  # noqa: BLE001
            root = None
    if root:
        chunks.append(f"Workspace path: {root}")

    chunks.append(_AVAILABLE_TOOLS)
    chunks.append(_DEFAULT_OPERATION)

    entries: list[dict] = []
    if file_operator is not None:
        try:
            entries = file_operator.list_tree()
        except Exception:  # noqa: BLE001
            entries = []

    if entries:
        paths = sorted({e["path"] for e in entries}, key=str.lower)
        shown, rest = paths[:_MAX_TREE_PATHS], len(paths) - _MAX_TREE_PATHS
        tree = f"Files in the workspace ({len(paths)}):\n  - " + "\n  - ".join(shown)
        if rest > 0:
            tree += f"\n  - … and {rest} more"
        chunks.append(tree)
    else:
        chunks.append(_EMPTY_TREE)

    recent = _recent_modified(entries)
    if recent:
        chunks.append("Recently modified:\n" + "\n".join(recent))

    chat = _chat_lines(memory, conversation_id)
    if chat:
        chunks.append("Current conversation (stay on this chat's context):\n" + "\n".join(chat))

    return "\n\n".join(chunks)
