"""Workspace-first awareness brief & specialist prompt builder — injected before EVERY specialist call.

Phase XIII / Architect Directive: the agent operates inside the user's real project
with full file tools and conversation memory. Every model invocation receives a clean,
host-controlled prompt with:

- Project identity and workspace path
- Current chat history: last 8 conversation turns
- Conversation scope summary: key facts & discussion topics
- Live workspace file list (up to 60 paths) + recently modified
- Target file content when modifying existing files
- Non-negotiable rules for structured file actions
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from synapse.domain.enums import MemoryScope

#: how many most-recently-modified files to surface.
_MAX_RECENT = 6
#: cap on how many workspace paths are printed before collapsing the list.
_MAX_TREE_PATHS = 40

_WORKSPACE_HEADER = (
    "You are inside Synapse, a real project workspace on disk "
    "(connected to the user's real project, not an abstract chat)."
)

_AVAILABLE_TOOLS = (
    "Available workspace tools (all enabled, no permission needed):\n"
    "  ✓ Read Files\n"
    "  ✓ Write Files\n"
    "  ✓ Create Files & Folders\n"
    "  ✓ Rename / Move / Delete Files\n"
    "  ✓ Search the Workspace\n"
    "  ✓ Read Images & Documents\n"
    "You may freely read, create, edit, rename, delete, and search files."
)

_DEFAULT_OPERATION = (
    "Operating rules:\n"
    "  - Always read real file contents before editing. Never invent contents.\n"
    "  - When the user asks for code, docs, tests or any artifact, create or edit the real files. "
    "The files are the deliverable; keep the chat reply short and factual.\n"
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


def _find_chat_records_from_disk(
    project_id: str | None,
    conversation_id: str | None,
    memory=None,
    limit: int = 8,
) -> list[tuple[str, str]]:
    if not conversation_id:
        return []

    candidates: list[Path] = []
    env_home = os.environ.get("SYNAPSE_HOME")
    if env_home:
        candidates.append(Path(env_home))

    try:
        from synapse.config.paths import _default_home
        dh = _default_home()
        if dh not in candidates:
            candidates.append(dh)
    except Exception:
        pass

    if memory is not None and getattr(memory, "_root", None):
        try:
            m_path = Path(memory._root)
            if m_path not in candidates:
                candidates.append(m_path)
                candidates.append(m_path.parent)
                candidates.append(m_path.parent.parent)
        except Exception:
            pass

    for root_dir in candidates:
        search_paths: list[Path] = []
        if project_id:
            search_paths.extend([
                root_dir / "data" / "projects" / "chats" / project_id / f"{conversation_id}.json",
                root_dir / "chats" / project_id / f"{conversation_id}.json",
                root_dir / "projects" / "chats" / project_id / f"{conversation_id}.json",
            ])
        if (root_dir / "chats").is_dir():
            search_paths.extend(list((root_dir / "chats").glob(f"*/{conversation_id}.json")))
        if (root_dir / "data" / "projects" / "chats").is_dir():
            search_paths.extend(list((root_dir / "data" / "projects" / "chats").glob(f"*/{conversation_id}.json")))

        for p in search_paths:
            if p.is_file():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    msgs = data.get("messages", [])
                    if msgs:
                        raw = []
                        for m in msgs[-limit:]:
                            raw.append((m.get("role", "user"), m.get("content", "")))
                        return raw
                except Exception:
                    pass
    return []


def _chat_lines(
    memory=None,
    conversation_id: str | None = None,
    limit: int = 5,
    chat_store=None,
    messages: list | None = None,
    project_id: str | None = None,
    current_prompt: str | None = None,
) -> list[str]:
    """Last 4–5 user/assistant messages of THIS chat (up to 8 for chat memory)."""
    raw_turns: list[tuple[str, str]] = []
    if messages:
        for m in messages[-limit:]:
            role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "user")
            content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else str(m))
            raw_turns.append((str(role), str(content)))
    elif chat_store is not None and conversation_id:
        try:
            stored = chat_store.messages(conversation_id)
            for m in stored[-limit:]:
                raw_turns.append((m.role, m.content))
        except Exception:  # noqa: BLE001
            pass

    if not raw_turns and memory is not None and conversation_id:
        try:
            entries = memory.recent(
                MemoryScope.CONVERSATION, limit + 4, conversation=conversation_id
            )
            for entry in reversed(entries[:limit]):
                if not entry.text:
                    continue
                who = "user" if entry.source == "user" else "assistant"
                raw_turns.append((who, entry.text))
        except Exception:  # noqa: BLE001
            pass

    if not raw_turns and conversation_id:
        raw_turns = _find_chat_records_from_disk(project_id, conversation_id, memory=memory, limit=limit)

    # Exclude trailing active user prompt if it matches the current executing request
    if raw_turns and current_prompt:
        last_role, last_text = raw_turns[-1]
        c_clean = current_prompt.strip().lower()
        t_clean = last_text.strip().lower()
        if last_role == "user" and (c_clean == t_clean or c_clean in t_clean or t_clean in c_clean):
            raw_turns = raw_turns[:-1]

    lines: list[str] = []
    for who, text in raw_turns:
        cleaned = " ".join(str(text).split())
        if len(cleaned) > 250:
            cleaned = f"{cleaned[:249]}…"
        lines.append(f"- {who}: {cleaned}")
    return lines


def _get_conversation_scope_summary(
    memory=None,
    conversation_id: str | None = None,
    chat_store=None,
    project_id: str | None = None,
    explicit_scope: str | None = None,
    messages: list | None = None,
    current_prompt: str | None = None,
) -> str:
    """Retrieve or build the short conversation_scope summary."""
    if explicit_scope:
        return explicit_scope.strip()

    if memory is not None and getattr(memory, "_root", None):
        scope_file = Path(memory._root) / "conversation_scope.json"
        if scope_file.is_file():
            try:
                data = json.loads(scope_file.read_text(encoding="utf-8"))
                if not conversation_id or data.get("chat_id") == conversation_id:
                    summary = data.get("summary")
                    if summary:
                        return str(summary).strip()
            except Exception:  # noqa: BLE001
                pass

    raw_turns: list[tuple[str, str]] = []
    if messages:
        for m in messages:
            role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "user")
            content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else str(m))
            raw_turns.append((str(role), str(content)))
    elif chat_store is not None and conversation_id:
        try:
            msgs = chat_store.messages(conversation_id)
            if msgs:
                raw_turns = [(m.role, m.content) for m in msgs]
        except Exception:
            pass

    if not raw_turns and memory is not None and conversation_id:
        try:
            entries = memory.recent(MemoryScope.CONVERSATION, 12, conversation=conversation_id)
            for entry in reversed(entries):
                if entry.text:
                    raw_turns.append(("user" if entry.source == "user" else "assistant", entry.text))
        except Exception:
            pass

    if not raw_turns and conversation_id:
        raw_turns = _find_chat_records_from_disk(project_id, conversation_id, memory=memory, limit=12)

    # Exclude trailing active user prompt if it matches the current executing request
    if raw_turns and current_prompt:
        last_role, last_text = raw_turns[-1]
        c_clean = current_prompt.strip().lower()
        t_clean = last_text.strip().lower()
        if last_role == "user" and (c_clean == t_clean or c_clean in t_clean or t_clean in c_clean):
            raw_turns = raw_turns[:-1]

    if raw_turns:
        user_msgs = [c for r, c in raw_turns if r == "user"]
        summary_topics = []
        for u in user_msgs:
            cleaned = " ".join(u.split())
            if cleaned and cleaned not in summary_topics:
                summary_topics.append(cleaned[:80])
        if summary_topics:
            topics_str = "; ".join(summary_topics[-3:])
            return f"Active chat focusing on: {topics_str} ({len(raw_turns)} turns total)."

    return f"Active workspace conversation for project '{project_id or 'workspace'}'."


def build_specialist_prompt(
    task_description: str,
    *,
    project_id: str | None = None,
    project_name: str | None = None,
    project_path: str | None = None,
    file_operator=None,
    memory=None,
    conversation_id: str | None = None,
    chat_store=None,
    messages: list | None = None,
    conversation_scope: str | None = None,
    target_edit_path: str | None = None,
    target_edit_content: str | None = None,
    prior_context: str | None = None,
    review_snippets: list[str] | None = None,
    workspace_context: str | None = None,
    is_chat: bool = False,
    include_file_rules: bool = True,
) -> str:
    """Build the single clean prompt template for specialists as mandated by Architect Directive."""
    p_name = project_name or project_id or "workspace"
    root = str(project_path) if project_path else (str(file_operator.root) if file_operator is not None else "/workspace")

    # 1. Chat history (4–5 turns by default, up to 8 if chat memory query)
    is_memory_query = is_chat or any(
        w in task_description.lower()
        for w in (
            "what is this chat",
            "what is our chat",
            "what is this conversation",
            "summarize this chat",
            "summarize this conversation",
            "summarize our conversation",
            "what have we done",
            "what did we do",
            "what have we accomplished",
            "prior",
            "earlier",
            "previous",
            "remember",
        )
    )
    chat_limit = 8 if is_memory_query else 5
    chat = _chat_lines(
        memory=memory,
        conversation_id=conversation_id,
        limit=chat_limit,
        chat_store=chat_store,
        messages=messages,
        project_id=project_id,
        current_prompt=task_description,
    )
    chat_text = "\n".join(chat) if chat else "(No previous chat messages in this session)"

    # 2. Conversation summary
    summary_text = _get_conversation_scope_summary(
        memory=memory,
        conversation_id=conversation_id,
        chat_store=chat_store,
        project_id=project_id,
        explicit_scope=conversation_scope,
        messages=messages,
        current_prompt=task_description,
    )

    # 3. Live files
    entries: list[dict] = []
    if file_operator is not None:
        try:
            entries = file_operator.list_tree()
        except Exception:
            entries = []

    if entries:
        paths = sorted({e["path"] for e in entries}, key=str.lower)
        shown, rest = paths[:_MAX_TREE_PATHS], len(paths) - _MAX_TREE_PATHS
        file_list_text = "\n".join(f"- {p}" for p in shown)
        if rest > 0:
            file_list_text += f"\n- … and {rest} more"
    else:
        file_list_text = "(No files in workspace yet)"

    # 4. Target file edit content
    edit_block = ""
    if target_edit_path and target_edit_content is not None:
        edit_block = (
            f"Current content of the file you must modify ({target_edit_path}):\n"
            f"```\n{target_edit_content}\n```\n\n"
        )

    # 5. Prior context / review snippets / workspace context
    context_block = ""
    if workspace_context:
        context_block += f"Workspace files:\n{workspace_context}\n\n"
    if prior_context:
        context_block += f"Previous step context:\n{prior_context}\n\n"
    if review_snippets:
        context_block += "Workspace actual files on disk for review:\n" + "\n\n".join(review_snippets) + "\n\n"

    rules_block = ""
    if include_file_rules and not is_chat:
        is_doc_task = any(ext in task_description.lower() for ext in (".pdf", ".docx", ".pptx", ".xlsx", ".csv", "pdf", "word", "powerpoint", "presentation", "spreadsheet", "excel", "report", "slides"))
        doc_rule = ""
        if is_doc_task:
            doc_rule = (
                f"- DOCUMENT CONTENT REQUIREMENT: For all documents (.pdf, .docx, .pptx, .xlsx, .csv), you MUST provide complete, detailed, readable text (full paragraphs, multi-point bullet lists, filled data tables, complete slide text). NEVER emit empty skeletons, title-only documents, or placeholder text. Verification strictly enforces content completeness.\n"
            )
        rules_block = (
            f"\n\nRules:\n"
            f"- OUTPUT FORMAT — If you create or change any file, end your response with exactly one JSON object and nothing after it.\n"
            f"- Preferred format (full overwrite – most reliable):\n"
            f'  {{"folders": ["optional"], "files": [{{"path": "relative/path", "content": "complete file body"}}]}}\n'
            f"{doc_rule}"
            f"- Only use edit/rename/delete actions when the change is tiny and you have the exact current snippet.\n"
            f"- Do not invent files that are not in the live list.\n"
            f"- Keep any normal text reply short. The files are the deliverable."
        )

    template = (
        f"You are a specialist inside Synapse working on a real project.\n\n"
        f"Project: {p_name}\n"
        f"Path: {root}\n\n"
        f"Current chat history (this chat only):\n"
        f"{chat_text}\n\n"
        f"Conversation summary (this chat):\n"
        f"{summary_text}\n\n"
        f"Live files:\n"
        f"{file_list_text}\n\n"
        f"{edit_block}"
        f"{context_block}"
        f"Your task: {task_description}"
        f"{rules_block}"
    )
    return template


def build_workspace_brief(
    *,
    project_id: str | None = None,
    project_name: str | None = None,
    project_path: str | None = None,
    file_operator=None,
    memory=None,
    conversation_id: str | None = None,
    chat_store=None,
    messages: list | None = None,
    conversation_scope: str | None = None,
) -> str:
    """Assemble the unified workspace awareness brief (compact reference)."""
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

    chat = _chat_lines(
        memory=memory,
        conversation_id=conversation_id,
        limit=5,
        chat_store=chat_store,
        messages=messages,
        project_id=project_id,
    )
    if chat:
        chunks.append(
            "Current chat history (stay on this conversation):\n" + "\n".join(chat)
        )

    scope_summary = _get_conversation_scope_summary(
        memory=memory,
        conversation_id=conversation_id,
        chat_store=chat_store,
        project_id=project_id,
        explicit_scope=conversation_scope,
        messages=messages,
    )
    if scope_summary:
        chunks.append(f"Conversation scope summary:\n{scope_summary}")

    return "\n\n".join(chunks)
