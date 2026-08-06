"""Manifest parser — extracts structured file operations from model output.

Phase 6 file-generation tasks tell the model to answer with a JSON manifest
instead of prose. This module is the tolerant reader side: it finds the JSON
in the result (behind markdown fences, trailing prose, etc.), normalizes the
dozens of shapes a model may produce, and rejects unsafe paths.

Accepted shapes (all equivalent)::

    {"files": [{"path": "a.py", "content": "..."}]}
    {"file":  {"path": "a.py", "content": "..."}}
    {"a.py": "content", "src/x.txt": "..."}
    {"operation": "create", "path": "a.py", "content": "..."}
    {"rename": {"from": "old", "to": "new"}}
    {"delete": "stale.txt"}   /  {"files": [{"action": "delete", "path": ...}]}
    {"folders": ["assets", "css", "js"]}
    {"files": [{"action": "edit", "path": "a.py", "old": "...", "new": "..."}]}

Phase 7: ``folders`` / ``dirs`` / ``directories`` arrays become
``create_folder`` ops (the backend makes real directories in the workspace),
and ``edit`` items become targeted replace operations instead of whole-file
overwrites.

Phase 7 fix: models are instructed to answer with ``MANIFEST_INSTRUCTION``
appended to the prompt. When a model still answers with prose + fenced code
blocks, the fallback converter now emits one write op per code block (using
``file:`` headers or the language's default name) instead of collapsing the
whole answer into a single ``output.txt``.
"""

from __future__ import annotations

import json
import re

from synapse.logging import get_logger
from synapse.workspace.operator import is_safe_relative_path

log = get_logger("synapse.workspace.manifest")

#: Written into the prompt of every file-output task (Phase 7 fix). Without
#: this instruction real models emit prose / fenced code; with it they return
#: a JSON manifest the backend executes. Kept next to the parser so the two
#: stay in lockstep.
MANIFEST_INSTRUCTION = """\
OUTPUT FORMAT — respond with ONE JSON object and nothing else: no prose, no \
markdown, no code fences. For example:
{"folders": ["assets"], "files": [{"path": "index.html", "content": "<p>hi</p>"}, {"path": "style.css", "content": "body{}"}]}

Rules:
- "files" is the single list of file contents; every item has "path" (relative to the project — no leading "/", no "..", no absolute paths) and "content" (the complete file body as one string).
- put every directory that must exist under "folders".
- to change an existing file use {"action":"edit","path":..., "old":"<exact snippet>", "new":"<replacement>"}.
- to rename a file use {"action":"rename","path":..., "to":...}.
- to delete a file use {"action":"delete","path":...}.
Only the paths inside the JSON will be created — nothing outside the JSON is read."""

#: Appended to every task that does NOT already carry the strict manifest
#: requirement (any request kind, per Phase X "universal workspace tool").
#: The model is told it works inside the user's project folder and may
#: produce files; a pure-prose answer remains perfectly valid.
WORKSPACE_TOOL_INSTRUCTION = """\
You are working inside the user's project folder. The file listings above are \
real files you can read and edit, and you are allowed to create new files \
anywhere inside the folder.

If your answer produces or changes files, end it with ONE JSON object and \
nothing else: {"folders": ["docs"], "files": [{"path": "docs/a.md", "content": "..."}]}
Each "files" item has "path" (relative to the project — no leading "/", no \
"..", no absolute paths) and "content" (the complete file body). To change an \
existing file use {"action":"edit","path":..., "old":"<exact snippet>", \
"new":"<replacement>"}; to rename {"action":"rename","path":...,"to":...}; to \
delete {"action":"delete","path":...}.

Only add a manifest when your answer actually creates or changes files — if \
it is a normal answer, plain prose is fine."""



#: language tag (from a fence) -> preferred extension (without the dot).
_LANG_EXT: dict[str, str] = {
    "python": "py", "py": "py",
    "javascript": "js", "js": "js", "json": "json", "jsonc": "json",
    "html": "html", "css": "css", "markdown": "md", "md": "md",
    "shell": "sh", "bash": "sh", "sh": "sh",
    "typescript": "ts", "ts": "ts", "yaml": "yaml", "yml": "yaml",
    "toml": "toml", "xml": "xml", "sql": "sql", "c": "c", "cpp": "cpp",
    "java": "java", "go": "go", "rust": "rs", "ruby": "rb", "php": "php",
    "text": "txt", "txt": "txt",
}

#: fenced code block regex — captures language and (optional) leading path.
_FENCE_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_+-]*)[ \t]*\n?(?P<body>.*?)```",
    re.DOTALL,
)
_PATH_RE = re.compile(r"^[#]?\s*(?:FILE|PATH)?\s*[:=]?\s*([A-Za-z0-9_./\-]+\.\w+)\s*$")


def _first_json(text: str) -> dict | None:
    """Locate and parse the first balanced JSON object in ``text``."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        import json

                        value = json.loads(candidate)
                        if isinstance(value, dict):
                            return value
                    except Exception:  # noqa: BLE001
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def _payload_to_ops(payload: dict) -> list[dict]:
    """Convert a parsed JSON object into normalized file ops."""
    ops: list[dict] = []
    # explicit folder creation — the backend makes real directories.
    for key in ("folders", "dirs", "directories"):
        value = payload.get(key)
        if isinstance(value, list):
            for rel in value:
                rel = str(rel)
                if is_safe_relative_path(rel):
                    ops.append({"action": "create_folder", "path": rel})
                else:
                    ops.append({"action": "create_folder", "path": rel, "error": "unsafe path"})
    for key in ("files", "create", "writes", "operations", "ops"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                op = _clean_op_item(item)
                if op:
                    ops.append(op)
        elif isinstance(value, dict):
            for rel, content in value.items():
                if is_safe_relative_path(str(rel)):
                    ops.append({"action": "write", "path": str(rel), "content": str(content)})
                else:
                    ops.append({"action": "write", "path": str(rel), "content": str(content), "error": "unsafe path"})
    if payload.get("file") and isinstance(payload["file"], dict):
        op = _clean_op_item(payload["file"])
        if op:
            ops.append(op)
    # top-level mapping {"path.py": content, ...} (no files/file key)
    if not ops:
        for rel, value in payload.items():
            if isinstance(value, str):
                if is_safe_relative_path(str(rel)):
                    ops.append({"action": "write", "path": str(rel), "content": value})
                else:
                    ops.append({"action": "write", "path": str(rel), "content": value, "error": "unsafe path"})
    return ops


def _dedupe_ops(ops: list[dict]) -> list[dict]:
    """Drop duplicate (action, path) ops, preserving order."""
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for op in ops:
        key = (op.get("action", "write"), op.get("path", op.get("to", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(op)
    return deduped


#: a manifest may arrive wrapped in a single JSON code fence.
_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?([\s\S]*?)\n?\s*```\s*$")


def _parse_strict_manifest(text: str) -> list[dict]:
    """Strict parse: the WHOLE model answer must be a JSON manifest.

    Used for tasks the planner did not mark as file-producing: an embedded
    JSON object inside ordinary prose (e.g. a model quoting the manifest
    format) is NOT enough to create files — only a pure-manifest answer is
    honoured. Prose answers return [].
    """
    candidate = text
    m = _JSON_FENCE_RE.match(text)
    if m:
        candidate = m.group(1).strip()
    try:
        payload = json.loads(candidate)
    except Exception:  # noqa: BLE001 - not JSON -> not a manifest
        return []
    if not isinstance(payload, dict):
        return []
    return _dedupe_ops(_payload_to_ops(payload))


def parse_file_manifest(
    result: str | None,
    hint: str = "",
    *,
    strict: bool = False,
) -> list[dict]:
    """Parse model output into a normalized list of file ops.

    Returns ops: ``{"action": "write"|"rename"|"delete", "path": ..., ...}``.
    Ops with an unsafe path keep their payload but get ``"error": ...`` so the
    orchestrator surfaces them instead of silently dropping work.

    ``strict=True`` (Phase X universal workspace tool) requires the whole
    output to BE a manifest — ordinary prose never becomes files, no fence
    fallback. Non-strict keeps the tolerant prose/fenced-code converter for
    tasks the planner explicitly marked as file-producing.
    """
    if not result:
        return []
    text = result.strip()
    if strict:
        return _parse_strict_manifest(text)

    payload = _first_json(text)
    if payload is not None and isinstance(payload, dict):
        ops = _dedupe_ops(_payload_to_ops(payload))
        if ops:
            return ops

    # fallback: fenced code blocks -> one write op per block (Phase 7 fix).
    fallback = _fences_to_ops(text, hint)
    if fallback:
        return fallback
    return []


def _clean_op_item(item: object) -> dict | None:
    if not isinstance(item, dict):
        return None
    action = str(item.get("action") or item.get("operation") or "").lower()
    raw_path = item.get("path") or item.get("file") or item.get("name")
    content = item.get("content") or item.get("text") or item.get("body")
    to = item.get("to") or item.get("dest") or item.get("new_path")

    if action in ("rename", "move") or to:
        src = str(item.get("from") or item.get("path") or "")
        dst = str(to or item.get("path") or "")
        safe = is_safe_relative_path(src) and is_safe_relative_path(dst)
        return {"action": "rename", "path": src, "to": dst, **({"error": "unsafe path"} if not safe else {})}
    if action in ("delete", "remove"):
        return {"action": "delete", "path": str(raw_path)}
    if action in ("edit", "patch", "replace"):
        old = item.get("old") or item.get("find") or item.get("from")
        new = item.get("new") or item.get("replace") or item.get("to")
        safe = is_safe_relative_path(str(raw_path))
        return {
            "action": "edit",
            "path": str(raw_path),
            "old": str(old or ""),
            "new": str(new or ""),
            **({"error": "unsafe path"} if not safe else {}),
        }
    if raw_path:
        safe = is_safe_relative_path(str(raw_path))
        return {
            "action": "write",
            "path": str(raw_path),
            "content": str(content or ""),
            **({"error": "unsafe path"} if not safe else {}),
        }
    return None


#: language tag -> sensible default file name (used by the fence fallback so
#: a "create a website" answer becomes index.html/style.css/script.js instead
#: of output.html/output_2.css/...).
_LANG_DEFAULT_PATH: dict[str, str] = {
    "html": "index.html",
    "css": "style.css",
    "javascript": "script.js", "js": "script.js",
    "typescript": "main.ts", "ts": "main.ts",
    "python": "main.py", "py": "main.py",
    "markdown": "README.md", "md": "README.md",
    "shell": "script.sh", "bash": "script.sh", "sh": "script.sh",
    "go": "main.go", "rust": "main.rs", "ruby": "main.rb",
    "sql": "query.sql", "json": "data.json",
}

#: header line inside a fence naming the destination file, e.g.
#: ``# file: index.html`` / ``// style.css`` / ``<!-- FILE = app.js -->`` /
#: ``index.html:``. The captured path must be the only real content of the
#: header line (code that merely starts with a comment still matches the
#: extension requirement below).
_PATH_HEADER_RE = re.compile(
    r"^\s*(?:#|//|/\*|<!--|--|%;|['\"])\s*(?:FILE|PATH)?\s*[:=]?\s*"
    r"([A-Za-z0-9_./\-]+\.\w+)\s*(?:\*/|-->)?\s*$",
    re.IGNORECASE,
)


def _fences_to_ops(text: str, hint: str) -> list[dict]:
    """Convert every fenced code block into a write op.

    Order is preserved. A block naming its destination in the first line
    (``file: path`` in any comment style, or a bare ``path.ext:`` line) keeps
    that path; otherwise the planner hint or the language's default name is
    used. Colliding names get a numeric suffix so no content is lost.
    """
    blocks = _FENCE_RE.findall(text)
    if not blocks:
        return []
    used: set[str] = set()
    ops: list[dict] = []
    for lang, body in blocks:
        raw = body.rstrip("\n")
        if not raw or not raw.strip():
            continue
        path, body = _fence_destination(raw, lang, hint)
        path = _unique_path(path, used)
        if not is_safe_relative_path(path):
            path = "output.txt"
        ops.append({"action": "write", "path": path, "content": body})
    return ops


def _fence_destination(body: str, lang: str, hint: str) -> tuple[str, str]:
    """Return (path, body-without-header) for one fenced block."""
    first_line, _, rest = body.partition("\n")
    head = first_line.strip()
    m = _PATH_HEADER_RE.match(head) or _PATH_RE.match(head)
    if m:
        return m.group(1), rest.strip("\n")
    if hint:
        return hint, body
    default = _LANG_DEFAULT_PATH.get(lang.strip().lower(), "")
    if default:
        return default, body
    ext = _LANG_EXT.get(lang.strip().lower(), "")
    return f"output.{ext}" if ext else "output.txt", body


def _unique_path(path: str, used: set[str]) -> str:
    """Append a numeric suffix when the name is already taken."""
    if path not in used:
        used.add(path)
        return path
    base, sep, ext = path.rpartition(".")
    n = 2
    while True:
        candidate = f"{base}_{n}.{ext}" if sep else f"{path}_{n}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        n += 1