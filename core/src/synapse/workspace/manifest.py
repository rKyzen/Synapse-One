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
OUTPUT FORMAT — If you create or change files, end your response with exactly one JSON object and nothing after it.

Preferred formats:
1. Full create / overwrite (recommended for new files, code, docs, and rich documents .pdf, .docx, .pptx, .xlsx, .csv):
{"folders": ["optional/dirs"], "files": [{"path": "relative/path.ext", "content": "complete file body or document markdown/data"}]}

2. Rich artifact generation (alternative format for presentations, spreadsheets, reports):
{"artifacts": [{"path": "docs/report.pdf", "content": "# Title\n\nContent..."}, {"path": "data/expenses.xlsx", "content": "Category,Amount\nFood,50"}]}

3. Edit (use only when the change is small and you have the exact current snippet):
{"action": "edit", "path": "relative/path.py", "old": "exact current snippet", "new": "replacement"}

4. Rename / Delete:
{"action": "rename", "path": "old.py", "to": "new.py"}
{"action": "delete", "path": "stale.py"}

Rules:
- Paths relative to project root only
- Prefer full-content overwrite — it is far more reliable
- For documents (.pdf, .docx, .pptx, .xlsx, .csv), generate the ACTUAL FULL CONTENT requested (real numbered questions for question papers, real paragraphs for reports, real slide bullet points for presentations, real data rows for spreadsheets); NEVER emit meta-descriptions, summaries, or placeholder text
- If no files change, reply with normal prose only (no JSON)"""

WORKSPACE_TOOL_INSTRUCTION = MANIFEST_INSTRUCTION



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
_PATH_RE = re.compile(r"^[#]?\s*(?:FILE|PATH)?\s*[:=]?\s*([A-Za-z0-9_./:\-]+\.\w+)\s*$", re.IGNORECASE)

#: placeholder file bodies models produce when they echo the manifest
#: instruction's example instead of writing real content ("...").
_PLACEHOLDER_CONTENT_RE = re.compile(r"^[.…]{1,6}$")


def _is_placeholder_content(content: str) -> bool:
    """True when ``content`` is only the dotted placeholder from the
    instruction examples (``"content": "..."``, a fence body of ``...``).

    Empty strings are NOT placeholders — creating an empty file is a real,
    parseable intent.
    """
    return bool(_PLACEHOLDER_CONTENT_RE.match((content or "").strip()))


def _json_objects(text: str) -> list[dict]:
    """Every balanced JSON object in ``text``, in order of appearance.

    Models frequently prefix their real manifest with an echo of the format
    example; the LAST object is the most likely answer.
    """
    found: list[dict] = []
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        i = start
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
                        value = json.loads(candidate, strict=False)
                    except Exception:  # noqa: BLE001
                        value = None
                    if isinstance(value, dict):
                        found.append(value)
                    break
        # Resume after the balanced object so JSON nested INSIDE a manifest
        # (e.g. an item of the "files" array) is not collected as a sibling.
        start = text.find("{", i + 1) if i + 1 < len(text) else -1
    return found


def _filter_manifest_ops(ops: list[dict]) -> list[dict]:
    """Drop instruction-example echoes before they reach the filesystem.

    A manifest whose every file write is placeholder content ("...") creates
    NOTHING — not even the example's folders (``{"folders": ["docs"],
    "files": [{"path": "docs/a.md", "content": "..."}]}`` must not materialize
    a phantom ``docs/a.md`` or an empty ``docs/``). Mixed manifests keep their
    real files; only the placeholder writes are dropped.
    """
    file_ops = [o for o in ops if o.get("action") == "write"]
    if file_ops and all(_is_placeholder_content(o.get("content")) for o in file_ops):
        return []
    return [o for o in ops if not _is_placeholder_content(o.get("content"))]


def is_safe_manifest_path(raw: str) -> bool:
    """True when ``raw`` does not attempt parent traversal or invalid escaping."""
    if not isinstance(raw, str) or not raw.strip() or raw.strip() in (".", "/", "\\"):
        return False
    clean = raw.strip().strip("'\"").replace("\\", "/")
    if clean.startswith("/") and len(clean) > 2 and clean[2] == ":":
        clean = clean[1:]
    parts = [p for p in clean.split("/") if p not in ("", ".")]
    if not parts or any(p in ("..",) for p in parts):
        return False
    if any(p in (".", "..") for p in clean.split("/")):
        return False
    return True


def _payload_to_ops(payload: dict) -> list[dict]:
    """Convert a parsed JSON object into normalized file ops."""
    ops: list[dict] = []
    # Single top-level action: {"action": "edit"|"rename"|"delete"|"write", ...}
    if payload.get("action") or payload.get("operation"):
        op = _clean_op_item(payload)
        if op:
            ops.append(op)
            return ops
    if payload.get("rename") and isinstance(payload["rename"], dict):
        op = _clean_op_item({"action": "rename", **payload["rename"]})
        if op:
            ops.append(op)
            return ops
    if payload.get("delete") and isinstance(payload["delete"], str):
        op = _clean_op_item({"action": "delete", "path": payload["delete"]})
        if op:
            ops.append(op)
            return ops

    # explicit folder creation — the backend makes real directories.
    for key in ("folders", "dirs", "directories"):
        value = payload.get(key)
        if isinstance(value, list):
            for rel in value:
                rel = str(rel)
                if is_safe_manifest_path(rel):
                    ops.append({"action": "create_folder", "path": rel})
                else:
                    ops.append({"action": "create_folder", "path": rel, "error": "unsafe path"})
    for key in ("files", "create", "writes", "operations", "ops", "artifacts", "documents"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                op = _clean_op_item(item)
                if op:
                    ops.append(op)
        elif isinstance(value, dict):
            for rel, content in value.items():
                if isinstance(content, (dict, list)):
                    content = json.dumps(content)
                if is_safe_manifest_path(str(rel)):
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
            if rel in ("action", "operation", "intent", "reasoning", "old", "new", "path", "file"):
                continue
            if isinstance(value, str):
                if is_safe_manifest_path(str(rel)):
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
    return _filter_manifest_ops(_dedupe_ops(_payload_to_ops(payload)))


def _is_code_or_markup(path: str, body: str) -> bool:
    """Check whether raw body matches the expected code/markup structure of path."""
    if not body or not body.strip():
        return False
    if _is_placeholder_content(body):
        return False
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext in ("md", "txt", "rst", "adoc"):
        return True
    if ext == "py":
        import ast
        try:
            ast.parse(body)
            return True
        except SyntaxError:
            py_markers = ("def ", "class ", "import ", "from ", "return ", "print(", "if __name__", "@")
            return any(m in body for m in py_markers)
    if ext in ("json", "jsonc"):
        import json
        try:
            json.loads(body)
            return True
        except Exception:
            return "{" in body and "}" in body
    if ext in ("html", "htm", "xml", "svg"):
        return "<" in body and ">" in body
    if ext in ("js", "ts", "jsx", "tsx", "mjs", "cjs"):
        js_markers = ("function", "const ", "let ", "var ", "=>", "import ", "export ", "console.", "{", "}")
        return any(m in body for m in js_markers)
    if ext in ("css", "scss", "sass", "less"):
        return "{" in body and "}" in body
    if ext in ("sh", "bash", "zsh"):
        sh_markers = ("#!/", "echo ", "export ", "if [", "then", "fi", "cd ", "mkdir ")
        return any(m in body for m in sh_markers)
    if ext in ("yaml", "yml", "toml", "ini", "cfg", "env"):
        return ":" in body or "=" in body
    if ext in ("sql",):
        sql_markers = ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "FROM", "WHERE")
        return any(m in body.upper() for m in sql_markers)
    return True


def _raw_code_to_ops(text: str, hint: str) -> list[dict]:
    """Fallback when no JSON and no markdown code fences are present.

    If the model output is raw code or script content with an explicit file header
    or a non-empty hint, extract it into a write op.
    """
    if not text or not text.strip():
        return []
    cleaned = text.strip()
    if _is_placeholder_content(cleaned):
        return []

    # Check for header line naming the file
    first_line, _, rest = cleaned.partition("\n")
    head = first_line.strip()
    m = _PATH_HEADER_RE.match(head) or _PATH_RE.match(head)
    detected_path = ""
    body = cleaned
    if m:
        cand = m.group(1).strip()
        if _is_valid_extracted_path(cand):
            detected_path = cand
            body = rest.strip("\n")

    if not detected_path and hint:
        sanitized = _sanitize_hint(hint)
        if sanitized and _is_valid_extracted_path(sanitized):
            detected_path = sanitized

    if detected_path and not _is_placeholder_content(body):
        if not _is_code_or_markup(detected_path, body):
            return []
        if not is_safe_manifest_path(detected_path):
            detected_path = "output.txt"
        return [{"action": "write", "path": detected_path, "content": body}]

    return []


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

    # Phase XVI — prefer the LAST JSON object: models commonly open their
    # answer with an echo of the instruction example and put the real
    # manifest at the end. Candidate echoes are rejected by
    # ``_filter_manifest_ops``; earlier objects are tried only when the last
    # one produced nothing.
    for payload in reversed(_json_objects(text)):
        ops = _filter_manifest_ops(_dedupe_ops(_payload_to_ops(payload)))
        if ops:
            return ops

    # fallback: fenced code blocks -> one write op per block (Phase 7 fix).
    fallback = _fences_to_ops(text, hint)
    if fallback:
        return fallback

    # fallback: raw code / script output without fences or JSON
    raw_ops = _raw_code_to_ops(text, hint)
    if raw_ops:
        return raw_ops

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
        safe = is_safe_manifest_path(src) and is_safe_manifest_path(dst)
        return {"action": "rename", "path": src, "to": dst, **({"error": "unsafe path"} if not safe else {})}
    if action in ("delete", "remove"):
        return {"action": "delete", "path": str(raw_path)}
    if action in ("edit", "patch", "replace"):
        old = item.get("old") or item.get("find") or item.get("from")
        new = item.get("new") or item.get("replace") or item.get("to")
        safe = is_safe_manifest_path(str(raw_path))
        return {
            "action": "edit",
            "path": str(raw_path),
            "old": str(old or ""),
            "new": str(new or ""),
            **({"error": "unsafe path"} if not safe else {}),
        }
    if action in ("create_pdf", "create_docx", "create_pptx", "create_xlsx", "create_csv", "create_artifact", "create_doc", "create_presentation", "create_sheet"):
        content_val = item.get("content") or item.get("text") or item.get("body") or item.get("slides") or item.get("sections") or item.get("rows")
        if isinstance(content_val, (dict, list)):
            content_val = json.dumps(content_val)
        safe = is_safe_manifest_path(str(raw_path))
        return {
            "action": "write",
            "path": str(raw_path),
            "content": str(content_val or ""),
            **({"error": "unsafe path"} if not safe else {}),
        }
    if raw_path:
        safe = is_safe_manifest_path(str(raw_path))
        content_str = content
        if isinstance(content_str, (dict, list)):
            content_str = json.dumps(content_str)
        return {
            "action": "write",
            "path": str(raw_path),
            "content": str(content_str or ""),
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
    "requirements": "requirements.txt", "pip": "requirements.txt",
    "dockerfile": "Dockerfile", "docker": "Dockerfile",
}

#: header line inside a fence naming the destination file, e.g.
#: ``# file: index.html`` / ``// style.css`` / ``<!-- FILE = app.js -->`` /
#: ``index.html:``. The captured path must be the only real content of the
#: header line (code that merely starts with a comment still matches the
#: extension requirement below).
_PATH_HEADER_RE = re.compile(
    r"^\s*(?:#|//|/\*|<!--|--|%;|['\"])\s*(?:FILE|PATH|filepath)?\s*[:=]?\s*"
    r"([A-Za-z0-9_./:\-]+\.\w+)\s*(?:\*/|-->)?\s*$",
    re.IGNORECASE,
)

_PRECEDING_PATH_RES = [
    # Explicit prefix: File: `path/to/file.ext` or File: path/to/file.ext
    re.compile(
        r"(?:file|path|filename|filepath|destination)\s*[:=]\s*[`*\"']?([A-Za-z0-9_./:\-]+\.[A-Za-z0-9]+)[`*\"']?",
        re.IGNORECASE,
    ),
    # Markdown heading: ### `path/to/file.ext` or ### path/to/file.ext
    re.compile(
        r"^#+\s*[`*\"']?([A-Za-z0-9_./:\-]+\.[A-Za-z0-9]+)[`*\"':]?",
        re.IGNORECASE,
    ),
    # List item: - `path/to/file.ext` or 1. `path/to/file.ext`
    re.compile(
        r"^(?:[-*+]|\d+[\.\)])\s*[`*\"']?([A-Za-z0-9_./:\-]+\.[A-Za-z0-9]+)[`*\"':]?",
        re.IGNORECASE,
    ),
    # Standalone path line or path with colon: `path/to/file.ext` or requirements.txt:
    re.compile(
        r"^\s*[`*\"']?([A-Za-z0-9_./:\-]+\.[A-Za-z0-9]+)[`*\"':]?\s*$",
        re.IGNORECASE,
    ),
    # Backticked or bold anywhere on the line
    re.compile(
        r"[`*]([A-Za-z0-9_./:\-]+\.[A-Za-z0-9]+)[`*]",
        re.IGNORECASE,
    ),
]


def _is_valid_extracted_path(cand: str) -> bool:
    if not cand or not is_safe_manifest_path(cand):
        return False
    cand_lower = cand.lower().strip()
    if cand_lower in ("e.g.", "i.e.", "etc.", "version.1"):
        return False
    if cand_lower in ("dockerfile", "makefile", "procfile", ".gitignore", ".env", ".env.example"):
        return True
    if "." not in cand:
        return False
    ext = cand.rsplit(".", 1)[-1].lower()
    if len(ext) < 1 or len(ext) > 12:
        return False
    if ext.isdigit():
        return False
    return True


def _extract_preceding_path(preceding_text: str) -> str:
    """Find a filename or file path in the lines immediately preceding a code fence."""
    lines = [line.strip() for line in preceding_text.splitlines() if line.strip()]
    if not lines:
        return ""
    candidates_to_check = lines[-5:][::-1]
    for line in candidates_to_check:
        if "```" in line:
            break
        for rx in _PRECEDING_PATH_RES:
            match = rx.search(line)
            if match:
                cand = match.group(1).strip().strip("`*\"':")
                if _is_valid_extracted_path(cand):
                    return cand
    return ""


def _sanitize_hint(hint: str, lang: str = "") -> str:
    """Normalize a planner hint into a valid relative file path."""
    if not hint or not isinstance(hint, str):
        return ""
    h = hint.strip().replace("\\", "/")
    if not is_safe_manifest_path(h):
        return ""
    # If hint already has a valid file extension (e.g. "src/app.py", "README.md")
    if "." in h and not h.endswith("."):
        ext = h.rsplit(".", 1)[-1]
        if not ext.isdigit() and 1 <= len(ext) <= 10:
            return h
    clean_lang = _LANG_EXT.get(lang.strip().lower(), "") or "txt"
    if h.lower() in _LANG_EXT:
        ext = _LANG_EXT[h.lower()]
        return f"README.{ext}" if ext == "md" else f"main.{ext}"
    if h.endswith("_") or h.endswith("/"):
        stem = h.rstrip("/_")
        target_ext = clean_lang if clean_lang != "txt" else "py"
        return f"test/test_{stem or 'app'}.{target_ext}" if "test" in h else f"{h}main.{target_ext}"
    return f"{h}.{clean_lang}"


def _fences_to_ops(text: str, hint: str) -> list[dict]:
    """Convert every fenced code block into a write op.

    Order is preserved. A block naming its destination in the first line
    (``file: path`` in any comment style, or a bare ``path.ext:`` line) keeps
    that path; otherwise preceding markdown context, the planner hint, or the
    language's default name is used. Colliding names get a numeric suffix.
    """
    if not text:
        return []
    matches = list(_FENCE_RE.finditer(text))
    if not matches:
        # Check for unclosed code fence
        unclosed = re.search(r"```(?P<lang>[a-zA-Z0-9_+-]*)[ \t]*\n?(?P<body>.*)$", text, re.DOTALL)
        if unclosed:
            lang = unclosed.group("lang") or ""
            body = unclosed.group("body") or ""
            raw = body.rstrip("\n")
            if raw and raw.strip():
                preceding_text = text[: unclosed.start()]
                preceding_path = _extract_preceding_path(preceding_text)
                path, body_content = _fence_destination(
                    raw, lang, hint, preceding_path=preceding_path
                )
                if not _is_placeholder_content(body_content):
                    if not is_safe_manifest_path(path):
                        path = "output.txt"
                    return [{"action": "write", "path": path, "content": body_content}]
        return []
    used: set[str] = set()
    ops: list[dict] = []
    for m in matches:
        lang = m.group("lang") or ""
        body = m.group("body") or ""
        raw = body.rstrip("\n")
        if not raw or not raw.strip():
            continue
        preceding_text = text[: m.start()]
        preceding_path = _extract_preceding_path(preceding_text)
        path, body_content = _fence_destination(
            raw, lang, hint, preceding_path=preceding_path
        )
        if _is_placeholder_content(body_content):
            continue
        path = _unique_path(path, used)
        if not is_safe_manifest_path(path):
            path = "output.txt"
        ops.append({"action": "write", "path": path, "content": body_content})
    return ops


def _fence_destination(
    body: str, lang: str, hint: str, preceding_path: str = ""
) -> tuple[str, str]:
    """Return (path, body-without-header) for one fenced block."""
    first_line, _, rest = body.partition("\n")
    head = first_line.strip()
    m = _PATH_HEADER_RE.match(head) or _PATH_RE.match(head)
    if m:
        cand = m.group(1).strip()
        if _is_valid_extracted_path(cand):
            return cand, rest.strip("\n")
    if preceding_path and _is_valid_extracted_path(preceding_path):
        return preceding_path, body
    if hint:
        sanitized = _sanitize_hint(hint, lang)
        if sanitized:
            return sanitized, body
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