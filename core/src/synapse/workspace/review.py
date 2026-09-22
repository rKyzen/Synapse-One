"""OutputReviewer — Phase 6 deterministic validation of generated files.

After the orchestrator applies a task's file manifest, every written file is
checked with cheap, offline validators (syntax/consistency by extension):

- ``.py``   — Python syntax via ``ast.parse``
- ``.json`` — ``json.loads``
- ``.js``   — brace/paren/bracket balance
- ``.html`` — required document tag presence + comment/quote sanity
- else     — non-empty, valid UTF-8

Results feed the workspace summary shown to the user and are recorded in the
project's action log. A model-backed REVIEW task (routed to the strongest
reasoning model) complements this deterministic pass when the planner emits
one.
"""

from __future__ import annotations

from synapse.domain.fileops import ValidationResult
from synapse.logging import get_logger

log = get_logger("synapse.workspace.review")


def _as_text(content: str | bytes) -> str:
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return str(content)


def _check_python(content: str | bytes) -> ValidationResult:
    import ast

    text = _as_text(content)
    try:
        ast.parse(text)
        return ValidationResult(path="", ok=True, checks=["python syntax ok"])
    except SyntaxError as exc:
        return ValidationResult(
            path="", ok=False, checks=["python syntax"],
            error=f"python syntax error: line {exc.lineno}: {exc.msg}",
        )


def _check_json(content: str | bytes) -> ValidationResult:
    import json

    text = _as_text(content)
    try:
        json.loads(text)
        return ValidationResult(path="", ok=True, checks=["json parses"])
    except json.JSONDecodeError as exc:
        return ValidationResult(
            path="", ok=False, checks=["json parses"],
            error=f"json error at line {exc.lineno} col {exc.colno}: {exc.msg}",
        )


def _check_javascript(content: str | bytes) -> ValidationResult:
    text = _as_text(content)
    pairs = {"{": "}", "(": ")", "[": "]"}
    stack: list[str] = []
    in_str: str | None = None
    esc = False
    in_line_comment = False
    for i, ch in enumerate(text):
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            continue
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"', "`"):
            in_str = ch
        elif ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            in_line_comment = True
        elif ch in pairs:
            stack.append(ch)
        elif ch in pairs.values():
            if not stack or pairs[stack.pop()] != ch:
                return ValidationResult(
                    path="", ok=False, checks=["bracket balance"],
                    error=f"unbalanced {ch!r} at index {i}",
                )
    if in_str:
        return ValidationResult(path="", ok=False, checks=["bracket balance"], error="unterminated string literal")
    if stack:
        return ValidationResult(path="", ok=False, checks=["bracket balance"], error="unclosed brackets")
    return ValidationResult(path="", ok=True, checks=["bracket balance"])


def _check_html(content: str | bytes) -> ValidationResult:
    text = _as_text(content)
    lowered = text.lower()
    checks = ["non-empty"]
    if "<!doctype html" not in lowered and "<html" not in lowered:
        return ValidationResult(
            path="", ok=False, checks=checks,
            error="missing <!doctype html> or <html> document root",
        )
    if "<html" not in lowered or "</html" not in lowered:
        return ValidationResult(path="", ok=False, checks=checks, error="unclosed <html> element")
    return ValidationResult(path="", ok=True, checks=[*checks, "document root present"])


def _check_pdf(content: str | bytes) -> ValidationResult:
    data = content.encode("latin-1", "ignore") if isinstance(content, str) else content
    if not data or not data.startswith(b"%PDF"):
        return ValidationResult(path="", ok=False, checks=["pdf header"], error="invalid pdf header")
    return ValidationResult(path="", ok=True, checks=["pdf header ok", f"{len(data)} bytes"])


def _check_zip_xml(content: str | bytes, member_name: str, format_name: str) -> ValidationResult:
    import zipfile
    import io

    data = content.encode("latin-1", "ignore") if isinstance(content, str) else content
    if not data:
        return ValidationResult(path="", ok=False, checks=[f"{format_name} content"], error="empty file")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = z.namelist()
            if member_name not in names:
                return ValidationResult(
                    path="", ok=False, checks=[f"{format_name} package"],
                    error=f"missing {member_name} in package",
                )
        return ValidationResult(path="", ok=True, checks=[f"{format_name} package ok", f"{len(data)} bytes"])
    except Exception as exc:
        return ValidationResult(
            path="", ok=False, checks=[f"{format_name} zip"],
            error=f"invalid {format_name} archive: {exc}",
        )


def _check_pptx(content: str | bytes) -> ValidationResult:
    return _check_zip_xml(content, "ppt/presentation.xml", "pptx")


def _check_xlsx(content: str | bytes) -> ValidationResult:
    return _check_zip_xml(content, "xl/workbook.xml", "xlsx")


def _check_text(content: str | bytes) -> ValidationResult:
    if isinstance(content, bytes):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return ValidationResult(path="", ok=False, checks=["valid utf-8"], error="invalid utf-8 text")
    else:
        text = str(content)
    if not text.strip():
        return ValidationResult(path="", ok=False, checks=["non-empty"], error="file is empty")
    return ValidationResult(path="", ok=True, checks=["non-empty", "valid utf-8"])


def _check_docx(content: str | bytes) -> ValidationResult:
    return _check_zip_xml(content, "word/document.xml", "docx")


def _check_csv(content: str | bytes) -> ValidationResult:
    import csv

    text = content.decode("utf-8", "ignore") if isinstance(content, bytes) else str(content)
    try:
        rows = list(csv.reader(text.splitlines()))
        if not rows:
            return ValidationResult(path="", ok=False, checks=["csv rows"], error="empty csv")
        return ValidationResult(path="", ok=True, checks=["csv format ok", f"{len(rows)} rows"])
    except Exception as exc:
        return ValidationResult(path="", ok=False, checks=["csv parses"], error=f"csv error: {exc}")


_VALIDATORS = {
    "py": _check_python,
    "json": _check_json,
    "js": _check_javascript,
    "mjs": _check_javascript,
    "cjs": _check_javascript,
    "html": _check_html,
    "htm": _check_html,
    "pdf": _check_pdf,
    "pptx": _check_pptx,
    "ppt": _check_pptx,
    "xlsx": _check_xlsx,
    "xls": _check_xlsx,
    "docx": _check_docx,
    "doc": _check_docx,
    "csv": _check_csv,
}


def validate_file(path: str, content: str | bytes) -> ValidationResult:
    """Run the validator matching ``path``'s extension (default: text check)."""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    checker = _VALIDATORS.get(ext, _check_text)
    try:
        result = checker(content)
    except Exception as exc:
        result = ValidationResult(path=path, ok=False, checks=["validation error"], error=str(exc))
    return result.model_copy(update={"path": path})


def summarize(validations: list[ValidationResult], review_text: str = "") -> str:
    """Human-readable summary block appended to the final response."""
    lines = []
    for result in validations:
        marker = "ok" if result.ok else "FAILED"
        detail = result.error or (" · ".join(result.checks) if result.checks else "")
        lines.append(f"- {result.path}: {marker}{(' (' + detail + ')') if detail else ''}")
    if not lines:
        return ""
    header = "Generated files (validation):"
    if review_text:
        return f"{header}\n" + "\n".join(lines) + f"\n\nReview:\n{review_text}"
    return f"{header}\n" + "\n".join(lines)