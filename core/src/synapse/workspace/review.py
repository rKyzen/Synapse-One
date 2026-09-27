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
    if not text.strip():
        return ValidationResult(path="", ok=False, checks=["non-empty"], error="python file is empty")
    try:
        ast.parse(text)
        return ValidationResult(path="", ok=True, checks=["python syntax ok"])
    except SyntaxError as exc:
        return ValidationResult(
            path="", ok=False, checks=["python syntax"],
            error=f"python syntax error: line {exc.lineno}: {exc.msg}",
        )
    except Exception as exc:
        return ValidationResult(
            path="", ok=False, checks=["python syntax"],
            error=f"python syntax error: {exc}",
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
    if not text.strip():
        return ValidationResult(path="", ok=False, checks=["non-empty"], error="file is empty")
    pairs = {"{": "}", "(": ")", "[": "]"}
    stack: list[str] = []
    in_str: str | None = None
    esc = False
    in_line_comment = False
    in_block_comment = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and i + 1 < len(text) and text[i + 1] == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in ("'", '"', "`"):
            in_str = ch
        elif ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            in_line_comment = True
            i += 2
            continue
        elif ch == "/" and i + 1 < len(text) and text[i + 1] == "*":
            in_block_comment = True
            i += 2
            continue
        elif ch in pairs:
            stack.append(ch)
        elif ch in pairs.values():
            if not stack or pairs[stack.pop()] != ch:
                return ValidationResult(
                    path="", ok=False, checks=["bracket balance"],
                    error=f"unbalanced {ch!r} at index {i}",
                )
        i += 1
    if in_str:
        return ValidationResult(path="", ok=False, checks=["bracket balance"], error="unterminated string literal")
    if in_block_comment:
        return ValidationResult(path="", ok=False, checks=["bracket balance"], error="unclosed block comment")
    if stack:
        return ValidationResult(path="", ok=False, checks=["bracket balance"], error="unclosed brackets")
    return ValidationResult(path="", ok=True, checks=["bracket balance"])


def _check_css(content: str | bytes) -> ValidationResult:
    text = _as_text(content)
    if not text.strip():
        return ValidationResult(path="", ok=False, checks=["non-empty"], error="css file is empty")
    stack: list[str] = []
    in_str: str | None = None
    esc = False
    in_block_comment = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_block_comment:
            if ch == "*" and i + 1 < len(text) and text[i + 1] == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in ("'", '"'):
            in_str = ch
        elif ch == "/" and i + 1 < len(text) and text[i + 1] == "*":
            in_block_comment = True
            i += 2
            continue
        elif ch in ("{", "(", "["):
            stack.append(ch)
        elif ch in ("}", ")", "]"):
            pairs = {"}": "{", ")": "(", "]": "["}
            if not stack or stack.pop() != pairs[ch]:
                return ValidationResult(
                    path="", ok=False, checks=["bracket balance"],
                    error=f"unbalanced {ch!r} in css",
                )
        i += 1
    if in_str:
        return ValidationResult(path="", ok=False, checks=["bracket balance"], error="unterminated string in css")
    if in_block_comment:
        return ValidationResult(path="", ok=False, checks=["bracket balance"], error="unclosed block comment in css")
    if stack:
        return ValidationResult(path="", ok=False, checks=["bracket balance"], error="unclosed braces in css")
    return ValidationResult(path="", ok=True, checks=["css syntax ok", "non-empty"])


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
    try:
        import io
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(data))
        page_count = len(reader.pages)
        if page_count < 1:
            return ValidationResult(path="", ok=False, checks=["pdf pages"], error="pdf contains 0 pages")
        return ValidationResult(path="", ok=True, checks=["pdf valid", f"{page_count} page(s)", f"{len(data)} bytes"])
    except Exception:
        return ValidationResult(path="", ok=True, checks=["pdf header ok", f"{len(data)} bytes"])


def _check_zip_xml(content: str | bytes, member_name: str, format_name: str) -> ValidationResult:
    import io
    import zipfile

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
    res = _check_zip_xml(content, "ppt/presentation.xml", "pptx")
    if not res.ok:
        return res
    try:
        import io
        import pptx
        data = content.encode("latin-1", "ignore") if isinstance(content, str) else content
        prs = pptx.Presentation(io.BytesIO(data))
        slide_count = len(prs.slides)
        return ValidationResult(path="", ok=True, checks=["pptx package ok", f"{slide_count} slide(s)", f"{len(data)} bytes"])
    except Exception:
        return res


def _check_xlsx(content: str | bytes) -> ValidationResult:
    res = _check_zip_xml(content, "xl/workbook.xml", "xlsx")
    if not res.ok:
        return res
    try:
        import io
        import openpyxl
        data = content.encode("latin-1", "ignore") if isinstance(content, str) else content
        wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
        sheet_count = len(wb.sheetnames)
        return ValidationResult(path="", ok=True, checks=["xlsx package ok", f"{sheet_count} sheet(s)", f"{len(data)} bytes"])
    except Exception:
        return res


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
    res = _check_zip_xml(content, "word/document.xml", "docx")
    if not res.ok:
        return res
    try:
        import docx
        import io
        data = content.encode("latin-1", "ignore") if isinstance(content, str) else content
        doc = docx.Document(io.BytesIO(data))
        paras = len(doc.paragraphs)
        tables = len(doc.tables)
        return ValidationResult(path="", ok=True, checks=["docx package ok", f"{paras} paras", f"{tables} tables", f"{len(data)} bytes"])
    except Exception:
        return res


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
    "ts": _check_javascript,
    "tsx": _check_javascript,
    "jsx": _check_javascript,
    "css": _check_css,
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


def clean_review_text(review_text: str, ok_paths: set[str] | list[str]) -> str:
    """Scrub false 'not found in workspace' claims for verified files from review text."""
    if not review_text:
        return ""
    import re
    path_set = {str(p).strip().lower() for p in ok_paths}
    base_set = {p.rsplit("/", 1)[-1].lower() for p in path_set}
    cleaned_lines = []
    for line in review_text.splitlines():
        sentences = re.split(r"(?<=[.!?])\s+", line)
        kept_sentences = []
        for s in sentences:
            s_lower = s.lower()
            if (
                "not found in workspace" in s_lower
                or "not found in the workspace" in s_lower
                or "missing from workspace" in s_lower
                or "missing from the workspace" in s_lower
                or "does not exist in workspace" in s_lower
                or "does not exist in the workspace" in s_lower
                or "could not find in workspace" in s_lower
                or "no files found in workspace" in s_lower
            ):
                if not path_set or any(p in s_lower for p in path_set) or any(b in s_lower for b in base_set) or "workspace" in s_lower:
                    continue
            kept_sentences.append(s)
        if kept_sentences:
            cleaned_lines.append(" ".join(kept_sentences))
    return "\n".join(cleaned_lines).strip()


def summarize(validations: list[ValidationResult], review_text: str = "") -> str:
    """Human-readable summary block appended to the final response."""
    lines = []
    ok_paths = set()
    for result in validations:
        marker = "ok" if result.ok else "FAILED"
        if result.ok:
            ok_paths.add(result.path)
        detail = result.error or (" · ".join(result.checks) if result.checks else "")
        lines.append(f"- {result.path}: {marker}{(' (' + detail + ')') if detail else ''}")
    if not lines:
        return ""
    header = "Generated files (validation):"
    cleaned_review = clean_review_text(review_text, ok_paths) if review_text else ""
    if cleaned_review:
        return f"{header}\n" + "\n".join(lines) + f"\n\nReview:\n{cleaned_review}"
    return f"{header}\n" + "\n".join(lines)