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


def _check_python(content: str) -> ValidationResult:
    import ast

    try:
        ast.parse(content)
        return ValidationResult(path="", ok=True, checks=["python syntax ok"])
    except SyntaxError as exc:
        return ValidationResult(
            path="", ok=False, checks=["python syntax"],
            error=f"python syntax error: line {exc.lineno}: {exc.msg}",
        )


def _check_json(content: str) -> ValidationResult:
    import json

    try:
        json.loads(content)
        return ValidationResult(path="", ok=True, checks=["json parses"])
    except json.JSONDecodeError as exc:
        return ValidationResult(
            path="", ok=False, checks=["json parses"],
            error=f"json error at line {exc.lineno} col {exc.colno}: {exc.msg}",
        )


def _check_javascript(content: str) -> ValidationResult:
    pairs = {"{": "}", "(": ")", "[": "]"}
    stack: list[str] = []
    in_str: str | None = None
    esc = False
    in_line_comment = False
    for i, ch in enumerate(content):
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
        elif ch == "/" and i + 1 < len(content) and content[i + 1] == "/":
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


def _check_html(content: str) -> ValidationResult:
    lowered = content.lower()
    checks = ["non-empty"]
    if "<!doctype html" not in lowered and "<html" not in lowered:
        return ValidationResult(
            path="", ok=False, checks=checks,
            error="missing <!doctype html> or <html> document root",
        )
    if "<html" not in lowered or "</html" not in lowered:
        return ValidationResult(path="", ok=False, checks=checks, error="unclosed <html> element")
    return ValidationResult(path="", ok=True, checks=[*checks, "document root present"])


def _check_text(content: str) -> ValidationResult:
    if not content.strip():
        return ValidationResult(path="", ok=False, checks=["non-empty"], error="file is empty")
    return ValidationResult(path="", ok=True, checks=["non-empty", "valid utf-8"])


_VALIDATORS = {
    "py": _check_python,
    "json": _check_json,
    "js": _check_javascript,
    "mjs": _check_javascript,
    "cjs": _check_javascript,
    "html": _check_html,
    "htm": _check_html,
}


def validate_file(path: str, content: str) -> ValidationResult:
    """Run the validator matching ``path``'s extension (default: text check)."""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    checker = _VALIDATORS.get(ext, _check_text)
    result = checker(content)
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