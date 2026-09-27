"""ActionEngine — the backend filesystem executor (Phase 7).

The engine is the ONLY code that touches the user's disk once a request is
classified as a file kind. The LLM never runs filesystem operations: it only
generates a structured manifest of instructions. The engine:

1. **plans** — expands a manifest into concrete operations (folders first,
   plus a canonical ``assets/`` directory for project generation);
2. **executes** — every operation via the safe ``FileOperator``, refusing
   deletes unless the user explicitly confirmed them;
3. **verifies** — checks each written file actually exists and passes the
   deterministic validators;
4. **summarizes** — returns a short human summary instead of dumping code
   into the chat.

``run_workspace_ops`` is the model-free shortcut for explicit instructions
(list / search / read / rename / move / delete / create folder): the backend
answers directly and no model is consulted.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Callable

from synapse.domain.enums import RequestKind
from synapse.domain.fileops import FileAction, ValidationResult
from synapse.logging import get_logger
from synapse.workspace.operator import WorkspaceSafetyError, is_safe_relative_path, normalize_workspace_rel
from synapse.workspace.review import summarize as summarize_validations
from synapse.workspace.review import validate_file

from synapse.workspace.artifacts import build_artifact_content

log = get_logger("synapse.actions.engine")


class ActionEngine:
    """Plans and executes filesystem actions on a project's workspace."""

    def __init__(self, operator) -> None:
        self._op = operator

    # -- tools: backend filesystem primitives --------------------------------

    def create_folder(self, rel: str) -> None:
        path = self._op.path_for(rel)
        path.mkdir(parents=True, exist_ok=True)

    def exists(self, rel: str) -> bool:
        return self._op.exists(rel)

    def read(self, rel: str) -> str | None:
        return self._op.read(rel)

    def write(self, rel: str, content: str | bytes) -> dict:
        data, _ = build_artifact_content(rel, content)
        return self._op.write(rel, data)

    def edit(self, rel: str, old: str, new: str) -> dict:
        """Targeted in-place edit; if old snippet match fails, falls back to full-content replace."""
        content = self._op.read(rel)
        if content is None:
            return {"ok": False, "error": f"file not found: {rel}"}
        target_old = old
        if target_old and target_old in content:
            updated = content.replace(target_old, new, 1) if content.count(target_old) == 1 else content.replace(target_old, new)
        elif target_old and target_old.strip() and target_old.strip() in content:
            updated = content.replace(target_old.strip(), new, 1)
        elif target_old and target_old.replace("\r\n", "\n") in content.replace("\r\n", "\n"):
            normalized_content = content.replace("\r\n", "\n")
            updated = normalized_content.replace(target_old.replace("\r\n", "\n"), new, 1)
        else:
            is_full_code = bool(
                new and (
                    "\n" in new
                    or len(new) > 40
                    or any(k in new for k in ("def ", "class ", "import ", "<html", "function", "var ", "let ", "const ", "{", "};"))
                )
            )
            if is_full_code:
                updated = new
            else:
                return {"ok": False, "error": "old text not found in file"}

        self._op.write(rel, updated)
        post_content = self._op.read(rel)
        if post_content is None or (new and new not in post_content and post_content != updated):
            return {"ok": False, "error": "edit not reflected on disk (verification failed)"}
        return {"ok": True, "bytes": len(post_content.encode())}

    def move(self, src: str, dst: str) -> dict:
        return self._op.rename(src, dst)

    def delete(self, rel: str, *, confirmed: bool = False) -> bool:
        """Delete a file. ``confirmed`` is required — silent deletes are refused."""
        if not confirmed:
            raise PermissionError(
                f"deletion requires confirmation: '{rel}' (say 'delete {rel}' to confirm)"
            )
        return self._op.delete(rel)

    def search(self, pattern: str, *, rel: str = "", max_results: int = 50) -> list[dict]:
        """Regex search across workspace files. Returns [{path, line, text}]."""
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE)
        hits: list[dict] = []
        for entry in self._op.list_tree(rel):
            content = self._op.read(entry["path"]) or ""
            for lineno, line in enumerate(content.splitlines(), 1):
                if rx.search(line):
                    hits.append({"path": entry["path"], "line": lineno, "text": line.strip()[:200]})
                    if len(hits) >= max_results:
                        return hits
        return hits

    def list_tree(self) -> list[dict]:
        return self._op.list_tree()

    # -- planning -------------------------------------------------------------

    @staticmethod
    def plan_project_ops(kind: RequestKind | None, ops: list[dict]) -> list[dict]:
        """Expand a manifest into the full action list.

        Project generation guarantees a real folder structure: an ``assets/``
        directory is created when the model did not already request folders.
        """
        if kind is not RequestKind.PROJECT_GENERATION or not ops:
            return ops
        has_folder = any(o.get("action") == "create_folder" for o in ops)
        if has_folder:
            return ops
        return [{"action": "create_folder", "path": "assets"}, *ops]

    # -- execution --------------------------------------------------------------

    def _verified(self, perform: Callable[[], object], check: Callable[[], bool], fail: str):
        """Run a mutation, verify the real filesystem state, retry once.

        ``perform`` must be safe to run twice (idempotent). When the check
        still fails after the retry, ``OSError(fail)`` is raised so the caller
        records a failed action — a silent non-write can never pass as "ok".
        """
        result = perform()
        if check():
            return result
        result = perform()
        if not check():
            raise OSError(fail)
        return result

    def _edit_verified(self, path: str, old: str, new: str) -> bool:
        """True when the edit landed: file exists and contains new content."""
        content = self._op.read(path)
        if content is None:
            return False
        if new and new not in content:
            return False
        return True

    def _content_matches(self, path: str, expected: str | bytes) -> bool:
        if not self._op.exists(path):
            return False
        if isinstance(expected, bytes):
            try:
                return self._op.path_for(path).read_bytes() == expected
            except OSError:
                return False
        content = self._op.read(path)
        if content is None:
            return False
        return content == expected or content.replace("\r\n", "\n") == str(expected).replace("\r\n", "\n")

    def apply(
        self,
        ops: list[dict],
        *,
        confirm_delete: bool = False,
        on_step: Callable[[dict], None] | None = None,
    ) -> tuple[list[FileAction], list[ValidationResult]]:
        """Execute manifest ops against the workspace; verify every mutation.

        Never raises: violations go through failed FileActions so the rest of
        the request proceeds. Deletes require ``confirm_delete``.

        ``on_step`` (when given) receives ``{"kind", "path", ...}`` for the
        mutation about to execute, so a live UI can show file activity.
        """
        actions: list[FileAction] = []
        validations: list[ValidationResult] = []
        for op in ops:
            action = op.get("action", "write")
            raw_path = op.get("path", "")
            path = normalize_workspace_rel(raw_path, self._op.root) or raw_path
            raw_to = op.get("to", "")
            to_path = normalize_workspace_rel(raw_to, self._op.root) or raw_to
            if op.get("error") and not is_safe_relative_path(path):
                entry = FileAction(path=path, action=action, status="failed", error=op["error"])
                actions.append(entry)
                self._log_result(entry)
                continue
            step = {"action": action, "path": path}
            if action in ("create_folder", "mkdir", "folder"):
                step["kind"] = "create_folder"
            elif action in ("rename", "move"):
                step["kind"] = "rename"
                step["to"] = to_path
            elif action in ("delete", "remove"):
                step["kind"] = "delete"
            elif action in ("edit", "patch", "replace"):
                step["kind"] = "edit"
            elif action in ("search", "list", "read"):
                step["kind"] = "informational"
            else:
                step["kind"] = "write"
            if on_step and step["kind"] != "informational":
                on_step(step)
            try:
                if action in ("create_folder", "mkdir", "folder"):
                    self._verified(
                        lambda: self.create_folder(path),
                        lambda: self._op.path_for(path).is_dir(),
                        "folder missing after create (verification failed)",
                    )
                    entry = FileAction(path=path, action="created_folder", status="ok")
                elif action in ("rename", "move"):
                    self._verified(
                        lambda: self._op.rename(path, to_path),
                        lambda: self._op.exists(to_path) and not self._op.exists(path),
                        "rename not applied (verification failed)",
                    )
                    content = self._op.read(to_path) or ""
                    check = validate_file(to_path, content)
                    validations.append(check)
                    entry = FileAction(
                        path=to_path, action="renamed", bytes=len(content.encode()),
                        validated=check.ok, validation=check.error or " · ".join(check.checks),
                    )
                elif action in ("delete", "remove"):
                    if not confirm_delete and not op.get("confirmed"):
                        entry = FileAction(
                            path=path, action="deleted", status="failed",
                            error="requires confirmation — say 'delete <path>' to confirm",
                        )
                        actions.append(entry)
                        self._log_result(entry)
                        continue
                    ok = self._op.delete(path)
                    if ok:
                        self._verified(
                            lambda: self._op.delete(path),
                            lambda: not self._op.exists(path),
                            "file still exists after delete (verification failed)",
                        )
                    entry = FileAction(
                        path=path, action="deleted",
                        status="ok" if ok else "failed",
                        error="" if ok else "not found",
                    )
                elif action in ("edit", "patch", "replace"):
                    old, new = op.get("old", ""), op.get("new", "")
                    result = self.edit(path, old, new)
                    if result["ok"]:
                        self._verified(
                            lambda: self.edit(path, old, new),
                            lambda: self._edit_verified(path, old, new),
                            "edit not applied (verification failed)",
                        )
                        content = self._op.read(path) or ""
                        check = validate_file(path, content)
                        validations.append(check)
                        entry = FileAction(
                            path=path, action="modified", bytes=result["bytes"],
                            validated=check.ok, validation=check.error or " · ".join(check.checks),
                        )
                    else:
                        entry = FileAction(path=path, action="edit", status="failed", error=result["error"])
                elif action in ("search", "list", "read"):
                    # informational actions executed for context; nothing mutated.
                    entry = FileAction(path=path or op.get("pattern", ""), action=action, status="ok")
                else:  # write
                    content = op.get("content", "")
                    content_data, _ = build_artifact_content(path, content)
                    result = self._verified(
                        lambda: self._op.write(path, content_data),
                        lambda: self._content_matches(path, content_data),
                        "file missing or content mismatch after write (verification failed)",
                    )
                    check = validate_file(path, content_data)
                    validations.append(check)
                    entry = FileAction(
                        path=path, action=result["action"], bytes=result["bytes"],
                        validated=check.ok, validation=check.error or " · ".join(check.checks),
                    )
            except WorkspaceSafetyError as exc:
                entry = FileAction(path=path, action=action, status="failed", error=str(exc))
            except (OSError, ValueError, TypeError) as exc:  # noqa: BLE001 - surfaced as a failed action
                log.warning("action_failed", action=action, path=path, error=str(exc)[:200])
                entry = FileAction(path=path, action=action, status="failed", error=str(exc))
            actions.append(entry)
            self._log_result(entry)
        self.verify(actions)
        return actions, validations

    @staticmethod
    def _log_result(entry: FileAction) -> None:
        log.info(
            "action_executed",
            action=entry.action,
            path=entry.path,
            status=entry.status,
            error=entry.error,
            size_bytes=entry.bytes,
        )

    def verify(self, actions: list[FileAction]) -> None:
        """Verification step: every created/modified file must exist on disk."""
        for action in actions:
            if action.status != "ok":
                continue
            if action.action not in ("created", "modified", "created_folder", "renamed"):
                continue
            if not self._op.exists(action.path):
                action.status = "failed"
                action.error = "file missing after write (verification failed)"

    # -- model-free workspace operations ------------------------------------------

    def run_workspace_ops(
        self,
        ops: list[dict],
        *,
        on_step: Callable[[dict], None] | None = None,
    ) -> tuple[list[str], list[FileAction]]:
        """Execute explicit backend-only operations; returns (display_lines, actions)."""
        lines: list[str] = []
        actions: list[FileAction] = []
        for op in ops:
            action = op.get("action")
            raw_path = op.get("path", "")
            path = normalize_workspace_rel(raw_path, self._op.root) if raw_path else ""
            raw_to = op.get("to", "")
            to_path = normalize_workspace_rel(raw_to, self._op.root) if raw_to else ""
            if on_step:
                step = {"action": action, "path": path or op.get("pattern", "")}
                if action in ("rename", "move"):
                    step["to"] = to_path
                if action == "list":
                    step["kind"] = "list"
                elif action == "search":
                    step["kind"] = "search"
                elif action == "read":
                    step["kind"] = "read"
                else:
                    step["kind"] = action or "step"
                on_step(step)
            try:
                if action == "list":
                    entries = self._op.list_tree(path)
                    paths = [e["path"] for e in entries]
                    folder_desc = f" in {path}" if path else " in workspace"
                    actions.append(FileAction(path=path or ".", action="listed", status="ok"))
                    lines.append(f"Files{folder_desc} ({len(paths)}):")
                    if paths:
                        lines.extend(f"- {p}" for p in paths)
                    else:
                        lines.append("- (empty)")
                elif action == "search":
                    hits = self.search(op.get("pattern", ""), rel=path)
                    pattern = op.get("pattern", "")
                    actions.append(FileAction(path=pattern, action="searched", status="ok", error="" if hits else "no matches"))
                    lines.append(f"Search '{pattern}' ({len(hits)} match(es)):")
                    if hits:
                        lines.extend(f"- {h['path']}:{h['line']}  {h['text']}" for h in hits)
                    else:
                        lines.append("- no matches")
                elif action == "read":
                    content = self._op.read(path)
                    if content is None:
                        actions.append(FileAction(path=path, action="read", status="failed", error="not found"))
                        lines.append(f"read {path}: not found")
                    else:
                        actions.append(FileAction(path=path, action="read", status="ok"))
                        max_chars = 8000
                        if len(content) <= max_chars:
                            snippet = content
                        else:
                            snippet = content[:max_chars] + f"\n... (truncated, file is {len(content)} chars)"
                        lines.append(f"### {path}")
                        lines.append(snippet)
                elif action in ("rename", "move"):
                    self._verified(
                        lambda: self._op.rename(path, to_path),
                        lambda: self._op.exists(to_path) and not self._op.exists(path),
                        "rename not applied (verification failed)",
                    )
                    actions.append(FileAction(path=path, action="renamed", status="ok"))
                    lines.append(f"Renamed {path} -> {to_path}")
                elif action in ("delete", "remove"):
                    ok = self._op.delete(path)
                    if ok:
                        self._verified(
                            lambda: self._op.delete(path),
                            lambda: not self._op.exists(path),
                            "file still exists after delete (verification failed)",
                        )
                    actions.append(
                        FileAction(path=path, action="deleted", status="ok" if ok else "failed", error="" if ok else "not found")
                    )
                    lines.append(f"Deleted {path}" if ok else f"{path}: not found")
                elif action in ("create_folder", "mkdir"):
                    self._verified(
                        lambda: self.create_folder(path),
                        lambda: self._op.path_for(path).is_dir(),
                        "folder missing after create (verification failed)",
                    )
                    actions.append(FileAction(path=path, action="created_folder", status="ok"))
                    lines.append(f"Created folder {path}/")
            except WorkspaceSafetyError as exc:
                actions.append(FileAction(path=path, action=action or "?", status="failed", error=str(exc)))
                lines.append(f"Failed: {exc}")
            except (OSError, ValueError) as exc:
                actions.append(FileAction(path=path, action=action or "?", status="failed", error=str(exc)))
                lines.append(f"Failed: {exc}")
        return lines, actions

    # -- context for analysis / modification ---------------------------------------

    def build_context(
        self,
        *,
        max_files: int = 8,
        max_bytes_per_file: int = 4000,
        on_file: Callable[[str], None] | None = None,
        prompt: str | None = None,
        target_paths: list[str] | None = None,
    ) -> str | None:
        """Snapshot of the workspace (listing + excerpts) to feed a model.

        Lets the model edit or analyze real existing files instead of
        hallucinating their content. Returns None when the workspace is empty.
        Only injects full contents for files targeted by the prompt or task.
        ``on_file`` (when given) is called with each excerpted path so a live
        UI can show per-file reads.
        """
        entries = self._op.list_tree()
        if not entries:
            return None
        parts = ["Workspace files:"]
        for entry in entries[:40]:
            parts.append(f"- {entry['path']} ({entry['size']} bytes)")
        if len(entries) > 40:
            parts.append(f"- ... and {len(entries) - 40} more files")

        # Determine which files need content injection
        targeted_entries = []
        lowered_prompt = (prompt or "").lower()
        if target_paths:
            t_set = {p.lower() for p in target_paths}
            targeted_entries = [e for e in entries if e["path"].lower() in t_set or Path(e["path"]).name.lower() in t_set]
        elif prompt:
            for entry in entries:
                rel = entry["path"].lower()
                name = Path(entry["path"]).name.lower()
                if (rel in lowered_prompt or name in lowered_prompt) and len(name) > 3:
                    targeted_entries.append(entry)

        if targeted_entries:
            for entry in targeted_entries[:max_files]:
                content = self._op.read(entry["path"])
                if content is None:
                    continue
                excerpt = content[:max_bytes_per_file]
                parts.append(f"\n### {entry['path']}\n{excerpt}")
                if on_file:
                    on_file(entry["path"])
        else:
            for entry in entries[:min(max_files, 2)]:
                content = self._op.read(entry["path"])
                if content is None:
                    continue
                excerpt = content[:max_bytes_per_file]
                parts.append(f"\n### {entry['path']}\n{excerpt}")
                if on_file:
                    on_file(entry["path"])

        return "\n".join(parts)

    # -- summary ---------------------------------------------------------------------

    @staticmethod
    def summarize(
        actions: list[FileAction],
        validations: list[ValidationResult],
        review_text: str = "",
    ) -> str:
        """Short human summary of what the engine did — never raw file content."""
        if not actions:
            return ""
        validation_block = summarize_validations(validations, review_text=review_text)
        lines = [validation_block] if validation_block else []
        lines.append(f"Actions ({len(actions)}):")
        for action in actions:
            marker = "ok" if action.status == "ok" else f"FAILED: {action.error}"
            lines.append(f"- {action.action} {action.path} [{marker}]")
        return "\n".join(lines)
