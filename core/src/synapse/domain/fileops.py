"""Phase 6 domain models — file actions, validation, and workspace summaries.

Pure data structures describing what the AI Workspace did to the user's
project: which files were created/edited/renamed/deleted, whether each passes
automated validation, and a human-readable summary of completed actions.
No behavior, no vendor logic.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FileAction(BaseModel):
    """One filesystem mutation applied to the project's work directory."""

    path: str  # relative to the work directory
    action: str  # created | modified | renamed | deleted
    status: str = "ok"  # ok | failed
    bytes: int = 0
    error: str = ""
    validated: bool = False
    validation: str = ""  # short human note e.g. "python syntax ok"


class ValidationResult(BaseModel):
    """Result of deterministic syntax/consistency checks on a generated file."""

    path: str
    ok: bool = True
    checks: list[str] = Field(default_factory=list)
    error: str = ""


class WorkspaceSummary(BaseModel):
    """Concise machine-readable record of a file-writing request."""

    files_created: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    files_deleted: list[str] = Field(default_factory=list)
    files_failed: list[str] = Field(default_factory=list)
    validations: list[ValidationResult] = Field(default_factory=list)
    review_text: str = ""