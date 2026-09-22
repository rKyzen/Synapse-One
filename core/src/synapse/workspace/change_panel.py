"""ChangePanel — tracks and displays file changes in the workspace.

Provides a structured view of all file modifications during a session:
    - Files created
    - Files modified
    - Files deleted
    - Files renamed

Similar to modern coding agents' change tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from synapse.logging import get_logger

log = get_logger("synapse.workspace.change_panel")


class ChangeType(str, Enum):
    """Types of file changes."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


@dataclass
class FileChange:
    """A single file change."""

    path: str
    change_type: ChangeType
    timestamp: str
    old_path: str | None = None  # For renames
    size_before: int | None = None
    size_after: int | None = None
    line_changes: int | None = None  # +lines, -lines
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_addition(self) -> bool:
        return self.change_type == ChangeType.CREATED

    @property
    def is_removal(self) -> bool:
        return self.change_type == ChangeType.DELETED

    @property
    def display_name(self) -> str:
        if self.change_type == ChangeType.RENAMED:
            return f"{self.old_path} -> {self.path}"
        return self.path


@dataclass
class ChangeSummary:
    """Summary of all changes in a session."""

    changes: list[FileChange]
    session_id: str = ""
    started_at: str = ""

    @property
    def created(self) -> list[FileChange]:
        return [c for c in self.changes if c.change_type == ChangeType.CREATED]

    @property
    def modified(self) -> list[FileChange]:
        return [c for c in self.changes if c.change_type == ChangeType.MODIFIED]

    @property
    def deleted(self) -> list[FileChange]:
        return [c for c in self.changes if c.change_type == ChangeType.DELETED]

    @property
    def renamed(self) -> list[FileChange]:
        return [c for c in self.changes if c.change_type == ChangeType.RENAMED]

    def to_dict(self) -> dict:
        return {
            "total": len(self.changes),
            "created": len(self.created),
            "modified": len(self.modified),
            "deleted": len(self.deleted),
            "renamed": len(self.renamed),
            "files": [
                {
                    "path": c.path,
                    "type": c.change_type.value,
                    "timestamp": c.timestamp,
                    "old_path": c.old_path,
                }
                for c in self.changes
            ],
        }

    def to_markdown(self) -> str:
        """Human-readable markdown summary."""
        lines = ["## File Changes\n"]

        if self.created:
            lines.append("### Created")
            for c in self.created:
                lines.append(f"- `{c.path}`")
            lines.append("")

        if self.modified:
            lines.append("### Modified")
            for c in self.modified:
                lines.append(f"- `{c.path}`")
            lines.append("")

        if self.deleted:
            lines.append("### Deleted")
            for c in self.deleted:
                lines.append(f"- `{c.path}`")
            lines.append("")

        if self.renamed:
            lines.append("### Renamed")
            for c in self.renamed:
                lines.append(f"- `{c.old_path}` -> `{c.path}`")
            lines.append("")

        if not self.changes:
            lines.append("No changes yet.")

        return "\n".join(lines)


class ChangePanel:
    """Tracks file changes during a session."""

    def __init__(self, session_id: str = "") -> None:
        self._session_id = session_id
        self._changes: list[FileChange] = []
        self._start_time = datetime.now(timezone.utc).isoformat()

    def record_created(self, path: str, size: int | None = None) -> None:
        """Record a file creation."""
        self._changes.append(FileChange(
            path=path,
            change_type=ChangeType.CREATED,
            timestamp=datetime.now(timezone.utc).isoformat(),
            size_after=size,
        ))

    def record_modified(
        self,
        path: str,
        size_before: int | None = None,
        size_after: int | None = None,
        line_changes: int | None = None,
    ) -> None:
        """Record a file modification."""
        self._changes.append(FileChange(
            path=path,
            change_type=ChangeType.MODIFIED,
            timestamp=datetime.now(timezone.utc).isoformat(),
            size_before=size_before,
            size_after=size_after,
            line_changes=line_changes,
        ))

    def record_deleted(self, path: str) -> None:
        """Record a file deletion."""
        self._changes.append(FileChange(
            path=path,
            change_type=ChangeType.DELETED,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))

    def record_renamed(self, old_path: str, new_path: str) -> None:
        """Record a file rename."""
        self._changes.append(FileChange(
            path=new_path,
            change_type=ChangeType.RENAMED,
            timestamp=datetime.now(timezone.utc).isoformat(),
            old_path=old_path,
        ))

    def get_summary(self) -> ChangeSummary:
        """Get a summary of all changes."""
        return ChangeSummary(
            changes=list(self._changes),
            session_id=self._session_id,
            started_at=self._start_time,
        )

    def get_recent(self, limit: int = 10) -> list[FileChange]:
        """Get the most recent changes."""
        return list(reversed(self._changes[-limit:]))

    def clear(self) -> None:
        """Clear all recorded changes."""
        self._changes.clear()
        self._start_time = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Serialize for API responses."""
        return self.get_summary().to_dict()
