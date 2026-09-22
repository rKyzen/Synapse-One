"""NoteStore — durable, atomic per-note persistence in internal storage.

Notes are the workspace's scratch layer: markdown entries, goal-linkable and
taggable. One JSON file per note under ``<internal>/notes/<project_id>/``,
following the shared JsonEntityStore pattern (atomic tmp + replace).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from synapse.domain.notes import Note
from synapse.projects.stores import JsonEntityStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class NoteStore(JsonEntityStore[Note]):
    """File-backed note records for one workspace/project."""

    def __init__(self, project_id: str, notes_dir: Path) -> None:
        super().__init__(project_id, notes_dir, Note, bucket_name="note")

    def create(
        self,
        title: str,
        content: str = "",
        *,
        goal_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Note:
        note = Note(
            id=self.new_id(),
            project_id=self._project_id,
            title=(title or "").strip() or self._title_from(content),
            content=content,
            goal_id=goal_id,
            tags=[t for t in (tags or []) if t],
        )
        return self.put(note)

    def update(
        self,
        note_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
        goal_id: str | None = None,
    ) -> Note | None:
        with self._lock:
            note = self._read(note_id)
            if note is None:
                return None
            if title is not None:
                note.title = title.strip() or note.title
            if content is not None:
                note.content = content
            if tags is not None:
                note.tags = [t for t in tags if t]
            if goal_id is not None:
                note.goal_id = goal_id or None
            note.updated_at = _now_iso()
            self._write(note)
            return note

    @staticmethod
    def _title_from(content: str) -> str:
        """Derive a title from the note's first non-empty line."""
        for line in content.splitlines():
            text = re.sub(r"^\s*[#>*\-]+\s*", "", line).strip()
            if text:
                return text if len(text) <= 60 else text[:60].rstrip() + "..."
        return "Note"
