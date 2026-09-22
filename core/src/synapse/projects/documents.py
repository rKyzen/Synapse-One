"""DocumentStore — durable, atomic per-document persistence in internal storage.

Documents are typed, longer-form workspace artifacts (report, essay, lesson
plan, business plan, ...). One JSON file per document under
``<internal>/documents/<project_id>/``, following the shared JsonEntityStore
pattern (atomic tmp + replace).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from synapse.domain.documents import Document
from synapse.domain.enums import DocumentStatus
from synapse.projects.stores import JsonEntityStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DocumentStore(JsonEntityStore[Document]):
    """File-backed typed document records for one workspace/project."""

    def __init__(self, project_id: str, documents_dir: Path) -> None:
        super().__init__(project_id, documents_dir, Document, bucket_name="document")

    def create(
        self,
        title: str,
        *,
        doc_type: str = "document",
        content: str = "",
        goal_id: str | None = None,
    ) -> Document:
        doc = Document(
            id=self.new_id(),
            project_id=self._project_id,
            title=(title or "").strip() or "Untitled document",
            doc_type=(doc_type or "document").strip(),
            content=content,
            status=DocumentStatus.DRAFT.value,
            goal_id=goal_id,
        )
        return self.put(doc)

    def update(
        self,
        document_id: str,
        *,
        title: str | None = None,
        doc_type: str | None = None,
        content: str | None = None,
        status: str | None = None,
        goal_id: str | None = None,
    ) -> Document | None:
        with self._lock:
            doc = self._read(document_id)
            if doc is None:
                return None
            if title is not None:
                doc.title = title.strip() or doc.title
            if doc_type is not None:
                doc.doc_type = doc_type.strip() or doc.doc_type
            if content is not None:
                doc.content = content
            if status is not None:
                try:
                    doc.status = DocumentStatus(status).value
                except ValueError:
                    doc.status = DocumentStatus.DRAFT.value
            if goal_id is not None:
                doc.goal_id = goal_id or None
            doc.updated_at = _now_iso()
            self._write(doc)
            return doc
