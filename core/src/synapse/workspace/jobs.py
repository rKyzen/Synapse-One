"""Indexing jobs — thread-safe progress tracking for file ingestion.

Uploads return instantly; indexing runs in a background thread and reports
stage + 0-100 progress through the JobRegistry so the web UI can poll
``GET /jobs/{id}``.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

log = logging.getLogger("synapse.workspace.jobs")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IndexingJob(BaseModel):
    job_id: str
    file_id: str
    file_name: str = ""
    status: str = "running"  # pending | running | completed | failed
    stage: str = "queued"
    progress: int = Field(default=0, ge=0, le=100)
    message: str = ""
    error: str | None = None
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class JobRegistry:
    """In-memory job store with a lock; never raises on read."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, IndexingJob] = {}

    def create(self, file_id: str, file_name: str = "") -> IndexingJob:
        job = IndexingJob(job_id=uuid.uuid4().hex[:12], file_id=file_id, file_name=file_name)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def update(self, job_id: str, *, stage: str | None = None, progress: int | None = None,
               message: str | None = None, status: str | None = None, error: str | None = None) -> IndexingJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if stage is not None:
                job.stage = stage
            if message is not None:
                job.message = message
            if error is not None:
                job.error = error
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = max(0, min(100, int(progress)))
            job.updated_at = _now()
        return job

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_dump() if job else None

    def latest_for(self, file_id: str) -> dict | None:
        with self._lock:
            match = [j for j in self._jobs.values() if j.file_id == file_id]
        if not match:
            return None
        return max(match, key=lambda j: j.created_at).model_dump()