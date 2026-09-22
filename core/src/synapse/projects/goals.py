"""GoalStore — durable, atomic per-goal persistence in internal storage.

Each goal is one JSON file under ``<internal>/goals/<project_id>/<goal_id>.json``,
following the exact persistence pattern of ChatStore (atomic tmp + replace on
every write). Goals are the atomic unit of the AI Operating Workspace: the
user states an outcome, and the workspace tracks progress, steps, and linked
artifacts around it. Nothing here raises on I/O.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from synapse.domain.enums import GoalStatus
from synapse.domain.goals import GoalInfo, GoalRecord, GoalStep
from synapse.logging import get_logger

log = get_logger("synapse.projects.goals")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


class GoalStore:
    """File-backed goal records for one workspace/project."""

    def __init__(self, project_id: str, goals_dir: Path) -> None:
        self._project_id = project_id
        self._dir = goals_dir
        self._lock = threading.RLock()
        self._dir.mkdir(parents=True, exist_ok=True)

    # -- paths --------------------------------------------------------------

    def _path(self, goal_id: str) -> Path:
        return self._dir / f"{goal_id}.json"

    # -- persistence --------------------------------------------------------

    def _read(self, goal_id: str) -> GoalRecord | None:
        path = self._path(goal_id)
        try:
            if not path.exists():
                return None
            record = GoalRecord(**json.loads(path.read_text(encoding="utf-8")))
            record.project_id = self._project_id
            return record
        except Exception:  # noqa: BLE001 - a corrupt goal never breaks a project
            log.warning("goal_load_failed", goal_id=goal_id, path=str(path))
            return None

    def _write(self, record: GoalRecord) -> None:
        try:
            tmp = self._path(record.id).with_suffix(".json.tmp")
            tmp.write_text(json.dumps(record.model_dump(), indent=2), encoding="utf-8")
            tmp.replace(self._path(record.id))
        except Exception:  # noqa: BLE001
            log.warning("goal_save_failed", goal_id=record.id)

    # -- API -----------------------------------------------------------------

    def list(self, include_archived: bool = False) -> list[GoalInfo]:
        with self._lock:
            out = []
            for path in sorted(self._dir.glob("*.json")):
                if path.name.endswith(".tmp"):
                    continue
                record = self._read(path.stem)
                if record is not None and (
                    include_archived or record.status != GoalStatus.ARCHIVED
                ):
                    out.append(record.to_info())
            out.sort(key=lambda g: g.updated_at, reverse=True)
            return out

    def get(self, goal_id: str) -> GoalInfo | None:
        with self._lock:
            record = self._read(goal_id)
            return record.to_info() if record else None

    def create(self, outcome: str, *, deadline: str | None = None) -> GoalInfo:
        """Create a goal from its outcome statement (primary intake point)."""
        goal_id = _uuid()
        record = GoalRecord(
            id=goal_id,
            project_id=self._project_id,
            outcome=outcome,
            title=self._title_from(outcome),
            deadline=deadline or None,
        )
        with self._lock:
            self._write(record)
        log.info("goal_created", goal_id=goal_id, project_id=self._project_id, title=record.title)
        return record.to_info()

    def rename(self, goal_id: str, title: str) -> GoalInfo | None:
        with self._lock:
            record = self._read(goal_id)
            if record is None:
                return None
            record.title = title.strip() or record.title
            record.updated_at = _now_iso()
            self._write(record)
            return record.to_info()

    def update_status(self, goal_id: str, status: GoalStatus) -> GoalInfo | None:
        with self._lock:
            record = self._read(goal_id)
            if record is None:
                return None
            record.status = status
            record.updated_at = _now_iso()
            self._write(record)
            log.info("goal_status_updated", goal_id=goal_id, status=status.value)
            return record.to_info()

    def set_progress(self, goal_id: str, progress: int) -> GoalInfo | None:
        """Update 0..100 progress (clamped). Called by the orchestrator."""
        with self._lock:
            record = self._read(goal_id)
            if record is None:
                return None
            record.progress = max(0, min(100, int(progress)))
            record.updated_at = _now_iso()
            self._write(record)
            return record.to_info()

    def set_steps(self, goal_id: str, steps: list[GoalStep]) -> GoalInfo | None:
        """Replace the planned step list (set by the planner/orchestrator)."""
        with self._lock:
            record = self._read(goal_id)
            if record is None:
                return None
            record.steps = list(steps)
            record.updated_at = _now_iso()
            self._write(record)
            return record.to_info()

    def update_step(
        self,
        goal_id: str,
        index: int,
        *,
        status: str | None = None,
        model: str | None = None,
        result: str | None = None,
    ) -> GoalInfo | None:
        """Update one planned step's status/model/result without rewriting the plan."""
        with self._lock:
            record = self._read(goal_id)
            if record is None or not (0 <= index < len(record.steps)):
                return None
            step = record.steps[index]
            if status is not None:
                step.status = status
            if model is not None:
                step.model = model
            if result is not None:
                step.result = result
            if record.steps:
                done = sum(1 for s in record.steps if s.status in ("completed", "skipped"))
                record.progress = min(100, round(done * 100 / len(record.steps)))
            record.updated_at = _now_iso()
            self._write(record)
            return record.to_info()

    def link(
        self, goal_id: str, *, chat_id: str | None = None, file_id: str | None = None,
        note_id: str | None = None, task_id: str | None = None,
    ) -> GoalInfo | None:
        """Link an artifact (chat/file/note/task) to the goal. Idempotent."""
        with self._lock:
            record = self._read(goal_id)
            if record is None:
                return None
            if chat_id and chat_id not in record.linked_chats:
                record.linked_chats.append(chat_id)
            if file_id and file_id not in record.linked_files:
                record.linked_files.append(file_id)
            if note_id and note_id not in record.linked_notes:
                record.linked_notes.append(note_id)
            if task_id and task_id not in record.linked_tasks:
                record.linked_tasks.append(task_id)
            record.updated_at = _now_iso()
            self._write(record)
            return record.to_info()

    def delete(self, goal_id: str) -> bool:
        with self._lock:
            path = self._path(goal_id)
            try:
                if path.exists():
                    path.unlink()
                    log.info("goal_deleted", goal_id=goal_id, project_id=self._project_id)
                    return True
            except OSError:
                log.warning("goal_unlink_failed", goal_id=goal_id)
            return False

    @staticmethod
    def _title_from(outcome: str) -> str:
        """A clean one-line title derived from the outcome statement."""
        text = re.sub(r"\s+", " ", outcome).strip()
        text = text.lstrip("#>*-\t ")
        if not text:
            return "Goal"
        return text if len(text) <= 60 else text[:60].rstrip() + "..."
