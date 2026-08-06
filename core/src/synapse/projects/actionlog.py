"""ActionLog — Phase 6 durable, append-only record of AI workspace actions.

One JSON-lines file per project at ``<project>/logs/actions.jsonl``. Every
file-writing request writes one entry capturing:

- the prompt and resulting task plan (kinds)
- which models were used
- which tools/tool actions were used
- which files were created/modified/renamed/deleted
- validation results
- total execution time
- failures

Appends are atomic for a single record (serialized under a per-store lock);
reads never crash. Writes are best-effort — a logging failure must never
break a chat request.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from synapse.logging import get_logger

log = get_logger("synapse.projects.actionlog")

LOG_FILE = "actions.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ActionLog:
    """Append-only JSONL action store for one project."""

    def __init__(self, project_id: str, logs_dir: str | Path) -> None:
        self._project_id = project_id
        self._logs_dir = Path(logs_dir)
        self._path = self._logs_dir / LOG_FILE
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    def record(self, entry: dict) -> dict:
        """Append one record; returns it with a timestamp. Never raises."""
        record = dict(entry or {})
        record.setdefault("project_id", self._project_id)
        record.setdefault("ts", _now_iso())
        try:
            with self._lock:
                self._logs_dir.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, default=str) + "\n")
                    handle.flush()
        except Exception:  # noqa: BLE001
            log.warning("action_log_write_failed", project_id=self._project_id)
        return record

    def recent(self, limit: int = 50) -> list[dict]:
        """Most recent entries, newest first."""
        try:
            with self._lock:
                if not self._path.exists():
                    return []
                raw = self._path.read_text(encoding="utf-8").splitlines()
            entries = []
            for line in raw[-limit:]:
                try:
                    entries.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue
            return list(reversed(entries))
        except Exception:  # noqa: BLE001
            log.warning("action_log_read_failed", project_id=self._project_id)
            return []

    def count(self) -> int:
        return len(self.recent(limit=10 ** 9))