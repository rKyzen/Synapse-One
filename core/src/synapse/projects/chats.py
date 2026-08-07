"""ChatStore — durable, atomic per-chat persistence in internal storage.

Each chat is one JSON file under ``<internal>/chats/<project_id>/<chat_id>.json``
(Phase 7: chats never live inside the user's workspace folder). Each file holds
its header (id, title, timestamps) plus every message. Every append rewrites
the file atomically (tmp + replace), so a message is durable the moment it is
recorded and survives an unexpected shutdown. Nothing here raises on I/O.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from synapse.domain.projects import ChatInfo, ChatRecord, StoredMessage
from synapse.logging import get_logger

log = get_logger("synapse.projects.chats")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


class ChatStore:
    """File-backed chat records for one project."""

    def __init__(self, project_id: str, chats_dir: Path) -> None:
        self._project_id = project_id
        self._dir = chats_dir
        self._lock = threading.RLock()
        self._dir.mkdir(parents=True, exist_ok=True)

    # -- paths --------------------------------------------------------------

    def _path(self, chat_id: str) -> Path:
        return self._dir / f"{chat_id}.json"

    # -- persistence --------------------------------------------------------

    def _read(self, chat_id: str) -> ChatRecord | None:
        path = self._path(chat_id)
        try:
            if not path.exists():
                return None
            record = ChatRecord(**json.loads(path.read_text(encoding="utf-8")))
            record.project_id = self._project_id
            return record
        except Exception:  # noqa: BLE001 - a corrupt chat never breaks a project
            log.warning("chat_load_failed", chat_id=chat_id, path=str(path))
            return None

    def _write(self, record: ChatRecord) -> None:
        try:
            tmp = self._path(record.id).with_suffix(".json.tmp")
            tmp.write_text(json.dumps(record.model_dump(), indent=2), encoding="utf-8")
            tmp.replace(self._path(record.id))
        except Exception:  # noqa: BLE001
            log.warning("chat_save_failed", chat_id=record.id)

    # -- API -----------------------------------------------------------------

    def list(self) -> list[ChatInfo]:
        with self._lock:
            out = []
            for path in sorted(self._dir.glob("*.json")):
                if path.name.endswith(".tmp"):
                    continue
                record = self._read(path.stem)
                if record is not None:
                    out.append(self._to_info(record))
            out.sort(key=lambda c: c.updated_at, reverse=True)
            return out

    def get(self, chat_id: str) -> ChatInfo | None:
        with self._lock:
            record = self._read(chat_id)
            return self._to_info(record) if record else None

    def create(self, title: str = "New chat") -> ChatInfo:
        chat_id = _uuid()
        record = ChatRecord(
            id=chat_id,
            project_id=self._project_id,
            title=title or "New chat",
            created_at=_now_iso(),
            updated_at=_now_iso(),
            messages=[],
        )
        with self._lock:
            self._write(record)
        log.info("chat_created", chat_id=chat_id, project_id=self._project_id, title=record.title)
        return self._to_info(record)

    def rename(self, chat_id: str, title: str) -> ChatInfo | None:
        with self._lock:
            record = self._read(chat_id)
            if record is None:
                return None
            record.title = title.strip() or record.title
            record.updated_at = _now_iso()
            self._write(record)
            return self._to_info(record)

    def delete(self, chat_id: str) -> bool:
        with self._lock:
            path = self._path(chat_id)
            try:
                if path.exists():
                    path.unlink()
                    log.info("chat_deleted", chat_id=chat_id, project_id=self._project_id)
                    return True
            except OSError:
                log.warning("chat_unlink_failed", chat_id=chat_id)
            return False

    def messages(self, chat_id: str) -> list[StoredMessage]:
        with self._lock:
            record = self._read(chat_id)
            return list(record.messages) if record else []

    def append(
        self,
        chat_id: str,
        *,
        role: str,
        content: str,
        meta: list[str] | None = None,
        trace: dict | None = None,
        files: list[str] | None = None,
    ) -> StoredMessage | None:
        """Append a message and write the chat file immediately."""
        message = StoredMessage(
            role=role,
            content=content,
            created_at=_now_iso(),
            meta=meta or [],
            trace=trace,
            files=files or [],
        )
        with self._lock:
            record = self._read(chat_id)
            if record is None:
                return None
            record.messages.append(message)
            record.updated_at = message.created_at
            # Auto-rename from conversation context: the first meaningful user
            # message seeds the chat's title (works for every default
            # placeholder — "New chat", the "General Discussion" created by
            # ``ensure_chat``, or an empty title). Whitespace is collapsed so
            # a multi-line prompt becomes a clean one-line title. A title the
            # user set explicitly is never overwritten.
            if message.role == "user" and self._default_title(record.title):
                record.title = self._title_from(message.content)
            self._write(record)
            log.info(
                "message_recorded",
                chat_id=chat_id,
                project_id=self._project_id,
                role=message.role,
                chars=len(content),
            )
            return message

    @staticmethod
    def _default_title(title: str) -> bool:
        return title.strip().lower() in ("", "new chat", "general discussion")

    @staticmethod
    def _title_from(content: str) -> str:
        """A clean one-line title derived from the conversation's first message."""
        text = re.sub(r"\s+", " ", content).strip()
        text = text.lstrip("#>*-\t ")
        if not text:
            return "Chat"
        return text if len(text) <= 50 else text[:50].rstrip() + "..."

    @staticmethod
    def _to_info(record: ChatRecord) -> ChatInfo:
        return ChatInfo(
            id=record.id,
            project_id=record.project_id,
            title=record.title,
            created_at=record.created_at,
            updated_at=record.updated_at,
            message_count=len(record.messages),
        )