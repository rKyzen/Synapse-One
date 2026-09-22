"""TaskQueue — background task management with cancellation support.

Supports:
    - Background execution
    - Cancel
    - Retry
    - Pause/Resume
    - Progress tracking
    - Priority ordering
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from synapse.logging import get_logger

log = get_logger("synapse.tasks.queue")


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RETRYING = "retrying"


class TaskPriority(str, Enum):
    """Task priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class TaskDefinition:
    """Definition of a background task."""

    id: str
    name: str
    func: Callable[..., Any]
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    timeout: int = 300  # seconds
    created_at: str = ""
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    progress: float = 0.0
    retries: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    cancellation_token: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        """Request cancellation."""
        self.cancellation_token.set()

    @property
    def is_cancelled(self) -> bool:
        return self.cancellation_token.is_set()


@dataclass
class TaskInfo:
    """Public info about a task."""

    id: str
    name: str
    priority: str
    status: str
    progress: float
    created_at: str
    started_at: str | None
    completed_at: str | None
    error: str | None
    retries: int


class TaskQueue:
    """Manages background task execution."""

    def __init__(self, max_workers: int = 2) -> None:
        self._max_workers = max_workers
        self._tasks: dict[str, TaskDefinition] = {}
        self._queue: list[str] = []
        self._running: dict[str, threading.Thread] = {}
        self._lock = threading.RLock()
        self._counter = 0

    def submit(
        self,
        name: str,
        func: Callable[..., Any],
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        timeout: int = 300,
        **kwargs,
    ) -> str:
        """Submit a task for background execution. Returns task ID."""
        with self._lock:
            self._counter += 1
            task_id = f"task_{self._counter}"
            task = TaskDefinition(
                id=task_id,
                name=name,
                func=func,
                args=args,
                kwargs=kwargs,
                priority=priority,
                max_retries=max_retries,
                timeout=timeout,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._tasks[task_id] = task
            self._queue.append(task_id)
            self._sort_queue()
            self._try_start_next()
            log.info("task_submitted", task_id=task_id, name=name, priority=priority.value)
            return task_id

    def cancel(self, task_id: str) -> bool:
        """Cancel a task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
                return False
            task.cancel()
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now(timezone.utc).isoformat()
            # Remove from queue if pending
            if task_id in self._queue:
                self._queue.remove(task_id)
            log.info("task_cancelled", task_id=task_id)
            return True

    def pause(self, task_id: str) -> bool:
        """Pause a running task (cooperative)."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != TaskStatus.RUNNING:
                return False
            task.status = TaskStatus.PAUSED
            log.info("task_paused", task_id=task_id)
            return True

    def resume(self, task_id: str) -> bool:
        """Resume a paused task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != TaskStatus.PAUSED:
                return False
            task.status = TaskStatus.RUNNING
            log.info("task_resumed", task_id=task_id)
            return True

    def retry(self, task_id: str) -> bool:
        """Retry a failed task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != TaskStatus.FAILED:
                return False
            if task.retries >= task.max_retries:
                return False
            task.status = TaskStatus.RETRYING
            task.retries += 1
            task.error = None
            task.progress = 0.0
            self._queue.append(task_id)
            self._sort_queue()
            self._try_start_next()
            log.info("task_retrying", task_id=task_id, retries=task.retries)
            return True

    def get_task(self, task_id: str) -> TaskInfo | None:
        """Get task info."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            return TaskInfo(
                id=task.id,
                name=task.name,
                priority=task.priority.value,
                status=task.status.value,
                progress=task.progress,
                created_at=task.created_at,
                started_at=task.started_at,
                completed_at=task.completed_at,
                error=task.error,
                retries=task.retries,
            )

    def list_tasks(
        self,
        status: TaskStatus | None = None,
        limit: int = 50,
    ) -> list[TaskInfo]:
        """List tasks with optional filtering."""
        with self._lock:
            tasks = list(self._tasks.values())
            if status:
                tasks = [t for t in tasks if t.status == status]
            tasks.sort(key=lambda t: t.created_at, reverse=True)
            return [
                TaskInfo(
                    id=t.id,
                    name=t.name,
                    priority=t.priority.value,
                    status=t.status.value,
                    progress=t.progress,
                    created_at=t.created_at,
                    started_at=t.started_at,
                    completed_at=t.completed_at,
                    error=t.error,
                    retries=t.retries,
                )
                for t in tasks[:limit]
            ]

    def update_progress(self, task_id: str, progress: float) -> None:
        """Update task progress (0.0 to 1.0)."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.progress = min(1.0, max(0.0, progress))

    def cleanup(self, max_age_hours: int = 24) -> int:
        """Remove old completed tasks. Returns count removed."""
        with self._lock:
            cutoff = time.time() - (max_age_hours * 3600)
            to_remove = []
            for task_id, task in self._tasks.items():
                if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED):
                    if task.completed_at:
                        try:
                            completed = datetime.fromisoformat(task.completed_at)
                            if completed.timestamp() < cutoff:
                                to_remove.append(task_id)
                        except ValueError:
                            pass
            for task_id in to_remove:
                del self._tasks[task_id]
            return len(to_remove)

    # -- Private methods -----------------------------------------------------

    def _sort_queue(self) -> None:
        """Sort queue by priority."""
        priority_order = {
            TaskPriority.URGENT: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.NORMAL: 2,
            TaskPriority.LOW: 3,
        }
        self._queue.sort(key=lambda tid: priority_order.get(
            self._tasks[tid].priority, 2
        ))

    def _try_start_next(self) -> None:
        """Try to start the next task in the queue."""
        if len(self._running) >= self._max_workers:
            return
        if not self._queue:
            return

        task_id = self._queue.pop(0)
        task = self._tasks.get(task_id)
        if task is None or task.status not in (TaskStatus.PENDING, TaskStatus.RETRYING):
            return

        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(timezone.utc).isoformat()

        thread = threading.Thread(
            target=self._run_task,
            args=(task,),
            daemon=True,
        )
        self._running[task_id] = thread
        thread.start()

    def _run_task(self, task: TaskDefinition) -> None:
        """Execute a task in a worker thread."""
        try:
            result = task.func(*task.args, **task.kwargs)
            if not task.is_cancelled:
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.progress = 1.0
        except Exception as exc:
            if not task.is_cancelled:
                task.error = str(exc)
                task.status = TaskStatus.FAILED
                log.warning("task_failed", task_id=task.id, error=str(exc)[:200])
        finally:
            task.completed_at = datetime.now(timezone.utc).isoformat()
            with self._lock:
                self._running.pop(task.id, None)
            self._try_start_next()
