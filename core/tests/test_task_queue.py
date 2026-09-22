"""Tests for the task queue system."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synapse.tasks.queue import TaskQueue, TaskStatus, TaskPriority


class TestTaskQueue:
    """Tests for TaskQueue."""

    def test_queue_creation(self):
        queue = TaskQueue(max_workers=2)
        assert queue._max_workers == 2

    def test_submit_task(self):
        queue = TaskQueue(max_workers=2)
        task_id = queue.submit("test_task", lambda: "result")
        assert task_id is not None
        info = queue.get_task(task_id)
        assert info is not None
        assert info.name == "test_task"

    def test_task_execution(self):
        queue = TaskQueue(max_workers=2)
        task_id = queue.submit("fast_task", lambda: 42)

        # Wait for completion
        time.sleep(0.5)

        info = queue.get_task(task_id)
        assert info.status == TaskStatus.COMPLETED.value

    def test_task_cancellation(self):
        queue = TaskQueue(max_workers=2)

        def slow_task():
            time.sleep(10)
            return "done"

        task_id = queue.submit("slow_task", slow_task)
        time.sleep(0.1)  # Let it start

        cancelled = queue.cancel(task_id)
        assert cancelled

        info = queue.get_task(task_id)
        assert info.status == TaskStatus.CANCELLED.value

    def test_task_list(self):
        queue = TaskQueue(max_workers=2)
        queue.submit("task1", lambda: 1)
        queue.submit("task2", lambda: 2)

        tasks = queue.list_tasks()
        assert len(tasks) == 2

    def test_task_priority(self):
        queue = TaskQueue(max_workers=1)
        # The low-priority task sleeps so the single worker stays occupied and
        # the high-priority task is guaranteed to remain queued when asserted.
        id1 = queue.submit("low", lambda: time.sleep(0.5), priority=TaskPriority.LOW)
        id2 = queue.submit("high", lambda: 2, priority=TaskPriority.HIGH)

        # High priority should be first in queue
        assert queue._queue[0] == id2

    def test_task_retry(self):
        queue = TaskQueue(max_workers=2)

        def failing_task():
            raise ValueError("test error")

        task_id = queue.submit("failing", failing_task, max_retries=2)
        time.sleep(0.5)

        info = queue.get_task(task_id)
        assert info.status == TaskStatus.FAILED.value

        # Retry
        retried = queue.retry(task_id)
        assert retried

    def test_cleanup_old_tasks(self):
        queue = TaskQueue(max_workers=2)
        task_id = queue.submit("task", lambda: 1)
        time.sleep(0.5)

        # Cleanup with 0 age should remove completed tasks
        removed = queue.cleanup(max_age_hours=0)
        assert removed >= 1
