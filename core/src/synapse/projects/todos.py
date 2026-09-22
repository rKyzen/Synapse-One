"""TodoStore — durable, atomic per-todo persistence in internal storage.

Todo items are the user's to-do list within a workspace: goal-linkable,
prioritized, with optional due dates. One JSON file per todo under
``<internal>/todos/<project_id>/``, following the shared JsonEntityStore
pattern (atomic tmp + replace).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from synapse.domain.enums import TodoPriority, TodoStatus
from synapse.domain.todos import TodoTask
from synapse.projects.stores import JsonEntityStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TodoStore(JsonEntityStore[TodoTask]):
    """File-backed todo records for one workspace/project."""

    def __init__(self, project_id: str, todos_dir: Path) -> None:
        super().__init__(project_id, todos_dir, TodoTask, bucket_name="todo")

    def create(
        self,
        title: str,
        *,
        description: str = "",
        status: TodoStatus | str = TodoStatus.TODO,
        priority: TodoPriority | str = TodoPriority.MEDIUM,
        due_date: str | None = None,
        goal_id: str | None = None,
    ) -> TodoTask:
        try:
            status_enum = status if isinstance(status, TodoStatus) else TodoStatus(status)
        except ValueError:
            status_enum = TodoStatus.TODO
        try:
            priority_enum = (
                priority if isinstance(priority, TodoPriority) else TodoPriority(priority)
            )
        except ValueError:
            priority_enum = TodoPriority.MEDIUM
        todo = TodoTask(
            id=self.new_id(),
            project_id=self._project_id,
            title=(title or "").strip() or "Untitled task",
            description=description,
            status=status_enum,
            priority=priority_enum,
            due_date=due_date or None,
            goal_id=goal_id,
        )
        return self.put(todo)

    def update(
        self,
        todo_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        status: TodoStatus | str | None = None,
        priority: TodoPriority | str | None = None,
        due_date: str | None = None,
        goal_id: str | None = None,
    ) -> TodoTask | None:
        with self._lock:
            todo = self._read(todo_id)
            if todo is None:
                return None
            if title is not None:
                todo.title = title.strip() or todo.title
            if description is not None:
                todo.description = description
            if status is not None:
                try:
                    todo.status = status if isinstance(status, TodoStatus) else TodoStatus(status)
                except ValueError:
                    pass
            if priority is not None:
                try:
                    todo.priority = (
                        priority if isinstance(priority, TodoPriority) else TodoPriority(priority)
                    )
                except ValueError:
                    pass
            if due_date is not None:
                todo.due_date = due_date or None
            if goal_id is not None:
                todo.goal_id = goal_id or None
            todo.updated_at = _now_iso()
            self._write(todo)
            return todo
