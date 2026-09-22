"""Live execution timeline — turns bus events into an SSE stream for the UI.

The chat UI calls ``POST /request/stream``; the hub runs the normal request
on a background thread and relays the timeline-relevant bus events (request
analysis, workspace reads, model load, task generation, file writes) as SSE
``data:`` lines. The final event carries the full :class:`AgentResponse` so
the frontend can render the reply exactly as with ``/request``.

Single active stream by design: the UI already guards against concurrent
requests with its ``busy`` flag.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from typing import Any, Callable

from fastapi.responses import StreamingResponse

from synapse.events import Event, EventBus, Events

log = logging.getLogger("synapse.api.timeline")


class TimelineHub:
    """Captures bus events while a request runs and streams them to the UI."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._queue: queue.Queue | None = None
        bus.subscribe("*", self._forward)

    # -- public -------------------------------------------------------------------

    def stream(self, run: Callable[[], tuple[Any, list[str]]]) -> StreamingResponse:
        """Run ``run()`` on a background thread and SSE its timeline events."""
        q: queue.Queue = queue.Queue()
        self._queue = q

        def worker() -> None:
            try:
                response, meta = run()
                self._queue = None
                q.put({"kind": "done", "response": response, "meta": meta})
            except Exception as exc:  # noqa: BLE001 - surfaced as an SSE error event
                log.exception("request_stream_failed")
                self._queue = None
                q.put({"kind": "stream_error", "detail": str(exc)[:500]})

        threading.Thread(target=worker, daemon=True).start()

        def generator():
            while True:
                step = q.get()
                if step["kind"] == "done":
                    yield _encode(
                        {
                            "kind": "done",
                            "response": _dump(step["response"]),
                            "meta": step["meta"],
                        }
                    )
                    break
                if step["kind"] == "stream_error":
                    yield _encode({"kind": "stream_error", "detail": step["detail"]})
                    break
                yield _encode(step)

        return StreamingResponse(generator(), media_type="text/event-stream")

    # -- plumbing -----------------------------------------------------------------

    def _forward(self, event: Event) -> None:
        q = self._queue
        if q is None:
            return
        step = self._map(event)
        if step is not None:
            q.put(step)

    @staticmethod
    def _map(event: Event) -> dict | None:
        name = event.name
        payload = event.payload or {}
        if name == Events.TIMELINE:
            return {"kind": payload.get("kind", "step"), "text": payload.get("text", "")}
        if name == Events.REQUEST_RECEIVED:
            return {"kind": "understand", "text": "Understanding request…"}
        if name == Events.REQUEST_ANALYZED:
            return {"kind": "analyze", "text": "Analyzing the request…"}
        if name == Events.TASK_PLANNED:
            count = payload.get("task_count", "")
            return {"kind": "plan_tasks", "text": f"Planning tasks ({count} task{'s' if count != 1 else ''})…"}
        if name == Events.RETRIEVAL_RAN:
            return {"kind": "search", "text": "Reading workspace knowledge…"}
        if name == Events.CAPABILITY_SELECTED:
            cap = payload.get("capability", "general")
            return {"kind": "select_capability", "text": f"Selecting capability: {cap}"}
        if name == Events.TASK_ROUTED:
            model = payload.get("model", "model")
            task_id = payload.get("task_id", "")
            return {"kind": "routing", "text": f"Routing task {task_id} to {model}…"}
        if name == Events.MODEL_LOADED:
            return {"kind": "model", "text": f"Loading {payload.get('model_id', 'model')}…"}
        if name == Events.MODEL_REUSED:
            return {"kind": "model", "text": f"Using loaded model {payload.get('model_id', 'model')}…"}
        if name == Events.TASK_STARTED:
            desc = payload.get("description", payload.get("task_id", "task"))
            return {"kind": "execute_task", "text": f"Executing: {desc[:50]}…"}
        if name == Events.TASK_WAITING:
            task_id = payload.get("task_id", "")
            deps = ", ".join(payload.get("depends_on", []))
            return {"kind": "waiting", "text": f"Task {task_id} waiting for {deps}…"}
        if name == Events.FILE_WRITTEN:
            return {"kind": "create_file", "text": f"Creating file: {payload.get('path', 'file')}…"}
        if name == Events.FILE_EDITED:
            return {"kind": "edit_file", "text": f"Editing file: {payload.get('path', 'file')}…"}
        if name == Events.FILE_READ:
            return {"kind": "read_files", "text": f"Reading file: {payload.get('path', 'file')}…"}
        if name == Events.TESTS_RUN:
            return {"kind": "run_tests", "text": f"Running tests: {payload.get('suite', 'test suite')}…"}
        if name == Events.RESULT_VALIDATED:
            status = payload.get("status", "valid")
            return {"kind": "validate_result", "text": f"Validating result: {status}…"}
        if name == Events.TASK_COMPLETED:
            task_id = payload.get("task_id", "")
            ms = payload.get("latency_ms", 0)
            return {"kind": "complete", "text": f"Task {task_id} completed ({ms:.0f}ms)" if task_id else f"Step completed ({ms:.0f}ms)"}
        if name == Events.TASK_FAILED:
            return {"kind": "error", "text": f"Task failed: {payload.get('reason', 'unknown error')}"}
        if name == Events.PIPELINE_STARTED:
            return {"kind": "pipeline", "text": "Starting multi-agent pipeline…"}
        if name == Events.PIPELINE_STAGE_STARTED:
            return {"kind": "stage", "text": f"Stage: {payload.get('description', 'processing')}…"}
        if name == Events.PIPELINE_STAGE_COMPLETED:
            return {"kind": "stage_done", "text": f"Completed: {payload.get('description', 'stage')}"}
        if name == Events.TOOL_CALLED:
            return {"kind": "tool", "text": f"Using tool: {payload.get('tool_name', 'tool')}…"}
        if name == Events.REQUEST_COMPLETED:
            return {"kind": "finished", "text": "Finished"}
        return None


def _dump(response: Any) -> dict:
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        return dump()
    return response


def _encode(step: dict) -> str:
    return f"data: {json.dumps(step)}\n\n"