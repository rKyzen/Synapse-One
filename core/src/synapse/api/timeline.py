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
        if name == Events.RETRIEVAL_RAN:
            return {"kind": "search", "text": "Searching workspace…"}
        if name == Events.MODEL_LOADED:
            return {"kind": "model", "text": f"Loading {payload.get('model_id', 'model')}…"}
        if name == Events.MODEL_REUSED:
            return {"kind": "model", "text": f"Using {payload.get('model_id', 'model')}…"}
        if name == Events.TASK_FAILED:
            return {"kind": "error", "text": "A task failed"}
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