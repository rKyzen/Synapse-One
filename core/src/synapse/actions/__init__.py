"""Action Engine package — Phase 7: backend-executed filesystem behavior.

The Action Engine sits between the planner and the models:

    Planner -> Execution Plan -> Action Engine -> Filesystem Tools
                                     |                   |
                                     v                   v
                                 Summary         Model(s) generate content

Every request is classified into one of the seven ``RequestKind`` kinds. File
kinds never dump code into the chat: the engine plans folders, writes files,
edits, renames, deletes (with confirmation) and searches — all on the real
workspace folder — then reports a short summary.
"""

from __future__ import annotations

from synapse.actions.classifier import (
    classify_request,
    extract_workspace_ops,
    requires_workspace_access,
)
from synapse.actions.engine import ActionEngine
from synapse.domain.enums import RequestKind

__all__ = [
    "ActionEngine",
    "RequestKind",
    "classify_request",
    "extract_workspace_ops",
    "requires_workspace_access",
]
