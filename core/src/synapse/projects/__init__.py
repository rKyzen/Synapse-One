"""Phase 5 — Workspace System.

Provides project-level isolation for files, memory, vector indexes, and
chat histories.  The :class:`WorkspaceSystem` facade is the single entry
point; per-project Workspace and WorkspaceMemory instances are created and
cached lazily on demand.
"""

from synapse.projects.chats import ChatStore
from synapse.projects.manager import ProjectManager
from synapse.projects.system import WorkspaceSystem

__all__ = ["ChatStore", "ProjectManager", "WorkspaceSystem"]