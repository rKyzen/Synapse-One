"""Path resolution — the single place that knows where Synapse stores things.

No module anywhere in the codebase may hardcode a filesystem path. Everything
routes through these resolvers so the layout can change per-OS, per-profile, or
via SYNAPSE_HOME without touching feature code.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _default_home() -> Path:
    """OS-aware default internal data home (respects XDG on Linux).

    This is Synapse's own private storage root — it is **never** exposed to
    the user as a workspace. User project folders live wherever the user
    picks them; this tree holds only Synapse metadata (projects.db, chats,
    memory, logs, cache, settings, embeddings).
    """
    override = os.environ.get("SYNAPSE_HOME")
    if override:
        return Path(override).expanduser()

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
        return Path(base) / "Synapse"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Synapse"
    # linux / bsd
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "synapse"


@dataclass(frozen=True)
class SynapsePaths:
    """Resolved, immutable directory layout for the application.

    The whole tree is **internal Synapse storage** (``%LOCALAPPDATA%/Synapse``
    on Windows). User project workspaces are plain folders anywhere on disk;
    only their paths are recorded in ``projects.db``. Per the Phase 7
    architecture, nothing user-facing is ever stored here — chats, memory,
    logs, caches, and embeddings are keyed by project id instead.
    """

    home: Path
    config_dir: Path
    data_dir: Path
    cache_dir: Path
    logs_dir: Path

    def ensure(self) -> None:
        for p in (
            self.config_dir,
            self.data_dir,
            self.cache_dir,
            self.logs_dir,
            self.settings_dir,
            self.chats_dir,
            self.goals_dir,
            self.notes_dir,
            self.documents_dir,
            self.todos_dir,
            self.memory_dir,
            self.embeddings_dir,
            self.actions_dir,
            self.temp_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def config_file(self, name: str = "config.toml") -> Path:
        return self.config_dir / name

    # -- internal storage helpers -------------------------------------------
    #
    # The Phase 7 internal layout mirrors the documented contract:
    #   projects.db   — SQLite project registry (workspace paths only)
    #   chats/        — per-project chat records (<pid>/<chat_id>.json)
    #   memory/       — per-project WorkspaceMemory scopes
    #   embeddings/   — per-project vector stores + uploaded file catalog
    #   logs/actions/ — per-project JSONL audit trails
    #   cache/        — transient data (cleaned on shutdown)
    #   settings/     — application settings files

    @property
    def projects_db(self) -> Path:
        """SQLite registry storing only each project's workspace path."""
        return self.data_dir / "projects.db"

    @property
    def projects_root(self) -> Path:
        """Internal home for projects created without a user-chosen parent."""
        return self.data_dir / "projects"

    @property
    def chats_dir(self) -> Path:
        return self.data_dir / "chats"

    @property
    def goals_dir(self) -> Path:
        """Per-project goal records (AI Operating Workspace, Phase A)."""
        return self.data_dir / "goals"

    @property
    def notes_dir(self) -> Path:
        """Per-project notes (AI Operating Workspace, Phase B)."""
        return self.data_dir / "notes"

    @property
    def documents_dir(self) -> Path:
        """Per-project documents (AI Operating Workspace, Phase B)."""
        return self.data_dir / "documents"

    @property
    def todos_dir(self) -> Path:
        """Per-project todo items (AI Operating Workspace, Phase B)."""
        return self.data_dir / "todos"

    @property
    def memory_dir(self) -> Path:
        return self.data_dir / "memory"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def actions_dir(self) -> Path:
        return self.logs_dir / "actions"

    @property
    def settings_dir(self) -> Path:
        return self.config_dir / "settings"

    @property
    def temp_dir(self) -> Path:
        return self.cache_dir / "tmp"

    @property
    def session_file(self) -> Path:
        return self.data_dir / "session.json"

    @classmethod
    def discover(
        cls,
        home: Path | None = None,
        config_dir: Path | None = None,
    ) -> "SynapsePaths":
        home = home or _default_home()
        config = config_dir or home / "config"
        return cls(
            home=home,
            config_dir=config,
            data_dir=home / "data",
            cache_dir=home / "cache",
            logs_dir=home / "logs",
        )


def bundled_config_dir() -> Path:
    """Locate the sample config shipped with the source tree, if present.

    Resolves relative to this file (src/synapse/config/) up to the repo root's
    config/ directory. Returns an empty path when not in a source checkout.
    """
    candidate = Path(__file__).resolve().parents[3] / "config"
    return candidate if (candidate / "config.toml").exists() else Path()
