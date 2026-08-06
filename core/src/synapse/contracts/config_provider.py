"""Configuration provider contract.

Any concrete source (TOML, env, secrets, profiles) must implement this. The
rest of the system consumes settings through this abstraction only, so future
config sources (user profiles, DB-backed settings) can be added without
touching consumers.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ConfigProvider(Protocol):
    def get(self, key: str, default: Any = None) -> Any:
        """Return a config value by dotted key, e.g. ``providers.openai.base_url``."""
