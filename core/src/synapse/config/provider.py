"""The concrete ConfigProvider: merges defaults + TOML + env + secret files.

Implements the ConfigProvider protocol and exposes the fully-merged,
type-validated SynapseSettings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from synapse.config.paths import SynapsePaths
from synapse.config.settings import ENV_PREFIX, Settings

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _flatten(prefix: str, d: dict, acc: dict[str, str]) -> None:
    """Flatten nested dict to dotted keys for env lookup."""
    for key, value in d.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _flatten(dotted, value, acc)
        else:
            acc[dotted.upper()] = str(value)


class ConfigProvider:
    """Layered, validated configuration access.

    Layers (low→high): defaults < TOML < env vars < secret files.
    Supports dotted key lookup for future dynamic consumers and resolves the
    final, validated :class:`SynapseSettings`.
    """

    def __init__(
        self,
        paths: SynapsePaths,
        *,
        env: dict[str, str] | None = None,
        toml: dict | None = None,
    ) -> None:
        self._paths = paths
        self._env = env if env is not None else os.environ
        self._raw: dict = {}
        self._settings: Settings | None = None
        if toml is not None:
            self._raw = toml

    # -- loading -----------------------------------------------------------

    def load(self) -> "ConfigProvider":
        """Load and merge all layers. Returns self for chaining.

        Config file search order:
            1. ``<SYNAPSE_HOME>/config/config.toml`` (user override)
            2. bundled sample config from the source checkout (fallback)
            3. defaults only
        """
        merged: dict = {}
        toml_file = self._paths.config_file()
        candidates = [toml_file]
        from synapse.config.paths import bundled_config_dir

        sample = bundled_config_dir() / "config.toml"
        if not toml_file.exists() and sample.exists():
            candidates.insert(0, sample)
        for candidate in candidates:
            if candidate.exists():
                with candidate.open("rb") as fh:
                    merged = _deep_merge(merged, tomllib.load(fh))
        # env overrides (only keys that already exist, to avoid polluting)
        self._raw = _apply_env(merged, self._env)
        self._settings = Settings.from_raw(self._raw, self._paths)
        return self

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            raise RuntimeError("ConfigProvider.load() must be called before access")
        return self._settings

    # -- ConfigProvider protocol -------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Dotted-key lookup into the merged raw config."""
        node: Any = self._raw
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _apply_env(raw: dict, env: dict[str, str]) -> dict:
    """Overlay env vars matching ``SYNAPSE_<DOTTED.KEY>`` onto raw config.

    Only keys already present in raw are overridden (or new keys explicitly
    named by prefix). Secret file paths are resolved here too.
    """
    out = _deep_merge({}, raw)

    flat: dict[str, str] = {}
    _flatten("", raw, flat)

    for dotted_key, _ in flat.items():
        env_key = f"{ENV_PREFIX}{dotted_key.replace('.', '_')}"
        if env_key in env:
            _set_dotted(out, dotted_key.lower(), _coerce(env[env_key]))

    # resolve secret file indirection: if a value points at an existing file
    # ending in common secret markers, keep as-is; resolution happens in settings.
    return out


def _coerce(value: str) -> Any:
    """Best-effort type coercion for env-provided scalars."""
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        if lowered.startswith("0x"):
            return int(value, 16)
        if value.isdigit():
            return int(value)
        return float(value)
    except ValueError:
        return value


def _set_dotted(target: dict, dotted: str, value: Any) -> None:
    node = target
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
