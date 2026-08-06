"""Configuration subsystem tests."""

from __future__ import annotations

import pytest

from synapse.config.paths import SynapsePaths
from synapse.config.provider import ConfigProvider


def _write_toml(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_paths_discovery_respects_synapse_home(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNAPSE_HOME", str(tmp_path / "custom"))
    paths = SynapsePaths.discover()
    assert paths.home == tmp_path / "custom"
    assert str(paths.config_dir).endswith("custom\\config")


def test_config_loads_toml(tmp_path):
    cfg = tmp_path / "config" / "config.toml"
    _write_toml(
        cfg,
        '[logging]\nlevel = "DEBUG"\n\n[providers.ollama]\nenabled = true\nbase_url = "http://x"\n',
    )
    paths = SynapsePaths.discover(home=tmp_path / "home", config_dir=cfg.parent)
    provider = ConfigProvider(paths=paths).load()
    assert provider.settings.logging.level == "DEBUG"
    assert provider.get("providers.ollama.base_url") == "http://x"


def test_env_overrides_toml(tmp_path):
    cfg = tmp_path / "config" / "config.toml"
    _write_toml(cfg, '[logging]\nlevel = "INFO"\n')
    paths = SynapsePaths.discover(home=tmp_path / "home", config_dir=cfg.parent)
    env = {"SYNAPSE_LOGGING_LEVEL": "ERROR"}
    provider = ConfigProvider(paths=paths, env=env).load()
    assert provider.settings.logging.level == "ERROR"


def test_secret_file_resolution(tmp_path):
    key_file = tmp_path / "openai.key"
    key_file.write_text("sk-test-abc", encoding="utf-8")
    # Forward slashes are TOML-safe and accepted on Windows by pathlib.
    key_path = str(key_file).replace("\\", "/")
    cfg = tmp_path / "config" / "config.toml"
    _write_toml(
        cfg,
        '[providers.openai]\nenabled = true\napi_key_file = "'
        + key_path
        + '"\n',
    )
    paths = SynapsePaths.discover(home=tmp_path / "home", config_dir=cfg.parent)
    provider = ConfigProvider(paths=paths).load()
    ep = provider.settings.providers["openai"]
    assert ep.api_key() == "sk-test-abc"


def test_defaults_when_no_config(tmp_path):
    paths = SynapsePaths.discover(home=tmp_path / "home", config_dir=tmp_path / "config")
    provider = ConfigProvider(paths=paths).load()
    assert provider.settings.logging.level == "INFO"
    assert provider.settings.name == "synapse"