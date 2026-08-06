"""ProviderManager tests."""

from __future__ import annotations

from synapse.config.paths import SynapsePaths
from synapse.config.provider import ConfigProvider
from synapse.domain import ProviderKind
from synapse.events import EventBus
from synapse.providers.factory import ProviderFactory
from synapse.providers.manager import ProviderManager


def _config(tmp_path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.toml").write_text(
        """
[providers.ollama]
enabled = true
base_url = "http://localhost:11434"

[providers.openai]
enabled = false
""",
        encoding="utf-8",
    )
    paths = SynapsePaths.discover(home=tmp_path / "home", config_dir=cfg_dir)
    return ConfigProvider(paths=paths).load()


def test_load_all_initializes_enabled_only(tmp_path, monkeypatch):
    config = _config(tmp_path)
    events = EventBus()
    manager = ProviderManager(config, ProviderFactory(config, events), events)

    # Ollama init builds a client (no network yet); models discovery is deferred.
    providers = manager.load_all()
    assert "ollama" in providers
    assert "openai" not in providers


def test_manager_state_and_shutdown(tmp_path):
    config = _config(tmp_path)
    events = EventBus()
    manager = ProviderManager(config, ProviderFactory(config, events), events)
    manager.load_all()
    assert manager.provider_ids() == ["ollama"]
    manager.shutdown_all()
    assert manager.provider_ids() == []
