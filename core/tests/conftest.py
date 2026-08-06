"""Shared fixtures."""

from __future__ import annotations

import os

import pytest

from synapse.bootstrap import create_container
from synapse.config.paths import SynapsePaths
from synapse.di import Container


def pytest_configure(config):  # noqa: ANN001
    """Keep `import synapse.api` from auto-booting real services in tests."""
    os.environ.setdefault("SYNAPSE_TEST", "1")


@pytest.fixture
def temp_paths(tmp_path) -> SynapsePaths:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(
        """
[logging]
level = "INFO"

[providers.ollama]
enabled = true
base_url = "http://localhost:11434"

[providers.openai]
enabled = false

[models."llama3.2"]
provider = "ollama"
kind = "local"
capabilities = { coding = 0.5, reasoning = 0.5 }
""",
        encoding="utf-8",
    )
    return SynapsePaths.discover(
        home=tmp_path / "home",
        config_dir=config_dir,
    )


@pytest.fixture
def container(temp_paths, monkeypatch) -> Container:
    """A wired container that ignores real env config."""
    monkeypatch.setenv("SYNAPSE_HOME", str(temp_paths.home))
    return create_container(paths=temp_paths)
