"""Model registry tests — verify config-driven catalog, zero hardcoded names."""

from __future__ import annotations

import pytest

from synapse.contracts import ModelRegistry
from synapse.domain import Capability, ProviderKind
from synapse.events import EventBus
from synapse.registry import ConfigModelRegistry


@pytest.fixture
def registry(temp_paths, monkeypatch) -> ConfigModelRegistry:
    cfg_file = temp_paths.config_dir / "config.toml"
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(
        """
[models."qwen2.5:7b"]
provider = "ollama"
kind = "local"
capabilities = { coding = 0.8, reasoning = 0.7 }
context_window = 32768

[models."gpt-4o-mini"]
provider = "openai"
kind = "cloud"
capabilities = { vision = true }
estimated_cost_per_1k = 0.00015
""",
        encoding="utf-8",
    )
    return ConfigModelRegistry(_make_config(cfg_file), EventBus())


def _make_config(cfg_file):
    from synapse.config.paths import SynapsePaths
    from synapse.config.provider import ConfigProvider

    paths = SynapsePaths.discover(home=cfg_file.parent.parent, config_dir=cfg_file.parent)
    return ConfigProvider(paths=paths).load()


def test_loads_from_config(registry):
    assert len(registry.all()) == 2


def test_parses_phase25_profile_fields(registry):
    m = registry.get("qwen2.5:7b")
    assert m.capabilities.coding == 0.8
    assert m.capabilities.chat == 0.0  # default when absent
    assert m.priority == 0
    assert m.preferred_tasks == []


def test_get(registry):
    m = registry.get("gpt-4o-mini")
    assert m is not None
    assert m.provider_id == "openai"
    assert m.kind == ProviderKind.CLOUD
    assert m.capabilities.vision is True
    assert m.estimated_cost_per_1k == pytest.approx(0.00015)


def test_by_provider(registry):
    assert len(registry.by_provider("ollama")) == 1


def test_by_kind(registry):
    assert len(registry.by_kind(ProviderKind.LOCAL)) == 1
    assert len(registry.by_kind(ProviderKind.CLOUD)) == 1


def test_supports_capability(registry):
    models = registry.supports(Capability.VISION)
    assert [m.id for m in models] == ["gpt-4o-mini"]


def test_registry_is_contract_compliant(registry):
    assert isinstance(registry, ModelRegistry)


def test_event_published_on_load():
    from synapse.config.paths import SynapsePaths
    from synapse.config.provider import ConfigProvider

    bus = EventBus()
    events = []
    bus.subscribe("model_catalog.loaded", events.append)
    from pathlib import Path

    paths = SynapsePaths.discover(home=Path(".") / "x", config_dir=Path(".") / "y")
    ConfigModelRegistry(ConfigProvider(paths=paths).load(), bus)
    assert len(events) == 1


class FakeProvider:
    provider_id = "ollama"

    def __init__(self, descriptors, metas):
        self._descriptors = descriptors
        self._metas = metas

    def list_models(self):
        return self._descriptors

    def to_metadata(self, descriptor):
        return self._metas.get(descriptor.id)


def test_sync_adds_discovered_models(registry):
    from synapse.domain import ModelDescriptor, ModelMetadata

    provider = FakeProvider(
        [ModelDescriptor(id="qwen3:4b", provider_id="ollama")],
        {
            "qwen3:4b": ModelMetadata(
                id="qwen3:4b",
                provider_id="ollama",
                kind=ProviderKind.LOCAL,
                capabilities={"reasoning": 0.6, "coding": 0.6, "writing": 0.7, "math": 0.6},
            )
        },
    )
    result = registry.sync([provider])
    assert registry.get("qwen3:4b") is not None
    assert len(result) == 3


def test_sync_never_overwrites_catalog(registry):
    from synapse.domain import ModelDescriptor, ModelMetadata

    provider = FakeProvider(
        [ModelDescriptor(id="qwen2.5:7b", provider_id="ollama")],
        {
            "qwen2.5:7b": ModelMetadata(
                id="qwen2.5:7b",
                provider_id="ollama",
                kind=ProviderKind.LOCAL,
                capabilities={"coding": 0.1},  # would downgrade if applied
            )
        },
    )
    registry.sync([provider])
    assert registry.get("qwen2.5:7b").capabilities.coding == 0.8  # catalog wins


def test_sync_skips_unmatching_provider_id(registry):
    from synapse.domain import ModelDescriptor, ModelMetadata

    provider = FakeProvider(
        [ModelDescriptor(id="qwen3:4b", provider_id="other")],
        {},
    )
    registry.sync([provider])
    assert registry.get("qwen3:4b") is None


def test_sync_tolerates_failing_provider(registry):
    class Boom:
        provider_id = "boom"

        def list_models(self):
            raise RuntimeError("down")

    registry.sync([Boom()])
    assert len(registry.all()) == 2


def test_sync_skips_none_metadata(registry):
    from synapse.domain import ModelDescriptor

    provider = FakeProvider([ModelDescriptor(id="qwen3:4b", provider_id="ollama")], {})
    registry.sync([provider])
    assert registry.get("qwen3:4b") is None
