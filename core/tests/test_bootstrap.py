"""Bootstrap / composition-root tests."""

from __future__ import annotations

from synapse.bootstrap import Boot, create_container
from synapse.contracts import HardwareProvider, ModelProvider, ModelRegistry
from synapse.di import Container
from synapse.events import EventBus
from synapse.providers.manager import ProviderManager


def test_create_container_wires_graph(container):
    assert isinstance(container, Container)
    # All key abstractions resolve.
    assert container.resolve(HardwareProvider) is not None
    assert container.resolve(ModelRegistry) is not None
    assert container.resolve(ProviderManager) is not None
    assert container.resolve(EventBus) is not None


def test_boot_entry_points(container):
    boot = Boot(container)
    assert boot.registry.all()  # config catalog parsed (may be empty w/o config)
    assert boot.hardware.scan().os_name
    boot.start()
    boot.shutdown()
