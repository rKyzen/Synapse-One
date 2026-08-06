"""ProviderManager — the only handle the Master may use to reach providers.

Exposes ModelProvider instances and their registry metadata. It performs no
routing (that is a later subsystem); it only manages provider lifecycle and
lookup. Vendor names never appear above this layer except via provider_id
strings resolved here.
"""

from __future__ import annotations

import structlog

from synapse.config.settings import ProviderEndpoint
from synapse.contracts import ConfigProvider, ModelProvider
from synapse.domain.enums import ProviderState
from synapse.domain.models import ModelMetadata
from synapse.events import Events
from synapse.providers.factory import ProviderFactory

log = structlog.get_logger("synapse.providers.manager")


class ProviderManager:
    """Owns provider instances, their lifecycle, and model metadata lookups."""

    def __init__(self, config: ConfigProvider, factory: ProviderFactory, events) -> None:
        self._config = config
        self._factory = factory
        self._events = events
        self._providers: dict[str, ModelProvider] = {}
        self._states: dict[str, ProviderState] = {}
        self._metadata: dict[str, ModelMetadata] = {}

    def load_all(self) -> dict[str, ModelProvider]:
        """Instantiate + initialize every enabled provider in config."""
        enabled = {
            pid: ep
            for pid, ep in (self._config.settings.providers or {}).items()
            if ep.enabled
        }
        for provider_id, endpoint in enabled.items():
            try:
                provider = self._factory.create(provider_id, endpoint)
                provider.initialize()
                self._providers[provider_id] = provider
                self._states[provider_id] = ProviderState.READY
                self.index_models(provider)
                log.info("provider_registered", provider_id=provider_id)
            except Exception as exc:  # noqa: BLE001
                self._states[provider_id] = ProviderState.FAILED
                log.exception("provider_failed_to_initialize", provider_id=provider_id, error=str(exc))
        return self._providers

    def index_models(self, provider: ModelProvider) -> None:
        """Notify the bus that a provider's model list is discoverable.

        Metadata enrichment against the registry is a Phase 2+ concern
        (router); today we only publish discovery so subscribers can react.
        """
        self._events.publish(Events.MODELS_DISCOVERED, {"provider_id": provider.provider_id})

    # -- public API (all the Master may use) --------------------------------

    def provider_ids(self) -> list[str]:
        return list(self._providers)

    def get(self, provider_id: str) -> ModelProvider | None:
        return self._providers.get(provider_id)

    def all(self) -> list[ModelProvider]:
        return list(self._providers.values())

    def state(self, provider_id: str) -> ProviderState:
        return self._states.get(provider_id, ProviderState.DISCOVERED)

    def health(self, provider_id: str) -> bool:
        provider = self._providers.get(provider_id)
        return bool(provider and provider.health())

    def shutdown_all(self) -> None:
        for provider in self._providers.values():
            try:
                provider.shutdown()
            except Exception:  # noqa: BLE001
                log.exception("provider_shutdown_error", provider_id=provider.provider_id)
        self._states = {k: ProviderState.SHUTDOWN for k in self._states}
        self._providers.clear()