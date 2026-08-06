"""ProviderFactory — the single place that maps provider_id -> concrete class.

Adding a provider means: (1) one new class in ``providers/`` implementing
ModelProvider, (2) one entry here, (3) config entries. Nothing else changes.
"""

from __future__ import annotations

from synapse.config.settings import ProviderEndpoint
from synapse.contracts import ConfigProvider
from synapse.events import EventBus
from synapse.providers.gemini import GeminiProvider
from synapse.providers.ollama import OllamaProvider
from synapse.providers.openai import OpenAIChatCompatibleProvider

_REGISTRY: dict[str, type] = {
    "ollama": OllamaProvider,
    "openai": OpenAIChatCompatibleProvider,
    "gemini": GeminiProvider,
}


class ProviderFactory:
    def __init__(self, config: ConfigProvider, events: EventBus) -> None:
        self._config = config
        self._events = events

    @classmethod
    def known_ids(cls) -> list[str]:
        return list(_REGISTRY)

    def create(self, provider_id: str, endpoint: ProviderEndpoint):
        cls = _REGISTRY.get(provider_id)
        if cls is None:
            raise KeyError(f"unknown provider id: {provider_id!r}")
        return cls(self._config, self._events, endpoint)