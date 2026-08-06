"""Model registry contract.

The registry is a read-only catalog the router consults. It is deliberately
decoupled from any specific config source so it can later be backed by SQLite,
the filesystem, or a remote catalog without touching routing code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from synapse.contracts.model_provider import ModelProvider
from synapse.domain.enums import Capability, ProviderKind
from synapse.domain.models import ModelMetadata


class ModelRegistry(ABC):
    @abstractmethod
    def all(self) -> list[ModelMetadata]:
        """Return every known model."""

    @abstractmethod
    def get(self, model_id: str) -> ModelMetadata | None:
        """Look up a single model by its canonical id."""

    @abstractmethod
    def by_provider(self, provider_id: str) -> list[ModelMetadata]:
        """Filter models by provider id."""

    @abstractmethod
    def by_kind(self, kind: ProviderKind) -> list[ModelMetadata]:
        """Filter models by local/cloud kind."""

    @abstractmethod
    def supports(self, capability: Capability) -> list[ModelMetadata]:
        """Models that satisfy a boolean capability."""

    @abstractmethod
    def sync(self, providers: list[ModelProvider]) -> list[ModelMetadata]:
        """Merge live-discovered models from providers into the catalog.

        Catalog (config) entries always win; discovery only ADDS models the
        provider reports as installed but that are not configured. Returns
        the updated full list. Never touches vendor names.
        """
