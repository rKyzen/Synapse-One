"""The one interface the Master/planner/routing layers talk to.

NO vendor-specific names or shapes may ever leak out of a ModelProvider
implementation. The Master communicates only with this interface (through
ProviderManager). Adding a provider == implementing this one class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from synapse.domain.enums import Capability, ProviderKind
from synapse.domain.models import ModelDescriptor, ModelMetadata
from synapse.domain.requests import ChatRequest, ChatResponse


class ModelProvider(ABC):
    """Hot-swappable contract implemented by every model provider.

    All methods are synchronous: providers are thin adapters around local
    sockets or HTTP, and a full async transport layer is a later concern.
    The router/master depends on this abstraction only.
    """

    #: Stable identifier, e.g. "ollama" / "openai" / "gemini".
    provider_id: str = "abstract"

    kind: ProviderKind = ProviderKind.LOCAL

    @abstractmethod
    def initialize(self) -> None:
        """Prepare the provider (connect, load discovery data).

        Must be idempotent and safe to call repeatedly.
        """

    @abstractmethod
    def list_models(self) -> list[ModelDescriptor]:
        """Discover currently available models from this provider."""

    @abstractmethod
    def chat(self, request: ChatRequest) -> ChatResponse:
        """Execute a normalized chat completion."""

    @abstractmethod
    def health(self) -> bool:
        """True when the provider can serve requests right now."""

    @abstractmethod
    def supports(self, capability: Capability) -> bool:
        """Whether the provider offers the requested capability at all."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources. Idempotent."""

    @abstractmethod
    def to_metadata(self, descriptor: ModelDescriptor) -> ModelMetadata | None:
        """Map a live model descriptor into registry metadata, or None.

        This is the only place vendor-specific capability knowledge may live.
        """

    # -- optional lifecycle (Phase 2.5+) -------------------------------------

    def list_loaded(self) -> dict[str, float]:
        """Return ``{model_id: ram_gb}`` for models currently resident, if known.

        Providers that cannot report live load state (or for which the API is
        unavailable) return an empty dict. Used by the ModelLifecycleManager to
        populate memory usage and to detect models that became loaded outside
        Synapse (e.g. via ``ollama run``).
        """
        return {}

    def is_loaded(self, model_id: str) -> bool | None:
        """True/False when known; None when the provider cannot tell."""
        loaded = self.list_loaded()
        if model_id in loaded:
            return True
        if model_id not in loaded:
            return None
        return None

    def load_model(self, model_id: str) -> bool:
        """Hint the provider to make ``model_id`` resident (best-effort).

        Default: no-op, returns False. Providers that support pre-loading (e.g.
        Ollama ``keep_alive: -1``) override this. Returning True means the load
        request was accepted (loading is asynchronous server-side).
        """
        return False

    def unload_model(self, model_id: str) -> bool:
        """Ask the provider to unload ``model_id`` (best-effort).

        Default: no-op, returns False. Providers that support unloading (e.g.
        Ollama ``keep_alive: 0``) override this. Returning True is not a hard
        guarantee the model is evicted, only that the request was accepted.
        """
        return False

    # -- optional embeddings (Phase 3: workspace memory) --------------------

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]] | None:
        """Embed ``texts`` into vectors, or None when unsupported/unavailable.

        ``model`` names a specific embedding model when the caller has one
        configured; otherwise the provider picks its own default. A provider
        that cannot embed (or has no embedding model installed) returns None —
        callers must degrade gracefully.
        """
        return None
