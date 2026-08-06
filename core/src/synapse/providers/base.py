"""Shared base class for HTTP-based model providers.

Holds the common pieces (config access, an httpx client, small helpers) so each
vendor provider stays small and focuses only on its wire format. Providers are
strictly bound to the ModelProvider contract.
"""

from __future__ import annotations

import httpx
import structlog

from synapse.config.settings import ProviderEndpoint
from synapse.contracts import ConfigProvider, ModelProvider
from synapse.domain import Capability, ChatRequest, ChatResponse, ModelDescriptor, ProviderKind, ProviderMetrics, Usage
from synapse.events import EventBus, Events

log = structlog.get_logger("synapse.providers")


class HttpBaseProvider(ModelProvider):
    """Common scaffold for REST-based providers.

    Subclasses set ``provider_id`` and ``kind`` and implement
    ``list_models``/``chat``/``supports``/``to_metadata``/
    ``_health_ok``. Connection details come from config, never hardcoded.
    """

    provider_id = "abstract"
    kind = ProviderKind.LOCAL

    def __init__(self, config: ConfigProvider, events: EventBus, endpoint: ProviderEndpoint) -> None:
        self._config = config
        self._events = events
        self.endpoint = endpoint
        self._client: httpx.Client | None = None
        self._ready = False
        self._log = structlog.get_logger(f"synapse.providers.{self.provider_id}")

    # lifecycle -------------------------------------------------------------

    def initialize(self) -> None:
        self._client = self._build_client()
        self._ready = True
        self._events.publish(Events.PROVIDER_INITIALIZED, {"provider_id": self.provider_id})
        self._log.info("provider_initialized", provider_id=self.provider_id)

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._ready = False
        self._log.info("provider_shutdown", provider_id=self.provider_id)

    # helpers ---------------------------------------------------------------

    def _client_timeout(self) -> httpx.Timeout:
        t = self.endpoint.timeouts
        # No overall httpx limit: the generation budget is enforced by the
        # provider (it can return partial output), while connect/read/write stay
        # per-phase so a slow-but-alive CPU stream is never killed.
        return httpx.Timeout(None, connect=t.connect, read=t.read, write=t.connect, pool=t.connect)

    def _build_client(self) -> httpx.Client:
        headers = {}
        return httpx.Client(timeout=self._client_timeout(), headers=headers)

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            raise RuntimeError(f"provider '{self.provider_id}' not initialized")
        return self._client

    # contract bits that subclasses supply ---------------------------------

    def _health_ok(self) -> bool:
        raise NotImplementedError

    def health(self) -> bool:
        if not self._ready:
            return False
        try:
            return self._health_ok()
        except Exception as exc:  # noqa: BLE001
            self._log.warning("health_check_failed", provider_id=self.provider_id, reason=str(exc)[:200])
            return False

    @staticmethod
    def _response(
        provider_id: str,
        model_id: str,
        kind: ProviderKind,
        content: str,
        usage: Usage | None = None,
        raw: dict | None = None,
        metrics: ProviderMetrics | None = None,
    ) -> ChatResponse:
        return ChatResponse(
            provider_id=provider_id,
            model_id=model_id,
            kind=kind,
            content=content,
            usage=usage or Usage(),
            metrics=metrics or ProviderMetrics(),
            raw=raw,
        )