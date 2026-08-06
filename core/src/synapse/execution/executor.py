"""Executor — performs the routed AI call through the provider interface.

The Executor is the ONLY place the Master's flow touches a ModelProvider, and
it does so through the ModelProvider abstraction provided by ProviderManager.
It knows no vendor details.
"""

from __future__ import annotations

import time

import structlog

from synapse.contracts import Executor
from synapse.domain import ChatMessage, ChatRequest, ChatResponse, RoutingDecision
from synapse.providers.manager import ProviderManager

log = structlog.get_logger("synapse.execution.executor")


class ProviderUnavailable(RuntimeError):
    """Raised when the routed provider cannot be reached."""


class Executor(Executor):
    """Executes a routed request. Replaceable behind the Executor contract
    (e.g. future: streaming, parallel multi-model execution)."""

    def __init__(self, providers: ProviderManager) -> None:
        self._providers = providers

    def execute(
        self,
        routing: RoutingDecision,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        if not routing.provider_id:
            raise ProviderUnavailable("no provider in routing decision")

        provider = self._providers.get(routing.provider_id)
        if provider is None:
            raise ProviderUnavailable(f"provider '{routing.provider_id}' not registered")

        request = ChatRequest(
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=temperature,
            max_tokens=max_tokens,
            model=routing.model_id,
        )
        start = time.perf_counter()
        try:
            response = provider.chat(request)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "provider_call_failed",
                provider_id=routing.provider_id,
                model_id=routing.model_id,
                error=str(exc)[:200],
            )
            raise ProviderUnavailable(f"provider call failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info(
            "provider_call_ok",
            provider_id=routing.provider_id,
            model_id=routing.model_id,
            latency_ms=round(elapsed_ms, 1),
        )
        # Ensure the response carries the routed model id.
        return response.model_copy(update={"model_id": routing.model_id})