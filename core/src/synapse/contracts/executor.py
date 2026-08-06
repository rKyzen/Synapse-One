"""Executor contract — runs a routed request against a provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from synapse.domain.diagnosis import RoutingDecision
from synapse.domain.requests import ChatResponse


class Executor(ABC):
    @abstractmethod
    def execute(
        self,
        routing: RoutingDecision,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Execute a prompt via the routed provider. Returns a normalized
        ChatResponse. Raises ProviderUnavailable when the provider is missing
        or fails."""
