"""Chat request/response entities used across the provider interface."""

from __future__ import annotations

from pydantic import BaseModel, Field

from synapse.domain.enums import ProviderKind


class ChatMessage(BaseModel):
    role: str
    content: str
    #: Phase 4 — base64-encoded image payloads (raw or data-URI) attached to a
    #: user message. Providers translate to their vendor wire format; models
    #: without vision support reject or ignore them.
    images: list[str] = Field(default_factory=list)


class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ProviderMetrics(BaseModel):
    """Latency/throughput observed by the provider during one generation.

    ``first_token_latency_s`` — seconds from request to the first content token.
    ``total_latency_s``       — seconds for the whole generation (wall clock).
    ``tokens_generated``      — tokens produced (when the provider reports them).
    ``tokens_per_second``     — tokens_generated / total_latency_s.
    ``interrupted``           — generation ended early (timeout, drop).
    ``interrupt_reason``      — "connection", "read", "generation" or transport.
    """

    first_token_latency_s: float | None = None
    total_latency_s: float | None = None
    tokens_generated: int | None = None
    tokens_per_second: float | None = None
    interrupted: bool = False
    interrupt_reason: str | None = None


class ChatRequest(BaseModel):
    """A normalized chat request — the only shape providers understand.

    Providers translate this into their vendor wire format and translate the
    response back into ChatResponse. The Master never sees vendor formats.
    """

    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = None
    stream: bool = False
    model: str | None = None
    #: Optional structured-output hint: for Ollama this is the JSON-schema
    #: object (``format``) forcing constrained decoding; cloud providers that
    #: lack the feature simply ignore it. Never part of prompt text.
    format: dict | None = None


class ChatResponse(BaseModel):
    """Normalized chat completion from any provider."""

    provider_id: str
    model_id: str
    kind: ProviderKind
    content: str
    usage: Usage = Field(default_factory=Usage)
    metrics: ProviderMetrics = Field(default_factory=ProviderMetrics)
    raw: dict | None = None
