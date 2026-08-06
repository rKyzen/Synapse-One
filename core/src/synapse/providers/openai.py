"""OpenAI-compatible provider — the cloud fallback (and a template for any
vendor exposing a Chat Completions-compatible API).

The OpenAI wire protocol is used by many hosted providers (OpenAI, Azure
OpenAI, Together, Groq, etc.), so this class is written to be reusable as a
base for future cloud providers through config (base_url + api key) alone.
"""

from __future__ import annotations

import httpx

from synapse.config.settings import ProviderEndpoint
from synapse.contracts import ConfigProvider
from synapse.domain import (
    Capability,
    ChatRequest,
    ChatResponse,
    ModelDescriptor,
    ModelMetadata,
    ProviderKind,
    Usage,
)
from synapse.events import EventBus
from synapse.providers.base import HttpBaseProvider

_OPENAI_EP = "https://api.openai.com/v1"


class OpenAIChatCompatibleProvider(HttpBaseProvider):
    """Chat Completions-compatible cloud provider."""

    provider_id = "openai"
    kind = ProviderKind.CLOUD

    _DEFAULT_MODEL = "gpt-4o-mini"
    _BASE_CAPS = {
        "chat": 0.9,
        "reasoning": 0.8,
        "coding": 0.85,
        "writing": 0.9,
        "math": 0.85,
        "translation": 0.85,
        "planning": 0.7,
        "debugging": 0.8,
        "architecture": 0.8,
        "ocr": 0.6,
        "pdf": 0.5,
        "json_capability": 0.9,
        "terminal": 0.5,
        "vision": False,
        "tools": True,
    }

    def __init__(self, config: ConfigProvider, events: EventBus, endpoint: ProviderEndpoint) -> None:
        super().__init__(config, events, endpoint)

    def _build_client(self) -> httpx.Client:
        base = self.endpoint.base_url or _OPENAI_EP
        api_key = self.endpoint.api_key()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        return httpx.Client(base_url=base, timeout=self.endpoint.options.get("timeout", 60.0), headers=headers)

    # -- discovery ----------------------------------------------------------

    def list_models(self) -> list[ModelDescriptor]:
        # Chat Completions has no open "list models" endpoint for all vendors;
        # return configured models from the catalog instead of failing.
        config_models = self._config.get("models", {}) or {}
        return [
            ModelDescriptor(
                id=mid,
                provider_id=self.provider_id,
                display_name=cfg.get("display_name", mid),
                metadata={},
            )
            for mid, cfg in config_models.items()
            if isinstance(cfg, dict) and cfg.get("provider") == self.provider_id
        ]

    def to_metadata(self, descriptor: ModelDescriptor) -> ModelMetadata | None:
        cfg = self._config.get(f"models.{descriptor.id}", {}) or {}
        caps = cfg.get("capabilities", {}) if isinstance(cfg.get("capabilities"), dict) else {}
        return ModelMetadata(
            id=descriptor.id,
            provider_id=self.provider_id,
            kind=self.kind,
            display_name=cfg.get("display_name", descriptor.id),
            context_window=cfg.get("context_window"),
            latency=cfg.get("latency", "medium"),
            estimated_cost_per_1k=float(cfg.get("estimated_cost_per_1k", 0.0)),
            privacy_score=float(cfg.get("privacy_score", 0.0)),
            priority=int(cfg.get("priority", 0)),
            preferred_tasks=list(cfg.get("preferred_tasks", [])),
            capabilities={
                "chat": caps.get("chat", 0.9),
                "reasoning": caps.get("reasoning", 0.8),
                "coding": caps.get("coding", 0.85),
                "writing": caps.get("writing", 0.9),
                "math": caps.get("math", 0.85),
                "translation": caps.get("translation", 0.85),
                "planning": caps.get("planning", 0.7),
                "debugging": caps.get("debugging", 0.8),
                "architecture": caps.get("architecture", 0.8),
                "ocr": caps.get("ocr", 0.6),
                "pdf": caps.get("pdf", 0.5),
                "json_capability": caps.get("json_capability", 0.9),
                "terminal": caps.get("terminal", 0.5),
                "embeddings": caps.get("embeddings", 0.0),
                "vision": bool(caps.get("vision", False)),
                "tools": bool(caps.get("tools", True)),
            },
        )

    # -- execution ----------------------------------------------------------

    def chat(self, request: ChatRequest) -> ChatResponse:
        model = self._resolve_model(request)
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": request.stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        resp = self.client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "") or ""
        u = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=u.get("prompt_tokens"),
            completion_tokens=u.get("completion_tokens"),
            total_tokens=u.get("total_tokens"),
        )
        return self._response(self.provider_id, model, self.kind, content, usage, raw=data)

    def _resolve_model(self, request: ChatRequest) -> str:
        if request.model:
            return request.model
        configured = self._config.get("providers.openai.default_model")
        return str(configured) if configured else self._DEFAULT_MODEL

    # -- support/health -----------------------------------------------------

    def supports(self, capability: Capability) -> bool:
        # Cloud models are capable across the board; specific gating happens in
        # the router via registry metadata.
        return capability in (Capability.REASONING, Capability.CODING, Capability.WRITING, Capability.MATH)

    def _health_ok(self) -> bool:
        resp = self.client.get("/models")
        return resp.status_code in (200, 401)  # 401 = key configured but forbidden; endpoint alive