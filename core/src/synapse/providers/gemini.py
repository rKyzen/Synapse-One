"""Gemini (Google) provider — a cloud provider using the gemini-1.5 REST API.

Reuses the OpenAI-compatible transport scaffold; only the endpoint path,
request/response shapes differ. Kept as its own class so it can diverge
(streaming, thinking modes) without affecting the OpenAI adapter.
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

_GEMINI_EP = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_MODEL = "gemini-1.5-flash"


class GeminiProvider(HttpBaseProvider):
    provider_id = "gemini"
    kind = ProviderKind.CLOUD

    _BASE_CAPS = {
        "chat": 0.85,
        "reasoning": 0.75,
        "coding": 0.8,
        "writing": 0.85,
        "math": 0.8,
        "translation": 0.8,
        "planning": 0.7,
        "debugging": 0.7,
        "architecture": 0.7,
        "ocr": 0.7,
        "pdf": 0.6,
        "json_capability": 0.8,
        "terminal": 0.5,
        "vision": False,
        "tools": True,
    }

    def __init__(self, config: ConfigProvider, events: EventBus, endpoint: ProviderEndpoint) -> None:
        super().__init__(config, events, endpoint)

    def _build_client(self) -> httpx.Client:
        return httpx.Client(timeout=self.endpoint.options.get("timeout", 60.0))

    def _api_key(self) -> str:
        return self.endpoint.api_key() or ""

    # -- discovery ----------------------------------------------------------

    def list_models(self) -> list[ModelDescriptor]:
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
                "chat": caps.get("chat", 0.85),
                "reasoning": caps.get("reasoning", 0.75),
                "coding": caps.get("coding", 0.8),
                "writing": caps.get("writing", 0.85),
                "math": caps.get("math", 0.8),
                "translation": caps.get("translation", 0.8),
                "planning": caps.get("planning", 0.7),
                "debugging": caps.get("debugging", 0.7),
                "architecture": caps.get("architecture", 0.7),
                "ocr": caps.get("ocr", 0.7),
                "pdf": caps.get("pdf", 0.6),
                "json_capability": caps.get("json_capability", 0.8),
                "terminal": caps.get("terminal", 0.5),
                "embeddings": caps.get("embeddings", 0.0),
                "vision": bool(caps.get("vision", False)),
                "tools": bool(caps.get("tools", True)),
            },
        )

    # -- execution ----------------------------------------------------------

    def chat(self, request: ChatRequest) -> ChatResponse:
        model = self._resolve_model(request)
        url = f"{_GEMINI_EP}/models/{model}:generateContent?key={self._api_key()}"
        payload = {
            "contents": [
                {
                    "parts": [{"text": m.content}],
                    "role": ("model" if m.role == "assistant" else "user"),
                }
                for m in request.messages
            ]
        }
        if request.temperature is not None:
            payload["generationConfig"] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            payload["generationConfig"] = payload.get("generationConfig", {}) | {"maxOutputTokens": request.max_tokens}

        resp = self.client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        content = "".join(p.get("text", "") for p in parts)
        usage = Usage(
            prompt_tokens=data.get("usageMetadata", {}).get("promptTokenCount"),
            completion_tokens=data.get("usageMetadata", {}).get("candidatesTokenCount"),
        )
        return self._response(self.provider_id, model, self.kind, content, usage, raw=data)

    def _resolve_model(self, request: ChatRequest) -> str:
        if request.model:
            return request.model
        configured = self._config.get("providers.gemini.default_model")
        return str(configured) if configured else _DEFAULT_MODEL

    # -- support/health -----------------------------------------------------

    def supports(self, capability: Capability) -> bool:
        return capability in (Capability.REASONING, Capability.CODING, Capability.WRITING, Capability.MATH)

    def _health_ok(self) -> bool:
        return bool(self._api_key())