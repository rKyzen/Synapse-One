"""Ollama provider — the default local-first model provider.

Talks to a local Ollama server (configurable base_url, default localhost:11434)
over its REST API. Converts normalized ChatRequest <-> Ollama wire format.
"""

from __future__ import annotations

import json
import time

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
    ProviderMetrics,
    Usage,
)
from synapse.events import EventBus
from synapse.providers.base import HttpBaseProvider

_OLLAMA_EP = "http://localhost:11434"


def _strip_data_uri(image: str) -> str:
    """Return raw base64 from a data-URI like ``data:image/png;base64,xxxx``."""
    if image.startswith("data:") and "," in image:
        return image.split(",", 1)[1]
    return image


class OllamaProvider(HttpBaseProvider):
    """Implements ModelProvider for a local Ollama server."""

    provider_id = "ollama"
    kind = ProviderKind.LOCAL

    # Base capability surface for Ollama's general-purpose models. Per-model
    # refinement happens in to_metadata using advertised metadata.
    _BASE_CAPS = {
        "chat": 0.7,
        "reasoning": 0.6,
        "coding": 0.6,
        "writing": 0.7,
        "math": 0.6,
        "translation": 0.6,
        "planning": 0.5,
        "vision": False,
        "tools": False,
    }

    #: Safe defaults for models without a config profile: embedding models can
    #: never serve chat, vision models advertise vision, everything else gets a
    #: conservative neutral profile so unknowns never dominate routing.
    _EMBEDDING_CAPS = {
        "chat": 0.0,
        "reasoning": 0.0,
        "coding": 0.0,
        "writing": 0.0,
        "math": 0.0,
        "embeddings": 1.0,
        "vision": False,
        "tools": False,
    }
    _VISION_CAPS = {
        "chat": 0.6,
        "reasoning": 0.5,
        "coding": 0.25,
        "writing": 0.5,
        "math": 0.4,
        "vision": True,
        "ocr": 1.0,
        "pdf": 1.0,
        "tools": False,
    }
    _NEUTRAL_CAPS = {
        "chat": 0.5,
        "reasoning": 0.4,
        "coding": 0.4,
        "writing": 0.5,
        "math": 0.4,
        "translation": 0.4,
        "planning": 0.4,
        "vision": False,
        "tools": False,
    }

    @classmethod
    def _default_caps(cls, model_id: str) -> dict:
        """Capability fallback for a discovered model with no config profile."""
        lowered = model_id.lower()
        if any(hint in lowered for hint in ("embed", "bge", "mxbai", "minilm", "e5-", "gte", "arctic")):
            return dict(cls._EMBEDDING_CAPS)
        if any(hint in lowered for hint in ("-vl", "vl:", "vision", "llava", "minicpm", "moondream", "gemma3")):
            return dict(cls._VISION_CAPS)
        return dict(cls._NEUTRAL_CAPS)

    def __init__(self, config: ConfigProvider, events: EventBus, endpoint: ProviderEndpoint) -> None:
        super().__init__(config, events, endpoint)

    def _build_client(self) -> httpx.Client:
        base = self.endpoint.base_url or _OLLAMA_EP
        return httpx.Client(base_url=base, timeout=self._client_timeout())

    # -- discovery ----------------------------------------------------------

    def list_models(self) -> list[ModelDescriptor]:
        resp = self.client.get("/api/tags")
        resp.raise_for_status()
        tags = resp.json().get("models", [])
        return [
            ModelDescriptor(
                id=m["name"],  # full name (e.g. "qwen3:4b") is the stable id
                provider_id=self.provider_id,
                display_name=m["name"],
                size_bytes=m.get("size"),
                metadata={"tags": m.get("name"), "details": m.get("details")},
            )
            for m in tags
        ]

    def to_metadata(self, descriptor: ModelDescriptor) -> ModelMetadata | None:
        """Map a live Ollama model into registry metadata.

        Uses registry/capability defaults from config where available and
        Ollama's own metadata (context length, quantization) when present.
        """
        config_entry = self._config.get(f"models.{descriptor.id}", {})
        if not isinstance(config_entry, dict) or not config_entry:
            # Ollama tags discovered models with ":latest"; config keys are
            # untagged. Match either form so profiles are reused, then fall
            # back to name-based safe defaults.
            base, _, tag = descriptor.id.rpartition(":")
            if tag == "latest":
                config_entry = self._config.get(f"models.{base}", {})
        caps = (
            config_entry.get("capabilities", self._default_caps(descriptor.id))
            if isinstance(config_entry, dict)
            else self._default_caps(descriptor.id)
        )
        context = descriptor.metadata.get("details") or {}
        context_window = context.get("context_length") if isinstance(context, dict) else None
        return ModelMetadata(
            id=descriptor.id,
            provider_id=self.provider_id,
            kind=self.kind,
            display_name=descriptor.display_name,
            role=config_entry.get("role", "") if isinstance(config_entry, dict) else "",
            strengths=list(config_entry.get("strengths", [])) if isinstance(config_entry, dict) else [],
            weaknesses=list(config_entry.get("weaknesses", [])) if isinstance(config_entry, dict) else [],
            speed_tier=config_entry.get("speed_tier", "fast") if isinstance(config_entry, dict) else "fast",
            hardware_tier=config_entry.get("hardware_tier", "") if isinstance(config_entry, dict) else "",
            modalities=list(config_entry.get("modalities", ["text"])) if isinstance(config_entry, dict) else ["text"],
            tools_supported=bool(config_entry.get("tools_supported", caps.get("tools", False))) if isinstance(config_entry, dict) else False,
            context_window=context_window,
            required_ram_gb=float(config_entry.get("required_ram_gb", 0.0)) if isinstance(config_entry, dict) else 0.0,
            required_vram_gb=float(config_entry.get("required_vram_gb", 0.0)) if isinstance(config_entry, dict) else 0.0,
            size_bytes=descriptor.size_bytes,
            latency="fast",
            privacy_score=1.0,
            priority=int(config_entry.get("priority", 0)) if isinstance(config_entry, dict) else 0,
            preferred_tasks=list(config_entry.get("preferred_tasks", [])) if isinstance(config_entry, dict) else [],
            capabilities={
                "chat": caps.get("chat", 0.7),
                "reasoning": caps.get("reasoning", 0.6),
                "coding": caps.get("coding", 0.6),
                "writing": caps.get("writing", 0.7),
                "math": caps.get("math", 0.6),
                "translation": caps.get("translation", 0.6),
                "planning": caps.get("planning", 0.5),
                "debugging": caps.get("debugging", 0.3),
                "architecture": caps.get("architecture", 0.3),
                "ocr": caps.get("ocr", 0.0),
                "pdf": caps.get("pdf", 0.0),
                "json_capability": caps.get("json_capability", 0.3),
                "terminal": caps.get("terminal", 0.3),
                "embeddings": caps.get("embeddings", 0.0),
                "vision": bool(caps.get("vision", False)),
                "tools": bool(caps.get("tools", False)),
            },
        )

    # -- execution ----------------------------------------------------------

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Run a generation, always streaming from Ollama internally.

        Streaming is used even for non-streaming requests because it is the only
        way to measure first-token latency, token throughput, and to keep the
        partial output when generation is interrupted. The aggregated result is
        returned as a single :class:`ChatResponse`.
        """
        model = self._resolve_model(request)
        payload_messages = []
        for m in request.messages:
            msg: dict = {"role": m.role, "content": m.content}
            if m.images:
                msg["images"] = [_strip_data_uri(image) for image in m.images]
            payload_messages.append(msg)
        payload = {
            "model": model,
            "messages": payload_messages,
            "stream": True,
        }
        options = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if options:
            payload["options"] = options
        if request.format is not None:
            # Structured outputs: Ollama constrains decoding to a JSON schema.
            payload["format"] = request.format

        content, usage, metrics = self._run_stream(model, payload)
        return self._response(self.provider_id, model, self.kind, content, usage, metrics=metrics)

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]] | None:
        """Embed texts via ``/api/embed``. Returns None when unavailable.

        ``model`` defaults to the provider's configured embedding model; if
        none is resolvable (nothing embedding-capable installed), returns
        None so callers can degrade to keyword search.
        """
        if not texts:
            return []
        resolved = model or self._embedding_model()
        if not resolved:
            self._log.debug("embed_skipped_no_model", provider_id=self.provider_id)
            return None
        try:
            resp = self.client.post(
                "/api/embed",
                json={"model": resolved, "input": texts},
                timeout=self.endpoint.timeouts.connect + 60.0,
            )
            resp.raise_for_status()
            embeddings = resp.json().get("embeddings")
            if not embeddings:
                return None
            return embeddings
        except Exception:  # noqa: BLE001 - best effort
            self._log.debug("embed_failed", provider_id=self.provider_id, model_id=resolved)
            return None

    def _embedding_model(self) -> str | None:
        """First installed embedding-capable model, or None."""
        try:
            for desc in self.list_models():
                meta = self.to_metadata(desc)
                if meta and meta.capabilities.embeddings > 0.9:
                    return desc.id
        except Exception:  # noqa: BLE001
            return None
        return None

    def _run_stream(self, model: str, payload: dict) -> tuple[str, Usage, ProviderMetrics]:
        """Consume Ollama's newline-delimited JSON stream into normalized output.

        Timeouts are differentiated:
          connectivity  -> ``httpx.ConnectTimeout``  (re-raised: no output)
          read          -> ``httpx.ReadTimeout``     (idle gap, partial kept)
          generation    -> enforced wall-clock budget (partial kept)
        Every timeout is logged with its duration and, when text exists, the
        partial output is returned instead of failing.
        """
        t = self.endpoint.timeouts
        gen_start = time.monotonic()
        deadline = gen_start + t.generation
        parts: list[str] = []
        first_token_s: float | None = None
        done: dict = {}
        interrupted = False
        reason: str | None = None

        try:
            with self.client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if time.monotonic() >= deadline:
                        interrupted = True
                        reason = "generation"
                        self._log.warning(
                            "provider_generation_timeout",
                            model=model,
                            generation_timeout_s=t.generation,
                            duration_s=round(time.monotonic() - gen_start, 3),
                            partial_chars=len("".join(parts)),
                        )
                        break
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    piece = (chunk.get("message") or {}).get("content", "")
                    if piece:
                        if first_token_s is None:
                            first_token_s = time.monotonic() - gen_start
                        parts.append(piece)
                    if chunk.get("done"):
                        done = chunk
                        break
        except httpx.ConnectTimeout as exc:
            self._log.warning(
                "provider_connect_timeout",
                model=model,
                connect_timeout_s=t.connect,
                duration_s=round(time.monotonic() - gen_start, 3),
            )
            raise
        except httpx.ReadTimeout as exc:
            interrupted = True
            reason = "read"
            self._log.warning(
                "provider_read_timeout",
                model=model,
                read_timeout_s=t.read,
                duration_s=round(time.monotonic() - gen_start, 3),
                partial_chars=len("".join(parts)),
            )
        except httpx.TransportError as exc:
            if not parts:
                self._log.error("provider_transport_error", model=model, error=str(exc)[:200])
                raise
            interrupted = True
            reason = "transport"
            self._log.warning(
                "provider_stream_interrupted",
                model=model,
                error=str(exc)[:200],
                duration_s=round(time.monotonic() - gen_start, 3),
                partial_chars=len("".join(parts)),
            )

        # The stream can end without a done chunk right when the budget expires
        # (e.g. the final token arrived as the deadline passed).
        if not interrupted and not done and time.monotonic() >= deadline:
            interrupted = True
            reason = "generation"
            self._log.warning(
                "provider_generation_timeout",
                model=model,
                generation_timeout_s=t.generation,
                duration_s=round(time.monotonic() - gen_start, 3),
                partial_chars=len("".join(parts)),
            )

        total_s = time.monotonic() - gen_start
        prompt_tokens = done.get("prompt_eval_count")
        completion_tokens = done.get("eval_count")
        content = "".join(parts)
        tokens_per_s = round(completion_tokens / total_s, 2) if completion_tokens and total_s > 0 else None
        metrics = ProviderMetrics(
            first_token_latency_s=round(first_token_s, 4) if first_token_s is not None else None,
            total_latency_s=round(total_s, 4),
            tokens_generated=completion_tokens,
            tokens_per_second=tokens_per_s,
            interrupted=interrupted,
            interrupt_reason=reason,
        )
        return content, Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens), metrics

    @staticmethod
    def _strip_latest(model_id: str) -> str:
        """Normalize an id: ``foo:latest`` and ``foo:max`` are ``foo``."""
        base, _, tag = model_id.rpartition(":")
        return base if tag in ("latest", "max") else model_id

    def _resolve_model(self, request: ChatRequest) -> str:
        """Pick the model name to send to Ollama.

        Priority: the routed model (explicit request) > configured default
        (only when actually installed) > first installed model. Never falls
        back to a hardcoded name and never sends an uninstalled model.
        """
        if request.model:
            return request.model
        installed = [d.id for d in self.list_models()]
        if not installed:
            raise RuntimeError("ollama reports no installed models")
        configured = self._config.get("providers.ollama.default_model")
        if configured:
            configured = str(configured)
            for installed_id in installed:
                if self._strip_latest(installed_id) == self._strip_latest(configured):
                    return installed_id
        return installed[0]

    # -- support/health -----------------------------------------------------

    def supports(self, capability: Capability) -> bool:
        if capability in (Capability.CODING, Capability.WRITING, Capability.REASONING, Capability.MATH):
            return True
        if capability == Capability.VISION:
            return any(m.metadata.get("details", {}).get("architecture") == "clip" for m in self.list_models())
        return False

    def _health_ok(self) -> bool:
        resp = self.client.get("/api/tags")
        return resp.status_code == 200

    # -- lifecycle (Phase 2.5+) ----------------------------------------------

    def list_loaded(self) -> dict[str, float]:
        """Return {model_id: ram_gb} for models currently loaded in Ollama.

        Uses the official ``/api/ps`` endpoint. CPU-resident models report
        ``size_vram=0`` and the full ``size`` in system RAM, so fall back to
        ``size`` when VRAM is empty. Returns empty dict on error.
        """
        try:
            resp = self.client.get("/api/ps")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", [])
            result: dict[str, float] = {}
            for m in models:
                name = m.get("name")
                size = m.get("size_vram") or m.get("size") or 0
                if not name or not size:
                    continue
                # bytes -> GB
                result[name] = round(size / (1024 ** 3), 2)
            return result
        except Exception:  # noqa: BLE001 - best effort
            self._log.debug("list_loaded_failed", provider_id=self.provider_id)
            return {}

    def is_loaded(self, model_id: str) -> bool | None:
        loaded = self.list_loaded()
        if model_id in loaded:
            return True
        if loaded:
            return False
        return None

    def load_model(self, model_id: str) -> bool:
        """Pre-load a model using keep_alive=-1 (indefinite)."""
        try:
            # Minimal generate request with keep_alive=-1 to trigger load
            payload = {
                "model": model_id,
                "prompt": "",
                "keep_alive": -1,
                "stream": False,
            }
            resp = self.client.post("/api/generate", json=payload, timeout=30.0)
            return resp.status_code == 200
        except Exception:  # noqa: BLE001
            self._log.debug("load_model_failed", model_id=model_id)
            return False

    def unload_model(self, model_id: str) -> bool:
        """Unload a model using keep_alive=0 (immediate)."""
        try:
            # Empty generate request with keep_alive=0 triggers unload
            payload = {
                "model": model_id,
                "prompt": "",
                "keep_alive": 0,
                "stream": False,
            }
            resp = self.client.post("/api/generate", json=payload, timeout=10.0)
            return resp.status_code == 200
        except Exception:  # noqa: BLE001
            self._log.debug("unload_model_failed", model_id=model_id)
            return False