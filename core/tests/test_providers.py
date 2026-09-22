"""Provider tests — validate contract compliance and wire translation offline."""

from __future__ import annotations

import pytest
import httpx
from unittest import mock

from synapse.config.paths import SynapsePaths
from synapse.config.provider import ConfigProvider
from synapse.config.settings import ProviderEndpoint, ProviderTimeouts
from synapse.contracts import ModelProvider
from synapse.domain import ChatRequest, ChatMessage, ModelDescriptor
from synapse.events import EventBus
from synapse.providers.factory import ProviderFactory
from synapse.providers.gemini import GeminiProvider
from synapse.providers.ollama import OllamaProvider
from synapse.providers.openai import OpenAIChatCompatibleProvider


def _config(tmp_path, toml_text=""):
    paths = SynapsePaths.discover(home=tmp_path / "home", config_dir=tmp_path / "config")
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.toml").write_text(toml_text, encoding="utf-8")
    return ConfigProvider(paths=paths).load()


def _ep(base_url="http://localhost:11434", api_key=None, generation_timeout=None):
    timeouts = ProviderTimeouts()
    if generation_timeout is not None:
        timeouts = ProviderTimeouts(generation=generation_timeout)
    return ProviderEndpoint(enabled=True, base_url=base_url, api_key_env=None, timeouts=timeouts)


class FakeResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json = json_data

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")
        return None


class FakeClient:
    """Minimal httpx.Client stand-in for provider tests."""

    def __init__(self, responses: dict[str, FakeResponse] | None = None):
        self._responses = responses or {}
        self.posted: list[tuple[str, dict]] = []
        self.streamed: list[tuple[str, str, dict]] = []
        self.closed = False

    def get(self, url, **kwargs):
        return self._responses.get("get", FakeResponse(200, {}))

    def post(self, url, json=None, **kwargs):
        self.posted.append((url, json))
        return self._responses.get("post", FakeResponse(200, {"message": {"content": "hi"}}))

    def stream(self, method, url, json=None, **kwargs):
        self.streamed.append((method, url, json))
        return self._responses.get("stream", FakeStreamResponse())

    def close(self):
        self.closed = True


class FakeStreamResponse:
    """Fake httpx streaming response: NDJSON lines, optional mid-stream failure."""

    def __init__(self, lines: list[str] | None = None, status_code: int = 200, interrupt_after: int | None = None, interrupt_exc: Exception | None = None):
        self._lines = lines or []
        self.status_code = status_code
        self._interrupt_after = interrupt_after
        self._interrupt_exc = interrupt_exc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_lines(self):
        if self._interrupt_after is not None and self._interrupt_after <= 0:
            raise self._interrupt_exc
        yielded = 0
        for line in self._lines:
            yield line
            yielded += 1
            if self._interrupt_after is not None and yielded >= self._interrupt_after:
                raise self._interrupt_exc


@pytest.fixture
def ollama(tmp_path):
    provider = OllamaProvider(_config(tmp_path), EventBus(), _ep())
    provider._client = FakeClient(
        {
            "get": FakeResponse(200, {"models": [{"name": "llama3.2:latest", "size": 123}]}),
            "stream": FakeStreamResponse(
                [
                    '{"model":"m","message":{"role":"assistant","content":"hello "},"done":false}',
                    '{"model":"m","message":{"role":"assistant","content":"from ollama"},"done":false}',
                    '{"model":"m","message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":5,"eval_count":3}',
                ]
            ),
        }
    )
    provider._ready = True
    return provider


def test_providers_satisfy_contract():
    for cls in (OllamaProvider, OpenAIChatCompatibleProvider, GeminiProvider):
        assert issubclass(cls, ModelProvider)
        for method in ("initialize", "list_models", "chat", "health", "supports", "shutdown", "to_metadata"):
            assert callable(getattr(cls, method)), f"{cls.__name__} missing {method}"


def test_ollama_chat_translates_wire_format(ollama):
    response = ollama.chat(ChatRequest(messages=[ChatMessage(role="user", content="ping")]))
    assert response.provider_id == "ollama"
    assert response.content == "hello from ollama"
    assert response.usage.prompt_tokens == 5
    # Verify the wire payload matches Ollama's /api/chat shape.
    method, url, payload = ollama._client.streamed[0]
    assert method == "POST"
    assert "/api/chat" in url
    assert payload["messages"][0] == {"role": "user", "content": "ping"}
    assert payload["stream"] is True


def test_ollama_list_models(ollama):
    models = ollama.list_models()
    assert models[0].id == "llama3.2:latest"
    assert models[0].provider_id == "ollama"


def test_ollama_to_metadata_reuses_untagged_config_profile(tmp_path):
    provider = OllamaProvider(
        _config(
            tmp_path,
            """
[models."nomic-embed-text"]
provider = "ollama"
capabilities = { embeddings = 1.0, chat = 0.0 }
""",
        ),
        EventBus(),
        _ep(),
    )
    metadata = provider.to_metadata(ModelDescriptor(id="nomic-embed-text:latest", provider_id="ollama"))
    assert metadata is not None
    assert metadata.capabilities.embeddings == 1.0
    assert metadata.capabilities.chat == 0.0
    # Router gate therefore excludes it from chat — never a 400 "does not support chat".


def test_ollama_to_metadata_embedding_defaults_when_unknown(tmp_path):
    provider = OllamaProvider(_config(tmp_path), EventBus(), _ep())
    metadata = provider.to_metadata(ModelDescriptor(id="bge-m3:latest", provider_id="ollama"))
    assert metadata is not None
    assert metadata.capabilities.embeddings == 1.0
    assert metadata.capabilities.chat == 0.0


def test_ollama_to_metadata_vision_defaults_when_unknown(tmp_path):
    provider = OllamaProvider(_config(tmp_path), EventBus(), _ep())
    metadata = provider.to_metadata(ModelDescriptor(id="qwen2.5vl:7b", provider_id="ollama"))
    assert metadata is not None
    assert metadata.capabilities.vision is True
    assert metadata.capabilities.ocr == 1.0


def test_ollama_to_metadata_neutral_defaults_when_unknown(tmp_path):
    provider = OllamaProvider(_config(tmp_path), EventBus(), _ep())
    metadata = provider.to_metadata(ModelDescriptor(id="some-random-model:7b", provider_id="ollama"))
    assert metadata is not None
    assert metadata.capabilities.chat == 0.5  # conservative — never dominates
    assert metadata.capabilities.embeddings == 0.0


def test_ollama_uses_routed_model(ollama):
    response = ollama.chat(ChatRequest(messages=[ChatMessage(role="user", content="ping")], model="qwen3:4b"))
    method, url, payload = ollama._client.streamed[0]
    assert payload["model"] == "qwen3:4b"
    assert response.model_id == "qwen3:4b"


def test_ollama_without_routed_model_uses_first_installed(ollama):
    response = ollama.chat(ChatRequest(messages=[ChatMessage(role="user", content="ping")]))
    method, url, payload = ollama._client.streamed[0]
    assert payload["model"] == "llama3.2:latest"
    assert response.model_id == "llama3.2:latest"


def test_ollama_default_model_falls_back_when_not_installed(tmp_path):
    toml = '[providers.ollama]\ndefault_model = "qwen2.5:3b"\n'
    provider = OllamaProvider(_config(tmp_path, toml), EventBus(), _ep())
    provider._client = FakeClient(
        {"get": FakeResponse(200, {"models": [{"name": "llama3.2:latest", "size": 123}]})}
    )
    provider._ready = True
    response = provider.chat(ChatRequest(messages=[ChatMessage(role="user", content="ping")]))
    assert response.model_id == "llama3.2:latest"


def test_ollama_default_model_matches_untagged_installed(tmp_path):
    toml = '[providers.ollama]\ndefault_model = "llama3.2"\n'
    provider = OllamaProvider(_config(tmp_path, toml), EventBus(), _ep())
    provider._client = FakeClient(
        {"get": FakeResponse(200, {"models": [{"name": "llama3.2:latest", "size": 123}]})}
    )
    provider._ready = True
    response = provider.chat(ChatRequest(messages=[ChatMessage(role="user", content="ping")]))
    # the configured default matches the installed :latest tag; the real
    # installed name is sent (never an uninstalled tag)
    assert response.model_id == "llama3.2:latest"


def test_ollama_health(ollama):
    assert ollama.health() is True


def test_ollama_streaming_reports_metrics(ollama):
    response = ollama.chat(ChatRequest(messages=[ChatMessage(role="user", content="ping")]))
    assert response.content == "hello from ollama"
    assert response.metrics.total_latency_s is not None
    assert response.metrics.total_latency_s >= 0
    assert response.metrics.first_token_latency_s is not None
    assert response.metrics.tokens_generated == 3
    assert response.metrics.tokens_per_second > 0
    assert response.metrics.interrupted is False
    assert response.metrics.interrupt_reason is None


def test_ollama_read_timeout_returns_partial(tmp_path):
    provider = OllamaProvider(_config(tmp_path), EventBus(), _ep())
    provider._client = FakeClient(
        {
            "stream": FakeStreamResponse(
                ['{"model":"m","message":{"role":"assistant","content":"Hello"},"done":false}'],
                interrupt_after=1,
                interrupt_exc=httpx.ReadTimeout("idle gap"),
            )
        }
    )
    provider._ready = True
    response = provider.chat(ChatRequest(messages=[ChatMessage(role="user", content="q")], model="qwen3:4b"))
    assert response.content == "Hello"
    assert response.metrics.interrupted is True
    assert response.metrics.interrupt_reason == "read"


def test_ollama_connect_timeout_raises(tmp_path):
    provider = OllamaProvider(_config(tmp_path), EventBus(), _ep())
    provider._client = FakeClient(
        {"stream": FakeStreamResponse([], interrupt_after=0, interrupt_exc=httpx.ConnectTimeout("no connection"))}
    )
    provider._ready = True
    with pytest.raises(httpx.ConnectTimeout):
        provider.chat(ChatRequest(messages=[ChatMessage(role="user", content="q")], model="qwen3:4b"))


def test_ollama_generation_timeout_returns_partial(tmp_path, monkeypatch):
    import synapse.providers.ollama as ollama_mod

    provider = OllamaProvider(_config(tmp_path), EventBus(), _ep(generation_timeout=0.5))
    provider._client = FakeClient(
        {
            "stream": FakeStreamResponse(
                [
                    '{"model":"m","message":{"role":"assistant","content":"partial "},"done":false}',
                    '{"model":"m","message":{"role":"assistant","content":"answer"},"done":false}',
                ]
            )
        }
    )
    provider._ready = True

    # Simulate time running out: the two chunks arrive instantly, then the
    # clock jumps far past the generation budget before the third read.
    reads = iter([100.0, 100.0, 100.0, 100.0, 300.0, 300.0, 300.0])
    monkeypatch.setattr(ollama_mod.time, "monotonic", lambda: next(reads))

    response = provider.chat(ChatRequest(messages=[ChatMessage(role="user", content="q")], model="qwen3:4b"))
    assert response.content == "partial answer"
    assert response.metrics.interrupted is True
    assert response.metrics.interrupt_reason == "generation"


def test_ollama_default_timeouts_are_generous():
    ep = ProviderEndpoint()
    assert ep.timeouts.generation == 180.0
    assert ep.timeouts.read == 180.0
    assert ep.timeouts.connect == 10.0


def test_ollama_timeouts_from_toml(tmp_path):
    config = _config(
        tmp_path,
        """
[providers.ollama.timeouts]
connect = 5
read = 300
generation = 600
""",
    )
    endpoint = config.settings.providers["ollama"]
    assert endpoint.timeouts.connect == 5
    assert endpoint.timeouts.read == 300
    assert endpoint.timeouts.generation == 600


def test_openai_chat_translates(monkeypatch, tmp_path):
    provider = OpenAIChatCompatibleProvider(_config(tmp_path), EventBus(), _ep("https://api.openai.com/v1"))
    provider._client = FakeClient(
        {
            "get": FakeResponse(200, {"data": []}),
            "post": FakeResponse(
                200,
                {
                    "choices": [{"message": {"content": "cloud reply"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                },
            ),
        }
    )
    provider._ready = True
    response = provider.chat(ChatRequest(messages=[ChatMessage(role="user", content="q")]))
    assert response.kind.value == "cloud"
    assert response.content == "cloud reply"
    url, payload = provider._client.posted[0]
    assert "/chat/completions" in url


def test_gemini_chat_translates(monkeypatch, tmp_path):
    provider = GeminiProvider(_config(tmp_path), EventBus(), _ep())
    provider._client = FakeClient(
        {
            "post": FakeResponse(
                200,
                {
                    "candidates": [{"content": {"parts": [{"text": "gemini says hi"}]}}],
                    "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2},
                },
            )
        }
    )
    provider._ready = True
    response = provider.chat(ChatRequest(messages=[ChatMessage(role="user", content="q")]))
    assert response.content == "gemini says hi"


def test_factory_creates_known(tmp_path):
    factory = ProviderFactory(_config(tmp_path), EventBus())
    assert set(factory.known_ids()) == {"ollama", "openai", "gemini"}
    assert isinstance(factory.create("ollama", _ep()), OllamaProvider)


def test_factory_rejects_unknown(tmp_path):
    factory = ProviderFactory(_config(tmp_path), EventBus())
    with pytest.raises(KeyError):
        factory.create("does-not-exist", _ep())
