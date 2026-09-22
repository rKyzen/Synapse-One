"""Embedding Engine — batched text vectors from a local embedding model.

Uses the same provider ``embed()`` capability the Phase 3 workspace memory
relies on; the model id comes from ``[workspace] embedding_model`` (default
``nomic-embed-text``). Returns ``None`` when no embedding-capable provider is
available so callers can degrade to keyword search.
"""

from __future__ import annotations

import logging

from synapse.workspace.contracts import EmbeddingEngine as EmbeddingEngineContract

log = logging.getLogger("synapse.workspace.embeddings")


def _embedding_provider(providers, embedding_model: str | None):
    """First provider that exposes a usable embedding model, else (None, None).

    Returns ``(provider, model)``: ``model`` is the configured embedding model
    when it is installed, otherwise ``None`` so the provider auto-resolves the
    best installed embedding model (adaptive to whatever the hardware tier
    installed).
    """
    for provider in providers.all():
        try:
            if getattr(provider, "embed", None) is None:
                continue
            installed = {d.id for d in provider.list_models()}
            if embedding_model and embedding_model in installed:
                return provider, embedding_model
            for desc in provider.list_models():
                meta = provider.to_metadata(desc)
                if meta and meta.capabilities.embeddings > 0.9:
                    return provider, None
        except Exception:  # noqa: BLE001
            continue
    return None, None


class EmbeddingEngine(EmbeddingEngineContract):
    """Batch embedding with size-gated requests and graceful degradation."""

    def __init__(self, providers, embedding_model: str | None, batch_size: int = 16) -> None:
        self._providers = providers
        self._model = embedding_model
        self._batch = max(1, batch_size)

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not texts:
            return []
        provider, model = _embedding_provider(self._providers, self._model)
        if provider is None:
            log.debug("embedding_no_provider")
            return None
        out: list[list[float]] = []
        try:
            for i in range(0, len(texts), self._batch):
                batch = texts[i:i + self._batch]
                vectors = provider.embed(batch, model=model)
                if not vectors or len(vectors) != len(batch):
                    return None
                for v in vectors:
                    if v is None or len(v) == 0:
                        return None
                out.extend(vectors)
        except Exception:  # noqa: BLE001
            log.debug("embedding_engine_failed")
            return None
        return out