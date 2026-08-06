"""Model registry — a config-driven catalog of known models.

The registry contains NO hardcoded model names or vendor specifics. Entries
are loaded from configuration (inline TOML or an external catalog file) and
publish a MODEL_CATALOG_LOADED event. The router consults this later.
"""

from __future__ import annotations

from pathlib import Path

from synapse.contracts import ConfigProvider, ModelRegistry
from synapse.domain.enums import Capability, ProviderKind
from synapse.domain.models import ModelCapabilities, ModelMetadata
from synapse.events import EventBus, Events

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


def _parse_entry(model_id: str, raw: dict) -> ModelMetadata:
    caps = raw.get("capabilities", {})
    kind = ProviderKind(raw.get("kind", "local"))
    metadata = ModelMetadata(
        id=model_id,
        provider_id=raw["provider"],
        kind=kind,
        display_name=raw.get("display_name", model_id),
        context_window=raw.get("context_window"),
        required_ram_gb=float(raw.get("required_ram_gb", 0.0)),
        required_gpu=raw.get("required_gpu"),
        required_vram_gb=float(raw.get("required_vram_gb", 0.0)),
        size_bytes=raw.get("size_bytes"),
        latency=raw.get("latency", "medium"),
        estimated_cost_per_1k=float(raw.get("estimated_cost_per_1k", 0.0)),
        privacy_score=float(raw.get("privacy_score", 1.0)),
        priority=int(raw.get("priority", 0)),
        preferred_tasks=list(raw.get("preferred_tasks", [])),
        average_tokens_per_second=raw.get("average_tokens_per_second"),
        average_latency_s=raw.get("average_latency_s"),
        capabilities=ModelCapabilities(**caps)
        if isinstance(caps, dict)
        else ModelCapabilities(),
    )
    return metadata


class ConfigModelRegistry(ModelRegistry):
    """ModelRegistry implementation backed by layered configuration."""

    def __init__(self, config: ConfigProvider, events: EventBus) -> None:
        self._config = config
        self._events = events
        self._models: dict[str, ModelMetadata] = {}
        self._load()

    def _load(self) -> None:
        raw_models: dict = dict(self._config.settings.models or {})

        catalog_file = self._config.settings.model_catalog_file
        if catalog_file:
            path = Path(catalog_file).expanduser()
            if path.exists():
                with path.open("rb") as fh:
                    raw_models.update(tomllib.load(fh).get("models", {}))

        for model_id, entry in raw_models.items():
            parsed = _parse_entry(model_id, entry)
            self._models[model_id] = parsed

        self._events.publish(Events.MODEL_CATALOG_LOADED, {"count": len(self._models)})

    # -- ModelRegistry contract --------------------------------------------

    def all(self) -> list[ModelMetadata]:
        return list(self._models.values())

    def get(self, model_id: str) -> ModelMetadata | None:
        return self._models.get(model_id)

    def by_provider(self, provider_id: str) -> list[ModelMetadata]:
        return [m for m in self._models.values() if m.provider_id == provider_id]

    def by_kind(self, kind: ProviderKind) -> list[ModelMetadata]:
        return [m for m in self._models.values() if m.kind == kind]

    def supports(self, capability: Capability) -> list[ModelMetadata]:
        return [m for m in self._models.values() if capability in m.supports]

    def sync(self, providers) -> list[ModelMetadata]:
        """Merge live-discovered models into the catalog (never overwrites)."""
        for provider in providers:
            try:
                descriptors = provider.list_models()
            except Exception:  # noqa: BLE001 - a failing provider must not break routing
                continue
            for descriptor in descriptors:
                if descriptor.provider_id != provider.provider_id:
                    continue
                if descriptor.id in self._models:
                    continue
                try:
                    metadata = provider.to_metadata(descriptor)
                except Exception:  # noqa: BLE001
                    continue
                if metadata is None:
                    continue
                self._models[metadata.id] = metadata
                self._events.publish(
                    Events.MODELS_DISCOVERED,
                    {"provider_id": provider.provider_id, "model_id": metadata.id},
                )
        return self.all()