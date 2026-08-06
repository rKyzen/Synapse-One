"""Model-related domain entities."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from synapse.domain.enums import Capability, LatencyTier, ProviderKind


class ModelCapabilities(BaseModel):
    """Weighted capability profile of a model, consulted by the router.

    Scored capabilities run 0.0..1.0 (e.g. ``coding = 0.95``). A model can
    also carry free-form extra capability data without schema changes. Boolean
    capabilities (``vision``, ``tools``) say whether the model supports them
    at all. ``chat`` marks conversational models: embedding-only models carry
    ``chat = 0.0`` and are never selected for text generation.
    """

    reasoning: float = Field(default=0.0, ge=0.0, le=1.0)
    coding: float = Field(default=0.0, ge=0.0, le=1.0)
    writing: float = Field(default=0.0, ge=0.0, le=1.0)
    math: float = Field(default=0.0, ge=0.0, le=1.0)
    vision: bool = False
    tools: bool = False
    chat: float = Field(default=0.0, ge=0.0, le=1.0)
    translation: float = Field(default=0.0, ge=0.0, le=1.0)
    planning: float = Field(default=0.0, ge=0.0, le=1.0)
    debugging: float = Field(default=0.0, ge=0.0, le=1.0)
    architecture: float = Field(default=0.0, ge=0.0, le=1.0)
    ocr: float = Field(default=0.0, ge=0.0, le=1.0)
    pdf: float = Field(default=0.0, ge=0.0, le=1.0)
    json_capability: float = Field(default=0.0, ge=0.0, le=1.0)
    terminal: float = Field(default=0.0, ge=0.0, le=1.0)
    embeddings: float = Field(default=0.0, ge=0.0, le=1.0)
    extra: dict[str, Any] = Field(default_factory=dict)

    def score_for(self, capability: Capability) -> float:
        """Profile strength for a capability: 0.0..1.0 (booleans map to 0/1)."""
        if capability == Capability.REASONING:
            return self.reasoning
        if capability == Capability.CODING:
            return self.coding
        if capability == Capability.WRITING:
            return self.writing
        if capability == Capability.MATH:
            return self.math
        if capability == Capability.CHAT:
            return self.chat
        if capability == Capability.TRANSLATION:
            return self.translation
        if capability == Capability.PLANNING:
            return self.planning
        if capability == Capability.DEBUGGING:
            return self.debugging
        if capability == Capability.ARCHITECTURE:
            return self.architecture
        if capability == Capability.OCR:
            return self.ocr
        if capability == Capability.PDF:
            return self.pdf
        if capability == Capability.JSON:
            return self.json_capability
        if capability == Capability.TERMINAL:
            return self.terminal
        if capability == Capability.EMBEDDINGS:
            return self.embeddings
        if capability == Capability.VISION:
            return 1.0 if self.vision else 0.0
        if capability == Capability.TOOLS:
            return 1.0 if self.tools else 0.0
        if capability == Capability.LANGUAGES:
            return 1.0 if self.extra.get("languages") else 0.0
        return 0.0


class ModelMetadata(BaseModel):
    """Registry metadata for one installed/configured model.

    This is the canonical description of a model the router can reason about:
    weighted capability profile, resource requirements, known performance
    characteristics, and routing hints. Everything is config-driven — the
    router never hardcodes model ids or names.
    """

    id: str
    provider_id: str
    kind: ProviderKind
    display_name: str = ""
    context_window: int | None = None
    required_ram_gb: float = 0.0
    required_gpu: str | None = None
    required_vram_gb: float = 0.0
    size_bytes: int | None = None
    latency: LatencyTier = LatencyTier.MEDIUM
    estimated_cost_per_1k: float = 0.0
    privacy_score: float = Field(default=1.0, ge=0.0, le=1.0)
    priority: int = Field(default=0, ge=0, le=100)
    preferred_tasks: list[str] = Field(default_factory=list)
    average_tokens_per_second: float | None = None
    average_latency_s: float | None = None
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)

    @property
    def supports(self) -> set[Capability]:
        """Derived set of capabilities for cheap filtering."""
        caps = set[Capability]()
        for capability in Capability:
            if capability == Capability.LONG_CONTEXT:
                if self.context_window and self.context_window >= 128_000:
                    caps.add(Capability.LONG_CONTEXT)
            elif capability == Capability.LANGUAGES:
                if self.capabilities.extra.get("languages"):
                    caps.add(Capability.LANGUAGES)
            elif self.capabilities.score_for(capability) > 0:
                caps.add(capability)
        return caps


class ModelDescriptor(BaseModel):
    """A live model as reported by a provider (e.g. Ollama tags endpoint)."""

    id: str
    provider_id: str
    display_name: str = ""
    size_bytes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
