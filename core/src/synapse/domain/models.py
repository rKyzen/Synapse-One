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
        cap_val = capability.value if hasattr(capability, "value") else str(capability)
        if cap_val in self.extra:
            val = self.extra[cap_val]
            if isinstance(val, bool):
                return 1.0 if val else 0.0
            if isinstance(val, (int, float)):
                return max(0.0, min(1.0, float(val)))

        if capability in (Capability.REASONING,):
            return self.reasoning
        if capability in (Capability.CODING, Capability.FILE_EDITING):
            return self.coding
        if capability in (Capability.WRITING, Capability.PDF_CREATION, Capability.DOCX_CREATION, Capability.PPT_CREATION):
            return self.writing
        if capability in (Capability.MATH,):
            return self.math
        if capability in (Capability.CHAT, Capability.CONVERSATION):
            return self.chat
        if capability in (Capability.TRANSLATION,):
            return self.translation
        if capability in (Capability.PLANNING, Capability.TASK_MANAGEMENT):
            return self.planning
        if capability in (Capability.DEBUGGING, Capability.TESTING):
            return self.debugging or self.coding
        if capability in (Capability.ARCHITECTURE,):
            return self.architecture
        if capability in (Capability.OCR,):
            return self.ocr
        if capability in (Capability.PDF, Capability.PDF_READING):
            return self.pdf or (1.0 if self.vision else self.ocr)
        if capability in (Capability.JSON,):
            return self.json_capability
        if capability in (Capability.TERMINAL,):
            return self.terminal
        if capability in (Capability.EMBEDDINGS, Capability.MEMORY, Capability.KNOWLEDGE_RETRIEVAL):
            return self.embeddings
        if capability in (Capability.VISION, Capability.IMAGE_UNDERSTANDING):
            return 1.0 if self.vision else 0.0
        if capability in (Capability.TOOLS, Capability.AUTOMATION):
            return 1.0 if self.tools else max(self.terminal, self.coding)
        if capability in (Capability.LANGUAGES,):
            return 1.0 if self.extra.get("languages") else 0.0
        if capability in (Capability.LONG_CONTEXT,):
            return 1.0 if self.extra.get("long_context") else 0.0
        if capability in (Capability.SUMMARIZATION,):
            return max(self.writing, self.chat, self.reasoning)
        if capability in (Capability.RESEARCH,):
            return max(self.reasoning, self.chat)
        if capability in (Capability.CODE_ANALYSIS,):
            return max(self.coding, self.reasoning, self.architecture)
        if capability in (Capability.FILE_READING,):
            return 1.0 if self.tools else max(self.coding, self.chat, 0.5)
        if capability in (Capability.FILE_CREATION,):
            return max(self.coding, self.writing, self.reasoning)
        if capability in (Capability.DOCUMENT_ANALYSIS,):
            return max(self.pdf, self.writing, self.reasoning)
        if capability in (Capability.SPREADSHEET_CREATION,):
            return max(self.math, self.json_capability, self.writing)
        if capability in (Capability.DATA_ANALYSIS,):
            return max(self.math, self.reasoning, self.coding)
        if capability in (Capability.PROJECT_CREATION,):
            return max(self.coding, self.architecture, self.planning)
        if capability in (Capability.PROJECT_ANALYSIS,):
            return max(self.architecture, self.coding, self.reasoning)
        if capability in (Capability.CITATIONS,):
            return max(self.reasoning, self.writing)
        if capability in (Capability.WEB_SEARCH,):
            return 1.0 if self.tools else max(self.reasoning, self.chat)
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
