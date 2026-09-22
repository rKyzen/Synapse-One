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
    role: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    speed_tier: str = ""
    hardware_tier: str = ""
    modalities: list[str] = Field(default_factory=lambda: ["text"])
    tools_supported: bool = False

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


def format_model_registry_summary(
    models: list[ModelMetadata],
    tier: str | None = None,
    available_ram_gb: float | None = None,
    vram_gb: float | None = None,
) -> str:
    """Format active models into structured ground-truth catalog for the Master Model prompt."""
    lines: list[str] = [
        "### REGISTERED SPECIALIST MODELS & TOOLS",
        "The following models and tools are registered on this system. You MUST select from these models/tools:",
        "",
    ]
    for m in models:
        # derive strong capabilities
        strong_caps: list[str] = []
        if m.capabilities.coding >= 0.6:
            strong_caps.append("coding")
        if m.capabilities.math >= 0.6:
            strong_caps.append("math")
        if m.capabilities.reasoning >= 0.6:
            strong_caps.append("reasoning")
        if m.capabilities.writing >= 0.6:
            strong_caps.append("writing")
        if m.capabilities.planning >= 0.6:
            strong_caps.append("agent_planning")
        if m.capabilities.vision:
            strong_caps.append("vision")
        if m.capabilities.tools or m.tools_supported:
            strong_caps.append("tools")
        if m.capabilities.chat >= 0.6:
            strong_caps.append("conversation")
        if m.capabilities.embeddings >= 0.6:
            strong_caps.append("embeddings")

        cap_str = ", ".join(strong_caps) if strong_caps else "general"
        strengths_str = "; ".join(m.strengths) if m.strengths else "General capability"
        weaknesses_str = "; ".join(m.weaknesses) if m.weaknesses else "None specific"
        modalities_str = ", ".join(m.modalities) if m.modalities else "text"
        speed = m.speed_tier or m.latency.value
        tool_support = "Yes" if (m.tools_supported or m.capabilities.tools) else "No"
        res = f"~{m.required_ram_gb}GB RAM"
        if m.required_vram_gb > 0:
            res += f", ~{m.required_vram_gb}GB VRAM"

        lines.append(f"Model: {m.id}")
        if m.role:
            lines.append(f"  Role: {m.role}")
        lines.append(f"  Capabilities: {cap_str}")
        lines.append(f"  Strengths: {strengths_str}")
        lines.append(f"  Weaknesses: {weaknesses_str}")
        lines.append(f"  Context Limit: {m.context_window or 32768} tokens")
        lines.append(f"  Modality: {modalities_str}")
        lines.append(f"  Approx Resource Requirements: {res}")
        lines.append(f"  Speed Characteristics: {speed}")
        if m.hardware_tier:
            lines.append(f"  Hardware Tier: {m.hardware_tier}")
        lines.append(f"  Tool / Function-calling Support: {tool_support}")
        lines.append("")

    lines.append("### AVAILABLE DETERMINISTIC TOOLS")
    lines.append("Tool: filesystem_tool")
    lines.append("  Role: File and folder operations (create, read, edit, delete, rename files)")
    lines.append("  Capability: file_creation, file_editing, file_reading")
    lines.append("  Resource Requirements: 0 GB (native filesystem)")
    lines.append("")
    lines.append("Tool: test_runner")
    lines.append("  Role: Project unit test execution and validation")
    lines.append("  Capability: testing, validation")
    lines.append("  Resource Requirements: 0 GB (local test subprocess)")
    lines.append("")

    return "\n".join(lines)


class ModelDescriptor(BaseModel):
    """A live model as reported by a provider (e.g. Ollama tags endpoint)."""

    id: str
    provider_id: str
    display_name: str = ""
    size_bytes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
