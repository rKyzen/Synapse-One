"""Decision Engine — combines analyzers + hardware + registry into a Decision.

Answers the core questions:
    - can this stay local?
    - does it require internet?
    - should cloud reasoning be used?
    - which capabilities are REQUIRED (hard gates)?
    - which capabilities are PREFERRED (soft boosts)?
    - which provider kind is preferred?
    - which workspace should be activated?
    - should memory be used? (placeholder, always False in Phase 2)

Phase 2.5: the engine also maps prompt modality (vision/embeddings, detected
by the intent analyzer) into capability requirements, and treats coding tasks
as reasoning-heavy so specialists win where they should.

Pure computation: no I/O, no vendor knowledge.
"""

from __future__ import annotations

from synapse.contracts import DecisionEngine
from synapse.domain import (
    Capability,
    ComplexityResult,
    Decision,
    HardwareProfile,
    IntentResult,
    PrivacyMode,
    PrivacyResult,
    ProviderKind,
    WorkspaceKind,
)
from synapse.domain.models import ModelMetadata

#: Capabilities implied by each intent (hard requirements).
_INTENT_CAPS: dict = {
    "coding": [Capability.CODING, Capability.REASONING],
    "research": [Capability.REASONING],
    "education": [Capability.REASONING],
    "writing": [Capability.WRITING],
    "business": [Capability.REASONING, Capability.WRITING],
    "creative": [Capability.REASONING],
    "planning": [Capability.REASONING],
    "general": [],
    "conversation": [],
}

#: Capabilities that boost (but never exclude) a model for an intent.
_INTENT_PREFERRED: dict = {
    "coding": [Capability.CHAT, Capability.DEBUGGING, Capability.ARCHITECTURE, Capability.JSON, Capability.TERMINAL],
    "research": [Capability.CHAT, Capability.PLANNING, Capability.TOOLS, Capability.LONG_CONTEXT],
    "education": [Capability.CHAT, Capability.PLANNING],
    "writing": [Capability.CHAT, Capability.TRANSLATION, Capability.PLANNING],
    "business": [Capability.CHAT, Capability.PLANNING, Capability.TOOLS],
    "creative": [Capability.CHAT, Capability.WRITING, Capability.PLANNING],
    "planning": [Capability.CHAT, Capability.PLANNING],
    "general": [Capability.CHAT, Capability.WRITING],
    "conversation": [Capability.CHAT],
}

#: Workspace implied by each intent.
_INTENT_WORKSPACE: dict = {
    "coding": WorkspaceKind.CODING,
    "writing": WorkspaceKind.WRITING,
}


class DecisionEngine(DecisionEngine):
    """Deterministic decision engine over analysis results."""

    def decide(
        self,
        intent: IntentResult,
        complexity: ComplexityResult,
        privacy: PrivacyResult,
        hardware: HardwareProfile,
        registry: list[ModelMetadata],
    ) -> Decision:
        caps = list(_INTENT_CAPS.get(intent.primary.value, []))
        preferred = list(_INTENT_PREFERRED.get(intent.primary.value, []))

        # Modality (Phase 2.5): image work REQUIRES a vision specialist;
        # retrieval work prefers (never forces) embeddings-capable models.
        if intent.vision_required:
            caps += [Capability.VISION, Capability.OCR, Capability.PDF]
            reasoning_vision = "modality=vision → vision/ocr/pdf capabilities required"
        else:
            reasoning_vision = None
        if intent.embeddings_required:
            preferred += [Capability.EMBEDDINGS, Capability.LONG_CONTEXT]
            reasoning_emb = "modality=embeddings → embeddings-capable models preferred"
        else:
            reasoning_emb = None

        reasoning: list[str] = []
        reasoning.append(f"intent={intent.primary.value} complexity={complexity.score} privacy={privacy.mode.value}")
        if reasoning_vision:
            reasoning.append(reasoning_vision)
        if reasoning_emb:
            reasoning.append(reasoning_emb)

        internet = privacy.internet_required

        # Provider availability from registry (which kinds are configured).
        kinds = {m.kind for m in registry}
        local_available = ProviderKind.LOCAL in kinds
        cloud_available = ProviderKind.CLOUD in kinds
        reasoning.append(f"providers available: local={local_available}, cloud={cloud_available}")

        # Can it stay local?
        hardware_ok = hardware.recommendations.can_run_local_llm
        can_stay_local = (
            local_available
            and hardware_ok
            and not internet
            and privacy.mode
            in (PrivacyMode.LOCAL_ONLY, PrivacyMode.PREFER_LOCAL, PrivacyMode.BALANCED)
        )
        reasoning.append(f"hardware ok for local: {hardware_ok}")

        # Should cloud reasoning be used?
        use_cloud = (
            internet
            or privacy.mode in (PrivacyMode.PREFER_CLOUD, PrivacyMode.CLOUD_REQUIRED)
            or (complexity.score >= 85 and cloud_available)
        )

        # Preferred provider kind.
        if privacy.mode == PrivacyMode.LOCAL_ONLY and local_available:
            preferred_kind = ProviderKind.LOCAL
        elif privacy.mode == PrivacyMode.CLOUD_REQUIRED and cloud_available:
            preferred_kind = ProviderKind.CLOUD
        elif privacy.mode in (PrivacyMode.PREFER_LOCAL, PrivacyMode.BALANCED):
            preferred_kind = ProviderKind.LOCAL
        else:
            preferred_kind = ProviderKind.CLOUD if cloud_available else ProviderKind.LOCAL

        workspace = _INTENT_WORKSPACE.get(intent.primary.value, WorkspaceKind.GENERAL)

        return Decision(
            can_stay_local=can_stay_local,
            internet_required=internet,
            use_cloud_reasoning=use_cloud,
            use_memory=False,  # placeholder — Memory Engine in a later phase
            privacy=privacy.mode,
            required_capabilities=caps,
            preferred_capabilities=preferred,
            preferred_kind=preferred_kind,
            workspace=workspace,
            reasoning=reasoning,
        )
