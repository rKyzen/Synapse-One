"""Hardware tier resolver — dynamically assigns the Master Agent model role.

Pure, deterministic, offline. Given a scanned HardwareProfile, resolves the
hardware tier and the ordered list of candidate models (per tier) that should
serve as the Master AI / Task Divider. The resolver knows nothing about
providers or registries — the orchestrator maps candidates to actually
installed, healthy models.

Tiers (feature spec, all thresholds configurable):

- **Tier 1** (< 12 GB available RAM, no dedicated VRAM): small chat models
  (``qwen2.5:1.5b`` / ``llama3.2:1b``).
- **Tier 2** (12–24 GB available RAM): ``qwen2.5:3b`` / ``llama3.2:3b``.
- **Tier 3** (> 24 GB available RAM or dedicated VRAM): ``qwen2.5:7b`` /
  ``llama3.1:8b``.
- **Cloud fallback** (available RAM < 6 GB): a cloud API model
  (``gpt-4o-mini`` / ``gemini-2.5-flash``) to prevent system lockups.

Model ids are NOT hardcoded defaults in behavior: they come from config
(``master_agent.tiers.*``) with these values as built-in defaults so the
feature runs unconfigured but stays reconfigurable.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from synapse.contracts import ConfigProvider
from synapse.domain.hardware import HardwareProfile


class HardwareTier(str, Enum):
    """Hardware tiers for the Master Agent model role."""

    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"
    TIER3_PLUS = "tier3_plus"
    CLOUD_FALLBACK = "cloud_fallback"


#: built-in candidate models per tier (overridden by config).
#: EXACTLY ONE default Master Model per hardware tier (small & fast).
_DEFAULT_CANDIDATES: dict[HardwareTier, list[str]] = {
    HardwareTier.TIER1: ["gemma3:1b"],
    HardwareTier.TIER2: ["gemma3:4b"],
    HardwareTier.TIER3: ["gemma3:4b"],
    HardwareTier.TIER3_PLUS: ["gemma3:4b"],
    HardwareTier.CLOUD_FALLBACK: ["gpt-4o-mini", "gemini-2.5-flash"],
}

#: resolver thresholds (GB) — configurable via ``master_agent.resolver.*``.
_DEFAULT_THRESHOLDS = {
    "tier2_min_ram_gb": 6.0,
    "tier3_min_ram_gb": 16.0,
    "tier3_min_vram_gb": 12.0,
    "tier3_plus_min_ram_gb": 32.0,
    "tier3_plus_min_vram_gb": 20.0,
    "cloud_max_ram_gb": 0.5,
}


class TierAssignment(BaseModel):
    """Result of resolving a HardwareProfile to a Master Agent tier."""

    tier: HardwareTier
    reason: str
    #: candidate model ids in preference order (as configured for the tier).
    candidate_models: list[str] = Field(default_factory=list)


class TierResolver:
    """Deterministic hardware → tier mapping. No providers, no I/O."""

    def __init__(
        self,
        config: ConfigProvider | None = None,
        *,
        thresholds: dict[str, float] | None = None,
        candidates: dict[HardwareTier, list[str]] | None = None,
    ) -> None:
        self._thresholds = dict(_DEFAULT_THRESHOLDS)
        self._candidates = dict(_DEFAULT_CANDIDATES)
        if config is not None:
            raw_t: dict | None = config.get("master_agent.resolver")
            if isinstance(raw_t, dict):
                for key, value in raw_t.items():
                    if not isinstance(value, (int, float)):
                        continue
                    self._thresholds[str(key)] = float(value)
            raw_c: dict | None = config.get("master_agent.tiers")
            if isinstance(raw_c, dict):
                for tier in HardwareTier:
                    listed = raw_c.get(tier.value)
                    if listed is None and tier is HardwareTier.CLOUD_FALLBACK:
                        listed = raw_c.get("cloud")  # "cloud" is the friendly key
                    if isinstance(listed, list):
                        ids = [str(m) for m in listed if isinstance(m, str) and m]
                        if ids:
                            self._candidates[tier] = ids
        if thresholds:
            self._thresholds.update(thresholds)
        if candidates:
            self._candidates.update(candidates)

    # -- public -------------------------------------------------------------

    def resolve(self, hardware: HardwareProfile) -> TierAssignment:
        """Map a hardware profile to a tier + ordered candidate models."""
        available_gb = float(hardware.memory.available_gb or 0.0)
        vram_gb = float(hardware.gpu.vram_gb) if hardware.gpu and hardware.gpu.vram_gb else 0.0
        t = self._thresholds

        if available_gb < t["cloud_max_ram_gb"]:
            return TierAssignment(
                tier=HardwareTier.CLOUD_FALLBACK,
                reason=(
                    f"{available_gb:.1f}GB available RAM below "
                    f"{t['cloud_max_ram_gb']:.1f}GB — cloud master model required "
                    "(prevents local OOM lockups)"
                ),
                candidate_models=list(self._candidates[HardwareTier.CLOUD_FALLBACK]),
            )
        if available_gb >= t["tier3_plus_min_ram_gb"] or vram_gb >= t["tier3_plus_min_vram_gb"]:
            tier = HardwareTier.TIER3_PLUS
            reason = (
                f"{available_gb:.1f}GB available RAM"
                + (f" (or {vram_gb:.0f}GB dedicated VRAM)" if vram_gb else "")
                + " — high-end workstation tier with optional deep reasoning"
            )
        elif available_gb >= t["tier3_min_ram_gb"] or vram_gb >= t["tier3_min_vram_gb"]:
            tier = HardwareTier.TIER3
            reason = (
                f"{available_gb:.1f}GB available RAM"
                + (f" (or {vram_gb:.0f}GB dedicated VRAM)" if vram_gb else "")
                + " — high-end local tier"
            )
        elif available_gb >= t["tier2_min_ram_gb"]:
            tier = HardwareTier.TIER2
            reason = f"{available_gb:.1f}GB available RAM — mid-size local tier"
        else:
            tier = HardwareTier.TIER1
            reason = f"{available_gb:.1f}GB available RAM — lightweight local tier"
        return TierAssignment(
            tier=tier,
            reason=reason,
            candidate_models=list(self._candidates[tier]),
        )

    def candidates_for(self, tier: HardwareTier) -> list[str]:
        """Ordered candidate model ids for one tier (config or defaults)."""
        return list(self._candidates.get(tier, []))

    def can_support_model(
        self,
        model_required_ram_gb: float | str,
        hardware: HardwareProfile | float,
        safety_buffer_gb: float | HardwareProfile = 0.5,
    ) -> bool:
        """Check if hardware safely has enough memory to load a model.

        Supports both:
          can_support_model(required_ram_gb, hardware_profile, safety_buffer=0.5)
          can_support_model(model_name, required_ram_gb, hardware_profile, safety_buffer=0.5)
        """
        if isinstance(model_required_ram_gb, str):
            actual_ram = float(hardware) if isinstance(hardware, (int, float)) else 0.0
            actual_hw = safety_buffer_gb if isinstance(safety_buffer_gb, HardwareProfile) else None
            actual_buffer = 0.5
        else:
            actual_ram = float(model_required_ram_gb)
            actual_hw = hardware if isinstance(hardware, HardwareProfile) else None
            actual_buffer = float(safety_buffer_gb) if isinstance(safety_buffer_gb, (int, float)) else 0.5

        if actual_hw is None:
            return True
        available_gb = float(actual_hw.memory.available_gb or 0.0)
        vram_gb = float(actual_hw.gpu.vram_gb) if actual_hw.gpu and actual_hw.gpu.vram_gb else 0.0
        if vram_gb > 0 and actual_ram <= vram_gb:
            return True
        return available_gb >= (actual_ram + actual_buffer)