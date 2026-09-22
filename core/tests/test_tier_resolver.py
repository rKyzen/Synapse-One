"""Hardware tier resolver tests — Master Agent model role assignment."""

from __future__ import annotations

from synapse.domain.hardware import GpuInfo, HardwareProfile, MemoryInfo
from synapse.hardware import TierResolver
from synapse.hardware.tier_resolver import HardwareTier, _DEFAULT_CANDIDATES


def _profile(available_gb: float, vram_gb: float | None = None) -> HardwareProfile:
    gpu = GpuInfo(name="fake", vram_gb=vram_gb, vendor="nvidia") if vram_gb else None
    return HardwareProfile(
        memory=MemoryInfo(total_gb=64.0, available_gb=available_gb),
        gpu=gpu,
    )


def test_default_candidates_match_feature_spec():
    assert _DEFAULT_CANDIDATES[HardwareTier.TIER1] == ["gemma3:1b"]
    assert _DEFAULT_CANDIDATES[HardwareTier.TIER2] == ["gemma3:4b"]
    assert _DEFAULT_CANDIDATES[HardwareTier.TIER3] == ["gemma3:4b"]
    assert _DEFAULT_CANDIDATES[HardwareTier.TIER3_PLUS] == ["gemma3:4b"]
    assert _DEFAULT_CANDIDATES[HardwareTier.CLOUD_FALLBACK] == ["gpt-4o-mini", "gemini-2.5-flash"]


def test_tier1_below_6gb():
    assignment = TierResolver().resolve(_profile(4.0))
    assert assignment.tier is HardwareTier.TIER1
    assert assignment.candidate_models == ["gemma3:1b"]


def test_tier2_between_6_and_16gb():
    assignment = TierResolver().resolve(_profile(8.0))
    assert assignment.tier is HardwareTier.TIER2
    assert assignment.candidate_models == ["gemma3:4b"]


def test_tier2_boundary_inclusive():
    assert TierResolver().resolve(_profile(6.0)).tier is HardwareTier.TIER2


def test_tier3_between_16_and_32gb():
    assignment = TierResolver().resolve(_profile(20.0))
    assert assignment.tier is HardwareTier.TIER3
    assert assignment.candidate_models == ["gemma3:4b"]


def test_tier3_dedicated_vram_low_ram():
    assignment = TierResolver().resolve(_profile(4.0, vram_gb=12.0))
    assert assignment.tier is HardwareTier.TIER3
    assert assignment.candidate_models == ["gemma3:4b"]


def test_tier3_plus_above_32gb():
    assignment = TierResolver().resolve(_profile(36.0))
    assert assignment.tier is HardwareTier.TIER3_PLUS
    assert assignment.candidate_models == ["gemma3:4b"]


def test_tier3_plus_dedicated_vram():
    assignment = TierResolver().resolve(_profile(4.0, vram_gb=24.0))
    assert assignment.tier is HardwareTier.TIER3_PLUS
    assert assignment.candidate_models == ["gemma3:4b"]


def test_cloud_fallback_below_half_gb():
    assignment = TierResolver().resolve(_profile(0.3))
    assert assignment.tier is HardwareTier.CLOUD_FALLBACK
    assert assignment.candidate_models == ["gpt-4o-mini", "gemini-2.5-flash"]
    assert "lockups" in assignment.reason.lower()


def test_can_support_model_safety():
    resolver = TierResolver()
    profile = _profile(10.0, vram_gb=0.0)
    # Model needing 6 GB fits in 10 GB available (buffer 0.5)
    assert resolver.can_support_model("qwen2.5-coder:7b", 6.0, profile) is True
    # Model needing 12 GB exceeds 10 GB available
    assert resolver.can_support_model("gemma3:12b", 12.0, profile) is False
    # Model needing 14 GB fits if GPU has 16 GB VRAM
    gpu_profile = _profile(4.0, vram_gb=16.0)
    assert resolver.can_support_model("qwen2.5-coder:14b", 14.0, gpu_profile) is True


def test_config_override_thresholds_and_candidates():
    class FakeConfig:
        def get(self, key, default=None):
            if key == "master_agent.resolver":
                return {"tier2_min_ram_gb": 20.0, "tier3_min_ram_gb": 40.0, "tier3_plus_min_ram_gb": 60.0, "cloud_max_ram_gb": 1.0}
            if key == "master_agent.tiers":
                return {"tier1": ["tiny-model"], "tier2": ["mid-model"], "cloud": ["sky-model"]}
            return default

    resolver = TierResolver(FakeConfig())
    assert resolver.resolve(_profile(16.0)).tier is HardwareTier.TIER1
    assert resolver.resolve(_profile(16.0)).candidate_models == ["tiny-model"]
    assert resolver.resolve(_profile(35.0)).tier is HardwareTier.TIER2
    assert resolver.resolve(_profile(35.0)).candidate_models == ["mid-model"]
    assert resolver.resolve(_profile(0.8)).tier is HardwareTier.CLOUD_FALLBACK
    assert resolver.resolve(_profile(0.8)).candidate_models == ["sky-model"]
    assert resolver.resolve(_profile(50.0)).tier is HardwareTier.TIER3


def test_explicit_constructor_overrides():
    resolver = TierResolver(thresholds={"tier2_min_ram_gb": 9.0})
    assert resolver.resolve(_profile(10.0)).tier is HardwareTier.TIER2
    resolver = TierResolver(candidates={HardwareTier.TIER1: ["custom:1b"]})
    assert resolver.resolve(_profile(4.0)).candidate_models == ["custom:1b"]