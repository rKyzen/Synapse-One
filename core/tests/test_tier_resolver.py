"""Hardware tier resolver tests — Master Agent model role assignment."""

from __future__ import annotations

from synapse.domain.hardware import GpuInfo, HardwareProfile, MemoryInfo
from synapse.hardware import TierResolver
from synapse.hardware.tier_resolver import HardwareTier, _DEFAULT_CANDIDATES


def _profile(available_gb: float, vram_gb: float | None = None) -> HardwareProfile:
    gpu = GpuInfo(name="fake", vram_gb=vram_gb, vendor="nvidia") if vram_gb else None
    return HardwareProfile(
        memory=MemoryInfo(total_gb=32.0, available_gb=available_gb),
        gpu=gpu,
    )


def test_default_candidates_match_feature_spec():
    assert _DEFAULT_CANDIDATES[HardwareTier.TIER1] == ["qwen2.5:1.5b", "llama3.2:1b"]
    assert _DEFAULT_CANDIDATES[HardwareTier.TIER2] == ["qwen2.5:3b", "llama3.2:3b"]
    assert _DEFAULT_CANDIDATES[HardwareTier.TIER3] == ["qwen2.5:7b", "llama3.1:8b"]
    assert _DEFAULT_CANDIDATES[HardwareTier.CLOUD_FALLBACK] == ["gpt-4o-mini", "gemini-2.5-flash"]


def test_tier1_below_12gb_no_vram():
    assignment = TierResolver().resolve(_profile(8.0))
    assert assignment.tier is HardwareTier.TIER1
    assert assignment.candidate_models == ["qwen2.5:1.5b", "llama3.2:1b"]


def test_tier2_between_12_and_24gb():
    assignment = TierResolver().resolve(_profile(16.0))
    assert assignment.tier is HardwareTier.TIER2
    assert assignment.candidate_models == ["qwen2.5:3b", "llama3.2:3b"]


def test_tier2_boundary_inclusive():
    assert TierResolver().resolve(_profile(12.0)).tier is HardwareTier.TIER2


def test_tier3_above_24gb():
    assignment = TierResolver().resolve(_profile(32.0))
    assert assignment.tier is HardwareTier.TIER3
    assert assignment.candidate_models == ["qwen2.5:7b", "llama3.1:8b"]


def test_tier3_dedicated_vram_low_ram():
    assignment = TierResolver().resolve(_profile(8.0, vram_gb=12.0))
    assert assignment.tier is HardwareTier.TIER3


def test_tier3_boundary_exclusive_for_ram():
    assert TierResolver().resolve(_profile(24.0)).tier is HardwareTier.TIER2
    assert TierResolver().resolve(_profile(24.01)).tier is HardwareTier.TIER3


def test_cloud_fallback_below_6gb():
    assignment = TierResolver().resolve(_profile(4.0))
    assert assignment.tier is HardwareTier.CLOUD_FALLBACK
    assert assignment.candidate_models == ["gpt-4o-mini", "gemini-2.5-flash"]
    assert "lockups" in assignment.reason.lower()


def test_config_override_thresholds_and_candidates():
    class FakeConfig:
        def get(self, key, default=None):
            if key == "master_agent.resolver":
                return {"tier2_min_ram_gb": 20.0, "tier3_min_ram_gb": 40.0, "cloud_max_ram_gb": 8.0}
            if key == "master_agent.tiers":
                return {"tier1": ["tiny-model"], "tier2": ["mid-model"], "cloud": ["sky-model"]}
            return default

    resolver = TierResolver(FakeConfig())
    assert resolver.resolve(_profile(16.0)).tier is HardwareTier.TIER1
    assert resolver.resolve(_profile(16.0)).candidate_models == ["tiny-model"]
    assert resolver.resolve(_profile(35.0)).tier is HardwareTier.TIER2
    assert resolver.resolve(_profile(35.0)).candidate_models == ["mid-model"]
    assert resolver.resolve(_profile(6.0)).tier is HardwareTier.CLOUD_FALLBACK
    assert resolver.resolve(_profile(6.0)).candidate_models == ["sky-model"]
    assert resolver.resolve(_profile(50.0)).tier is HardwareTier.TIER3


def test_explicit_constructor_overrides():
    resolver = TierResolver(thresholds={"tier2_min_ram_gb": 9.0})
    assert resolver.resolve(_profile(10.0)).tier is HardwareTier.TIER2
    resolver = TierResolver(candidates={HardwareTier.TIER1: ["custom:1b"]})
    assert resolver.resolve(_profile(8.0)).candidate_models == ["custom:1b"]