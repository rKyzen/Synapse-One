"""Hardware domain entities."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CpuInfo(BaseModel):
    model: str = "unknown"
    cores: int = 0
    threads: int = 0
    frequency_mhz: float | None = None
    architecture: str = "unknown"


class GpuInfo(BaseModel):
    name: str | None = None
    vram_gb: float | None = None
    vendor: str | None = None


class MemoryInfo(BaseModel):
    total_gb: float = 0.0
    available_gb: float = 0.0
    used_percent: float = 0.0


class StorageInfo(BaseModel):
    total_gb: float = 0.0
    free_gb: float = 0.0
    filesystem: str = "unknown"


class RecommendedModelLimits(BaseModel):
    """Hardware-derived recommendation of what can run locally.

    Pure data — the router decides how to use it, never the scanner.
    """

    max_ram_gb: float = 0.0
    max_quantized_params_billions: float = 0.0
    vram_available_gb: float = 0.0
    can_run_local_llm: bool = False
    notes: list[str] = Field(default_factory=list)


class HardwareProfile(BaseModel):
    """Immutable snapshot of machine capabilities at scan time."""

    os_name: str = "unknown"
    os_version: str = "unknown"
    arch: str = "unknown"
    cpu: CpuInfo = Field(default_factory=CpuInfo)
    gpu: GpuInfo | None = None
    memory: MemoryInfo = Field(default_factory=MemoryInfo)
    storage: StorageInfo = Field(default_factory=StorageInfo)
    recommendations: RecommendedModelLimits = Field(default_factory=RecommendedModelLimits)
