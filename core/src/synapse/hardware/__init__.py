"""Hardware scanning as an independent service.

Produces an immutable HardwareProfile. Contains NO routing/heuristic logic —
the router consumes this profile later. Uses psutil with graceful fallbacks so
a scanner never crashes on an unusual machine.

The tier resolver (Master Agent model role) is also exported here: it is pure
hardware knowledge — no providers, no models beyond configured candidates.
"""

from __future__ import annotations

import platform
import shutil
import sys

import psutil

from synapse.config.paths import SynapsePaths
from synapse.contracts import HardwareProvider, ConfigProvider
from synapse.domain.hardware import (
    CpuInfo,
    GpuInfo,
    HardwareProfile,
    MemoryInfo,
    RecommendedModelLimits,
    StorageInfo,
)
from synapse.hardware.tier_resolver import HardwareTier, TierAssignment, TierResolver

__all__ = [
    "HardwareScanner",
    "HardwareTier",
    "TierAssignment",
    "TierResolver",
]


def _recommend(memory: MemoryInfo) -> RecommendedModelLimits:
    """Pure mapping: hardware -> what can likely run locally.

    Notes are advisory only; the router (later) makes the final call.
    Simple heuristic by RAM/VRAM, kept isolated from routing logic.
    """
    notes: list[str] = []
    max_ram_gb = memory.total_gb
    # Rough rule of thumb: a ~2-4B quantized model needs ~8GB RAM; scale linearly.
    usable = max(memory.total_gb * 0.6, 0.0)
    max_params = usable / 2.0
    if memory.total_gb < 8:
        notes.append("low RAM: only small quantized models practical")
        can_local = True
    else:
        notes.append("sufficient RAM for local small-to-medium models")
        can_local = True
    return RecommendedModelLimits(
        max_ram_gb=max_ram_gb,
        max_quantized_params_billions=round(max_params, 1),
        vram_available_gb=0.0,  # filled if a discrete GPU is found
        can_run_local_llm=can_local,
        notes=notes,
    )


def _gpu() -> GpuInfo | None:
    """Best-effort GPU detection. Returns None when unknown."""
    try:
        # NVIDIA via nvidia-smi if available
        import subprocess  # noqa: PLC0415

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip():
            name, vram = out.stdout.strip().splitlines()[0].split(",")
            return GpuInfo(name=name.strip(), vram_gb=float(vram.strip()), vendor="nvidia")
    except Exception:  # noqa: BLE001 - GPU detection is best-effort
        pass
    # WMI on Windows for a generic name
    if sys.platform == "win32":
        try:
            from winreg import (  # noqa: PLC0415
                HKEY_LOCAL_MACHINE,
                KEY_READ,
                OpenKey,
                QueryValueEx,
            )

            key = OpenKey(HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000")
            vendor, _ = QueryValueEx(key, "ProviderName")
            name, _ = QueryValueEx(key, "DeviceDescription")
            return GpuInfo(name=name, vendor=vendor)
        except OSError:
            pass
    return None


class HardwareScanner(HardwareProvider):
    """Default HardwareProvider backed by psutil + stdlib.

    Injected dependencies: config/paths are available for future per-OS tuning.
    """

    def __init__(self, config: ConfigProvider, paths: SynapsePaths) -> None:
        self._config = config
        self._paths = paths

    def scan(self) -> HardwareProfile:
        cpu_freq = None
        try:
            cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else None
        except Exception:  # noqa: BLE001
            pass

        cpu = CpuInfo(
            model=platform.processor() or platform.machine() or "unknown",
            cores=psutil.cpu_count(logical=False) or 0,
            threads=psutil.cpu_count(logical=True) or 0,
            frequency_mhz=cpu_freq,
            architecture=platform.machine() or "unknown",
        )

        vm = psutil.virtual_memory()
        memory = MemoryInfo(
            total_gb=round(vm.total / (1024**3), 2),
            available_gb=round(vm.available / (1024**3), 2),
            used_percent=vm.percent,
        )

        du = shutil.disk_usage(self._paths.home)
        storage = StorageInfo(
            total_gb=round(du.total / (1024**3), 2),
            free_gb=round(du.free / (1024**3), 2),
            filesystem=platform.uname().system.lower(),
        )

        gpu = _gpu()
        rec = _recommend(memory)
        if gpu and gpu.vram_gb:
            rec = rec.model_copy(
                update={
                    "vram_available_gb": gpu.vram_gb,
                    "max_quantized_params_billions": round(gpu.vram_gb / 4.0, 1),
                }
            )

        return HardwareProfile(
            os_name=platform.system(),
            os_version=platform.release(),
            arch=cpu.architecture,
            cpu=cpu,
            gpu=gpu,
            memory=memory,
            storage=storage,
            recommendations=rec,
        )