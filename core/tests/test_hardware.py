"""Hardware scanner tests."""

from __future__ import annotations

from synapse.contracts import HardwareProvider
from synapse.hardware import HardwareScanner


def test_scanner_returns_profile(temp_paths, container):
    scanner = container.resolve(HardwareProvider)
    assert isinstance(scanner, HardwareScanner)
    profile = scanner.scan()
    assert profile.os_name
    assert profile.cpu.cores > 0
    assert profile.memory.total_gb > 0
    assert profile.storage.total_gb > 0
    assert profile.recommendations.can_run_local_llm is True
    assert profile.recommendations.max_quantized_params_billions > 0


def test_scan_is_independent_of_router(temp_paths):
    # The scanner must not import/require routing concepts.
    import synapse.hardware  # noqa: F401

    assert "router" not in [f.name for f in (__import__("pathlib").Path("src/synapse/hardware").iterdir())]
