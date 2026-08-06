"""Hardware provider contract.

Implemented by the hardware scanner; consumed later by the router and model
registry recommendations. Hardware logic must never leak into routing — the
router sees the resulting profile only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from synapse.domain.hardware import HardwareProfile


class HardwareProvider(ABC):
    @abstractmethod
    def scan(self) -> HardwareProfile:
        """Produce a fresh snapshot of machine capabilities."""
