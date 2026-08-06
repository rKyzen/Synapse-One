"""Phase 4: Multi-Step Verification contract.

For factual, technical, mathematical, or coding questions:
- Generate an initial answer
- Verify the answer before returning it
- Detect contradictions
- Correct obvious mistakes automatically
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class VerificationStatus(Enum):
    """Result of verification."""

    VERIFIED = "verified"
    CORRECTED = "corrected"
    CONTRADICTION = "contradiction"
    UNCERTAIN = "uncertain"
    FAILED = "failed"


@dataclass(frozen=True)
class VerificationResult:
    """Result of verifying a response."""

    status: VerificationStatus
    original_response: str
    verified_response: str
    corrections: list[str]
    confidence: float
    reason: str
    sources_used: list[str]


class Verifier(ABC):
    """Verifies responses for correctness and consistency."""

    @abstractmethod
    def verify(
        self,
        prompt: str,
        response: str,
        *,
        context: str | None = None,
        capability: str | None = None,
        expected_format: str | None = None,
    ) -> VerificationResult:
        """Verify a response against the prompt and context.

        Args:
            prompt: Original user prompt
            response: Response to verify
            context: Context that was used (RAG, memory, etc.)
            capability: Task capability (coding, math, factual, etc.)
            expected_format: Expected output format (json, code, etc.)

        Returns:
            VerificationResult with status and corrected response
        """


class FactChecker(ABC):
    """Checks factual claims against known sources."""

    @abstractmethod
    def check_claim(self, claim: str, sources: list[str]) -> dict:
        """Check a single factual claim against sources."""


class CodeVerifier(ABC):
    """Verifies code correctness (syntax, imports, logic)."""

    @abstractmethod
    def verify_code(self, code: str, language: str, context: str | None = None) -> dict:
        """Verify code for common issues."""


class MathVerifier(ABC):
    """Verifies mathematical computations."""

    @abstractmethod
    def verify_math(self, expression: str, expected: str | None = None) -> dict:
        """Verify a mathematical expression or result."""