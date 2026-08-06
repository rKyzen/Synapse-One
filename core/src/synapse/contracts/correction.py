"""Phase 4: Self-Correction contract — Hallucination Detection.

Detect common hallucinations:
- Invented APIs
- Non-existent files
- Fake package names
- Imaginary project functions
- Incorrect commands

If uncertainty is high, explicitly say so instead of guessing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class HallucinationType(Enum):
    """Types of hallucinations we detect."""

    INVENTED_API = "invented_api"
    NONEXISTENT_FILE = "nonexistent_file"
    FAKE_PACKAGE = "fake_package"
    IMAGINARY_FUNCTION = "imaginary_function"
    INCORRECT_COMMAND = "incorrect_command"
    FAKE_CITATION = "fake_citation"
    CONTRADICTION = "contradiction"
    UNVERIFIABLE_CLAIM = "unverifiable_claim"


@dataclass(frozen=True)
class HallucinationFlag:
    """A detected potential hallucination."""

    type: HallucinationType
    span: str  # The text span that's suspicious
    reason: str
    confidence: float  # How confident we are this is a hallucination
    suggested_correction: str | None = None


class HallucinationDetector(ABC):
    """Detects common hallucination patterns in responses."""

    @abstractmethod
    def scan(
        self,
        response: str,
        *,
        prompt: str,
        context: str | None = None,
        available_apis: list[str] | None = None,
        existing_files: list[str] | None = None,
        installed_packages: list[str] | None = None,
        project_functions: list[str] | None = None,
    ) -> list[HallucinationFlag]:
        """Scan a response for potential hallucinations.

        Args:
            response: The response to scan
            prompt: Original prompt
            context: Context used (RAG, memory, etc.)
            available_apis: Known APIs in the project
            existing_files: Known files in the workspace
            installed_packages: Installed packages
            project_functions: Known functions in the project

        Returns:
            List of HallucinationFlag for each detected issue
        """


class SelfCorrector(ABC):
    """Attempts to auto-correct detected hallucinations."""

    @abstractmethod
    def correct(
        self,
        response: str,
        flags: list[HallucinationFlag],
        *,
        prompt: str,
        context: str | None = None,
    ) -> str:
        """Apply corrections for detected hallucinations.

        Args:
            response: Original response
            flags: Detected hallucination flags
            prompt: Original prompt
            context: Context used

        Returns:
            Corrected response
        """