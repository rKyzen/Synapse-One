"""Analyzer contracts — replaceable analysis engines.

Analyzers are pure: prompt in, structured result out. The Master calls them
directly. A rule-based analyzer can later be swapped for a model-backed one
without touching the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from synapse.domain.diagnosis import ComplexityResult, IntentResult, PrivacyResult


class IntentAnalyzer(ABC):
    @abstractmethod
    def analyze(self, prompt: str) -> IntentResult:
        """Detect the primary/secondary intent of a user prompt."""


class ComplexityAnalyzer(ABC):
    @abstractmethod
    def analyze(self, prompt: str) -> ComplexityResult:
        """Score prompt difficulty on a 0-100 scale."""


class PrivacyAnalyzer(ABC):
    @abstractmethod
    def analyze(self, prompt: str, complexity: ComplexityResult, context: dict | None = None) -> PrivacyResult:
        """Return a privacy decision for the prompt.

        ``context`` may carry provider availability and user preference so the
        analyzer can reason about local/cloud options.
        """