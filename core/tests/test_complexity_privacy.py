"""Complexity + privacy analyzer tests."""

from __future__ import annotations

import pytest

from synapse.analyzers import ComplexityAnalyzer, PrivacyAnalyzer
from synapse.domain import PrivacyMode


def test_complexity_simple_math():
    result = ComplexityAnalyzer().analyze("what is 5 x 5")
    assert result.score <= 10


def test_complexity_summarization_mid():
    result = ComplexityAnalyzer().analyze("summarize this article about the economy")
    assert 30 <= result.score <= 50


def test_complexity_research_high():
    result = ComplexityAnalyzer().analyze("Write a full research paper with literature review")
    assert result.score >= 85


def test_complexity_website():
    result = ComplexityAnalyzer().analyze("Build a complete portfolio website with a web app backend")
    assert result.score >= 60


def test_complexity_bounds():
    for prompt in ("hi", "explain photosynthesis", "build a flutter app"):
        r = ComplexityAnalyzer().analyze(prompt)
        assert 0 <= r.score <= 100
        assert r.reasoning


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("Hello", 1),
        ("2+2", 2),
        ("Write a resignation letter", 20),
        ("QuickSort", 35),
        ("Explain Quantum Computing", 55),
        ("Design a distributed chat architecture", 80),
        ("Build a complete ERP platform", 95),
    ],
)
def test_complexity_calibration(prompt, expected):
    assert ComplexityAnalyzer().analyze(prompt).score == expected


def test_complexity_is_dynamic_not_constant():
    scores = [ComplexityAnalyzer().analyze(p).score for p in ("hi", "2+2", "summarize a report", "build an erp")]
    assert len(set(scores)) > 2


def test_privacy_sensitive_local_only():
    result = PrivacyAnalyzer().analyze(
        "my credit card is 1234, help me with a password",
        ComplexityAnalyzer().analyze("my credit card password"),
        context={"user_preference": "balanced", "provider_availability": {"local": True, "cloud": True}},
    )
    assert result.mode == PrivacyMode.LOCAL_ONLY


def test_privacy_internet_cloud_required():
    result = PrivacyAnalyzer().analyze(
        "what is the latest news today on the web?",
        ComplexityAnalyzer().analyze("latest news"),
        context={"user_preference": "balanced", "provider_availability": {"local": True, "cloud": True}},
    )
    assert result.internet_required is True
    assert result.mode == PrivacyMode.CLOUD_REQUIRED


def test_privacy_prefer_local_default():
    result = PrivacyAnalyzer().analyze(
        "explain photosynthesis",
        ComplexityAnalyzer().analyze("explain photosynthesis"),
        context={"user_preference": "balanced", "provider_availability": {"local": True, "cloud": True}},
    )
    assert result.mode == PrivacyMode.BALANCED


def test_privacy_user_preference():
    result = PrivacyAnalyzer().analyze(
        "explain photosynthesis",
        ComplexityAnalyzer().analyze("explain photosynthesis"),
        context={"user_preference": "local", "provider_availability": {"local": True, "cloud": True}},
    )
    assert result.mode == PrivacyMode.PREFER_LOCAL
