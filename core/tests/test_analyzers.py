"""Intent analyzer tests."""

from __future__ import annotations

from synapse.analyzers import IntentAnalyzer
from synapse.domain import IntentType


def test_coding_intent():
    result = IntentAnalyzer().analyze("Write a python script to parse JSON and fix the bug")
    assert result.primary == IntentType.CODING
    assert result.confidence > 0.5


def test_writing_intent():
    result = IntentAnalyzer().analyze("Write an essay about climate change")
    assert result.primary == IntentType.WRITING


def test_conversation_intent():
    result = IntentAnalyzer().analyze("hi, how are you?")
    assert result.primary == IntentType.CONVERSATION


def test_research_intent():
    result = IntentAnalyzer().analyze("I need to research and analyze the literature for my paper")
    assert result.primary == IntentType.RESEARCH


def test_general_fallback():
    result = IntentAnalyzer().analyze("what time is it")
    assert result.primary == IntentType.GENERAL
    assert result.reasoning


def test_secondary_intent():
    result = IntentAnalyzer().analyze("Write a business email about revenue strategy")
    # writing and business both fire; one of them is secondary
    assert result.secondary in (IntentType.BUSINESS, IntentType.WRITING)


def test_confidence_in_range():
    result = IntentAnalyzer().analyze("Plan a schedule for the sprint")
    assert 0.0 <= result.confidence <= 1.0


def test_vision_modality_detected():
    result = IntentAnalyzer().analyze("Analyze this screenshot and describe what you see")
    assert result.vision_required is True
    assert any("vision" in r for r in result.reasoning)


def test_embeddings_modality_detected():
    result = IntentAnalyzer().analyze("semantic search my notes for similar documents")
    assert result.embeddings_required is True


def test_plain_text_has_no_vision_modality():
    result = IntentAnalyzer().analyze("Hello, how are you?")
    assert result.vision_required is False
    assert result.embeddings_required is False


def test_coding_prompt_not_marked_vision():
    result = IntentAnalyzer().analyze("write a graph traversal algorithm in python")
    assert result.vision_required is False
