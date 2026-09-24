"""Intent analysis — heuristic keyword scoring + modality detection.

Rule-based by design (deterministic, offline). A model-backed intent detector
could replace this later behind the same IntentAnalyzer contract.
Phase 2.5 adds prompt-derived modality flags (vision/embeddings) so the
router never picks a vision specialist for a text-only task — and never
ignores one when images are actually involved.
"""

from __future__ import annotations

import re

from synapse.contracts import IntentAnalyzer
from synapse.domain import IntentResult, IntentType

#: (intent, base_weight, keywords) — matched on lowercase prompt text with word boundaries.
_INTENT_TABLE: list[tuple[IntentType, int, tuple[str, ...]]] = [
    (IntentType.PLANNING, 12, ("plan", "planning", "roadmap", "schedule", "timeline", "organize", "steps to", "outline to")),
    (IntentType.CODING, 15, ("code", "coding", "program", "programming", "debug", "debugging", "bug", "bugs", "fix", "fixes", "api", "function", "functions", "script", "scripts", "website", "web app", "flutter", "python", "javascript", "repository", "git", "refactor", "algorithm", "build", "develop", "implement", "platform", "quicksort", "software")),
    (IntentType.RESEARCH, 14, ("research", "study", "analyze", "investigate", "survey", "paper", "citation", "literature", "sources", "evidence", "findings")),
    (IntentType.WRITING, 13, ("write", "writing", "essay", "email", "letter", "story", "poem", "grammar", "rewrite", "draft", "summarize", "translate", "proofread")),
    (IntentType.BUSINESS, 12, ("business", "revenue", "profit", "client", "contract", "pitch", "strategy", "sales", "startup", "invoice", "market")),
    (IntentType.CREATIVE, 11, ("creative", "brainstorm", "design", "imagine", "moodboard", "illustration", "artwork", "logo")),
    (IntentType.EDUCATION, 10, ("teach", "learn", "student", "homework", "explain", "explaining", "explanation", "lesson", "tutorial", "quiz")),
    (IntentType.CONVERSATION, 8, ("hi", "hello", "hey", "how are you", "thanks", "thank you", "who are you", "bye", "good morning")),
]

_COMPILED_INTENT_TABLE: list[tuple[IntentType, int, list[re.Pattern]]] = [
    (
        intent,
        weight,
        [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in keywords],
    )
    for intent, weight, keywords in _INTENT_TABLE
]

#: Unambiguous image-understanding signals. Deliberately excludes ambiguous
#: words (graph, scan, pdf alone) that appear in coding/security contexts.
_VISION_HINTS: tuple[str, ...] = (
    "screenshot", "screen shot", "image", "images", "photo", "photos", "picture",
    "pictures", "ocr", "optical character", "thumbnail", "meme", "infographic",
    "diagram", "chart", "look at this", "describe this image", "analyze this image",
    "visual inspection", "image analysis", "caption this",
)

_VISION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b" + re.escape(hint) + r"\b", re.IGNORECASE)
    for hint in _VISION_HINTS
]

#: Semantic-retrieval signals (memory engine is a later phase; the router
#: boosts embeddings-capable models when these appear).
_EMBEDDING_HINTS: tuple[str, ...] = (
    "semantic search", "vector search", "find similar", "search my notes",
    "retrieve", "embedding", "similar documents", "remember that", "recall",
    "search my memory", "what did i ask", "my previous work",
)

_EMBEDDING_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b" + re.escape(hint) + r"\b", re.IGNORECASE)
    for hint in _EMBEDDING_HINTS
]


def _score_prompt(text: str) -> dict[IntentType, int]:
    """Map each intent to a weighted score for the prompt text."""
    scores: dict[IntentType, int] = {}
    for intent, weight, patterns in _COMPILED_INTENT_TABLE:
        hits = sum(1 for p in patterns if p.search(text))
        if hits:
            scores[intent] = weight * hits
    return scores


def _modality(text: str) -> tuple[bool, bool, list[str]]:
    vision = any(p.search(text) for p in _VISION_PATTERNS)
    embeddings = any(p.search(text) for p in _EMBEDDING_PATTERNS)
    reasons = []
    if vision:
        reasons.append("prompt references images/screenshots/documents → vision required")
    if embeddings:
        reasons.append("prompt asks for semantic retrieval → embeddings preferred")
    return vision, embeddings, reasons


class IntentAnalyzer(IntentAnalyzer):
    """Deterministic keyword-weight intent detector + modality analysis."""

    def analyze(self, prompt: str) -> IntentResult:
        text = prompt.lower()
        scores = _score_prompt(text)
        reasoning: list[str] = []

        vision, embeddings, modality_reasons = _modality(text)
        reasoning.extend(modality_reasons)

        if not scores:
            reasoning.append("no strong intent keyword; treated as general")
            return IntentResult(
                primary=IntentType.GENERAL,
                confidence=0.4,
                reasoning=reasoning,
                vision_required=vision,
                embeddings_required=embeddings,
            )

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        primary, primary_score = ranked[0]
        secondary = ranked[1][0] if len(ranked) > 1 else None
        total = sum(scores.values())
        confidence = round(min(0.99, primary_score / total), 2)

        reasoning.append(f"detected intent {primary.value} with score {primary_score}")
        if secondary:
            reasoning.append(f"secondary intent {secondary.value}")
        return IntentResult(
            primary=primary,
            confidence=confidence,
            secondary=secondary,
            reasoning=reasoning,
            vision_required=vision,
            embeddings_required=embeddings,
        )