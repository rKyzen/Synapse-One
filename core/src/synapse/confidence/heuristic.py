"""Heuristic Confidence Engine — Phase 4.

A lightweight, dependency-free scorer. It weighs:
- response coverage of the prompt (lexical overlap)
- response length sanity (too short for the question, or padded)
- uncertainty markers (\"I'm not sure\", hedges) that lower confidence
- retrieval grounding (overlap with retrieved chunks raises confidence)
- capability-specific signals (code fences for coding, digit checks for math)

The real confidence ceiling comes from verification and escalation stages;
this engine decides *whether* those stages should run.
"""

from __future__ import annotations

import re

from synapse.contracts.confidence import ConfidenceEngine, ConfidenceScore
from synapse.domain.enums import Capability

_UNCERTAINTY_MARKERS = (
    "i'm not sure",
    "i am not sure",
    "i don't know",
    "i do not know",
    "not certain",
    "probably",
    "maybe",
    "might be",
    "could be",
    "i think",
    "i guess",
    "it seems",
    "not sure",
)

_REFUSAL_MARKERS = (
    "i cannot",
    "i can't",
    "i won't",
    "i will not",
    "unable to",
    "not able to",
    "cannot help",
    "can't help",
)


class HeuristicConfidenceEngine(ConfidenceEngine):
    """Scoring engine based on cheap lexical/structural heuristics."""

    def __init__(
        self,
        base_score: float = 0.55,
        refuse_threshold: float = 0.3,
        verify_threshold: float = 0.7,
        escalate_threshold: float = 0.6,
    ) -> None:
        self._base = base_score
        self._refuse_threshold = refuse_threshold
        self._verify_threshold = verify_threshold
        self._escalate_threshold = escalate_threshold

    # -- ConfidenceEngine ----------------------------------------------------

    def score(
        self,
        response: str,
        prompt: str,
        *,
        context: str | None = None,
        retrieved_chunks: list[dict] | None = None,
        model_id: str | None = None,
        capability_requirements: list[Capability] | None = None,
    ) -> ConfidenceScore:
        del context, model_id  # unused in the heuristic scorer
        if not response or not response.strip():
            return ConfidenceScore(score=0.0, reason="empty response")

        score = self._base
        reasons: list[str] = []

        lowered = response.lower()

        # 1. Prompt coverage — key words from the prompt appear in the answer.
        prompt_words = {w for w in re.findall(r"[a-z]{3,}", prompt.lower())}
        response_words = set(re.findall(r"[a-z]{3,}", lowered))
        if prompt_words:
            covered = len(prompt_words & response_words) / len(prompt_words)
            if covered > 0.5:
                score += 0.15 * min(covered, 1.0)
                reasons.append(f"prompt coverage {covered:.0%}")
            elif covered < 0.15:
                score -= 0.15
                reasons.append(f"poor prompt coverage {covered:.0%}")

        # 2. Length sanity relative to the question.
        words = len(response.split())
        if words < 3:
            score -= 0.25
            reasons.append("response too short")
        elif words < 15 and len(prompt_words) >= 4:
            score -= 0.1
            reasons.append("short response for question length")
        if words > 2000:
            score -= 0.1
            reasons.append("suspiciously long response")

        # 3. Uncertainty markers.
        markers = [m for m in _UNCERTAINTY_MARKERS if m in lowered]
        if markers:
            score -= 0.12 * min(len(markers), 3)
            reasons.append(f"uncertainty markers: {markers[0]}")

        # 4. Refusal markers — a refusal is honest but not a confident answer.
        if any(m in lowered for m in _REFUSAL_MARKERS):
            score -= 0.2
            reasons.append("refusal markers present")

        # 5. Retrieval grounding — overlap with provided chunks.
        if retrieved_chunks:
            chunk_text = " ".join(str(c.get("text", "")) for c in retrieved_chunks).lower()
            chunk_words = set(re.findall(r"[a-z]{3,}", chunk_text))
            if chunk_words:
                grounded = len(response_words & chunk_words) / len(chunk_words)
                score += 0.2 * min(grounded, 1.0)
                if grounded > 0.4:
                    reasons.append(f"grounded in retrieval {grounded:.0%}")

        # 6. Capability-specific checks.
        breakdown: dict[Capability, float] = {}
        for cap in capability_requirements or []:
            cap_score = score
            if cap == Capability.CODING:
                if "```" in response or "def " in response:
                    cap_score += 0.1
                    reasons.append("code present")
                else:
                    cap_score -= 0.1
            elif cap == Capability.MATH:
                if re.search(r"\d", response):
                    cap_score += 0.05
                else:
                    cap_score -= 0.15
                    reasons.append("no numeric output for math task")
            breakdown[cap] = max(0.0, min(1.0, cap_score))

        score = max(0.0, min(1.0, score))
        reason = "; ".join(reasons) if reasons else "heuristic baseline"
        return ConfidenceScore(score=score, reason=reason, breakdown=breakdown or None)

    def should_escalate(self, score: ConfidenceScore, threshold: float | None = None) -> bool:
        return score.score < (threshold if threshold is not None else self._escalate_threshold)

    def should_verify(self, score: ConfidenceScore, threshold: float | None = None) -> bool:
        return score.score < (threshold if threshold is not None else self._verify_threshold)

    def should_refuse(self, score: ConfidenceScore, threshold: float | None = None) -> bool:
        return score.score < (threshold if threshold is not None else self._refuse_threshold)
