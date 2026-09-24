"""Router V2 (Phase 2.5) — the intelligent supervisor's model selection.

The router behaves like a supervisor that understands WHY each model exists
instead of treating models as equals. Pipeline per request:

    capability matching  -> which capabilities does this task need?
    capability gating    -> hard requirements (privacy, hardware, health,
                            availability, required capabilities) — every
                            failure is recorded with a reason.
    weighted scoring     -> required capabilities weight strongly, preferred
                            capabilities boost, unused specializations are
                            PENALIZED (vision models score zero on text-only
                            tasks, coding specialists lose priority on
                            non-coding tasks).
    latency prediction   -> model size + prompt length + expected output +
                            hardware + historical performance.
    selection            -> best score wins; zero-scoring models never route.

Works purely on ModelMetadata + hardware + health + the Decision. NO vendor
names or hardcoded model ids anywhere.
"""

from __future__ import annotations

from synapse.contracts import Router
from synapse.domain import (
    Capability,
    Decision,
    HardwareProfile,
    LatencyTier,
    PrivacyMode,
    ProviderKind,
    RoutingDecision,
)
from synapse.domain.models import ModelMetadata
from synapse.hardware.model_matrix import get_exact_model, matches_model_id
from synapse.hardware.tier_resolver import TierResolver

#: Weight of each REQUIRED capability in the score.
_REQUIRED_WEIGHTS: dict[Capability, float] = {
    Capability.REASONING: 0.45,
    Capability.CODING: 0.50,
    Capability.WRITING: 0.50,
    Capability.MATH: 0.40,
    Capability.VISION: 0.55,
    Capability.OCR: 0.40,
    Capability.PDF: 0.35,
    Capability.TOOLS: 0.25,
    Capability.LONG_CONTEXT: 0.25,
    Capability.DEBUGGING: 0.30,
    Capability.ARCHITECTURE: 0.30,
    Capability.PLANNING: 0.30,
    Capability.TRANSLATION: 0.30,
    Capability.EMBEDDINGS: 0.25,
    Capability.JSON: 0.20,
    Capability.TERMINAL: 0.15,
    Capability.CHAT: 0.15,
    Capability.CONVERSATION: 0.15,
    Capability.LANGUAGES: 0.15,
    Capability.RESEARCH: 0.40,
    Capability.SUMMARIZATION: 0.35,
    Capability.CODE_ANALYSIS: 0.45,
    Capability.FILE_READING: 0.25,
    Capability.FILE_CREATION: 0.40,
    Capability.FILE_EDITING: 0.45,
    Capability.TESTING: 0.40,
    Capability.IMAGE_UNDERSTANDING: 0.55,
    Capability.PDF_READING: 0.40,
    Capability.PDF_CREATION: 0.35,
    Capability.DOCX_CREATION: 0.35,
    Capability.DOCUMENT_ANALYSIS: 0.40,
    Capability.PPT_CREATION: 0.35,
    Capability.SPREADSHEET_CREATION: 0.35,
    Capability.DATA_ANALYSIS: 0.45,
    Capability.PROJECT_CREATION: 0.45,
    Capability.PROJECT_ANALYSIS: 0.45,
    Capability.TASK_MANAGEMENT: 0.30,
    Capability.MEMORY: 0.25,
    Capability.KNOWLEDGE_RETRIEVAL: 0.30,
    Capability.CITATIONS: 0.25,
    Capability.AUTOMATION: 0.35,
}

#: PREFERRED capabilities contribute at this fraction of the required weight.
_PREFERRED_MULTIPLIER = 0.6

#: Penalties for unused specializations (task-independent).
_VISION_UNUSED_PENALTY = 0.35
_OCR_UNUSED_PENALTY = 0.25
_PDF_UNUSED_PENALTY = 0.20
_CODING_UNUSED_THRESHOLD = 0.95
_CODING_UNUSED_PENALTY = 0.35

#: Deep-reasoning bonus: high-complexity tasks favor reasoning specialists.
_DEEP_REASONING_THRESHOLD = 75
_DEEP_REASONING_BONUS = 0.4

#: Small structural biases (kept deliberately small — capability fit dominates).
_LOCAL_BIAS = 0.05
_PRIVACY_BIAS = 0.05
_LATENCY_WEIGHT = {LatencyTier.FAST: 1.0, LatencyTier.MEDIUM: 0.6, LatencyTier.SLOW: 0.2}
_LATENCY_BONUS = 0.1

#: Cost penalty (dollars per 1k tokens -> score), capped so it cannot
#: dominate the capability match.
_COST_SCALE = 1000.0
_COST_PENALTY_CAP = 0.08

#: Embedding-only models can never serve text generation.
_EMBEDDINGS_ONLY_THRESHOLD = 0.9

_LONG_CONTEXT_WINDOW = 128_000

#: Historical-performance reliability (Phase 3): once a model has at least
#: this many recorded executions, its score is scaled by reliability.
_RELIABILITY_MIN_SAMPLES = 3
#: (score) * (1 - _RELIABILITY_MAX_PENALTY * (1 - success_rate)) — a model
#: with 50% success is penalized by up to this fraction.
_RELIABILITY_MAX_PENALTY = 0.15
#: timeouts/failures additionally shave this fraction off the score.
_RELIABILITY_FAILURE_PENALTY = 0.05


def pick_primary_capability(caps: list[Capability] | None) -> Capability:
    """Select the most specific capability from a task's requirements."""
    if not caps:
        return Capability.CHAT
    for specific_cap in (
        Capability.VISION,
        Capability.IMAGE_UNDERSTANDING,
        Capability.OCR,
        Capability.CODING,
        Capability.FILE_CREATION,
        Capability.FILE_EDITING,
        Capability.TESTING,
        Capability.DEBUGGING,
        Capability.PLANNING,
        Capability.ARCHITECTURE,
        Capability.MATH,
        Capability.REASONING,
        Capability.TRANSLATION,
        Capability.LANGUAGES,
        Capability.RESEARCH,
        Capability.WRITING,
        Capability.SUMMARIZATION,
        Capability.EMBEDDINGS,
    ):
        if specific_cap in caps:
            return specific_cap
    return caps[0]


class Router(Router):
    """Deterministic rule-based intelligent router. Replaceable by a learned
    router later — the interface stays the same."""

    def route(
        self,
        decision: Decision,
        hardware: HardwareProfile,
        registry: list[ModelMetadata],
        provider_health: dict[str, bool],
        available_models: dict[str, set[str]] | None = None,
        *,
        complexity: int | None = None,
        prompt: str | None = None,
        performance: dict[str, dict] | None = None,
    ) -> RoutingDecision:
        complexity = complexity or 0
        candidates, excluded = self._filter(registry, hardware, decision, provider_health, available_models)
        if not candidates:
            return RoutingDecision(
                provider_id="",
                model_id="",
                reason="no model satisfies all constraints",
                excluded_models=excluded,
            )

        required = set(decision.required_capabilities)
        preferred = [c for c in decision.preferred_capabilities if c not in required]

        tier = TierResolver().resolve(hardware).tier
        primary_cap = pick_primary_capability(decision.required_capabilities)
        is_deep = (complexity >= _DEEP_REASONING_THRESHOLD and (Capability.REASONING in required or Capability.MATH in required))
        locked_model = get_exact_model(tier, primary_cap, is_deep_reasoning=is_deep)

        scored = sorted(
            (
                (self._score(m, decision, required, preferred, complexity, locked_model_id=locked_model), m)
                for m in candidates
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        scored = [
            (self._reliability_adjust(score, m, performance), m)
            for score, m in scored
        ]
        best_score, best = scored[0]

        if best_score <= 0:
            return RoutingDecision(
                provider_id="",
                model_id="",
                reason="no model positively matches the request capabilities",
                excluded_models=excluded,
            )

        second_score = scored[1][0] if len(scored) > 1 else 0.0
        candidates_view = [
            {
                "provider": m.provider_id,
                "model": m.id,
                "kind": m.kind.value,
                "score": round(s, 3),
                "estimated_latency_s": self._estimate_latency_s(m, prompt or "", complexity, hardware, performance),
                "reliability": self._reliability_summary(m, performance),
            }
            for s, m in scored
        ]

        expected_tokens = self._expected_output_tokens(complexity)
        history = (performance or {}).get(f"{best.provider_id}/{best.id}")
        reliability_note = self._reliability_note(best, performance)

        return RoutingDecision(
            provider_id=best.provider_id,
            model_id=best.id,
            kind=best.kind,
            confidence=round(min(0.99, best_score / (best_score + second_score + 1e-9)), 2),
            estimated_cost_per_1k=best.estimated_cost_per_1k,
            reason=(
                f"best fit: {best.id} via {best.provider_id} "
                f"(capability score {best_score:.3f}, est {self._estimate_latency_s(best, prompt or '', complexity, hardware, performance)}s, "
                f"~{expected_tokens} output tokens{reliability_note})"
            ),
            candidates=candidates_view,
            capability_score=round(best_score, 3),
            excluded_models=excluded,
            estimated_latency_s=self._estimate_latency_s(best, prompt or "", complexity, hardware, performance),
            expected_output_tokens=expected_tokens,
            historical_performance=history,
        )

    # -- gating -------------------------------------------------------------

    def is_suitable(
        self,
        m: ModelMetadata,
        hardware: HardwareProfile,
        decision: Decision,
        provider_health: dict[str, bool],
        available_models: dict[str, set[str]] | None,
    ) -> bool:
        """True when ``m`` passes every hard constraint for ``decision``.

        Public so the lifecycle manager can test already-loaded models for
        reuse without duplicating the gating rules.
        """
        return Router._exclusion_reason(m, hardware, decision, provider_health, available_models, decision.required_capabilities) is None

    def capability_score(
        self,
        m: ModelMetadata,
        decision: Decision,
        *,
        complexity: int = 0,
        locked_model_id: str | None = None,
    ) -> float:
        """Requirement/preference-weighted capability score for ``m``.

        Public so the lifecycle manager can compare an already-loaded model
        against the ideal candidate and decide whether reuse is acceptable.
        """
        required = set(decision.required_capabilities)
        preferred = [c for c in decision.preferred_capabilities if c not in required]
        return self._score(m, decision, required, preferred, complexity, locked_model_id=locked_model_id)

    @staticmethod
    def _filter(
        registry: list[ModelMetadata],
        hardware: HardwareProfile,
        decision: Decision,
        provider_health: dict[str, bool],
        available_models: dict[str, set[str]] | None,
    ) -> tuple[list[ModelMetadata], list[dict[str, str]]]:
        """Apply hard constraints. Returns (admissible, excluded with reasons)."""
        out: list[ModelMetadata] = []
        excluded: list[dict[str, str]] = []
        required = decision.required_capabilities
        for m in registry:
            reason = Router._exclusion_reason(m, hardware, decision, provider_health, available_models, required)
            if reason:
                excluded.append({"model": m.id, "reason": reason})
                continue
            out.append(m)
        return out, excluded

    @staticmethod
    def _exclusion_reason(
        m: ModelMetadata,
        hardware: HardwareProfile,
        decision: Decision,
        provider_health: dict[str, bool],
        available_models: dict[str, set[str]] | None,
        required: list[Capability],
    ) -> str | None:
        health = provider_health.get(m.provider_id)
        if health is False:
            return f"provider {m.provider_id} unhealthy"

        if available_models is not None:
            installed = available_models.get(m.provider_id)
            # Ollama reports tags ("llama3.2:latest"); config keys are untagged.
            # None = provider couldn't report (transient failure) — don't
            # exclude; empty set = genuinely no models — exclude.
            if installed is not None:
                if len(installed) == 0:
                    return f"no models installed on {m.provider_id}"
                matched = m.id in installed or f"{m.id}:latest" in installed
                if not matched:
                    return f"not reported installed by {m.provider_id}"

        if decision.preferred_kind == ProviderKind.LOCAL and m.kind != ProviderKind.LOCAL:
            return f"privacy requires local execution, model is {m.kind.value}"
        if decision.preferred_kind == ProviderKind.CLOUD and m.kind != ProviderKind.CLOUD:
            return f"decision requires cloud execution, model is {m.kind.value}"
        if decision.privacy == PrivacyMode.LOCAL_ONLY and m.kind != ProviderKind.LOCAL:
            return "privacy LOCAL_ONLY forbids cloud execution"

        if m.required_ram_gb > hardware.memory.available_gb:
            return f"needs {m.required_ram_gb}GB RAM, {hardware.memory.available_gb}GB available"
        vram = hardware.gpu.vram_gb if hardware.gpu else 0.0
        if m.required_vram_gb > vram:
            return f"needs {m.required_vram_gb}GB VRAM, {vram}GB available"

        for cap in required:
            if cap == Capability.LONG_CONTEXT:
                if not (m.context_window or 0) >= _LONG_CONTEXT_WINDOW:
                    return f"lacks required capability {cap.value} (<{_LONG_CONTEXT_WINDOW} context)"
            elif m.capabilities.score_for(cap) <= 0:
                return f"lacks required capability {cap.value}"

        if (
            m.capabilities.embeddings > _EMBEDDINGS_ONLY_THRESHOLD
            and m.capabilities.chat <= 0
        ):
            return "embeddings-only model cannot serve text generation"
        return None

    # -- scoring ------------------------------------------------------------

    def _reliability_adjust(
        self,
        score: float,
        m: ModelMetadata,
        performance: dict[str, dict] | None,
    ) -> float:
        """Scale a candidate score by its historical reliability (Phase 3).

        Uses success rate and failure/timeout rate recorded in the
        PerformanceStore: unreliable models are demoted so a stable specialist
        can outrank a flaky one with identical capabilities. Models with fewer
        than ``_RELIABILITY_MIN_SAMPLES`` executions are untouched.
        """
        if not performance:
            return score
        history = performance.get(f"{m.provider_id}/{m.id}")
        if not history:
            return score
        samples = int(history.get("samples", 0) or 0)
        if samples < _RELIABILITY_MIN_SAMPLES:
            return score

        success_rate = float(history.get("success_rate", 1.0) or 0.0)
        timeouts = float(history.get("timeout_rate", 0.0) or 0.0)
        failures = float(history.get("failure_rate", 0.0) or 0.0)
        reliability = 1.0 - _RELIABILITY_MAX_PENALTY * (1.0 - success_rate)
        penalty = _RELIABILITY_FAILURE_PENALTY * (timeouts + failures)
        adjusted = max(0.0, score * reliability - penalty)
        return round(adjusted, 3)

    @staticmethod
    def _reliability_summary(m: ModelMetadata, performance: dict[str, dict] | None) -> dict | None:
        if not performance:
            return None
        history = performance.get(f"{m.provider_id}/{m.id}")
        if not history or int(history.get("samples", 0) or 0) < _RELIABILITY_MIN_SAMPLES:
            return None
        return {
            "samples": history.get("samples"),
            "success_rate": history.get("success_rate"),
            "timeout_rate": history.get("timeout_rate"),
            "failure_rate": history.get("failure_rate"),
        }

    def explain_routing(
        self,
        decision: Decision,
        hardware: HardwareProfile,
        registry: list[ModelMetadata],
        provider_health: dict[str, bool],
        available_models: dict[str, set[str]] | None = None,
        *,
        performance: dict[str, dict] | None = None,
    ) -> list[dict]:
        """Return detailed routing analysis for all models.

        Each entry contains:
            - model: model id
            - suitable: whether it passes all hard constraints
            - exclusion_reason: why it was excluded (if applicable)
            - capability_score: raw capability match score
            - adjusted_score: after reliability/performance adjustments
            - rank: position in final ranking (1 = best)
            - scoring_breakdown: detailed scoring components
        """
        results = []
        candidates, excluded = self._filter(
            registry, hardware, decision, provider_health, available_models
        )

        # Add excluded models
        for exc in excluded:
            results.append({
                "model": exc["model"],
                "suitable": False,
                "exclusion_reason": exc["reason"],
                "capability_score": 0,
                "adjusted_score": 0,
                "rank": None,
                "scoring_breakdown": {},
            })

        # Score candidates
        required = set(decision.required_capabilities)
        preferred = [c for c in decision.preferred_capabilities if c not in required]
        tier = TierResolver().resolve(hardware).tier
        primary_cap = pick_primary_capability(decision.required_capabilities)
        is_deep = Capability.REASONING in required or Capability.MATH in required
        locked_model = get_exact_model(tier, primary_cap, is_deep_reasoning=is_deep)

        scored = []
        for m in candidates:
            raw_score = self._score(m, decision, required, preferred, 0, locked_model_id=locked_model)
            adjusted = self._reliability_adjust(raw_score, m, performance)
            breakdown = self._score_breakdown(m, decision, required, preferred)
            scored.append({
                "model": m.id,
                "suitable": True,
                "exclusion_reason": None,
                "capability_score": round(raw_score, 3),
                "adjusted_score": round(adjusted, 3),
                "rank": None,
                "scoring_breakdown": breakdown,
            })

        # Sort and rank
        scored.sort(key=lambda x: x["adjusted_score"], reverse=True)
        for i, entry in enumerate(scored):
            entry["rank"] = i + 1

        results.extend(scored)
        return results

    def _score_breakdown(
        self,
        m: ModelMetadata,
        decision: Decision,
        required: set[Capability],
        preferred: list[Capability],
    ) -> dict:
        """Detailed breakdown of scoring components."""
        breakdown = {
            "required_capabilities": {},
            "preferred_capabilities": {},
            "penalties": {},
            "bonuses": {},
        }

        # Required capability scores
        for cap in required:
            score = m.capabilities.score_for(cap)
            weight = _REQUIRED_WEIGHTS.get(cap, 0.0)
            breakdown["required_capabilities"][cap.value] = {
                "score": round(score, 3),
                "weight": weight,
                "contribution": round(score * weight, 3),
            }

        # Preferred capability scores
        for cap in preferred:
            score = m.capabilities.score_for(cap)
            weight = _REQUIRED_WEIGHTS.get(cap, 0.0) * _PREFERRED_MULTIPLIER
            breakdown["preferred_capabilities"][cap.value] = {
                "score": round(score, 3),
                "weight": round(weight, 3),
                "contribution": round(score * weight, 3),
            }

        # Penalties
        has_vision_req = (Capability.VISION in required or Capability.IMAGE_UNDERSTANDING in required)
        is_vision_model = (
            bool(m.capabilities.vision)
            or "vl" in m.id.lower()
            or "moondream" in m.id.lower()
        )
        if not has_vision_req:
            if is_vision_model:
                breakdown["penalties"]["vision_unused"] = 1.0
            if m.capabilities.ocr > 0:
                breakdown["penalties"]["ocr_unused"] = _OCR_UNUSED_PENALTY
            if m.capabilities.pdf > 0:
                breakdown["penalties"]["pdf_unused"] = _PDF_UNUSED_PENALTY

        # Bonuses
        breakdown["bonuses"]["local"] = _LOCAL_BIAS if m.kind == ProviderKind.LOCAL else 0
        breakdown["bonuses"]["privacy"] = round(m.privacy_score * _PRIVACY_BIAS, 3)
        breakdown["bonuses"]["latency"] = round(
            _LATENCY_WEIGHT.get(m.latency, 0.5) * _LATENCY_BONUS, 3
        )

        return breakdown

    def _reliability_note(self, m: ModelMetadata, performance: dict[str, dict] | None) -> str:
        summary = self._reliability_summary(m, performance)
        if not summary:
            return ""
        return (
            f", reliability {summary['success_rate']:.0%} "
            f"({summary['samples']} samples)"
        )

    def _score(
        self,
        m: ModelMetadata,
        decision: Decision,
        required: set[Capability],
        preferred: list[Capability],
        complexity: int,
        locked_model_id: str | None = None,
    ) -> float:
        score = 0.0

        # Exact Model Matrix locked priority
        if locked_model_id and matches_model_id(m.id, locked_model_id):
            score += 2.0 if complexity < _DEEP_REASONING_THRESHOLD else 0.8

        for cap in required:
            score += m.capabilities.score_for(cap) * _REQUIRED_WEIGHTS.get(cap, 0.0)
        for cap in preferred:
            score += m.capabilities.score_for(cap) * _REQUIRED_WEIGHTS.get(cap, 0.0) * _PREFERRED_MULTIPLIER

        # Deep-reasoning tasks favor reasoning specialists.
        if complexity >= _DEEP_REASONING_THRESHOLD:
            score += m.capabilities.reasoning * _DEEP_REASONING_BONUS

        # Strengths and weaknesses matching
        weaknesses_text = " ".join(m.weaknesses).lower() if m.weaknesses else ""
        strengths_text = " ".join(m.strengths).lower() if m.strengths else ""

        if Capability.MATH in required or Capability.REASONING in required:
            if "deep math" in weaknesses_text or "mathematical proof" in weaknesses_text:
                score -= 0.30
            if "math reasoning" in strengths_text or "advanced reasoning" in strengths_text or "deep math" in strengths_text:
                score += 0.20

        if Capability.CODING in required:
            if "heavy code" in weaknesses_text or "code generation" in weaknesses_text:
                score -= 0.30
            if "code generation" in strengths_text or "refactoring" in strengths_text or "debugging" in strengths_text:
                score += 0.15

        # Unused specializations are penalized.
        has_vision_req = (Capability.VISION in required or Capability.IMAGE_UNDERSTANDING in required)
        if not has_vision_req:
            if m.capabilities.vision or "vl" in m.id.lower() or "moondream" in m.id.lower():
                score -= _VISION_UNUSED_PENALTY
            if m.capabilities.ocr > 0:
                score -= _OCR_UNUSED_PENALTY
            if m.capabilities.pdf > 0:
                score -= _PDF_UNUSED_PENALTY
        if Capability.CODING not in required and m.capabilities.coding >= _CODING_UNUSED_THRESHOLD:
            score -= _CODING_UNUSED_PENALTY

        # Structural biases.
        score += _LOCAL_BIAS if m.kind == ProviderKind.LOCAL else 0.0
        score += m.privacy_score * _PRIVACY_BIAS
        # Only favor fast latency on simple tasks; do not let small models overpower reasoning on hard tasks
        latency_scale = 1.0 if complexity < _DEEP_REASONING_THRESHOLD else 0.2
        score += _LATENCY_WEIGHT.get(m.latency, 0.5) * _LATENCY_BONUS * latency_scale
        score -= min(_COST_PENALTY_CAP, m.estimated_cost_per_1k * _COST_SCALE)

        return max(0.0, score)

    # -- latency prediction -------------------------------------------------

    @staticmethod
    def _expected_output_tokens(complexity: int) -> int:
        """Stepwise mapping: harder tasks produce longer answers."""
        if complexity <= 3:
            return 20
        if complexity <= 15:
            return 100
        if complexity <= 30:
            return 200
        if complexity <= 50:
            return 400
        if complexity <= 75:
            return 800
        if complexity <= 90:
            return 1200
        return 1600

    @staticmethod
    def _params_b(m: ModelMetadata) -> float:
        if m.size_bytes:
            return max(0.5, m.size_bytes / 1.5e9)
        if m.required_ram_gb:
            return max(0.5, m.required_ram_gb / 2.0)
        return 3.0

    def _estimate_tokens_per_s(self, m: ModelMetadata, hardware: HardwareProfile) -> float:
        """Model-size and hardware-aware throughput estimate (tokens/sec)."""
        params = self._params_b(m)
        gpu = hardware.gpu and hardware.gpu.vram_gb and (
            not m.required_vram_gb or m.required_vram_gb <= hardware.gpu.vram_gb
        )
        base = 120.0 / params if gpu else 20.0 / params
        return max(1.0, round(base, 2))

    def _estimate_latency_s(
        self,
        m: ModelMetadata,
        prompt: str,
        complexity: int,
        hardware: HardwareProfile,
        performance: dict[str, dict] | None,
    ) -> float:
        history = (performance or {}).get(f"{m.provider_id}/{m.id}") or {}

        tps = None
        if history.get("samples", 0) >= 3 and history.get("avg_tokens_per_second"):
            tps = float(history["avg_tokens_per_second"])
        else:
            tps = self._estimate_tokens_per_s(m, hardware)

        prompt_tokens = max(1, len(prompt) // 4)
        expected = self._expected_output_tokens(complexity)
        latency = (prompt_tokens * 0.6 + expected) / tps

        if history.get("samples", 0) >= 3 and history.get("avg_latency_s"):
            latency = 0.5 * float(history["avg_latency_s"]) + 0.5 * latency
        return round(latency, 1)
