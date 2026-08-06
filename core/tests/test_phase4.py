"""Phase 4 tests — answer quality, reliability, and trustworthiness.

Covers the six new subsystems: confidence engine, context builder, verifier,
hallucination detection/self-correction, escalation, and citations/grounding.
"""

from __future__ import annotations

import pytest

from synapse.citations.impl import GroundingValidatorImpl, InlineCitationEngine
from synapse.confidence.heuristic import HeuristicConfidenceEngine
from synapse.context.builder import DefaultContextBuilder, DefaultContextPolicy
from synapse.correction.impl import HeuristicHallucinationDetector, HeuristicSelfCorrector
from synapse.domain import Capability, ChatRequest, ChatResponse, ModelDescriptor, ModelMetadata, ProviderKind
from synapse.domain.enums import MemoryScope
from synapse.escalation.impl import CapabilityBasedEscalationPolicy, OllamaEscalationEngine
from synapse.verification.impl import (
    CompositeVerifier,
    HeuristicCodeVerifier,
    HeuristicMathVerifier,
)
from synapse.verification.impl import HeuristicFactChecker
from synapse.contracts.verification import VerificationStatus


class _MockProvider:
    """Fake provider: no network, deterministic replies."""

    provider_id = "mock"
    kind = ProviderKind.LOCAL

    def __init__(self) -> None:
        self.calls: list[str] = []

    def initialize(self) -> None:
        pass

    def list_models(self) -> list[ModelDescriptor]:
        return [ModelDescriptor(id="mock-1", provider_id="mock")]

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request.messages[-1].content)
        return ChatResponse(
            provider_id="mock",
            model_id="mock-1",
            kind=self.kind,
            content="mock reply",
            raw={},
        )

    def health(self) -> bool:
        return True

    def supports(self, capability: Capability) -> bool:
        return True

    def shutdown(self) -> None:
        pass

    def to_metadata(self, descriptor: ModelDescriptor) -> ModelMetadata | None:
        return None


@pytest.fixture
def mock_boot_phase4(temp_paths, monkeypatch):
    """Boot with the mock provider registered and a single routing target."""
    from synapse.bootstrap import Boot, create_container

    monkeypatch.setenv("SYNAPSE_HOME", str(temp_paths.home))
    boot = Boot(create_container(paths=temp_paths))

    manager = boot.providers
    mock = _MockProvider()
    manager._providers["mock"] = mock
    from synapse.domain.enums import ProviderState

    manager._states["mock"] = ProviderState.READY

    registry = boot.registry
    registry._models["mock-1"] = ModelMetadata(
        id="mock-1",
        provider_id="mock",
        kind=ProviderKind.LOCAL,
        privacy_score=1.0,
        capabilities={"reasoning": 0.9, "coding": 0.9, "writing": 0.9, "math": 0.9},
    )
    return boot, mock


# --------------------------------------------------------------------------
# Confidence Engine
# --------------------------------------------------------------------------


class TestConfidenceEngine:
    def test_empty_response_scores_zero(self):
        engine = HeuristicConfidenceEngine()
        score = engine.score("", "hello")
        assert score.score == 0.0
        assert engine.should_refuse(score)

    def test_confident_response_scores_high(self):
        engine = HeuristicConfidenceEngine()
        prompt = "What is the capital of France and its population"
        response = "The capital of France is Paris. Paris has a population of roughly two million people living within the city limits."
        score = engine.score(response, prompt)
        assert score.score > 0.6
        assert not engine.should_escalate(score)

    def test_uncertainty_markers_lower_score(self):
        engine = HeuristicConfidenceEngine()
        score = engine.score(
            "I'm not sure, I don't know, maybe it could be Paris",
            "What is the capital of France",
        )
        assert score.score < 0.5

    def test_grounded_in_retrieval_scores_higher(self):
        engine = HeuristicConfidenceEngine()
        prompt = "Explain the nitrogen cycle"
        chunks = [{"text": "The nitrogen cycle moves nitrogen through the atmosphere, soil, and living organisms."}]
        grounded = engine.score(
            "The nitrogen cycle moves nitrogen through the atmosphere, soil, and living organisms.",
            prompt,
            retrieved_chunks=chunks,
        )
        ungrounded = engine.score(
            "The nitrogen cycle moves nitrogen through the atmosphere, soil, and living organisms.",
            prompt,
        )
        assert grounded.score > ungrounded.score

    def test_thresholds_configurable(self):
        engine = HeuristicConfidenceEngine(
            refuse_threshold=0.5, verify_threshold=0.8, escalate_threshold=0.75
        )
        score = engine.score("short", "a question with several words here")
        assert score.score < 0.75
        assert engine.should_escalate(score)
        assert engine.should_verify(score)


# --------------------------------------------------------------------------
# Context Builder
# --------------------------------------------------------------------------


class TestContextBuilder:
    def test_builds_bundle_with_counts(self):
        builder = DefaultContextBuilder()
        bundle = builder.build(
            "Summarize the report",
            task_capabilities=["writing"],
            memory_scope=MemoryScope.PROJECT,
            memory_results=[{"text": "the report covers quarterly revenue growth", "query": "report"}],
            retrieval_chunks=[{"text": "Quarterly revenue grew by twelve percent, as the report shows.", "file_name": "q3.md", "score": 0.8}],
            conversation_history=[{"role": "user", "content": "please summarize the report"}],
        )
        assert bundle.system_prompt
        assert "report" in bundle.user_prompt
        assert bundle.memory_entries_used == 1
        assert bundle.retrieval_chunks_used == 1
        assert bundle.conversation_turns_used == 1
        assert bundle.total_chars == len(bundle.user_prompt)

    def test_irrelevant_entries_filtered_by_policy(self):
        builder = DefaultContextBuilder()
        bundle = builder.build(
            "Refactor the database migration script",
            retrieval_chunks=[
                {"text": "migration script handles schema changes safely", "file_name": "migrate.py", "score": 0.9},
                {"text": "cooking pasta requires salted boiling water", "file_name": "kitchen.md", "score": 0.9},
            ],
        )
        assert bundle.retrieval_chunks_used == 1
        assert "migration" in bundle.user_prompt
        assert "cooking pasta" not in bundle.user_prompt

    def test_character_budget_enforced(self):
        builder = DefaultContextBuilder(max_context_chars=200)
        bundle = builder.build(
            "Tell me everything",
            retrieval_chunks=[{"text": "word " * 400, "file_name": "big.txt", "score": 0.9}],
        )
        assert bundle.total_chars <= 200 + len("…(context truncated)") + 2

    def test_policy_weights(self):
        policy = DefaultContextPolicy()
        assert policy.weight_context("user_preferences", []) == 1.0
        assert policy.weight_context("unknown", []) == 0.5
        assert policy.should_include_retrieval({"text": "x", "score": 0.1}, []) is False
        assert policy.should_include_retrieval({"text": "x", "score": 0.9}, []) is True


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


class TestVerification:
    def test_verifier_passes_clean_response(self):
        verifier = CompositeVerifier()
        result = verifier.verify(
            "Define a function that doubles a number",
            "Here is a function that doubles a number:\n```python\ndef double(x):\n    return x * 2\n```",
            capability="coding",
        )
        assert result.status is VerificationStatus.VERIFIED

    def test_verifier_catches_unbalanced_brackets(self):
        verifier = CompositeVerifier()
        result = verifier.verify(
            "Write a function",
            "```python\ndef f(x):\n    return (x * 2\n```",
            capability="coding",
        )
        assert result.status is not VerificationStatus.VERIFIED
        assert any("bracket" in c for c in result.corrections)

    def test_fact_checker_supports_grounded_claim(self):
        checker = HeuristicFactChecker()
        assert checker.check_claim(
            "The database migration script handles schema changes",
            ["The database migration script handles schema changes safely"],
        )["supported"]

    def test_fact_checker_flags_invented_claim(self):
        checker = HeuristicFactChecker()
        assert not checker.check_claim(
            "The unicorn module processes stardust quickly",
            ["The database migration script handles schema changes"],
        )["supported"]

    def test_math_verifier_evaluates_expression(self):
        verifier = HeuristicMathVerifier()
        result = verifier.verify_math("2 + 3 * 4")
        assert result["ok"] is True
        assert result["result"] == 14.0

    def test_math_verifier_rejects_unsafe_expression(self):
        verifier = HeuristicMathVerifier()
        result = verifier.verify_math("__import__('os').system('rm -rf /')")
        assert result["ok"] is False

    def test_code_verifier_balanced_fences(self):
        verifier = HeuristicCodeVerifier()
        result = verifier.verify_code("```python\nprint('hi')\n```", "python")
        assert result["ok"] is True


# --------------------------------------------------------------------------
# Hallucination Detection & Self-Correction
# --------------------------------------------------------------------------


class TestHallucinationCorrection:
    def test_detects_nonexistent_file(self):
        detector = HeuristicHallucinationDetector()
        flags = detector.scan(
            "Edit config.py to enable the feature",
            prompt="edit config",
            existing_files=["main.py", "README.md"],
        )
        assert any(f.type.value == "nonexistent_file" for f in flags)

    def test_does_not_flag_existing_file(self):
        detector = HeuristicHallucinationDetector()
        flags = detector.scan(
            "Edit main.py to enable the feature",
            prompt="edit config",
            existing_files=["main.py", "README.md"],
        )
        assert not any(f.type.value == "nonexistent_file" for f in flags)

    def test_detects_fake_citation_without_sources(self):
        detector = HeuristicHallucinationDetector()
        flags = detector.scan(
            "The result is confirmed by research, see [1]",
            prompt="research question",
            context=None,
        )
        assert any(f.type.value == "fake_citation" for f in flags)

    def test_no_fake_citation_when_context_present(self):
        detector = HeuristicHallucinationDetector()
        flags = detector.scan(
            "The result is confirmed by research, see [1]",
            prompt="research question",
            context="source text about the research",
        )
        assert not any(f.type.value == "fake_citation" for f in flags)

    def test_self_corrector_annotates_missing_files(self):
        corrector = HeuristicSelfCorrector()
        detector = HeuristicHallucinationDetector()
        flags = detector.scan(
            "Edit missing_file.py to enable the feature",
            prompt="edit",
            existing_files=["main.py"],
        )
        corrected = corrector.correct("Edit missing_file.py to enable the feature", flags, prompt="edit")
        assert "not found in workspace" in corrected

    def test_self_corrector_appends_disclaimer_when_no_fix(self):
        corrector = HeuristicSelfCorrector()
        from synapse.contracts.correction import HallucinationFlag, HallucinationType

        flags = [HallucinationFlag(type=HallucinationType.UNVERIFIABLE_CLAIM, span="x", reason="y", confidence=0.5)]
        corrected = corrector.correct("Some unverifiable claim here", flags, prompt="p")
        assert "could not be verified" in corrected


# --------------------------------------------------------------------------
# Escalation
# --------------------------------------------------------------------------


def _model(model_id: str, reasoning: float, kind: ProviderKind = ProviderKind.LOCAL, ram: float = 0.0) -> ModelMetadata:
    return ModelMetadata(
        id=model_id,
        provider_id="mock",
        kind=kind,
        required_ram_gb=ram,
        capabilities={"reasoning": reasoning, "coding": reasoning, "math": reasoning, "chat": reasoning},
    )


class TestEscalation:
    def test_policy_escalates_to_stronger_local_model(self):
        policy = CapabilityBasedEscalationPolicy()
        decision = policy.decide(
            current_model=_model("tiny", 0.3),
            confidence_score=0.2,
            available_models=[_model("tiny", 0.3), _model("big", 0.9)],
            task_capabilities=["reasoning"],
        )
        assert decision.should_escalate
        assert decision.target_model is not None
        assert decision.target_model.id == "big"

    def test_policy_does_not_escalate_at_acceptable_confidence(self):
        policy = CapabilityBasedEscalationPolicy()
        decision = policy.decide(
            current_model=_model("tiny", 0.3),
            confidence_score=0.8,
            available_models=[_model("tiny", 0.3), _model("big", 0.9)],
            task_capabilities=["reasoning"],
        )
        assert not decision.should_escalate

    def test_policy_skips_cloud_models(self):
        policy = CapabilityBasedEscalationPolicy()
        decision = policy.decide(
            current_model=_model("tiny", 0.3),
            confidence_score=0.1,
            available_models=[_model("tiny", 0.3), _model("cloud-big", 0.95, kind=ProviderKind.CLOUD)],
            task_capabilities=["reasoning"],
        )
        assert not decision.should_escalate
        assert decision.reason == "no stronger local model available"

    def test_engine_falls_back_when_provider_missing(self):
        class EmptyManager:
            def get(self, provider_id):
                return None

        engine = OllamaEscalationEngine(EmptyManager())  # type: ignore[arg-type]
        decision = CapabilityBasedEscalationPolicy().decide(
            current_model=_model("tiny", 0.3),
            confidence_score=0.2,
            available_models=[_model("tiny", 0.3), _model("big", 0.9)],
            task_capabilities=["reasoning"],
        )
        response, conf = engine.escalate("question", "weak answer", decision)
        assert response == "weak answer"
        assert conf == decision.current_score


# --------------------------------------------------------------------------
# Citations & Grounding
# --------------------------------------------------------------------------


class TestCitations:
    def test_footnote_citations_attached_when_overlapping(self):
        engine = InlineCitationEngine()
        grounded = engine.attach_citations(
            "The nitrogen cycle moves nitrogen through the atmosphere, soil, and living organisms.",
            [{"text": "The nitrogen cycle moves nitrogen through the atmosphere, soil, and living organisms.", "file_name": "science.md", "chunk_index": 3}],
            citation_format="footnote",
        )
        assert grounded.citations
        assert "science.md" in grounded.response
        assert grounded.citations[0].chunk_index == 3

    def test_no_citations_for_unrelated_chunks(self):
        engine = InlineCitationEngine()
        grounded = engine.attach_citations(
            "The capital of France is Paris",
            [{"text": "recipes for sourdough bread require flour", "file_name": "baking.md"}],
        )
        assert grounded.citations == []
        assert grounded.response == "The capital of France is Paris"

    def test_grounding_validator_flags_ungrounded_claims(self):
        validator = GroundingValidatorImpl()
        grounded, ungrounded = validator.validate(
            "The migration script handles schema changes. Unicorns power the engine.",
            [{"text": "The migration script handles schema changes safely."}],
        )
        assert not grounded
        assert any("Unicorns" in claim for claim in ungrounded)

    def test_grounding_validator_passes_grounded_response(self):
        validator = GroundingValidatorImpl()
        grounded, ungrounded = validator.validate(
            "The migration script handles schema changes safely.",
            [{"text": "The migration script handles schema changes safely."}],
        )
        assert grounded
        assert ungrounded == []


# --------------------------------------------------------------------------
# Bootstrap wiring
# --------------------------------------------------------------------------


def test_phase4_components_wired(container):
    from synapse.contracts import (
        CitationEngine,
        ConfidenceEngine,
        ContextBuilder,
        EscalationEngine,
        EscalationPolicy,
        GroundingValidator,
        HallucinationDetector,
        SelfCorrector,
        Verifier,
    )

    assert container.resolve(ConfidenceEngine) is not None
    assert container.resolve(ContextBuilder) is not None
    assert container.resolve(Verifier) is not None
    assert container.resolve(EscalationPolicy) is not None
    assert container.resolve(EscalationEngine) is not None
    assert container.resolve(HallucinationDetector) is not None
    assert container.resolve(SelfCorrector) is not None
    assert container.resolve(CitationEngine) is not None
    assert container.resolve(GroundingValidator) is not None


def test_master_process_runs_phase4_quality_loop(mock_boot_phase4):
    """End-to-end: the phase 4 quality pipeline runs without breaking the response."""
    boot, mock = mock_boot_phase4
    response = boot.master.process("Build a python script to fix this bug")
    assert response.response  # non-empty final answer
    assert response.model == "mock-1"
    assert mock.calls
