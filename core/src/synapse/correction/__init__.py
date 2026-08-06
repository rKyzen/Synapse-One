"""Phase 4: Correction / Hallucination Detection module."""

from synapse.correction.impl import HeuristicHallucinationDetector, HeuristicSelfCorrector

__all__ = ["HeuristicHallucinationDetector", "HeuristicSelfCorrector"]