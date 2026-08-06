"""Abstract contracts.

Everything in Synapse depends on these interfaces — never on concrete vendor
classes. Adding a provider, registry backend, hardware probe, config source,
analyzer, router, or executor requires implementing the relevant contract and
registering it via DI.
"""

from synapse.contracts.analyzer import ComplexityAnalyzer, IntentAnalyzer, PrivacyAnalyzer
from synapse.contracts.citations import CitationEngine, GroundingValidator
from synapse.contracts.config_provider import ConfigProvider
from synapse.contracts.confidence import ConfidenceEngine
from synapse.contracts.context import ContextBuilder, ContextPolicy
from synapse.contracts.correction import HallucinationDetector, SelfCorrector
from synapse.contracts.decision_engine import DecisionEngine
from synapse.contracts.escalation import EscalationEngine, EscalationPolicy
from synapse.contracts.executor import Executor
from synapse.contracts.hardware_provider import HardwareProvider
from synapse.contracts.memory import MemoryStore
from synapse.contracts.model_provider import ModelProvider
from synapse.contracts.model_registry import ModelRegistry
from synapse.contracts.performance import PerformanceStore
from synapse.contracts.planner import ExecutionPlanner
from synapse.contracts.router import Router
from synapse.contracts.synthesizer import Synthesizer
from synapse.contracts.task_planner import TaskPlanner
from synapse.contracts.verification import CodeVerifier, FactChecker, MathVerifier, Verifier

__all__ = [
    "ConfigProvider",
    "HardwareProvider",
    "ModelProvider",
    "ModelRegistry",
    "IntentAnalyzer",
    "ComplexityAnalyzer",
    "PrivacyAnalyzer",
    "DecisionEngine",
    "ExecutionPlanner",
    "Router",
    "Executor",
    "PerformanceStore",
    "TaskPlanner",
    "MemoryStore",
    "Synthesizer",
    # Phase 4
    "ConfidenceEngine",
    "ContextBuilder",
    "ContextPolicy",
    "Verifier",
    "FactChecker",
    "CodeVerifier",
    "MathVerifier",
    "EscalationEngine",
    "EscalationPolicy",
    "HallucinationDetector",
    "SelfCorrector",
    "CitationEngine",
    "GroundingValidator",
]
