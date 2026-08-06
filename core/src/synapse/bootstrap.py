"""Composition root — wires the full application graph via DI.

Called once at startup. Nothing here is business logic; it only assembles
services and their dependencies so no module constructs its own collaborators.
"""

from __future__ import annotations

from synapse.analyzers import ComplexityAnalyzer, IntentAnalyzer, PrivacyAnalyzer
from synapse.config.paths import SynapsePaths
from synapse.config.provider import ConfigProvider
from synapse.confidence.heuristic import HeuristicConfidenceEngine
from synapse.contracts import (
    CitationEngine,
    CodeVerifier,
    ComplexityAnalyzer as ComplexityAnalyzerProtocol,
    ConfidenceEngine,
    ConfigProvider as ConfigProviderProtocol,
    ContextBuilder,
    DecisionEngine as DecisionEngineProtocol,
    EscalationEngine,
    EscalationPolicy,
    ExecutionPlanner as ExecutionPlannerProtocol,
    FactChecker,
    GroundingValidator,
    HallucinationDetector,
    HardwareProvider,
    IntentAnalyzer as IntentAnalyzerProtocol,
    MathVerifier,
    MemoryStore,
    ModelRegistry,
    PerformanceStore,
    PrivacyAnalyzer as PrivacyAnalyzerProtocol,
    Router as RouterProtocol,
    Executor as ExecutorProtocol,
    SelfCorrector,
    Synthesizer,
    TaskPlanner,
    Verifier,
)
from synapse.context.builder import DefaultContextBuilder, DefaultContextPolicy
from synapse.correction.impl import HeuristicHallucinationDetector, HeuristicSelfCorrector
from synapse.citations.impl import GroundingValidatorImpl, InlineCitationEngine
from synapse.decision import DecisionEngine
from synapse.di import Container
from synapse.escalation.impl import CapabilityBasedEscalationPolicy, OllamaEscalationEngine
from synapse.events import EventBus
from synapse.execution import Executor, ExecutionPlanner
from synapse.hardware import HardwareScanner
from synapse.lifecycle import LifecycleSettings, ModelLifecycleManager
from synapse.master import MasterAgent
from synapse.memory import WorkspaceMemory
from synapse.performance import FilePerformanceStore
from synapse.planner import HeuristicTaskPlanner
from synapse.projects import WorkspaceSystem
from synapse.providers import ProviderFactory, ProviderManager
from synapse.registry import ConfigModelRegistry
from synapse.router import Router
from synapse.synthesis import TemplateSynthesizer
from synapse.verification.impl import CompositeVerifier, HeuristicCodeVerifier, HeuristicFactChecker, HeuristicMathVerifier
from synapse.workspace import Workspace

_VERSION = __import__("synapse", fromlist=["__version__"]).__version__


class Boot:
    """An assembled application root: container + resolved entry points."""

    def __init__(self, container: Container) -> None:
        self.container = container

    @property
    def providers(self) -> ProviderManager:
        return self.container.resolve(ProviderManager)

    @property
    def registry(self) -> ModelRegistry:
        return self.container.resolve(ModelRegistry)

    @property
    def hardware(self) -> HardwareProvider:
        return self.container.resolve(HardwareProvider)

    @property
    def events(self) -> EventBus:
        return self.container.resolve(EventBus)

    @property
    def master(self) -> MasterAgent:
        return self.container.resolve(MasterAgent)

    @property
    def lifecycle(self) -> ModelLifecycleManager:
        return self.container.resolve(ModelLifecycleManager)

    @property
    def workspace(self) -> Workspace:
        return self.container.resolve(Workspace)

    @property
    def projects(self) -> WorkspaceSystem:
        return self.container.resolve(WorkspaceSystem)

    def start(self) -> None:
        from synapse.logging import get_logger

        log = get_logger("synapse.bootstrap")
        log.info("synapse_boot", version=_VERSION)
        self.providers.load_all()
        self.lifecycle.preload_embedding()
        self.lifecycle.start_background_task()

    def shutdown(self) -> None:
        self.lifecycle.stop_background_task()
        self.providers.shutdown_all()
        try:
            self.projects.cleanup()
        except Exception:  # noqa: BLE001 - cleanup must never break shutdown
            pass


def create_container(paths: SynapsePaths | None = None, env: dict | None = None) -> Container:
    """Build and wire the container. ``env`` overrides real env (tests)."""
    paths = paths or SynapsePaths.discover()
    paths.ensure()

    container = Container()

    config = ConfigProvider(paths=paths, env=env).load()
    container.bind_instance(ConfigProviderProtocol, config)

    events = EventBus()
    container.bind_instance(EventBus, events)

    container.register(HardwareProvider, lambda: HardwareScanner(config, paths))
    container.register(ModelRegistry, lambda: ConfigModelRegistry(config, events))
    container.register(ProviderFactory, lambda: ProviderFactory(config, events))
    container.register(ProviderManager, lambda: ProviderManager(config, container.resolve(ProviderFactory), events))

    # Phase 2 — analyzers + pipeline.
    container.register(IntentAnalyzerProtocol, lambda: IntentAnalyzer())
    container.register(ComplexityAnalyzerProtocol, lambda: ComplexityAnalyzer())
    container.register(PrivacyAnalyzerProtocol, lambda: PrivacyAnalyzer())
    container.register(DecisionEngineProtocol, lambda: DecisionEngine())
    container.register(ExecutionPlannerProtocol, lambda: ExecutionPlanner())
    container.register(RouterProtocol, lambda: Router())
    container.register(
        ExecutorProtocol,
        lambda: Executor(container.resolve(ProviderManager)),
    )
    # Phase 2.5+ — model lifecycle manager (RAM resident-model management).
    container.register(
        ModelLifecycleManager,
        lambda: ModelLifecycleManager(
            settings=LifecycleSettings.from_config(config.settings.lifecycle),
            providers=container.resolve(ProviderManager),
            registry=container.resolve(ModelRegistry),
            router=container.resolve(RouterProtocol),
            events=events,
            hardware=container.resolve(HardwareProvider),
        ),
    )
    # Phase 3 — task orchestration, workspace memory, result synthesis.
    container.register(
        TaskPlanner,
        lambda: HeuristicTaskPlanner(**config.settings.planner),
    )
    container.register(
        MemoryStore,
        lambda: WorkspaceMemory(
            paths,
            config,
            container.resolve(ProviderManager),
            enabled=bool(config.get("memory.enabled", True)),
            top_k=int(config.get("memory.top_k", 3)),
            min_similarity=float(config.get("memory.min_similarity", 0.35)),
            embedding_model=config.get("memory.embedding_model"),
        ),
    )
    container.register(Synthesizer, lambda: TemplateSynthesizer())
    # Phase 4 — multimodal workspace (files, RAG, vision, code scan).
    container.register(
        Workspace,
        lambda: Workspace(
            paths,
            config,
            container.resolve(ProviderManager),
            container.resolve(ModelRegistry),
            events,
            memory=container.resolve(MemoryStore),
            lifecycle=container.resolve(ModelLifecycleManager),
        ),
    )
    # Phase 2.5 — performance learning loop.
    container.register(
        PerformanceStore,
        lambda: FilePerformanceStore(
            paths,
            enabled=bool(config.settings.performance.enabled),
            max_entries=config.settings.performance.max_entries,
            events=events,
        ),
    )
    # Phase 4 — Answer Quality & Reliability
    container.register(
        ConfidenceEngine,
        lambda: HeuristicConfidenceEngine(
            refuse_threshold=config.get("confidence.refuse_threshold", 0.3),
            verify_threshold=config.get("confidence.verify_threshold", 0.7),
            escalate_threshold=config.get("confidence.escalate_threshold", 0.6),
        ),
    )
    container.register(
        ContextBuilder,
        lambda: DefaultContextBuilder(
            policy=DefaultContextPolicy(),
            max_context_chars=int(config.get("context.max_chars", 8000)),
        ),
    )
    container.register(
        Verifier,
        lambda: CompositeVerifier(
            fact_checker=HeuristicFactChecker(),
            code_verifier=HeuristicCodeVerifier(),
            math_verifier=HeuristicMathVerifier(),
        ),
    )
    container.register(
        FactChecker,
        lambda: HeuristicFactChecker(),
    )
    container.register(
        CodeVerifier,
        lambda: HeuristicCodeVerifier(),
    )
    container.register(
        MathVerifier,
        lambda: HeuristicMathVerifier(),
    )
    container.register(
        EscalationPolicy,
        lambda: CapabilityBasedEscalationPolicy(),
    )
    container.register(
        EscalationEngine,
        lambda: OllamaEscalationEngine(container.resolve(ProviderManager)),
    )
    container.register(
        HallucinationDetector,
        lambda: HeuristicHallucinationDetector(),
    )
    container.register(
        SelfCorrector,
        lambda: HeuristicSelfCorrector(),
    )
    container.register(
        CitationEngine,
        lambda: InlineCitationEngine(),
    )
    container.register(
        GroundingValidator,
        lambda: GroundingValidatorImpl(),
    )
    container.register(
        MasterAgent,
        lambda: MasterAgent(
            intent_analyzer=container.resolve(IntentAnalyzerProtocol),
            complexity_analyzer=container.resolve(ComplexityAnalyzerProtocol),
            privacy_analyzer=container.resolve(PrivacyAnalyzerProtocol),
            decision_engine=container.resolve(DecisionEngineProtocol),
            planner=container.resolve(ExecutionPlannerProtocol),
            router=container.resolve(RouterProtocol),
            executor=container.resolve(ExecutorProtocol),
            providers=container.resolve(ProviderManager),
            hardware=container.resolve(HardwareProvider),
            registry=container.resolve(ModelRegistry),
            events=events,
            config=config,
            performance=container.resolve(PerformanceStore),
            lifecycle=container.resolve(ModelLifecycleManager),
            task_planner=container.resolve(TaskPlanner),
            memory=container.resolve(MemoryStore),
            synthesizer=container.resolve(Synthesizer),
            workspace=container.resolve(Workspace),
            confidence_engine=container.resolve(ConfidenceEngine),
            context_builder=container.resolve(ContextBuilder),
            verifier=container.resolve(Verifier),
            escalation_engine=container.resolve(EscalationEngine),
            escalation_policy=container.resolve(EscalationPolicy),
            hallucination_detector=container.resolve(HallucinationDetector),
            self_corrector=container.resolve(SelfCorrector),
            citation_engine=container.resolve(CitationEngine),
            grounding_validator=container.resolve(GroundingValidator),
        ),
    )
    # Phase 5 — Workspace System (projects, chats, per-project isolation).
    container.register(
        WorkspaceSystem,
        lambda: WorkspaceSystem(
            root=paths.data_dir / "projects",
            paths=paths,
            config=config,
            providers=container.resolve(ProviderManager),
            registry=container.resolve(ModelRegistry),
            events=events,
            lifecycle=container.resolve(ModelLifecycleManager),
            memory_config=dict(config.get("memory", {})),
        ),
    )
    return container