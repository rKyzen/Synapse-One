"""Master Agent — the orchestrator and ONLY public entry for AI requests.

The Master performs no provider-specific work. It:
    1. receives the user request
    2. loads the hardware profile
    3. queries the registry
    4. runs analyzers (intent, complexity, privacy)
    5. asks the Decision Engine
    6. builds the Execution Plan
    7. asks the Router (with performance history for latency prediction)
    8. executes via the Executor (through the ModelProvider interface)
    9. records the execution outcome into the PerformanceStore (the learning
       loop) and aggregates into a normalized AgentResponse + DecisionTrace
   10. Phase 4: confidence scoring, verification, escalation, self-correction,
       grounding, and citations.

Every dependency is an interface; every subsystem can be swapped independently.
"""

from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path

import structlog

from synapse.actions import (
    ActionEngine,
    IntentRouter,
    RequestKind,
    classify_request,
    extract_workspace_ops,
    requires_workspace_access,
)
from synapse.actions.followup import has_workspace_context, is_follow_up
from synapse.analyzers import ComplexityAnalyzer, IntentAnalyzer, PrivacyAnalyzer
from synapse.contracts import (
    CitationEngine,
    ConfigProvider,
    ConfidenceEngine,
    ContextBuilder,
    DecisionEngine,
    ExecutionPlanner,
    GroundingValidator,
    HardwareProvider,
    MemoryStore,
    ModelRegistry,
    PerformanceStore,
    Router,
    Synthesizer,
    TaskPlanner,
    Verifier,
)
from synapse.contracts.correction import HallucinationDetector, SelfCorrector
from synapse.contracts.escalation import EscalationEngine, EscalationPolicy
from synapse.contracts.verification import VerificationStatus
from synapse.decision import DecisionEngine as ConcreteDecisionEngine
from synapse.domain import (
    AgentResponse,
    Capability,
    ChatResponse,
    ComplexityResult,
    Decision,
    DecisionTrace,
    ExecutionGraph,
    ExecutionPlan,
    GraphEdge,
    GraphNode,
    IntentResult,
    MemoryScope,
    PrivacyMode,
    PrivacyResult,
    ProviderKind,
    RoutingDecision,
    Task,
    TaskKind,
    TaskStatus,
    WorkspaceOutcome,
)
from synapse.domain.enums import IntentKind, IntentType
from synapse.domain.fileops import FileAction, ValidationResult
from synapse.domain.tasks import TaskDAG
from synapse.events import EventBus, Events
from synapse.execution import Executor, ProviderUnavailable
from synapse.master.fast_path import FastPathResult, FastPathType, check_fast_path
from synapse.master.schemas import ExecutionMode, MasterAnalysis, ReasoningComplexity
from synapse.workspace.brief import build_workspace_brief
from synapse.workspace.manifest import (
    MANIFEST_INSTRUCTION,
    WORKSPACE_TOOL_INSTRUCTION,
    parse_file_manifest,
)
from synapse.workspace.operator import is_safe_relative_path

log = structlog.get_logger("synapse.master")

#: message returned when every task failed to route.
_NO_ROUTE_GUIDANCE = (
    "No model could satisfy the current constraints. "
    "Install or enable a matching provider/model in configuration, "
    "then retry."
)

#: claim verbs a model uses to assert it produced a file, followed by a path.
#: Prose lines matching this for a path the action log did not verify are
#: scrubbed from the final response (Phase XVI — responses reflect real
#: execution only, never model claims).
_FILE_CLAIM_RE = re.compile(
    r"\b(?:created|wrote|written|saved|generated|added|updated|modified|deleted|renamed|built|writes|creates)\b"
    r"[^.\n]{0,80}?\b([A-Za-z0-9_./\-]+\.[a-zA-Z0-9]+)\b",
    re.IGNORECASE,
)

#: a line that looks like a raw manifest the model echoed back (contains a
#: ``"path":`` key plus file/folder keys) — never legitimate prose.
_MANIFEST_FRAG_RE = re.compile(r'"path"\s*:\s*"')

#: an explicit file operand (a name with an extension, e.g. ``main.py``) — the
#: one signal that turns artifact wording into a real file-editing request even
#: when no workspace context exists yet (Phase XVIII).
_EXPLICIT_FILE_TARGET_RE = re.compile(r"(?:[\w\-/]+\.)[a-z0-9]+", re.IGNORECASE)


def _manifest_paths(result: str) -> list[str]:
    """Extract only the paths a file-manifest answer declares — never contents."""
    paths: list[str] = []
    for m in re.finditer(r'"path"\s*:\s*"([^"]+)"', result or ""):
        if is_safe_relative_path(m.group(1)):
            paths.append(m.group(1))
    return list(dict.fromkeys(paths))


def _is_scratch_project(project_id: str | None) -> bool:
    """True when the request rides the automatic scratch project (id
    ``general``, created at boot with only a generated README) instead of a
    user-adopted project. Scratch auto-generated files are NOT workspace
    context for follow-up resolution (Phases XVII–XVIII)."""
    return not project_id or project_id == "general"


class MasterAgent:
    """Orchestrates the full request pipeline. Interface-only dependencies."""

    def __init__(
        self,
        intent_analyzer: IntentAnalyzer,
        complexity_analyzer: ComplexityAnalyzer,
        privacy_analyzer: PrivacyAnalyzer,
        decision_engine: DecisionEngine,
        planner: ExecutionPlanner,
        router: Router,
        executor: Executor,
        providers: ProviderManager,
        hardware: HardwareProvider,
        registry: ModelRegistry,
        events: EventBus,
        config: ConfigProvider,
        performance: PerformanceStore | None = None,
        lifecycle=None,
        task_planner: TaskPlanner | None = None,
        memory: MemoryStore | None = None,
        synthesizer: Synthesizer | None = None,
        workspace: "Workspace | None" = None,
        # Phase 4
        confidence_engine: ConfidenceEngine | None = None,
        context_builder: ContextBuilder | None = None,
        verifier: Verifier | None = None,
        escalation_engine: EscalationEngine | None = None,
        escalation_policy: EscalationPolicy | None = None,
        hallucination_detector: HallucinationDetector | None = None,
        self_corrector: SelfCorrector | None = None,
        citation_engine: CitationEngine | None = None,
        grounding_validator: GroundingValidator | None = None,
    ) -> None:
        self._intent = intent_analyzer
        self._complexity = complexity_analyzer
        self._privacy = privacy_analyzer
        self._decision = decision_engine
        self._planner = planner
        self._router = router
        self._executor = executor
        self._providers = providers
        self._hardware = hardware
        self._registry = registry
        self._events = events
        self._config = config
        self._performance = performance
        self._lifecycle = lifecycle
        self._task_planner = task_planner
        self._memory = memory
        self._synthesizer = synthesizer
        self._workspace: "Workspace | None" = workspace
        # Phase 4
        self._confidence = confidence_engine
        self._context_builder = context_builder
        self._verifier = verifier
        self._escalation = escalation_engine
        self._escalation_policy = escalation_policy
        self._hallucination = hallucination_detector
        self._corrector = self_corrector
        self._citations = citation_engine
        self._grounding = grounding_validator

    def _analyze_request(self, prompt: str, conversation_id: str | None = None) -> MasterAnalysis:
        if self._task_planner is not None and hasattr(self._task_planner, "analyze"):
            try:
                return self._task_planner.analyze(prompt, conversation_id=conversation_id)
            except TypeError:
                return self._task_planner.analyze(prompt)
        from synapse.master.orchestrator import AIMasterOrchestrator
        return AIMasterOrchestrator._fallback_analysis(prompt, "direct fallback")

    # -- public entry -------------------------------------------------------

    def process(
        self,
        prompt: str,
        files: list[str] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        workspace=None,
        memory=None,
        # Phase 6 — per-project workspace tools (safe file CRUD + action log).
        file_operator=None,
        action_log=None,
        project_id: str | None = None,
        conversation_id: str | None = None,
        # Phase XIII — project identity surfaced inside every model prompt.
        project_name: str | None = None,
        project_path: str | None = None,
        # Phase XV — per-project indexing, change tracking, diagnostics.
        project_index=None,
        change_panel=None,
        diagnostics=None,
    ) -> AgentResponse:
        start = time.perf_counter()
        self._events.publish(Events.REQUEST_RECEIVED, {"prompt_length": len(prompt)})

        # 0. Deterministic Fast Path (Arithmetic, Greetings, Identity, Workspace, Status)
        fast_result = check_fast_path(prompt, has_files=bool(files))
        if fast_result.is_fast_path:
            if fast_result.path_type == FastPathType.WORKSPACE_QUERY and fast_result.direct_response is None:
                if file_operator is not None:
                    items = file_operator.list_tree()
                    if items:
                        lines = [f"- `{f.get('path')}` ({f.get('size', 0)} bytes)" for f in items]
                        fast_result.direct_response = "### Project Files:\n\n" + "\n".join(lines)
                    else:
                        fast_result.direct_response = "Project workspace is currently empty."
                else:
                    fast_result.direct_response = "No active workspace folder opened."

            elif fast_result.path_type == FastPathType.LOADED_MODELS and fast_result.direct_response is None:
                if self._lifecycle is not None:
                    loaded = self._lifecycle.list_loaded()
                    if loaded:
                        fast_result.direct_response = "### Currently Loaded Models:\n\n" + "\n".join(
                            [f"- **{m}** (RAM: {r:.1f} GB)" for m, r in loaded.items()]
                        )
                    else:
                        fast_result.direct_response = "No specialist models currently active in memory."
                else:
                    fast_result.direct_response = "No active loaded models reported."

            elif fast_result.path_type == FastPathType.SYSTEM_STATUS and fast_result.direct_response is None:
                hw = self._hardware.scan()
                gpu_str = f"{hw.gpu.name} ({hw.gpu.vram_gb:.1f} GB VRAM)" if hw.gpu and hw.gpu.name else "None"
                fast_result.direct_response = (
                    f"### System Status:\n\n"
                    f"- **CPU**: {hw.cpu.model}\n"
                    f"- **RAM**: {hw.memory.available_gb:.1f} GB available / {hw.memory.total_gb:.1f} GB total\n"
                    f"- **GPU**: {gpu_str}\n"
                    f"- **Local LLM Capable**: {'Yes' if hw.recommendations.can_run_local_llm else 'No'}"
                )

            if fast_result.direct_response is not None:
                elapsed_ms = (time.perf_counter() - start) * 1000
                timings = {
                    "fast_path_ms": round(elapsed_ms, 2),
                    "master_analysis_ms": 0.0,
                    "context_retrieval_ms": 0.0,
                    "worker_inference_ms": 0.0,
                    "tools_ms": 0.0,
                    "total_ms": round(elapsed_ms, 2),
                }
                self._save_memory(prompt, fast_result.direct_response, memory=memory, conversation_id=conversation_id)
                self._publish_timeline("fast_path", f"Fast path: {fast_result.path_type.value}")
                self._events.publish(Events.REQUEST_ROUTED, {"provider": "fast_path", "model": "deterministic"})
                self._events.publish(Events.REQUEST_COMPLETED, {"latency_ms": round(elapsed_ms, 1), "tasks": 0, "fast_path": True})

                hardware = self._hardware.scan()
                intent_res = IntentResult(primary=fast_result.intent, confidence=1.0)
                comp_res = ComplexityResult(score=fast_result.complexity)
                priv_res = PrivacyResult(mode=PrivacyMode.LOCAL_ONLY)
                dec_res = Decision(can_stay_local=True, preferred_kind=ProviderKind.LOCAL)
                routing_res = RoutingDecision(
                    provider_id="fast_path",
                    model_id="deterministic",
                    kind=ProviderKind.LOCAL,
                    reason=f"deterministic fast path: {fast_result.path_type.value}",
                    capability_score=1.0,
                )
                fast_analysis = MasterAnalysis(
                    intent="direct_answer",
                    domain="general",
                    goal=prompt[:80],
                    workspace_needed=False,
                    files_needed=False,
                    artifact_required=False,
                    reasoning_complexity=ReasoningComplexity.TRIVIAL,
                    required_capabilities=["chat"],
                    recommended_model_role="Fast Path",
                    execution_mode=ExecutionMode.DIRECT_ANSWER,
                    confidence=1.0,
                )
                trace = self._build_trace(
                    intent_res, comp_res, priv_res, dec_res, routing_res, hardware, elapsed_ms, response=None,
                    master_analysis=fast_analysis,
                    timings=timings,
                )
                return AgentResponse(
                    response=fast_result.direct_response,
                    intent=fast_result.intent,
                    complexity=fast_result.complexity,
                    privacy=PrivacyMode.LOCAL_ONLY,
                    provider="fast_path",
                    model="deterministic",
                    execution_plan=ExecutionPlan(
                        intent=fast_result.intent,
                        complexity=fast_result.complexity,
                        privacy=PrivacyMode.LOCAL_ONLY,
                    ),
                    decision_trace=trace,
                    latency_ms=round(elapsed_ms, 1),
                    execution_graph=ExecutionGraph(total_latency_ms=round(elapsed_ms, 1)),
                    workspace=None,
                    actions=[],
                )
            elif fast_result.path_type == FastPathType.FAST_CHAT:
                master_analysis = MasterAnalysis(
                    intent="direct_answer",
                    domain="general",
                    goal=prompt[:80],
                    workspace_needed=False,
                    files_needed=False,
                    artifact_required=False,
                    reasoning_complexity=ReasoningComplexity.TRIVIAL,
                    required_capabilities=["chat"],
                    recommended_model_role="General Chat",
                    execution_mode=ExecutionMode.DIRECT_ANSWER,
                    confidence=1.0,
                )
                master_analysis_ms = 0.0

        if not fast_result.is_fast_path or fast_result.path_type != FastPathType.FAST_CHAT:
            # 1. Master Model Analysis First (runs before touching workspace or tools)
            analysis_start = time.perf_counter()
            master_analysis = self._analyze_request(prompt, conversation_id=conversation_id)
            if files:
                master_analysis.files_needed = True
                master_analysis.workspace_needed = True
            master_analysis_ms = (time.perf_counter() - analysis_start) * 1000

        self._publish_timeline("master_analysis", "Master Model analyzing request…")
        self._publish_timeline("intent_decision", f"Intent: {master_analysis.intent.replace('_', ' ').title()}")
        self._publish_timeline("domain_decision", f"Domain: {master_analysis.domain.replace('_', ' ').title()}")
        self._publish_timeline("artifact_decision", f"Artifact required: {'Yes' if master_analysis.artifact_required else 'No'}")
        self._publish_timeline("files_decision", f"Files required: {'Yes' if master_analysis.files_needed else 'No'}")
        self._publish_timeline("workspace_decision", f"Workspace required: {'Yes' if master_analysis.workspace_needed else 'No'}")
        caps_label = " + ".join([c.replace("_", " ").title() for c in master_analysis.required_capabilities]) or "General"
        self._publish_timeline("capability_decision", f"Capability: {caps_label}")

        kind = classify_request(prompt)
        if master_analysis.artifact_required or master_analysis.execution_mode in (
            ExecutionMode.ARTIFACT_GENERATION, ExecutionMode.WORKSPACE_AGENT
        ):
            if kind not in (
                RequestKind.FILE_CREATION,
                RequestKind.PROJECT_GENERATION,
                RequestKind.FILE_MODIFICATION,
                RequestKind.DOCUMENTATION,
            ):
                kind = (
                    RequestKind.PROJECT_GENERATION
                    if "project" in master_analysis.domain or "web" in master_analysis.domain
                    else RequestKind.FILE_CREATION
                )

        intent_kind = IntentRouter().route(prompt)
        if master_analysis.artifact_required:
            intent_kind = IntentKind.FILE_GENERATION

        log.info(
            "request_classified",
            kind=kind.value,
            intent=intent_kind.value,
            master_intent=master_analysis.intent,
            domain=master_analysis.domain,
            artifact_required=master_analysis.artifact_required,
            files_needed=master_analysis.files_needed,
            workspace_needed=master_analysis.workspace_needed,
            prompt=prompt[:120],
        )

        ws = workspace if workspace is not None else self._workspace
        mem = memory if memory is not None else self._memory

        hardware = self._hardware.scan()
        self._registry.sync(self._providers.all())
        registry_models = self._registry.all()

        complexity_map = {
            ReasoningComplexity.TRIVIAL: 10,
            ReasoningComplexity.EASY: 30,
            ReasoningComplexity.MEDIUM: 55,
            ReasoningComplexity.HARD: 82,
            ReasoningComplexity.VERY_HARD: 95,
        }
        master_complexity_score = complexity_map.get(master_analysis.reasoning_complexity, 30)

        complexity = self._complexity.analyze(prompt)
        if master_analysis.reasoning_complexity in (ReasoningComplexity.HARD, ReasoningComplexity.VERY_HARD):
            complexity = complexity.model_copy(
                update={"score": max(complexity.score, master_complexity_score)}
            )
        elif master_analysis.reasoning_complexity == ReasoningComplexity.TRIVIAL:
            complexity = complexity.model_copy(
                update={"score": min(complexity.score, master_complexity_score)}
            )

        intent = self._intent.analyze(prompt)
        if master_analysis.domain in ("mathematics", "math"):
            intent = intent.model_copy(update={"primary": IntentType.RESEARCH})
        elif master_analysis.domain in ("logic", "reasoning", "puzzle"):
            intent = intent.model_copy(update={"primary": IntentType.RESEARCH})
        elif master_analysis.artifact_required or master_analysis.coding_needed:
            intent = intent.model_copy(update={"primary": IntentType.CODING})
        elif (
            not master_analysis.coding_needed
            and intent.primary is IntentType.CODING
            and not _EXPLICIT_FILE_TARGET_RE.search(prompt)
            and kind not in (RequestKind.FILE_MODIFICATION, RequestKind.FILE_CREATION, RequestKind.PROJECT_GENERATION)
        ):
            intent = intent.model_copy(update={"primary": IntentType.CONVERSATION})

        privacy = self._privacy.analyze(
            prompt,
            complexity,
            context={
                "user_preference": self._config.get("privacy.user_preference", "balanced"),
                "provider_availability": self._provider_availability(),
            },
        )
        self._events.publish(Events.REQUEST_ANALYZED, {"intent": intent.primary.value, "complexity": complexity.score})

        follow_up_override = False
        if intent_kind.is_chat_only and not master_analysis.artifact_required and not master_analysis.files_needed:
            if (
                is_follow_up(prompt)
                and not _is_scratch_project(project_id)
                and has_workspace_context(
                    memory=mem,
                    conversation_id=conversation_id,
                    file_operator=file_operator,
                    action_log=action_log,
                )
            ):
                follow_up_override = True
                intent_kind = IntentKind.FILE_EDITING
                log.info(
                    "follow_up_context_resolved",
                    prompt=prompt[:120],
                    kind=kind.value,
                )
            else:
                return self._chat_direct(
                    prompt, intent, complexity, privacy, hardware, registry_models,
                    model=model, temperature=temperature, max_tokens=max_tokens,
                    files=files, ws=ws, memory=mem, conversation_id=conversation_id,
                    start=start, master_analysis=master_analysis,
                    master_analysis_ms=master_analysis_ms, context_retrieval_ms=0.0,
                    project_id=project_id, project_name=project_name, project_path=project_path,
                    file_operator=file_operator,
                )
        elif (
            intent_kind is IntentKind.FILE_EDITING
            and not master_analysis.artifact_required
            and not _EXPLICIT_FILE_TARGET_RE.search(prompt)
            and (
                _is_scratch_project(project_id)
                or not has_workspace_context(
                    memory=mem,
                    conversation_id=conversation_id,
                    file_operator=file_operator,
                    action_log=action_log,
                )
            )
        ):
            intent_kind = IntentKind.QUESTION_ANSWERING
            return self._chat_direct(
                prompt, intent, complexity, privacy, hardware, registry_models,
                model=model, temperature=temperature, max_tokens=max_tokens,
                files=files, ws=ws, memory=mem, conversation_id=conversation_id,
                start=start, master_analysis=master_analysis,
                master_analysis_ms=master_analysis_ms, context_retrieval_ms=0.0,
                project_id=project_id, project_name=project_name, project_path=project_path,
                file_operator=file_operator,
            )

        # Context loading (strictly opt-in based on Master Analysis)
        context_retrieval_start = time.perf_counter()
        workspace_outcome: WorkspaceOutcome | None = None
        if (master_analysis.workspace_needed or master_analysis.files_needed) and ws is not None and ws.enabled:
            if files:
                self._events.publish(Events.FILE_ATTACHED, {"file_ids": files})
            context_text, workspace_outcome = ws.prepare(prompt, files)

        decision = self._decision.decide(intent, complexity, privacy, hardware, registry_models)
        req_caps = list(decision.required_capabilities or [])
        if master_analysis.domain in ("mathematics", "math") or master_analysis.reasoning_complexity in (
            ReasoningComplexity.HARD, ReasoningComplexity.VERY_HARD
        ):
            if Capability.REASONING not in req_caps:
                req_caps.append(Capability.REASONING)
            if Capability.MATH not in req_caps and master_analysis.domain in ("mathematics", "math"):
                req_caps.append(Capability.MATH)
        if (master_analysis.coding_needed or master_analysis.artifact_required) and Capability.CODING not in req_caps:
            req_caps.append(Capability.CODING)
        if not master_analysis.coding_needed and not master_analysis.artifact_required and Capability.CODING in req_caps:
            req_caps = [c for c in req_caps if c != Capability.CODING]

        decision = decision.model_copy(update={"required_capabilities": req_caps})

        if workspace_outcome is not None and workspace_outcome.local_only and (
            workspace_outcome.context_chars > 0 or workspace_outcome.vision_descriptions
        ):
            decision = decision.model_copy(
                update={
                    "can_stay_local": True,
                    "use_cloud_reasoning": False,
                    "preferred_kind": ProviderKind.LOCAL,
                    "privacy": PrivacyMode.LOCAL_ONLY,
                    "reasoning": [
                        *decision.reasoning,
                        "workspace content kept local (allow_cloud_forwarding=false)",
                    ],
                }
            )
        plan = self._planner.build_plan(decision, intent, complexity, privacy)

        provider_ids = sorted({m.provider_id for m in registry_models} | set(self._providers.provider_ids()))
        health = {pid: self._providers.health(pid) for pid in provider_ids}
        available = self._provider_available_models()
        perf_stats = self._performance.stats() if self._performance else None

        brief = build_workspace_brief(
            project_id=project_id,
            project_name=project_name,
            project_path=project_path,
            file_operator=file_operator,
            memory=mem,
            conversation_id=conversation_id,
        )

        web_start = time.perf_counter()
        web_context = ""
        web_ms = 0.0
        if master_analysis is not None and master_analysis.web_needed:
            self._publish_timeline("web_search", f"Searching web for: {prompt[:60]}…")
            try:
                from synapse.pipeline.tools import ToolRegistry
                tool_reg = ToolRegistry()
                search_results = tool_reg._web_search(prompt, max_results=4)
                if search_results:
                    res_lines = []
                    for item in search_results:
                        title = item.get("title", "")
                        snippet = item.get("snippet", "")
                        url = item.get("url", "")
                        res_lines.append(f"- {title} ({url}):\n  {snippet}")
                    web_context = "Fresh Web Search Results:\n" + "\n".join(res_lines) + "\n\n"
            except Exception as exc:  # noqa: BLE001
                log.warning("web_search_failed", error=str(exc)[:200])
            web_ms = (time.perf_counter() - web_start) * 1000

        context = None
        if file_operator is not None and (
            master_analysis.workspace_needed
            or master_analysis.artifact_required
            or requires_workspace_access(prompt)
            or follow_up_override
        ):
            self._publish_timeline("read_workspace", "Reading workspace…")
            context = ActionEngine(file_operator).build_context(
                on_file=lambda p: self._publish_timeline("read_file", f"Reading {p}")
            )
        if web_context:
            context = f"{web_context}{context}" if context else web_context
        context_retrieval_ms = (time.perf_counter() - context_retrieval_start) * 1000

        dag = self._plan_tasks(prompt, intent, complexity, privacy, decision, master_analysis=master_analysis)
        log.info(
            "task_plan",
            kind=kind.value,
            tasks=[
                {
                    "id": t.id,
                    "kind": t.kind.value,
                    "file_output": bool(t.file_output),
                    "hint": t.file_hint,
                    "description": t.description[:120],
                }
                for t in dag.tasks
            ],
        )
        self._events.publish(Events.TASK_PLANNED, {"task_count": len(dag.tasks)})

        # Phase 7 — explicit workspace operations are answered by the backend
        # alone (list / search / rename / move / delete / folders): no model is
        # consulted and no code is ever printed into the chat.
        backend_ops = (
            extract_workspace_ops(prompt)
            if file_operator is not None and kind is RequestKind.WORKSPACE_OPERATION
            else []
        )
        backend_only = bool(backend_ops)
        actions: list[FileAction] = []
        validations: list[ValidationResult] = []
        verification_ms = 0.0
        worker_inference_start = time.perf_counter()
        if backend_only:
            lines, actions = ActionEngine(file_operator).run_workspace_ops(
                backend_ops, on_step=self._timeline_for_op
            )
            final_response = "\n".join(lines)
            graph, executed, primary_routing = ExecutionGraph(), [], None
            worker_inference_ms = 0.0
            tools_ms = (time.perf_counter() - worker_inference_start) * 1000
        else:
            # Execute DAG with Phase 4 quality checks
            graph, executed, primary_routing = self._execute_dag_with_quality(
                dag, prompt, decision, hardware, registry_models, health, available, complexity.score, perf_stats,
                workspace_outcome, files, temperature=temperature, max_tokens=max_tokens,
                workspace=ws, memory=mem, conversation_id=conversation_id,
                file_operator=file_operator, kind=kind,
                project_id=project_id,
                project_name=project_name,
                project_path=project_path,
                workspace_context=context,
            )
            worker_inference_ms = (time.perf_counter() - worker_inference_start) * 1000

            # Phase 6/7 — apply generated files to the project workspace via
            # the Action Engine and validate them. ``actions`` drive the
            # summary block and the action log.
            tools_start = time.perf_counter()
            if file_operator is not None:
                actions, validations = self._apply_file_outputs(dag, executed, file_operator, kind=kind)
                ver_start = time.perf_counter()
                # Phase XIII — write → verify → index → remember: confirm the
                # files landed, syntax-check what we can, refresh the project
                # index and record the write at project scope.
                self._verify_and_remember(
                    actions, file_operator,
                    workspace=ws, memory=mem, conversation_id=conversation_id,
                    project_index=project_index, change_panel=change_panel,
                    diagnostics=diagnostics,
                )
                verification_ms = (time.perf_counter() - ver_start) * 1000
            tools_ms = (time.perf_counter() - tools_start) * 1000

            # Synthesize final response (excluding raw file manifests, which
            # are replaced by the action summary).
            prose = self._synthesize(prompt, [t for t in executed if not t.file_output], dag)
            summary_only = (
                kind
                in (
                    RequestKind.FILE_CREATION,
                    RequestKind.FILE_MODIFICATION,
                    RequestKind.PROJECT_GENERATION,
                    RequestKind.DOCUMENTATION,
                )
                and file_operator is not None
            )
            final_response = self._compose_response(
                prose, actions, validations, dag, summary_only=summary_only
            )

        # Phase 6 — audit trail: plan, models, tools, files, failures, time.
        if action_log is not None:
            self._record_action_log(
                action_log, prompt, dag, actions, validations, project_id=project_id,
                elapsed_ms=(time.perf_counter() - start) * 1000,
            )

        # Phase 4: Post-synthesis quality checks on final response
        if self._confidence and executed:
            # Score the synthesized response
            primary_model = primary_routing.model_id if primary_routing else (executed[0].model_id if executed else None)
            retrieved_chunks = workspace_outcome.retrieval if workspace_outcome else None
            memory_entries = mem.recent(MemoryScope.CONVERSATION, 5, conversation=conversation_id) if mem else None

            conf_score = self._confidence.score(
                final_response,
                prompt,
                context=workspace_outcome.retrieval if workspace_outcome else None,
                retrieved_chunks=[{"text": c.text, "file_name": c.file_name, "score": c.score} for c in retrieved_chunks] if retrieved_chunks else None,
                model_id=primary_model,
                capability_requirements=decision.required_capabilities,
            )

            # Self-correction / hallucination detection
            if self._hallucination and not actions:
                # Gather project context for hallucination detection
                existing_files = []
                if file_operator is not None:
                    for entry in file_operator.list_tree():
                        existing_files.append(entry["path"])
                        existing_files.append(Path(entry["path"]).name)
                elif ws is not None:
                    existing_files = [f.name for f in ws.files.list()]

                installed_packages = []  # Could be populated from project
                project_functions = []  # Could be populated from code index

                flags = self._hallucination.scan(
                    final_response,
                    prompt=prompt,
                    context=workspace_outcome.retrieval if workspace_outcome else None,
                    existing_files=existing_files,
                    installed_packages=installed_packages,
                    project_functions=project_functions,
                )
                if flags and self._corrector:
                    final_response = self._corrector.correct(final_response, flags, prompt=prompt)

            # Escalation if confidence is low
            if self._escalation and self._escalation_policy and self._confidence.should_escalate(conf_score):
                current_model = None
                if primary_model:
                    current_model = next((m for m in registry_models if m.id == primary_model), None)
                esc_decision = self._escalation_policy.decide(
                    current_model,
                    conf_score.score,
                    list(registry_models),
                    [c.value for c in (decision.required_capabilities or [])],
                    hardware_constraints={"ram_gb": hardware.memory.total_gb},
                )
                if esc_decision.should_escalate and esc_decision.target_model is not None:
                    escalated_response, new_conf = self._escalation.escalate(
                        prompt,
                        final_response,
                        esc_decision,
                        context=workspace_outcome.retrieval if workspace_outcome else None,
                    )
                    final_response = escalated_response
                    log.info(
                        "escalation_applied",
                        to_model=esc_decision.target_model.id,
                        new_confidence=round(new_conf, 2),
                    )

            # Citation attachment
            if self._citations and retrieved_chunks:
                grounded = self._citations.attach_citations(
                    final_response,
                    [{"text": c.text, "file_name": c.file_name, "chunk_index": c.chunk_index,
                      "page": c.page, "lines_start": c.lines_start, "lines_end": c.lines_end, "score": c.score}
                     for c in retrieved_chunks],
                    citation_format="footnote",
                )
                final_response = grounded.response

            # Grounding validation
            if self._grounding and retrieved_chunks:
                is_grounded, ungrounded = self._grounding.validate(
                    final_response,
                    [{"text": c.text} for c in retrieved_chunks],
                    require_citation_for_claims=True,
                )
                if not is_grounded:
                    log.warning("response_not_grounded", ungrounded_claims=ungrounded[:3])

        if workspace_outcome is not None and workspace_outcome.vision_descriptions and final_response:
            final_response = (
                "Vision analysis:\n\n"
                + "\n\n".join(workspace_outcome.vision_descriptions)
                + "\n\n"
                + final_response
            )
        self._save_memory(prompt, final_response, memory=mem, conversation_id=conversation_id)

        elapsed_ms = (time.perf_counter() - start) * 1000
        timings = {
            "master_analysis_ms": round(master_analysis_ms, 2),
            "context_retrieval_ms": round(context_retrieval_ms, 2),
            "model_load_ms": 0.0,
            "worker_inference_ms": round(worker_inference_ms, 2),
            "tools_ms": round(tools_ms, 2),
            "web_ms": round(web_ms, 2),
            "verification_ms": round(verification_ms, 2),
            "total_ms": round(elapsed_ms, 2),
        }
        routing = primary_routing or RoutingDecision(reason="no task routed")
        if primary_routing is None and not backend_only:
            final_response = _NO_ROUTE_GUIDANCE
        self._events.publish(Events.REQUEST_ROUTED, {"provider": routing.provider_id, "model": routing.model_id})
        graph.total_latency_ms = round(elapsed_ms, 1)
        trace = self._build_trace(
            intent, complexity, privacy, decision, routing, hardware, elapsed_ms, response=None,
            master_analysis=master_analysis,
            timings=timings,
        )
        self._events.publish(Events.REQUEST_COMPLETED, {"latency_ms": round(elapsed_ms, 1), "tasks": len(dag.tasks)})
        log.info(
            "request_timing_breakdown",
            master_analysis_ms=round(master_analysis_ms, 2),
            context_retrieval_ms=round(context_retrieval_ms, 2),
            model_load_ms=0.0,
            worker_inference_ms=round(worker_inference_ms, 2),
            tools_ms=round(tools_ms, 2),
            web_ms=round(web_ms, 2),
            verification_ms=round(verification_ms, 2),
            total_ms=round(elapsed_ms, 2),
        )
        log.info(
            "request_completed",
            latency_ms=round(elapsed_ms, 1),
            tasks=len(dag.tasks),
            provider=routing.provider_id,
            model=routing.model_id,
            kind=kind.value,
            files_created=len([a for a in actions if a.action in ("created", "created_folder") and a.status == "ok"]),
            files_modified=len([a for a in actions if a.action == "modified" and a.status == "ok"]),
            action_failures=len([a for a in actions if a.status == "failed"]),
        )

        return AgentResponse(
            response=final_response,
            intent=intent.primary,
            complexity=complexity.score,
            privacy=privacy.mode,
            provider=routing.provider_id,
            model=routing.model_id,
            execution_plan=plan,
            decision_trace=trace,
            latency_ms=round(elapsed_ms, 1),
            execution_graph=graph,
            workspace=workspace_outcome,
            actions=actions,
        )

    # -- task orchestration -------------------------------------------------

    def _chat_direct(
        self,
        prompt: str,
        intent,
        complexity,
        privacy,
        hardware,
        registry_models,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        files: list[str] | None = None,
        ws=None,
        memory=None,
        conversation_id: str | None = None,
        start: float,
        master_analysis: MasterAnalysis | None = None,
        master_analysis_ms: float = 0.0,
        context_retrieval_ms: float = 0.0,
        project_id: str | None = None,
        project_name: str | None = None,
        project_path: str | None = None,
        file_operator=None,
    ) -> AgentResponse:
        """Chat-only request path (AI Operating Workspace Intent Router).

        A conversation or question-answering intent never enters the workspace
        pipeline: no planner, no brief, no Action Engine, no folders, no
        manifests, no files, no JSON. One decision → one route → one model
        call → the answer, plus conversation memory so chat continuity works.

        Chat stays workspace-AWARE without becoming workspace-WRITING: when a
        workspace exists, its index/vision retrieval is used read-only to
        answer (e.g. "summarize the design notes", attached screenshots), but
        nothing is planned, written, or recorded as an action.
        """
        # A chat path is by definition NOT an editing pipeline: when the
        # wording analyzer still reports a coding primary (bug/fix/script
        # words), demanding CODING capabilities would pin a coding specialist
        # for a plain conversational ask. Reset to conversation so the routed
        # model is a chat-capable one.
        if intent.primary is IntentType.CODING:
            intent = intent.model_copy(
                update={
                    "primary": IntentType.CONVERSATION,
                    "secondary": intent.primary,
                    "reasoning": [
                        *intent.reasoning,
                        "chat path with coding wording: required capabilities "
                        "reset to conversation",
                    ],
                }
            )
        decision = self._decision.decide(intent, complexity, privacy, hardware, registry_models)
        req_caps = list(decision.required_capabilities or [])
        if master_analysis is not None:
            if master_analysis.domain in ("mathematics", "math") or master_analysis.reasoning_complexity in (
                ReasoningComplexity.HARD, ReasoningComplexity.VERY_HARD
            ):
                if Capability.REASONING not in req_caps:
                    req_caps.append(Capability.REASONING)
                if Capability.MATH not in req_caps and master_analysis.domain in ("mathematics", "math"):
                    req_caps.append(Capability.MATH)
        decision = decision.model_copy(update={"required_capabilities": req_caps})
        # Chat never invokes the (workspace/artifact) planner: the execution
        # plan is a static empty shell so the response envelope stays complete
        # without running any planning machinery for conversation.
        plan = ExecutionPlan(
            intent=intent.primary,
            complexity=complexity.score,
            privacy=privacy.mode,
            steps=[],
        )

        # Read-only workspace context (RAG hits, vision descriptions, code
        # scan markers). Purely informational: a chat must still never create
        # folders/files/actions, and a failing workspace must never break chat.
        workspace_outcome = None
        chat_context = ""
        if (master_analysis is None or master_analysis.workspace_needed or master_analysis.files_needed) and ws is not None and ws.enabled:
            ctx_start = time.perf_counter()
            try:
                context_text, workspace_outcome = ws.prepare(prompt, files)
                if context_text:
                    chat_context = f"{context_text}\n\n"
            except Exception as exc:  # noqa: BLE001 - chat resilience is paramount
                log.warning("chat_workspace_prepare_failed", error=str(exc)[:200])
                workspace_outcome = None
            context_retrieval_ms = (time.perf_counter() - ctx_start) * 1000

        provider_ids = sorted({m.provider_id for m in registry_models} | set(self._providers.provider_ids()))
        health = {pid: self._providers.health(pid) for pid in provider_ids}
        available = self._provider_available_models()
        perf_stats = self._performance.stats() if self._performance else None

        mem = memory if memory is not None else self._memory

        is_chat_memory = False
        if master_analysis is not None and (
            master_analysis.memory_needed
            or master_analysis.intent == "chat_memory"
            or master_analysis.domain == "chat_memory"
            or any(
                w in prompt.lower()
                for w in (
                    "what is this chat",
                    "what is our chat",
                    "what is this conversation",
                    "summarize this chat",
                    "summarize this conversation",
                    "summarize our conversation",
                    "what have we done",
                    "what did we do",
                    "what have we accomplished",
                    "tell me about this chat",
                    "what is the context of this chat",
                )
            )
        ):
            is_chat_memory = True

        web_start = time.perf_counter()
        web_context = ""
        web_ms = 0.0
        if master_analysis is not None and master_analysis.web_needed:
            self._publish_timeline("web_search", f"Searching web for: {prompt[:60]}…")
            try:
                from synapse.pipeline.tools import ToolRegistry
                tool_reg = ToolRegistry()
                search_results = tool_reg._web_search(prompt, max_results=4)
                if search_results:
                    res_lines = []
                    for item in search_results:
                        title = item.get("title", "")
                        snippet = item.get("snippet", "")
                        url = item.get("url", "")
                        res_lines.append(f"- {title} ({url}):\n  {snippet}")
                    web_context = "Fresh Web Search Results:\n" + "\n".join(res_lines) + "\n\n"
            except Exception as exc:  # noqa: BLE001
                log.warning("web_search_failed", error=str(exc)[:200])
            web_ms = (time.perf_counter() - web_start) * 1000

        if is_chat_memory:
            from synapse.workspace.brief import build_specialist_prompt
            effective_prompt = build_specialist_prompt(
                prompt,
                project_id=project_id,
                project_name=project_name,
                project_path=project_path,
                file_operator=file_operator,
                memory=mem,
                conversation_id=conversation_id,
                chat_store=self._chat_store if hasattr(self, "_chat_store") else None,
                workspace_context=((web_context + chat_context).strip()) if (web_context or chat_context) else None,
                is_chat=True,
                include_file_rules=False,
            )
        else:
            effective_prompt = f"{web_context}{chat_context}{prompt}"

        chat_max_tokens = max_tokens if max_tokens is not None else 768

        worker_start = time.perf_counter()
        routing = self._route_chat(
            decision, hardware, registry_models, health, available,
            complexity=complexity.score, prompt=prompt, perf_stats=perf_stats,
        )
        if routing is None:
            final_response = _NO_ROUTE_GUIDANCE
        else:
            try:
                response = self._execute_with_lifecycle(
                    routing, effective_prompt, temperature=temperature, max_tokens=chat_max_tokens
                )
                final_response = response.content
            except ProviderUnavailable:
                final_response = _NO_ROUTE_GUIDANCE
            self._events.publish(
                Events.REQUEST_ROUTED, {"provider": routing.provider_id, "model": routing.model_id}
            )
        worker_inference_ms = (time.perf_counter() - worker_start) * 1000

        self._save_memory(prompt, final_response, memory=memory, conversation_id=conversation_id)

        elapsed_ms = (time.perf_counter() - start) * 1000
        timings = {
            "master_analysis_ms": round(master_analysis_ms, 2),
            "context_retrieval_ms": round(context_retrieval_ms, 2),
            "model_load_ms": 0.0,
            "worker_inference_ms": round(worker_inference_ms, 2),
            "tools_ms": 0.0,
            "web_ms": round(web_ms, 2),
            "verification_ms": 0.0,
            "total_ms": round(elapsed_ms, 2),
        }
        graph = ExecutionGraph()
        graph.total_latency_ms = round(elapsed_ms, 1)
        trace_routing = routing or RoutingDecision(reason="no chat route")
        trace = self._build_trace(
            intent, complexity, privacy, decision, trace_routing, hardware, elapsed_ms, response=None,
            master_analysis=master_analysis,
            timings=timings,
        )
        self._events.publish(
            Events.REQUEST_COMPLETED, {"latency_ms": round(elapsed_ms, 1), "tasks": 0, "chat": True}
        )
        log.info(
            "request_timing_breakdown",
            master_analysis_ms=round(master_analysis_ms, 2),
            context_retrieval_ms=round(context_retrieval_ms, 2),
            model_load_ms=0.0,
            worker_inference_ms=round(worker_inference_ms, 2),
            tools_ms=0.0,
            web_ms=round(web_ms, 2),
            verification_ms=0.0,
            total_ms=round(elapsed_ms, 2),
        )
        log.info(
            "chat_completed",
            latency_ms=round(elapsed_ms, 1),
            provider=trace_routing.provider_id,
            model=trace_routing.model_id,
        )
        return AgentResponse(
            response=final_response,
            intent=intent.primary,
            complexity=complexity.score,
            privacy=privacy.mode,
            provider=trace_routing.provider_id,
            model=trace_routing.model_id,
            execution_plan=plan,
            decision_trace=trace,
            latency_ms=round(elapsed_ms, 1),
            execution_graph=graph,
            workspace=workspace_outcome,
            actions=[],
        )

    def _route_chat(
        self,
        decision: Decision,
        hardware,
        registry_models,
        health,
        available,
        *,
        complexity: int,
        prompt: str,
        perf_stats,
    ) -> RoutingDecision | None:
        """Route a chat-only request: loaded-model reuse first, then Router."""
        if self._lifecycle is not None:
            reuse = self._lifecycle.find_reuse(
                decision, hardware, health, available, complexity=complexity
            )
            if reuse is not None:
                return reuse
        return self._router.route(
            decision,
            hardware,
            registry_models,
            health,
            available,
            complexity=complexity,
            prompt=prompt,
            performance=perf_stats,
        )

    def _plan_tasks(
        self, prompt, intent, complexity, privacy, decision, master_analysis: MasterAnalysis | None = None
    ) -> TaskDAG:
        if self._task_planner is not None:
            dag = self._task_planner.plan(prompt, intent, complexity, privacy, decision)
            if master_analysis is not None and master_analysis.artifact_required:
                file_tasks = [t for t in dag.tasks if t.file_output and t.kind != TaskKind.SYNTHESIS]
                if not file_tasks:
                    non_synth = [t for t in dag.tasks if t.kind != TaskKind.SYNTHESIS]
                    if non_synth:
                        first = non_synth[0]
                        first.file_output = True
                        if Capability.CODING not in first.required_capabilities:
                            first.required_capabilities.append(Capability.CODING)
            return dag

        file_output = False
        caps = list(decision.required_capabilities or [Capability.CHAT])
        if master_analysis is not None and master_analysis.artifact_required:
            file_output = True
            if Capability.CODING not in caps:
                caps.append(Capability.CODING)
        return TaskDAG(
            tasks=[
                Task(
                    id="t1",
                    kind=TaskKind.CODING if (master_analysis and (master_analysis.coding_needed or master_analysis.artifact_required)) else TaskKind.GENERAL,
                    description=prompt,
                    required_capabilities=caps,
                    preferred_capabilities=list(decision.preferred_capabilities),
                    file_output=file_output,
                )
            ]
        )

    def _execute_dag(
        self,
        dag: TaskDAG,
        prompt: str,
        decision: Decision,
        hardware,
        registry_models,
        health,
        available,
        complexity: int,
        perf_stats,
    ) -> tuple[ExecutionGraph, list[Task], RoutingDecision | None]:
        """Execute every task in topological order, routing each independently.

        Tasks are executed sequentially today; the DAG structure is
        parallel-ready (any task whose dependencies are done could run
        concurrently with its siblings).
        """
        return self._execute_dag_with_quality(
            dag, prompt, decision, hardware, registry_models, health, available, complexity, perf_stats,
            workspace_outcome=None, files=None
        )

    def _execute_dag_with_quality(
        self,
        dag: TaskDAG,
        prompt: str,
        decision: Decision,
        hardware,
        registry_models,
        health,
        available,
        complexity: int,
        perf_stats,
        workspace_outcome: WorkspaceOutcome | None = None,
        files: list[str] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        workspace=None,
        memory=None,
        conversation_id: str | None = None,
        file_operator=None,
        kind: RequestKind | None = None,
        project_id: str | None = None,
        project_name: str | None = None,
        project_path: str | None = None,
        chat_store=None,
        workspace_context: str | None = None,
    ) -> tuple[ExecutionGraph, list[Task], RoutingDecision | None]:
        """Execute DAG with Phase 4 quality checks (confidence, verification, escalation, self-correction)."""
        ws = workspace if workspace is not None else self._workspace
        mem = memory if memory is not None else self._memory
        graph = ExecutionGraph()
        executed: list[Task] = []
        primary: RoutingDecision | None = None
        order = 0

        for task in dag.topological_order():
            order += 1
            task.order = order
            if task.kind == TaskKind.SYNTHESIS:
                continue  # synthesized by the Result Synthesizer, not a model

            # Dependency gate
            if task.depends_on:
                incomplete_deps = [dep for dep in task.depends_on if not any(t.id == dep and t.status == TaskStatus.COMPLETED for t in executed)]
                if incomplete_deps:
                    task.status = TaskStatus.WAITING
                    self._events.publish(Events.TASK_WAITING, {"task_id": task.id, "depends_on": incomplete_deps})

            task_decision = self._decision_for_task(task, decision)

            # Publish capability selection
            if task.required_capabilities:
                self._events.publish(
                    Events.CAPABILITY_SELECTED,
                    {"task_id": task.id, "capability": ", ".join(c.value for c in task.required_capabilities)},
                )

            # Pass previous step context forward
            prior = self._prior_context(task, executed)

            # If targeting an existing file for edit/modification:
            target_edit_path = None
            existing_file_content = None
            if file_operator is not None:
                if task.file_hint and file_operator.exists(task.file_hint):
                    target_edit_path = task.file_hint
                elif not task.file_hint:
                    for candidate in ("main.py", "app.py", "src/main.py", "index.html", "script.py", "expense_tracker.py", "style.css", "script.js"):
                        if file_operator.exists(candidate) and any(w in task.description.lower() for w in ("fastapi", "code", "app", "python", "endpoint", "edit", "add to", "modify", "update", "button", "html", "style", "css", "script", "js", "website")):
                            target_edit_path = candidate
                            break

            if target_edit_path and file_operator is not None:
                existing_file_content = file_operator.read(target_edit_path)

            # If reviewing files, inject real files from disk that passed verification
            review_snippets = None
            if task.kind == TaskKind.REVIEW and file_operator is not None:
                try:
                    from synapse.workspace.review import validate_file
                    verified_ok_paths: set[str] = set()
                    failed_paths: set[str] = set()
                    for t in executed:
                        if hasattr(t, "_actions") and t._actions:
                            for a in t._actions:
                                if a.status == "ok" and getattr(a, "validated", True):
                                    verified_ok_paths.add(a.path)
                                elif a.status == "failed" or not getattr(a, "validated", True):
                                    failed_paths.add(a.path)

                    tree = file_operator.list_tree()
                    if tree:
                        snippets = []
                        for item in tree[:10]:
                            p = item.get("path")
                            if p in failed_paths and p not in verified_ok_paths:
                                continue
                            c = file_operator.read(p)
                            if c is not None:
                                v_res = validate_file(p, c)
                                if not v_res.ok:
                                    continue
                                max_c = 2500
                                snip = c if len(c) <= max_c else c[:max_c] + "\n... [truncated]"
                                snippets.append(f"### File on disk: {p}\n{snip}")
                        if snippets:
                            review_snippets = snippets
                except Exception:
                    pass

            from synapse.workspace.brief import build_specialist_prompt
            if "You are a specialist inside Synapse" in task.description or "You are inside Synapse" in task.description:
                effective_prompt = task.description
                if prior and prior not in effective_prompt:
                    effective_prompt = f"{effective_prompt}\n\n{prior}"
                if task.file_output and "OUTPUT FORMAT" not in effective_prompt and "Rules:" not in effective_prompt:
                    effective_prompt = f"{effective_prompt}\n\n{MANIFEST_INSTRUCTION}"
                if target_edit_path and existing_file_content and target_edit_path not in effective_prompt:
                    effective_prompt = (
                        f"{effective_prompt}\n\n"
                        f"Current content of the file you must modify ({target_edit_path}):\n"
                        f"```\n{existing_file_content}\n```"
                    )
                if review_snippets and "Workspace actual files on disk" not in effective_prompt:
                    effective_prompt = f"{effective_prompt}\n\nWorkspace actual files on disk for review:\n" + "\n\n".join(review_snippets)
            else:
                effective_prompt = build_specialist_prompt(
                    task.description,
                    project_id=project_id,
                    project_name=project_name,
                    project_path=project_path,
                    file_operator=file_operator,
                    memory=mem,
                    conversation_id=conversation_id,
                    chat_store=chat_store or (self._chat_store if hasattr(self, "_chat_store") else None),
                    target_edit_path=target_edit_path,
                    target_edit_content=existing_file_content,
                    prior_context=prior,
                    review_snippets=review_snippets,
                    workspace_context=workspace_context,
                )

            routing = self._route_task(
                task_decision, hardware, registry_models, health, available, complexity, effective_prompt, perf_stats,
                task=task,
            )
            node = GraphNode(
                task_id=task.id,
                kind=task.kind,
                description=task.description,
                status=task.status,
                depends_on=list(task.depends_on),
                preferred_model=task.preferred_model or task.model_hint,
                fallback_model=task.fallback_model,
                required_tools=list(task.required_tools),
                outputs=list(task.outputs),
                validation_state=task.validation_state,
                order=order,
                memory_context_used=mem is not None,
            )
            if routing is None or not routing.model_id:
                reason = (routing.reason if routing else "no model available")
                task.status = TaskStatus.FAILED
                task.error = reason
                node.status = TaskStatus.FAILED
                node.reason = reason
                self._events.publish(Events.TASK_FAILED, {"task_id": task.id, "reason": reason})
                graph.nodes.append(node)
                continue

            task.provider_id = routing.provider_id
            task.model_id = routing.model_id
            task.capability_score = routing.capability_score
            task.reason = routing.reason
            task.status = TaskStatus.ROUTED
            log.info(
                "model_selected",
                task_id=task.id,
                kind=task.kind.value,
                file_output=bool(task.file_output),
                provider=routing.provider_id,
                model=routing.model_id,
                reason=routing.reason,
                capability_score=routing.capability_score,
            )
            node.provider_id = routing.provider_id
            node.model_id = routing.model_id
            node.capability_score = routing.capability_score
            node.reason = routing.reason
            node.status = TaskStatus.ROUTED
            self._events.publish(
                Events.TASK_ROUTED,
                {"task_id": task.id, "provider": routing.provider_id, "model": routing.model_id},
            )

            try:
                task.status = TaskStatus.RUNNING
                node.status = TaskStatus.RUNNING
                self._events.publish(
                    Events.TASK_STARTED,
                    {"task_id": task.id, "description": task.description, "model": routing.model_id},
                )
                task_max_tokens = max_tokens
                if task_max_tokens is None:
                    if task.file_output or (kind and kind in (RequestKind.FILE_CREATION, RequestKind.PROJECT_GENERATION)):
                        task_max_tokens = 2048
                    elif target_edit_path or (kind and kind is RequestKind.FILE_MODIFICATION):
                        task_max_tokens = 1536
                    elif task.kind in (TaskKind.REVIEW, TaskKind.SYNTHESIS):
                        task_max_tokens = 768
                    else:
                        task_max_tokens = 1024

                task_start = time.perf_counter()
                response = self._execute_with_lifecycle(routing, effective_prompt, temperature=temperature, max_tokens=task_max_tokens)
                task.latency_ms = (time.perf_counter() - task_start) * 1000
                task.result = response.content
                task.status = TaskStatus.COMPLETED
                node.status = TaskStatus.COMPLETED
                node.latency_ms = round(task.latency_ms, 1)

                # Progressive file execution and quality verification
                if file_operator is not None and task.file_output and task.kind not in (TaskKind.SYNTHESIS, TaskKind.REVIEW):
                    # Vision Isolation: vision models never produce file ops
                    if Capability.VISION not in (task.required_capabilities or []):
                        hint_path = target_edit_path or task.file_hint or ""
                        ops = parse_file_manifest(task.result, hint=hint_path, strict=False)
                        if not ops:
                            # Auto re-prompt once with stronger instruction
                            target_path = hint_path or "main.py"
                            reprompt_msg = (
                                f"{effective_prompt}\n\n"
                                f"ACTION REQUIRED: You must emit the file changes as JSON now.\n"
                                f"End your response with ONE JSON object containing the complete updated content:\n"
                                f'{{"files": [{{"path": "{target_path}", "content": "<complete updated code>"}}]}}'
                            )
                            try:
                                retry_resp = self._execute_with_lifecycle(routing, reprompt_msg, temperature=0.1, max_tokens=task_max_tokens)
                                retry_ops = parse_file_manifest(retry_resp.content, hint=hint_path, strict=False)
                                if retry_ops:
                                    ops = retry_ops
                                    task.result = retry_resp.content
                            except Exception:
                                pass

                        if ops:
                            engine = ActionEngine(file_operator)
                            ops = engine.plan_project_ops(kind, ops)
                            task_actions, task_validations = engine.apply(ops, on_step=self._timeline_for_op)

                            # Post-Creation Verification & Auto-Fix Loop (Universal across .py, .json, .js, .css, .html, etc.)
                            failing_items: list[tuple[str, str]] = []
                            for v in task_validations:
                                if not v.ok:
                                    failing_items.append((v.path, v.error or "validation failed"))
                            for a in task_actions:
                                if a.status == "failed" and a.path and not any(p == a.path for p, _ in failing_items):
                                    failing_items.append((a.path, a.error or "action failed"))
                            for a in task_actions:
                                if a.path.endswith(".py") and a.status == "ok" and not any(p == a.path for p, _ in failing_items):
                                    py_content = file_operator.read(a.path)
                                    if py_content is not None:
                                        try:
                                            ast.parse(py_content)
                                        except SyntaxError as syn_exc:
                                            failing_items.append((a.path, f"python syntax error: line {syn_exc.lineno}: {syn_exc.msg}"))

                            if failing_items:
                                log.warning("post_creation_verification_failed_retrying", failing_count=len(failing_items), files=[p for p, _ in failing_items])
                                errors_desc = []
                                for file_path, err_msg in failing_items:
                                    is_binary = any(file_path.endswith(ext) for ext in (".pdf", ".docx", ".pptx", ".xlsx"))
                                    curr_content = file_operator.read(file_path) if not is_binary else None
                                    if curr_content:
                                        body_display = f"Current Content on Disk:\n```\n{curr_content[:1500]}\n```"
                                    else:
                                        file_size = file_operator.path_for(file_path).stat().st_size if file_operator.exists(file_path) else 0
                                        body_display = f"Current File Status: on disk ({file_size} bytes, empty or content-less)"

                                    errors_desc.append(
                                        f"File: {file_path}\n"
                                        f"Verification Error: {err_msg}\n"
                                        f"{body_display}"
                                    )

                                retry_prompt = (
                                    f"{effective_prompt}\n\n"
                                    f"AUTOMATIC POST-CREATION VERIFICATION FAILED:\n"
                                    f"The following file(s) generated failed syntax, structure, or content completeness verification checks:\n\n"
                                    + "\n\n".join(errors_desc) + "\n\n"
                                    f"REPAIR INSTRUCTIONS:\n"
                                    f"1. Fix the errors identified above. If the file was empty or contained little/no real text, generate substantial, rich, readable content (full paragraphs, detailed bullets, filled tables, full slide text).\n"
                                    f"2. Return the complete, fully working corrected file content (do not omit anything, do not emit skeleton-only or placeholder text).\n"
                                    f"3. Return the corrected files in JSON format:\n"
                                    f'{{\n  "files": [\n    {{"path": "<path>", "content": "<complete corrected content or markdown>"}}\n  ]\n}}'
                                )
                                try:
                                    fix_resp = self._execute_with_lifecycle(routing, retry_prompt, temperature=0.1, max_tokens=task_max_tokens)
                                    fix_ops = parse_file_manifest(fix_resp.content, hint=failing_items[0][0], strict=False)
                                    if fix_ops:
                                        task.result = fix_resp.content
                                        fix_ops = engine.plan_project_ops(kind, fix_ops)
                                        repaired_actions, repaired_validations = engine.apply(fix_ops, on_step=self._timeline_for_op)
                                        orig_actions = {a.path: a.action for a in task_actions}
                                        for ra in repaired_actions:
                                            if orig_actions.get(ra.path) == "created" and ra.action == "modified":
                                                ra.action = "created"
                                        repaired_paths = {a.path for a in repaired_actions}
                                        task_actions = [a for a in task_actions if a.path not in repaired_paths] + repaired_actions
                                        repaired_val_paths = {v.path for v in repaired_validations}
                                        task_validations = [v for v in task_validations if v.path not in repaired_val_paths] + repaired_validations
                                except Exception as exc:  # noqa: BLE001
                                    log.warning("auto_repair_failed", error=str(exc)[:200])

                            task._actions = task_actions
                            task._validations = task_validations
                        else:
                            fail_path = hint_path or "output.py"
                            task._actions = [FileAction(path=fail_path, action="write", status="failed", error="no file manifest found in model output")]
                            task._validations = []

                if task.kind == TaskKind.REVIEW:
                    self._events.publish(Events.RESULT_VALIDATED, {"task_id": task.id, "status": "verified"})
                    task.validation_state = "verified"
                    node.validation_state = "verified"
                elif "test" in task.description.lower():
                    self._events.publish(Events.TESTS_RUN, {"task_id": task.id, "suite": task.description})
                    task.validation_state = "tested"
                    node.validation_state = "tested"
                else:
                    task.validation_state = "completed"
                    node.validation_state = "completed"

                log.info(
                    "task_completed",
                    task_id=task.id,
                    model=routing.model_id,
                    chars=len(task.result or ""),
                    has_json="{" in (task.result or ""),
                    fences=(task.result or "").count("```") // 2,
                    latency_ms=round(task.latency_ms, 1),
                )
                self._events.publish(Events.TASK_COMPLETED, {"task_id": task.id, "latency_ms": round(task.latency_ms, 1)})

                # Phase 4: Quality checks on task result
                if self._confidence:
                    retrieved_chunks = [{"text": c.text, "file_name": c.file_name, "score": c.score} for c in workspace_outcome.retrieval] if workspace_outcome and workspace_outcome.retrieval else None
                    conf_score = self._confidence.score(
                        task.result,
                        task.description,
                        context=workspace_outcome.retrieval if workspace_outcome else None,
                        retrieved_chunks=retrieved_chunks,
                        model_id=routing.model_id,
                        capability_requirements=task.required_capabilities,
                    )
                    task.confidence_score = conf_score.score

                    # Verification
                    if self._verifier and self._confidence.should_verify(conf_score):
                        capability = task.required_capabilities[0].value if task.required_capabilities else "general"
                        ver_result = self._verifier.verify(
                            task.description,
                            task.result,
                            context=workspace_outcome.retrieval if workspace_outcome else None,
                            capability=capability,
                        )
                        if ver_result.status is not VerificationStatus.VERIFIED:
                            task.result = ver_result.verified_response
                            log.info("verification_corrected", task_id=task.id, status=ver_result.status.value)

                    # Hallucination detection + self-correction
                    # (skipped for file tasks: their output is a structured
                    # manifest whose paths are creation intent, not claims —
                    # the engine's post-write verification is the real check.)
                    if self._hallucination and not task.file_output:
                        existing_files = []
                        if file_operator is not None:
                            for entry in file_operator.list_tree():
                                existing_files.append(entry["path"])
                                existing_files.append(Path(entry["path"]).name)
                        elif ws is not None:
                            existing_files = [f.name for f in ws.files.list()]
                        flags = self._hallucination.scan(
                            task.result,
                            prompt=task.description,
                            context=workspace_outcome.retrieval if workspace_outcome else None,
                            existing_files=existing_files,
                        )
                        if flags and self._corrector:
                            task.result = self._corrector.correct(task.result, flags, prompt=task.description)
                            log.info("self_correction_applied", task_id=task.id, flags=len(flags))

                    # Escalation
                    if self._escalation and self._confidence.should_escalate(conf_score):
                        from synapse.contracts.escalation import EscalationDecision

                        target_model = None
                        if self._escalation_policy:
                            current_model = next(
                                (m for m in registry_models if m.id == routing.model_id),
                                None,
                            )
                            esc_decision = self._escalation_policy.decide(
                                current_model,
                                conf_score.score,
                                list(registry_models),
                                [c.value for c in (task.required_capabilities or [])],
                                hardware_constraints={"ram_gb": hardware.memory.total_gb},
                            )
                        else:
                            esc_decision = EscalationDecision(
                                should_escalate=False,
                                target_model=target_model,
                                reason="no escalation policy configured",
                                current_score=conf_score.score,
                                expected_improvement=0.0,
                            )
                        if esc_decision.should_escalate and esc_decision.target_model is not None:
                            try:
                                escalated_resp, new_conf = self._escalation.escalate(
                                    task.description,
                                    task.result,
                                    esc_decision,
                                    context=workspace_outcome.retrieval if workspace_outcome else None,
                                )
                                task.result = escalated_resp
                                task.confidence_score = new_conf
                                log.info("task_escalated", task_id=task.id, to_model=esc_decision.target_model.id)
                            except ProviderUnavailable:
                                log.warning("task_escalation_unavailable", task_id=task.id)

                executed.append(task)
                if primary is None:
                    primary = routing
            except ProviderUnavailable as exc:
                task.status = TaskStatus.FAILED
                task.error = str(exc)
                node.status = TaskStatus.FAILED
                node.reason = str(exc)[:200]
                self._events.publish(Events.TASK_FAILED, {"task_id": task.id, "reason": str(exc)[:200]})
                log.warning("task_failed", task_id=task.id, error=str(exc)[:200])
            graph.nodes.append(node)

        graph.edges = [GraphEdge(source=dep, target=t.id) for t in dag.tasks for dep in t.depends_on]
        graph.execution_order = [t.id for t in dag.topological_order()]

        synthesis_task = dag.get("t-synthesis")
        if synthesis_task is not None:
            order += 1
            graph.nodes.append(
                GraphNode(
                    task_id=synthesis_task.id,
                    kind=synthesis_task.kind,
                    description=synthesis_task.description,
                    status=TaskStatus.COMPLETED,
                    depends_on=list(synthesis_task.depends_on),
                    provider_id="synthesizer",
                    model_id="synthesizer",
                    reason=f"merged {len(executed)} task outputs",
                    order=order,
                )
            )
            graph.synthesized = True
        return graph, executed, primary

    def _prior_context(
        self,
        task: Task,
        executed: list[Task],
    ) -> str:
        """Context from previously executed models (Phase XVI).

        Dependency results are preferred; when a task declares no deps, the
        most recently executed task's result is passed instead, so consecutive
        parts of a multi-model chain always see the previous step's output.
        File-manifest results are reduced to their paths — raw manifest
        contents are never re-fed to another model (the filesystem, after
        apply, is the source of truth).
        """
        if not executed:
            return ""
        deps = [t for t in executed if t.id in task.depends_on]
        if not deps:
            deps = [executed[-1]]
        parts: list[str] = []
        for prior in deps[-2:]:
            if prior.file_output:
                paths = _manifest_paths(prior.result or "")
                blob = f"[files produced in this step: {', '.join(paths) or 'none'}]"
            else:
                blob = (prior.result or "")[:2000]
            parts.append(
                f"--- Result from step {prior.id} "
                f"(kind: {prior.kind.value}, model: {prior.model_id}) ---\n{blob}"
            )
        return "Context from the previous step(s):\n" + "\n\n".join(parts)

    def _decision_for_task(self, task: Task, base: Decision) -> Decision:
        return Decision(
            can_stay_local=base.can_stay_local,
            internet_required=base.internet_required,
            privacy=base.privacy,
            required_capabilities=list(task.required_capabilities or [Capability.CHAT]),
            preferred_capabilities=list(task.preferred_capabilities),
            preferred_kind=base.preferred_kind,
            workspace=base.workspace,
            reasoning=[f"task {task.id} ({task.kind.value}) routed independently"],
        )

    def _route_task(
        self,
        task_decision: Decision,
        hardware,
        registry_models,
        health,
        available,
        complexity: int,
        prompt: str,
        perf_stats,
        task: Task | None = None,
    ) -> RoutingDecision | None:
        """Route one task: planner hint first, then loaded-model reuse, then the router."""
        if task is not None:
            hinted = self._resolve_task_model(task, registry_models, health, available)
            if hinted is not None:
                return RoutingDecision(
                    provider_id=hinted.provider_id,
                    model_id=hinted.id,
                    kind=hinted.kind,
                    confidence=1.0,
                    reason=(
                        "strongest available reasoning model (review)"
                        if task.kind == TaskKind.REVIEW
                        else "planner model hint"
                    ),
                    capability_score=1.0,
                )
        if self._lifecycle is not None:
            reuse = self._lifecycle.find_reuse(
                task_decision, hardware, health, available, complexity=complexity
            )
            if reuse is not None:
                return reuse
        return self._router.route(
            task_decision,
            hardware,
            registry_models,
            health,
            available,
            complexity=complexity,
            prompt=prompt,
            performance=perf_stats,
        )

    def _retrieve_memory_context(self, task: Task, memory=None) -> str | None:
        """Pull relevant conversation/project history to enrich a task prompt."""
        mem = memory if memory is not None else self._memory
        try:
            hits = mem.search(MemoryScope.CONVERSATION, task.description, k=2)
            if not hits and mem.search is not None:
                hits = mem.search(MemoryScope.PROJECT, task.description, k=1)
            if hits:
                lines = [f"- {h.text[:400]}" for h in hits]
                return "Relevant context from workspace memory:\n" + "\n".join(lines)
        except Exception:  # noqa: BLE001 - memory must never break the pipeline
            log.debug("memory_retrieval_failed")
        return None

    def _synthesize(self, prompt: str, executed: list[Task], dag: TaskDAG) -> str:
        """Produce the final response from per-task outputs."""
        if not executed:
            return _NO_ROUTE_GUIDANCE
        if self._synthesizer is not None:
            result = self._synthesizer.synthesize(prompt, executed)
            self._events.publish(
                Events.RESPONSE_SYNTHESIZED,
                {"task_ids": result.task_ids, "models": result.models_used},
            )
            return result.response
        return executed[0].result or ""

    def _publish_timeline(self, kind: str, text: str) -> None:
        """Publish a live-timeline step so the UI can show the agent working."""
        try:
            self._events.publish(Events.TIMELINE, {"kind": kind, "text": text})
        except Exception:  # noqa: BLE001 - a timeline must never break the pipeline
            log.debug("timeline_publish_failed", kind=kind)

    def _timeline_for_op(self, step: dict) -> None:
        """Turn an engine operation step into a readable timeline line."""
        kind = step.get("kind") or step.get("action")
        path = step.get("path") or ""
        if kind == "list":
            self._publish_timeline("list", "Listing workspace")
        elif kind == "search":
            self._publish_timeline("search", f"Searching {path or 'workspace'}…")
        elif kind == "read":
            self._publish_timeline("read", f"Reading {path or 'workspace'}…")
        elif kind == "create_folder":
            self._publish_timeline("create_folder", f"Creating folder {path}")
        elif kind == "rename":
            self._publish_timeline("rename", f"Renaming {path} → {step.get('to', '')}")
        elif kind == "delete":
            self._publish_timeline("delete", f"Deleting {path}")
        elif kind == "write":
            self._publish_timeline("write", f"Writing {path}")
        elif kind == "edit":
            self._publish_timeline("edit", f"Updating {path}")

    # -- Phase 6/7: file outputs, review, and action logging --------------------

    def _apply_file_outputs(
        self,
        dag: TaskDAG,
        executed: list[Task],
        file_operator,
        kind: RequestKind | None = None,
    ) -> tuple[list[FileAction], list[ValidationResult]]:
        """Apply each executed task's file manifest to the workspace via the
        Action Engine.

        Phase X — universal workspace tool: ANY task (analysis, research,
        chat, docs, coding) that ends with a manifest or fenced code blocks
        writes those files to the project folder, exactly like the coding
        agents. ``file_output`` tasks still additionally require a manifest
        (their missing-manifest case is reported as a failed action). Returns
        ``(actions, validations)``. A violation never raises — it becomes a
        failed FileAction so the rest of the request proceeds.
        """
        engine = ActionEngine(file_operator)
        actions: list[FileAction] = []
        validations: list[ValidationResult] = []
        for task in executed:
            if not task.result:
                continue
            if task.kind in (TaskKind.SYNTHESIS, TaskKind.REVIEW):
                continue  # review commentary must never become files
            # Vision isolation: vision models never produce file ops
            if Capability.VISION in (task.required_capabilities or []):
                continue
            if hasattr(task, "_actions") and task._actions:
                actions.extend(task._actions)
                if hasattr(task, "_validations") and task._validations:
                    validations.extend(task._validations)
                continue

            # Planned file tasks get the tolerant prose/fenced-code converter;
            # universal-tool tasks (file intent not planned) must produce a
            # real manifest — ordinary prose never becomes files on disk.
            ops = parse_file_manifest(
                task.result, hint=task.file_hint, strict=not task.file_output
            )
            log.info(
                "actions_generated",
                task_id=task.id,
                hint=task.file_hint,
                ops=[
                    {"action": o.get("action"), "path": o.get("path")}
                    for o in ops
                ],
            )
            if not ops:
                if task.file_output:
                    actions.append(
                        FileAction(
                            path=task.file_hint or "output",
                            action="write",
                            status="failed",
                            error="no file manifest found in model output",
                        )
                    )
                continue
            ops = engine.plan_project_ops(kind, ops)
            task_actions, task_validations = engine.apply(ops, on_step=self._timeline_for_op)
            log.info(
                "actions_executed",
                task_id=task.id,
                created=[
                    a.path for a in task_actions
                    if a.action in ("created", "created_folder") and a.status == "ok"
                ],
                modified=[
                    a.path for a in task_actions
                    if a.action == "modified" and a.status == "ok"
                ],
                renamed=[
                    a.path for a in task_actions
                    if a.action == "renamed" and a.status == "ok"
                ],
                failed=[
                    {"path": a.path, "error": a.error}
                    for a in task_actions if a.status == "failed"
                ],
            )
            actions.extend(task_actions)
            validations.extend(task_validations)
        log.info(
            "request_filesystem_outcome",
            created=[
                a.path for a in actions if a.action in ("created", "created_folder") and a.status == "ok"
            ],
            modified=[a.path for a in actions if a.action == "modified" and a.status == "ok"],
            failures=[{"path": a.path, "error": a.error} for a in actions if a.status == "failed"],
        )
        return actions, validations

    def _verify_and_remember(
        self,
        actions: list[FileAction],
        file_operator,
        *,
        workspace=None,
        memory=None,
        conversation_id: str | None = None,
        project_index=None,
        change_panel=None,
        diagnostics=None,
    ) -> None:
        """Phase XIII — the write→verify→index→remember close of every request.

        After files are applied: confirm each changed file exists on the
        workspace disk, syntax-check what we can, refresh the project index for
        the written files (so grounding/search reflects the new state), record
        the change in the change panel, run post-write diagnostics, and record
        the write in project memory (so a later request can recall what the
        agent produced without re-reading the folder). Never raises.
        """
        changed = [
            a for a in actions
            if a.action in ("created", "modified", "renamed") and a.status == "ok"
        ]
        if not changed:
            return

        # Phase XV — file change panel: created / modified / renamed / deleted.
        if change_panel is not None:
            try:
                for a in actions:
                    if a.status != "ok":
                        continue
                    if a.action in ("created", "created_folder"):
                        change_panel.record_created(a.path, size=a.bytes or None)
                    elif a.action == "modified":
                        change_panel.record_modified(a.path)
                    elif a.action == "renamed":
                        change_panel.record_renamed("(moved)", a.path)
                    elif a.action == "deleted":
                        change_panel.record_deleted(a.path)
            except Exception:  # noqa: BLE001 - change tracking never breaks the request
                log.debug("change_panel_record_failed", exc_info=True)

        rels = [a.path for a in changed if a.action in ("created", "modified")]
        verified = 0
        for rel in rels:
            try:
                if file_operator.exists(rel):
                    verified += 1
                    content = file_operator.read(rel)
                    if content is not None and not self._simple_syntax_check(rel, content):
                        self._publish_timeline("index", f"{rel} failed syntax check")
                    else:
                        self._publish_timeline("verified", f"Verified {rel}")
            except Exception:  # noqa: BLE001 - verification never breaks the request
                log.debug("work_verify_failed", path=rel, exc_info=True)
        if verified and workspace is not None:
            self._index_changed_files(workspace, file_operator, rels)

        # Phase XV — keep the project code index in sync with what we wrote.
        if project_index is not None:
            try:
                for rel in rels + [a.path for a in changed if a.action == "renamed"]:
                    entry = project_index.update_file(rel)
                    if entry is not None:
                        self._publish_timeline("index", f"Indexed {rel}")
            except Exception:  # noqa: BLE001 - index refresh is best-effort
                log.debug("project_index_update_failed", exc_info=True)

        # Phase XV — post-write diagnostics: surface syntax problems early.
        if diagnostics is not None:
            try:
                for rel in rels:
                    diags = diagnostics.analyze_file(rel)
                    for d in diags:
                        if d.severity == "error":
                            self._publish_timeline(
                                "diagnostic",
                                f"{rel}:{d.line or 0} {d.message}",
                            )
            except Exception:  # noqa: BLE001 - diagnostics never break the request
                log.debug("diagnostics_after_write_failed", exc_info=True)

        if memory is not None:
            try:
                summary = ", ".join(f"{a.action} {a.path}" for a in changed)
                memory.save(
                    MemoryScope.PROJECT,
                    f"The agent wrote to the workspace: {summary}.",
                    source="workspace",
                    metadata={"source": "phase_xiii"},
                )
                self._events.publish(
                    Events.MEMORY_WRITTEN,
                    {"entries": 1, "scope": MemoryScope.PROJECT.value},
                )
            except Exception:  # noqa: BLE001
                log.debug("work_remember_failed", exc_info=True)

    def _index_changed_files(self, workspace, file_operator, rels: list[str]) -> None:
        """Import written files into the project index (sha256 dedupe) and
        kick off embedding jobs so the refreshed state is searchable."""
        for rel in rels:
            try:
                data = file_operator.read(rel)
            except Exception:  # noqa: BLE001
                continue
            if not data:
                continue
            try:
                info = workspace.upload(rel, data.encode("utf-8"))
                workspace.index(info.id)
                self._publish_timeline("index", f"Indexed {rel}")
            except Exception:  # noqa: BLE001 - index refresh is best-effort
                log.debug("work_index_failed", path=rel, exc_info=True)

    @staticmethod
    def _simple_syntax_check(rel: str, content: str) -> bool:
        """Cheap local syntax validation for known plain-text formats. Unknown
        languages are reported OK (no interpreter available at this layer)."""
        if rel.endswith(".py"):
            try:
                ast.parse(content)
                return True
            except SyntaxError:
                return False
        if rel.endswith((".json", ".jsonc")):
            try:
                json.loads(content)
                return True
            except ValueError:
                return False
        return True

    def _compose_response(
        self,
        prose: str,
        actions: list[FileAction],
        validations: list[ValidationResult],
        dag: TaskDAG,
        *,
        summary_only: bool = False,
    ) -> str:
        """Merge the synthesized prose with the file-action summary block.

        ``summary_only`` (Phase 7) returns just the summary for file kinds —
        generated code is written to the workspace, never dumped into chat.
        When nothing could be parsed for a file kind, a short notice is shown
        instead of the raw model text.
        """
        prose = self._scrub_unverified_claims(prose, actions)
        if not actions:
            if summary_only:
                return (
                    "No files were written for this request — the model output "
                    "could not be converted into file actions. Check the action "
                    "log for details."
                )
            if prose:
                return prose
            return (
                "The model produced no answer for this request — its output "
                "contained no statements that could be verified. Check the "
                "action log for details."
            )
        review_text = ""
        review_task = dag.get("t-review")
        if review_task is not None and review_task.result:
            review_text = review_task.result.strip()
        ok_paths = {a.path for a in actions if a.status == "ok"}
        if review_text:
            from synapse.workspace.review import clean_review_text
            review_text = clean_review_text(review_text, ok_paths)
        block = ActionEngine.summarize(actions, validations, review_text=review_text)
        if summary_only:
            return block
        if prose and prose != _NO_ROUTE_GUIDANCE:
            return f"{block}\n\n{prose}"
        return block

    def _scrub_unverified_claims(self, prose: str, actions: list[FileAction]) -> str:
        """Remove model-prose statements that claim filesystem work the
        verified action log does not back.

        Phase XVI — the final response must reflect actual execution. Lines
        asserting a file was created/written/modified/deleted/renamed without
        a matching ok action, and raw manifest echoes, are dropped.
        """
        if not prose:
            return prose
        verified = {a.path for a in actions if a.status == "ok"}
        kept: list[str] = []
        for line in prose.splitlines():
            if self._line_is_unverified_claim(line, verified):
                continue
            kept.append(line)
        return "\n".join(kept)

    @staticmethod
    def _line_is_unverified_claim(line: str, verified: set[str]) -> bool:
        for m in _FILE_CLAIM_RE.finditer(line):
            if m.group(1) not in verified:
                return True
        return bool(_MANIFEST_FRAG_RE.search(line)) and any(
            key in line for key in ("content", "folders", '"files"')
        )

    def _record_action_log(
        self,
        action_log,
        prompt: str,
        dag: TaskDAG,
        actions: list[FileAction],
        validations: list[ValidationResult],
        project_id: str | None,
        elapsed_ms: float,
    ) -> None:
        """Write one Phase 6 audit entry for this request."""
        from synapse.domain import TaskStatus

        models_used = sorted({t.model_id for t in dag.tasks if t.model_id})
        tools_used = sorted({f"filesystem.{a.action}" for a in actions})
        entry = {
            "project_id": project_id or "",
            "prompt": prompt[:500],
            "task_plan": [
                {"id": t.id, "kind": t.kind.value, "file_output": bool(t.file_output), "status": t.status.value}
                for t in dag.tasks
            ],
            "models_used": models_used,
            "tools_used": tools_used,
            "tool_actions": [a.model_dump() for a in actions],
            "files_created": [a.path for a in actions if a.action == "created" and a.status == "ok"],
            "files_modified": [a.path for a in actions if a.action == "modified" and a.status == "ok"],
            "files_renamed": [a.path for a in actions if a.action == "renamed" and a.status == "ok"],
            "files_deleted": [a.path for a in actions if a.action == "deleted" and a.status == "ok"],
            "files_failed": [a.path for a in actions if a.status == "failed"],
            "validation": [
                {"path": v.path, "ok": v.ok, "error": v.error} for v in validations
            ],
            "execution_time_ms": round(elapsed_ms, 1),
            "failures": [
                {"task_id": t.id, "error": t.error} for t in dag.tasks if t.status == TaskStatus.FAILED
            ],
        }
        action_log.record(entry)

    # -- model selection hints (planner → model) -----------------------------

    def _resolve_task_model(self, task: Task, registry_models, health, available):
        """Planner-level model selection: REVIEW → strongest reasoning model;
        otherwise honor ``task.preferred_model`` or ``task.model_hint`` when
        available and safe. Returns None to let the router decide."""
        if task.kind == TaskKind.REVIEW:
            return self._strongest_reasoning_model(registry_models, health, available)
        hint = task.preferred_model or task.model_hint
        if not hint:
            return None
        for m in registry_models:
            avail_set = available.get(m.provider_id) or set()
            if m.id == hint and health.get(m.provider_id) and (m.id in avail_set):
                # Deterministic hardware safety check
                try:
                    hw = self._hardware.scan()
                    avail = float(hw.memory.available_gb or 0.0)
                    vram = float(hw.gpu.vram_gb) if hw.gpu and hw.gpu.vram_gb else 0.0
                    # If model requires more RAM than available (and cannot fit in VRAM), reject it
                    if m.required_ram_gb > avail and (vram <= 0 or m.required_ram_gb > vram):
                        log.warning(
                            "model_hint_rejected_hardware_safety",
                            model_id=m.id,
                            required_ram=m.required_ram_gb,
                            available_ram=avail,
                            vram=vram,
                        )
                        return None
                except Exception:
                    pass
                return m
        return None

    @staticmethod
    def _strongest_reasoning_model(registry_models, health, available):
        """Highest reasoning-capability model that is healthy and available."""
        best = None
        best_score = -1.0
        for m in registry_models:
            score = m.capabilities.score_for(Capability.REASONING)
            if score <= 0:
                continue
            if not health.get(m.provider_id):
                continue
            avail_set = available.get(m.provider_id) or set()
            if m.id not in avail_set:
                continue
            if score > best_score or (score == best_score and best is not None and m.id < best.id):
                best = m
                best_score = score
        return best

    def _save_memory(self, prompt: str, response: str, memory=None, conversation_id: str | None = None) -> None:
        mem = memory if memory is not None else self._memory
        if mem is None:
            return
        try:
            mem.save(MemoryScope.CONVERSATION, prompt, source="user", conversation=conversation_id)
            mem.save(MemoryScope.CONVERSATION, response, source="assistant", conversation=conversation_id)
            self._events.publish(Events.MEMORY_WRITTEN, {"entries": 2, "scope": MemoryScope.CONVERSATION.value})
        except Exception:  # noqa: BLE001
            log.debug("memory_save_failed")

    # -- internals ----------------------------------------------------------

    def _provider_availability(self) -> dict[str, bool]:
        kinds = {pid: self._providers.get(pid).kind.value for pid in self._providers.provider_ids()}
        return {"local": "local" in kinds.values(), "cloud": "cloud" in kinds.values()}

    def _provider_available_models(self) -> dict[str, set[str] | None]:
        """provider_id -> set of model ids the provider reports as installed.

        Returns ``None`` for a provider whose ``list_models()`` raised,
        distinguishing transient failures from genuinely empty installs.
        """
        available: dict[str, set[str] | None] = {}
        for provider in self._providers.all():
            try:
                ids = {d.id for d in provider.list_models()}
            except Exception:  # noqa: BLE001 - a failing provider never blocks routing
                ids = None
            available[provider.provider_id] = ids
        return available

    def _execute_with_lifecycle(
        self, routing: RoutingDecision, prompt: str, *, temperature: float = 0.7, max_tokens: int | None = None,
    ) -> ChatResponse:
        """Execute, bracketing the request with lifecycle load/active/idle hooks."""
        if self._lifecycle is not None:
            self._lifecycle.note_request_started(routing.provider_id, routing.model_id)
        try:
            return self._execute_and_record(routing, prompt, temperature=temperature, max_tokens=max_tokens)
        finally:
            if self._lifecycle is not None:
                self._lifecycle.note_request_completed(routing.provider_id, routing.model_id)

    def _execute_and_record(self, routing: RoutingDecision, prompt: str, *, temperature: float = 0.7, max_tokens: int | None = None) -> ChatResponse:
        """Execute and feed the outcome back into the performance store."""
        self._publish_timeline("generate", "Generating…")
        try:
            response = self._executor.execute(routing, prompt, temperature=temperature, max_tokens=max_tokens)
        except ProviderUnavailable as exc:
            log.warning("execution_failed", error=str(exc)[:200])
            if self._performance:
                self._performance.record(
                    routing.provider_id,
                    routing.model_id,
                    latency_s=0.0,
                    success=False,
                )
            raise
        if self._performance:
            metrics = response.metrics
            self._performance.record(
                routing.provider_id,
                routing.model_id,
                latency_s=metrics.total_latency_s if metrics.total_latency_s is not None else 0.0,
                tokens_per_s=metrics.tokens_per_second,
                success=True,
                interrupted=metrics.interrupted,
            )
        return response

    @staticmethod
    def _build_trace(
        intent,
        complexity,
        privacy,
        decision: Decision,
        routing: RoutingDecision,
        hardware,
        elapsed_ms: float,
        response: ChatResponse | None,
        master_analysis: MasterAnalysis | None = None,
        timings: dict[str, float] | None = None,
    ) -> DecisionTrace:
        return DecisionTrace(
            intent=intent.primary,
            intent_confidence=intent.confidence,
            complexity=complexity.score,
            privacy=privacy.mode,
            internet_required=privacy.internet_required,
            hardware={
                "cpu": hardware.cpu.model,
                "ram_gb": hardware.memory.total_gb,
                "gpu": hardware.gpu.name if hardware.gpu else None,
                "can_run_local_llm": hardware.recommendations.can_run_local_llm,
            },
            provider=routing.provider_id,
            model=routing.model_id,
            reason=routing.reason,
            execution_time_ms=round(elapsed_ms, 1),
            candidate_count=len(routing.candidates),
            capability_score=routing.capability_score,
            excluded_models=routing.excluded_models,
            estimated_latency_s=routing.estimated_latency_s,
            expected_output_tokens=routing.expected_output_tokens,
            historical_performance=routing.historical_performance,
            intent_reasoning=intent.reasoning,
            complexity_reasoning=complexity.reasoning,
            privacy_reasoning=privacy.reasoning,
            hardware_reasoning=decision.reasoning,
            master_analysis=master_analysis.model_dump() if master_analysis else None,
            workspace_needed=master_analysis.workspace_needed if master_analysis else None,
            reasoning_complexity=master_analysis.reasoning_complexity.value if master_analysis else None,
            domain=master_analysis.domain if master_analysis else None,
            timings=timings or {},
        )
