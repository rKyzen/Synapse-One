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
import time

import structlog

from synapse.actions import (
    ActionEngine,
    RequestKind,
    classify_request,
    extract_workspace_ops,
    requires_workspace_access,
)
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
    Decision,
    DecisionTrace,
    ExecutionGraph,
    ExecutionPlan,
    GraphEdge,
    GraphNode,
    MemoryScope,
    PrivacyMode,
    ProviderKind,
    RoutingDecision,
    Task,
    TaskKind,
    TaskStatus,
    WorkspaceOutcome,
)
from synapse.domain.fileops import FileAction, ValidationResult
from synapse.domain.tasks import TaskDAG
from synapse.events import EventBus, Events
from synapse.execution import Executor, ProviderUnavailable
from synapse.providers.manager import ProviderManager
from synapse.router import Router as ConcreteRouter
from synapse.workspace.brief import build_workspace_brief
from synapse.workspace.manifest import (
    MANIFEST_INSTRUCTION,
    WORKSPACE_TOOL_INSTRUCTION,
    parse_file_manifest,
)

log = structlog.get_logger("synapse.master")

#: message returned when every task failed to route.
_NO_ROUTE_GUIDANCE = (
    "No model could satisfy the current constraints. "
    "Install or enable a matching provider/model in configuration, "
    "then retry."
)


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
    ) -> AgentResponse:
        start = time.perf_counter()
        self._events.publish(Events.REQUEST_RECEIVED, {"prompt_length": len(prompt)})
        # Phase 7 — every request is classified into one of the seven kinds so
        # the execution path can be chosen before any model is consulted.
        kind = classify_request(prompt)
        log.info(
            "request_classified",
            kind=kind.value,
            prompt=prompt[:120],
        )
        # Resolve per-project scoped workspace/memory (Phase 5), else fall back
        # to the globally bound defaults.
        ws = workspace if workspace is not None else self._workspace
        mem = memory if memory is not None else self._memory

        hardware = self._hardware.scan()
        self._registry.sync(self._providers.all())
        registry_models = self._registry.all()

        intent = self._intent.analyze(prompt)
        complexity = self._complexity.analyze(prompt)
        privacy = self._privacy.analyze(
            prompt,
            complexity,
            context={
                "user_preference": self._config.get("privacy.user_preference", "balanced"),
                "provider_availability": self._provider_availability(),
            },
        )
        self._events.publish(Events.REQUEST_ANALYZED, {"intent": intent.primary.value, "complexity": complexity.score})

        # Phase 4 — attach workspace files (vision/RAG/code scan) and build context.
        workspace_outcome: WorkspaceOutcome | None = None
        if ws is not None and ws.enabled:
            if files:
                self._events.publish(Events.FILE_ATTACHED, {"file_ids": files})
            context_text, workspace_outcome = ws.prepare(prompt, files)

        decision = self._decision.decide(intent, complexity, privacy, hardware, registry_models)
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

        # Phase X — the Action Engine feeds EVERY agent real filesystem
        # context (the backend reads; the model edits). Workspace access is
        # universal: any request kind may read and write inside the project
        # folder, exactly like the coding agents. The context is appended to
        # each task's description AFTER planning so it never interferes with
        # prompt decomposition.
        #
        # Phase XIII — workspace-first: a compact awareness brief (project
        # identity, path, tools, defaults, file tree, recently modified files,
        # current chat summary) is injected into EVERY task description —
        # including the reviewer and the final synthesis — so no model call
        # ever forgets it is operating inside the user's project. Full file
        # excerpts (``build_context``) are reserved for requests that actually
        # touch the workspace (the deterministic workspace-access gate);
        # pure-chat requests still get the brief, but not the whole tree read.
        brief = None
        context = None
        if file_operator is not None:
            self._publish_timeline("read_workspace", "Reading workspace…")
            brief = build_workspace_brief(
                project_id=project_id,
                project_name=project_name,
                project_path=project_path,
                file_operator=file_operator,
                memory=mem,
                conversation_id=conversation_id,
            )
            if requires_workspace_access(prompt):
                context = ActionEngine(file_operator).build_context(
                    on_file=lambda p: self._publish_timeline("read_file", f"Reading {p}")
                )

        dag = self._plan_tasks(prompt, intent, complexity, privacy, decision)
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
        if brief:
            for task in dag.tasks:
                if context and task.kind not in (TaskKind.SYNTHESIS, TaskKind.REVIEW):
                    task.description = f"{brief}\n\n{context}\n\n{task.description}"
                else:
                    task.description = f"{brief}\n\n{task.description}"
        elif context:
            prefix = f"{context}\n\n"
            for task in dag.tasks:
                if task.kind in (TaskKind.SYNTHESIS, TaskKind.REVIEW):
                    continue
                task.description = f"{prefix}{task.description}"
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
        if backend_only:
            lines, actions = ActionEngine(file_operator).run_workspace_ops(
                backend_ops, on_step=self._timeline_for_op
            )
            final_response = "\n".join(lines)
            graph, executed, primary_routing = ExecutionGraph(), [], None
        else:
            # Execute DAG with Phase 4 quality checks
            graph, executed, primary_routing = self._execute_dag_with_quality(
                dag, prompt, decision, hardware, registry_models, health, available, complexity.score, perf_stats,
                workspace_outcome, files, temperature=temperature, max_tokens=max_tokens,
                workspace=ws, memory=mem, conversation_id=conversation_id,
            )

            # Phase 6/7 — apply generated files to the project workspace via
            # the Action Engine and validate them. ``actions`` drive the
            # summary block and the action log.
            if file_operator is not None:
                actions, validations = self._apply_file_outputs(dag, executed, file_operator, kind=kind)
                # Phase XIII — write → verify → index → remember: confirm the
                # files landed, syntax-check what we can, refresh the project
                # index and record the write at project scope.
                self._verify_and_remember(
                    actions, file_operator,
                    workspace=ws, memory=mem, conversation_id=conversation_id,
                )

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
            if self._hallucination:
                # Gather project context for hallucination detection
                existing_files = [f.name for f in ws.files.list()] if ws else []
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
        routing = primary_routing or RoutingDecision(reason="no task routed")
        if primary_routing is None and not backend_only:
            final_response = _NO_ROUTE_GUIDANCE
        self._events.publish(Events.REQUEST_ROUTED, {"provider": routing.provider_id, "model": routing.model_id})
        graph.total_latency_ms = round(elapsed_ms, 1)
        trace = self._build_trace(
            intent, complexity, privacy, decision, routing, hardware, elapsed_ms, response=None
        )
        self._events.publish(Events.REQUEST_COMPLETED, {"latency_ms": round(elapsed_ms, 1), "tasks": len(dag.tasks)})
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

    def _plan_tasks(self, prompt, intent, complexity, privacy, decision) -> TaskDAG:
        if self._task_planner is not None:
            return self._task_planner.plan(prompt, intent, complexity, privacy, decision)
        return TaskDAG(
            tasks=[
                Task(
                    id="t1",
                    kind=TaskKind.GENERAL,
                    description=prompt,
                    required_capabilities=list(decision.required_capabilities or [Capability.CHAT]),
                    preferred_capabilities=list(decision.preferred_capabilities),
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

            task_decision = self._decision_for_task(task, decision)

            # Phase 4: Build smart context using ContextBuilder
            effective_prompt = task.description
            if self._context_builder:
                memory_results = None
                if mem:
                    hits = mem.search(MemoryScope.CONVERSATION, task.description, k=2, conversation=conversation_id)
                    if hits:
                        memory_results = [{"text": h.text, "scope": h.scope.value, "source": h.source} for h in hits]

                retrieval_chunks = None
                if workspace_outcome and workspace_outcome.retrieval:
                    retrieval_chunks = [{"text": c.text, "file_name": c.file_name, "score": c.score} for c in workspace_outcome.retrieval]

                conversation_history = None
                if mem:
                    recent = mem.recent(MemoryScope.CONVERSATION, 5, conversation=conversation_id)
                    conversation_history = [{"role": h.source, "content": h.text} for h in recent]

                context_bundle = self._context_builder.build(
                    task.description,
                    task_capabilities=[c.value for c in task.required_capabilities] if task.required_capabilities else [],
                    memory_results=memory_results,
                    retrieval_chunks=retrieval_chunks,
                    conversation_history=conversation_history,
                    workspace_outcome={"retrieval": workspace_outcome.retrieval, "vision_descriptions": workspace_outcome.vision_descriptions} if workspace_outcome else None,
                )
                effective_prompt = context_bundle.user_prompt

            # Phase 7 fix — a file task model is told exactly what to return.
            # Without this real LLMs answer in prose/fenced code instead of a
            # manifest, so the backend would have nothing structured to apply.
            if task.file_output:
                effective_prompt = f"{effective_prompt}\n\n{MANIFEST_INSTRUCTION}"
            elif task.kind not in (TaskKind.SYNTHESIS, TaskKind.REVIEW):
                # Phase X — universal workspace tool: every other task may
                # also produce files inside the project folder (soft rule:
                # plain prose answers stay perfectly valid).
                effective_prompt = f"{effective_prompt}\n\n{WORKSPACE_TOOL_INSTRUCTION}"

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
                task_start = time.perf_counter()
                response = self._execute_with_lifecycle(routing, effective_prompt, temperature=temperature, max_tokens=max_tokens)
                task.latency_ms = (time.perf_counter() - task_start) * 1000
                task.result = response.content
                task.status = TaskStatus.COMPLETED
                node.status = TaskStatus.COMPLETED
                node.latency_ms = round(task.latency_ms, 1)
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
                        existing_files = [f.name for f in ws.files.list()] if ws else []
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
    ) -> None:
        """Phase XIII — the write→verify→index→remember close of every request.

        After files are applied: confirm each changed file exists on the
        workspace disk, syntax-check what we can, refresh the project index for
        the written files (so grounding/search reflects the new state), and
        record the write in project memory (so a later request can recall what
        the agent produced without re-reading the folder). Never raises.
        """
        changed = [
            a for a in actions
            if a.action in ("created", "modified", "renamed") and a.status == "ok"
        ]
        if not changed:
            return

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
        if not actions:
            if summary_only:
                return (
                    "No files were written for this request — the model output "
                    "could not be converted into file actions. Check the action "
                    "log for details."
                )
            return prose or _NO_ROUTE_GUIDANCE
        review_text = ""
        review_task = dag.get("t-review")
        if review_task is not None and review_task.result:
            review_text = review_task.result.strip()
        block = ActionEngine.summarize(actions, validations, review_text=review_text)
        if summary_only:
            return block
        if prose and prose != _NO_ROUTE_GUIDANCE:
            return f"{block}\n\n{prose}"
        return block

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
        otherwise honor ``task.model_hint`` when available. Returns None to
        let the router decide."""
        if task.kind == TaskKind.REVIEW:
            return self._strongest_reasoning_model(registry_models, health, available)
        hint = task.model_hint
        if not hint:
            return None
        for m in registry_models:
            if m.id == hint and health.get(m.provider_id) and (m.id in available.get(m.provider_id, set())):
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
            if m.id not in available.get(m.provider_id, set()):
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

    def _provider_available_models(self) -> dict[str, set[str]]:
        """provider_id -> set of model ids the provider reports as installed."""
        available: dict[str, set[str]] = {}
        for provider in self._providers.all():
            try:
                ids = {d.id for d in provider.list_models()}
            except Exception:  # noqa: BLE001 - a failing provider never blocks routing
                ids = set()
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
        )
