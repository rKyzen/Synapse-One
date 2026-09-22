"""AI Master Orchestrator — a model-backed Task Planner (Task Divider).

Replaces (behind the same :class:`TaskPlanner` contract) the deterministic
decomposition with a small LLM acting as the Master Agent: it reads the user
prompt, understands the intent, and emits a structured
:class:`TaskDecompositionPlan` (JSON) before any deterministic
``Router.route()`` call happens.

Flow per request:

1. Resolve the hardware tier (``TierResolver``) -> candidate model ids.
2. Pick the first candidate that is registered AND its provider is healthy —
   the "Master AI" role is thus assigned dynamically per machine, so
   commercial distribution never risks an OOM crash.
3. Call the Master AI through the provider abstraction with the JSON schema
   of ``TaskDecompositionPlan`` attached (``ChatRequest.format``) plus a
   system prompt that forbids solving the prompt.
4. Parse + validate with Pydantic; repair obvious fence/extra-noise damage;
   retry once on hallucinated/invalid/timed-out output.
5. Any failure -> graceful fallback (single pass-through task with the raw
   prompt; when an injected fallback planner exists, it is used instead so
   behavior degrades to the current deterministic planner exactly).

No vendor code lives here: everything goes through ProviderManager and the
ModelProvider interface.
"""

from __future__ import annotations

import json
import re
import structlog
from typing import Any

from pydantic import ValidationError

from synapse.config import ConfigProvider
from synapse.contracts import HardwareProvider, ModelRegistry, TaskPlanner
from synapse.domain import (
    Capability,
    ChatMessage,
    ChatRequest,
    ComplexityResult,
    Decision,
    IntentResult,
    PrivacyResult,
)
from synapse.domain.enums import TaskKind
from synapse.domain.models import format_model_registry_summary
from synapse.domain.tasks import Task, TaskDAG
from synapse.events import EventBus
from synapse.hardware.tier_resolver import HardwareTier, TierAssignment, TierResolver
from synapse.master.schemas import (
    ExecutionMode,
    MasterAnalysis,
    ReasoningComplexity,
    TaskDecompositionPlan,
)
from synapse.providers.manager import ProviderManager

log = structlog.get_logger("synapse.master.orchestrator")

#: system instruction for Master Analysis — runs on the raw prompt before any files/tools.
_ANALYSIS_SYSTEM_PROMPT = (
    "You are the Synapse One Master AI Orchestrator. Your ONLY job is to analyze "
    "the user's request BEFORE any files, workspace, or tools are loaded.\n\n"
    "CRITICAL RULES:\n"
    "1. You are an orchestrator and router, NOT a worker. NEVER solve the task directly or generate file contents here.\n"
    "2. Distinguish between ANSWERING ABOUT SOMETHING vs DOING/CREATING SOMETHING:\n"
    "   - 'Explain HTML' -> direct answer, artifact_required=false, files_needed=false, workspace_needed=false\n"
    "   - 'Show me an HTML example' -> direct answer, artifact_required=false\n"
    "   - 'Generate/Create/Build me an HTML/CSS landing page' -> artifact_generation, artifact_required=true, files_needed=true, coding_needed=true, execution_mode=artifact_generation\n"
    "   - 'Create a landing page in my project/workspace' -> workspace_agent, artifact_required=true, workspace_needed=true, files_needed=true, coding_needed=true\n"
    "   - Math problems / logic puzzles / standalone queries -> direct_answer, workspace_needed=false, files_needed=false, tools_needed=false\n"
    "3. WORKSPACE OPT-IN: Do NOT set workspace_needed=true unless the request specifically asks to inspect, modify, or operate on existing workspace/project files.\n"
    "4. ARTIFACT INDEPENDENCE: A request can require creating files/artifacts (artifact_required=true, files_needed=true) even if existing workspace inspection is not needed.\n"
    "5. DIFFICULTY EVALUATION: Evaluate reasoning_complexity ('trivial', 'easy', 'medium', 'hard', 'very_hard'). Simple arithmetic is trivial/easy; mathematical proofs or complex logical puzzles (e.g. chessboard dominoes) are hard.\n"
    "6. Output ONLY valid JSON matching the MasterAnalysis schema."
)

#: system instruction — the Master AI is an orchestrator/router, never a worker.
_DIVIDER_SYSTEM_PROMPT = (
    "You are the Synapse One Master AI Orchestrator. Your ONLY job is to analyze "
    "the user's request, detect required capabilities, decompose complex requests into "
    "minimal subtasks, build a Task DAG with dependencies, and select the appropriate specialist "
    "model or tool for each subtask from the provided Model Registry.\n\n"
    "CRITICAL RULES:\n"
    "1. You are an orchestrator and router, NOT a worker. NEVER directly solve the specialist task yourself.\n"
    "2. Each subtask MUST be assigned a target capability and a specialist model or tool from the registry.\n"
    "3. Different subtasks can and should route to different specialist models (e.g. coding to Qwen Coder, "
    "math/chat to Gemma, vision to vision model, file creation to filesystem_tool).\n"
    "4. Do NOT use heavy models like 32B unless the subtask genuinely requires deep reasoning.\n"
    "5. Output ONLY valid JSON matching the provided schema."
)

#: fallback search order when the assigned tier has no usable model installed:
#: remaining local tiers (ascending size), then the cloud tier.
_LOCAL_TIER_ORDER = (HardwareTier.TIER1, HardwareTier.TIER2, HardwareTier.TIER3, HardwareTier.TIER3_PLUS)

#: strip ```json ... ``` fences and any prose around the JSON object.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort repair: pull the first balanced JSON object out of ``text``.

    Handles fences, leading prose, trailing prose and stray backticks — the
    common failure modes of instruction-following models. Raises ValueError
    when no JSON object can be found.
    """
    cleaned = _FENCE_RE.sub("", text or "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(cleaned[start : end + 1])


class MasterModelSelection:
    """Which provider/model serves the Master AI role for one request."""

    __slots__ = ("provider_id", "model_id", "tier", "reason")

    def __init__(self, provider_id: str, model_id: str, tier: HardwareTier, reason: str) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        self.tier = tier
        self.reason = reason


class AIMasterOrchestrator(TaskPlanner):
    """Model-backed front planner; deterministic fallback behind it.

    Composes with any other TaskPlanner: the injected ``fallback_planner``
    receives the whole prompt untouched whenever the Master AI cannot produce
    a valid plan (and when disabled).
    """

    def __init__(
        self,
        *,
        providers: ProviderManager,
        registry: ModelRegistry,
        hardware: HardwareProvider,
        config: ConfigProvider,
        resolver: TierResolver | None = None,
        fallback_planner: TaskPlanner | None = None,
        enabled: bool = True,
        temperature: float = 0.1,
        max_retries: int = 1,
        max_tasks: int = 8,
        events: EventBus | None = None,
    ) -> None:
        self._providers = providers
        self._registry = registry
        self._hardware = hardware
        self._config = config
        self._resolver = resolver or TierResolver(config)
        self._fallback = fallback_planner
        self._enabled = enabled
        self._temperature = temperature
        self._max_retries = max(0, int(max_retries))
        self._max_tasks = max(1, int(max_tasks))
        self._events = events

        # observability for the last plan() call.
        self.last_tier: HardwareTier | None = None
        self.last_model: str | None = None
        self.last_analysis: MasterAnalysis | None = None
        self.used_ai: bool = False

    # -- Master Analysis (runs first before any workspace / files / tools) --

    def analyze(self, prompt: str) -> MasterAnalysis:
        """Analyze the user request using the Master Model before loading workspace or tools."""
        if not self._enabled:
            analysis = self._fallback_analysis(prompt, "master ai disabled")
            self.last_analysis = analysis
            return analysis

        selection = self._select_master_model()
        if selection is None:
            analysis = self._fallback_analysis(prompt, "no master model available")
            self.last_analysis = analysis
            return analysis

        self.last_tier = selection.tier
        self.last_model = selection.model_id

        failures: list[str] = []
        for attempt in range(self._max_retries + 1):
            try:
                raw = self._call_master_analysis(selection, prompt)
                analysis = self._to_analysis(raw, prompt)
                self.last_analysis = analysis
                log.info(
                    "ai_master_analysis_ok",
                    intent=analysis.intent,
                    domain=analysis.domain,
                    workspace_needed=analysis.workspace_needed,
                    artifact_required=analysis.artifact_required,
                    complexity=analysis.reasoning_complexity.value,
                    model=selection.model_id,
                )
                return analysis
            except Exception as exc:  # noqa: BLE001 - any failure falls back
                failures.append(str(exc)[:200])
                log.warning(
                    "ai_master_analysis_failed",
                    attempt=attempt,
                    provider=selection.provider_id,
                    model=selection.model_id,
                    error=str(exc)[:200],
                )

        analysis = self._fallback_analysis(
            prompt, f"master ai analysis failed: {failures[-1] if failures else 'unknown'}"
        )
        self.last_analysis = analysis
        return analysis

    def _call_master_analysis(self, selection: MasterModelSelection, prompt: str) -> str:
        provider = self._providers.get(selection.provider_id)
        if provider is None:
            raise RuntimeError(f"master provider '{selection.provider_id}' not registered")

        registry_models = self._registry.all()
        hardware = None
        try:
            hardware = self._hardware.scan()
        except Exception:
            pass
        avail_ram = hardware.memory.available_gb if hardware and hardware.memory else None
        vram = hardware.gpu.vram_gb if hardware and hardware.gpu else None

        registry_summary = format_model_registry_summary(
            registry_models,
            tier=selection.tier.value,
            available_ram_gb=avail_ram,
            vram_gb=vram,
        )

        user_content = (
            f"SYSTEM HARDWARE CONTEXT:\n"
            f"- Detected Tier: {selection.tier.value}\n"
            f"- Available RAM: {avail_ram if avail_ram is not None else 'unknown'} GB\n"
            f"- Dedicated GPU VRAM: {vram if vram is not None else 0} GB\n\n"
            f"{registry_summary}\n\n"
            "MASTER ANALYSIS INSTRUCTIONS:\n"
            "Analyze the user's prompt below. Output strict JSON conforming to the MasterAnalysis schema.\n"
            "Determine:\n"
            "- intent: artifact_generation, question_answering, coding, mathematical_reasoning, logical_reasoning, direct_answer, etc.\n"
            "- domain: web_development, mathematics, python, general, logic, etc.\n"
            "- goal: summary of user goal\n"
            "- workspace_needed (boolean): true ONLY if existing workspace files must be inspected\n"
            "- files_needed (boolean): true if files must be created or edited on disk\n"
            "- coding_needed (boolean): true if code must be written/generated\n"
            "- artifact_required (boolean): true if an artifact (files, web page, project) must be produced\n"
            "- reasoning_complexity: trivial, easy, medium, hard, very_hard\n"
            "- required_capabilities: list of required capability strings (e.g. ['html', 'css', 'code_generation', 'file_creation'])\n"
            "- recommended_model_role: specialist role name from registry\n"
            "- execution_mode: direct_answer, workspace_agent, tool_execution, artifact_generation, multi_step_agent\n"
            "- confidence: 0.0 to 1.0\n\n"
            "Do NOT solve the task or generate the file content yourself. Output ONLY valid JSON.\n"
            f"Schema: {json.dumps(MasterAnalysis.model_json_schema())}\n\n"
            f"USER PROMPT:\n{prompt}"
        )

        request = ChatRequest(
            messages=[
                ChatMessage(role="system", content=_ANALYSIS_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_content),
            ],
            temperature=self._temperature,
            model=selection.model_id,
            format=MasterAnalysis.model_json_schema(),
        )
        response = provider.chat(request)
        content = (response.content or "").strip()
        if not content:
            raise RuntimeError("master provider returned empty output for analysis")
        return content

    def _to_analysis(self, raw: str, prompt: str) -> MasterAnalysis:
        obj = extract_json_object(raw)
        if "reasoning_complexity" in obj:
            val = str(obj["reasoning_complexity"]).lower().strip()
            for member in ReasoningComplexity:
                if member.value in val or val in member.value:
                    obj["reasoning_complexity"] = member.value
                    break
            else:
                obj["reasoning_complexity"] = ReasoningComplexity.EASY.value
        if "execution_mode" in obj:
            val = str(obj["execution_mode"]).lower().strip()
            for member in ExecutionMode:
                if member.value in val or val in member.value:
                    obj["execution_mode"] = member.value
                    break
            else:
                obj["execution_mode"] = ExecutionMode.DIRECT_ANSWER.value
        return MasterAnalysis.model_validate(obj)

    @staticmethod
    def _fallback_analysis(prompt: str, reason: str = "") -> MasterAnalysis:
        log.info("ai_master_fallback_analysis", reason=reason)
        text = (prompt or "").strip()
        lowered = text.lower()

        # Check workspace queries / cues
        has_workspace_cues = any(
            w in lowered
            for w in (
                "project", "workspace", "codebase", "repository", "notes",
                "these files", "my files", "in this", "existing", "tests",
                "test suite", "overview", "readme", "documentation", "files",
                "design notes"
            )
        )

        # 1. Math and logic reasoning
        is_math_puzzle = any(
            hint in lowered
            for hint in (
                "chessboard", "domino", "dominoes", "corner squares", "62 squares",
                "father is 4 times", "how old are they", "in 20 years", "algebra",
                "calculus", "equation", "derivative", "integral", "solve for",
                "logic puzzle", "riddle", "knights and knaves"
            )
        )
        is_hard_reasoning = any(
            hint in lowered
            for hint in (
                "chessboard", "domino", "mutilated", "opposite corner", "62 squares",
                "proof", "prove that", "theorem", "quantum", "relativity", "distributed consensus"
            )
        )
        if is_math_puzzle:
            complexity = ReasoningComplexity.HARD if is_hard_reasoning else ReasoningComplexity.MEDIUM
            role = "General Chat & Math/Reasoning (Tier 3)" if is_hard_reasoning else "Math & Reasoning Specialist"
            return MasterAnalysis(
                intent="mathematical_reasoning" if "father" in lowered or "algebra" in lowered else "logical_reasoning",
                domain="mathematics",
                goal="Solve the mathematical / logical reasoning problem accurately with full proof or derivation",
                workspace_needed=False,
                workspace_reason="Standalone math/logic question does not require workspace files",
                files_needed=False,
                memory_needed=False,
                tools_needed=False,
                coding_needed=False,
                vision_needed=False,
                document_processing_needed=False,
                web_needed=False,
                artifact_required=False,
                reasoning_complexity=complexity,
                required_capabilities=["reasoning", "math"],
                recommended_model_role=role,
                execution_mode=ExecutionMode.DIRECT_ANSWER,
                confidence=0.95,
            )

        # 2. Explanations (without create/build intent)
        is_explanation = (
            lowered.startswith(("explain", "what is", "how do", "why does", "tell me about", "describe", "summarize", "summarise", "overview"))
            and not any(verb in lowered for verb in ("generate", "create", "build", "make", "implement", "scaffold", "write a", "write me", "write the", "write to"))
        )
        if is_explanation:
            return MasterAnalysis(
                intent="question_answering",
                domain="workspace_analysis" if has_workspace_cues else "general_knowledge",
                goal=f"Provide educational explanation for: {text[:80]}",
                workspace_needed=has_workspace_cues,
                workspace_reason="Workspace inspection needed for project context" if has_workspace_cues else "General explanation query does not need workspace access",
                files_needed=False,
                memory_needed=has_workspace_cues,
                tools_needed=False,
                coding_needed=False,
                vision_needed=False,
                document_processing_needed=False,
                web_needed=False,
                artifact_required=False,
                reasoning_complexity=ReasoningComplexity.EASY,
                required_capabilities=["chat", "writing"],
                recommended_model_role="General Chat",
                execution_mode=ExecutionMode.DIRECT_ANSWER,
                confidence=0.95,
            )

        # 3. Artifact generation / file creation
        is_create = any(
            v in lowered
            for v in ("create", "generate", "build", "make", "implement", "scaffold", "develop", "write a", "write me", "write the", "write to", "save to", "dump to")
        )
        is_web = any(
            w in lowered
            for w in ("html", "css", "landing page", "website", "web page", "frontend", "web app", "site")
        )
        is_explicit_workspace = any(
            w in lowered
            for w in ("in my project", "in my workspace", "in this repository", "in existing project", "project overview")
        )

        if is_create and (is_web or "file" in lowered or "script" in lowered or "app" in lowered or "output." in lowered or "readme" in lowered or "doc" in lowered or "test" in lowered):
            domain = "web_development" if is_web else "software_development"
            caps = ["html", "css", "code_generation", "file_creation"] if is_web else ["coding", "file_creation"]
            mode = ExecutionMode.WORKSPACE_AGENT if (is_explicit_workspace or has_workspace_cues) else ExecutionMode.ARTIFACT_GENERATION
            return MasterAnalysis(
                intent="artifact_generation",
                domain=domain,
                goal=f"Generate and create artifact: {text[:80]}",
                workspace_needed=is_explicit_workspace or has_workspace_cues,
                workspace_reason="Targeting existing project workspace" if (is_explicit_workspace or has_workspace_cues) else "New artifact creation",
                files_needed=True,
                memory_needed=False,
                tools_needed=True,
                coding_needed=True,
                vision_needed=False,
                document_processing_needed=False,
                web_needed=False,
                artifact_required=True,
                reasoning_complexity=ReasoningComplexity.EASY,
                required_capabilities=caps,
                recommended_model_role="Coding Specialist",
                execution_mode=mode,
                confidence=0.95,
            )

        # 4. Testing / tools
        if "test" in lowered or "run " in lowered:
            return MasterAnalysis(
                intent="tool_execution",
                domain="software_development",
                goal=text[:100],
                workspace_needed=True,
                workspace_reason="Running tests or inspecting workspace",
                files_needed=False,
                memory_needed=True,
                tools_needed=True,
                coding_needed=False,
                vision_needed=False,
                document_processing_needed=False,
                web_needed=False,
                artifact_required=False,
                reasoning_complexity=ReasoningComplexity.EASY,
                required_capabilities=["tools", "chat"],
                recommended_model_role="General Chat",
                execution_mode=ExecutionMode.WORKSPACE_AGENT,
                confidence=0.9,
            )

        # 5. Fallback general request
        return MasterAnalysis(
            intent="general_request",
            domain="general",
            goal=text[:100],
            workspace_needed=False,
            workspace_reason="Standard query",
            files_needed=False,
            memory_needed=False,
            tools_needed=False,
            coding_needed=False,
            vision_needed=False,
            document_processing_needed=False,
            web_needed=False,
            artifact_required=False,
            reasoning_complexity=ReasoningComplexity.EASY,
            required_capabilities=["chat"],
            recommended_model_role="General Chat",
            execution_mode=ExecutionMode.DIRECT_ANSWER,
            confidence=0.8,
        )

    # -- contract -----------------------------------------------------------

    def plan(
        self,
        prompt: str,
        intent: IntentResult,
        complexity: ComplexityResult,
        privacy: PrivacyResult,
        decision: Decision,
    ) -> TaskDAG:
        self.used_ai = False
        self.last_tier = None
        self.last_model = None
        if not self._enabled:
            return self._fallback_plan(prompt, intent, complexity, privacy, decision, "disabled")

        selection = self._select_master_model()
        if selection is None:
            return self._fallback_plan(
                prompt, intent, complexity, privacy, decision,
                "no master model available on this machine",
            )
        self.last_tier = selection.tier
        self.last_model = selection.model_id
        log.info(
            "ai_master_model_selected",
            tier=selection.tier.value,
            provider=selection.provider_id,
            model=selection.model_id,
            reason=selection.reason,
        )

        plan: TaskDecompositionPlan | None = None
        failures: list[str] = []
        for attempt in range(self._max_retries + 1):
            try:
                raw = self._call_master(selection, prompt)
                plan = self._to_plan(raw, prompt)
                break
            except Exception as exc:  # noqa: BLE001 - any failure falls back
                failures.append(str(exc)[:200])
                log.warning(
                    "ai_master_plan_failed",
                    attempt=attempt,
                    provider=selection.provider_id,
                    model=selection.model_id,
                    error=str(exc)[:200],
                )

        if plan is None:
            log.warning("ai_master_fallback", prompt_length=len(prompt), failures=failures)
            return self._fallback_plan(
                prompt, intent, complexity, privacy, decision,
                "master ai failed to produce a valid plan",
            )

        dag = plan.to_dag()
        self.used_ai = True
        log.info(
            "ai_master_plan_ok",
            tasks=len([t for t in dag.tasks if t.kind.value != "synthesis"]),
            strategy=plan.execution_strategy.value,
            provider=selection.provider_id,
            model=selection.model_id,
        )
        return dag

    # -- the AI call --------------------------------------------------------

    def _call_master(self, selection: MasterModelSelection, prompt: str) -> str:
        provider = self._providers.get(selection.provider_id)
        if provider is None:
            raise RuntimeError(f"master provider '{selection.provider_id}' not registered")

        # Dynamically generate Model Registry summary from actual registry
        registry_models = self._registry.all()
        hardware = None
        try:
            hardware = self._hardware.scan()
        except Exception:
            pass
        avail_ram = hardware.memory.available_gb if hardware and hardware.memory else None
        vram = hardware.gpu.vram_gb if hardware and hardware.gpu else None

        registry_summary = format_model_registry_summary(
            registry_models,
            tier=selection.tier.value,
            available_ram_gb=avail_ram,
            vram_gb=vram,
        )

        user_content = (
            f"SYSTEM HARDWARE CONTEXT:\n"
            f"- Detected Tier: {selection.tier.value}\n"
            f"- Available RAM: {avail_ram if avail_ram is not None else 'unknown'} GB\n"
            f"- Dedicated GPU VRAM: {vram if vram is not None else 0} GB\n\n"
            f"{registry_summary}\n\n"
            "TASK DECOMPOSITION INSTRUCTIONS:\n"
            "Analyze the user's prompt below. Decompose it into minimal subtasks with clear dependencies.\n"
            "Assign each subtask an intent, target capability, and the exact specialist model or tool "
            "from the Model Registry above that is best suited to execute it.\n"
            "Do NOT perform the work yourself. Output ONLY valid JSON matching the schema below.\n"
            f"Schema: {json.dumps(TaskDecompositionPlan.model_json_schema())}\n\n"
            f"USER PROMPT:\n{prompt}"
        )

        request = ChatRequest(
            messages=[
                ChatMessage(role="system", content=_DIVIDER_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_content),
            ],
            temperature=self._temperature,
            model=selection.model_id,
            format=TaskDecompositionPlan.model_json_schema(),
        )
        response = provider.chat(request)
        content = (response.content or "").strip()
        if not content:
            raise RuntimeError("master provider returned empty output")
        return content

    def _to_plan(self, raw: str, prompt: str) -> TaskDecompositionPlan:
        """Parse + validate raw model output into a TaskDecompositionPlan."""
        obj = extract_json_object(raw)
        tasks = obj.get("tasks")
        if isinstance(tasks, list) and len(tasks) > self._max_tasks:
            kept = tasks[: self._max_tasks]
            kept_ids = {t.get("task_id") for t in kept if isinstance(t, dict)}
            for t in kept:
                if isinstance(t, dict):
                    deps = t.get("dependencies")
                    if isinstance(deps, list):
                        #: dependencies on dropped tasks are removed so the
                        #: capped DAG stays valid.
                        t["dependencies"] = [d for d in deps if d in kept_ids]
            obj = {**obj, "tasks": kept}

        # Guardrail: enforce 32B model restriction (only tier3_plus and genuine deep reasoning)
        if isinstance(obj.get("tasks"), list):
            for t in obj["tasks"]:
                if not isinstance(t, dict):
                    continue
                assigned = str(t.get("assigned_model", "")).lower()
                intent = str(t.get("intent", "")).upper()
                capability = str(t.get("capability", "")).lower()
                reasoning = str(t.get("reasoning", "")).lower()
                if "32b" in assigned:
                    is_deep_reasoning = (
                        intent == "DEEP_REASONING"
                        or "deep" in capability
                        or "deep reasoning" in reasoning
                        or "complex analysis" in reasoning
                    )
                    is_tier3_plus = self.last_tier is HardwareTier.TIER3_PLUS
                    if not (is_deep_reasoning and is_tier3_plus):
                        # Safely fallback to the tier's standard reasoning model
                        fallback_model = (
                            "gemma3:12b"
                            if self.last_tier in (HardwareTier.TIER3, HardwareTier.TIER3_PLUS)
                            else ("gemma3:4b" if self.last_tier is HardwareTier.TIER2 else "gemma3:1b")
                        )
                        t["assigned_model"] = fallback_model
                        log.info(
                            "32b_guardrail_applied",
                            original=assigned,
                            fallback=fallback_model,
                            is_deep_reasoning=is_deep_reasoning,
                            is_tier3_plus=is_tier3_plus,
                        )

        plan = TaskDecompositionPlan.model_validate(obj)
        if not plan.tasks:
            raise ValueError("decomposition plan contains no tasks")
        return plan

    # -- model selection -----------------------------------------------------

    def _select_master_model(self) -> MasterModelSelection | None:
        """Pick the first usable candidate: assigned tier, then escalation."""
        try:
            hardware = self._hardware.scan()
        except Exception:  # noqa: BLE001 - a broken scanner must not break planning
            log.warning("ai_master_hardware_scan_failed")
            return None
        assignment = self._resolver.resolve(hardware)
        registry = self._registry.all()

        tiers = self._tier_search_order(assignment)
        health_cache: dict[str, bool] = {}
        for tier in tiers:
            for model_id in self._resolver.candidates_for(tier):
                meta = next((m for m in registry if m.id == model_id), None)
                if meta is None:
                    continue
                pid = meta.provider_id
                if pid not in health_cache:
                    health_cache[pid] = self._providers.health(pid)
                if not health_cache[pid]:
                    continue
                return MasterModelSelection(
                    provider_id=pid,
                    model_id=model_id,
                    tier=tier,
                    reason=f"hardware tier {assignment.tier.value}: {assignment.reason}",
                )
        log.warning(
            "ai_master_no_candidate",
            assigned_tier=assignment.tier.value,
            registry_count=len(registry),
        )
        return None

    @staticmethod
    def _tier_search_order(assignment: TierAssignment) -> tuple[HardwareTier, ...]:
        """Assigned tier first (excl. cloud), then ascending local, then cloud."""
        assigned = assignment.tier
        if assigned is HardwareTier.CLOUD_FALLBACK:
            return (HardwareTier.CLOUD_FALLBACK,)
        return (
            assigned,
            *(t for t in _LOCAL_TIER_ORDER if t is not assigned),
            HardwareTier.CLOUD_FALLBACK,
        )

    # -- fallback -------------------------------------------------------------

    def _fallback_plan(
        self,
        prompt: str,
        intent: IntentResult,
        complexity: ComplexityResult,
        privacy: PrivacyResult,
        decision: Decision,
        reason: str,
    ) -> TaskDAG:
        if self._fallback is not None:
            log.info("ai_master_fallback_planner", reason=reason)
            return self._fallback.plan(prompt, intent, complexity, privacy, decision)
        log.info("ai_master_fallback_single", reason=reason)
        return TaskDAG(
            tasks=[
                Task(
                    id="t1",
                    kind=self._fallback_kind(prompt),
                    description=prompt,
                    required_capabilities=list(decision.required_capabilities or [Capability.CHAT]),
                    preferred_capabilities=list(decision.preferred_capabilities),
                )
            ]
        )

    @staticmethod
    def _fallback_kind(prompt: str) -> TaskKind:
        """Smallest deterministic kind guess for the pass-through fallback."""
        from synapse.planner.heuristic import _KIND_HINTS  # noqa: PLC0415

        lowered = prompt.lower()
        for kind, hints in _KIND_HINTS:
            if any(hint in lowered for hint in hints):
                return kind
        return TaskKind.GENERAL