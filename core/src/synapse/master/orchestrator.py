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
import threading
import time
from typing import Any

import structlog

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
    "   - 'Explain photosynthesis in simple terms' / 'Explain HTML' -> direct answer, artifact_required=false, files_needed=false, workspace_needed=false, web_needed=false, required_capabilities=['conversation', 'reasoning'], recommended_model_role='gemma3:4b'\n"
    "   - 'Show me an HTML example' -> direct answer, artifact_required=false, web_needed=false\n"
    "   - 'Generate/Create/Build me an HTML/CSS landing page' -> artifact_generation, artifact_required=true, files_needed=true, coding_needed=true, web_needed=false, execution_mode=artifact_generation\n"
    "   - 'Create a landing page in my project/workspace' -> workspace_agent, artifact_required=true, workspace_needed=true, files_needed=true, coding_needed=true, web_needed=false\n"
    "   - 'Add ... to the FastAPI code' / 'Edit existing file' / 'Modify main.py' -> coding, workspace_needed=true, files_needed=true, coding_needed=true, artifact_required=true, web_needed=false, execution_mode=edit_existing\n"
    "   - 'What is the latest version of FastAPI?' / 'Recent news about AI' -> question_answering, web_needed=true, tools_needed=true, workspace_needed=false, files_needed=false\n"
    "   - Math problems / logic puzzles / standalone queries -> direct_answer, workspace_needed=false, files_needed=false, tools_needed=false, web_needed=false\n"
    "3. WEB ACCESS (web_needed): Set web_needed=true ONLY if the request clearly requires current, live, or real-time web information (e.g. latest version/release, recent news, live prices, current weather, recent documentation). Pure coding, offline logic, workspace file editing, math, and general conversation MUST have web_needed=false.\n"
    "4. WORKSPACE OPT-IN: Do NOT set workspace_needed=true unless the request specifically asks to inspect, modify, or operate on existing workspace/project files.\n"
    "5. ARTIFACT INDEPENDENCE: A request can require creating files/artifacts (artifact_required=true, files_needed=true) even if existing workspace inspection is not needed.\n"
    "6. DIFFICULTY EVALUATION: Evaluate reasoning_complexity ('trivial', 'easy', 'medium', 'hard', 'very_hard'). Simple arithmetic is trivial/easy; mathematical proofs or complex logical puzzles (e.g. chessboard dominoes) are hard.\n"
    "7. Output ONLY valid JSON matching the MasterAnalysis schema."
)

#: system instruction — the Master AI is an orchestrator/router, never a worker.
_DIVIDER_SYSTEM_PROMPT = (
    "You are the Synapse One Master AI Orchestrator (≤1.7B parameter router). Your ONLY job is to analyze "
    "the user's request, detect required capabilities, decompose complex requests into "
    "minimal subtasks with clear dependencies, and select the appropriate specialist "
    "model or tool for each subtask from the provided Model Registry.\n\n"
    "CRITICAL NON-NEGOTIABLE RULES:\n"
    "1. You are strictly a planner and router, NOT a worker. NEVER directly solve the task yourself, never write code, never do math proofs, never generate documents, and never claim a file was created.\n"
    "2. Real filesystem tools are the ONLY way files are created or modified on disk.\n"
    "3. Each subtask MUST be assigned an intent, target capability, and the exact specialist model or tool from the registry.\n"
    "4. Different subtasks can and should route to different specialist models (e.g. coding to Qwen Coder, "
    "math/chat to Gemma, vision to vision model, file creation to filesystem_tool).\n"
    "5. Do NOT use heavy models like 32B unless the subtask genuinely requires deep reasoning on high-end hardware.\n"
    "6. Output strictly valid JSON matching the TaskDecompositionPlan schema.\n"
    "7. For rich document artifact subtasks (.pdf, .docx, .pptx, .xlsx, .csv), sub_prompt MUST specify producing complete and comprehensive readable content (detailed paragraphs, multi-point bullet lists, filled tables, full slide text), never empty skeletons.\n\n"
    "FEW-SHOT EXAMPLES OF VALID TASK DECOMPOSITION PLANS:\n\n"
    "Example 1: Compound Coding Project with Tests and Docs\n"
    "User: 'Create a CLI expense tracker in Python with tests and documentation'\n"
    "{\n"
    '  "execution_strategy": "SEQUENTIAL",\n'
    '  "tasks": [\n'
    '    {"task_id": 1, "intent": "PLANNING", "sub_prompt": "Plan architecture and CLI structure for expense tracker", "assigned_model": "qwen3:4b", "capability": "planning", "reasoning": "system design planning", "dependencies": []},\n'
    '    {"task_id": 2, "intent": "CODING", "sub_prompt": "Implement expense tracker in src/expense_tracker.py", "assigned_model": "qwen2.5-coder:7b", "capability": "coding", "reasoning": "code specialist implementation", "dependencies": [1]},\n'
    '    {"task_id": 3, "intent": "CODING", "sub_prompt": "Write pytest suite in test/test_expense_tracker.py", "assigned_model": "qwen2.5-coder:7b", "capability": "testing", "reasoning": "test suite implementation", "dependencies": [2]},\n'
    '    {"task_id": 4, "intent": "WRITING", "sub_prompt": "Write documentation in docs/architecture.md", "assigned_model": "gemma3:4b", "capability": "writing", "reasoning": "documentation writing", "dependencies": [2]}\n'
    "  ]\n"
    "}\n\n"
    "Example 2: Mathematical Proof / Deep Reasoning\n"
    "User: 'Prove why opposite corners of a chessboard cannot be covered by 31 dominoes'\n"
    "{\n"
    '  "execution_strategy": "SEQUENTIAL",\n'
    '  "tasks": [\n'
    '    {"task_id": 1, "intent": "DEEP_REASONING", "sub_prompt": "Formalize chessboard coloring proof and deduce contradiction", "assigned_model": "gemma3:4b", "capability": "reasoning", "reasoning": "deep mathematical proof", "dependencies": []}\n'
    "  ]\n"
    "}\n\n"
    "Example 3: Bug Fix in Workspace\n"
    "User: 'Fix the IndexError in parser.py'\n"
    "{\n"
    '  "execution_strategy": "SEQUENTIAL",\n'
    '  "tasks": [\n'
    '    {"task_id": 1, "intent": "CODING", "sub_prompt": "Diagnose and fix IndexError in parser.py", "assigned_model": "qwen2.5-coder:7b", "capability": "debugging", "reasoning": "code debugging specialist", "dependencies": []}\n'
    "  ]\n"
    "}\n\n"
    "Example 4: Image Analysis\n"
    "User: 'Inspect chart.png and describe the data trends'\n"
    "{\n"
    '  "execution_strategy": "SEQUENTIAL",\n'
    '  "tasks": [\n'
    '    {"task_id": 1, "intent": "VISION", "sub_prompt": "Extract and interpret data trends from chart.png", "assigned_model": "qwen2.5vl:7b", "capability": "vision", "reasoning": "multimodal vision specialist", "dependencies": []}\n'
    "  ]\n"
    "}\n\n"
    "Example 5: Simple Text Explanation / Question Answering\n"
    "User: 'Explain photosynthesis in simple terms'\n"
    "{\n"
    '  "execution_strategy": "SEQUENTIAL",\n'
    '  "tasks": [\n'
    '    {"task_id": 1, "intent": "CONVERSATION", "sub_prompt": "Explain photosynthesis in simple terms", "assigned_model": "gemma3:4b", "capability": "conversation", "reasoning": "educational text explanation", "dependencies": []}\n'
    "  ]\n"
    "}\n\n"
    "Example 6: Rich Document Generation (PDF/DOCX/PPTX/XLSX)\n"
    "User: 'Create a 1-page PDF report on the benefits of local AI in docs/local_ai.pdf'\n"
    "{\n"
    '  "execution_strategy": "SEQUENTIAL",\n'
    '  "tasks": [\n'
    '    {"task_id": 1, "intent": "DOCUMENT_GENERATION", "sub_prompt": "Generate full, comprehensive readable content with detailed paragraphs and benefits for docs/local_ai.pdf", "assigned_model": "qwen2.5-coder:7b", "capability": "document_generation", "reasoning": "document generation specialist", "dependencies": []}\n'
    "  ]\n"
    "}"
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


def validate_plan_dag(plan: TaskDecompositionPlan) -> None:
    """Validate task graph invariants before execution.

    Checks:
    - Non-empty tasks list
    - Unique non-empty task IDs
    - All sub_prompts are non-empty strings
    - Dependencies reference existing tasks
    - No self-dependencies
    - Acyclicity (verified via topological sort)
    """
    if not plan.tasks:
        raise ValueError("decomposition plan contains no tasks")
    ids = set()
    for task in plan.tasks:
        if task.task_id in ids:
            raise ValueError(f"duplicate task_id {task.task_id} in decomposition plan")
        ids.add(task.task_id)
        if not task.sub_prompt or not task.sub_prompt.strip():
            raise ValueError(f"task {task.task_id} has an empty prompt")
        for dep in task.dependencies:
            if dep == task.task_id:
                raise ValueError(f"task {task.task_id} depends on itself")
            if dep not in ids and dep not in {t.task_id for t in plan.tasks}:
                raise ValueError(f"task {task.task_id} depends on unknown task {dep}")
    dag = plan.to_dag()
    dag.topological_order()


class AnalysisCache:
    """Thread-safe LRU + TTL cache for MasterAnalysis results."""

    def __init__(self, max_size: int = 128, ttl_seconds: float = 120.0) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[float, MasterAnalysis]] = {}
        self._lock = threading.Lock()

    def _normalize_key(self, prompt: str, conversation_id: str | None = None) -> str:
        cleaned = " ".join(prompt.strip().lower().split())
        return f"{conversation_id or 'global'}::{cleaned}"

    def get(self, prompt: str, conversation_id: str | None = None) -> MasterAnalysis | None:
        key = self._normalize_key(prompt, conversation_id)
        now = time.time()
        with self._lock:
            if key in self._cache:
                timestamp, analysis = self._cache[key]
                if now - timestamp <= self._ttl:
                    return analysis.model_copy(deep=True)
                del self._cache[key]
        return None

    def set(self, prompt: str, analysis: MasterAnalysis, conversation_id: str | None = None) -> None:
        key = self._normalize_key(prompt, conversation_id)
        now = time.time()
        with self._lock:
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (now, analysis.model_copy(deep=True))

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


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
        self._analysis_cache = AnalysisCache(max_size=128, ttl_seconds=120.0)

        # observability for the last plan() call.
        self.last_tier: HardwareTier | None = None
        self.last_model: str | None = None
        self.last_analysis: MasterAnalysis | None = None
        self.used_ai: bool = False

    # -- Master Analysis (runs first before any workspace / files / tools) --

    def analyze(self, prompt: str, conversation_id: str | None = None) -> MasterAnalysis:
        """Analyze the user request using the Master Model before loading workspace or tools."""
        cached = self._analysis_cache.get(prompt, conversation_id)
        if cached is not None:
            log.info("ai_master_analysis_cache_hit", prompt=prompt[:60])
            self.last_analysis = cached
            return cached

        if not self._enabled:
            analysis = self._fallback_analysis(prompt, "master ai disabled")
            self._analysis_cache.set(prompt, analysis, conversation_id)
            self.last_analysis = analysis
            return analysis

        selection = self._select_master_model()
        if selection is None:
            analysis = self._fallback_analysis(prompt, "no master model available")
            self._analysis_cache.set(prompt, analysis, conversation_id)
            self.last_analysis = analysis
            return analysis

        self.last_tier = selection.tier
        self.last_model = selection.model_id

        failures: list[str] = []
        for attempt in range(self._max_retries + 1):
            try:
                raw = self._call_master_analysis(selection, prompt)
                analysis = self._to_analysis(raw, prompt)
                self._analysis_cache.set(prompt, analysis, conversation_id)
                self.last_analysis = analysis
                log.info(
                    "ai_master_analysis_ok",
                    intent=analysis.intent,
                    domain=analysis.domain,
                    workspace_needed=analysis.workspace_needed,
                    artifact_required=analysis.artifact_required,
                    web_needed=analysis.web_needed,
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
        self._analysis_cache.set(prompt, analysis, conversation_id)
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
            "- web_needed (boolean): true ONLY if request requires live/current web information (latest versions, news, live prices)\n"
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
            max_tokens=512,
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

        # 2. Chat Memory & Conversation Summaries
        is_chat_memory = any(
            hint in lowered
            for hint in (
                "what is this chat about",
                "what is our chat about",
                "what is this conversation about",
                "summarize this chat",
                "summarize this conversation",
                "summarize our conversation",
                "what have we done so far",
                "what did we do so far",
                "what have we accomplished",
                "what did we discuss",
                "chat summary",
                "conversation summary",
                "what are we working on in this chat",
                "what have we been doing",
            )
        )
        if is_chat_memory:
            return MasterAnalysis(
                intent="question_answering",
                domain="chat_memory",
                goal="Summarize current chat conversation history and accomplishments",
                workspace_needed=False,
                workspace_reason="Chat memory query relies on injected conversation history",
                files_needed=False,
                memory_needed=True,
                tools_needed=False,
                coding_needed=False,
                vision_needed=False,
                document_processing_needed=False,
                web_needed=False,
                artifact_required=False,
                reasoning_complexity=ReasoningComplexity.EASY,
                required_capabilities=["chat", "writing", "conversation"],
                recommended_model_role="General Chat",
                execution_mode=ExecutionMode.DIRECT_ANSWER,
                confidence=0.99,
            )

        # 3. Web search & live information queries
        is_web_query = any(
            kw in lowered
            for kw in (
                "latest version", "current version", "latest release", "newest version",
                "what is the latest", "who is the current", "latest news", "current price",
                "stock price", "current weather", "weather in", "latest documentation",
                "recent updates", "recent news", "today's news", "who won", "score of",
                "latest stable", "latest fastapi", "latest python", "latest react",
                "release date of", "current exchange rate"
            )
        )
        if is_web_query:
            return MasterAnalysis(
                intent="question_answering",
                domain="web_research",
                goal=f"Retrieve fresh/current web information for: {text[:80]}",
                workspace_needed=False,
                workspace_reason="Web query does not require workspace inspection",
                files_needed=False,
                memory_needed=False,
                tools_needed=True,
                coding_needed=False,
                vision_needed=False,
                document_processing_needed=False,
                web_needed=True,
                artifact_required=False,
                reasoning_complexity=ReasoningComplexity.EASY,
                required_capabilities=["web_search", "chat"],
                recommended_model_role="Research & Search Specialist",
                execution_mode=ExecutionMode.TOOL_EXECUTION,
                confidence=0.95,
            )

        # 4. Explanations (without create/build intent)
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

        # 3. Edit existing file / modify code in workspace
        is_edit = (
            any(
                v in lowered
                for v in (
                    "edit", "modify", "update", "change the", "refactor", "patch",
                    "add to", "add a", "in the index.html", "in the file", "in index.html",
                    "add to the", "add a route to", "add an endpoint to", "in the fastapi",
                    "in the existing", "in existing file", "to the fastapi code",
                    "to the code", "in the code", "add if successful", "print successful",
                    "add a button", "add a learn more", "button"
                )
            )
            or ("in " in lowered and (".html" in lowered or ".py" in lowered or ".js" in lowered or ".css" in lowered))
        )
        if is_edit:
            return MasterAnalysis(
                intent="coding",
                domain="software_development",
                goal=f"Edit existing file in workspace: {text[:80]}",
                workspace_needed=True,
                workspace_reason="Modifying existing workspace file",
                files_needed=True,
                memory_needed=True,
                tools_needed=True,
                coding_needed=True,
                vision_needed=False,
                document_processing_needed=False,
                web_needed=False,
                artifact_required=True,
                reasoning_complexity=ReasoningComplexity.EASY,
                required_capabilities=["coding", "file_creation", "file_editing"],
                recommended_model_role="Coding Specialist",
                execution_mode=ExecutionMode.EDIT_EXISTING,
                confidence=0.95,
            )

        # 4. Artifact generation / rich document creation
        is_create = any(
            v in lowered
            for v in ("create", "generate", "build", "make", "implement", "scaffold", "develop", "write a", "write me", "write the", "write to", "save to", "dump to", "save it as", "produce")
        )
        is_web = any(
            w in lowered
            for w in ("html", "css", "landing page", "website", "web page", "frontend", "web app", "site", "javascript", "js")
        )
        is_doc = any(
            w in lowered
            for w in (
                ".pdf", "pdf", ".docx", "docx", "word doc", "word document",
                ".pptx", "pptx", "powerpoint", "presentation", "slides",
                ".xlsx", "xlsx", "excel", "spreadsheet", ".csv", "csv",
                "report", "slideshow"
            )
        )
        is_explicit_workspace = any(
            w in lowered
            for w in ("in my project", "in my workspace", "in this repository", "in existing project", "project overview")
        )

        if (is_create and (is_web or is_doc or "file" in lowered or "script" in lowered or "app" in lowered or "output." in lowered or "readme" in lowered or "doc" in lowered or "test" in lowered or "index.html" in lowered or ".py" in lowered or ".js" in lowered or ".css" in lowered)) or (is_doc and any(v in lowered for v in ("create", "generate", "build", "make", "write", "produce", "save"))):
            if is_doc:
                domain = "document_generation"
                caps = ["document_generation", "coding", "file_creation"]
                role = "Document & Artifact Specialist"
            elif is_web:
                domain = "web_development"
                caps = ["coding", "html", "css", "code_generation", "file_creation"]
                role = "Coding Specialist"
            else:
                domain = "software_development"
                caps = ["coding", "file_creation"]
                role = "Coding Specialist"
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
                document_processing_needed=is_doc,
                web_needed=False,
                artifact_required=True,
                reasoning_complexity=ReasoningComplexity.EASY,
                required_capabilities=caps,
                recommended_model_role=role,
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
                raw = self._call_master(
                    selection,
                    prompt,
                    retry_error=failures[-1] if failures else None,
                )
                plan = self._to_plan(raw, prompt)
                validate_plan_dag(plan)
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

    def _call_master(
        self,
        selection: MasterModelSelection,
        prompt: str,
        retry_error: str | None = None,
    ) -> str:
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

        repair_prompt = ""
        if retry_error:
            repair_prompt = (
                f"\n\nCRITICAL FIX: Your previous response was rejected due to error:\n"
                f"'{retry_error}'\n"
                "Please output strictly valid JSON conforming to the TaskDecompositionPlan schema "
                "with non-empty prompts, valid task IDs, and no cycles."
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
            f"{repair_prompt}"
        )

        request = ChatRequest(
            messages=[
                ChatMessage(role="system", content=_DIVIDER_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_content),
            ],
            temperature=self._temperature,
            max_tokens=768,
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