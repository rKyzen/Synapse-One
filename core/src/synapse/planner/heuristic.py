"""Task Planner — deterministic heuristic prompt decomposition.

Splits a large prompt into smaller tasks, classifies each task's role
(coding, writing, math, ...) and capability needs, and wires them into a DAG
with an explicit synthesis task when the prompt decomposes. Deterministic and
offline by design — a model-backed planner can replace this behind the same
contract later.

Phase 6 additions:

- **file generation** — prompts that ask to *create / write / generate /
  refactor* files get ``Task.file_output = True`` (plus a ``file_hint`` and a
  ``model_hint`` from config) so the orchestrator knows to expect a structured
  file manifest from the chosen model and apply it to the workspace.
- **review step** — when a request produces files, a ``TaskKind.REVIEW`` task
  is appended that depends on every file task; the orchestrator routes it to
  the strongest available reasoning model ("Review → strongest model").
"""

from __future__ import annotations

import re
import uuid

import structlog

from synapse.contracts import TaskPlanner
from synapse.domain import (
    Capability,
    ComplexityResult,
    Decision,
    IntentResult,
    PrivacyResult,
)
from synapse.domain.enums import TaskKind
from synapse.domain.tasks import Task, TaskDAG

log = structlog.get_logger("synapse.planner")

#: task kind -> (required capabilities, preferred capabilities)
_TASK_PROFILES: dict[TaskKind, tuple[list[Capability], list[Capability]]] = {
    TaskKind.PLANNING: ([Capability.REASONING, Capability.PLANNING], [Capability.CHAT]),
    TaskKind.REASONING: ([Capability.REASONING], [Capability.CHAT, Capability.PLANNING]),
    TaskKind.CODING: ([Capability.CODING, Capability.REASONING], [Capability.DEBUGGING, Capability.ARCHITECTURE, Capability.JSON, Capability.TERMINAL, Capability.CHAT]),
    TaskKind.WRITING: ([Capability.WRITING], [Capability.CHAT, Capability.TRANSLATION, Capability.PLANNING]),
    TaskKind.MATH: ([Capability.MATH, Capability.REASONING], [Capability.CHAT]),
    TaskKind.RESEARCH: ([Capability.REASONING], [Capability.CHAT, Capability.PLANNING, Capability.TOOLS]),
    TaskKind.GENERAL: ([Capability.CHAT], [Capability.WRITING]),
    TaskKind.SYNTHESIS: ([Capability.CHAT], [Capability.WRITING, Capability.REASONING]),
    TaskKind.REVIEW: ([Capability.REASONING], [Capability.CHAT, Capability.JSON, Capability.CODING]),
}

#: keyword -> kind used to classify sub-prompts (matched on lowercase text).
_KIND_HINTS: list[tuple[TaskKind, tuple[str, ...]]] = [
    (TaskKind.PLANNING, ("plan", "roadmap", "schedule", "timeline", "organize", "steps to", "outline to", "architecture of")),
    (TaskKind.CODING, ("code", "script", "function", "api", "program", "debug", "implement", "refactor", "algorithm", "bug", "build", "flutter", "python", "javascript", "repository", "website", "sql", "endpoint", "class ", "method")),
    (TaskKind.MATH, ("calculate", "math", "equation", "sum of", "product of", "probability", "derivative", "integral", "solve")),
    (TaskKind.WRITING, ("write", "essay", "email", "letter", "story", "poem", "draft", "summarize", "summarise", "rewrite", "proofread", "translate", "document")),
    (TaskKind.RESEARCH, ("research", "study", "analyze", "investigate", "survey", "compare", "evidence", "findings", "literature", "review")),
    (TaskKind.REASONING, ("explain", "why", "reason", "think through", "logic", "what if", "prove", "infer", "justify", "evaluate")),
]

_SYNTHESIS_INSTRUCTION = "Combine the results of the tasks above into one cohesive, complete final answer for the user."

_REVIEW_INSTRUCTION = (
    "Review the generated files for correctness and internal consistency. "
    "Check that all referenced files exist, syntax is valid, and the pieces fit "
    "together. Report concrete issues, if any, and confirm what is good."
)

#: Patterns that mark a prompt as expecting filesystem output (Phase 6/7).
_FILE_TASK_PATTERNS: list[tuple[str, str]] = [
    # "create/build/make/write/generate" + (a/an) + artifact or language
    (r"\b(create|build|make|write|generate|scaffold|design)\s+(an?\s+)?(?:(?:[\w\-]+\s+){0,2})(website|web\s*app|web\s*page|site|portfolio|app|application|script|program|tool|project|cli|dashboard|game|plugin|extension|module|package|class|function|api|endpoint|unit\s+test|python|javascript|typescript|html|css|bash|shell|go|rust|java|c\+\+|ruby|php|sql|react|vue|flask|django|fastapi|store|shop|ecommerce|blog|forum|cms|chatbot|bot|calculator|todo\s*list\s*app|todo\s*app|todo\s*list|landing\s*page|wiki|timer|website)\b", ""),
    (r"\b(write|generate|create|make)\s+(a\s+|an\s+)?(?:[\w\-]+\s+)?(readme|documentation|doc\b|docs|markdown|manual|guide|changelog)\b", "md"),
    (r"\b(generate|create|write|make)\s+(files?|codebase|project|scaffold|structure)\b", ""),
    (r"\brefactor\b", ""),
    (r"\b(generate|create|write)\s+documentation\b", "md"),
    # Phase 7 — modifications target existing files and expect a manifest too.
    (r"\b(edit|update|modify|change|rewrite|improve|adjust|tweak|patch|redesign|restyle)\s+(the\s+)?(code|file|script|website|web\s*app|app|function|class|test|readme|css|html|style|color|layout|design|theme|[a-z0-9_.\-/]+\.[a-z0-9]+)\b", ""),
    (r"\bfix\s+(the\s+)?(bug|code|script|website|web\s*app|app|file|function|error)\b", ""),
    (r"\bmake\s+my\s+(website|web\s*app|app|script|code|project|file|readme)\b", ""),
    (r"\b(add|remove|change)\s+(a\s+)?(feature|button|color|style|dark\s*mode|light\s*mode|page|section|function|method|test|route|navbar)\b", ""),
    # "save/write ... to <file.ext>" — explicit destination, any kind of task
    # (research/analysis/chat included). Extension required so chat phrases
    # like "write a blog post to my site" never match.
    (r"\b(save|write|dump|export|output)\s+.*?\bto\b\s+([A-Za-z0-9_.\-/]+\.[a-z0-9]+)\b", ""),
]

#: extension hint derived from the prompt, when unambiguous.
_EXT_HINT_RE = re.compile(
    r"\b([A-Za-z0-9_\-]+\.(?:py|js|ts|html|css|json|md|sh|toml|yaml|yml|sql))\b"
)

#: model hints per task kind (override via config ``planner.model_hints``).
#: REVIEW is resolved dynamically to the strongest available reasoning model.
_DEFAULT_MODEL_HINTS: dict[str, str] = {}

#: a standalone part that is only a file name (``script.js``, ``src/app.ts``)
#: — the residue of splitting "with index.html, style.css and script.js".
_BARE_PATH_RE = re.compile(r"[A-Za-z0-9_.\-/]+(?:\.[A-Za-z0-9]+)")

#: separators that indicate a multi-part prompt (in priority order).
_SPLIT_PATTERNS: list[tuple[str, str]] = [
    # numbered list: "1. ... 2. ..."
    (r"\n\s*\d+[\.\)]\s+", r"(?:\n\s*\d+[\.\)]\s+)"),
    # "First ... Then ..."
    (r"\s+then\s+", r"\s+then\s+"),
    # newline-separated requests
    (r"\n+", r"\n+"),
    # explicit " and " between imperative clauses (short prompts only)
    (r"\s+and\s+", r"\s+and\s+"),
]


class HeuristicTaskPlanner(TaskPlanner):
    """Deterministic decomposition planner with per-task capability profiles."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        min_split_complexity: int = 40,
        max_tasks: int = 6,
        min_part_length: int = 8,
        file_output_enabled: bool = True,
        model_hints: dict[str, str] | None = None,
        add_review: bool = True,
    ) -> None:
        self._enabled = enabled
        self._min_split_complexity = min_split_complexity
        self._max_tasks = max_tasks
        self._min_part_length = min_part_length
        self._file_output_enabled = file_output_enabled
        self._model_hints = dict(_DEFAULT_MODEL_HINTS)
        self._model_hints.update(model_hints or {})
        self._add_review = add_review

    # -- contract -----------------------------------------------------------

    def plan(
        self,
        prompt: str,
        intent: IntentResult,
        complexity: ComplexityResult,
        privacy: PrivacyResult,
        decision: Decision,
    ) -> TaskDAG:
        if not self._enabled:
            return self._single_task(prompt, decision)
        parts = self._split(prompt, complexity.score)
        # Bare path fragments ("script.js" from "with index.html and style.css
        # and script.js") are absorbed into the previous task — they are not
        # standalone tasks, and letting them through would leak generated code
        # into the chat synthesis as a non-file task.
        parts = self._absorb_fragments(parts)
        if len(parts) <= 1:
            return self._single_task(prompt, decision)

        tasks = [self._task_for_part(f"t{i+1}", part) for i, part in enumerate(parts)]
        tasks = tasks[: self._max_tasks]
        has_file_output = any(t.file_output for t in tasks)
        if has_file_output and self._add_review:
            review = Task(
                id="t-review",
                kind=TaskKind.REVIEW,
                description=_REVIEW_INSTRUCTION,
                required_capabilities=list(_TASK_PROFILES[TaskKind.REVIEW][0]),
                preferred_capabilities=list(_TASK_PROFILES[TaskKind.REVIEW][1]),
                depends_on=[t.id for t in tasks if t.file_output],
            )
            tasks.append(review)
        if len(tasks) > 1:
            synthesis = Task(
                id="t-synthesis",
                kind=TaskKind.SYNTHESIS,
                description=_SYNTHESIS_INSTRUCTION,
                required_capabilities=list(_TASK_PROFILES[TaskKind.SYNTHESIS][0]),
                preferred_capabilities=list(_TASK_PROFILES[TaskKind.SYNTHESIS][1]),
                depends_on=[t.id for t in tasks],
            )
            tasks.append(synthesis)

        dag = TaskDAG(tasks=tasks)
        dag.topological_order()  # validate acyclicity up front
        log.info(
            "task_plan",
            task_count=len(tasks),
            complexity=complexity.score,
            intent=intent.primary.value,
            file_output=has_file_output,
        )
        return dag

    # -- decomposition ------------------------------------------------------

    def _split(self, prompt: str, complexity: int) -> list[str]:
        if complexity < self._min_split_complexity:
            return [prompt]
        text = prompt.strip()
        for _, pattern in _SPLIT_PATTERNS:
            if re.search(pattern, text):
                parts = [p.strip() for p in re.split(pattern, text) if p.strip()]
                if len(parts) >= 2:
                    return self._merge_small(parts)
        return [text]

    def _merge_small(self, parts: list[str]) -> list[str]:
        """Drop near-empty parts and merge tiny fragments into the previous one."""
        merged: list[str] = []
        for part in parts:
            if len(part) < self._min_part_length:
                if merged:
                    merged[-1] = f"{merged[-1]} {part}".strip()
                continue
            merged.append(part)
        return merged or [parts[0]]

    @staticmethod
    def _absorb_fragments(parts: list[str]) -> list[str]:
        """Merge bare-path parts (``script.js``) back into the previous task."""
        merged: list[str] = []
        for part in parts:
            if merged and _BARE_PATH_RE.fullmatch(part):
                merged[-1] = f"{merged[-1]} {part}".strip()
            else:
                merged.append(part)
        return merged

    def _task_for_part(self, task_id: str, part: str) -> Task:
        kind = self._classify(part)
        required, preferred = _TASK_PROFILES[kind]
        file_output, file_hint = self._file_task_signal(part)
        if file_output and kind == TaskKind.GENERAL:
            kind = TaskKind.CODING
            required, preferred = _TASK_PROFILES[kind]
        if file_output:
            # Phase X — a file-writing agent is routed tools- and JSON-aware
            # (manifest output) even for research/reasoning/analysis tasks,
            # while pure chat tasks keep their normal routing weights.
            preferred = self._with_tool_caps(preferred)
        return Task(
            id=task_id,
            kind=kind,
            description=part,
            required_capabilities=list(required),
            preferred_capabilities=list(preferred),
            file_output=file_output,
            file_hint=file_hint,
            model_hint=self._model_hints.get(kind.value),
        )

    def _file_task_signal(self, part: str) -> tuple[bool, str]:
        """Is this part a file-generation request? Returns (flag, ext hint)."""
        if not self._file_output_enabled:
            return False, ""
        lowered = part.lower()
        hit = any(re.search(pattern, lowered) for pattern, _ in _FILE_TASK_PATTERNS)
        hint = ""
        if hit:
            m = _EXT_HINT_RE.search(part)
            if m:
                hint = m.group(1)
            else:
                known = [
                    (r"\b(python|\.py)\b", "py"), (r"\b(html)\b", "html"),
                    (r"\b(css)\b", "css"), (r"\b(javascript|\.js)\b", "js"),
                    (r"\b(typescript|\.ts)\b", "ts"), (r"\b(readme|documentation|markdown)\b", "md"),
                    (r"\b(bash|shell)\b", "sh"), (r"\b(json)\b", "json"),
                    (r"\b(golang|go)\b", "go"), (r"\b(rust)\b", "rs"),
                ]
                for pattern, ext in known:
                    if re.search(pattern, lowered):
                        hint = f"output.{ext}"
                        break
        return hit, hint

    @staticmethod
    def _with_tool_caps(preferred: list[Capability]) -> list[Capability]:
        """Add TOOLS + JSON preference so file-writing agents route to
        tool- and manifest-capable models without excluding any model."""
        caps = list(preferred)
        for cap in (Capability.TOOLS, Capability.JSON):
            if cap not in caps:
                caps.append(cap)
        return caps

    @staticmethod
    def _classify(part: str) -> TaskKind:
        lowered = part.lower()
        for kind, hints in _KIND_HINTS:
            if any(hint in lowered for hint in hints):
                return kind
        return TaskKind.GENERAL

    def _single_task(self, prompt: str, decision: Decision) -> TaskDAG:
        kind = TaskKind.GENERAL
        lowered = prompt.lower()
        if any(hint in lowered for hint in _KIND_HINTS[1][1]):
            kind = TaskKind.CODING
        elif any(hint in lowered for hint in _KIND_HINTS[4][1]):
            kind = TaskKind.WRITING
        elif any(hint in lowered for hint in _KIND_HINTS[3][1]):
            kind = TaskKind.RESEARCH

        required, preferred = _TASK_PROFILES[kind]
        file_output, file_hint = self._file_task_signal(prompt)
        # File-producing prompts must never route as a general chat task:
        # generation needs a coding-capable model, docs a writing-capable one.
        if file_output:
            if kind == TaskKind.GENERAL and file_hint == "":
                kind = TaskKind.CODING
            if any(
                hint in lowered
                for hint in ("readme", "documentation", "markdown", "manual", "guide", "changelog")
            ):
                kind = TaskKind.WRITING
            required, preferred = _TASK_PROFILES[kind]
            preferred = self._with_tool_caps(preferred)
        tasks: list[Task] = [
            Task(
                id="t1",
                kind=kind,
                description=prompt,
                required_capabilities=list(required),
                preferred_capabilities=list(preferred),
                file_output=file_output,
                file_hint=file_hint,
                model_hint=self._model_hints.get(kind.value),
            )
        ]
        if file_output and self._add_review:
            tasks.append(
                Task(
                    id="t-review",
                    kind=TaskKind.REVIEW,
                    description=_REVIEW_INSTRUCTION,
                    required_capabilities=list(_TASK_PROFILES[TaskKind.REVIEW][0]),
                    preferred_capabilities=list(_TASK_PROFILES[TaskKind.REVIEW][1]),
                    depends_on=["t1"],
                )
            )
        return TaskDAG(tasks=tasks)