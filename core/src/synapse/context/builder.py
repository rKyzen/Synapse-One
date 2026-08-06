"""Default Context Builder — Phase 4.

Constructs focused system/user prompts from only the information that is
actually relevant to the task: memories, retrieval chunks, conversation
history, workspace outcome, and user preferences. Everything is bounded by a
character budget so context stays within the model's window.
"""

from __future__ import annotations

from synapse.contracts.context import ContextBundle, ContextBuilder, ContextPolicy
from synapse.domain.enums import MemoryScope

_SYSTEM_BASE = (
    "You are Synapse, a local-first AI assistant. "
    "Answer the user's request using the provided context. "
    "If the context is insufficient, say so explicitly — never fabricate."
)


class DefaultContextPolicy(ContextPolicy):
    """Relevance policy: only include entries whose content overlaps the task.

    A cheap lexical relevance check (shared keywords) keeps the prompt lean:
    if an entry shares none of the task's content words, it is dropped.
    """

    def should_include_memory(self, entry: dict, task_capabilities: list[str]) -> bool:
        del task_capabilities
        return _relevant(entry.get("text", ""), entry.get("query", ""))

    def should_include_retrieval(self, chunk: dict, task_capabilities: list[str]) -> bool:
        del task_capabilities
        score = chunk.get("score")
        if score is not None:
            try:
                return float(score) >= 0.3
            except (TypeError, ValueError):
                pass
        return bool(chunk.get("text", "").strip())

    def weight_context(self, context_type: str, task_capabilities: list[str]) -> float:
        del task_capabilities
        weights = {
            "user_preferences": 1.0,
            "conversation": 0.8,
            "memory": 0.7,
            "retrieval": 0.9,
            "workspace": 0.6,
        }
        return weights.get(context_type, 0.5)


def _relevant(text: str, query: str) -> bool:
    """True when ``text`` and ``query`` share at least one content word."""
    if not query:
        return True
    import re

    text_words = {w for w in re.findall(r"[a-z]{3,}", text.lower())}
    query_words = {w for w in re.findall(r"[a-z]{3,}", query.lower())}
    return bool(text_words & query_words)


class DefaultContextBuilder(ContextBuilder):
    """Builds prompts with a strict character budget, most-relevant first."""

    def __init__(self, policy: ContextPolicy | None = None, max_context_chars: int = 8000) -> None:
        self._policy = policy or DefaultContextPolicy()
        self._max_context_chars = max_context_chars

    def build(
        self,
        prompt: str,
        *,
        task_capabilities: list[str] | None = None,
        memory_scope: MemoryScope | None = None,
        memory_results: list[dict] | None = None,
        retrieval_chunks: list[dict] | None = None,
        conversation_history: list[dict] | None = None,
        workspace_outcome: dict | None = None,
        user_preferences: dict | None = None,
        max_context_chars: int | None = None,
    ) -> ContextBundle:
        capabilities = task_capabilities or []
        effective_max = max_context_chars if max_context_chars is not None else self._max_context_chars
        memory_entries = self._filtered(memory_results, capabilities, query=prompt)
        chunks = self._filtered(retrieval_chunks, capabilities, query=prompt)
        history = self._filtered(conversation_history, capabilities, text_key="content")

        parts: list[str] = [prompt]
        counts = {
            "memory_entries_used": len(memory_entries),
            "retrieval_chunks_used": len(chunks),
            "conversation_turns_used": len(history),
            "workspace_files_used": 0,
        }

        if workspace_outcome:
            vision = workspace_outcome.get("vision_descriptions")
            if vision:
                parts.append("Vision descriptions of attached images:\n" + "\n".join(str(v) for v in vision))
            counts["workspace_files_used"] = workspace_outcome.get("file_count", 0)

        if user_preferences:
            lines = [f"- {k}: {v}" for k, v in user_preferences.items() if isinstance(v, (str, int, float, bool))]
            if lines:
                parts.append("User preferences:\n" + "\n".join(lines))

        if history:
            turns = "\n".join(f"{h.get('role', 'user')}: {str(h.get('content', ''))[:500]}" for h in history)
            parts.append("Recent conversation:\n" + turns)

        if memory_entries:
            items = "\n".join(f"- {e.get('text', '')[:400]}" for e in memory_entries)
            parts.append("Relevant memories:\n" + items)

        if chunks:
            items = []
            for i, c in enumerate(chunks, start=1):
                file_name = c.get("file_name", "")
                items.append(f"[{i}] {c.get('text', '')[:600]}" + (f" (from {file_name})" if file_name else ""))
            parts.append("Retrieved knowledge:\n" + "\n".join(items))

        body = "\n\n".join(parts)
        if effective_max > 0 and len(body) > effective_max:
            body = body[:effective_max].rsplit("\n\n", 1)[0] + "\n…(context truncated)"

        return ContextBundle(
            system_prompt=_SYSTEM_BASE,
            user_prompt=body,
            memory_entries_used=counts["memory_entries_used"],
            retrieval_chunks_used=counts["retrieval_chunks_used"],
            conversation_turns_used=counts["conversation_turns_used"],
            workspace_files_used=counts["workspace_files_used"],
            total_chars=len(body),
        )

    def _filtered(
        self,
        entries: list[dict] | None,
        capabilities: list[str],
        text_key: str = "text",
        query: str | None = None,
    ) -> list[dict]:
        if not entries:
            return []
        policy = self._policy
        selected: list[dict] = []
        for entry in entries:
            if text_key == "content":
                include = policy.should_include_memory(entry, capabilities)
            else:
                include = policy.should_include_retrieval(entry, capabilities)
                if include and query:
                    include = _relevant(entry.get("text", ""), query)
            if include:
                selected.append(entry)
        return selected[:10]
