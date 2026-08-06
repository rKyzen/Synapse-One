"""Phase 4: Context Builder contract.

Builds smart prompts from relevant information only:
- Current task
- Relevant memories
- Retrieved documents
- Recent conversation
- User preferences

Only includes information that is actually relevant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from synapse.domain.enums import MemoryScope


@dataclass(frozen=True)
class ContextBundle:
    """A constructed context bundle for a task."""

    system_prompt: str
    user_prompt: str
    # Metadata for logging/debugging
    memory_entries_used: int = 0
    retrieval_chunks_used: int = 0
    conversation_turns_used: int = 0
    workspace_files_used: int = 0
    total_chars: int = 0


class ContextBuilder(ABC):
    """Constructs smart, relevant prompts for model execution."""

    @abstractmethod
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
        max_context_chars: int = 8000,
    ) -> ContextBundle:
        """Build a context-aware prompt bundle.

        Args:
            prompt: The user's current prompt
            task_capabilities: Required capabilities for this task
            memory_scope: Memory scope to search (conversation/project/global)
            memory_results: Pre-retrieved memory entries
            retrieval_chunks: Retrieved document chunks from workspace
            conversation_history: Recent conversation turns
            workspace_outcome: Workspace preparation outcome
            user_preferences: User preferences for context building
            max_context_chars: Maximum characters in final context

        Returns:
            ContextBundle with system_prompt, user_prompt, and metadata
        """


class ContextPolicy(ABC):
    """Policy for what context to include and how to weight it."""

    @abstractmethod
    def should_include_memory(self, entry: dict, task_capabilities: list[str]) -> bool:
        """Whether a memory entry is relevant."""

    @abstractmethod
    def should_include_retrieval(self, chunk: dict, task_capabilities: list[str]) -> bool:
        """Whether a retrieval chunk is relevant."""

    @abstractmethod
    def weight_context(self, context_type: str, task_capabilities: list[str]) -> float:
        """Weight for a context type (0.0 - 1.0)."""