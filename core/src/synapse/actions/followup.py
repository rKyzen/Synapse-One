"""Follow-up intent resolution — conversational wording, workspace intent.

Phase XVII. The Intent Router is context-free on purpose: it classifies a
single prompt with no knowledge of the conversation or the project. That is
correct for greetings and general questions, but inside an active project a
message that is worded conversationally is often a continuation of the
workspace work. "fix the bugs", "continue", "change the UI", "add search" and
"why isn't this working?" carry no explicit file or artifact target, so the
router alone falls through to CONVERSATION / QUESTION_ANSWERING and the
request is answered in chat: no planner, no tools, no files.

This module resolves that gap deterministically and offline. It never
replaces the router — it only answers two questions:

1. ``is_follow_up`` — does the wording read as a continuation of project
   work (continuation verbs, bug/error reports, project modification verbs,
   deictic references to project artifacts, standalone deictic pronouns)?
2. ``has_workspace_context`` — does the actual state prove project activity
   (prior conversation turns, prior recorded workspace actions, or existing
   project files)?

When both are true the Master overrides the chat-only verdict and routes the
request through the planner and the file tools with full workspace context.
Nothing here is a model call; a failing store never changes routing.
"""

from __future__ import annotations

import re

from synapse.domain.enums import MemoryScope

#: conversational wording that only makes sense as a continuation of the
#: active project's work. Priority-ordered alternatives, first match wins:
#: 1) explicit continuation / iteration; 2) bug / error / failure reporting;
#: 3) project modification / improvement verbs; 4) deictic references to a
#: project artifact; 5) add/remove a concrete thing (a feature-like noun,
#: never a number or connector); 6) a standalone deictic pronoun.
_FOLLOW_UP_RE = re.compile(
    r"(?:"
    r"\b(?:continue|keep\s+going|keep\s+working|keep\s+it\s+up|go\s+on|proceed|"
    r"carry\s+on|same\s+(?:thing|idea|task)|do\s+it|do\s+that|go\s+ahead|"
    r"once\s+more|again|retry|redo|rerun|re-?run|restart)\b"
    r"|\b(?:bug|bugs|broken|crash|crashing|crashes|error|errors|exception|"
    r"failing|failed|fails|doesn'?t(?:\s+[\w'-]+){0,2}\s+work\b|"
    r"does\s+not(?:\s+[\w'-]+){0,2}\s+work\b|"
    r"isn'?t(?:\s+[\w'-]+){0,2}\s+working\b|"
    r"is\s+not(?:\s+[\w'-]+){0,2}\s+working\b|not\s+working\b|malfunction|"
    r"(?<!no\s)(?<!nothing\s)(?:wrong|problem|glitch|issue))\b"
    r"|\b(?:change|update|modify|improve|enhance|adjust|tweak|redesign|restyle|"
    r"edit|rewrite|polish|clean\s+up|make\s+it\s+better|make\s+this|fix)\b"
    r"|\b(?:the|this|my|our|that)\s+(?:code|app|application|project|website|"
    r"web\s*app|script|file|files|test|tests|ui|interface|layout|design|feature|"
    r"function|class|module|readme|codebase|repo|repository)\b"
    r"|\b(?:add|remove)\s+(?:a\s+|an\s+|the\s+)?(?!\d|and\b|to\b|more\b|it\b|"
    r"them\b|one\b|two\b|three\b|four\b|five\b|six\b|seven\b|eight\b|nine\b|"
    r"ten\b)[\w-]{2,24}\b"
    r"|^(?:it|this|that|them|those|same)[.?!\s]*$"
    r")",
    re.IGNORECASE,
)


def is_follow_up(prompt: str) -> bool:
    """True when the wording reads as a continuation of project work.

    Deterministic and offline; never a model call. Greetings, thanks,
    general questions and social phrases do not match.
    """
    return bool(_FOLLOW_UP_RE.search(prompt or ""))


def has_workspace_context(
    *,
    memory=None,
    conversation_id: str | None = None,
    file_operator=None,
    action_log=None,
) -> bool:
    """True when the actual project state proves workspace activity.

    Signals, cheapest first, any of which is enough:

    - prior conversation turns scoped to THIS chat (the chat store history
      already lives in the conversation memory store);
    - prior recorded workspace actions for the project (the action log only
      receives entries from the workspace pipeline, never from pure chat);
    - existing project files (an opened workspace with content).

    Every probe is defensive: a failing store degrades to the remaining
    signals instead of changing routing.
    """
    if memory is not None and conversation_id:
        try:
            if memory.recent(MemoryScope.CONVERSATION, 1, conversation=conversation_id):
                return True
        except Exception:  # noqa: BLE001 - a failing store never changes routing
            pass
    if action_log is not None:
        try:
            if action_log.recent(1):
                return True
        except Exception:  # noqa: BLE001
            pass
    if file_operator is not None:
        try:
            if file_operator.list_tree():
                return True
        except Exception:  # noqa: BLE001
            pass
    return False
