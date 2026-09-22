"""GoalPlanner — goal decomposition into capability-driven steps (Phase B).

The AI Operating Workspace is goal-centric: the user states an outcome and
the planner produces an ordered, capability-labeled plan — WITHOUT naming any
model. Model selection happens later, at execution time, by the Router
(capabilities first, models second). Templates are keyed by workspace type:
a research workspace plans research-shaped steps, a developer workspace plans
implementation-shaped steps — coding is one template among many.

Phase B scope: plans are recorded onto the Goal (GoalStore.set_steps) and
executed step-by-step by the GoalExecutor (text deliverables). File-output
manifests remain the legacy /request path until Phase D unifies them.
"""

from __future__ import annotations

import re

from synapse.domain.enums import Capability
from synapse.domain.goals import GoalStep

#: workspace type -> [(step description template, capability, preferred caps)]
#: ``{outcome}`` is replaced by the (shortened) goal outcome at plan time.
GOAL_TEMPLATES: dict[str, list[tuple[str, str, list[str]]]] = {
    "general": [
        ("Understand what the goal requires: {outcome}", "reasoning", ["chat"]),
        ("Plan the steps to achieve: {outcome}", "planning", ["reasoning"]),
        ("Execute the core work for: {outcome}", "chat", ["writing"]),
        ("Review the work done for correctness and completeness", "reasoning", ["chat"]),
        ("Summarize the outcome of the goal", "writing", ["chat"]),
    ],
    "student": [
        ("Map the study plan for: {outcome}", "planning", ["reasoning"]),
        ("Gather the key materials and concepts for: {outcome}", "reasoning", ["tools"]),
        ("Summarize the core knowledge into clear notes", "writing", ["reasoning"]),
        ("Generate practice questions to test understanding", "reasoning", ["writing"]),
        ("Review answers and close knowledge gaps", "reasoning", ["chat"]),
    ],
    "research": [
        ("Survey existing knowledge and sources for: {outcome}", "reasoning", ["tools"]),
        ("Organize findings into themes and evidence", "reasoning", ["planning"]),
        ("Outline the structure of the deliverable", "planning", ["reasoning"]),
        ("Draft the deliverable from the evidence", "writing", ["reasoning"]),
        ("Verify claims and add citations", "reasoning", ["chat"]),
        ("Review and polish the final deliverable", "writing", ["chat"]),
    ],
    "business": [
        ("Research the context and constraints for: {outcome}", "reasoning", ["tools"]),
        ("Draft the business deliverable for: {outcome}", "writing", ["reasoning"]),
        ("Review risks, numbers, and feasibility", "reasoning", ["math"]),
        ("Finalize the deliverable for presentation", "writing", ["chat"]),
    ],
    "developer": [
        ("Understand the requirements behind: {outcome}", "reasoning", ["chat"]),
        ("Design the architecture and structure", "architecture", ["reasoning"]),
        ("Implement the solution", "coding", ["debugging", "terminal"]),
        ("Test and fix issues in the implementation", "debugging", ["coding"]),
        ("Review the work for correctness and consistency", "reasoning", ["coding"]),
        ("Document how to use and maintain it", "writing", ["chat"]),
    ],
    "writer": [
        ("Define the outline and angle for: {outcome}", "planning", ["reasoning"]),
        ("Draft the full piece", "writing", ["chat"]),
        ("Revise for structure, clarity, and flow", "writing", ["reasoning"]),
        ("Polish language, tone, and final details", "writing", ["chat"]),
    ],
    "designer": [
        ("Interpret the brief and constraints for: {outcome}", "reasoning", ["chat"]),
        ("Develop concept and visual direction", "reasoning", ["vision"]),
        ("Produce the visual assets and iterations", "vision", ["reasoning"]),
        ("Review against the brief and refine", "reasoning", ["vision"]),
        ("Prepare the final presentation of the work", "writing", ["chat"]),
    ],
    "teacher": [
        ("Define learning objectives for: {outcome}", "planning", ["reasoning"]),
        ("Draft the lesson plan structure", "writing", ["planning"]),
        ("Create the teaching materials", "writing", ["reasoning"]),
        ("Build assessment questions and answers", "writing", ["reasoning"]),
        ("Review the lesson for coherence and pacing", "reasoning", ["chat"]),
    ],
    "custom": [
        ("Understand what the goal requires: {outcome}", "reasoning", ["chat"]),
        ("Plan the steps to achieve: {outcome}", "planning", ["reasoning"]),
        ("Execute the core work for: {outcome}", "chat", ["writing"]),
        ("Review the work done for correctness and completeness", "reasoning", ["chat"]),
        ("Summarize the outcome of the goal", "writing", ["chat"]),
    ],
}

_VALID_CAPS = {c.value for c in Capability}


class GoalPlanner:
    """Decomposes a goal outcome into capability-labeled steps."""

    def plan(self, outcome: str, workspace_type: str = "general") -> list[GoalStep]:
        """Return the ordered plan for a goal in the given workspace type.

        ``workspace_type`` defaults to ``general`` for unknown/custom types —
        the planner never fails on an unfamiliar workspace.
        """
        template = GOAL_TEMPLATES.get(workspace_type, GOAL_TEMPLATES["general"])
        steps: list[GoalStep] = []
        for description, capability, preferred in template:
            cap = capability if capability in _VALID_CAPS else "reasoning"
            steps.append(
                GoalStep(
                    description=description.format(outcome=self._shorten(outcome, 140)),
                    capability=cap,
                    status="pending",
                )
            )
        return steps

    @staticmethod
    def _shorten(outcome: str, limit: int = 140) -> str:
        text = re.sub(r"\s+", " ", outcome or "").strip()
        return text if len(text) <= limit else text[:limit].rstrip() + "..."
