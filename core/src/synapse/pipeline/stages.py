"""Pipeline stages — defines the different types of work a pipeline can perform.

Each stage type:
    - Has specific capability requirements
    - Has a preferred model type
    - Has a prompt template
    - Can produce tool calls
    - Has validation rules

Stage types:
    - PLANNING: decompose task, design architecture
    - CODING: write code, implement features
    - REVIEW: review code, find issues
    - DOCUMENTATION: write docs, comments
    - RESEARCH: analyze requirements, gather info
    - SYNTHESIS: combine outputs into final response
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from synapse.domain import Capability


class StageType(str, Enum):
    """Types of pipeline stages."""

    PLANNING = "planning"
    CODING = "coding"
    REVIEW = "review"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    SYNTHESIS = "synthesis"
    TOOL_EXECUTION = "tool_execution"


@dataclass
class StageDefinition:
    """Defines a pipeline stage type with its requirements."""

    stage_type: StageType
    required_capabilities: list[Capability]
    preferred_capabilities: list[Capability] = field(default_factory=list)
    description: str = ""
    prompt_template: str = ""
    max_tokens: int = 4000
    temperature: float = 0.7
    produces_files: bool = False
    can_use_tools: bool = False


# Stage definitions with capability requirements
STAGE_DEFINITIONS: dict[StageType, StageDefinition] = {
    StageType.PLANNING: StageDefinition(
        stage_type=StageType.PLANNING,
        required_capabilities=[Capability.REASONING, Capability.PLANNING],
        preferred_capabilities=[Capability.ARCHITECTURE],
        description="Decompose task and design approach",
        prompt_template="""You are a software architect. Analyze the task and create a plan.

Task: {task}

Context:
{context}

Provide:
1. Problem analysis
2. Required files/components
3. Implementation steps
4. Dependencies between steps

Output a structured plan.""",
        max_tokens=2000,
        temperature=0.3,
    ),
    StageType.CODING: StageDefinition(
        stage_type=StageType.CODING,
        required_capabilities=[Capability.CODING],
        preferred_capabilities=[Capability.DEBUGGING, Capability.ARCHITECTURE],
        description="Implement code changes",
        prompt_template="""You are an expert programmer. Implement the requested code changes.

Task: {task}

Plan from previous stage:
{previous_output}

Workspace context:
{workspace_context}

Existing files to consider:
{file_contents}

Write clean, production-ready code. Use the write_file tool to create/modify files.""",
        max_tokens=4000,
        temperature=0.3,
        produces_files=True,
        can_use_tools=True,
    ),
    StageType.REVIEW: StageDefinition(
        stage_type=StageType.REVIEW,
        required_capabilities=[Capability.REASONING, Capability.CODING],
        preferred_capabilities=[Capability.DEBUGGING],
        description="Review code for issues",
        prompt_template="""You are a senior code reviewer. Review the implementation.

Original task: {task}

Code written:
{previous_output}

Files modified:
{files_modified}

Check for:
1. Bugs or logic errors
2. Security issues
3. Performance problems
4. Code style violations
5. Missing edge cases

Output issues found and suggested fixes.""",
        max_tokens=2000,
        temperature=0.3,
    ),
    StageType.DOCUMENTATION: StageDefinition(
        stage_type=StageType.DOCUMENTATION,
        required_capabilities=[Capability.WRITING],
        preferred_capabilities=[Capability.CODING],
        description="Write documentation",
        prompt_template="""You are a technical writer. Document the implementation.

Task: {task}

Implementation:
{previous_output}

Write clear, concise documentation including:
1. Purpose and functionality
2. Usage examples
3. API changes (if any)
4. Configuration options

Output the documentation content.""",
        max_tokens=2000,
        temperature=0.5,
        produces_files=True,
        can_use_tools=True,
    ),
    StageType.RESEARCH: StageDefinition(
        stage_type=StageType.RESEARCH,
        required_capabilities=[Capability.REASONING],
        preferred_capabilities=[Capability.PLANNING],
        description="Research and analyze",
        prompt_template="""You are a researcher. Analyze the requirements and gather context.

Task: {task}

Workspace files:
{workspace_context}

Provide:
1. Current state analysis
2. Relevant patterns in codebase
3. Potential challenges
4. Recommended approach

Output your analysis.""",
        max_tokens=2000,
        temperature=0.5,
    ),
    StageType.SYNTHESIS: StageDefinition(
        stage_type=StageType.SYNTHESIS,
        required_capabilities=[Capability.CHAT, Capability.WRITING],
        preferred_capabilities=[Capability.REASONING],
        description="Combine outputs into final response",
        prompt_template="""You are a helpful assistant. Synthesize the work done into a clear response.

Original request: {task}

Work completed:
{previous_output}

Files changed:
{files_summary}

Provide a concise summary of what was done, what files were created/modified,
and any important notes for the user.""",
        max_tokens=1500,
        temperature=0.7,
    ),
}


@dataclass
class PipelineStage:
    """An instance of a stage in the pipeline."""

    id: str
    definition: StageDefinition
    task_description: str
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"
    result: Any = None

    @property
    def stage_type(self) -> StageType:
        return self.definition.stage_type

    @property
    def required_capabilities(self) -> list[Capability]:
        return self.definition.required_capabilities

    def format_prompt(
        self,
        context: dict[str, str],
        previous_output: str = "",
        file_contents: dict[str, str] | None = None,
        files_summary: str = "",
    ) -> str:
        """Format the prompt template with context values."""
        template = self.definition.prompt_template
        values = {
            "task": self.task_description,
            "context": context.get("workspace_brief", ""),
            "previous_output": previous_output,
            "workspace_context": context.get("workspace_brief", ""),
            "file_contents": self._format_files(file_contents or {}),
            "files_modified": files_summary,
            "files_summary": files_summary,
        }
        try:
            return template.format(**values)
        except KeyError:
            # Fallback: just return task + context
            return f"{self.task_description}\n\n{context.get('workspace_brief', '')}"

    @staticmethod
    def _format_files(files: dict[str, str]) -> str:
        if not files:
            return "No files loaded."
        parts = []
        for path, content in list(files.items())[:5]:
            truncated = content[:1000] + "..." if len(content) > 1000 else content
            parts.append(f"--- {path} ---\n{truncated}")
        return "\n\n".join(parts)
