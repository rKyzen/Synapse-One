"""PipelineOrchestrator — manages the full multi-agent pipeline lifecycle.

Decomposes requests into stages, routes each to the best model, executes
sequentially or in parallel where possible, and synthesizes the final output.

Key features:
    - Automatic stage decomposition based on request complexity
    - Per-stage model routing with capability matching
    - Tool calling support for file operations
    - Progress tracking via activity feed
    - Context preservation across stages
    - Graceful error handling and recovery
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import structlog

from synapse.domain import Capability, Decision, HardwareProfile, RoutingDecision
from synapse.domain.models import ModelMetadata
from synapse.execution.executor import Executor, ProviderUnavailable
from synapse.pipeline.context import PipelineContext, StageResult
from synapse.pipeline.stages import (
    STAGE_DEFINITIONS,
    PipelineStage,
    StageType,
)
from synapse.pipeline.tools import ToolCall, ToolRegistry, parse_tool_calls
from synapse.router.router import Router
from synapse.workspace.brief import build_workspace_brief
from synapse.workspace.operator import FileOperator

log = structlog.get_logger("synapse.pipeline.orchestrator")


class PipelineOrchestrator:
    """Orchestrates multi-stage execution across multiple models."""

    def __init__(
        self,
        router: Router,
        executor: Executor,
        hardware: HardwareProfile,
        registry: list[ModelMetadata],
        provider_health: dict[str, bool],
        available_models: dict[str, set[str]] | None = None,
        performance: dict[str, dict] | None = None,
        file_operator: FileOperator | None = None,
        on_activity: Callable[[str, str], None] | None = None,
    ) -> None:
        self._router = router
        self._executor = executor
        self._hardware = hardware
        self._registry = registry
        self._health = provider_health
        self._available = available_models
        self._performance = performance
        self._on_activity = on_activity or (lambda *a: None)
        self._tool_registry = ToolRegistry(file_operator) if file_operator else None

    def decompose(self, prompt: str, context: PipelineContext) -> list[PipelineStage]:
        """Decompose a request into pipeline stages.

        For simple requests (single-model tasks), returns a single stage.
        For complex requests, returns multiple stages with dependencies.
        """
        # Check for explicit multi-step indicators
        is_complex = self._is_complex_request(prompt)

        if not is_complex:
            # Single-stage: determine the best type from the prompt
            stage_type = self._infer_stage_type(prompt)
            return [PipelineStage(
                id="stage_0",
                definition=STAGE_DEFINITIONS[stage_type],
                task_description=prompt,
            )]

        # Multi-stage decomposition
        stages = []

        # Always start with planning for complex tasks
        stages.append(PipelineStage(
            id="stage_0",
            definition=STAGE_DEFINITIONS[StageType.PLANNING],
            task_description=prompt,
        ))

        # Add coding stage
        stages.append(PipelineStage(
            id="stage_1",
            definition=STAGE_DEFINITIONS[StageType.CODING],
            task_description=prompt,
            dependencies=["stage_0"],
        ))

        # Add review stage
        stages.append(PipelineStage(
            id="stage_2",
            definition=STAGE_DEFINITIONS[StageType.REVIEW],
            task_description=prompt,
            dependencies=["stage_1"],
        ))

        # Add synthesis
        stages.append(PipelineStage(
            id="stage_3",
            definition=STAGE_DEFINITIONS[StageType.SYNTHESIS],
            task_description=prompt,
            dependencies=["stage_0", "stage_1", "stage_2"],
        ))

        return stages

    def execute(
        self,
        prompt: str,
        context: PipelineContext,
    ) -> PipelineContext:
        """Execute the full pipeline for a request."""
        self._on_activity("pipeline_start", "Planning approach...")
        stages = self.decompose(prompt, context)
        self._on_activity("pipeline_plan", f"Decomposed into {len(stages)} stages")

        # Execute stages in dependency order
        completed: dict[str, StageResult] = {}

        for stage in stages:
            # Wait for dependencies
            for dep_id in stage.dependencies:
                if dep_id not in completed:
                    self._on_activity("pipeline_wait", f"Waiting for {dep_id}...")
                    # In real implementation, this could be parallel
                    break

            # Execute stage
            self._on_activity(
                "stage_start",
                f"Executing {stage.definition.description}...",
            )

            result = self._execute_stage(stage, context, completed)
            completed[stage.id] = result
            context.add_stage_result(result)

            if not result.success:
                self._on_activity("stage_error", f"Stage failed: {result.error}")
                # Continue with other stages if possible
                continue

            self._on_activity("stage_complete", f"Completed {stage.definition.description}")

        # Extract final response from synthesis stage
        synthesis = completed.get("stage_3") or completed.get(f"stage_{len(stages)-1}")
        if synthesis:
            context.final_response = synthesis.content

        return context

    def _execute_stage(
        self,
        stage: PipelineStage,
        context: PipelineContext,
        previous_results: dict[str, StageResult],
    ) -> StageResult:
        """Execute a single pipeline stage."""
        # Build prompt with context
        previous_output = self._build_previous_output(previous_results, stage.dependencies)
        files_summary = self._build_files_summary(context)

        prompt = stage.format_prompt(
            context=context.to_dict(),
            previous_output=previous_output,
            file_contents=context.file_contents,
            files_summary=files_summary,
        )

        # Route to best model
        self._on_activity("routing", f"Selecting model for {stage.stage_type.value}...")
        routing = self._route_for_stage(stage)
        if not routing or not routing.provider_id:
            return StageResult(
                stage_id=stage.id,
                stage_type=stage.stage_type.value,
                model_id="",
                provider_id="",
                content="",
                success=False,
                error="No model available for this stage",
            )

        self._on_activity("model_selected", f"Using {routing.model_id}")

        # Execute with tool support if enabled
        if stage.definition.can_use_tools and self._tool_registry:
            return self._execute_with_tools(stage, routing, prompt, context)
        else:
            return self._execute_simple(stage, routing, prompt)

    def _execute_simple(
        self,
        stage: PipelineStage,
        routing: RoutingDecision,
        prompt: str,
    ) -> StageResult:
        """Execute a stage without tool support."""
        try:
            response = self._executor.execute(
                routing,
                prompt,
                temperature=stage.definition.temperature,
                max_tokens=stage.definition.max_tokens,
            )
            return StageResult(
                stage_id=stage.id,
                stage_type=stage.stage_type.value,
                model_id=routing.model_id,
                provider_id=routing.provider_id,
                content=response.content,
                success=True,
            )
        except Exception as exc:
            log.warning("stage_execution_failed", stage=stage.id, error=str(exc)[:200])
            return StageResult(
                stage_id=stage.id,
                stage_type=stage.stage_type.value,
                model_id=routing.model_id,
                provider_id=routing.provider_id,
                content="",
                success=False,
                error=str(exc),
            )

    def _execute_with_tools(
        self,
        stage: PipelineStage,
        routing: RoutingDecision,
        prompt: str,
        context: PipelineContext,
    ) -> StageResult:
        """Execute a stage with tool calling loop."""
        tool_schemas = self._tool_registry.get_schemas() if self._tool_registry else []
        tool_prompt = f"{prompt}\n\nAvailable tools: {json.dumps(tool_schemas)}"

        max_iterations = 5
        all_tool_calls = []
        all_content = []

        for iteration in range(max_iterations):
            try:
                response = self._executor.execute(
                    routing,
                    tool_prompt,
                    temperature=stage.definition.temperature,
                    max_tokens=stage.definition.max_tokens,
                )
            except Exception as exc:
                return StageResult(
                    stage_id=stage.id,
                    stage_type=stage.stage_type.value,
                    model_id=routing.model_id,
                    provider_id=routing.provider_id,
                    content="\n".join(all_content),
                    success=False,
                    error=str(exc),
                    tool_calls=[tc.__dict__ for tc in all_tool_calls],
                )

            content = response.content
            all_content.append(content)

            # Parse tool calls from response
            tool_calls = parse_tool_calls(content)
            if not tool_calls:
                # No more tool calls, we're done
                break

            # Execute tool calls
            tool_outputs = []
            for tc in tool_calls:
                self._on_activity("tool_call", f"Calling {tc.tool_name}...")
                result = self._tool_registry.execute(tc)
                all_tool_calls.append(tc)
                if result.success:
                    tool_outputs.append(f"Tool {tc.tool_name} succeeded: {result.output}")
                    # Track file changes
                    if tc.tool_name == "write_file" and "path" in tc.arguments:
                        context.files_created.append(tc.arguments["path"])
                else:
                    tool_outputs.append(f"Tool {tc.tool_name} failed: {result.error}")

            # Add tool results to context for next iteration
            tool_context = "\n\nTool results:\n" + "\n".join(tool_outputs)
            tool_prompt = f"{prompt}\n\nPrevious output:\n{content}{tool_context}\n\nAvailable tools: {json.dumps(tool_schemas)}"

        return StageResult(
            stage_id=stage.id,
            stage_type=stage.stage_type.value,
            model_id=routing.model_id,
            provider_id=routing.provider_id,
            content="\n".join(all_content),
            success=True,
            tool_calls=[tc.__dict__ for tc in all_tool_calls],
        )

    def _route_for_stage(self, stage: PipelineStage) -> RoutingDecision | None:
        """Route to the best model for a specific stage."""
        from synapse.domain import Decision, PrivacyMode, ProviderKind

        # Create a decision with the stage's required capabilities
        decision = Decision(
            required_capabilities=stage.required_capabilities,
            preferred_capabilities=stage.definition.preferred_capabilities,
            preferred_kind=ProviderKind.LOCAL,
            privacy=PrivacyMode.LOCAL_ONLY,
            reasoning=[],
        )

        return self._router.route(
            decision=decision,
            hardware=self._hardware,
            registry=self._registry,
            provider_health=self._health,
            available_models=self._available,
            performance=self._performance,
        )

    def _is_complex_request(self, prompt: str) -> bool:
        """Determine if a request needs multiple stages."""
        indicators = [
            "build ", "create ", "implement ", "refactor ",
            "design ", "architect ", "develop ", "add feature",
            "write a ", "make a ", "set up ",
            " and ", " then ", " also ",
            " with tests", " with documentation",
        ]
        prompt_lower = prompt.lower()
        return any(ind in prompt_lower for ind in indicators)

    def _infer_stage_type(self, prompt: str) -> StageType:
        """Infer the best stage type for a simple request."""
        prompt_lower = prompt.lower()
        if any(w in prompt_lower for w in ["read", "what", "show", "list", "find"]):
            return StageType.RESEARCH
        if any(w in prompt_lower for w in ["write", "create", "add", "implement"]):
            return StageType.CODING
        if any(w in prompt_lower for w in ["review", "check", "fix", "debug"]):
            return StageType.REVIEW
        if any(w in prompt_lower for w in ["document", "explain", "comment"]):
            return StageType.DOCUMENTATION
        if any(w in prompt_lower for w in ["plan", "design", "architect"]):
            return StageType.PLANNING
        return StageType.SYNTHESIS

    def _build_previous_output(
        self,
        results: dict[str, StageResult],
        dep_ids: list[str],
    ) -> str:
        """Combine outputs from dependency stages."""
        parts = []
        for dep_id in dep_ids:
            if dep_id in results:
                r = results[dep_id]
                parts.append(f"[{r.stage_type}]:\n{r.content[:3000]}")
        return "\n\n".join(parts)

    def _build_files_summary(self, context: PipelineContext) -> str:
        """Build a summary of file changes so far."""
        changes = context.get_change_summary()
        parts = []
        if changes["created"]:
            parts.append(f"Created: {', '.join(changes['created'])}")
        if changes["modified"]:
            parts.append(f"Modified: {', '.join(changes['modified'])}")
        if changes["deleted"]:
            parts.append(f"Deleted: {', '.join(changes['deleted'])}")
        return "\n".join(parts) if parts else "No files changed yet."
