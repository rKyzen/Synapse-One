"""Tests for the multi-agent pipeline system."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synapse.pipeline.context import PipelineContext, StageResult
from synapse.pipeline.stages import (
    STAGE_DEFINITIONS,
    PipelineStage,
    StageType,
)
from synapse.pipeline.tools import ToolCall, ToolRegistry, parse_tool_calls
from synapse.pipeline.orchestrator import PipelineOrchestrator


class TestPipelineContext:
    """Tests for PipelineContext."""

    def test_context_creation(self):
        ctx = PipelineContext(prompt="test prompt")
        assert ctx.prompt == "test prompt"
        assert ctx.stage_results == []
        assert ctx.files_created == []

    def test_add_stage_result(self):
        ctx = PipelineContext(prompt="test")
        result = StageResult(
            stage_id="s1",
            stage_type="planning",
            model_id="model1",
            provider_id="provider1",
            content="plan content",
            files_written=["file1.py"],
        )
        ctx.add_stage_result(result)
        assert len(ctx.stage_results) == 1
        assert ctx.files_created == ["file1.py"]

    def test_get_previous_outputs(self):
        ctx = PipelineContext(prompt="test")
        ctx.stage_results.append(StageResult(
            stage_id="s1",
            stage_type="planning",
            model_id="m1",
            provider_id="p1",
            content="output1",
        ))
        outputs = ctx.get_previous_outputs()
        assert "output1" in outputs

    def test_get_change_summary(self):
        ctx = PipelineContext(prompt="test")
        ctx.files_created = ["a.py", "b.py"]
        ctx.files_modified = ["c.py"]
        summary = ctx.get_change_summary()
        assert summary["created"] == ["a.py", "b.py"]
        assert summary["modified"] == ["c.py"]
        assert summary["deleted"] == []


class TestPipelineStages:
    """Tests for pipeline stages."""

    def test_stage_definitions_exist(self):
        assert StageType.PLANNING in STAGE_DEFINITIONS
        assert StageType.CODING in STAGE_DEFINITIONS
        assert StageType.REVIEW in STAGE_DEFINITIONS
        assert StageType.SYNTHESIS in STAGE_DEFINITIONS

    def test_stage_has_required_capabilities(self):
        for stage_type, defn in STAGE_DEFINITIONS.items():
            assert len(defn.required_capabilities) > 0

    def test_pipeline_stage_creation(self):
        stage = PipelineStage(
            id="s1",
            definition=STAGE_DEFINITIONS[StageType.CODING],
            task_description="write code",
        )
        assert stage.stage_type == StageType.CODING
        assert "write code" in stage.task_description

    def test_stage_prompt_formatting(self):
        stage = PipelineStage(
            id="s1",
            definition=STAGE_DEFINITIONS[StageType.CODING],
            task_description="implement feature",
        )
        prompt = stage.format_prompt(
            context={"workspace_brief": "project info"},
            previous_output="plan from planner",
        )
        assert "implement feature" in prompt
        assert "project info" in prompt


class TestToolRegistry:
    """Tests for tool registry."""

    def test_registry_creation(self):
        registry = ToolRegistry()
        tools = registry.list_tools()
        assert len(tools) > 0

    def test_builtin_tools_exist(self):
        registry = ToolRegistry()
        assert registry.get("read_file") is not None
        assert registry.get("write_file") is not None
        assert registry.get("list_files") is not None

    def test_tool_schemas(self):
        registry = ToolRegistry()
        schemas = registry.get_schemas()
        assert len(schemas) > 0
        assert all("name" in s for s in schemas)

    def test_parse_json_tool_calls(self):
        text = '```json\n{"tool": "read_file", "args": {"path": "test.py"}}\n```'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].tool_name == "read_file"
        assert calls[0].arguments["path"] == "test.py"

    def test_parse_inline_tool_calls(self):
        text = 'TOOL:write_file(path="test.py", content="hello")'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].tool_name == "write_file"

    def test_unknown_tool_returns_error(self):
        registry = ToolRegistry()
        call = ToolCall(id="tc1", tool_name="unknown_tool", arguments={})
        result = registry.execute(call)
        assert not result.success
        assert "Unknown tool" in result.error


class TestToolCalling:
    """Tests for tool execution."""

    def test_read_file_tool(self, tmp_path):
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world", encoding="utf-8")

        from synapse.workspace.operator import FileOperator
        operator = FileOperator(tmp_path)
        registry = ToolRegistry(file_operator=operator)

        call = ToolCall(id="tc1", tool_name="read_file", arguments={"path": "test.txt"})
        result = registry.execute(call)
        assert result.success
        assert result.output == "hello world"

    def test_write_file_tool(self, tmp_path):
        from synapse.workspace.operator import FileOperator
        operator = FileOperator(tmp_path)
        registry = ToolRegistry(file_operator=operator)

        call = ToolCall(
            id="tc1",
            tool_name="write_file",
            arguments={"path": "new.txt", "content": "new content"},
        )
        result = registry.execute(call)
        assert result.success
        assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "new content"

    def test_list_files_tool(self, tmp_path):
        from synapse.workspace.operator import FileOperator
        operator = FileOperator(tmp_path)
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "b.txt").write_text("b", encoding="utf-8")
        registry = ToolRegistry(file_operator=operator)

        call = ToolCall(id="tc1", tool_name="list_files", arguments={})
        result = registry.execute(call)
        assert result.success
        assert len(result.output) == 2
