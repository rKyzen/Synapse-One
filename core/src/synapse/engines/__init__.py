"""Synapse One engines — the 8-engine taxonomy (AI Operating Workspace).

This package is the canonical *namespace* of the redesigned architecture.
Each engine re-exports the existing subsystem's public entry points WITHOUT
moving any logic: the modules listed below remain the implementations and
keep their current names for backward compatibility (675-test suite, API
contracts). The engines become the seams that later phases refactor behind.

Engine map (name -> existing implementation):

    workspace_engine   -> synapse.projects.system (WorkspaceSystem),
                           synapse.projects.manager (ProjectManager),
                           synapse.workspace.operator (FileOperator),
                           synapse.workspace.change_panel, synapse.projects.chats,
                           synapse.projects.goals (Phase A)
    memory_engine      -> synapse.memory.workspace_memory (WorkspaceMemory),
                           synapse.domain.memory (MemoryEntry)
    knowledge_engine   -> synapse.workspace (files, indexer, vectors, embeddings,
                           retrieval, parsers, chunking, vision, jobs),
                           synapse.search.semantic_search
    ai_orchestrator    -> synapse.master.agent (MasterAgent),
                           synapse.planner.heuristic, synapse.execution,
                           synapse.pipeline (orchestrator/stages/tools),
                           synapse.verification, synapse.confidence, synapse.correction,
                           synapse.citations, synapse.escalation
    tool_engine        -> synapse.pipeline.tools (ToolRegistry),
                           synapse.actions (ActionEngine, classifier),
                           synapse.terminal.runner, synapse.tasks.queue,
                           synapse.diagnostics.analyzer, synapse.actions.code_editor
    model_engine       -> synapse.providers (ProviderManager, ollama, gemini, openai),
                           synapse.registry, synapse.router, synapse.lifecycle,
                           synapse.hardware, synapse.performance, synapse.execution
    automation_engine  -> synapse.tasks.queue, synapse.events (Phase E: schedules,
                           triggers, workflows on top of the existing bus)
    ui_engine          -> synapse.api (build_app) + api/web_ui.py (Phase C: replaced
                           by the workspace shell; api layer stays)

Design rules the engines encode (vision gate):
    1. The user's GOAL is the center — chat is one view, never the product.
    2. AI is one engine among eight — the orchestrator supervises, never
       specializes, and models execute capabilities in the background.
    3. Engines depend on contracts, never on each other's internals; the
       event bus is the only cross-engine side channel.
"""

from __future__ import annotations

# -- workspace_engine -----------------------------------------------------
# Projects/workspaces, files, chats, goals, changes, terminal roots.

from synapse.projects.system import WorkspaceSystem  # noqa: F401
from synapse.projects.manager import ProjectManager  # noqa: F401
from synapse.projects.chats import ChatStore  # noqa: F401
from synapse.projects.goals import GoalStore  # noqa: F401
from synapse.workspace.operator import FileOperator, WorkspaceSafetyError  # noqa: F401
from synapse.workspace.change_panel import ChangePanel  # noqa: F401

# -- memory_engine ----------------------------------------------------------
# Scoped semantic memory: conversation / project / global / user (Phase A).

from synapse.memory.workspace_memory import WorkspaceMemory  # noqa: F401
from synapse.domain.memory import MemoryEntry  # noqa: F401
from synapse.domain.enums import MemoryScope  # noqa: F401

# -- knowledge_engine -------------------------------------------------------
# parse -> chunk -> embed -> index -> retrieve across all document kinds.

from synapse.workspace.workspace import Workspace  # noqa: F401
from synapse.workspace.parsers import parse  # noqa: F401
from synapse.workspace.chunking import chunk_text, chunk_code  # noqa: F401
from synapse.workspace.embeddings import EmbeddingEngine  # noqa: F401
from synapse.workspace.vectors import create_vector_store  # noqa: F401
from synapse.workspace.indexer import WorkspaceIndexer  # noqa: F401
from synapse.workspace.retrieval import Retrieval  # noqa: F401
from synapse.workspace.vision import VisionPipeline  # noqa: F401
from synapse.search.semantic_search import SemanticSearch  # noqa: F401

# -- ai_orchestrator --------------------------------------------------------
# Goal intake -> planning -> capability selection -> execution -> validation.

from synapse.master.agent import MasterAgent  # noqa: F401
from synapse.planner.heuristic import HeuristicTaskPlanner  # noqa: F401
from synapse.execution import ExecutionPlanner, Executor  # noqa: F401
from synapse.pipeline.orchestrator import PipelineOrchestrator  # noqa: F401
from synapse.verification.impl import CompositeVerifier  # noqa: F401
from synapse.confidence.heuristic import HeuristicConfidenceEngine  # noqa: F401

# -- tool_engine ------------------------------------------------------------
# The registry is the single execution surface for every tool; the model only
# emits tool calls, the engine performs them (Phase A: wiring step).

from synapse.pipeline.tools import ToolRegistry, ToolDefinition  # noqa: F401
from synapse.actions.engine import ActionEngine  # noqa: F401
from synapse.actions.classifier import classify_request  # noqa: F401
from synapse.terminal.runner import TerminalRunner  # noqa: F401
from synapse.tasks.queue import TaskQueue  # noqa: F401
from synapse.diagnostics.analyzer import DiagnosticAnalyzer  # noqa: F401
from synapse.actions.code_editor import CodeEditor  # noqa: F401

# -- model_engine -----------------------------------------------------------
# Providers, registry, router, lifecycle — models are execution engines that
# satisfy capabilities; the product never exposes them directly.

from synapse.providers.manager import ProviderManager  # noqa: F401
from synapse.registry import ConfigModelRegistry  # noqa: F401
from synapse.router.router import Router  # noqa: F401
from synapse.lifecycle.manager import ModelLifecycleManager  # noqa: F401
from synapse.hardware import HardwareScanner  # noqa: F401
from synapse.performance.store import FilePerformanceStore  # noqa: F401

# -- automation_engine (Phase E placeholder) --------------------------------
# Schedules/triggers/workflows run on the task queue + event bus.

from synapse.tasks.queue import TaskQueue as _AutomationTaskQueue  # noqa: F401

__all__ = [
    "WorkspaceSystem",
    "ProjectManager",
    "ChatStore",
    "GoalStore",
    "FileOperator",
    "WorkspaceSafetyError",
    "ChangePanel",
    "WorkspaceMemory",
    "MemoryEntry",
    "MemoryScope",
    "Workspace",
    "parse",
    "chunk_text",
    "chunk_code",
    "EmbeddingEngine",
    "create_vector_store",
    "WorkspaceIndexer",
    "Retrieval",
    "VisionPipeline",
    "SemanticSearch",
    "MasterAgent",
    "HeuristicTaskPlanner",
    "ExecutionPlanner",
    "Executor",
    "PipelineOrchestrator",
    "CompositeVerifier",
    "HeuristicConfidenceEngine",
    "ToolRegistry",
    "ToolDefinition",
    "ActionEngine",
    "classify_request",
    "TerminalRunner",
    "TaskQueue",
    "DiagnosticAnalyzer",
    "CodeEditor",
    "ProviderManager",
    "ConfigModelRegistry",
    "Router",
    "ModelLifecycleManager",
    "HardwareScanner",
    "FilePerformanceStore",
]
