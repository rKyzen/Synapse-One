"""Multi-Agent Pipeline — decomposes requests into stages and routes each
to the best-fit model. Handles context passing, tool calls, and progress
tracking via the activity feed.

Architecture:
    Request -> Decompose -> [Stage1, Stage2, ...] -> Execute -> Synthesize
    Each stage: Route -> Execute -> Validate -> Pass Context to Next

Key components:
    - PipelineOrchestrator: main entry point, manages stage execution
    - PipelineStage: individual stage with routing and execution
    - PipelineContext: shared state between stages
    - ToolRegistry: tool definitions and execution
    - ActivityTracker: live progress updates
"""
