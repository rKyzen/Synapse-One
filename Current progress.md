# Current Progress — Master Model Routing & Architecture Refactor

## 1. Summary of Changes & Architecture Upgrades

### 1.1 Master Analysis First
- **Strict Schema Enforcement**: Implemented `MasterAnalysis` model in `core/src/synapse/master/schemas.py` containing 18 structured fields (`intent`, `domain`, `goal`, `workspace_needed`, `workspace_reason`, `files_needed`, `memory_needed`, `tools_needed`, `coding_needed`, `vision_needed`, `document_processing_needed`, `web_needed`, `artifact_required`, `reasoning_complexity`, `required_capabilities`, `recommended_model_role`, `execution_mode`, `recommended_workers`).
- **Pre-Flight Execution**: The Master Model analyzes incoming prompts **before** touching the filesystem, scanning workspace directories, or loading tools.
- **Master Prompt & Deterministic Fallback**: Added `_ANALYSIS_SYSTEM_PROMPT` in `core/src/synapse/master/orchestrator.py` and a robust deterministic `_fallback_analysis` method that produces full `MasterAnalysis` even in offline/fallback modes.

### 1.2 Intent Classification & Routing Fixes
- **Indirect Phrasing Support**: Updated regex patterns across `core/src/synapse/actions/classifier.py` and `core/src/synapse/actions/intent_router.py` to match indirect clauses (e.g., `"Generate Me a Simple Html Css Landing PAge..."`, `"Build us an app..."`).
- **Artifact Generation Recognition**: Queries containing creation verbs (`create`, `generate`, `build`, `make`, `write`, `develop`) targeting deliverables are classified as `artifact_generation` (`artifact_required=True`).
- **Conversational Queries Preserved**: Direct explanatory queries (e.g., `"Explain how HTML works"`) route cleanly to direct answers with `artifact_required=False`.

### 1.3 Physical File Creation Pipeline
- **Enforced File Output**: When `artifact_required=True`, the execution planner sets `file_output=True` and appends `Capability.CODING` to the task DAG.
- **Manifest Instruction**: Tasks producing files inject structured manifest instructions into worker prompts.
- **Disk Generation**: The Action Engine and `FileOperator` parse file manifests (`parse_file_manifest`) and physically create and verify files on disk (e.g. `index.html`, `style.css`).

### 1.4 Strict Workspace Opt-In & Isolation
- **No Unnecessary Scans**: Standalone math, logic, and conversational requests evaluate to `workspace_needed=False`, ensuring zero file scans, zero workspace context retrieval, and zero unneeded disk I/O.
- **Explicit Workspace Cues**: Workspace access is only activated when explicitly requested (e.g. `"in my workspace"`, `"in my existing project"`) or when attached files are present.

### 1.5 Difficulty-Aware Specialist Model Selection
- **Reasoning Complexity Scoring**: Mapped `reasoning_complexity` levels (`TRIVIAL`, `EASY`, `MEDIUM`, `HARD`, `VERY_HARD`) directly to complexity scores.
- **Strengths & Weaknesses Evaluation**: Upgraded `core/src/synapse/router/router.py` to evaluate model profile strengths and weaknesses against reasoning complexity. Hard math and combinatorial logic problems require `Capability.REASONING` and `Capability.MATH`, routing to capable models (`gemma3:4b` / `gemma3:12b`) while heavily penalizing models with math weaknesses (`qwen3:1.7b`).

### 1.6 UI Observability & Granular Timings
- **Timeline Events**: Published rich timeline events for every decision phase (`master_analysis`, `intent_decision`, `domain_decision`, `artifact_decision`, `files_decision`, `workspace_decision`, `capability_decision`).
- **Timing Breakdown**: Added timing metrics to `DecisionTrace.timings`:
  - `master_analysis_ms`
  - `context_retrieval_ms`
  - `worker_inference_ms`
  - `tools_ms`
  - `total_ms`

---

## 2. Modified & Created Files

| File | Changes Made |
| :--- | :--- |
| `core/src/synapse/master/schemas.py` | Added `ReasoningComplexity`, `ExecutionMode`, and `MasterAnalysis` Pydantic models. |
| `core/src/synapse/master/__init__.py` | Exported `MasterAnalysis`, `ReasoningComplexity`, `ExecutionMode`. |
| `core/src/synapse/domain/diagnosis.py` | Extended `DecisionTrace` with `master_analysis`, `workspace_needed`, `reasoning_complexity`, `domain`, and `timings`. |
| `core/src/synapse/actions/classifier.py` | Updated regexes for create/build/generate to match indirect objects and prioritized file creation. |
| `core/src/synapse/actions/intent_router.py` | Updated `FILE_GENERATION` intent routing regex for indirect object support. |
| `core/src/synapse/master/orchestrator.py` | Implemented `analyze()`, structured JSON schema prompting, and deterministic `_fallback_analysis()`. |
| `core/src/synapse/master/agent.py` | Wired Master Analysis pre-flight, timeline publishing, execution mode routing, and timing calculations. |
| `core/src/synapse/router/router.py` | Added strengths/weaknesses matching and latency scaling for hard reasoning tasks. |
| `core/tests/test_master_analysis.py` | Created comprehensive regression test suite covering Tests 1 to 5. |

---

## 3. Regression Test Matrix

- **Test 1**: `"Generate Me a Simple Html Css Landing PAge for A Product"` -> `intent="artifact_generation"`, `artifact_required=True`, routes to coding worker, creates `index.html` & `style.css` on disk.
- **Test 2**: `"Explain how HTML and CSS work."` -> `intent="direct_answer"`, `artifact_required=False`, no files created.
- **Test 3**: `"Create a landing page in my existing project."` -> `intent="artifact_generation"`, `workspace_needed=True`, creates files in workspace.
- **Test 4**: Father/Son age word problem -> standalone math problem, `workspace_needed=False`, zero workspace scan, routes to math/chat worker.
- **Test 5**: Chessboard dominoes proof -> hard reasoning puzzle, `reasoning_complexity="hard"`, routes to capable reasoning model (`gemma3:4b` / `gemma3:12b`), rejects weak models (`qwen3:1.7b`).
