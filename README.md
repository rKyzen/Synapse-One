# Synapse One — AI Operating Workspace

> The world's first truly intelligent AI Operating Workspace.
> Users define the outcome. Synapse handles everything else.

**Vision document:** [`References/Synapse_one.md`](References/Synapse_one.md) — the single source of truth.

**Current status: Phase 13 — Workspace-First Operating Environment (every model call — including synthesis and review — is prefixed with a compact workspace brief: project identity, grounding path, real file tools, live file tree, recent modifications, current chat; a deterministic `requires_workspace_access` gate feeds file excerpts only to prompts that actually touch the workspace, leaving greetings chat-only; the write→verify→index→remember loop confirms files on disk, syntax-checks Python/JSON, refreshes the project index, and records the write at project memory scope; the timeline UI gains 🛡️ verified / 🗂️ indexed steps). Core: Phase 7 (Action-Based Execution Engine — every request classified into one of seven kinds (Chat Response / File Creation / File Modification / Project Generation / Project Analysis / Documentation / Workspace Operation); the backend executes filesystem actions (create/write/edit/rename/move/delete/search folders+files), verifies every write, and returns only a short summary; code is written directly into the workspace (index.html, style.css, script.js, assets/), never dumped into chat. Includes: Workspace Engine Architecture — Synapse storage separated from user workspaces, SQLite project registry, native folder picker, reconnect + delete modes, lifecycle cleanup).** Phase 1 built the foundation (domain models, contracts, hardware scanner, model registry, provider abstraction, event bus, configuration, DI, logging). Phase 2 made Synapse intelligent: it now understands a request (intent), scores its complexity, decides privacy, builds an execution plan, routes to the best provider/model, executes, and returns both the answer and the reasoning behind every decision — through the Master Agent, the only public entry point for AI requests. Phase 2.5 upgraded the router into an intelligent supervisor: capability-weighted specialist profiles, vision-exclusion rules, calibrated complexity, latency prediction, a performance learning loop, and explainable traces. Phase 2.5+ added the model lifecycle (RAM-resident tracking, idle auto-unload, embedding pinning, loaded-model reuse). Phase 3 starts genuinely multi-step orchestration: large prompts are decomposed into a task DAG (Task Planner), every task is routed and executed independently (per-task routing with reliability-adjusted scoring), relevant conversation/project history is recalled into each task prompt (Workspace Memory), and per-task outputs are merged into a single answer (Result Synthesizer) with the full execution graph returned on every response. Phase 4 turns Synapse from a chat interface into a true AI Workspace: drag & drop multimodal files with automatic per-file pipeline decisions, a local vector store with nomic-embed-text embeddings, background indexing with live job progress, and an enhanced web UI. Workspace content is enforced local-only. **Phase 5 adds a multi-project workspace system:** each project gets its own isolated directory (files, vector store, memory, chats), durable per-chat message persistence, automatic session recovery across restarts, and a redesigned web UI with server-backed project/chat management. **Phase 6 completes the cycle — Synapse can now *do* the work:** projects can be created in user-chosen folders, and for file-producing requests the task pipeline writes real files through a safe, contained per-project filesystem operator, validates every output (python/JSON/JS/HTML), appends a REVIEW task routed to the strongest reasoning model, merges that review into the answer, and records an on-disk audit trail (plan, models, tools, files, failures, timing) per project. **Phase 7 re-architects the Workspace System like a professional IDE:** a project's folder is now *only* the project — source, generated files, and Git — while all Synapse metadata (chats, memory, embeddings, logs, session) lives in OS appdata (`%LOCALAPPDATA%\Synapse`) keyed by a unique project id in a SQLite registry; projects are created through a native OS folder picker, can be deleted in two modes (detach keeps the files, erase removes the folder), reconnect automatically when the folder is moved, and startup/shutdown cleanup sweeps stale temp and orphaned internal state.

---

## 1. Repository Structure

```
Synapse One/
├── references/
│   └── Synapse_one.md               # Source of truth (vision doc)
├── core/                            # Python AI Core (the "brain")
│   ├── pyproject.toml               # Package + dependency manifest
│   ├── config/
│   │   └── config.toml              # Default configuration (sample catalog)
│   ├── scripts/
│   │   └── verify.py                # CLI Phase 1 verification
│   ├── src/synapse/
│   │   ├── __init__.py              # Version
│   │   ├── bootstrap.py             # Composition root (DI wiring)
│   │   ├── actions/                 # Phase 7 Action-Based Execution Engine
│   │   │   ├── __init__.py          #   exports: ActionEngine, RequestKind, classify_request, extract_workspace_ops
│   │   │   ├── classifier.py        #   RequestKind classifier (7 kinds) + explicit op extraction
│   │   │   └── engine.py            #   ActionEngine — backend FS tools + plan → execute → verify → summarize
│   │   ├── api/                     # FastAPI surface (status, /request, /files, /jobs, /projects (+pick/reconnect/verify), /work, /actions, /chats, /session, /ui)
│   │   ├── api/web_ui.py            # Server-backed chat UI (Phase 5/6/7)
│   │   ├── master/                  # Master Agent — the orchestrator
│   │   ├── projects/                # Phase 5/6/7 workspace system (projects, chats, isolation, picker)
│   │   │   ├── chats.py             #   ChatStore — atomic per-chat persistence (internal storage)
│   │   │   ├── manager.py           #   ProjectManager — SQLite registry (workspace paths only) + CRUD
│   │   │   ├── actionlog.py         #   ActionLog — per-project audit trail (JSONL, Phase 6)
│   │   │   ├── picker.py            #   native OS folder picker (PowerShell/zenity/osascript, Phase 7)
│   │   │   └── system.py            #   WorkspaceSystem — facade (projects, chats, internal per-project storage, sessions, file operator, cleanup)
│   │   ├── workspace/               # Phase 4 + Phase 6 workspace subsystems
│   │   │   ├── workspace.py         #   facade the Master talks to (prepare/upload/index)
│   │   │   ├── files.py             #   File Manager (sha256 dedupe, catalog, blobs)
│   │   │   ├── parsers.py           #   Document Parser (PDF/DOCX/text)
│   │   │   ├── chunking.py          #   text + code chunking with overlap
│   │   │   ├── embeddings.py        #   Embedding Engine (nomic-embed-text, batched)
│   │   │   ├── vectors.py           #   FAISS + numpy/local vector stores
│   │   │   ├── retrieval.py         #   semantic/keyword RAG + deterministic code scan
│   │   │   ├── vision.py            #   Vision Pipeline (base64 → vision model)
│   │   │   ├── indexer.py           #   background parse → chunk → embed → store
│   │   │   ├── jobs.py              #   index job registry/progress
│   │   │   ├── operator.py          #   FileOperator — safe contained file CRUD (Phase 6)
│   │   │   ├── manifest.py          #   parse_file_manifest — model output → file writes (Phase 6)
│   │   │   └── review.py            #   OutputReviewer — validate_file + summarize (Phase 6)
│   │   ├── analyzers/               # Intent, Complexity, Privacy analyzers
│   │   ├── decision/                # Decision Engine
│   │   ├── router/                  # Router V2 (intelligent supervisor)
│   │   ├── execution/               # Planner + Executor
│   │   ├── performance/             # Performance learning loop (Phase 2.5)
│   │   ├── planner/                 # Task Planner — prompt → task DAG (Phase 3)
│   │   ├── memory/                  # Workspace Memory — scoped semantic store (Phase 3)
│   │   ├── synthesis/               # Result Synthesizer — task output merger (Phase 3)
│   │   ├── config/                  # Layered config (toml → env → secret files)
│   │   ├── contracts/               # Abstract interfaces (all subsystems)
│   │   ├── di/                      # Dependency injection container
│   │   ├── domain/                  # Typed domain entities + enums
│   │   ├── events/                  # Pub/sub event bus
│   │   ├── hardware/                # Hardware scanner (independent service)
│   │   ├── logging/                 # Structured logging (structlog)
│   │   ├── providers/               # Provider abstraction + vendors
│   │   │   ├── base.py              #   HTTP provider base class
│   │   │   ├── factory.py           #   provider_id → class map (only place vendors are named)
│   │   │   ├── manager.py           #   lifecycle + lookup (what the Master sees)
│   │   │   ├── ollama.py            #   Local provider
│   │   │   ├── openai.py            #   Cloud provider (OpenAI-compatible)
│   │   │   └── gemini.py            #   Cloud provider (Google)
│   │   └── registry/                # Config-driven model catalog
│   └── tests/                       # pytest suite (452 tests, offline)
├── rust/                            # Rust perf modules (future phase)
└── .gitignore
```

**Design invariants (non-negotiable):**
- The Master Agent never knows vendor names. It talks only to `ModelProvider` interfaces through `ProviderManager`.
- No vendor code exists outside `providers/`.
- No business logic in the UI (future). No hardcoded paths or model names in code.
- Hardware logic never mixes with routing.
- Every pipeline subsystem (analyzers, decision engine, router, planner, executor) is replaceable behind its contract.

---

## 2. Required Software

| Tool | Required | Notes |
|---|---|---|
| Python | Yes | 3.11+ (tested on 3.14) |
| Ollama | Recommended | Needed for the local provider. Install from [ollama.com](https://ollama.com) |
| Rust | Future phase | Not needed for Phase 1 |

---

## 3. Python Version

- **Minimum:** 3.11
- **Tested:** 3.14.3 (Windows)

Check your version:

```powershell
python --version
```

---

## 4. Rust Version

Not required for Phase 1. Rust is reserved for future performance modules (hardware monitoring, secure IPC) exposed via Tauri/native bindings. No Rust toolchain is needed today.

---

## 6. Required Environment Variables

| Variable | Purpose | Required? |
|---|---|---|
| `SYNAPSE_HOME` | Override the data/config home directory | Optional (defaults to OS app-data dir) |
| `OPENAI_API_KEY` | OpenAI API key | Only if OpenAI provider is enabled |
| `GEMINI_API_KEY` | Google Gemini API key | Only if Gemini provider is enabled |
| `SYNAPSE_LOGGING_LEVEL` | Log level override (e.g. `DEBUG`) | Optional |
| `SYNAPSE_LOGGING_JSON_LINES` | `true` for JSON log output | Optional |

**Secrets policy:** API keys are never committed. Supply them via environment variables or a secret file referenced in `config.toml` (`api_key_file = "C:/secure/openai.key"`). Secret files are covered by `.gitignore`.

Setting on Windows PowerShell (example — replace the path with your own):

```powershell
# internal data home (default: $env:LOCALAPPDATA\Synapse)
$env:SYNAPSE_HOME = "$env:LOCALAPPDATA\Synapse"
$env:OPENAI_API_KEY = "sk-..."
```

---

## 7. config.toml Example

Located at `core/config/config.toml` by default (also copied to `$SYNAPSE_HOME/config/config.toml` if you customize).

```toml
[logging]
level = "INFO"
json_lines = false

[providers.ollama]
enabled = true
base_url = "http://localhost:11434"
default_model = "llama3.2"

# Timeouts (seconds) — generous for local CPU inference:
#   connect    : TCP connection establishment
#   read       : max idle gap between streamed chunks (model "thinking" time)
#   generation : total wall-clock budget; exceeded mid-stream, partial output is returned
[providers.ollama.timeouts]
connect = 10
read = 180
generation = 180

# Model lifecycle — RAM-resident model management.
#   idle_timeout_small      : unload idle models after N seconds (small models)
#   idle_timeout_large      : unload idle LARGE models sooner (they waste RAM)
#   idle_timeout_embedding  : never auto-unload the embedding model
#   cleanup_interval        : background cleanup runs every N seconds
#   keep_embedding_loaded   : keep the embedding model resident
#   low_memory_threshold    : GB of free RAM that triggers immediate idle unload
#   large_model_min_ram_gb  : models requiring >= this RAM are "large"
#   max_loaded_models       : soft ceiling on concurrently loaded models
[lifecycle]
idle_timeout_small = 60
idle_timeout_large = 30
idle_timeout_embedding = 1e18
cleanup_interval = 15
keep_embedding_loaded = true
low_memory_threshold = 4
large_model_min_ram_gb = 8
max_loaded_models = 4

# Task Planner — decomposes large prompts into a task DAG.
#   enabled             : set false to always treat the whole prompt as one task
#   min_split_complexity: below this complexity score, no decomposition
#   max_tasks           : ceiling on the number of execution tasks
# Phase 6 — AI Workspace execution:
#   file_output_enabled : detect create/write/generate/refactor prompts and
#                         expect a structured file manifest from the model
#   add_review          : append a REVIEW task routed to the strongest
#                         available reasoning model after file tasks
#   model_hints         : kind -> preferred model id, e.g.
#                         { coding = "qwen2.5-coder", writing = "llama3.1" }
[planner]
enabled = true
min_split_complexity = 40
max_tasks = 6
file_output_enabled = true
add_review = true

# Workspace Memory — scoped (conversation/project/global) semantic memory.
#   embedding_model : model used for embeddings (must be installed in Ollama)
#   top_k           : results returned per search
#   min_similarity  : cosine threshold; below it, entries are dropped
[memory]
enabled = true
embedding_model = "nomic-embed-text"
top_k = 3
min_similarity = 0.35

# Phase 4 — AI Workspace (multimodal files + RAG + vision + code scan).
#   enabled              : master switch for the whole workspace subsystem
#   storage_dir          : base dir under $SYNAPSE_HOME/data
#   chunk_size/overlap   : document chunking window (chars) + overlap
#   code_chunk_lines     : code chunking window (lines) + line overlap
#   embedding_model      : used by the Embedding Engine (installed in Ollama)
#   vision_model         : used by the Vision Pipeline (e.g. qwen2.5vl:7b)
#   top_k / min_similarity : retrieval beam size + cosine threshold
#   embed_batch_size     : vectors sent per embedding request
#   allow_cloud_forwarding : FALSE = workspace content is enforced local-only.
#   auto_retrieve        : when no file is attached, retrieve over ALL indexed
#                          files automatically (follow-up questions)
#   code_scan_patterns   : regexes for deterministic code scans (TODO/FIXME…)
#   code_scan_triggers   : words in the prompt that trigger a code scan
[workspace]
enabled = true
storage_dir = "workspace"
chunk_size = 1200
chunk_overlap = 200
code_chunk_lines = 200
code_chunk_overlap = 20
embedding_model = "nomic-embed-text"
vision_model = "qwen2.5vl:7b"
top_k = 4
min_similarity = 0.25
embed_batch_size = 16
allow_cloud_forwarding = false
auto_retrieve = true
code_scan_patterns = ["TODO", "FIXME", "HACK", "XXX"]
code_scan_triggers = ["todo", "fixme", "hack"]

# File-type → pipeline mapping (image | document | code). Anything not listed
# falls back to document. Omit a category to use the built-in defaults.
[workspace.pipelines]
image = ["png", "jpg", "jpeg", "webp", "gif", "bmp"]
document = ["pdf", "docx", "txt", "md", "markdown", "rtf"]
code = ["py", "js", "ts", "dart", "rs", "go", "java", "c", "cpp", "h", "hpp",
        "cs", "rb", "php", "sh", "bash", "ps1", "sql", "html", "css", "scss",
        "json", "jsonc", "yaml", "yml", "toml", "xml", "proto", "kt", "swift"]

# Vector store backend: "faiss" (FAISS flat-IP = cosine, default) or "local"
# (pure numpy/JSON, used automatically if faiss is not installed).
[vector_store]
provider = "faiss"
```

The Ollama provider **streams internally** on every call. This enables first-token
latency and token-throughput metrics, and lets it return the **partial output** if
generation is interrupted (generation budget exceeded or the stream goes idle past
`read`). A `connect` timeout is surfaced as an error (nothing was generated); `read`
and `generation` timeouts return whatever text was produced so far, logged with the
timeout kind and duration.

Provider metrics are attached to every chat response:

```json
{
  "metrics": {
    "first_token_latency_s": 33.05,
    "total_latency_s": 33.88,
    "tokens_generated": 448,
    "tokens_per_second": 13.22,
    "interrupted": false,
    "interrupt_reason": null
  }
}
```

For the full sample config (all providers + model catalog), see `core/config/config.toml`.

**Precedence (low → high):** built-in defaults → `config.toml` → environment variables → secret files.

---

## 8. How to Install Dependencies

The project is fully portable — no absolute paths, no assumptions about the drive letter or machine. Everything resolves relative to the project folder, so it works from any location (C:, D:, a USB stick, anywhere).

**One-command setup (recommended):** from `core/`, double-click or run:

```powershell
.\bootstrap.ps1
```

This creates the virtual environment (`.venv`), upgrades pip, and installs the package editable with `dev` + `server` extras — automatically, from any location.

**Or manually**, from `core/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,server]"
```

> **Moving machines?** The `.venv` folder is bound to the machine/drive that created it. After copying the project, delete `.venv` and run `.\bootstrap.ps1` again (or follow the steps above). Never reuse a moved virtual environment.

What the extras include:

| Extra | Packages |
|---|---|
| `dev` | pytest, pytest-asyncio, pytest-cov |
| `server` | fastapi, uvicorn |

Install just tests (no server) with `-e ".[dev]"`.

---

## 9. How to Start the FastAPI Backend

**One-command launcher (recommended):** from `core/`:

```powershell
.\start.ps1
```

It auto-installs anything missing (see Section 8), then starts the server on `127.0.0.1:8000`. CMD users: `start.bat`. Add `--reload` for development.

**Or manually** from `core/`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn synapse.api:app --host 127.0.0.1 --port 8000
```

Then open:

- **Interactive docs (Swagger):** <http://127.0.0.1:8000/docs>
- **JSON status:** <http://127.0.0.1:8000/status>

**The status endpoint is read-only verification.** It reports hardware, registry, and provider state. All real business logic arrives in later phases.

---

## 10. How to Verify Ollama Is Detected

Ollama runs as a local service on `localhost:11434`.

**Step 1 — is the Ollama service up?** (Requires Ollama installed, e.g. `ollama serve` or the desktop app):

```powershell
Invoke-RestMethod -Uri "http://localhost:11434/api/tags"
```

You should see a JSON list with `models`. If this fails, start Ollama first.

**Step 2 — is it detected by Synapse?**

```powershell
.\.venv\Scripts\python.exe scripts\verify.py --providers
```

Expected (Ollama running):

```
== Providers ==
  ollama     kind=local  state=ready      health=OK
```

If `health=DOWN`, see **Section 13 (debugging)**.

---

## 11. How to Verify OpenAI / Gemini Providers

These are cloud providers, disabled by default (local-first philosophy).

**Step 1 — enable one in `config.toml`:**

```toml
[providers.openai]
enabled = true
```

**Step 2 — provide the key via env or secret file** (Section 6).

**Step 3 — verify:**

```powershell
$env:OPENAI_API_KEY = "sk-..."
.\.venv\Scripts\python.exe scripts\verify.py --providers
```

Expected:

```
  openai     kind=cloud  state=ready      health=OK
```

Notes:
- OpenAI health returns OK even with a 401 when the key is present (proves endpoint reachability without leaking auth state).
- Gemini health checks that a key is configured.
- Full chat execution is validated by the test suite using offline wire-format tests (`tests/test_providers.py`).

---

## 12. How to Run Every Test

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Or with coverage:

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=synapse --cov-report=term-missing
```

Current result: **595 passed**. No network access is required — providers are tested against mocked HTTP responses, the workspace subsystem runs on deterministic fake embeddings, and Phase 6/7/8 end-to-end tests use a hermetic fake multi-model provider.

---

## 13. How to Debug Startup Failures

**Common failures and fixes:**

| Symptom | Cause / Fix |
|---|---|
| `ModuleNotFoundError: No module named 'synapse'` | Dependencies not installed — run the install step (Section 8) and check `.venv`. |
| `ImportError: cannot import name ...` | Stale editable install — `python -m pip install --force-reinstall --no-deps -e .` |
| Provider `health=DOWN` for `ollama` | Ollama service not running — start Ollama, or fix `base_url` in `config.toml`. |
| Provider `health=DOWN` for `openai` | Key missing — set `OPENAI_API_KEY` or `api_key_file`. |
| `uvicorn` not found | Missing `server` extra — `pip install -e ".[dev,server]"` |
| Port already in use | Change port: `--port 8001` |

**If the backend fails to start, run in the foreground with debug logging:**

```powershell
$env:SYNAPSE_LOGGING_LEVEL = "DEBUG"
.\.venv\Scripts\python.exe -m uvicorn synapse.api:app --port 8000
```

Read the traceback top-down: the first frames tell you which subsystem failed.

---

## 14. How to View Logs

Logs go to the console by default (structured via structlog).

**Development (human-readable, default):**

```powershell
.\.venv\Scripts\python.exe scripts\verify.py
```

**Production (JSON lines):**

```toml
[logging]
json_lines = true
```

Every subsystem logs under its own hierarchical logger:
`synapse.bootstrap`, `synapse.hardware`, `synapse.providers.ollama`, `synapse.providers.manager`, etc.

Log files land in `<SYNAPSE_HOME>/logs/` (auto-created at startup).

---

## 15. Expected Startup Output

Start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn synapse.api:app --host 127.0.0.1 --port 8000
```

You should see uvicorn's startup banner. Query the status:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/status" | ConvertTo-Json -Depth 5
```

Expected shape (values will match your machine):

```json
{
  "name": "synapse-one",
  "version": "0.1.0",
  "providers": [
    { "provider_id": "ollama", "kind": "local", "state": "ready", "ready": true }
  ],
  "hardware": [
    { "key": "os", "value": "Windows" },
    { "key": "cpu", "value": "AMD64 ... (6c/12t)" },
    { "key": "ram_gb", "value": 27.9 },
    { "key": "ram_available_gb", "value": 11.2 },
    { "key": "gpu", "value": null },
    { "key": "can_run_local_llm", "value": true },
    { "key": "max_params_billions", "value": 8.4 }
  ],
  "registry_count": 3,
  "models": ["llama3.2", "gpt-4o-mini", "gemini-1.5-flash"]
}
```

Run the CLI verification for the human-readable summary:

```powershell
.\.venv\Scripts\python.exe scripts\verify.py
```

Expected:

```
== Hardware ==
  OS:          Windows 11 (AMD64)
  CPU:         AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD (6c/12t)
  RAM:         27.9 GB total, 11.2 GB available
  GPU:         not detected
  Storage:     133.6 GB free
  Local LLM:   yes (max ~8.4B params)
== Model Registry (3 models) ==
  gemini-1.5-flash         provider=gemini   kind=cloud  ctx=1000000
  llama3.2                 provider=ollama   kind=local  ctx=131072
  gpt-4o-mini              provider=openai   kind=cloud  ctx=128000
== Providers ==
  ollama     kind=local  state=ready      health=OK
```

**Success criteria for Phase 1:**
1. `pytest` → 83 passed
2. `verify.py` prints hardware + registry + providers
3. `GET /status` returns the JSON above
4. Ollama `health=OK` when the service is running

---

# Phase 2 — Master Agent & Decision Engine

## Architecture

Phase 2 adds the intelligence layer on top of the Phase 1 foundation. Every request now flows through the **Master Agent** — the single public entry point — which orchestrates analysis, decision, planning, routing, and execution. The Master never touches a vendor: it only consumes interfaces.

```
                      ┌───────────────────────────────┐
        prompt ──────▶│         Master Agent          │
                      │   (orchestrates, never does)  │
                      └───────┬───────┬───────┬───────┘
                              │       │       │
                    ┌─────────▼──┐ ┌──▼───────▼──┐ ┌───────────┐
                    │ Hardware   │ │  Registry   │ │  Events   │
                    │ Scanner    │ │ (catalog)   │ │  bus      │
                    └────────────┘ └─────────────┘ └───────────┘
                              │
                              ▼
   ┌──────────────┬──────────────────┬──────────────────┐
   ▼              ▼                  ▼                  ▼
 Intent        Complexity        Privacy          Decision
 Analyzer      Analyzer          Analyzer         Engine
   └──────────────┴──────────────────┴──────────────────┘
                              │
                              ▼
                     Execution Planner
                              │
                              ▼
                         Router V1
                              │  (registry metadata + health + hardware)
                              ▼
                        Executor
                              │  ModelProvider interface only
                              ▼
                    Ollama / OpenAI / Gemini
                              │
                              ▼
               Normalized AgentResponse + DecisionTrace
```

## Subsystem Responsibilities

| Subsystem | Module | Responsibility |
|---|---|---|
| **Master Agent** | `master/agent.py` | Orchestrates the pipeline; the ONLY public entry for AI requests. Never performs provider-specific work. |
| **Intent Analyzer** | `analyzers/intent.py` | Detects primary/secondary intent (general, writing, coding, research, education, business, creative, planning, conversation) + confidence + reasoning. |
| **Complexity Analyzer** | `analyzers/complexity.py` | Scores 0–100 difficulty (calibrated to the vision examples: math 5, grammar 15, summarization 35, essay 60, website 80, research paper 95). |
| **Privacy Analyzer** | `analyzers/privacy.py` | Returns LOCAL_ONLY / PREFER_LOCAL / BALANCED / PREFER_CLOUD / CLOUD_REQUIRED from sensitivity, internet need, complexity, and user preference. |
| **Decision Engine** | `decision/engine.py` | Answers: can it stay local? does it need internet? cloud reasoning? which capabilities? which provider kind? which workspace? memory? (placeholder). |
| **Execution Planner** | `execution/planner.py` | Produces the reusable `ExecutionPlan` (strategy, steps, latency tier, cost). |
| **Router V1** | `router/router.py` | Vendor-agnostic model selection: capability fit, hardware gate, health gate, kind preference, cost/latency weights. Returns provider + model + confidence + reason + candidate list. |
| **Executor** | `execution/executor.py` | Executes the routed request through the `ModelProvider` interface only; measures latency. |
| **Decision Trace** | `domain/diagnosis.py` | Developer-mode trace of one request: intent, confidence, complexity, privacy, internet, hardware, provider, model, reason, timing, candidate count. |

## How the Subsystems Communicate

- **Interfaces over classes:** the Master is constructed with `IntentAnalyzer`, `ComplexityAnalyzer`, `PrivacyAnalyzer`, `DecisionEngine`, `ExecutionPlanner`, `Router`, `Executor`, `ProviderManager`, `HardwareProvider`, `ModelRegistry`, `EventBus`, `ConfigProvider` — all from `synapse.contracts`. Swapping any one is a DI registration change.
- **Data via domain models:** each stage hands the next a typed result (`IntentResult` → `ComplexityResult` → `PrivacyResult` → `Decision` → `ExecutionPlan` → `RoutingDecision` → `ChatResponse` → `AgentResponse`).
- **Events for observability:** `request.received` / `request.analyzed` / `request.routed` / `request.completed` are published to the bus; future subsystems (memory, workspace engine, UI bridge) subscribe without the Master knowing.
- **The Master never sees a vendor:** the Router picks `provider_id`+`model_id`; the Executor resolves them through `ProviderManager.get()` — the same interface the Master would use for anything else. No vendor name appears in `master/`, `router/`, `decision/`, `execution/`, or `analyzers/`.

## Sequence Diagram (one request)

```
User/API     Master        Intent  Complexity  Privacy  Decision  Planner   Router    Executor   ProviderManager   ModelProvider
  │           │             │         │          │        │         │        │          │           │                │
  │  prompt   │             │         │          │        │         │        │          │           │                │
  │──────────▶│             │         │          │        │         │        │          │           │                │
  │           │──hardware──▶│         │          │        │         │        │          │           │                │
  │           │◀──profile───│         │          │        │         │        │          │           │                │
  │           │──analyze───▶│         │          │        │         │        │          │           │                │
  │           │◀──intent────│         │          │        │         │        │          │           │                │
  │           │──analyze────────────▶│         │        │         │        │          │           │                │
  │           │◀──complexity─────────│         │        │         │        │          │           │                │
  │           │──analyze──────────────────────▶│        │         │        │          │           │                │
  │           │◀──privacy──────────────────────│        │         │        │          │           │                │
  │           │──decide───────────────────────────────▶│        │         │        │          │           │                │
  │           │◀──Decision─────────────────────────────│        │         │        │          │           │                │
  │           │──plan─────────────────────────────────────────▶│         │        │          │           │                │
  │           │◀──ExecutionPlan───────────────────────────────│         │        │          │           │                │
  │           │──route──────────────────────────────────────────────────▶│        │          │           │                │
  │           │◀──RoutingDecision────────────────────────────────────────│        │          │           │                │
  │           │──health/availability──────────────────────────────────────────────│          │           │                │
  │           │──execute──────────────────────────────────────────────────────────────────────▶│           │                │
  │           │──────────────────────────get(provider)───────────────────────────────────────▶│           │                │
  │           │──────────────────────────────────────────────────────────chat()─────────────────────────────▶│                │
  │           │──────────────────────────────────────────────────────────◀──normalized response────────────────│                │
  │◀──AgentResponse + DecisionTrace ───│        │         │          │        │         │        │          │           │                │
```

## Class Diagram (core pipeline)

```
┌───────────────────────┐
│     MasterAgent       │
├───────────────────────┤
│ process(prompt) →     │
│   AgentResponse       │
└──────┬────────────────┘
       │ uses (all via contracts)
       ▼
┌──────────────────────────────────────────────────────────────┐
│  contracts                                                    │
│  IntentAnalyzer   ComplexityAnalyzer   PrivacyAnalyzer        │
│  DecisionEngine   ExecutionPlanner     Router   Executor      │
│  HardwareProvider ModelRegistry ProviderManager ConfigProvider │
└──────────────────────────────────────────────────────────────┘
       ▲                       ▲
       │ implements            │ implements
┌──────┴────────┐   ┌──────────┴─────────┐
│ analyzers/    │   │ decision/ router/  │
│ intent        │   │ execution/         │
│ complexity    │   │ master/            │
│ privacy       │   └────────────────────┘
└───────────────┘
Domain (shared types): IntentResult, ComplexityResult, PrivacyResult,
Decision, ExecutionPlan, RoutingDecision, DecisionTrace, AgentResponse
```

## Routing Decision Examples

| Prompt | Intent | Complexity | Privacy | Routed to | Why |
|---|---|---|---|---|---|
| `what is 5 x 5` | general | 10 | BALANCED | local (qwen3:4b) | trivial, stays local |
| `fix this python bug` | coding | 50 | BALANCED | local with coding ability | local-first, coding capability required |
| `summarize this article` | writing | 40 | BALANCED | local | low cost, no sensitivity |
| `research the latest literature` | research | 90 | CLOUD_REQUIRED | cloud model | needs internet + heavy reasoning |
| `help with my credit card details` | general | low | LOCAL_ONLY | local only | sensitive data never leaves device |

## API: `POST /request`

```powershell
$body = @{ prompt = "What is 5 x 5?" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/request" -Method Post -ContentType "application/json" -Body $body
```

Response (abridged):

```json
{
  "response": "twenty five",
  "intent": "general",
  "complexity": 10,
  "privacy": "BALANCED",
  "provider": "ollama",
  "model": "qwen3:4b",
  "execution_plan": { "strategy": "local", "steps": ["..."], "estimated_latency_tier": "fast" },
  "decision_trace": {
    "intent": "general", "intent_confidence": 0.4, "complexity": 10,
    "privacy": "BALANCED", "internet_required": false,
    "hardware": { "cpu": "AMD64 ...", "ram_gb": 27.9 },
    "provider": "ollama", "model": "qwen3:4b",
    "reason": "best fit: qwen3:4b via ollama (...)",
    "execution_time_ms": 23773.5, "candidate_count": 1
  },
  "latency_ms": 23773.5
}
```

## Phase 2 Verification Checklist

- [x] `pytest` → **118 passed** (16 test files; analyzers, decision engine, router V2, executor, master, performance store; mock providers, no internet)
- [ ] `POST /request` returns `response` + `intent` + `complexity` + `privacy` + `provider` + `model` + `execution_plan` + `decision_trace` + `latency_ms`
- [ ] `GET /status` still works (Phase 1 regression)
- [ ] Routing honors privacy: sensitive prompt → local model; internet prompt → cloud-required decision
- [ ] Router never selects a model from an unhealthy/unregistered provider (health gate)
- [ ] Decision trace is present for every request (`decision_trace` object)
- [ ] Live check with Ollama: pull a model, add it to `config.toml` with matching `id`, and see a real answer with the trace

> **Note:** Phase 2 now includes **model-presence gating** — the router filters out registry models that are not actually installed on the provider. The registry also **auto-syncs** discovered models (e.g. `qwen3:4b` from Ollama) into the routing catalog so routing works without manual config edits. If a catalog model is not installed and no discovered model satisfies the request, you'll get a clear "no model could satisfy" guidance message instead of a 503.

---

# Phase 2.5 — Intelligent Routing Engine

## What Changed

The router stopped treating models as interchangeable workers and started acting as a **supervisor that understands WHY each model exists**. Providers remain dumb pipes — only the Master decides.

| Area | Phase 2 (V1) | Phase 2.5 (V2) |
|---|---|---|
| Model knowledge | flat capability booleans | weighted specialist profiles (`ModelCapabilities.score_for`) |
| Vision handling | none | vision/OCR/PDF hints in intent → hard requirements; vision models score **zero** on text-only tasks |
| Complexity | static keywords | dynamic 0–100 heuristic calibrated to spec examples (Hello=1 … ERP=95) |
| Latency | latency tier only | size+hardware+prompt+expected-output estimate, blended with learned history |
| Learning | none | `PerformanceStore` records latency/tokens/s/success/timeout per model; router blends history in |
| Explainability | reason string | full trace: capability score, excluded models + reasons, complexity, estimated latency, expected output tokens, historical performance |

## Calibration (complexity analyzer)

| Prompt | Score |
|---|---|
| `Hello` | 1 |
| `2+2` | 2 |
| `Write a resignation letter` | 20 |
| `QuickSort` | 35 |
| `Explain Quantum Computing` | 55 |
| `Design a distributed chat architecture` | 80 |
| `Build a complete ERP platform` | 95 |

## Router V2 Scoring Rules

- **Required capabilities** dominate (weight 0.15–0.55 per capability); **preferred capabilities** boost at 60% weight.
- **Unused specializations are penalized**: vision −0.35, OCR −0.25, PDF −0.20 when the task needs no vision; coding specialists lose −0.35 on non-coding tasks.
- **Deep-reasoning bonus**: +0.4 × reasoning for complexity ≥ 75.
- **Structural biases stay small**: local +0.05, privacy +0.05, latency tier up to +0.1, cost capped at −0.08.
- **Hard gates never negotiate**: privacy, hardware (RAM/VRAM), provider health, installation status, required capabilities, embeddings-only models (can never serve text), LONG_CONTEXT < 128k.

## Performance Learning Loop

`FilePerformanceStore` persists per-model statistics to `<home>/data/performance.json` (JSON, thread-safe, survives restarts, never raises on I/O). After ≥3 samples the router blends measured latency/tokens-per-second into its prediction. Recording happens inside the Master for every executed request.

## Backward Compatibility

- `Router.route()` signature unchanged — Phase 2.5 inputs are optional keyword-only parameters (`complexity`, `prompt`, `performance`).
- All existing contracts, providers, and the API surface are untouched; new entities are additive-only fields with defaults.
- No vendor or model names are hardcoded anywhere — every profile (including `qwen2.5:3b`, `qwen2.5-coder:7b`, `llama3.1:8b`, `qwen2.5-vl:7b`, `nomic-embed-text`) lives in `config.toml`.

## Phase 2.5 Verification Checklist

- [ ] `pytest` → 144 passed (includes router V2 scenarios, complexity calibration, modality detection, performance store, model lifecycle)
- [x] Greeting routes to a general chat specialist, coding tasks to a coding specialist, reasoning to a reasoning specialist
- [x] Vision task → vision model; text-only task → vision model never selected (score 0.0 + excluded reason)
- [x] `decision_trace` explains every choice: capability score, excluded models + reasons, estimated latency, expected output tokens, historical performance
- [x] Performance file grows after requests and feeds back into latency prediction

---

# Phase 2.5+ — Model Lifecycle & RAM Management

## What Changed

Providers stay dumb pipes, but the Master no longer ignores what is already
resident. The `ModelLifecycleManager` tracks the load state of every model and
makes RAM a first-class resource:

- **Tracked states**: `offline → loading → active → idle → unloading → offline`
  (plus `failed`). The Master marks a model *active* before generation and *idle*
  after each response; the provider's `/api/ps` is the source of truth for what
  is actually resident.
- **Auto-unload idle models**: after `idle_timeout_small`/`idle_timeout_large`
  seconds an idle model is unloaded via the provider's `unload_model` (Ollama
  `keep_alive: 0`). Large models get a shorter timeout.
- **Embedding model is protected**: the embedding model (`nomic-embed-text`) is
  never auto-unloaded while `keep_embedding_loaded` is on, and is preloaded at
  startup (`keep_alive: -1`).
- **Reuse before routing**: if an already-loaded model passes the same hard
  gates the router uses and its capability score is within
  `prefer_loaded_model_margin` (default 5%, relative) of the best cold
  candidate, the Master reuses it — no load latency, no extra RAM.
- **Memory-pressure reaction**: when free RAM drops below
  `low_memory_threshold`, every idle model is immediately evicted except models
  in use and (optionally) the embedding model.
- **Capacity ceiling**: `max_loaded_models` evicts the least-recently-used idle
  model when the ceiling is exceeded.
- **Consistency**: periodic cleanup reconciles Synapse's view with the provider
  (`/api/ps`), adopting models loaded outside Synapse (e.g. `ollama run`) and
  sampling their RAM footprint.
- **Observability**: `GET /lifecycle` returns loaded models, settings, aggregate
  metrics (loads/unloads/reuses, RAM reclaimed, cleanup cycles, peak concurrency)
  and per-model metrics. Dedicated events fire on load/unload/idle-timeout/
  memory-pressure/reuse.

## How It Delegates to Providers

The manager calls only `ModelProvider` lifecycle methods —
`list_loaded()`, `is_loaded()`, `load_model()`, `unload_model()`. Providers
implement them natively (Ollama uses `/api/ps` and `keep_alive`), and providers
that don't support lifecycle return `False` — lifecycle then simply skips them.
No vendor code ever lives above the provider layer.

## API: `GET /lifecycle`

```json
{
  "enabled": true,
  "loaded_models": [{"model_id": "llama3.2", "state": "idle", "ram_gb": 2.39, "idle_s": 3.1}],
  "loaded_count": 1,
  "settings": {"idle_timeout_small_s": 60, "low_memory_threshold_gb": 4, "...": "..."},
  "system_metrics": {"total_loads": 2, "total_unloads": 1, "...": "..."},
  "model_metrics": [{"model_id": "llama3.2", "load_count": 1, "...": "..."}],
  "cleanup_stats": {"total_cleanup_cycles": 4, "...": "..."}
}
```

## Phase 2.5+ Verification Checklist

- [ ] N models loaded concurrently on a RAM-capped machine
- [x] Idle model auto-unloaded after its timeout (verified against live Ollama: model evicted, RAM reclaimed tracked)
- [x] Second request to a resident model is served via `model.reused` (no cold load)
- [x] Embedding model preloaded at startup and never evicted by pressure
- [x] `/lifecycle` reports loaded state + aggregate + per-model metrics

---

# Phase 3 — Task Orchestration & Workspace Memory

## What Changed

Phase 1–2.5+ turned Synapse into a *smart single-shot router*. Phase 3 makes it
a *step planner*: one complex request may become many self-contained tasks, each
routed, executed, and attributed independently, then merged back into one answer.
The Master still orchestrates and never touches a vendor.

```
                 ┌──────────────────────────────────────────────┐
   prompt ──────▶│                Master Agent                  │
                 │  1. Task Planner      → TaskDAG (parallel-   │
                 │                           ready task graph)  │
                 │  2. Per-task routing  → loaded-model reuse   │
                 │                           first, then Router  │
                 │  3. Execution         → each task, one call  │
                 │  4. Result Synthesizer → merge into answer   │
                 │  5. Workspace Memory   → conversation log    │
                 │                            (scoped, scoped)  │
                 └──────────────────────────────────────────────┘
                       │                │                │
                       ▼                ▼                ▼
                 HeuristicTaskPlanner ─▶ executes each  TemplateSynthesizer
                 (deterministic)        task via Router  ─▶ merged response
                                        + Executor
                                        (Workspace Memory recalls relevant
                                         history into every task prompt)
```

## New Subsystems

| Subsystem | Module | Responsibility |
|---|---|---|
| **Task Planner** | `planner/heuristic.py` | Decomposes high-complexity prompts into a `TaskDAG`: numbered lists, "then", newline-separated requests, and imperative "and" clauses. Classifies each part via `_KIND_HINTS` and applies per-kind capability profiles. Appends an explicit `t-synthesis` task. Deterministic, offline. |
| **Task DAG** | `domain/tasks.py` | Parallel-ready execution graph (`TaskDAG`) with `topological_order()` (Kahn's, cycle-safe, deterministic). Execution is sequential today; any task is runnable once its dependencies complete. |
| **Workspace Memory** | `memory/workspace_memory.py` | Scoped (conversation/project/global) memory persisted to `<data>/memory/*.json`. Semantic search via provider embeddings (`embed()` — Ollama `/api/embed`, `nomic-embed-text`) with cosine similarity; degraded keyword-overlap fallback when embeddings are unavailable or fail. Memory **never breaks the pipeline**. |
| **Result Synthesizer** | `synthesis/merger.py` | Merges completed per-task outputs into one response: headed sections `## Coding — …` with `*via {model}*` attribution. Single task passes through untouched. |
| **Reliability router** | `router/router.py` | After ≥3 recorded executions, a model's capability score is scaled by its historical `success_rate` (up to a 15% penalty) minus a failure/timeout penalty (5% each). Flaky models lose to stable equivalents with identical capabilities. |
| **Execution Graph** | `domain/tasks.py` | `AgentResponse.execution_graph` carries nodes (task, kind, status, provider, model, score, latency, reason, memory-context flag), edges (dependencies), `execution_order`, `synthesized`, `total_latency_ms`. |

## How the Master Orchestrates Now

1. **Analyze + decide** (unchanged pipeline): `intent → complexity → privacy →
   decision`.
2. **Plan**: `TaskPlanner.plan(...)` returns a `TaskDAG`; the fallback (no
   planner wired) is a single `t1` task. New event `task.planned`.
3. **Execute the DAG**: for each task in topological order — skip `t-synthesis`
   (merged by the synthesizer, never a model call); build a per-task `Decision`
   from the task's capability profile; recall relevant conversation/project
   history (`memory.search`) and prepend it to the prompt; route via
   loaded-model reuse first, then `Router.route(...)` (with `performance`);
   mark `routed → completed/failed`, record latency, publish
   `task.routed`/`task.completed`/`task.failed`. Every task contributes a
   `GraphNode`; edges and `execution_order` come from the DAG. The synthesis
   node is appended as `provider_id="synthesizer"`.
4. **Synthesize**: `Synthesizer.synthesize(...)` merges finished task outputs
   (or the single task passes through). New event `response.synthesized`.
5. **Save memory**: user prompt + assistant response written to the
   `conversation` scope. New event `memory.written`.
6. **Return**: `AgentResponse` now includes `execution_graph` alongside
   `decision_trace`.

## Reliability Routing

`FilePerformanceStore.record()` now also tracks **failures** and their rate.
`Router` demotes candidates with weak history (`_RELIABILITY_MIN_SAMPLES = 3`,
max penalty 15% on success rate + 5% per failure/timeout), surfaces a
`reliability` object in `candidates`, and appends
`reliability 90% (10 samples)` to the routing reason.

## API: `POST /request` — execution graph

```json
{
  "response": "Here is the consolidated result combining 3 sub-tasks (1 model(s)).\n\n## Coding — … *via llama3.2*\n…",
  "intent": "coding", "complexity": 61, "privacy": "BALANCED",
  "provider": "ollama", "model": "llama3.2",
  "execution_graph": {
    "nodes": [
      {"task_id": "t1", "kind": "coding", "status": "completed",
       "provider_id": "ollama", "model_id": "llama3.2", "order": 1,
       "memory_context_used": false},
      {"task_id": "t-synthesis", "kind": "synthesis", "status": "completed",
       "provider_id": "synthesizer", "model_id": "synthesizer", "order": 4,
       "reason": "merged 3 task outputs"}
    ],
    "edges": [{"source": "t1", "target": "t-synthesis"}],
    "execution_order": ["t1", "t2", "t3", "t-synthesis"],
    "synthesized": true,
    "total_latency_ms": 4123.4
  }
}
```

## Phase 3 Verification Checklist

- [x] `pytest` → **223 passed** (25 files; adds `test_workspace` — files,
      parsers, chunking, vector stores, retrieval, vision, jobs, facade
      pipeline decisions, Master integration — and `test_api_workspace` for
      the /files, /jobs, /workspace, /ui endpoints — all offline)
- [x] Complex numbered prompt → response opens with "consolidated result" and
      the execution graph shows `t1/t2/t3` completed + `t-synthesis`
- [x] Simple prompt → single `t1` node passes through unchanged
- [x] Flaky model (≥3 samples, low success rate) loses to a stable equivalent
- [x] `conversation.json` memory file grows with each request and relevant
      history is recalled into later task prompts
- [ ] Live check with Ollama: `POST /request` a 2-part numbered prompt and link
      the returned graph nodes to `/lifecycle` loaded models

---

# Phase 4 — AI Workspace (multimodal files, RAG, vision, code scan)

## What Changed

Synapse stopped being just a chat box. It is now a **workspace**: drop files
in, ask questions about them, and get answers grounded in local indexes —
with every file's content enforced **local-only**.

```
   drag & drop / picker (web UI at /ui)
        │
        ▼
  POST /files ──▶ File Manager (sha256 dedupe, catalog, blob storage)
        │
        ▼
  POST /files/{id}/index ──▶ IndexingJob (background thread, progress 0-100)
        │                      parsed by extension → pipeline decision:
        ├── image    → Vision Pipeline (qwen2.5vl) → description embedded
        ├── document → pypdf/python-docx/text → chunk → embed → vector store
        └── code     → line-bounded chunks → embed → store (+ TODO scan later)
        │
        ▼
  POST /request {"prompt", "files": [ids]} ──▶ Master Agent
        │                                         │
        │    attached images → vision.analyze()    │
        │    attached docs   → Retrieval.retrieve()│
        │    attached code   → scan_code()         │
        │    no files        → auto-retrieve over ALL indexed files
        ▼                                         ▼
  AgentResponse + workspace outcome (files used, hits, matches, local_only)
```

## New Subsystems

| Subsystem | Module | Responsibility |
|---|---|---|
| **File Manager** | `workspace/files.py` | Upload with sha256 dedupe, atomic `catalog.json`, blob storage under `<data>/workspace/files/`. Pure bookkeeping, never raises. |
| **Document Parser** | `workspace/parsers.py` | PDF via pypdf (page-aware), DOCX via python-docx (paragraphs + tables), text with encoding fallbacks. |
| **Chunking** | `workspace/chunking.py` | Documents: char chunks (1200/200) on paragraph seams with a hard-split fallback. Code: line-bounded chunks with line overlap, preserving start/end line numbers. |
| **Embedding Engine** | `workspace/embeddings.py` | Batched embeddings (batch 16) via the provider `embed()` capability (`nomic-embed-text`); returns `None` to degrade to keyword search when unavailable. |
| **Vector Store** | `workspace/vectors.py` | `faiss` backend = `IndexFlatIP` over L2-normalized vectors (cosine), persisted as `.index` + `.npy` + JSON metadata; `local` backend = pure numpy/JSON with identical semantics; `create_vector_store` falls back automatically. |
| **Retrieval** | `workspace/retrieval.py` | Semantic search (threshold `min_similarity`) → keyword-overlap fallback; deterministic regex code scan (`scan_code`); context-block renderer for prompt injection. |
| **Vision Pipeline** | `workspace/vision.py` | Base64 image → chat `images` field → vision model (`qwen2.5vl:7b`). Model resolved from config with capability-based fallback; raises `WorkspaceVisionUnavailable` when none installed. |
| **Indexer + Jobs** | `workspace/indexer.py`, `workspace/jobs.py` | Background parse → chunk → embed → store with stage + 0–100 progress; thread-safe `JobRegistry`. Indexed files become permanently queryable (follow-ups work without re-uploading). |
| **Workspace facade** | `workspace/workspace.py` | `prepare(prompt, file_ids)` — the single call the Master makes: resolves attachments, decides pipelines, runs retrieval/vision/code scan, applies the local-only guard, returns `(context, WorkspaceOutcome)`. |

## Privacy: enforced Local-Only

When `allow_cloud_forwarding = false` (the shipped default), **any** request that
uses workspace content is forced to the local-only execution path — the Master
overrides its decision with `PrivacyMode.LOCAL_ONLY` and `use_cloud_reasoning =
false` the moment workspace context or a vision analysis is attached. File
contents are read only by local parsers, local embeddings, and the local
vector store; they are never sent to a cloud provider.

## API

```powershell
# upload (indexing auto-starts; returns file + job_id)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/files" -Method Post -Form @{ file = Get-Item "notes.pdf" }

# indexing progress (stage + progress 0-100)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/jobs/<job_id>"

# workspace summary (counts + chunks + backend)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/workspace"

# ask with attachments (or no files → auto-retrieve over indexed files)
$body = @{ prompt = "Summarize the PDF"; files = @("<file id>") } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/request" -Method Post -ContentType "application/json" -Body $body
```

The response now carries a `workspace` object:

```json
"workspace": {
  "files_attached": ["3f9a1c2b4d01"],
  "files_used": ["3f9a1c2b4d01"],
  "pipelines": {"document": ["3f9a1c2b4d01"]},
  "retrieval": [{"file_id": "3f9a1c2b4d01", "file_name": "notes.pdf",
                 "chunk_index": 0, "text": "...", "score": 0.62, "page": 1}],
  "code_matches": [{"file_id": "...", "file_name": "app.py", "line": 7,
                    "text": "# TODO: add retry"}],
  "vision_descriptions": ["The screenshot shows..."],
  "context_chars": 4102,
  "local_only": true,
  "used_memory": true
}
```

## Frontend

- **Web UI**: `GET /ui` — drag & drop upload, live indexing progress, chat
  with per-file attachments and the workspace-outcome footer.

## Phase 4 Verification Checklist

- [x] `pytest` → **223 passed** — files/parsers/chunking/vector stores (faiss +
      local parity)/retrieval/vision/jobs/workspace facade/Master integration/API
- [x] Upload dedupes by content hash; delete removes blobs, catalog, and vectors
- [x] Indexing runs in the background with stage + progress and finishes for real PDF/DOCX/text
- [x] Images route to the vision pipeline; documents/code to RAG + code scan
- [x] `workspace.local_only=true` whenever workspace content is used (guard forced)
- [x] No files attached → follow-up questions auto-retrieve over all indexed files
- [ ] Live check with Ollama: upload a real PDF + screenshot, watch `/jobs` progress,
      ask questions grounded in them, confirm `local_only: true`

# Phase 6 — AI Workspace Execution (files on disk, multi-model pipelines, review, audit)

## What Changed

Synapse stopped producing only text: for file-producing requests it now makes the
task pipeline **do the work** — write files, validate them, review them, and log
every action. The Master still orchestrates and never touches a vendor.

```
  "Create a python app" (web UI / POST /request)
        │
        ▼
  Task Planner ──► t1 (coding, file_output=true, model hint → coder)
        │              │
        │              ▼ model returns a file manifest
        │        parse_file_manifest (JSON shapes / fenced blocks)
        │              ▼
        │        FileOperator.write ×N  →  <project>/work (atomic, contained)
        │              ▼
        │        OutputReviewer.validate_file (ast / json / braces / html)
        │              ▼
        │        t-review ──► strongest available reasoning model
        │              ▼
        ▼        ActionLog.append (plan, models, tools, files, failures)
  Response: per-file actions (created/modified/… + validation) + review
```

## New Subsystems

| Subsystem | Module | Responsibility |
|---|---|---|
| **FileOperator** | `workspace/operator.py` | Safe, contained file CRUD rooted at `<project>/work`. Every write is atomic (tmp + replace); paths are resolved and verified inside the root — traversal, absolute, and symlink escapes raise `WorkspaceSafetyError`; `read/rename/delete/list_tree/export/import` for the UI and onboarding. The only path to disk for generated output. |
| **Manifest Parser** | `workspace/manifest.py` | Turns a model response into file actions: JSON `{"files": [...]}`, `{"create": [...]}` / `{"rename"}/{delete}` forms, fenced code blocks, and flat maps; deduplicates by path; a code fence with language or `file_hint` falls back when no manifest is emitted; unsafe paths are reported as failed actions (never dropped silently). |
| **Output Reviewer** | `workspace/review.py` | Validates each written file (python → `ast`, json → `json.loads`, js/ts → brace balance, html → doctype/root, else non-empty UTF-8) and summarizes a write batch (`created N file(s) …`) into the answer. |
| **Review task** | `planner/heuristic.py` | `add_review` appends a `t-review` task depending only on the file tasks, routed to the strongest available **reasoning** model (capability-scored, health/install gated) to check the generated file set. |
| **Planner hints** | `planner/heuristic.py` | `_FILE_TASK_PATTERNS` detect create/write/build/generate/refactor prompts; `model_hints` map task kind → preferred model (e.g. `coding → qwen2.5-coder`); `file_hint` infers an extension from the prompt. |
| **Action Log** | `projects/actionlog.py` | Per-project append-only JSONL at `<project>/logs/actions.jsonl` (Phase 7: moved to internal `logs/actions/<pid>/actions.jsonl`); one record per disk-touching request: task plan, models used, tools (`filesystem.created/.modified/.renamed/.deleted`), per-file status + validation, failures, `execution_time_ms`. |
| **External projects** | `projects/manager.py` | `create(name, parent_dir=…)` builds `<parent>/<Name>/` and registers it; Phase 7 superseded the JSON registry with the SQLite `projects.db` (workspace paths only) + reconnect/verify/delete modes. |

## Four Realizations of "the model did something"

1. **Files land on disk** under `<project>/work`, atomically written with full
   path containment.
2. **Every file is validated** on write — a Python file that doesn't parse is
   recorded as a failed action with the compiler error, not silently accepted.
3. **A review pass runs** — a separate reasoning model examines the generated
   file set; its verdict is merged into the final answer.
4. **Nothing is invisible** — `GET /projects/{id}/work` lists produced files,
   `GET /projects/{id}/actions` shows the audit trail, and the `/request`
   response carries `actions[]` + `meta.created_files`.

## API (new in Phase 6)

```powershell
# create a project inside a user-chosen folder (optional context)
$body = @{ name = "Site"; location = "C:\dev" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/projects" -Method Post -ContentType "application/json" -Body $body

# list produced files
Invoke-RestMethod -Uri "http://127.0.0.1:8000/projects/<id>/work"

# read / write / rename / delete one produced file
Invoke-RestMethod -Uri "http://127.0.0.1:8000/projects/<id>/work/file?path=hello.py"

# audit trail for a project
Invoke-RestMethod -Uri "http://127.0.0.1:8000/projects/<id>/actions"
```

`POST /request` for a file prompt now returns, alongside the usual fields:

```json
"actions": [
  {"path": "hello.py", "action": "created", "status": "ok", "bytes": 16,
   "validated": true, "validation": "python syntax ok"}
],
"meta": {"created_files": 2, "task_count": 2}
```

## Phase 6 Verification Checklist

- [x] `pytest` → **409 passed** (adds `tests/test_phase6.py`: workspace creation at
      user folders + registry, filesystem safety, FileOperator CRUD, manifest
      parsing, planner file/review behavior, output validation, end-to-end
      multi-model file generation through the API, interruption recovery,
      session/work persistence across reboots — hermetic fake provider, offline)
- [x] Project created at `C:\<folder>\<slug>` gets the full project tree, and is
      rediscovered after a fresh `ProjectManager` via `registry.json`
- [x] Escape attempts (`../`, absolute paths, symlink) are rejected; files land
      under `<project>/work` only
- [x] "Create a python app" → `t1` (coder) writes `hello.py`+`README.md`, both
      validated, then `t-review` runs on the reasoning model
- [x] A failing sub-task still keeps the files the successful one wrote, and the
      failure appears in the project action log
- [ ] Live check with Ollama: create a project at a real folder, ask it to build
      a Python script, then open `<folder>/<Name>/hello.py` locally

---

# Phase 7 — Workspace Engine Architecture (IDE-grade separation, registry, picker, lifecycle)

## What Changed

The Workspace System now follows a professional IDE architecture: **the
user's folder is the project and nothing else**. All Synapse metadata was
moved out of project folders into OS appdata (`%LOCALAPPDATA%\Synapse`),
keyed by a unique project id stored in a SQLite registry. Projects are
created with a native OS folder picker, deleted in two modes (detach vs
erase), reconnected when moved, and all temporary/orphaned state is swept
at startup and shutdown.

## New Subsystems / Reworked

| Subsystem | Module | Responsibility |
|---|---|---|
| **SQLite registry** | `projects/manager.py` | `projects.db` stores only each project's `workspace_path`. Unique slug ids key all internal storage. Persistence across restarts; `verify()` reports workspaces missing on disk. |
| **Native folder picker** | `projects/picker.py` | OS-native directory dialog (Windows `FolderBrowserDialog` via PowerShell, `zenity`/`kdialog` on Linux, `osascript` on macOS) exposed as `POST /projects/pick`. The user picks a parent folder; Synapse creates `<parent>/<Name>`. |
| **Internal storage** | `config/paths.py` | Home moved to `%LOCALAPPDATA%\Synapse` (was `%APPDATA%\SynapseOne`). Layout: `projects.db`, `chats/<pid>/`, `memory/<pid>/`, `embeddings/<pid>/`, `logs/actions/<pid>/`, `cache/`, `config/settings/`. |
| **Workspace lifecycle** | `projects/system.py` | `create` (folder + README seed + best-effort `git init`) → register → use; `delete` detaches (keeps files) or erases the folder; `reconnect` repoints a moved workspace; `cleanup()` runs at boot and shutdown. |
| **Cleanup** | `projects/system.py` | Startup/shutdown sweep: temp cache wiped, internal per-project state for unregistered ids removed, leftover empty internal workspaces from failed creations deleted (never touches user-chosen folders). |
| **FileOperator** | `workspace/operator.py` | Rooted at the workspace folder (agent files are real user files and persist in Git); `.git/` internals are read-only and skipped in listings. |
| **Domain** | `domain/projects.py` | `ProjectInfo.workspace_path` (with `root` alias) + `exists` flag for the reconnect UI. |

## API (new in Phase 7)

```powershell
# open the OS-native folder picker (returns the chosen parent dir)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/projects/pick" -Method Post

# create a project inside a user-chosen parent folder
$body = @{ name = "Portfolio Website"; parent_dir = "D:\Projects" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/projects" -Method Post -ContentType "application/json" -Body $body

# delete: detach (default) or also erase the workspace folder
Invoke-RestMethod -Uri "http://127.0.0.1:8000/projects/<id>" -Method Delete
Invoke-RestMethod -Uri "http://127.0.0.1:8000/projects/<id>?delete_workspace=true" -Method Delete

# reconnect a project whose workspace was moved on disk
$body = @{ workspace_path = "D:\Projects\Portfolio Website" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/projects/<id>/reconnect" -Method Post -ContentType "application/json" -Body $body

# health: does the workspace folder still exist?
Invoke-RestMethod -Uri "http://127.0.0.1:8000/projects/<id>/verify" -Method Post
```

`GET /projects` now returns `workspace_path` + `exists` per project; the web
UI shows a "locate folder" reconnect banner for missing workspaces and a
two-step delete (keep files / erase files).

## Phase 7 Verification Checklist

- [x] `pytest` → **579 passed** — Action-Based Execution Engine: 7-kind classifier (Chat Response / File Creation / File Modification / Project Generation / Project Analysis / Documentation / Workspace Operation); ActionEngine with backend filesystem tools (create_folder/write/read/edit/rename/move/delete/search); delete confirmation gate; existence verification; manifest extensions (folders + edit ops); planner modification patterns; context injection for modification/analysis; model-free workspace operations (list/search/read/rename/delete/create_folder), including explicit `read <file>` which prints the file's contents back without consulting the model (whole-project read requests like "read the project" still route to Project Analysis); summary-only responses for file kinds; instruction injection into file-task prompts (models are told to return a JSON manifest); tolerant fallback converter turns prose + fenced code into one file per block (`file:` headers, language default names); generation prompts (blog/store/todo/…) no longer misclassify as chat; project registry is SQLite; workspace folders contain only project files (README seed + agent output); chats/memory/embeddings/action logs live in internal appdata storage keyed by project id; native picker endpoint (mocked), reconnect, verify, delete detach/erase modes, cleanup sweeps (temp, orphans, stale empty workspaces), `.git` read-only guard
- [x] **Phase X universal workspace tool** — every agent (any request kind: chat, analysis, research, docs, coding) can read AND write inside the project folder: workspace context is injected into every task (was only modification/analysis); non-file tasks get a soft "you may produce files" instruction (strict JSON-only manifest parse — ordinary prose never becomes files); file-writing tasks keep the tolerant prose/fenced-code converter; planner gains a "save/write … to <file.ext>" file pattern so named destinations become file tasks for ANY kind (analysis → report.md, research → findings.md); file-producing tasks route TOOLS/JSON-capable and keep pure chat on normal routing; hallucination self-correction no longer mangles manifest paths (it was rewriting "analysis.md" → "`analysis.md` (not found in workspace)" whenever the workspace had any files); `HeuristicFactChecker` tolerates non-string sources (RetrievedChunk objects) instead of crashing
- [x] `scripts/verify.py` boots clean with home at `%LOCALAPPDATA%\Synapse`
- [x] Live check: create a project via the picker, ask Synapse to build a
      script, confirm `D:\Projects\<Name>\hello.py` + a Git repo, then delete
      the project with "keep files" and reconnect to the same folder
- [x] Live check: "Create a portfolio website" → Synapse creates `assets/`, `css/`, `js/` + `index.html`/`style.css`/`script.js` in the workspace; chat shows only the action summary (no code blocks)

## Phase XI (Live Execution Timeline) Verification Checklist

- [x] `pytest` → **581 passed** — live execution timeline replaces the static "thinking…" indicator: new `POST /request/stream` SSE endpoint runs the request on a background thread and streams timeline-relevant bus events (request received → workspace read → per-file reads → model load/reuse → per-task generation → file writes/edits → finished) plus a final `done` event carrying the identical `AgentResponse` payload as `/request`; `api/timeline.py` hub maps legacy bus events (`MODEL_LOADED`, `MODEL_REUSED`, `RETRIEVAL_RAN`, `REQUEST_COMPLETED`, …) into structured steps; `Events.TIMELINE` + `_publish_timeline` in the master agent; `ActionEngine` gains `on_step` callbacks (`apply`, `run_workspace_ops`) and `on_file` (`build_context`) so reads/writes surface live; UI streams the SSE with `ReadableStream`, renders an emoji timeline (🧠 📂 📄 🔎 🤖 ✍ 💾 🔄 ✅) with pulse-on-last-line, dedupe, auto-scroll — and removes the strip the moment the response is ready so the reply stream types in clean; `/request` unchanged (chat recording, meta incl. intent, 503 on provider failure); the UI reads meta from the streamed response instead of rebuilding it client-side

## Phase XII (Per-Chat Isolation + Auto-Rename) Verification Checklist

- [x] `pytest` → **586 passed** — each chat is truly isolated: conversation memory entries are stamped with a `conversation` id (`MemoryEntry.conversation`) so `save`/`search`/`recent` only return that chat's own history (`WorkspaceMemory` + `MemoryStore` contract); the Master passes `conversation_id=chat.id` straight through `process()` → `_execute_dag_with_quality` (context-search + conversation-history) and `_save_memory`, so chat A never sees chat B's memory; chats already had isolated per-chat title/history/attachments files; auto-rename now fires from the **first user message in any default chat** (fixes the `ensure_chat` "General Discussion" case that never renamed before) and collapses whitespace/markdown into a clean ≤50-char one-line title, while titles the user set explicitly are never overwritten

## Phase X (Workspace IDE) Verification Checklist

- [x] `pytest` → **579 passed** — opening existing projects: `ProjectManager.register` adopts any live folder (name derived from folder); `projects/scanner.py` walks a folder honouring ignore rules (`.git`, `node_modules`, `venv`, …), detects the language/framework stack (manifests: `package.json`, `pyproject.toml`, `go.mod`, `Cargo.toml`, `Gemfile`, `composer.json`, `pubspec.yaml`, `*.csproj`, `mix.exs`; extension histogram fallback), and imports files into the project's index with background indexing jobs + sha256 dedupe (rescan is idempotent); `ProjectInfo` now carries `language`/`framework`/`indexed_count` persisted in registry settings; new endpoints `POST /projects/{id}/scan`, `GET /projects/{id}/dashboard` (stack, index health, recent files, chat count) and `GET /files/{id}/content` (read a workspace file's text); UI gains an "Open Existing Project" button (picks a folder → adopts it → auto-indexes) plus a "new chat" button, and a "scan files" re-index action; the agent can read AND write inside an opened project — "read app/main.py" returns the file's contents directly, "analyze the project and save the summary to analysis.md" writes the report (universal workspace tool: every agent has full folder access like the coding agents, routed to tool/JSON-capable models)

---

# Adaptive Model Installation (SetupModels.bat)

Synapse installs Ollama models **hardware-aware**: machines with <8 GB RAM or
older CPUs (Tier 1) get only lightweight models; machines with ≥8 GB RAM and
modern CPUs (Tier 2) get the standard set. No unnecessary model is ever
downloaded.

| File | Role |
|---|---|
| `models.json` | **Single source of truth** — Ollama URL, hardware thresholds, and the per-tier model lists. Change model recommendations here, never in code. |
| `SetupModels.bat` | Windows installer: checks/installs Ollama (silently), waits for the service, detects hardware, picks the tier, pulls **only missing** models, prints a summary (installed / skipped / failed). |
| `setup_models.ps1` | Shared helper (tier detection + installed-model status) used by the bat and the launchers. |
| `core/start.ps1`, `core/start.bat` | Onboarding gate: at launch, missing models trigger *"Synapse needs to install AI models before first use."* → prompt → `SetupModels.bat` → normal startup continues. Skip with `SYNAPSE_SKIP_MODEL_SETUP=1`. |

```powershell
# from the repo root
.\SetupModels.bat            # interactive install
.\SetupModels.bat --yes      # install and exit without pausing
.\SetupModels.bat --check    # report only; exit 0=ready 2=no Ollama 3=missing
```

**Tier rules (configurable in `models.json`):**

| Tier | Condition | Models |
|---|---|---|
| 1 | RAM < 8 GB **or** older CPU (e.g. pre-Ryzen-3000 / pre-Intel-10th-gen) | `llama3.2:1b`, `qwen2.5:3b`, `qwen2.5-coder:1.5b`, `qwen3:1.7b`, `all-minilm`, `moondream` — **no 7B/8B models** |
| 2 | RAM ≥ 8 GB **and** modern CPU (Ryzen 3000+, Intel 10th gen+, Core Ultra) | `llama3.1:8b`, `qwen2.5:7b`, `qwen2.5-coder:7b`, `qwen3:4b`, `qwen2.5vl:7b`, `nomic-embed-text` |

The router automatically works with whichever set was installed: the registry
syncs live-installed models into the catalog every request, `config.toml`
carries capability profiles for both tiers (uninstalled profiles are ignored),
and the embedding/vision engines fall back to the installed embedding/vision
models (`all-minilm` / `moondream` on Tier 1, `nomic-embed-text` /
`qwen2.5vl:7b` on Tier 2) when the configured one is absent. The Ollama
provider also skips an uninstalled configured default model and uses the first
installed one instead.

---

## Phase Roadmap

| Phase | Scope |
|---|---|
| **1 (done)** | Foundation: models, contracts, providers, registry, hardware, events, config, DI, logging |
| **2 (done)** | Master Agent + Decision Engine: intent/complexity/privacy analyzers, planner, router V1, executor, `/request` API, decision trace |
| **2.5 (done)** | Intelligent Routing Engine: specialist profiles, vision exclusion, calibrated complexity, latency prediction, performance learning loop, explainable traces |
| **2.5+ (done)** | Model Lifecycle: RAM-resident tracking, idle auto-unload, embedding pinning, loaded-model reuse, memory-pressure eviction, `/lifecycle` metrics |
| **3 (done)** | Task Orchestration: Task Planner + task DAG, per-task routing, Workspace Memory (conversation/project/global), Result Synthesizer, reliability-adjusted routing, execution graph on every response |
| **4 (done)** | AI Workspace: multimodal files (images/PDF/DOCX/code), automatic pipeline decisions, local RAG with FAISS + `nomic-embed-text`, vision answers, code scan, background indexing jobs, web UI at `/ui`, enforced local-only |
| **5 (done)** | Workspace System: multi-project isolation (files, vector store, memory, chats), durable chat persistence, session recovery, server-backed project/chat UI |
| **6 (done)** | AI Workspace Execution: projects in user folders + registry, safe contained FileOperator, file-producing tasks write validated files to disk, manifest parsing, review pass on the strongest reasoning model, per-project action-log audit trail, `/work` + `/actions` endpoints |
| **7 (done)** | **Action-Based Execution Engine: 7-kind classifier, ActionEngine (backend FS tools), manifest extensions (folders/edit), model-free workspace ops, summary-only responses, context injection, existence verification** |
| **X (done)** | **Workspace IDE: "Open Existing Project" (adopt any folder, auto-scan + index + stack detection), project dashboard (language/framework/index status/recent files), file content reads, sidebar scan action, agent read/write on project files — universal workspace tool: every agent type (chat/analysis/research/coding) can read and write inside the folder, routed to tools/JSON-capable models** |
| **11 (done)** | **Live Execution Timeline: `/request/stream` SSE streams real agent activity (request analysis → workspace/file reads → model load → task generation → file writes → finished) while the UI renders a live emoji timeline instead of a generic "thinking…" loader, removing it once the reply is ready** |
| **12 (done)** | **Per-chat isolation + auto-rename: conversation memory is scoped to the owning chat (isolated history, context, memory, title), and chats auto-title from the conversation's first message with clean whitespace handling** |
| **13 (done)** | **Workspace-First Operating Environment: every model call (including synthesis and review) is prefixed with a compact workspace brief (project identity, folder path, available file tools, default operation, file tree, recent modifications, current-chat summary); a deterministic access gate (`requires_workspace_access`) decides which prompts get ActionEngine excerpts while greetings stay chat-only; default dev-task verbs now target real files; the write→verify→index→remember loop confirms every created/modified file on disk, syntax-checks Python/JSON, refreshes the project index, and records the write in project memory (🛡️ verified / 🗂️ index timeline steps)** |
| **14 (done)** | **Adaptive Model Installation: hardware-tier-aware Ollama model setup (`SetupModels.bat` + `models.json` single source of truth — Tier 1 lightweight set for <8 GB RAM/older CPUs, Tier 2 standard set otherwise), automatic silent Ollama install with service wait, pull-only-missing with installed/skipped/failed summary, first-launch onboarding gate in `start.ps1`/`start.bat` ("Synapse needs to install AI models before first use." → SetupModels.bat → continue), tier-1 model profiles in the catalog, and tier-adaptive embedding/vision/default-model fallbacks so the router works with whichever set was installed** |
| **8** | Plugin framework + internal plugins |
| **9** | Workspace Engine hardening, multi-agent, voice |
| Later | Rust perf modules, marketplace |

---

*Synapse One: users define the outcome. Synapse handles everything else.*
