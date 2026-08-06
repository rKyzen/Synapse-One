# Synapse One
## The AI Operating Workspace

> **Version:** Vision Document v1.0
>
> **Codename:** Synapse One
>
> **Mission:**
> Build the world's first truly intelligent AI Operating Workspace that acts as an AI companion rather than just another chatbot.

---

# Table of Contents

1. Vision
2. Problem Statement
3. Core Philosophy
4. Design Principles
5. High-Level Architecture
6. Complete Workflow
7. Master Agent
8. Intelligence Engine
9. Model Routing System
10. Local-First AI
11. Cloud AI
12. Hybrid Execution
13. Project Memory
14. User Memory
15. Tool System
16. Workspace System
17. Multi-Agent Architecture
18. Execution Pipeline
19. Privacy
20. Hardware Awareness
21. AI Model Registry
22. Example Workflows
23. Future Vision

---

# Vision

Synapse One is not another AI chatbot.

It is not another note-taking application.

It is not another IDE.

It is not another productivity suite.

Synapse One is an **AI Operating Workspace** that understands what the user wants to accomplish and automatically orchestrates the required intelligence, tools, memory, and workflows.

Instead of users learning software, Synapse learns the user.

The user simply defines the goal.

Everything else is managed automatically.

---

# Problem Statement

Today's workflow looks like this:

User

↓

Google Search

↓

ChatGPT

↓

Gemini

↓

Claude

↓

Notion

↓

VS Code

↓

Google Docs

↓

Spreadsheet

↓

GitHub

↓

Browser

↓

Repeat...

Users constantly switch between applications and repeatedly explain context to multiple AI systems.

Every assistant starts from zero.

Every project loses context.

Every workflow becomes fragmented.

---

# Synapse One Solution

Synapse One replaces this fragmented workflow with one intelligent operating workspace.

User

↓

Synapse One

↓

Everything Happens Automatically

The platform understands:

- the task
- the project
- previous conversations
- files
- goals
- deadlines
- preferred writing style
- preferred programming language
- preferred frameworks
- device specifications
- privacy requirements

before choosing the best execution plan.

---

# Core Philosophy

Synapse follows one simple philosophy.

> Users should focus on goals.
>
> Synapse should handle everything else.

Users never need to ask:

- Which AI model should I use?
- Which application should I open?
- Which document contains my notes?
- Which version of this file is correct?

Synapse already knows.

---

# Core Design Principles

## Local First

Every task should be attempted locally first.

Cloud is never the default.

Cloud is the fallback.

---

## Privacy First

All personal data remains on the user's computer.

Nothing leaves the device unless absolutely necessary.

Cloud requests should be minimized.

Users remain in complete control of their information.

---

## Adaptive Intelligence

Synapse adapts to:

- user
- project
- hardware
- task complexity
- privacy mode
- available AI models

---

## Companion Instead of Chatbot

Synapse behaves like an intelligent operating companion.

It remembers projects.

It remembers goals.

It remembers workflows.

It assists continuously.

---

# High-Level Architecture

```
                         USER

                           │

                           ▼

                  Synapse Workspace

                           │

                           ▼

                 Master Supervisor AI

                           │

        ┌──────────────────┼───────────────────┐

        ▼                  ▼                   ▼

 Intent Engine      Memory Engine      Hardware Engine

        │                  │                   │

        └──────────────────┼───────────────────┘

                           ▼

                 Complexity Analyzer

                           ▼

                  Execution Planner

                           ▼

                 Model Routing Engine

                           ▼

      ┌───────────────┬──────────────┬──────────────┐

      ▼               ▼              ▼

 Local Models    Cloud Models      Tool Engine

      ▼               ▼              ▼

           Output Aggregation Layer

                      ▼

                    USER
```

---

# Complete Workflow

Every interaction follows the same execution pipeline.

---

## Step 1

User submits a request.

Example:

```
Build a Flutter weather app.
```

---

## Step 2

Master Agent receives the request.

The Master never immediately forwards the prompt.

Instead it starts thinking.

Questions:

What is the user trying to achieve?

Is this part of an existing project?

Do I need project memory?

Do I need internet?

Can this stay local?

How difficult is this?

Which tools are required?

---

## Step 3

Intent Detection

Example:

Task

Software Development

Framework

Flutter

Output

Code

Need Internet

No

Need Memory

Yes

Need Documentation

Yes

---

## Step 4

Project Memory

Synapse checks.

Does this belong to:

Hackathon

Personal App

Office Project

Research

Existing Repository

If yes:

Load entire project context.

---

## Step 5

Hardware Analysis

Synapse scans:

CPU

GPU

RAM

Battery

Storage

Internet

Available Models

Example:

Ryzen 5

16GB RAM

Integrated Graphics

Qwen Installed

Gemma Installed

DeepSeek Installed

---

## Step 6

Complexity Analysis

Complexity Score

0-100

Example

"What is 5 x 5?"

Score

2

↓

Local

---

Explain Photosynthesis

18

↓

Local

---

Build Website

74

↓

Hybrid

---

Research Paper

93

↓

Cloud

---

## Step 7

Privacy Analysis

Privacy Mode

Maximum

Balanced

Performance

Balanced is default.

---

## Step 8

Execution Planning

Master creates a workflow.

Example

Website

↓

Planning

↓

UI

↓

Frontend

↓

Backend

↓

Database

↓

Documentation

↓

Testing

---

Each becomes a task.

---

## Step 9

Task Assignment

Master delegates.

UI

↓

Claude

Frontend

↓

DeepSeek

Documentation

↓

GPT

Logo

↓

Image Generator

Testing

↓

Local Model

---

## Step 10

Parallel Execution

Every model works simultaneously.

Not sequentially.

Maximum performance.

---

## Step 11

Result Review

Master checks.

Does frontend match UI?

Does documentation match code?

Any conflicts?

If yes:

Send revision request.

---

## Step 12

Aggregation

Everything becomes one unified result.

User never sees multiple models.

Only Synapse.

---

# Master Agent

The Master is the brain.

It never specializes.

It supervises.

Responsibilities:

Intent Understanding

Planning

Delegation

Memory

Routing

Validation

Aggregation

Optimization

Privacy Enforcement

---

# Intelligence Engine

The Intelligence Engine decides:

Can this remain local?

Should this use cloud?

Can I split this task?

Should multiple models collaborate?

Can I reuse previous work?

---

# Local-First AI

Rule One

Always prefer local.

Cloud should only activate when:

Local confidence is too low.

Internet is required.

Context exceeds local capability.

Large reasoning required.

---

# Cloud AI

Cloud models become specialists.

Examples:

Research

Scientific reasoning

Large codebases

Advanced planning

Long context

Image generation

---

# Hybrid Intelligence

Example

Research Paper

↓

Cloud researches

↓

Local summarizes

↓

Local formats

↓

Master reviews

↓

Done

---

# Memory System

Memory exists in three layers.

## Project Memory

Stores:

Files

Chats

Tasks

Deadlines

Generated Content

Research

Images

Each project's memory is isolated — memories, files, and vector indexes from one project are never mixed into another. The WorkspaceSystem creates a separate `WorkspaceMemory` and `Workspace` instance per project, both rooted at the project directory.

---

## User Preferences

Stores:

Writing Style

Preferred Languages

Favorite Frameworks

Preferred AI Modes

Preferred Workspace

Theme

Coding Style

NOT personal sensitive information.

---

## Session Memory

Stores temporary execution context.

Deleted after session.

---

# Tool System

Everything is a plugin.

Examples

PDF Reader

Spreadsheet

Markdown Editor

GitHub

Email

Calendar

Browser

File Explorer

Terminal

Database Explorer

The Master never manipulates tools directly.

It invokes plugins.

---

# Workspace Engine

Synapse automatically changes UI.

Coding

↓

IDE

Research

↓

Browser + Notes

Writing

↓

Document Workspace

Business

↓

Spreadsheet + Documents

Teaching

↓

Lesson Workspace

## Projects (Phase 5)

Every workspace operation is scoped to a **project**. Projects are created, renamed, archived, and deleted through the UI or API. *(Phase 7 re-architected the storage split — see "Workspace vs Synapse Storage" below; the Phase 5 per-project directory layout was superseded.)*

**Phase 5 layout per project** (historical; `data/projects/<slug>/`):

- `chats/` — per-chat JSON records (durable message history)
- `memory/` — per-project semantic memory scope
- `files/` — uploaded blobs + catalog
- `index/` — vector database for this project
- `logs/` — reserved for project-scoped logs
- `config.json` — project identity, settings, timestamps

## Workspace vs Synapse Storage (Phase 7)

Phase 7 splits the system like a professional IDE:

- **The workspace folder** (user-chosen, e.g. `D:\Projects\Portfolio Website`) is the *project and nothing else* — source files, generated files, and Git data. Synapse never writes metadata into it.
- **Internal Synapse storage** lives in OS appdata (`%LOCALAPPDATA%\Synapse` on Windows, XDG on Linux) and holds everything Synapse owns:

```
Synapse/
├── projects.db     # SQLite registry — ONLY workspace paths per project
├── chats/          # chats/<project_id>/<chat_id>.json
├── memory/         # memory/<project_id>/*.json scopes
├── embeddings/     # embeddings/<project_id>/ (vector stores + upload catalog)
├── logs/           # logs/actions/<project_id>/actions.jsonl (+ app logs)
├── cache/          # transient data, wiped at shutdown
└── config/settings/ # application settings
```

Every piece of internal state is keyed by the project's **unique id**, so a workspace folder can be moved, copied, or shared without dragging Synapse data along.

## Chat Persistence (Phase 5)

Every chat is a JSON file under `<project>/chats/<chat_id>.json`. Messages are appended atomically (tmp + replace) on every turn, so a message is durable the moment it is recorded and survives unexpected shutdowns. The web UI loads messages from the server on chat switch, and the server records both user and assistant messages automatically.

## Session Recovery (Phase 5)

The last active project + chat are saved to `data/session.json` after every switch or message. On boot, `WorkspaceSystem.recover()` restores the session, so the user picks up exactly where they left off — even after a crash.

## Projects in User Folders (Phase 6)

*(Phase 7 replaced the registry and folder layout — see "Project Registry", "Native Folder Picker", "Workspace Lifecycle" below.)* A project can be created at a **user-chosen folder**: the workspace folder is created at `<parent>/<Name>`, seeded with a `README.md`, and optionally initialized as a Git repository. The folder itself holds only project files.

## Controlled Filesystem Access (Phase 6)

Every project gets a `FileOperator` rooted at the **workspace folder** (Phase 7: agent-generated files are real user files that persist in the workspace and its Git repo — there is no temp cache). It is the **only** way the Master or the model touches disk to produce output: writes are atomic (`tmp` + replace), every path is resolved and verified to stay inside the project root (traversal, absolute paths, and symlink escapes are rejected with `WorkspaceSafetyError`), Git internals (`.git/`) are read-only for the operator and skipped in listings, and listing/export/import are provided for the UI and onboarding. Content never leaves the project.

## Task Execution on Disk (Phase 6)

File-producing prompts ("create/write/build/generate …") route through the execution pipeline and write **real files**:

1. The planner flags the task as a file task (`file_output`) and attaches a `file_hint` (extension) when one is implied.
2. The routed model returns a structured **file manifest** (JSON `{"files": [...]}` or fenced code blocks) — parsed, flattened, deduplicated, and validated by `workspace/manifest.py`.
3. The operator writes each file to `<project>/work`, and every write is **validated** (`workspace/review.py`: python → `ast`, json → parse, js → brace balance, html → doctype/root, text → non-empty UTF-8).
4. A **review task** (`t-review`) is appended and routed to the strongest available *reasoning* model to check the generated file set for consistency and correctness.
5. The response reports each action (`created/modified/renamed/deleted`), its validation, and a human summary; files created by "OK" tool actions then appear in the project's `work/` listing.

## Multi-Model Pipeline (Phase 6)

Planner hints (`[planner] model_hints`) route each task kind to a specialist: e.g. `coding → qwen2.5-coder`, `writing → llama3.1`, `vision → qwen2.5-vl`, and REVIEW to the strongest available reasoning model. Every model still enters through the standard router so health/install/RAM gates are never bypassed.

## Audit Trail (Phase 6)

Every request that touches disk writes a **project action log** (`logs/actions/<project_id>/actions.jsonl` in internal storage, append-only JSON lines): the task plan, models used, tools used (`filesystem.created` / `.modified` / `.renamed` / `.deleted`), per-file status + validation, failures, and execution time. `GET /projects/{id}/actions` exposes it for the UI and debugging.

## Project Registry (Phase 7)

The registry is a **SQLite database** (`projects.db`) in internal storage. It stores only the workspace path per project — the single on-disk fact Synapse needs to find it. Unique slug ids (collision-safe) key all internal per-project state (chats, memory, embeddings, logs). `verify()` checks each registered workspace on disk so a moved/deleted folder is detected instead of silently failing.

## Native Folder Picker (Phase 7)

The web UI cannot open a native directory dialog, so the local server asks the OS on the user's behalf via `POST /projects/pick` (Windows `FolderBrowserDialog` through PowerShell, `zenity`/`kdialog` on Linux, `osascript` on macOS; the API falls back to a manual path prompt when no dialog exists). The user picks the **parent** directory; Synapse then creates `<parent>/<Name>` and registers it.

## Workspace Lifecycle (Phase 7)

- **Create** — pick a parent (or use internal storage) → folder created, seeded with `README.md`, best-effort `git init`, registered.
- **Delete (detach)** — removes the project from the registry and purges its internal chats/memory/embeddings/action logs; the user's folder is untouched.
- **Delete (erase)** — additionally removes the workspace folder itself.
- **Reconnect** — when a workspace was moved on disk, `POST /projects/{id}/reconnect` repoints the registry entry; the UI shows a "locate folder" banner on missing workspaces.
- **Cleanup** — on every startup and shutdown, Synapse wipes the temp cache, removes internal per-project state for unregistered ids, and deletes leftover empty internal workspaces from failed creations — never touching user-chosen folders.

## Action-Based Execution Engine (Phase 7)

Phase 7 refactors the execution pipeline so Synapse behaves like a real AI coding agent instead of a chatbot that returns code as text. Every request is classified into one of seven kinds:

1. **Chat Response** — plain conversation, no filesystem action
2. **File Creation** — write a script, function, class, test
3. **File Modification** — edit, refactor, fix, or change existing files
4. **Project Generation** — scaffold a website, app, dashboard, game, CLI, plugin
5. **Project Analysis** — analyze, explain, or review existing code
6. **Documentation** — write or update README, guides, changelogs
7. **Workspace Operation** — list, search, rename, move, delete, create folders

File kinds route through the **Action Engine** between the planner and models:

    Planner → Execution Plan → Action Engine → Filesystem Tools
                                     |                  |
                                     v                  v
                                Summary          Model(s) generate content

The backend performs **all filesystem operations**; the LLM only generates content and structured manifests. The Action Engine:

- **plans** — expands manifests into concrete operations, ensuring folder structure (e.g. `assets/` for project generation);
- **executes** — create folders, write/read/edit/rename/move/delete/search files via the safe `FileOperator`; deletes require explicit user confirmation;
- **verifies** — every written file is checked for existence and validated (Python/JSON/JS/HTML syntax, non-empty);
- **summarizes** — the chat response contains only a short action summary (created/modified/renamed/deleted + validation status); generated code is written directly into the workspace (`index.html`, `style.css`, `script.js`, `assets/`, etc.), never dumped into the chat.

For explicit workspace operations (list/search/rename/delete/create folder) the backend answers **without consulting any model** — the operation is executed immediately and the user sees a clean result.

The existing Phase 6 pipeline (manifest parser, FileOperator, OutputReviewer, review task on the strongest reasoning model) is fully reused; the Action Engine wraps it with guaranteed action execution, folder creation, and summary-only responses.

# Hardware Awareness

Synapse understands hardware.

Example

8GB RAM

↓

Recommend

Phi

Qwen 4B

Gemma

NOT

70B Models

The AI adapts to hardware.

Not the opposite.

---

# AI Model Registry

Every installed model includes metadata.

Capabilities

Writing

Coding

Reasoning

Vision

Math

Languages

Memory Usage

Latency

Privacy

Cost

Speed

Cloud

Local

Required RAM

Required GPU

The Master consults this registry before routing work.

---

# Privacy System

No conversations stored remotely.

No project files uploaded by default.

No telemetry.

Cloud execution only when needed.

User always owns their data.

---

# Example Workflow

User

"Build a portfolio website."

Master

↓

Intent Detection

↓

Complexity

↓

Privacy

↓

Hardware

↓

Memory

↓

Planner

↓

Design Task

↓

Coding Task

↓

Documentation Task

↓

Testing Task

↓

Review

↓

Merge

↓

Deliver

Everything happens automatically.

---

# Future Vision

Future versions may include:

Voice Companion

Desktop Automation

Android Integration

IoT Control

Smart Scheduling

Multi-PC Synchronization

AI Team Collaboration

Offline Voice Assistant

Plugin Marketplace

Self-improving Workflows

Enterprise Deployment

Private AI Clusters

Distributed Local AI Networks

---

# Final Vision Statement

Synapse One is not designed to be another chatbot.

It is designed to become the intelligent operating layer between humans and computers.

Rather than asking users to understand AI models, software, workflows, or tools, Synapse One understands them instead.

Every request is analyzed.

Every project is remembered.

Every decision is optimized.

Every task begins locally whenever possible.

Every cloud request is justified.

Every model works together under the supervision of the Master Agent.

The result is a privacy-first, hardware-aware, adaptive AI operating workspace that transforms artificial intelligence from a collection of disconnected models into one unified companion capable of planning, reasoning, delegating, remembering, and executing complex work on behalf of its users.

Synapse One does not replace software.

It orchestrates it.

It does not replace intelligence.

It amplifies it.

Its ultimate goal is simple:

**The user defines the outcome. Synapse handles everything else.**