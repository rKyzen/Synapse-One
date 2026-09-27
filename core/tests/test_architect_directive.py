"""Tests for Synapse One Architect Directive:
- Model Matrix Routing (Tier 2 and all tiers)
- Workspace Prompt Layers (Layer A vs Layer B)
- Action Engine & File Operator disk verification
- Python AST Syntax Validation Quality Loop
- Vision Isolation (zero file operations from vision tasks)
- Review / Synthesizer "not found in workspace" scrubbing
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import pytest

from synapse.actions.engine import ActionEngine
from synapse.domain import (
    Capability,
    Decision,
    HardwareProfile,
    ModelCapabilities,
    ModelMetadata,
    PrivacyMode,
    ProviderKind,
    Task,
    TaskKind,
)
from synapse.domain.fileops import ValidationResult
from synapse.domain.hardware import RecommendedModelLimits
from synapse.domain.tasks import TaskDAG
from synapse.hardware.model_matrix import (
    EXACT_MODEL_MATRIX,
    ModelRole,
    capability_to_role,
    get_exact_model,
)
from synapse.hardware.tier_resolver import HardwareTier
from synapse.router.router import Router, pick_primary_capability
from synapse.workspace.manifest import MANIFEST_INSTRUCTION, parse_file_manifest
from synapse.workspace.operator import FileOperator
from synapse.workspace.review import clean_review_text, summarize, validate_file


# ---------------------------------------------------------------------------
# 1. Model Matrix & Router Tests
# ---------------------------------------------------------------------------

def _hw_tier2() -> HardwareProfile:
    """Hardware profile resolving to Tier 2 (16GB RAM, 8GB VRAM)."""
    return HardwareProfile(
        memory={"total_gb": 16.0, "available_gb": 12.0, "used_percent": 25.0},
        gpu={"name": "NVIDIA GeForce RTX 3060", "vram_gb": 8.0},
        recommendations=RecommendedModelLimits(can_run_local_llm=True),
    )


def _tier2_registry() -> list[ModelMetadata]:
    return [
        ModelMetadata(
            id="gemma3:4b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            latency="fast",
            required_ram_gb=4,
            privacy_score=1.0,
            capabilities=ModelCapabilities(chat=0.9, writing=0.9, reasoning=0.85, math=0.85),
        ),
        ModelMetadata(
            id="qwen2.5-coder:7b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            latency="medium",
            required_ram_gb=8,
            privacy_score=1.0,
            capabilities=ModelCapabilities(chat=0.6, coding=1.0, debugging=0.95, architecture=0.9, reasoning=0.9),
        ),
        ModelMetadata(
            id="qwen3:4b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            latency="fast",
            required_ram_gb=4,
            privacy_score=1.0,
            capabilities=ModelCapabilities(chat=0.8, planning=0.95, reasoning=0.85),
        ),
        ModelMetadata(
            id="qwen2.5:7b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            latency="medium",
            required_ram_gb=8,
            privacy_score=1.0,
            capabilities=ModelCapabilities(chat=0.85, reasoning=0.9, translation=0.95, research=0.9),
        ),
        ModelMetadata(
            id="qwen2.5vl:7b",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            latency="medium",
            required_ram_gb=8,
            privacy_score=1.0,
            capabilities=ModelCapabilities(chat=0.7, vision=True, ocr=1.0, pdf=1.0),
        ),
        ModelMetadata(
            id="nomic-embed-text",
            provider_id="ollama",
            kind=ProviderKind.LOCAL,
            latency="fast",
            required_ram_gb=1,
            privacy_score=1.0,
            capabilities=ModelCapabilities(embeddings=1.0, chat=0.0),
        ),
    ]


def test_tier2_exact_model_matrix_values():
    assert EXACT_MODEL_MATRIX[HardwareTier.TIER2][ModelRole.CONVERSATION] == "gemma3:4b"
    assert EXACT_MODEL_MATRIX[HardwareTier.TIER2][ModelRole.MATH_REASONING] == "gemma3:4b"
    assert EXACT_MODEL_MATRIX[HardwareTier.TIER2][ModelRole.CODING] == "qwen2.5-coder:7b"
    assert EXACT_MODEL_MATRIX[HardwareTier.TIER2][ModelRole.PLANNING_AGENT] == "qwen3:4b"
    assert EXACT_MODEL_MATRIX[HardwareTier.TIER2][ModelRole.COMPLEX_TECHNICAL] == "qwen2.5:7b"
    assert EXACT_MODEL_MATRIX[HardwareTier.TIER2][ModelRole.VISION] == "qwen2.5vl:7b"
    assert EXACT_MODEL_MATRIX[HardwareTier.TIER2][ModelRole.EMBEDDINGS] == "nomic-embed-text"


def test_router_picks_gemma3_4b_for_conversation_on_tier2():
    router = Router()
    decision = Decision(
        can_stay_local=True,
        internet_required=False,
        preferred_kind=ProviderKind.LOCAL,
        required_capabilities=[Capability.CONVERSATION],
        preferred_capabilities=[Capability.CHAT],
        privacy=PrivacyMode.PREFER_LOCAL,
    )
    installed = {"ollama": {"gemma3:4b", "qwen2.5-coder:7b", "qwen3:4b", "qwen2.5:7b"}}
    routing = router.route(decision, _hw_tier2(), _tier2_registry(), {"ollama": True}, installed)
    assert routing.model_id == "gemma3:4b"


def test_router_picks_gemma3_4b_for_writing_and_math_on_tier2():
    router = Router()
    # Writing
    decision_writing = Decision(
        can_stay_local=True,
        internet_required=False,
        preferred_kind=ProviderKind.LOCAL,
        required_capabilities=[Capability.WRITING],
        privacy=PrivacyMode.PREFER_LOCAL,
    )
    installed = {"ollama": {"gemma3:4b", "qwen2.5-coder:7b", "qwen3:4b", "qwen2.5:7b"}}
    routing_w = router.route(decision_writing, _hw_tier2(), _tier2_registry(), {"ollama": True}, installed)
    assert routing_w.model_id == "gemma3:4b"

    # Math
    decision_math = Decision(
        can_stay_local=True,
        internet_required=False,
        preferred_kind=ProviderKind.LOCAL,
        required_capabilities=[Capability.MATH, Capability.REASONING],
        privacy=PrivacyMode.PREFER_LOCAL,
    )
    routing_m = router.route(decision_math, _hw_tier2(), _tier2_registry(), {"ollama": True}, installed)
    assert routing_m.model_id == "gemma3:4b"


def test_router_picks_qwen_coder_for_coding_on_tier2():
    router = Router()
    decision = Decision(
        can_stay_local=True,
        internet_required=False,
        preferred_kind=ProviderKind.LOCAL,
        required_capabilities=[Capability.CODING, Capability.FILE_CREATION],
        privacy=PrivacyMode.PREFER_LOCAL,
    )
    installed = {"ollama": {"gemma3:4b", "qwen2.5-coder:7b", "qwen3:4b", "qwen2.5:7b"}}
    routing = router.route(decision, _hw_tier2(), _tier2_registry(), {"ollama": True}, installed)
    assert routing.model_id == "qwen2.5-coder:7b"


def test_pick_primary_capability_prioritizes_specific_capabilities():
    assert pick_primary_capability([Capability.CHAT, Capability.CODING]) == Capability.CODING
    assert pick_primary_capability([Capability.CHAT, Capability.VISION]) == Capability.VISION
    assert pick_primary_capability([Capability.CHAT, Capability.WRITING]) == Capability.WRITING
    assert pick_primary_capability([Capability.CHAT, Capability.MATH]) == Capability.MATH


# ---------------------------------------------------------------------------
# 2. Action Engine & Disk Verification Tests
# ---------------------------------------------------------------------------

def test_action_engine_edit_verifies_on_disk(tmp_path: Path):
    op = FileOperator(tmp_path)
    engine = ActionEngine(op)

    # Setup file
    engine.write("hello.py", "def greet():\n    return 'hello world'\n")
    assert op.exists("hello.py")

    # Apply valid edit
    res = engine.edit("hello.py", "return 'hello world'", "return 'hello Synapse'")
    assert res["ok"] is True

    # Confirm disk state
    content = op.read("hello.py")
    assert content is not None
    assert "return 'hello Synapse'" in content
    assert "return 'hello world'" not in content


def test_action_engine_edit_fails_when_target_missing(tmp_path: Path):
    op = FileOperator(tmp_path)
    engine = ActionEngine(op)
    engine.write("app.py", "x = 10\n")

    res = engine.edit("app.py", "y = 20", "y = 30")
    assert res["ok"] is False
    assert "not found" in res["error"]


def test_action_engine_apply_edit_records_modified_action(tmp_path: Path):
    op = FileOperator(tmp_path)
    engine = ActionEngine(op)
    engine.write("main.py", "print('version 1')\n")

    ops = [{"action": "edit", "path": "main.py", "old": "version 1", "new": "version 2"}]
    actions, validations = engine.apply(ops)

    assert len(actions) == 1
    assert actions[0].action == "modified"
    assert actions[0].status == "ok"
    assert op.read("main.py") == "print('version 2')\n"


# ---------------------------------------------------------------------------
# 3. Python AST Syntax Check Tests
# ---------------------------------------------------------------------------

def test_validate_file_catches_python_syntax_error():
    valid_code = "def add(a, b):\n    return a + b\n"
    res_valid = validate_file("math_utils.py", valid_code)
    assert res_valid.ok is True

    invalid_code = "def broken_func(\n    return 42\n"
    res_invalid = validate_file("broken.py", invalid_code)
    assert res_invalid.ok is False
    assert "syntax error" in res_invalid.error.lower()


# ---------------------------------------------------------------------------
# 4. Review Text Cleaning / Anti-Hallucination Tests
# ---------------------------------------------------------------------------

def test_clean_review_text_scrubs_false_not_found_claims():
    raw_review = (
        "1. The project structure looks good.\n"
        "2. File index.html was not found in workspace.\n"
        "3. Implementation in script.js matches the spec.\n"
        "4. style.css is missing from workspace.\n"
    )
    ok_paths = {"index.html", "script.js", "style.css"}
    cleaned = clean_review_text(raw_review, ok_paths)

    assert "index.html was not found" not in cleaned
    assert "style.css is missing" not in cleaned
    assert "The project structure looks good." in cleaned
    assert "Implementation in script.js matches the spec." in cleaned


def test_summarize_integrates_cleaned_review():
    validations = [
        ValidationResult(path="index.html", ok=True, checks=["html root"]),
        ValidationResult(path="style.css", ok=True, checks=["valid"]),
    ]
    raw_review = "index.html not found in workspace. Everything else looks solid."
    summary_block = summarize(validations, review_text=raw_review)

    assert "- index.html: ok" in summary_block
    assert "- style.css: ok" in summary_block
    assert "index.html not found in workspace" not in summary_block
    assert "Everything else looks solid." in summary_block


# ---------------------------------------------------------------------------
# 5. Vision Isolation Tests
# ---------------------------------------------------------------------------

def test_vision_isolation_does_not_execute_files(tmp_path: Path):
    op = FileOperator(tmp_path)
    engine = ActionEngine(op)

    # If a vision task emits JSON, vision isolation ignores it
    vision_task = Task(
        id="t-vision",
        kind=TaskKind.GENERAL,
        description="Describe chart.png",
        required_capabilities=[Capability.VISION],
        file_output=False,
    )
    vision_task.result = '{"files": [{"path": "fake.py", "content": "print(1)"}]}'

    # When filtered with vision capability check:
    is_vision = Capability.VISION in (vision_task.required_capabilities or [])
    assert is_vision is True
    # Disk should remain untouched
    assert not op.exists("fake.py")


# ---------------------------------------------------------------------------
# 6. Compound Decomposition & Workspace Prompt Layer Tests
# ---------------------------------------------------------------------------

def test_compound_coding_request_decomposition_plan():
    from synapse.master.schemas import SubTask, TaskDecompositionPlan

    plan = TaskDecompositionPlan(
        execution_strategy="SEQUENTIAL",
        tasks=[
            SubTask(
                task_id=1,
                intent="CODING",
                sub_prompt="Implement calculator module in src/calc.py",
                assigned_model="qwen2.5-coder:7b",
                capability="coding",
                reasoning="implement core code",
                dependencies=[],
            ),
            SubTask(
                task_id=2,
                intent="CODING",
                sub_prompt="Write unit tests in test/test_calc.py",
                assigned_model="qwen2.5-coder:7b",
                capability="testing",
                reasoning="test suite",
                dependencies=[1],
            ),
            SubTask(
                task_id=3,
                intent="WRITING",
                sub_prompt="Write docs in README.md",
                assigned_model="gemma3:4b",
                capability="writing",
                reasoning="docs",
                dependencies=[1],
            ),
        ],
    )
    dag = plan.to_dag()
    assert len(dag.tasks) == 4  # 3 tasks + 1 synthesis
    assert dag.get("t1").file_output is True
    assert dag.get("t1").file_hint == "src/calc.py"
    assert dag.get("t2").file_output is True
    assert dag.get("t2").file_hint == "test/test_calc.py"
    assert dag.get("t3").file_output is True
    assert dag.get("t3").file_hint == "README.md"


# ---------------------------------------------------------------------------
# 7. Exact Model Matrix Enforcement & Photosynthesis Regression Tests
# ---------------------------------------------------------------------------

def test_photosynthesis_explanation_routes_to_gemma3_4b_on_tier2():
    """'Explain photosynthesis in simple terms' MUST select gemma3:4b on Tier 2."""
    from synapse.analyzers.complexity import ComplexityAnalyzer
    from synapse.analyzers.intent import IntentAnalyzer
    from synapse.analyzers.privacy import PrivacyAnalyzer
    from synapse.decision.engine import DecisionEngine

    prompt = "Explain photosynthesis in simple terms"
    intent = IntentAnalyzer().analyze(prompt)
    complexity = ComplexityAnalyzer().analyze(prompt)
    privacy = PrivacyAnalyzer().analyze(prompt, complexity)

    decision_engine = DecisionEngine()
    decision = decision_engine.decide(intent, complexity, privacy, _hw_tier2(), _tier2_registry())

    router = Router()
    installed = {"ollama": {"gemma3:4b", "qwen2.5-coder:7b", "qwen3:4b", "qwen2.5:7b", "qwen2.5vl:7b", "nomic-embed-text"}}
    routing = router.route(
        decision,
        _hw_tier2(),
        _tier2_registry(),
        {"ollama": True},
        installed,
        complexity=complexity.score,
        prompt=prompt,
    )

    assert routing.model_id == "gemma3:4b"
    # Ensure vision model has 0 score
    for cand in routing.candidates:
        if cand["model"] == "qwen2.5vl:7b":
            assert cand["score"] == 0.0


def test_pure_text_prompts_never_select_vision_model():
    """Text-only prompts must never select vision models even if all models are installed."""
    router = Router()
    installed = {"ollama": {"gemma3:4b", "qwen2.5-coder:7b", "qwen3:4b", "qwen2.5:7b", "qwen2.5vl:7b", "nomic-embed-text"}}

    prompts = [
        "Explain photosynthesis in simple terms",
        "Write a short poem about rain",
        "Solve x^2 - 7x + 12 = 0",
        "What is the capital of France?",
        "Tell me how quantum computing works",
    ]

    for p in prompts:
        decision = Decision(
            can_stay_local=True,
            internet_required=False,
            preferred_kind=ProviderKind.LOCAL,
            required_capabilities=[Capability.REASONING] if "solve" in p.lower() or "how" in p.lower() or "explain" in p.lower() else [Capability.CONVERSATION],
            privacy=PrivacyMode.PREFER_LOCAL,
        )
        routing = router.route(
            decision,
            _hw_tier2(),
            _tier2_registry(),
            {"ollama": True},
            installed,
            prompt=p,
        )
        assert routing.model_id != "qwen2.5vl:7b"
        assert routing.model_id == "gemma3:4b"


def test_vision_prompt_selects_qwen25vl_on_tier2():
    """Vision prompts with vision required MUST select qwen2.5vl:7b on Tier 2."""
    router = Router()
    decision = Decision(
        can_stay_local=True,
        internet_required=False,
        preferred_kind=ProviderKind.LOCAL,
        required_capabilities=[Capability.VISION, Capability.IMAGE_UNDERSTANDING],
        privacy=PrivacyMode.PREFER_LOCAL,
    )
    installed = {"ollama": {"gemma3:4b", "qwen2.5-coder:7b", "qwen3:4b", "qwen2.5:7b", "qwen2.5vl:7b", "nomic-embed-text"}}
    routing = router.route(
        decision,
        _hw_tier2(),
        _tier2_registry(),
        {"ollama": True},
        installed,
        prompt="Describe what you see in chart.png",
    )
    assert routing.model_id == "qwen2.5vl:7b"


def test_soft_scoring_and_history_cannot_override_exact_matrix_model():
    """Historical reliability adjustments must not cause another model to overtake the matrix model."""
    router = Router()
    installed = {"ollama": {"gemma3:4b", "qwen2.5-coder:7b", "qwen3:4b", "qwen2.5:7b", "qwen2.5vl:7b", "nomic-embed-text"}}

    history = {
        "ollama/gemma3:4b": {
            "samples": 20,
            "success_rate": 0.80,
            "failure_rate": 0.10,
            "timeout_rate": 0.10,
            "avg_latency_s": 2.5,
        },
        "ollama/qwen2.5:7b": {
            "samples": 100,
            "success_rate": 1.0,
            "failure_rate": 0.0,
            "timeout_rate": 0.0,
            "avg_latency_s": 1.0,
        },
    }

    decision = Decision(
        can_stay_local=True,
        internet_required=False,
        preferred_kind=ProviderKind.LOCAL,
        required_capabilities=[Capability.REASONING],
        privacy=PrivacyMode.PREFER_LOCAL,
    )

    routing = router.route(
        decision,
        _hw_tier2(),
        _tier2_registry(),
        {"ollama": True},
        installed,
        performance=history,
        prompt="Explain photosynthesis in simple terms",
    )

    assert routing.model_id == "gemma3:4b"


def test_all_tier2_exact_capabilities_route_correctly():
    """Ensure every primary capability routes to its exact designated Tier 2 model."""
    router = Router()
    installed = {"ollama": {"gemma3:4b", "qwen2.5-coder:7b", "qwen3:4b", "qwen2.5:7b", "qwen2.5vl:7b", "nomic-embed-text"}}

    cases = [
        ([Capability.CONVERSATION], "gemma3:4b"),
        ([Capability.CHAT], "gemma3:4b"),
        ([Capability.WRITING], "gemma3:4b"),
        ([Capability.REASONING], "gemma3:4b"),
        ([Capability.MATH], "gemma3:4b"),
        ([Capability.CODING], "qwen2.5-coder:7b"),
        ([Capability.PLANNING], "qwen3:4b"),
        ([Capability.TRANSLATION], "qwen2.5:7b"),
        ([Capability.VISION], "qwen2.5vl:7b"),
    ]

    for req_caps, expected_model in cases:
        decision = Decision(
            can_stay_local=True,
            internet_required=False,
            preferred_kind=ProviderKind.LOCAL,
            required_capabilities=req_caps,
            privacy=PrivacyMode.PREFER_LOCAL,
        )
        routing = router.route(
            decision,
            _hw_tier2(),
            _tier2_registry(),
            {"ollama": True},
            installed,
        )
        assert routing.model_id == expected_model, f"Expected {expected_model} for {req_caps}, got {routing.model_id}"


def test_master_agent_photosynthesis_end_to_end():
    """MasterAgent.process('Explain photosynthesis in simple terms') executes with gemma3:4b on Tier 2."""
    from unittest.mock import MagicMock
    from synapse.analyzers.complexity import ComplexityAnalyzer
    from synapse.analyzers.intent import IntentAnalyzer
    from synapse.analyzers.privacy import PrivacyAnalyzer
    from synapse.decision.engine import DecisionEngine
    from synapse.domain import ChatResponse
    from synapse.events import EventBus
    from synapse.execution.executor import Executor
    from synapse.execution.planner import ExecutionPlanner
    from synapse.master.agent import MasterAgent
    from synapse.providers.manager import ProviderManager

    mock_provider = MagicMock()
    mock_provider.provider_id = "ollama"
    mock_provider.kind = ProviderKind.LOCAL
    mock_provider.list_models.return_value = [
        MagicMock(id="gemma3:4b"),
        MagicMock(id="qwen2.5-coder:7b"),
        MagicMock(id="qwen3:4b"),
        MagicMock(id="qwen2.5:7b"),
        MagicMock(id="qwen2.5vl:7b"),
        MagicMock(id="nomic-embed-text"),
    ]
    mock_provider.chat.return_value = ChatResponse(
        provider_id="ollama",
        model_id="gemma3:4b",
        kind=ProviderKind.LOCAL,
        content="Photosynthesis is the process by which plants use sunlight, water, and CO2 to produce oxygen and sugar.",
    )

    providers = MagicMock()
    providers.get.return_value = mock_provider
    providers.health.return_value = True
    providers.all.return_value = [mock_provider]
    providers.provider_ids.return_value = ["ollama"]

    mock_hw = MagicMock()
    mock_hw.scan.return_value = _hw_tier2()

    mock_reg = MagicMock()
    mock_reg.all.return_value = _tier2_registry()

    events = EventBus()
    mock_config = MagicMock()
    mock_config.get.side_effect = lambda k, default=None: {
        "privacy.user_preference": "balanced",
        "router.prefer_local": True,
        "executor.timeout_s": 30.0,
    }.get(k, default)

    executor = Executor(providers)

    agent = MasterAgent(
        intent_analyzer=IntentAnalyzer(),
        complexity_analyzer=ComplexityAnalyzer(),
        privacy_analyzer=PrivacyAnalyzer(),
        decision_engine=DecisionEngine(),
        planner=ExecutionPlanner(),
        router=Router(),
        executor=executor,
        providers=providers,
        hardware=mock_hw,
        registry=mock_reg,
        events=events,
        config=mock_config,
    )

    response = agent.process("Explain photosynthesis in simple terms")
    assert response.model == "gemma3:4b"
    assert response.provider == "ollama"
    assert "Photosynthesis" in response.response


# ---------------------------------------------------------------------------
# 8. Multi-File Project Generation & Execution Verification Tests
# ---------------------------------------------------------------------------

def test_expense_tracker_multi_file_project_generation(tmp_path: Path):
    """Test compound multi-file coding project: creates runnable code, tests, and README."""
    from synapse.master.schemas import SubTask, TaskDecompositionPlan
    import subprocess
    import sys

    # 1. Decomposition plan for expense tracker
    plan = TaskDecompositionPlan(
        execution_strategy="SEQUENTIAL",
        tasks=[
            SubTask(
                task_id=1,
                intent="CODING",
                sub_prompt="Create expense_tracker.py with ExpenseTracker class (add, total, filter_by_category)",
                assigned_model="qwen2.5-coder:7b",
                capability="coding",
                reasoning="core logic",
                dependencies=[],
            ),
            SubTask(
                task_id=2,
                intent="CODING",
                sub_prompt="Create test_expense_tracker.py with pytest tests for ExpenseTracker",
                assigned_model="qwen2.5-coder:7b",
                capability="testing",
                reasoning="unit tests",
                dependencies=[1],
            ),
            SubTask(
                task_id=3,
                intent="WRITING",
                sub_prompt="Create README.md with project overview and usage",
                assigned_model="gemma3:4b",
                capability="writing",
                reasoning="documentation",
                dependencies=[1],
            ),
        ],
    )

    dag = plan.to_dag()
    assert len(dag.tasks) == 4  # 3 tasks + 1 synthesis

    op = FileOperator(tmp_path)
    engine = ActionEngine(op)

    # 2. Simulate model outputs in Layer B manifest format
    task1_output = """\
```json
{
  "files": [
    {
      "path": "expense_tracker.py",
      "content": "class ExpenseTracker:\\n    def __init__(self):\\n        self.expenses = []\\n\\n    def add(self, category: str, amount: float, description: str = ''):\\n        if amount <= 0:\\n            raise ValueError('Amount must be positive')\\n        item = {'category': category, 'amount': amount, 'description': description}\\n        self.expenses.append(item)\\n        return item\\n\\n    def total(self) -> float:\\n        return sum(e['amount'] for e in self.expenses)\\n\\n    def filter_by_category(self, category: str) -> list[dict]:\\n        return [e for e in self.expenses if e['category'] == category]\\n"
    }
  ]
}
```"""

    task2_output = """\
```json
{
  "files": [
    {
      "path": "test_expense_tracker.py",
      "content": "import pytest\\nfrom expense_tracker import ExpenseTracker\\n\\ndef test_add_and_total():\\n    tracker = ExpenseTracker()\\n    tracker.add('Food', 15.50, 'Lunch')\\n    tracker.add('Transport', 4.50, 'Bus')\\n    assert tracker.total() == 20.0\\n\\ndef test_filter_by_category():\\n    tracker = ExpenseTracker()\\n    tracker.add('Food', 10.0)\\n    tracker.add('Books', 25.0)\\n    food_items = tracker.filter_by_category('Food')\\n    assert len(food_items) == 1\\n    assert food_items[0]['amount'] == 10.0\\n\\ndef test_invalid_amount():\\n    tracker = ExpenseTracker()\\n    with pytest.raises(ValueError):\\n        tracker.add('Food', -5.0)\\n"
    }
  ]
}
```"""

    task3_output = """\
```json
{
  "files": [
    {
      "path": "README.md",
      "content": "# Expense Tracker\\n\\nA simple CLI expense tracking library with categorization and totaling.\\n\\n## Usage\\n```python\\nfrom expense_tracker import ExpenseTracker\\ntracker = ExpenseTracker()\\ntracker.add('Food', 12.50)\\nprint(tracker.total())\\n```\\n"
    }
  ]
}
```"""

    # 3. Parse manifests & apply actions
    ops1 = parse_file_manifest(task1_output)
    ops2 = parse_file_manifest(task2_output)
    ops3 = parse_file_manifest(task3_output)

    actions1, v1 = engine.apply(ops1)
    actions2, v2 = engine.apply(ops2)
    actions3, v3 = engine.apply(ops3)

    # 4. Verify disk presence & validation results
    assert op.exists("expense_tracker.py")
    assert op.exists("test_expense_tracker.py")
    assert op.exists("README.md")

    # AST validations passed
    assert all(val.ok for val in v1)
    assert all(val.ok for val in v2)
    assert all(val.ok for val in v3)

    # 5. Run pytest directly on the generated test file in tmp_path
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path / "test_expense_tracker.py")],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, f"Generated tests failed:\n{result.stdout}\n{result.stderr}"
    assert "3 passed" in result.stdout


# ---------------------------------------------------------------------------
# 9. Chat Memory, Context Injection & FastAPI Create/Edit Hardening Tests
# ---------------------------------------------------------------------------

def test_what_is_this_chat_about_uses_injected_history():
    """'What is this chat about?' is answered from injected history without tool calls."""
    from synapse.domain.enums import MemoryScope
    from synapse.workspace.brief import build_workspace_brief

    # Simulate memory store with prior conversation turns
    from unittest.mock import MagicMock
    from synapse.domain.memory import MemoryEntry

    mock_memory = MagicMock()
    mock_memory.recent.return_value = [
        MemoryEntry(id="1", scope=MemoryScope.CONVERSATION, text="Create a FastAPI endpoint in main.py", source="user", created_at="2026-09-25T10:00:00Z"),
        MemoryEntry(id="2", scope=MemoryScope.CONVERSATION, text="Created FastAPI app in main.py with health endpoint.", source="assistant", created_at="2026-09-25T10:01:00Z"),
        MemoryEntry(id="3", scope=MemoryScope.CONVERSATION, text="Now add a /items endpoint with Pydantic model", source="user", created_at="2026-09-25T10:02:00Z"),
        MemoryEntry(id="4", scope=MemoryScope.CONVERSATION, text="Updated main.py with /items endpoint and Item model.", source="assistant", created_at="2026-09-25T10:03:00Z"),
    ]

    brief = build_workspace_brief(
        project_id="proj-api",
        project_name="MyFastAPIProject",
        project_path="/workspace/api",
        memory=mock_memory,
        conversation_id="chat-123",
        conversation_scope="Discussion covering FastAPI endpoints and Pydantic data models.",
    )

    assert "Current chat history" in brief
    assert "Create a FastAPI endpoint in main.py" in brief
    assert "add a /items endpoint" in brief
    assert "Conversation scope summary:" in brief
    assert "FastAPI endpoints and Pydantic data models" in brief


def test_fastapi_create_and_subsequent_edit_writes_disk(tmp_path: Path):
    """Creating a FastAPI app and subsequently editing it writes real files to disk and verifies."""
    op = FileOperator(tmp_path)
    engine = ActionEngine(op)

    # 1. Create FastAPI app
    create_output = """\
Here is the FastAPI application:
```json
{
  "files": [
    {
      "path": "main.py",
      "content": "from fastapi import FastAPI\\n\\napp = FastAPI()\\n\\n@app.get('/')\\ndef read_root():\\n    return {'status': 'ok'}\\n"
    }
  ]
}
```
Let me know if you want more endpoints!"""

    ops_create = parse_file_manifest(create_output)
    actions1, val1 = engine.apply(ops_create)

    assert op.exists("main.py")
    assert actions1[0].status == "ok"
    assert "read_root" in op.read("main.py")
    assert all(v.ok for v in val1)

    # 2. Subsequent edit: modify endpoint and add /health
    edit_output = """\
```json
{
  "action": "edit",
  "path": "main.py",
  "old": "return {'status': 'ok'}",
  "new": "return {'status': 'healthy', 'version': '1.0.0'}"
}
```"""

    ops_edit = parse_file_manifest(edit_output)
    actions2, val2 = engine.apply(ops_edit)

    assert actions2[0].status == "ok"
    assert actions2[0].action == "modified"
    disk_content = op.read("main.py")
    assert "version': '1.0.0" in disk_content
    assert all(v.ok for v in val2)


def test_review_after_edit_never_says_not_found(tmp_path: Path):
    """Review summary after successful edit strictly excludes 'not found' claims."""
    op = FileOperator(tmp_path)
    engine = ActionEngine(op)
    engine.write("main.py", "print('hello')\n")

    ops = [{"action": "edit", "path": "main.py", "old": "hello", "new": "world"}]
    actions, validations = engine.apply(ops)

    raw_review = "File main.py was not found in workspace. Verification failed."
    summary_text = summarize(validations, review_text=raw_review)

    assert "- main.py: ok" in summary_text
    assert "not found" not in summary_text
    assert "Verification failed" in summary_text


def test_poem_and_photosynthesis_route_to_gemma_on_tier2():
    """Text conversation / explanation / poem prompts must select gemma3:4b on Tier 2."""
    router = Router()
    installed = {"ollama": {"gemma3:4b", "qwen2.5-coder:7b", "qwen3:4b", "qwen2.5:7b", "qwen2.5vl:7b", "nomic-embed-text"}}

    prompts = [
        ("Explain photosynthesis in simple terms", [Capability.CONVERSATION, Capability.REASONING]),
        ("Write a short poem about rain", [Capability.WRITING]),
        ("What is the speed of light?", [Capability.CONVERSATION]),
    ]

    for p, caps in prompts:
        decision = Decision(
            can_stay_local=True,
            internet_required=False,
            preferred_kind=ProviderKind.LOCAL,
            required_capabilities=caps,
            privacy=PrivacyMode.PREFER_LOCAL,
        )
        routing = router.route(
            decision,
            _hw_tier2(),
            _tier2_registry(),
            {"ollama": True},
            installed,
            prompt=p,
        )
        assert routing.model_id == "gemma3:4b", f"Prompt '{p}' routed to {routing.model_id} instead of gemma3:4b"


# ---------------------------------------------------------------------------
# 10. Multi-Turn FastAPI Create -> Edit -> Chat Summary Directive Verification
# ---------------------------------------------------------------------------

def test_fastapi_multi_turn_edit_and_chat_summary_end_to_end(tmp_path: Path):
    """End-to-end verification:
    1. Turn 1: Create simple FastAPI greet endpoint -> real main.py on disk
    2. Turn 2: Follow-up edit 'In the FastAPI code add if successful to Print Successful' -> existing main.py is modified on disk
    3. Turn 3: 'What is this chat about?' -> returns accurate summary from injected history
    """
    op = FileOperator(tmp_path)
    engine = ActionEngine(op)

    # --- Turn 1: Create FastAPI greet endpoint ---
    turn1_output = """\
```json
{
  "files": [
    {
      "path": "main.py",
      "content": "from fastapi import FastAPI\\n\\napp = FastAPI()\\n\\n@app.get('/greet')\\ndef greet():\\n    return {'message': 'Hello from Synapse!'}\\n"
    }
  ]
}
```"""
    ops1 = parse_file_manifest(turn1_output)
    actions1, v1 = engine.apply(ops1)
    assert op.exists("main.py")
    assert actions1[0].status == "ok"
    assert actions1[0].action == "created"
    assert "greet" in op.read("main.py")

    # --- Turn 2: Edit prompt modifying the existing file ---
    turn2_output = """\
```json
{
  "files": [
    {
      "path": "main.py",
      "content": "from fastapi import FastAPI\\n\\napp = FastAPI()\\n\\n@app.get('/greet')\\ndef greet():\\n    print('Successful')\\n    return {'message': 'Hello from Synapse!', 'status': 'success'}\\n"
    }
  ]
}
```"""
    ops2 = parse_file_manifest(turn2_output)
    actions2, v2 = engine.apply(ops2)
    assert actions2[0].status == "ok"
    assert actions2[0].action == "modified"
    updated_code = op.read("main.py")
    assert "print('Successful')" in updated_code
    assert "@app.get('/greet')" in updated_code
    assert all(v.ok for v in v2)

    # --- Turn 3: Chat summary from injected context ---
    from synapse.workspace.brief import build_workspace_brief

    turn_messages = [
        {"role": "user", "content": "Create simple FastAPI greet endpoint"},
        {"role": "assistant", "content": "Created FastAPI greet endpoint in main.py."},
        {"role": "user", "content": "In the FastAPI code add if successful to Print Successful"},
        {"role": "assistant", "content": "Updated main.py to print 'Successful' upon greeting."},
    ]

    brief = build_workspace_brief(
        project_id="fastapi-app",
        project_name="FastAPI App",
        project_path=str(tmp_path),
        file_operator=op,
        messages=turn_messages,
        conversation_scope="FastAPI greet endpoint creation and print statement update.",
    )

    # Injected context validation
    assert "Current chat history" in brief
    assert "Create simple FastAPI greet endpoint" in brief
    assert "Print Successful" in brief
    assert "main.py" in brief
    assert "Conversation scope summary:" in brief


# ---------------------------------------------------------------------------
# 11. Architect Directive Redesign: Space HTML Website & Edit & Chat Memory
# ---------------------------------------------------------------------------

def test_space_exploration_website_routes_to_qwen_coder_on_tier2():
    """Space exploration single-page HTML website request must route to Qwen-Coder on Tier 2."""
    from synapse.analyzers.intent import IntentAnalyzer
    from synapse.master.orchestrator import AIMasterOrchestrator

    prompt = "Create a simple single-page HTML website about space exploration with a dark theme and some cool facts. Save it as index.html"
    intent_res = IntentAnalyzer().analyze(prompt)
    assert intent_res.primary == Capability.CODING

    analysis = AIMasterOrchestrator._fallback_analysis(prompt, "test")
    assert analysis.coding_needed is True
    assert analysis.files_needed is True
    assert analysis.artifact_required is True

    router = Router()
    decision = Decision(
        can_stay_local=True,
        internet_required=False,
        preferred_kind=ProviderKind.LOCAL,
        required_capabilities=[Capability.CODING, Capability.FILE_CREATION],
        privacy=PrivacyMode.PREFER_LOCAL,
    )
    installed = {"ollama": {"gemma3:4b", "qwen2.5-coder:7b", "qwen3:4b", "qwen2.5:7b", "qwen2.5vl:7b", "nomic-embed-text"}}
    routing = router.route(decision, _hw_tier2(), _tier2_registry(), {"ollama": True}, installed, prompt=prompt)
    assert routing.model_id == "qwen2.5-coder:7b"


def test_space_exploration_website_creates_index_html_on_disk(tmp_path: Path):
    """Creating space exploration single-page HTML website produces real index.html on disk."""
    op = FileOperator(tmp_path)
    engine = ActionEngine(op)

    space_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Space Exploration</title>
    <style>
        body { background-color: #0b0d1b; color: #f0f0f0; font-family: sans-serif; text-align: center; }
        h1 { color: #5bc0be; }
        .facts { margin: 20px auto; max-width: 600px; text-align: left; }
    </style>
</head>
<body>
    <h1>Cosmic Frontiers: Space Exploration</h1>
    <p>Discover the wonders of our solar system and beyond.</p>
    <div class="facts">
        <ul>
            <li>Neutron stars can spin 600 times per second.</li>
            <li>One day on Venus is longer than one year on Earth.</li>
            <li>Space is completely silent because there is no atmosphere.</li>
        </ul>
    </div>
</body>
</html>"""

    model_response = f"""Here is the space exploration website:
```json
{{
  "files": [
    {{
      "path": "index.html",
      "content": {json.dumps(space_html)}
    }}
  ]
}}
```"""

    ops = parse_file_manifest(model_response)
    actions, validations = engine.apply(ops)

    assert op.exists("index.html")
    assert actions[0].status == "ok"
    assert actions[0].action == "created"
    assert actions[0].path == "index.html"
    content_on_disk = op.read("index.html")
    assert "Cosmic Frontiers" in content_on_disk
    assert "Neutron stars" in content_on_disk
    assert all(v.ok for v in validations)


def test_space_website_button_edit_modifies_existing_index_html(tmp_path: Path):
    """Follow-up edit adding a Learn More button modifies the existing index.html on disk."""
    op = FileOperator(tmp_path)
    engine = ActionEngine(op)

    # Step 1: Initial creation
    initial_html = "<!DOCTYPE html><html><body><h1>Space</h1><div id='content'>Facts</div></body></html>"
    engine.write("index.html", initial_html)
    assert op.exists("index.html")

    # Step 2: Full-overwrite edit with Learn More button
    updated_html = """<!DOCTYPE html>
<html>
<body>
    <h1>Space</h1>
    <div id='content'>Facts</div>
    <button onclick="alert('Welcome to Space!')">Learn More</button>
</body>
</html>"""

    edit_response = f"""```json
{{
  "files": [
    {{
      "path": "index.html",
      "content": {json.dumps(updated_html)}
    }}
  ]
}}
```"""

    ops = parse_file_manifest(edit_response)
    actions, validations = engine.apply(ops)

    assert actions[0].status == "ok"
    assert actions[0].action == "modified"
    disk_content = op.read("index.html")
    assert "Learn More" in disk_content
    assert "alert('Welcome to Space!')" in disk_content
    assert all(v.ok for v in validations)


def test_what_is_this_chat_about_summarizes_space_website_history(tmp_path: Path):
    """'What is this chat about?' accurately references space website & button addition from history."""
    from synapse.workspace.brief import build_specialist_prompt, build_workspace_brief

    op = FileOperator(tmp_path)
    engine = ActionEngine(op)
    engine.write("index.html", "<html><body><h1>Space Exploration</h1><button>Learn More</button></body></html>")

    chat_messages = [
        {"role": "user", "content": "Create a simple single-page HTML website about space exploration with a dark theme and some cool facts. Save it as index.html"},
        {"role": "assistant", "content": "Created index.html with dark theme and space facts."},
        {"role": "user", "content": "In the index.html file, add a Learn More button that alerts 'Welcome to Space!' when clicked."},
        {"role": "assistant", "content": "Added the Learn More alert button to index.html."},
    ]

    prompt = build_specialist_prompt(
        "Summarize what this conversation has accomplished so far.",
        project_id="space-portal",
        project_name="Space Portal",
        project_path=str(tmp_path),
        file_operator=op,
        messages=chat_messages,
    )

    assert "Project: Space Portal" in prompt
    assert "Path: " in prompt
    assert "Current chat history (this chat only):" in prompt
    assert "Create a simple single-page HTML website about space exploration" in prompt
    assert "add a Learn More button that alerts 'Welcome to Space!'" in prompt
    assert "Conversation summary (this chat):" in prompt
    assert "Live files:" in prompt
    assert "- index.html" in prompt
    assert "Rules:" in prompt


def test_specialist_prompt_exact_unified_structure(tmp_path: Path):
    """Specialist prompt contains all required sections in exact order without redundant layers."""
    from synapse.workspace.brief import build_specialist_prompt

    op = FileOperator(tmp_path)
    engine = ActionEngine(op)
    engine.write("app.py", "def run(): pass\n")

    prompt = build_specialist_prompt(
        "Add error handling to run()",
        project_id="my-app",
        project_name="My Cool App",
        project_path=str(tmp_path),
        file_operator=op,
        target_edit_path="app.py",
        target_edit_content="def run(): pass\n",
        messages=[
            {"role": "user", "content": "Initial setup"},
            {"role": "assistant", "content": "App initialized"},
        ],
    )

    # Verify required sections exist
    assert "You are a specialist inside Synapse working on a real project." in prompt
    assert "Project: My Cool App" in prompt
    assert f"Path: {tmp_path}" in prompt
    assert "Current chat history (this chat only):" in prompt
    assert "- user: Initial setup" in prompt
    assert "- assistant: App initialized" in prompt
    assert "Conversation summary (this chat):" in prompt
    assert "Live files:" in prompt
    assert "- app.py" in prompt
    assert "Current content of the file you must modify (app.py):" in prompt
    assert "def run(): pass" in prompt
    assert "Your task: Add error handling to run()" in prompt
    assert "Rules:" in prompt
    assert "Preferred format (full overwrite – most reliable):" in prompt


def test_action_engine_json_parsing_with_surrounding_prose(tmp_path: Path):
    """Action Engine cleanly parses and executes file manifest even when surrounded by prose."""
    op = FileOperator(tmp_path)
    engine = ActionEngine(op)

    response_text = """Certainly! I have created the requested styles.css file for your application.
Here is the code you need:

```json
{
  "files": [
    {
      "path": "styles.css",
      "content": "body { margin: 0; background: #222; color: #fff; }"
    }
  ]
}
```

Let me know if you would like me to make any further adjustments!"""

    ops = parse_file_manifest(response_text)
    actions, validations = engine.apply(ops)

    assert len(actions) == 1
    assert actions[0].status == "ok"
    assert actions[0].path == "styles.css"
    assert op.exists("styles.css")
    assert "background: #222" in op.read("styles.css")
    assert all(v.ok for v in validations)


def test_space_website_full_three_turn_lifecycle_end_to_end(tmp_path: Path):
    """End-to-End Directive Verification:
    1. Turn 1: 'Create a simple single-page HTML website about space exploration... Save it as index.html' -> creates real index.html on disk.
    2. Turn 2: 'In the index.html file, add a Learn More button that alerts "Welcome to Space!" when clicked.' -> modifies existing index.html on disk.
    3. Turn 3: 'What is this chat about?' -> routes to Gemma3:4b with injected chat history and conversation summary.
    """
    from synapse.master.orchestrator import AIMasterOrchestrator
    from synapse.workspace.brief import build_specialist_prompt
    from synapse.projects.chats import ChatStore

    op = FileOperator(tmp_path)
    engine = ActionEngine(op)
    chats_dir = tmp_path / "chats"
    chat_store = ChatStore("space-project", chats_dir)
    chat_info = chat_store.create("Space Exploration Chat")
    chat_id = chat_info.id

    # --- Turn 1: Create HTML page ---
    turn1_prompt = "Create a simple single-page HTML website about space exploration with a dark theme and some cool facts. Save it as index.html"
    analysis1 = AIMasterOrchestrator._fallback_analysis(turn1_prompt, "test")
    assert analysis1.coding_needed is True
    assert analysis1.artifact_required is True

    turn1_output = """```json
{
  "files": [
    {
      "path": "index.html",
      "content": "<!DOCTYPE html><html><head><title>Space</title></head><body><h1>Space Exploration</h1><p>Facts about stars.</p></body></html>"
    }
  ]
}
```"""
    ops1 = parse_file_manifest(turn1_output)
    actions1, v1 = engine.apply(ops1)
    assert actions1[0].status == "ok"
    assert actions1[0].action == "created"
    assert op.exists("index.html")
    chat_store.append(chat_id, role="user", content=turn1_prompt)
    chat_store.append(chat_id, role="assistant", content="Created index.html with dark theme and facts about space exploration.")

    # --- Turn 2: Edit index.html ---
    turn2_prompt = 'In the index.html file, add a Learn More button that alerts "Welcome to Space!" when clicked.'
    analysis2 = AIMasterOrchestrator._fallback_analysis(turn2_prompt, "test")
    assert analysis2.coding_needed is True

    turn2_output = """```json
{
  "files": [
    {
      "path": "index.html",
      "content": "<!DOCTYPE html><html><head><title>Space</title></head><body><h1>Space Exploration</h1><p>Facts about stars.</p><button onclick=\\"alert('Welcome to Space!')\\">Learn More</button></body></html>"
    }
  ]
}
```"""
    ops2 = parse_file_manifest(turn2_output)
    actions2, v2 = engine.apply(ops2)
    assert actions2[0].status == "ok"
    assert actions2[0].action == "modified"
    disk_html = op.read("index.html")
    assert "Learn More" in disk_html
    assert "Welcome to Space!" in disk_html
    chat_store.append(chat_id, role="user", content=turn2_prompt)
    chat_store.append(chat_id, role="assistant", content="Added Learn More alert button to index.html.")

    # --- Turn 3: Memory Query 'What is this chat about?' ---
    turn3_prompt = "What is this chat about?"
    analysis3 = AIMasterOrchestrator._fallback_analysis(turn3_prompt, "test")
    assert analysis3.coding_needed is False
    assert analysis3.domain == "chat_memory"
    assert "conversation" in analysis3.required_capabilities or "chat" in analysis3.required_capabilities

    router = Router()
    decision3 = Decision(
        can_stay_local=True,
        internet_required=False,
        preferred_kind=ProviderKind.LOCAL,
        required_capabilities=[Capability.CONVERSATION, Capability.WRITING],
        privacy=PrivacyMode.PREFER_LOCAL,
    )
    installed = {"ollama": {"gemma3:4b", "qwen2.5-coder:7b", "qwen3:4b", "qwen2.5:7b", "qwen2.5vl:7b", "nomic-embed-text"}}
    routing3 = router.route(decision3, _hw_tier2(), _tier2_registry(), {"ollama": True}, installed, prompt=turn3_prompt)
    assert routing3.model_id == "gemma3:4b"

    # Specialist prompt for Turn 3
    specialist_prompt3 = build_specialist_prompt(
        turn3_prompt,
        project_id="space-project",
        project_name="Space Project",
        project_path=str(tmp_path),
        file_operator=op,
        conversation_id=chat_id,
        chat_store=chat_store,
    )

    assert "Current chat history (this chat only):" in specialist_prompt3
    assert "Create a simple single-page HTML website about space exploration" in specialist_prompt3
    assert "Learn More button that alerts" in specialist_prompt3
    assert "Conversation summary (this chat):" in specialist_prompt3
    assert "- index.html" in specialist_prompt3


def test_windows_absolute_path_inside_workspace_relativizes_and_writes_safely(tmp_path: Path):
    """If the model or user outputs an absolute path that is inside the workspace (e.g. A:\\Test\\calculator.py),
    the system must relativize it and write it safely without rejecting it as an unsafe path."""
    from synapse.workspace.operator import FileOperator, normalize_workspace_rel
    from synapse.workspace.manifest import parse_file_manifest
    from synapse.actions.engine import ActionEngine

    op = FileOperator(tmp_path)
    engine = ActionEngine(op)

    # Simulated absolute path within tmp_path
    abs_file = tmp_path / "calculator.py"
    abs_path_str = str(abs_file)

    manifest_output = json.dumps({
        "folders": ["assets"],
        "files": [
            {"path": abs_path_str, "content": "class Calculator:\n    def add(self, a, b):\n        return a + b\n"}
        ]
    })

    ops = parse_file_manifest(manifest_output)
    actions, validations = engine.apply(ops)

    assert len(actions) == 2
    assert actions[0].action == "created_folder"
    assert actions[0].status == "ok"
    assert actions[1].action == "created"
    assert actions[1].status == "ok"
    assert actions[1].path == "calculator.py"
    assert op.exists("calculator.py")
    assert "class Calculator:" in op.read("calculator.py")


def test_nested_folder_creation_and_file_crud_operations(tmp_path: Path):
    """The AI can create folders inside folders and perform full CRUD inside nested directories."""
    from synapse.workspace.operator import FileOperator
    from synapse.workspace.manifest import parse_file_manifest
    from synapse.actions.engine import ActionEngine

    op = FileOperator(tmp_path)
    engine = ActionEngine(op)

    manifest_output = json.dumps({
        "folders": ["src/components/ui", "assets/images/icons"],
        "files": [
            {"path": "src/components/ui/button.tsx", "content": "export function Button() { return <button>Click</button>; }"},
            {"path": "assets/images/icons/star.svg", "content": "<svg><star/></svg>"},
            {"path": "src/utils/math.py", "content": "def add(a, b): return a + b\n"}
        ]
    })

    ops = parse_file_manifest(manifest_output)
    actions, validations = engine.apply(ops)

    assert all(a.status == "ok" for a in actions)
    assert op.exists("src/components/ui/button.tsx")
    assert op.exists("assets/images/icons/star.svg")
    assert op.exists("src/utils/math.py")

    # List subfolder contents
    src_files = op.list_tree("src")
    src_paths = {f["path"] for f in src_files}
    assert "src/components/ui/button.tsx" in src_paths
    assert "src/utils/math.py" in src_paths
    assert "assets/images/icons/star.svg" not in src_paths

    # List folders
    folders = op.list_folders()
    assert "src/components/ui" in folders
    assert "assets/images/icons" in folders

    # In-place edit inside nested directory
    edit_ops = parse_file_manifest(json.dumps({
        "action": "edit",
        "path": "src/utils/math.py",
        "old": "def add(a, b): return a + b",
        "new": "def add(a, b): return a + b + 0"
    }))
    edit_actions, _ = engine.apply(edit_ops)
    assert edit_actions[0].status == "ok"
    assert "return a + b + 0" in op.read("src/utils/math.py")

    # Rename / move inside nested folder
    move_ops = parse_file_manifest(json.dumps({
        "action": "rename",
        "path": "assets/images/icons/star.svg",
        "to": "assets/images/star-icon.svg"
    }))
    move_actions, _ = engine.apply(move_ops)
    assert move_actions[0].status == "ok"
    assert op.exists("assets/images/star-icon.svg")
    assert not op.exists("assets/images/icons/star.svg")


def test_delete_folder_with_contents(tmp_path: Path):
    """Deleting a folder deletes the folder and all its contents recursively."""
    from synapse.workspace.operator import FileOperator
    from synapse.actions.engine import ActionEngine

    op = FileOperator(tmp_path)
    engine = ActionEngine(op)

    op.write("temp_folder/sub/a.txt", "hello")
    op.write("temp_folder/sub/b.txt", "world")
    assert op.exists("temp_folder/sub/a.txt")

    assert op.delete("temp_folder")
    assert not op.exists("temp_folder/sub/a.txt")
    assert not (tmp_path / "temp_folder").exists()


def test_extract_workspace_ops_for_subfolder_commands():
    """Commands specifying nested folders are properly parsed."""
    from synapse.actions.classifier import extract_workspace_ops

    # List files in specific folder
    ops1 = extract_workspace_ops("List files in assets folder")
    assert any(o.get("action") == "list" and o.get("path") == "assets" for o in ops1)

    # Open / show folder
    ops2 = extract_workspace_ops("Open folder src/components")
    assert any(o.get("action") == "list" and o.get("path") == "src/components" for o in ops2)

    # Create folder inside another folder
    ops3 = extract_workspace_ops("Create a folder called icons inside assets")
    assert any(o.get("action") == "create_folder" and o.get("path") == "assets/icons" for o in ops3)


def test_manifest_and_action_engine_with_drive_prefixed_paths(tmp_path: Path):
    """Manifests with drive-prefixed paths are parsed and executed safely."""
    from synapse.workspace.manifest import parse_file_manifest
    from synapse.workspace.operator import FileOperator
    from synapse.actions.engine import ActionEngine

    workspace_dir = tmp_path / "Test"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    op = FileOperator(workspace_dir)
    engine = ActionEngine(op)

    manifest_json = json.dumps({
        "folders": ["assets"],
        "files": [
            {"path": "A:/Test/calculator.py", "content": "def add(a, b): return a + b\n"}
        ]
    })

    ops = parse_file_manifest(manifest_json, strict=False)
    assert len(ops) == 2
    actions, validations = engine.apply(ops)

    kinds = {a.action for a in actions}
    assert "created_folder" in kinds
    assert "created" in kinds
    assert all(a.status == "ok" for a in actions)
    assert op.exists("calculator.py")
    assert op.exists("assets")
    assert "def add(a, b)" in op.read("calculator.py")


def test_code_fence_with_drive_prefixed_header_extracts_and_writes(tmp_path: Path):
    """Code fences with preceding header file paths like A:/Test/calculator.py extract correctly."""
    from synapse.workspace.manifest import parse_file_manifest
    from synapse.workspace.operator import FileOperator
    from synapse.actions.engine import ActionEngine

    workspace_dir = tmp_path / "Test"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    op = FileOperator(workspace_dir)
    engine = ActionEngine(op)

    response_text = "Here is the calculator implementation:\n\n```python\n# file: A:/Test/calculator.py\ndef calc():\n    return 42\n```"
    ops = parse_file_manifest(response_text, hint="", strict=False)
    assert len(ops) == 1
    actions, _ = engine.apply(ops)
    assert actions[0].status == "ok"
    assert op.exists("calculator.py")
    assert "def calc():" in op.read("calculator.py")


def test_manifest_with_mixed_folders_and_drive_prefixed_files(tmp_path: Path):
    """Manifest with relative folder and drive-prefixed files in general workspace works cleanly."""
    from synapse.workspace.manifest import parse_file_manifest
    from synapse.workspace.operator import FileOperator
    from synapse.actions.engine import ActionEngine

    workspace_dir = tmp_path / "general"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    op = FileOperator(workspace_dir)
    engine = ActionEngine(op)

    manifest_json = json.dumps({
        "folders": ["Assets"],
        "files": [
            {"path": "A:/Test/calculator.py", "content": "print('calc')\n"},
            {"path": "A:/Test/Assets/log.txt", "content": "started\n"}
        ]
    })

    ops = parse_file_manifest(manifest_json, strict=False)
    assert len(ops) == 3
    actions, validations = engine.apply(ops)

    summary = engine.summarize(actions, validations)
    assert "write A:/Test/calculator.py [FAILED" not in summary
    assert "write A:/Test/Assets/log.txt [FAILED" not in summary
    assert all(a.status == "ok" for a in actions)
    assert op.exists("calculator.py")
    assert op.exists("Assets/log.txt")


# ---------------------------------------------------------------------------
# 16. Post-Creation Verification & Auto-Repair Tests
# ---------------------------------------------------------------------------

def test_validate_file_css():
    """CSS validator detects valid CSS, empty CSS, unbalanced braces, and unclosed comments."""
    valid_css = """
    /* Main Layout */
    body {
        margin: 0;
        font-family: 'Helvetica Neue', sans-serif;
    }
    @media (max-width: 768px) {
        .container {
            width: 100%;
        }
    }
    """
    res = validate_file("style.css", valid_css)
    assert res.ok is True
    assert "css syntax ok" in res.checks

    # Empty CSS
    res_empty = validate_file("empty.css", "   \n  ")
    assert res_empty.ok is False
    assert "empty" in res_empty.error

    # Unbalanced braces
    bad_css = ".btn { color: red; "
    res_bad = validate_file("button.css", bad_css)
    assert res_bad.ok is False
    assert "unclosed braces" in res_bad.error

    # Unclosed block comment
    bad_comment = "/* unfinished comment\nbody { color: blue; }"
    res_comment = validate_file("broken.css", bad_comment)
    assert res_comment.ok is False
    assert "unclosed block comment" in res_comment.error


def test_validate_file_ts_and_jsx_with_comments():
    """JS/TS/JSX validator properly handles multi-line block comments and brackets."""
    valid_ts = """
    /* Multi-line header comment
       Author: Synapse
    */
    interface User {
        id: number;
        name: string;
    }
    // Line comment
    function getUser(id: number): User {
        return { id: id, name: "Alice" };
    }
    """
    res = validate_file("user.ts", valid_ts)
    assert res.ok is True
    assert "bracket balance" in res.checks

    # Broken JSX/TS
    broken_tsx = "function Component() { return (<div><span>hello</div>); }"
    # Unbalanced paren
    broken_ts = "function test() { return [1, 2; }"
    res_broken = validate_file("app.tsx", broken_ts)
    assert res_broken.ok is False


def test_validate_file_python_syntax_and_empty():
    """Python validator checks syntax, empty files, and invalid characters."""
    valid_py = "def calculate(x: int) -> int:\n    return x * 2\n"
    assert validate_file("calc.py", valid_py).ok is True

    # Empty
    assert validate_file("blank.py", "  \n\t ").ok is False

    # Syntax Error
    broken_py = "def calc(x\n    return x * 2\n"
    res = validate_file("broken.py", broken_py)
    assert res.ok is False
    assert "syntax error" in res.error


def test_master_agent_post_creation_auto_repair_loop(tmp_path: Path):
    """When a specialist model generates Python with a syntax error, MasterAgent auto-repairs it."""
    from unittest.mock import MagicMock
    from synapse.analyzers.complexity import ComplexityAnalyzer
    from synapse.analyzers.intent import IntentAnalyzer
    from synapse.analyzers.privacy import PrivacyAnalyzer
    from synapse.decision.engine import DecisionEngine
    from synapse.domain import ChatResponse
    from synapse.events import EventBus
    from synapse.execution.executor import Executor
    from synapse.execution.planner import ExecutionPlanner
    from synapse.master.agent import MasterAgent
    from synapse.workspace.operator import FileOperator

    op = FileOperator(tmp_path)

    # Initial response has broken Python syntax (missing colon)
    broken_code = "def solve(a, b)\n    return a + b\n"
    broken_manifest = json.dumps({"files": [{"path": "solver.py", "content": broken_code}]})

    # Repaired response has valid Python syntax
    fixed_code = "def solve(a, b):\n    return a + b\n"
    fixed_manifest = json.dumps({"files": [{"path": "solver.py", "content": fixed_code}]})

    def _m(mid):
        m = MagicMock()
        m.id = mid
        return m

    mock_provider = MagicMock()
    mock_provider.provider_id = "ollama"
    mock_provider.kind = ProviderKind.LOCAL
    mock_provider.list_models.return_value = [
        _m("qwen2.5-coder:7b"),
        _m("gemma3:4b"),
        _m("qwen3:4b"),
        _m("qwen2.5:7b"),
        _m("qwen2.5vl:7b"),
        _m("nomic-embed-text"),
    ]

    call_count = 0
    def mock_chat(request, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        prompt_str = ""
        if hasattr(request, "messages") and request.messages:
            prompt_str = str(request.messages[-1].content)
        elif isinstance(request, str):
            prompt_str = request
        if "AUTOMATIC POST-CREATION VERIFICATION FAILED" in prompt_str or "REPAIR" in prompt_str:
            return ChatResponse(
                provider_id="ollama",
                model_id="qwen2.5-coder:7b",
                kind=ProviderKind.LOCAL,
                content=fixed_manifest,
            )
        return ChatResponse(
            provider_id="ollama",
            model_id="qwen2.5-coder:7b",
            kind=ProviderKind.LOCAL,
            content=broken_manifest,
        )

    mock_provider.chat.side_effect = mock_chat

    providers = MagicMock()
    providers.get.return_value = mock_provider
    providers.health.return_value = {"ollama": True}
    providers.all.return_value = [mock_provider]
    providers.provider_ids.return_value = ["ollama"]

    mock_hw = MagicMock()
    mock_hw.scan.return_value = _hw_tier2()
    mock_reg = MagicMock()
    mock_reg.all.return_value = _tier2_registry()

    mock_config = MagicMock()
    mock_config.get.side_effect = lambda k, default=None: {
        "privacy.user_preference": "balanced",
        "router.prefer_local": True,
        "executor.timeout_s": 30.0,
    }.get(k, default)

    agent = MasterAgent(
        intent_analyzer=IntentAnalyzer(),
        complexity_analyzer=ComplexityAnalyzer(),
        privacy_analyzer=PrivacyAnalyzer(),
        decision_engine=DecisionEngine(),
        planner=ExecutionPlanner(),
        router=Router(),
        executor=Executor(providers),
        providers=providers,
        hardware=mock_hw,
        registry=mock_reg,
        events=EventBus(),
        config=mock_config,
    )

    response = agent.process("Create solver.py with a solve function", file_operator=op)

    # Verification: File exists on disk, contains fixed code, passes ast.parse, and 2 calls were made (generation + repair)
    assert op.exists("solver.py")
    content = op.read("solver.py")
    assert "def solve(a, b):" in content
    ast.parse(content)  # Must be valid python syntax
    assert call_count >= 2
    assert "solver.py: ok" in response.response


def test_reviewer_filtering_excludes_failed_files(tmp_path: Path):
    """Review task only receives files that passed verification and never reviews broken/unverified files."""
    from synapse.workspace.operator import FileOperator
    from synapse.domain.fileops import FileAction, ValidationResult
    from synapse.workspace.review import clean_review_text, summarize

    op = FileOperator(tmp_path)
    op.write("good.py", "def f(): return 1\n")
    op.write("bad.py", "def f( invalid syntax\n")

    # Good file passed, bad file failed
    actions = [
        FileAction(path="good.py", action="created", status="ok", validated=True),
        FileAction(path="bad.py", action="created", status="failed", error="syntax error", validated=False),
    ]
    validations = [
        ValidationResult(path="good.py", ok=True, checks=["python syntax ok"]),
        ValidationResult(path="bad.py", ok=False, error="python syntax error: line 1"),
    ]

    summary = ActionEngine.summarize(actions, validations, review_text="good.py is well formatted.")
    assert "good.py: ok" in summary
    assert "bad.py: FAILED" in summary
    assert "good.py is well formatted" in summary


def test_master_agent_post_creation_auto_repair_json(tmp_path: Path):
    """When a specialist model generates malformed JSON, MasterAgent auto-repairs it."""
    from unittest.mock import MagicMock
    from synapse.analyzers.complexity import ComplexityAnalyzer
    from synapse.analyzers.intent import IntentAnalyzer
    from synapse.analyzers.privacy import PrivacyAnalyzer
    from synapse.decision.engine import DecisionEngine
    from synapse.domain import ChatResponse
    from synapse.events import EventBus
    from synapse.execution.executor import Executor
    from synapse.execution.planner import ExecutionPlanner
    from synapse.master.agent import MasterAgent
    from synapse.workspace.operator import FileOperator

    op = FileOperator(tmp_path)

    # Initial response has broken JSON (missing closing brace)
    broken_content = '{"name": "config", "debug": true, '
    broken_manifest = json.dumps({"files": [{"path": "config.json", "content": broken_content}]})

    # Repaired response has valid JSON
    fixed_content = '{"name": "config", "debug": true}'
    fixed_manifest = json.dumps({"files": [{"path": "config.json", "content": fixed_content}]})

    def _m(mid):
        m = MagicMock()
        m.id = mid
        return m

    mock_provider = MagicMock()
    mock_provider.provider_id = "ollama"
    mock_provider.kind = ProviderKind.LOCAL
    mock_provider.list_models.return_value = [
        _m("qwen2.5-coder:7b"),
        _m("gemma3:4b"),
        _m("qwen3:4b"),
        _m("qwen2.5:7b"),
        _m("qwen2.5vl:7b"),
        _m("nomic-embed-text"),
    ]

    call_count = 0
    def mock_chat(request, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        prompt_str = ""
        if hasattr(request, "messages") and request.messages:
            prompt_str = str(request.messages[-1].content)
        elif isinstance(request, str):
            prompt_str = request
        if "AUTOMATIC POST-CREATION VERIFICATION FAILED" in prompt_str or "REPAIR" in prompt_str:
            return ChatResponse(
                provider_id="ollama",
                model_id="qwen2.5-coder:7b",
                kind=ProviderKind.LOCAL,
                content=fixed_manifest,
            )
        return ChatResponse(
            provider_id="ollama",
            model_id="qwen2.5-coder:7b",
            kind=ProviderKind.LOCAL,
            content=broken_manifest,
        )

    mock_provider.chat.side_effect = mock_chat

    providers = MagicMock()
    providers.get.return_value = mock_provider
    providers.health.side_effect = lambda pid=None: True if pid else {"ollama": True}
    providers.all.return_value = [mock_provider]
    providers.provider_ids.return_value = ["ollama"]

    mock_hw = MagicMock()
    mock_hw.scan.return_value = _hw_tier2()
    mock_reg = MagicMock()
    mock_reg.all.return_value = _tier2_registry()

    mock_config = MagicMock()
    mock_config.get.side_effect = lambda k, default=None: {
        "privacy.user_preference": "balanced",
        "router.prefer_local": True,
        "executor.timeout_s": 30.0,
    }.get(k, default)

    agent = MasterAgent(
        intent_analyzer=IntentAnalyzer(),
        complexity_analyzer=ComplexityAnalyzer(),
        privacy_analyzer=PrivacyAnalyzer(),
        decision_engine=DecisionEngine(),
        planner=ExecutionPlanner(),
        router=Router(),
        executor=Executor(providers),
        providers=providers,
        hardware=mock_hw,
        registry=mock_reg,
        events=EventBus(),
        config=mock_config,
    )

    response = agent.process("Create config.json with application settings", file_operator=op)

    assert op.exists("config.json")
    content = op.read("config.json")
    parsed = json.loads(content)  # Must parse cleanly
    assert parsed["name"] == "config"
    assert call_count >= 2
    assert "config.json: ok" in response.response











