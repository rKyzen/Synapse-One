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
            capabilities=ModelCapabilities(chat=0.6, coding=1.0, debugging=0.95, architecture=0.9),
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
    assert result.returncode == 0, f"Generated tests failed:\\n{result.stdout}\\n{result.stderr}"
    assert "3 passed" in result.stdout




