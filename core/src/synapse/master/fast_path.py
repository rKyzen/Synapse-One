"""Deterministic Fast Path for Synapse One.

Provides immediate (<1ms), zero-LLM execution for trivial requests:
- Basic arithmetic & percentage calculations (safe AST evaluation)
- Standard greetings and social pleasantries
- System identity and health ping queries
- Fast-chat bypass for trivial single-turn conversational queries

Bypasses Master Model pre-flight analysis for maximum performance and zero overhead.
"""

from __future__ import annotations

import ast
import math
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel

from synapse.domain.enums import IntentType


class FastPathType(str, Enum):
    """Category of fast-path handling applied."""

    DIRECT_ARITHMETIC = "direct_arithmetic"
    DIRECT_GREETING = "direct_greeting"
    DIRECT_IDENTITY = "direct_identity"
    WORKSPACE_QUERY = "workspace_query"
    LOADED_MODELS = "loaded_models"
    SYSTEM_STATUS = "system_status"
    FAST_CHAT = "fast_chat"


class FastPathResult(BaseModel):
    """Result of fast-path analysis."""

    is_fast_path: bool = False
    path_type: FastPathType | None = None
    direct_response: str | None = None
    intent: IntentType = IntentType.CONVERSATION
    complexity: int = 1


#: Pure greetings
_GREETINGS = {
    "hi", "hello", "hey", "good morning", "good afternoon",
    "good evening", "howdy", "greetings", "sup", "yo",
    "hi there", "hello there", "hey there", "hola",
    "hey synapse", "hello synapse", "hi synapse", "good day",
    "morning", "evening", "afternoon", "hi there synapse",
    "hello there synapse", "hey there synapse", "whats up", "what's up",
}

#: Polite gratitude & farewells
_THANKS = {
    "thanks", "thank you", "thanks a lot", "thank you very much",
    "thanks so much", "thank you so much", "thx", "ty",
    "great thanks", "thanks a bunch", "much appreciated",
    "thank u", "tyvm", "thank you kindly", "thx so much", "thx a lot",
}

_FAREWELLS = {
    "bye", "goodbye", "see you", "see ya", "cya",
    "have a nice day", "have a good day", "good night",
    "bye bye", "take care", "see ya later", "goodbye synapse",
    "see you later", "catch you later", "talk to you later", "gn",
}

#: Identity questions
_IDENTITY = {
    "who are you", "what are you", "what is your name",
    "what's your name", "who created you", "who made you",
    "what can you do", "who created synapse", "what is synapse",
    "what is synapse one", "tell me who you are", "introduce yourself",
    "help", "what are your capabilities", "capabilities", "features",
    "what is your purpose",
}

_PING = {"ping", "test ping", "status check", "are you alive", "health ping", "pong"}

#: Instant zero-LLM workspace inspection patterns
_WORKSPACE_QUERIES = {
    "list files", "show files", "show project structure",
    "project structure", "list workspace", "show workspace",
    "list directory", "show directory", "what files are here",
    "what files are in this project", "tree", "workspace tree",
    "project files", "list project files", "show workspace files",
    "show files in workspace", "show files in project", "files in project",
    "show directory structure", "dir", "ls", "workspace files",
    "list workspace files",
}

#: Instant zero-LLM loaded models & status queries
_LOADED_MODELS_QUERIES = {
    "what models are loaded", "loaded models", "list models",
    "list loaded models", "model status", "models loaded",
    "which models are loaded", "active models", "which models are running",
    "show loaded models", "show active models", "running models",
    "models", "model list", "active model", "running model", "show models",
}

_SYSTEM_STATUS_QUERIES = {
    "system status", "health check", "hardware status",
    "system health", "overall status", "is system healthy",
    "check system health", "check health", "status", "sys status",
    "health", "system specs", "specs", "hardware info", "hardware specs",
}

#: Quick conversational acknowledgements & affirmations
_AFFIRMATIONS = {
    "ok", "okay", "sure", "cool", "nice", "awesome", "great",
    "got it", "understood", "yes", "no", "yep", "nope",
    "sounds good", "alright", "all right", "perfect",
}


class _SafeMathVisitor(ast.NodeVisitor):
    """Strict AST visitor that allows only safe arithmetic operations."""

    ALLOWED_NODES = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Call,
        ast.Name,
        ast.Load,
    )

    ALLOWED_FUNCS = {
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "ceil": math.ceil,
        "floor": math.floor,
    }

    def generic_visit(self, node: ast.AST) -> Any:
        if not isinstance(node, self.ALLOWED_NODES):
            raise ValueError(f"Unsafe node type: {type(node).__name__}")
        return super().generic_visit(node)

    def evaluate(self, node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return self.evaluate(node.body)
        if isinstance(node, ast.Constant):
            val = node.value
            if not isinstance(val, (int, float)):
                raise ValueError("Only numeric constants allowed")
            return val
        if isinstance(node, ast.UnaryOp):
            operand = self.evaluate(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            raise ValueError("Unsupported unary operator")
        if isinstance(node, ast.BinOp):
            left = self.evaluate(node.left)
            right = self.evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError("division by zero")
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                if right == 0:
                    raise ZeroDivisionError("division by zero")
                return left // right
            if isinstance(node.op, ast.Mod):
                if right == 0:
                    raise ZeroDivisionError("division by zero")
                return left % right
            if isinstance(node.op, ast.Pow):
                if abs(right) > 1000:
                    raise ValueError("Exponent too large")
                return left ** right
            raise ValueError("Unsupported binary operator")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only named math functions allowed")
            func_name = node.func.id.lower()
            if func_name not in self.ALLOWED_FUNCS:
                raise ValueError(f"Function '{func_name}' not permitted")
            func = self.ALLOWED_FUNCS[func_name]
            args = [self.evaluate(arg) for arg in node.args]
            return func(*args)
        raise ValueError(f"Unsupported node type: {type(node).__name__}")


def evaluate_arithmetic(expr: str) -> str | None:
    """Safely parse and evaluate an arithmetic string expression. Returns None on failure."""
    expr = expr.strip()
    # Normalize operators: ^ -> **
    expr = re.sub(r"\^", "**", expr)
    try:
        tree = ast.parse(expr, mode="eval")
        visitor = _SafeMathVisitor()
        visitor.visit(tree)
        result = visitor.evaluate(tree)
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        if isinstance(result, float):
            return f"{result:.6g}"
        return str(result)
    except Exception:  # noqa: BLE001
        return None


def _try_math_fast_path(cleaned: str) -> str | None:
    """Extract and evaluate arithmetic from common phrasing."""
    # Match percentage: "what is 15% of 200", "20% of 85"
    pct_match = re.match(
        r"^(?:what\s+is\s+|calculate\s+|compute\s+)?(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)(\d+(?:\.\d+)?)\??$",
        cleaned,
        re.IGNORECASE,
    )
    if pct_match:
        try:
            pct = float(pct_match.group(1))
            total = float(pct_match.group(2))
            res = (pct / 100.0) * total
            return str(int(res)) if res.is_integer() else f"{res:.6g}"
        except Exception:
            pass

    # Match direct calculation prefixes: "what is 2 + 2", "calculate 15 * 8", "100 / 4", "sqrt(144)"
    calc_prefix_match = re.match(
        r"^(?:what\s+is\s+|calculate\s+|compute\s+|evaluate\s+|how\s+much\s+is\s+|solve\s+)?\s*([0-9\.\+\-\*\/\%\^\(\)\s\w,]+)\??$",
        cleaned,
        re.IGNORECASE,
    )
    if calc_prefix_match:
        candidate = calc_prefix_match.group(1).strip()
        has_operator = bool(re.search(r"[\+\-\*\/\%\^]", candidate))
        has_func = any(fn in candidate.lower() for fn in _SafeMathVisitor.ALLOWED_FUNCS)
        if (has_operator or has_func) and re.search(r"\d", candidate):
            # Guard against code or words mixed in (e.g. "what is 2 + 2 in python")
            words = [w for w in re.findall(r"[a-zA-Z]+", candidate) if w.lower() not in _SafeMathVisitor.ALLOWED_FUNCS]
            if not words:
                res = evaluate_arithmetic(candidate)
                if res is not None:
                    return res

    return None


def check_fast_path(prompt: str, has_files: bool = False) -> FastPathResult:
    """Evaluate prompt against deterministic fast-path rules."""
    if has_files or not prompt or not isinstance(prompt, str):
        return FastPathResult(is_fast_path=False)

    text = prompt.strip()
    if not text:
        return FastPathResult(is_fast_path=False)

    # Clean punctuation for greeting/intent matching
    normalized = re.sub(r"[^\w\s\+\-\*\/\%\^\(\)\.]", "", text.lower()).strip()
    lowered_raw = text.lower().strip().rstrip("!?.,")

    # 1. Arithmetic fast path
    math_result = _try_math_fast_path(text)
    if math_result is not None:
        return FastPathResult(
            is_fast_path=True,
            path_type=FastPathType.DIRECT_ARITHMETIC,
            direct_response=f"The answer is {math_result}.",
            intent=IntentType.CONVERSATION,
            complexity=1,
        )

    # 2. Greetings
    if lowered_raw in _GREETINGS or normalized in _GREETINGS:
        return FastPathResult(
            is_fast_path=True,
            path_type=FastPathType.FAST_CHAT,
            direct_response=None,
            intent=IntentType.CONVERSATION,
            complexity=1,
        )

    # 3. Thanks
    if lowered_raw in _THANKS or normalized in _THANKS:
        return FastPathResult(
            is_fast_path=True,
            path_type=FastPathType.FAST_CHAT,
            direct_response=None,
            intent=IntentType.CONVERSATION,
            complexity=1,
        )

    # 4. Farewells
    if lowered_raw in _FAREWELLS or normalized in _FAREWELLS:
        return FastPathResult(
            is_fast_path=True,
            path_type=FastPathType.FAST_CHAT,
            direct_response=None,
            intent=IntentType.CONVERSATION,
            complexity=1,
        )

    # 5. Identity & Health
    if lowered_raw in _IDENTITY or normalized in _IDENTITY:
        return FastPathResult(
            is_fast_path=True,
            path_type=FastPathType.FAST_CHAT,
            direct_response=None,
            intent=IntentType.CONVERSATION,
            complexity=1,
        )

    if lowered_raw in _PING or normalized in _PING:
        return FastPathResult(
            is_fast_path=True,
            path_type=FastPathType.DIRECT_IDENTITY,
            direct_response="pong",
            intent=IntentType.CONVERSATION,
            complexity=1,
        )

    # 6. Workspace Queries (e.g. "list files", "show project structure")
    if lowered_raw in _WORKSPACE_QUERIES or normalized in _WORKSPACE_QUERIES:
        return FastPathResult(
            is_fast_path=True,
            path_type=FastPathType.WORKSPACE_QUERY,
            direct_response=None,
            intent=IntentType.RESEARCH,
            complexity=1,
        )

    # 7. Model and System Status
    if lowered_raw in _LOADED_MODELS_QUERIES or normalized in _LOADED_MODELS_QUERIES:
        return FastPathResult(
            is_fast_path=True,
            path_type=FastPathType.LOADED_MODELS,
            direct_response=None,
            intent=IntentType.CONVERSATION,
            complexity=1,
        )

    if lowered_raw in _SYSTEM_STATUS_QUERIES or normalized in _SYSTEM_STATUS_QUERIES:
        return FastPathResult(
            is_fast_path=True,
            path_type=FastPathType.SYSTEM_STATUS,
            direct_response=None,
            intent=IntentType.CONVERSATION,
            complexity=1,
        )

    # 8. Affirmations & Acknowledgements ("ok", "sure", "cool", "nice", etc.)
    if lowered_raw in _AFFIRMATIONS or normalized in _AFFIRMATIONS:
        return FastPathResult(
            is_fast_path=True,
            path_type=FastPathType.FAST_CHAT,
            direct_response=None,
            intent=IntentType.CONVERSATION,
            complexity=1,
        )

    # 9. Fast Chat Bypass (for trivial small prompts that don't need pre-flight analysis)
    # e.g. "Tell me a joke", "How are you", "Give me a quote", "What is the capital of..."
    is_simple_conversational = (
        len(text.split()) <= 8
        and not any(
            w in lowered_raw
            for w in (
                "create", "build", "make", "generate", "code", "file", "project",
                "workspace", "html", "css", "python", "script", "doc", "test",
                "proof", "solve", "math", "why", "explain", "analyze", "review",
                "edit", "delete", "rename", "run", "search", "list"
            )
        )
    )
    if is_simple_conversational and lowered_raw.startswith(
        ("tell me a", "how are you", "give me a", "say something", "what is the capital of", "who is the president of")
    ):
        return FastPathResult(
            is_fast_path=True,
            path_type=FastPathType.FAST_CHAT,
            direct_response=None,
            intent=IntentType.CONVERSATION,
            complexity=5,
        )

    return FastPathResult(is_fast_path=False)
