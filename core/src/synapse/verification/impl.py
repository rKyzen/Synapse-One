"""Heuristic Verification — Phase 4.

Combines cheap, dependency-free checks:
- FactChecker: claim-vs-source containment (lexical)
- CodeVerifier: balanced braces/parens, balanced code fences, common syntax slips
- MathVerifier: arithmetic re-evaluation via the ``numbers``/``ast`` modules
- CompositeVerifier: runs all three and produces a VerificationResult

These are heuristics, not proof: high-uncertainty output still flows to the
escalation stage. Every check is crash-safe — verification must never break
the pipeline.
"""

from __future__ import annotations

import ast
import operator
import re

from synapse.contracts.verification import (
    CodeVerifier,
    FactChecker,
    MathVerifier,
    VerificationResult,
    VerificationStatus,
    Verifier,
)


class HeuristicFactChecker(FactChecker):
    """Checks a claim against sources by lexical containment."""

    def check_claim(self, claim: str, sources: list[str]) -> dict:
        claim_words = {w for w in re.findall(r"[a-z]{3,}", claim.lower())}
        if not claim_words:
            return {"supported": False, "reason": "claim has no content words", "overlap": 0.0}
        best = 0.0
        for source in sources:
            # sources may be plain strings or rich objects (RetrievedChunk,
            # memory entries); verification must never crash on either.
            if not isinstance(source, str):
                source = str(source)
            source_words = {w for w in re.findall(r"[a-z]{3,}", source.lower())}
            if not source_words:
                continue
            overlap = len(claim_words & source_words) / len(claim_words)
            best = max(best, overlap)
        supported = best >= 0.6
        return {
            "supported": supported,
            "overlap": round(best, 3),
            "reason": f"max source overlap {best:.0%}",
        }


class HeuristicCodeVerifier(CodeVerifier):
    """Syntax-level code checks: fences, brackets, obvious slips."""

    _PAIRS = {"(": ")", "[": "]", "{": "}"}

    def verify_code(self, code: str, language: str, context: str | None = None) -> dict:
        del context
        issues: list[str] = []

        fences = code.count("```")
        if fences % 2 != 0:
            issues.append("unbalanced code fences")

        for opening, closing in self._PAIRS.items():
            depth = 0
            for ch in code:
                if ch == opening:
                    depth += 1
                elif ch == closing:
                    depth -= 1
                if depth < 0:
                    break
            if depth != 0:
                issues.append(f"unbalanced '{opening}{closing}' brackets")

        if re.search(r"\bdef\s+[A-Z][a-zA-Z0-9_]*\s*\(", code):
            issues.append("function name starts with uppercase (style)")

        return {
            "ok": not issues,
            "issues": issues,
            "language": language,
            "confidence": 1.0 if not issues else 0.4,
        }


class HeuristicMathVerifier(MathVerifier):
    """Re-evaluates simple arithmetic expressions safely."""

    _ALLOWED_BINARY = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
    }

    def verify_math(self, expression: str, expected: str | None = None) -> dict:
        try:
            value = self._evaluate(expression)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"could not evaluate: {exc}", "result": None}

        if expected is not None:
            try:
                match = abs(float(value) - float(expected)) < 1e-6
            except (TypeError, ValueError):
                match = str(value) == str(expected)
            return {
                "ok": match,
                "result": value,
                "expected": expected,
                "reason": "matches expected" if match else "does not match expected",
            }
        return {"ok": True, "result": value, "reason": "evaluated successfully"}

    def _evaluate(self, expression: str) -> float:
        tree = ast.parse(expression.strip(), mode="eval")
        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in self._ALLOWED_BINARY:
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self._ALLOWED_BINARY[type(node.op)]
            try:
                return float(op(left, right))
            except ZeroDivisionError as exc:
                raise ValueError("division by zero") from exc
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._eval_node(node.operand)
        raise ValueError(f"unsupported expression node: {type(node).__name__}")


class CompositeVerifier(Verifier):
    """Runs fact/code/math checks and merges them into one VerificationResult.

    If any check fails, the response is marked CORRECTED (when we produced a
    fix) or CONTRADICTION (when the response contradicts the context).
    """

    def __init__(
        self,
        fact_checker: FactChecker | None = None,
        code_verifier: CodeVerifier | None = None,
        math_verifier: MathVerifier | None = None,
    ) -> None:
        self._facts = fact_checker or HeuristicFactChecker()
        self._code = code_verifier or HeuristicCodeVerifier()
        self._math = math_verifier or HeuristicMathVerifier()

    def verify(
        self,
        prompt: str,
        response: str,
        *,
        context: str | None = None,
        capability: str | None = None,
        expected_format: str | None = None,
    ) -> VerificationResult:
        del expected_format
        corrections: list[str] = []
        sources_used: list[str] = []
        verified = response
        confidence = 0.8

        # Code check
        if capability == "coding" or "```" in response:
            code = self._strip_fences(response)
            code_result = self._code.verify_code(code, "python", context)
            sources_used.append("code-syntax")
            if not code_result["ok"]:
                corrections.append("; ".join(code_result["issues"]))
                confidence = min(confidence, code_result["confidence"])
                verified = self._fix_code_fences(verified)
            else:
                confidence = min(confidence, 0.95)

        # Math check
        if capability == "math":
            expressions = re.findall(r"\d+(?:\s*[+\-*/^]\s*\d+)+", response)
            if expressions:
                sources_used.append("arithmetic")
                for expr in expressions[:3]:
                    math_result = self._math.verify_math(expr)
                    if not math_result["ok"] and math_result.get("reason"):
                        corrections.append(f"arithmetic error in '{expr}': {math_result['reason']}")
                        confidence = min(confidence, 0.3)

        # Fact check against provided context
        if context:
            claims = self._extract_claims(response)
            sources_used.append("context")
            supported = 0
            for claim in claims:
                result = self._facts.check_claim(claim, [context])
                if result["supported"]:
                    supported += 1
            coverage = supported / len(claims) if claims else 1.0
            if coverage < 0.5 and claims:
                corrections.append(f"only {coverage:.0%} of claims grounded in context")
                confidence = min(confidence, 0.4)

        if not corrections:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                original_response=response,
                verified_response=response,
                corrections=[],
                confidence=confidence,
                reason="all checks passed",
                sources_used=sources_used,
            )
        return VerificationResult(
            status=VerificationStatus.CORRECTED,
            original_response=response,
            verified_response=verified,
            corrections=corrections,
            confidence=confidence,
            reason="; ".join(corrections[:3]),
            sources_used=sources_used,
        )

    @staticmethod
    def _extract_claims(response: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", response)
        return [s.strip() for s in sentences if len(s.split()) >= 5]

    @staticmethod
    def _strip_fences(response: str) -> str:
        return re.sub(r"^```[a-zA-Z0-9]*\n|```$", "", response.strip(), flags=re.MULTILINE)

    @staticmethod
    def _fix_code_fences(response: str) -> str:
        if response.count("```") % 2 != 0:
            return response.rstrip() + "\n```"
        return response
