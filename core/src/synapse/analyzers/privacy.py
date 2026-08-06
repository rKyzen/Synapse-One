"""Privacy analysis.

Decides LOCAL_ONLY / PREFER_LOCAL / BALANCED / PREFER_CLOUD / CLOUD_REQUIRED
from: sensitivity signals, internet requirement, complexity, user preference,
and provider availability (context). Deterministic and offline.
"""

from __future__ import annotations

from synapse.contracts import PrivacyAnalyzer
from synapse.domain import ComplexityResult, PrivacyMode, PrivacyResult

_SENSITIVE = ("password", "secret", "credit card", "ssn", "medical", "private key", "passport", "bank account")
_INTERNET = ("web search", "latest news", "internet", "does google say", "weather today", "current price", "stock price", "website live")

#: User preference fallback when context has none.
_DEFAULT_PREF = "balanced"


def _internet_required(text: str) -> bool:
    return any(kw in text for kw in _INTERNET)


def _sensitivity(text: str) -> tuple[float, list[str]]:
    hits = [k for k in _SENSITIVE if k in text]
    return (min(1.0, 0.3 * len(hits)), hits)


class PrivacyAnalyzer(PrivacyAnalyzer):
    """Rule-based privacy decision engine."""

    def analyze(
        self,
        prompt: str,
        complexity: ComplexityResult,
        context: dict | None = None,
    ) -> PrivacyResult:
        context = context or {}
        text = prompt.lower()

        internet = _internet_required(text)
        sensitivity, s_hits = _sensitivity(text)
        preferred = str(context.get("user_preference") or _DEFAULT_PREF).lower()

        availability = context.get("provider_availability") or {}  # {kind: bool}
        local_ok = availability.get("local", True)
        cloud_ok = availability.get("cloud", True)

        reasons: list[str] = []
        mode = PrivacyMode.BALANCED

        # 1. Hard privacy: sensitive data forces local.
        if sensitivity > 0:
            if local_ok:
                mode = PrivacyMode.LOCAL_ONLY
                reasons.append(f"sensitive terms [{', '.join(s_hits)}] → local-only")
            else:
                mode = PrivacyMode.PREFER_LOCAL
                reasons.append("sensitive terms but no local provider; degrade to prefer-local")

        # 2. Internet requirement forces cloud.
        elif internet:
            if cloud_ok:
                mode = PrivacyMode.CLOUD_REQUIRED
                reasons.append("internet-dependent task → cloud required")
            else:
                mode = PrivacyMode.PREFER_LOCAL
                reasons.append("internet-dependent but no cloud provider; using local")

        # 3. Complexity + preference.
        elif complexity.score >= 85:
            if cloud_ok:
                mode = PrivacyMode.PREFER_CLOUD
                reasons.append(f"high complexity ({complexity.score}) → prefer cloud reasoning")
            else:
                mode = PrivacyMode.PREFER_LOCAL
                reasons.append(f"high complexity but no cloud; prefer local")
        elif preferred == "local":
            mode = PrivacyMode.PREFER_LOCAL
            reasons.append("user prefers local execution")
        elif preferred == "cloud":
            mode = PrivacyMode.PREFER_CLOUD
            reasons.append("user prefers cloud execution")
        else:
            mode = PrivacyMode.BALANCED
            reasons.append("default balanced policy")

        if not reasons:
            reasons.append("no special privacy signals")

        return PrivacyResult(
            mode=mode,
            internet_required=internet,
            sensitivity=sensitivity,
            reasoning=reasons,
        )