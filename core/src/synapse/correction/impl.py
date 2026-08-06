"""Heuristic Hallucination Detection & Self-Correction — Phase 4.

Detects common hallucination patterns without external services:
- ``document.set_title()``-style invented APIs
- references to files that do not exist in the workspace
- fake package names (heuristic: "pip install <package>" where the package
  name never appears in the project)
- fabricated citations (references like "see [1]" with no sources provided)

Self-correction applies safe, surgical text fixes and otherwise flags the
uncertainty explicitly rather than guessing.
"""

from __future__ import annotations

import re

from synapse.contracts.correction import (
    HallucinationDetector,
    HallucinationFlag,
    HallucinationType,
    SelfCorrector,
)

_PYTHON_FN_CALL = re.compile(r"\b([a-z_][a-z0-9_]*)\(\)")
_FILE_REF = re.compile(r"(?:in|at|`)?([\w./-]+\.(?:py|md|txt|json|toml|cfg|ini|yaml|yml|log|csv|sql|js|ts|tsx|jsx|html|css))", re.IGNORECASE)
_PACKAGE_REF = re.compile(r"pip install\s+([\w.-]+)")
_CITATION_REF = re.compile(r"(?:see|per|according to)\s+\[(\d+)\]|\[(\d+)\]", re.IGNORECASE)


class HeuristicHallucinationDetector(HallucinationDetector):
    """Scans for the common hallucination classes listed in the contract."""

    def scan(
        self,
        response: str,
        *,
        prompt: str,
        context: str | None = None,
        available_apis: list[str] | None = None,
        existing_files: list[str] | None = None,
        installed_packages: list[str] | None = None,
        project_functions: list[str] | None = None,
    ) -> list[HallucinationFlag]:
        del prompt
        flags: list[HallucinationFlag] = []

        # 1. Invented APIs — function calls not in the known API surface.
        if available_apis:
            known = {a.strip() for a in available_apis if a.strip()}
            for match in _PYTHON_FN_CALL.finditer(response):
                name = match.group(1)
                if name in {"def", "if", "for", "while", "print", "len", "range"}:
                    continue
                if name not in known:
                    flags.append(
                        HallucinationFlag(
                            type=HallucinationType.INVENTED_API,
                            span=match.group(0),
                            reason=f"'{name}' not in known APIs",
                            confidence=0.5,
                        )
                    )

        # 2. Non-existent files.
        if existing_files:
            known_files = {f.lower() for f in existing_files}
            for match in _FILE_REF.finditer(response):
                path = match.group(1)
                if path.lower() not in known_files:
                    flags.append(
                        HallucinationFlag(
                            type=HallucinationType.NONEXISTENT_FILE,
                            span=path,
                            reason=f"'{path}' not in workspace",
                            confidence=0.5,
                        )
                    )

        # 3. Fake packages — pip install targets that are not installed.
        if installed_packages:
            known = {p.lower() for p in installed_packages}
            for match in _PACKAGE_REF.finditer(response):
                pkg = match.group(1)
                if pkg.lower() not in known:
                    flags.append(
                        HallucinationFlag(
                            type=HallucinationType.FAKE_PACKAGE,
                            span=f"pip install {pkg}",
                            reason=f"'{pkg}' is not an installed package",
                            confidence=0.4,
                        )
                    )

        # 4. Fake citations — numeric references with no sources provided.
        if _CITATION_REF.search(response) and not context:
            span = _CITATION_REF.search(response).group(0)
            flags.append(
                HallucinationFlag(
                    type=HallucinationType.FAKE_CITATION,
                    span=span,
                    reason="citation reference with no sources provided",
                    confidence=0.6,
                )
            )

        return flags


class HeuristicSelfCorrector(SelfCorrector):
    """Applies conservative text-level corrections to flagged responses."""

    def correct(
        self,
        response: str,
        flags: list[HallucinationFlag],
        *,
        prompt: str,
        context: str | None = None,
    ) -> str:
        del prompt, context
        corrected = response
        file_flags = [f for f in flags if f.type == HallucinationType.NONEXISTENT_FILE]
        citation_flags = [f for f in flags if f.type == HallucinationType.FAKE_CITATION]

        if file_flags:
            spans = {f.span for f in file_flags}
            for span in spans:
                corrected = corrected.replace(span, f"`{span}` (not found in workspace)")
                corrected = corrected.replace(
                    f"``{span}``", f"`{span}` (not found in workspace)"
                )

        if citation_flags:
            corrected = re.sub(
                r"(?:see|per|according to)\s+\[(\d+)\]",
                r"[source not provided]",
                corrected,
                flags=re.IGNORECASE,
            )

        if corrected == response:
            corrected = response + "\n\n(Note: parts of this answer could not be verified against available sources.)"

        return corrected
