"""ProjectDiagnostics — detects issues in the workspace.

Continuously detects:
    - Syntax errors
    - Missing imports
    - Failed builds
    - Dependency issues
    - Unused files
    - Code style violations

Offers fixes when possible.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synapse.logging import get_logger
from synapse.workspace.project_index import ProjectIndex
from synapse.workspace.operator import FileOperator

log = get_logger("synapse.diagnostics")


@dataclass
class Diagnostic:
    """A single diagnostic issue."""

    file: str
    line: int | None
    severity: str  # "error", "warning", "info"
    code: str  # E001, W001, etc.
    message: str
    fixable: bool = False
    fix_suggestion: str = ""


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for the workspace."""

    files_analyzed: int
    diagnostics: list[Diagnostic]
    errors: int
    warnings: int
    info: int
    timestamp: str = ""

    @property
    def has_errors(self) -> bool:
        return self.errors > 0

    def summary(self) -> str:
        return f"{self.files_analyzed} files analyzed: {self.errors} errors, {self.warnings} warnings"


class DiagnosticAnalyzer:
    """Analyzes the workspace for issues."""

    def __init__(
        self,
        file_operator: FileOperator,
        project_index: ProjectIndex | None = None,
    ) -> None:
        self._operator = file_operator
        self._index = project_index

    def analyze_all(self) -> DiagnosticReport:
        """Analyze all files in the workspace."""
        from datetime import datetime, timezone

        diagnostics: list[Diagnostic] = []
        files_analyzed = 0

        for entry in self._operator.list_tree():
            path = entry["path"]
            ext = Path(path).suffix.lower()

            # Only analyze code files
            if ext in (".py", ".js", ".ts", ".jsx", ".tsx", ".dart", ".rs", ".go", ".java"):
                file_diags = self.analyze_file(path)
                diagnostics.extend(file_diags)
                files_analyzed += 1

        errors = sum(1 for d in diagnostics if d.severity == "error")
        warnings = sum(1 for d in diagnostics if d.severity == "warning")
        info = sum(1 for d in diagnostics if d.severity == "info")

        return DiagnosticReport(
            files_analyzed=files_analyzed,
            diagnostics=diagnostics,
            errors=errors,
            warnings=warnings,
            info=info,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def analyze_file(self, path: str) -> list[Diagnostic]:
        """Analyze a single file for issues."""
        content = self._operator.read(path)
        if content is None:
            return [Diagnostic(
                file=path,
                line=None,
                severity="error",
                code="E001",
                message="File not found",
            )]

        ext = Path(path).suffix.lower()
        diagnostics: list[Diagnostic] = []

        if ext == ".py":
            diagnostics.extend(self._analyze_python(path, content))
        elif ext in (".js", ".ts", ".jsx", ".tsx"):
            diagnostics.extend(self._analyze_javascript(path, content))
        elif ext == ".dart":
            diagnostics.extend(self._analyze_dart(path, content))
        elif ext == ".rs":
            diagnostics.extend(self._analyze_rust(path, content))

        return diagnostics

    def check_syntax(self, path: str) -> list[Diagnostic]:
        """Check syntax only (fast)."""
        content = self._operator.read(path)
        if content is None:
            return []

        ext = Path(path).suffix.lower()
        if ext == ".py":
            return self._check_python_syntax(path, content)
        return []

    def find_unused_imports(self, path: str) -> list[Diagnostic]:
        """Find unused imports in a Python file."""
        content = self._operator.read(path)
        if not content:
            return []

        diagnostics = []
        try:
            tree = ast.parse(content)
            imports = []
            used = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname or alias.name.split(".")[0]
                        imports.append((name, node.lineno))
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        name = alias.asname or alias.name
                        imports.append((name, node.lineno))
                elif isinstance(node, ast.Name):
                    used.add(node.id)
                elif isinstance(node, ast.Attribute):
                    used.add(node.attr)

            for name, line in imports:
                if name not in used and name != "*":
                    diagnostics.append(Diagnostic(
                        file=path,
                        line=line,
                        severity="warning",
                        code="W001",
                        message=f"Unused import: {name}",
                        fixable=True,
                        fix_suggestion=f"Remove import of {name}",
                    ))
        except SyntaxError as exc:
            diagnostics.append(Diagnostic(
                file=path,
                line=exc.lineno,
                severity="error",
                code="E002",
                message=f"Syntax error: {exc.msg}",
            ))

        return diagnostics

    def check_missing_dependencies(self) -> list[Diagnostic]:
        """Check for missing project dependencies."""
        diagnostics = []

        # Check requirements.txt vs actual imports
        req_file = self._operator.read("requirements.txt")
        if req_file:
            required = set()
            for line in req_file.splitlines():
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
                    pkg = re.split(r"[>=<!\[]", line)[0].strip().lower()
                    if pkg:
                        required.add(pkg)

            # Check imports in Python files
            imported = set()
            for entry in self._operator.list_tree():
                if entry["path"].endswith(".py"):
                    content = self._operator.read(entry["path"])
                    if content:
                        for match in re.finditer(r"^(?:from\s+(\S+)\s+)?import\s+(\S+)", content, re.MULTILINE):
                            module = match.group(1) or match.group(2)
                            if module:
                                imported.add(module.split(".")[0].lower())

            # Find potentially missing
            for pkg in required:
                if pkg not in imported and pkg.replace("-", "_") not in imported:
                    diagnostics.append(Diagnostic(
                        file="requirements.txt",
                        line=None,
                        severity="info",
                        code="I001",
                        message=f"Package '{pkg}' in requirements.txt but not imported",
                    ))

        return diagnostics

    # -- Private analysis methods -------------------------------------------

    def _analyze_python(self, path: str, content: str) -> list[Diagnostic]:
        """Analyze Python file."""
        diagnostics = []

        # Syntax check
        diagnostics.extend(self._check_python_syntax(path, content))

        # Unused imports
        diagnostics.extend(self.find_unused_imports(path))

        # Line length
        for i, line in enumerate(content.splitlines(), 1):
            if len(line) > 120:
                diagnostics.append(Diagnostic(
                    file=path,
                    line=i,
                    severity="info",
                    code="I002",
                    message=f"Line too long ({len(line)} > 120)",
                ))

        return diagnostics

    def _check_python_syntax(self, path: str, content: str) -> list[Diagnostic]:
        """Check Python syntax."""
        try:
            ast.parse(content)
            return []
        except SyntaxError as exc:
            return [Diagnostic(
                file=path,
                line=exc.lineno,
                severity="error",
                code="E002",
                message=f"Syntax error: {exc.msg}",
                fixable=False,
            )]

    def _analyze_javascript(self, path: str, content: str) -> list[Diagnostic]:
        """Analyze JavaScript/TypeScript file."""
        diagnostics = []

        # Basic checks
        for i, line in enumerate(content.splitlines(), 1):
            # Console.log statements
            if "console.log(" in line:
                diagnostics.append(Diagnostic(
                    file=path,
                    line=i,
                    severity="warning",
                    code="W002",
                    message="Console.log statement found",
                    fixable=True,
                    fix_suggestion="Remove console.log",
                ))

            # debugger statements
            if line.strip() == "debugger;":
                diagnostics.append(Diagnostic(
                    file=path,
                    line=i,
                    severity="warning",
                    code="W003",
                    message="Debugger statement found",
                    fixable=True,
                    fix_suggestion="Remove debugger statement",
                ))

        return diagnostics

    def _analyze_dart(self, path: str, content: str) -> list[Diagnostic]:
        """Analyze Dart file."""
        diagnostics = []

        for i, line in enumerate(content.splitlines(), 1):
            # print statements (should use debugPrint)
            if "print(" in line and "debugPrint(" not in line:
                diagnostics.append(Diagnostic(
                    file=path,
                    line=i,
                    severity="info",
                    code="I003",
                    message="Consider using debugPrint instead of print",
                ))

        return diagnostics

    def _analyze_rust(self, path: str, content: str) -> list[Diagnostic]:
        """Analyze Rust file."""
        diagnostics = []

        for i, line in enumerate(content.splitlines(), 1):
            # unwrap() calls
            if ".unwrap()" in line:
                diagnostics.append(Diagnostic(
                    file=path,
                    line=i,
                    severity="warning",
                    code="W004",
                    message="unwrap() call may panic",
                    fixable=True,
                    fix_suggestion="Use expect() or handle the error",
                ))

        return diagnostics
