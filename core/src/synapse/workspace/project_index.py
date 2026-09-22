"""ProjectIndex — maintains a comprehensive index of the workspace.

Tracks:
    - Files (path, size, type, modified time)
    - Folders (structure)
    - Imports (Python, JS, TS, etc.)
    - Dependencies (package.json, requirements.txt, Cargo.toml, etc.)
    - Symbols (classes, functions, variables)
    - Documentation (README, docs)
    - Configuration files

The index is updated incrementally after edits and provides fast lookups
for context management and search.
"""

from __future__ import annotations

import ast
import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synapse.logging import get_logger

log = get_logger("synapse.workspace.project_index")


@dataclass
class FileIndex:
    """Index entry for a single file."""

    path: str
    name: str
    extension: str
    size: int
    modified_at: str
    language: str
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    symbols: list[SymbolInfo] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    is_test: bool = False
    is_config: bool = False
    is_documentation: bool = False


@dataclass
class SymbolInfo:
    """Information about a symbol (class, function, variable)."""

    name: str
    kind: str  # "class", "function", "variable", "method"
    line: int
    end_line: int | None = None
    parent: str | None = None  # class name for methods
    docstring: str | None = None
    signature: str | None = None


@dataclass
class ProjectStructure:
    """High-level project structure."""

    root: str
    total_files: int = 0
    total_folders: int = 0
    languages: dict[str, int] = field(default_factory=dict)
    main_language: str = ""
    has_tests: bool = False
    has_docs: bool = False
    config_files: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    dependencies_file: str | None = None


# Language detection by extension
LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".dart": "dart",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "shell",
    ".bash": "shell",
    ".ps1": "powershell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".md": "markdown",
    ".txt": "text",
}

# Config file patterns
CONFIG_PATTERNS = [
    "package.json",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    ".gitignore",
    ".env",
    "Makefile",
    "CMakeLists.txt",
    "docker-compose.yml",
    "Dockerfile",
]

# Test file patterns
TEST_PATTERNS = [
    "test_",
    "_test.py",
    "_test.js",
    "_test.ts",
    ".test.js",
    ".test.ts",
    ".spec.js",
    ".spec.ts",
    "tests/",
    "__tests__/",
]


class ProjectIndex:
    """Maintains a comprehensive index of the workspace."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._lock = threading.RLock()
        self._files: dict[str, FileIndex] = {}
        self._structure: ProjectStructure | None = None
        self._last_scan: datetime | None = None

    @property
    def root(self) -> Path:
        return self._root

    def scan(self) -> ProjectStructure:
        """Perform a full scan of the workspace."""
        with self._lock:
            self._files.clear()
            self._scan_directory(self._root)
            self._structure = self._build_structure()
            self._last_scan = datetime.now(timezone.utc)
            log.info(
                "project_indexed",
                root=str(self._root),
                files=len(self._files),
                languages=self._structure.languages,
            )
            return self._structure

    def update_file(self, rel_path: str) -> FileIndex | None:
        """Update index for a single file (after write/edit)."""
        with self._lock:
            path = self._root / rel_path
            if not path.is_file():
                # File was deleted
                self._files.pop(rel_path, None)
                return None
            entry = self._index_file(path)
            if entry:
                self._files[rel_path] = entry
            return entry

    def remove_file(self, rel_path: str) -> None:
        """Remove a file from the index."""
        with self._lock:
            self._files.pop(rel_path, None)

    def get_file(self, rel_path: str) -> FileIndex | None:
        """Get index entry for a file."""
        return self._files.get(rel_path)

    def get_all_files(self) -> list[FileIndex]:
        """Get all indexed files."""
        return list(self._files.values())

    def get_files_by_language(self, language: str) -> list[FileIndex]:
        """Get all files of a specific language."""
        return [f for f in self._files.values() if f.language == language]

    def search_symbols(self, query: str) -> list[tuple[FileIndex, SymbolInfo]]:
        """Search for symbols by name."""
        results = []
        query_lower = query.lower()
        for file_idx in self._files.values():
            for symbol in file_idx.symbols:
                if query_lower in symbol.name.lower():
                    results.append((file_idx, symbol))
        return results

    def search_imports(self, module: str) -> list[FileIndex]:
        """Find files that import a specific module."""
        results = []
        module_lower = module.lower()
        for file_idx in self._files.values():
            for imp in file_idx.imports:
                if module_lower in imp.lower():
                    results.append(file_idx)
                    break
        return results

    def get_dependencies(self) -> dict[str, list[str]]:
        """Get project dependencies from config files."""
        deps: dict[str, list[str]] = {}

        # Python dependencies
        for name in ["requirements.txt", "pyproject.toml", "setup.py"]:
            path = self._root / name
            if path.is_file():
                deps[name] = self._parse_python_deps(path)
                break

        # Node.js dependencies
        pkg_json = self._root / "package.json"
        if pkg_json.is_file():
            deps["package.json"] = self._parse_node_deps(pkg_json)

        # Rust dependencies
        cargo = self._root / "Cargo.toml"
        if cargo.is_file():
            deps["Cargo.toml"] = self._parse_rust_deps(cargo)

        return deps

    def get_import_graph(self) -> dict[str, list[str]]:
        """Build a dependency graph based on imports."""
        graph: dict[str, list[str]] = {}
        for path, file_idx in self._files.items():
            graph[path] = file_idx.imports
        return graph

    def get_context_files(self, query: str, max_files: int = 5) -> list[FileIndex]:
        """Get the most relevant files for a query.

        Uses symbol names, file names, and import relationships.
        """
        scores: list[tuple[float, FileIndex]] = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for file_idx in self._files.values():
            score = 0.0

            # Score by file name match
            if any(w in file_idx.name.lower() for w in query_words):
                score += 2.0

            # Score by symbol names
            for symbol in file_idx.symbols:
                if any(w in symbol.name.lower() for w in query_words):
                    score += 1.5

            # Score by imports
            for imp in file_idx.imports:
                if any(w in imp.lower() for w in query_words):
                    score += 0.5

            # Boost entry points and config files
            if file_idx.is_config:
                score += 0.3

            if score > 0:
                scores.append((score, file_idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scores[:max_files]]

    # -- Private methods -----------------------------------------------------

    def _scan_directory(self, directory: Path, depth: int = 0) -> None:
        """Recursively scan a directory."""
        if depth > 10:  # Prevent infinite recursion
            return

        try:
            for item in directory.iterdir():
                # Skip hidden directories and common non-project dirs
                if item.name.startswith(".") or item.name in ("node_modules", "__pycache__", "venv", ".venv"):
                    continue

                if item.is_file():
                    entry = self._index_file(item)
                    if entry:
                        try:
                            rel_path = str(item.relative_to(self._root)).replace("\\", "/")
                            self._files[rel_path] = entry
                        except ValueError:
                            pass
                elif item.is_dir():
                    self._scan_directory(item, depth + 1)
        except PermissionError:
            pass

    def _index_file(self, path: Path) -> FileIndex | None:
        """Index a single file."""
        try:
            stat = path.stat()
            ext = path.suffix.lower()
            language = LANGUAGE_MAP.get(ext, "unknown")
            name = path.name

            # Determine file category
            is_test = any(p in name.lower() or p in str(path).lower() for p in TEST_PATTERNS)
            is_config = name in CONFIG_PATTERNS or any(p in name for p in CONFIG_PATTERNS)
            is_doc = ext in (".md", ".rst", ".txt") and name.lower() in ("readme.md", "readme.txt", "readme.rst", "changelog.md")

            # Parse imports and symbols for code files
            imports = []
            symbols = []
            if language in ("python", "javascript", "typescript", "dart", "rust", "go", "java"):
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                    imports = self._extract_imports(content, language)
                    symbols = self._extract_symbols(content, language, path.name)
                except Exception:
                    pass

            return FileIndex(
                path=str(path.relative_to(self._root)).replace("\\", "/"),
                name=name,
                extension=ext,
                size=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                language=language,
                imports=imports,
                symbols=symbols,
                is_test=is_test,
                is_config=is_config,
                is_documentation=is_doc,
            )
        except Exception:
            return None

    def _extract_imports(self, content: str, language: str) -> list[str]:
        """Extract import statements from code."""
        imports = []

        if language == "python":
            # import x / from x import y
            for match in re.finditer(r"^(?:from\s+(\S+)\s+)?import\s+(\S+)", content, re.MULTILINE):
                module = match.group(1) or match.group(2)
                if module:
                    imports.append(module.split(".")[0])

        elif language in ("javascript", "typescript"):
            # import ... from '...' / require('...')
            for match in re.finditer(r"""(?:from\s+['"]([^'"]+)['"]|require\s*\(\s*['"]([^'"]+)['"]\s*\))""", content):
                module = match.group(1) or match.group(2)
                if module and not module.startswith("."):
                    imports.append(module)

        elif language == "dart":
            # import 'package:...'
            for match in re.finditer(r"import\s+'package:([^/]+)", content):
                imports.append(match.group(1))

        elif language == "rust":
            # use crate::... / extern crate ...
            for match in re.finditer(r"(?:extern\s+crate\s+(\w+)|use\s+(?:crate::)?(\w+))", content):
                crate = match.group(1) or match.group(2)
                if crate:
                    imports.append(crate)

        elif language == "go":
            # import "..."
            for match in re.finditer(r'import\s+"([^"]+)"', content):
                imports.append(match.group(1).split("/")[-1])

        elif language == "java":
            # import ...
            for match in re.finditer(r"import\s+([\w.]+)", content):
                imports.append(match.group(1).split(".")[-1])

        return list(set(imports))

    def _extract_symbols(self, content: str, language: str, filename: str) -> list[SymbolInfo]:
        """Extract symbols (classes, functions) from code."""
        symbols = []

        if language == "python":
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        symbols.append(SymbolInfo(
                            name=node.name,
                            kind="class",
                            line=node.lineno,
                            end_line=getattr(node, "end_lineno", None),
                            docstring=ast.get_docstring(node),
                        ))
                    elif isinstance(node, ast.FunctionDef):
                        parent = None
                        for parent_node in ast.walk(tree):
                            if isinstance(parent_node, ast.ClassDef):
                                for child in ast.iter_child_nodes(parent_node):
                                    if child is node:
                                        parent = parent_node.name
                                        break
                        symbols.append(SymbolInfo(
                            name=node.name,
                            kind="function" if not parent else "method",
                            line=node.lineno,
                            end_line=getattr(node, "end_lineno", None),
                            parent=parent,
                            docstring=ast.get_docstring(node),
                            signature=f"def {node.name}(...)",
                        ))
            except SyntaxError:
                pass

        elif language in ("javascript", "typescript"):
            # Simple regex-based extraction
            for match in re.finditer(r"(?:class|function|const|let|var)\s+(\w+)", content):
                kind = "class" if "class" in content[match.start():match.end()+20] else "function"
                symbols.append(SymbolInfo(
                    name=match.group(1),
                    kind=kind,
                    line=content[:match.start()].count("\n") + 1,
                ))

        return symbols

    def _build_structure(self) -> ProjectStructure:
        """Build high-level project structure summary."""
        languages: dict[str, int] = {}
        config_files = []
        entry_points = []

        for file_idx in self._files.values():
            # Count languages
            if file_idx.language != "unknown":
                languages[file_idx.language] = languages.get(file_idx.language, 0) + 1

            # Collect config files
            if file_idx.is_config:
                config_files.append(file_idx.path)

            # Detect entry points
            if file_idx.name in ("main.py", "app.py", "index.js", "index.ts", "main.dart", "main.rs", "main.go"):
                entry_points.append(file_idx.path)

        # Determine main language
        main_language = max(languages.items(), key=lambda x: x[1])[0] if languages else ""

        # Detect dependencies file
        dep_files = ["requirements.txt", "pyproject.toml", "package.json", "Cargo.toml", "go.mod"]
        dependencies_file = next((f for f in dep_files if (self._root / f).is_file()), None)

        return ProjectStructure(
            root=str(self._root),
            total_files=len(self._files),
            total_folders=len(set(
                str(Path(f.path).parent) for f in self._files.values()
                if str(Path(f.path).parent) != "."
            )),
            languages=languages,
            main_language=main_language,
            has_tests=any(f.is_test for f in self._files.values()),
            has_docs=any(f.is_documentation for f in self._files.values()),
            config_files=config_files,
            entry_points=entry_points,
            dependencies_file=dependencies_file,
        )

    def _parse_python_deps(self, path: Path) -> list[str]:
        """Parse Python dependency file."""
        deps = []
        try:
            content = path.read_text(encoding="utf-8")
            if path.name == "requirements.txt":
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("-"):
                        dep = re.split(r"[>=<!\[]", line)[0].strip()
                        if dep:
                            deps.append(dep)
            elif path.name == "pyproject.toml":
                # Simple extraction
                for match in re.finditer(r'"([a-zA-Z0-9_-]+)', content):
                    deps.append(match.group(1))
        except Exception:
            pass
        return deps

    def _parse_node_deps(self, path: Path) -> list[str]:
        """Parse package.json dependencies."""
        deps = []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            deps.extend(data.get("dependencies", {}).keys())
            deps.extend(data.get("devDependencies", {}).keys())
        except Exception:
            pass
        return deps

    def _parse_rust_deps(self, path: Path) -> list[str]:
        """Parse Cargo.toml dependencies."""
        deps = []
        try:
            content = path.read_text(encoding="utf-8")
            for match in re.finditer(r"(\w+)\s*=\s*", content):
                deps.append(match.group(1))
        except Exception:
            pass
        return deps

    def to_dict(self) -> dict:
        """Serialize index for API responses."""
        return {
            "root": str(self._root),
            "total_files": len(self._files),
            "structure": self._structure.__dict__ if self._structure else None,
            "files": {
                path: {
                    "name": f.name,
                    "language": f.language,
                    "size": f.size,
                    "symbols": len(f.symbols),
                    "imports": len(f.imports),
                }
                for path, f in self._files.items()
            },
        }
