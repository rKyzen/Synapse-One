"""Workspace folder scanner — discover, classify, and import existing files.

Phase X: opening an existing folder as a project should feel like opening a
project in an IDE — the folder's contents are walked (respecting ignore
rules), its language/framework stack is detected from manifest files, and the
files are imported into the project's internal workspace for indexing and
retrieval. Pure logic: no models, no I/O beyond reading files.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

IGNORED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "venv", ".venv", "env", "__pycache__",
    ".idea", ".vscode", ".tox", ".nox", "dist", "build", "target", ".next",
    ".nuxt", ".cache", ".pytest_cache", ".mypy_cache", "tmp", "temp", "coverage",
}

MAX_DEPTH = 10
MAX_FILES = 400
MAX_FILE_BYTES = 25 * 1024 * 1024

#: extension -> primary language label (histogram fallback when no manifest).
LANGUAGE_BY_EXT = {
    "py": "Python", "pyi": "Python",
    "js": "JavaScript", "mjs": "JavaScript", "cjs": "JavaScript",
    "ts": "TypeScript", "tsx": "TypeScript (React)", "jsx": "JavaScript (React)",
    "html": "HTML", "htm": "HTML", "css": "CSS", "scss": "SCSS",
    "go": "Go", "rs": "Rust", "rb": "Ruby", "java": "Java", "kt": "Kotlin",
    "cs": "C#", "php": "PHP", "cpp": "C++", "cc": "C++", "h": "C/C++",
    "swift": "Swift", "scala": "Scala", "ex": "Elixir", "exs": "Elixir",
    "vue": "Vue", "svelte": "Svelte", "dart": "Dart",
}

#: manifest files to look at (in priority order) for stack detection.
MANIFESTS = [
    "package.json", "pyproject.toml", "requirements.txt", "Pipfile", "setup.py",
    "go.mod", "Cargo.toml", "pom.xml", "build.gradle", "build.gradle.kts",
    "Gemfile", "composer.json", "pubspec.yaml", "Package.swift", "mix.exs",
]

_WS = re.compile(r"\s+")
_DEP_NAME = re.compile(r'"([a-zA-Z0-9@_\-./]+)"\s*:')


def iter_project_files(root: str | Path) -> list[Path]:
    """Walk a project folder, honouring ignore rules and size/depth caps."""
    root = Path(root)
    if not root.is_dir():
        return []
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        rel_depth = Path(dirpath).relative_to(root).parts
        if len(rel_depth) > MAX_DEPTH:
            dirnames[:] = []
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            out.append(p)
            if len(out) >= MAX_FILES:
                return out
    return out


def _package_json_deps(folder: Path) -> dict[str, str]:
    """Framework detection from package.json dependencies."""
    pkg = folder / "package.json"
    if not pkg.is_file():
        return {}
    try:
        data = json.loads(pkg.read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return {}
    merged: dict[str, str] = {}
    for section in ("dependencies", "devDependencies", "peerDependencies"):
        deps = data.get(section)
        if isinstance(deps, dict):
            merged.update({k.lower(): str(v) for k, v in deps.items()})
    return merged


def _python_stack(folder: Path) -> tuple[str, str]:
    """Python language + web framework from common manifest files."""
    framework = ""
    needles = {
        "django": "Django", "flask": "Flask", "fastapi": "FastAPI",
        "uvicorn": "FastAPI", "streamlit": "Streamlit", "pyramid": "Pyramid",
        "tornado": "Tornado", "quart": "Quart",
    }
    for manifest in ("pyproject.toml", "requirements.txt", "Pipfile", "setup.py"):
        path = folder / manifest
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        for needle, label in needles.items():
            if needle in text:
                framework = label
                break
        if framework:
            break
    return "Python", framework


def _node_stack(folder: Path, has_ts: bool) -> tuple[str, str]:
    deps = _package_json_deps(folder)
    if not deps:
        return "TypeScript" if has_ts else "JavaScript", ""
    name = "TypeScript" if has_ts else "JavaScript"
    for marker, label in (
        ("next", "Next.js"), ("nuxt", "Nuxt"), ("@nestjs/core", "NestJS"),
        ("react", "React"), ("vue", "Vue"), ("svelte", "Svelte"),
        ("@angular/core", "Angular"), ("express", "Express"), ("fastify", "Fastify"),
        ("sveltekit", "SvelteKit"), ("remix", "Remix"), ("astro", "Astro"),
    ):
        if marker in deps:
            return name, label
    return name, "Node.js"


def _csharp_stack(folder: Path) -> tuple[str, str]:
    for pattern in ("*.csproj", "*.fsproj"):
        matches = list(folder.glob(pattern))
        if not matches:
            continue
        try:
            text = matches[0].read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            text = ""
        framework = "ASP.NET Core" if ("microsoft.aspnetcore" in text or "sdk=\"microsoft.net.sdk.web" in text) else ".NET"
        return "C#", framework
    return "C#", ".NET"


def detect_stack(folder: str | Path) -> dict[str, str]:
    """Detect the dominant language + framework of a project folder.

    Returns ``{"language": str, "framework": str}``; both empty when nothing
    can be determined (e.g. an empty folder).
    """
    folder = Path(folder)
    language, framework = "", ""
    has_ts = False
    ext_counts: dict[str, int] = {}

    try:
        files = iter_project_files(folder)
    except OSError:
        files = []

    for p in files:
        ext = p.suffix.lstrip(".").lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        if p.name == "tsconfig.json" or ext in ("ts", "tsx"):
            has_ts = True

    manifest_names = {p.name for p in folder.iterdir() if p.is_file()} if folder.is_dir() else set()

    if "package.json" in manifest_names:
        language, framework = _node_stack(folder, has_ts)
    elif any(m in manifest_names for m in ("pyproject.toml", "requirements.txt", "Pipfile", "setup.py")):
        language, framework = _python_stack(folder)
    elif "go.mod" in manifest_names:
        language, framework = "Go", ""
    elif "Cargo.toml" in manifest_names:
        language, framework = "Rust", ""
    elif "pom.xml" in manifest_names or any(n in manifest_names for n in ("build.gradle", "build.gradle.kts")):
        language, framework = "Java", "Spring Boot" if "pom.xml" in manifest_names else "Gradle"
    elif "Gemfile" in manifest_names:
        gem = folder / "Gemfile"
        try:
            text = gem.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            text = ""
        language, framework = "Ruby", "Ruby on Rails" if "rails" in text else ""
    elif "composer.json" in manifest_names:
        try:
            data = json.loads((folder / "composer.json").read_text(encoding="utf-8", errors="replace"))
        except Exception:  # noqa: BLE001
            data = {}
        deps = data.get("require", {})
        language, framework = "PHP", "Laravel" if any("laravel" in k for k in deps) else ""
    elif "pubspec.yaml" in manifest_names:
        language, framework = "Dart", "Flutter"
    elif "Package.swift" in manifest_names:
        language, framework = "Swift", ""
    elif "mix.exs" in manifest_names:
        language, framework = "Elixir", "Phoenix"
    elif any(folder.glob("*.csproj")):
        language, framework = _csharp_stack(folder)

    if not language:
        if not ext_counts:
            language = ""
        else:
            max_count = max(ext_counts.values())
            top_exts = [e for e, c in ext_counts.items() if c == max_count]
            if any(e in ("html", "htm") for e in top_exts):
                language = "HTML / CSS / JavaScript"
            else:
                language = LANGUAGE_BY_EXT.get(top_exts[0], "Unknown")
    elif ext_counts and has_ts and language in ("JavaScript",):
        language = "TypeScript"

    return {"language": language, "framework": framework}
