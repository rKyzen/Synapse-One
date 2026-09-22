"""Tests for the project index system."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synapse.workspace.project_index import ProjectIndex, FileIndex, SymbolInfo


class TestProjectIndex:
    """Tests for ProjectIndex."""

    def test_index_creation(self, tmp_path):
        index = ProjectIndex(tmp_path)
        assert index.root == tmp_path

    def test_scan_empty_directory(self, tmp_path):
        index = ProjectIndex(tmp_path)
        structure = index.scan()
        assert structure.total_files == 0

    def test_scan_python_files(self, tmp_path):
        # Create test files
        (tmp_path / "main.py").write_text("def hello(): pass", encoding="utf-8")
        (tmp_path / "utils.py").write_text("import os", encoding="utf-8")

        index = ProjectIndex(tmp_path)
        structure = index.scan()
        assert structure.total_files == 2
        assert structure.languages.get("python") == 2
        assert structure.main_language == "python"

    def test_detect_entry_points(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hello')", encoding="utf-8")

        index = ProjectIndex(tmp_path)
        structure = index.scan()
        assert "main.py" in structure.entry_points

    def test_detect_config_files(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("flask", encoding="utf-8")

        index = ProjectIndex(tmp_path)
        structure = index.scan()
        assert "requirements.txt" in structure.config_files

    def test_extract_python_imports(self, tmp_path):
        (tmp_path / "app.py").write_text(
            "import os\nfrom pathlib import Path\nimport flask",
            encoding="utf-8",
        )

        index = ProjectIndex(tmp_path)
        index.scan()
        file_idx = index.get_file("app.py")
        assert file_idx is not None
        assert "os" in file_idx.imports
        assert "pathlib" in file_idx.imports  # Module name extracted
        assert "flask" in file_idx.imports

    def test_extract_python_symbols(self, tmp_path):
        (tmp_path / "models.py").write_text(
            "class User:\n    def __init__(self): pass\n\ndef create_user(): pass",
            encoding="utf-8",
        )

        index = ProjectIndex(tmp_path)
        index.scan()
        file_idx = index.get_file("models.py")
        assert file_idx is not None
        symbol_names = [s.name for s in file_idx.symbols]
        assert "User" in symbol_names
        assert "create_user" in symbol_names

    def test_search_symbols(self, tmp_path):
        (tmp_path / "a.py").write_text("class Foo: pass", encoding="utf-8")
        (tmp_path / "b.py").write_text("class FooBar: pass", encoding="utf-8")

        index = ProjectIndex(tmp_path)
        index.scan()
        results = index.search_symbols("Foo")
        assert len(results) == 2

    def test_get_context_files(self, tmp_path):
        (tmp_path / "auth.py").write_text("class Auth: pass", encoding="utf-8")
        (tmp_path / "utils.py").write_text("def helper(): pass", encoding="utf-8")

        index = ProjectIndex(tmp_path)
        index.scan()
        files = index.get_context_files("auth")
        assert len(files) > 0
        assert any("auth" in f.path for f in files)

    def test_update_file(self, tmp_path):
        (tmp_path / "test.py").write_text("x = 1", encoding="utf-8")

        index = ProjectIndex(tmp_path)
        index.scan()
        assert index.get_file("test.py") is not None

        # Update the file
        (tmp_path / "test.py").write_text("x = 2", encoding="utf-8")
        entry = index.update_file("test.py")
        assert entry is not None

    def test_remove_file(self, tmp_path):
        (tmp_path / "temp.py").write_text("temp", encoding="utf-8")

        index = ProjectIndex(tmp_path)
        index.scan()
        assert index.get_file("temp.py") is not None

        index.remove_file("temp.py")
        assert index.get_file("temp.py") is None

    def test_skip_hidden_directories(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("git config", encoding="utf-8")
        (tmp_path / "main.py").write_text("x = 1", encoding="utf-8")

        index = ProjectIndex(tmp_path)
        structure = index.scan()
        assert structure.total_files == 1

    def test_skip_node_modules(self, tmp_path):
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg.js").write_text("pkg", encoding="utf-8")
        (tmp_path / "app.js").write_text("console.log('hi')", encoding="utf-8")

        index = ProjectIndex(tmp_path)
        structure = index.scan()
        assert structure.total_files == 1


class TestFileIndex:
    """Tests for FileIndex dataclass."""

    def test_file_index_creation(self):
        entry = FileIndex(
            path="test.py",
            name="test.py",
            extension=".py",
            size=100,
            modified_at="2024-01-01T00:00:00",
            language="python",
        )
        assert entry.path == "test.py"
        assert entry.language == "python"
