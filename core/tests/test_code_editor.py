"""Tests for the code editing engine."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synapse.actions.code_editor import CodeEditor, EditOperation
from synapse.workspace.operator import FileOperator


class TestCodeEditor:
    """Tests for CodeEditor."""

    def test_editor_creation(self, tmp_path):
        operator = FileOperator(tmp_path)
        editor = CodeEditor(operator)
        assert editor._operator == operator

    def test_find_and_replace(self, tmp_path):
        (tmp_path / "test.py").write_text("x = 1\ny = 2", encoding="utf-8")
        operator = FileOperator(tmp_path)
        editor = CodeEditor(operator)

        result = editor.find_and_replace("test.py", "x = 1", "x = 100")
        assert result.success
        assert result.changes_made == 1
        assert (tmp_path / "test.py").read_text(encoding="utf-8") == "x = 100\ny = 2"

    def test_add_import(self, tmp_path):
        (tmp_path / "test.py").write_text("def hello(): pass", encoding="utf-8")
        operator = FileOperator(tmp_path)
        editor = CodeEditor(operator)

        result = editor.add_import("test.py", "import os")
        assert result.success
        content = (tmp_path / "test.py").read_text(encoding="utf-8")
        assert "import os" in content

    def test_add_import_duplicate(self, tmp_path):
        (tmp_path / "test.py").write_text("import os\ndef hello(): pass", encoding="utf-8")
        operator = FileOperator(tmp_path)
        editor = CodeEditor(operator)

        result = editor.add_import("test.py", "import os")
        assert result.success
        assert result.changes_made == 0  # No change needed

    def test_replace_function(self, tmp_path):
        (tmp_path / "test.py").write_text(
            "def greet():\n    return 'hello'\n\ndef main():\n    pass",
            encoding="utf-8",
        )
        operator = FileOperator(tmp_path)
        editor = CodeEditor(operator)

        result = editor.replace_function("test.py", "greet", "def greet():\n    return 'world'")
        assert result.success
        content = (tmp_path / "test.py").read_text(encoding="utf-8")
        assert "world" in content

    def test_edit_nonexistent_file(self, tmp_path):
        operator = FileOperator(tmp_path)
        editor = CodeEditor(operator)

        result = editor.find_and_replace("missing.py", "old", "new")
        assert not result.success
        assert "not found" in result.error

    def test_edit_operations(self, tmp_path):
        (tmp_path / "test.py").write_text("line1\nline2\nline3", encoding="utf-8")
        operator = FileOperator(tmp_path)
        editor = CodeEditor(operator)

        ops = [EditOperation(kind="append", target="", content="line4")]
        result = editor.edit_file("test.py", ops)
        assert result.success
        assert result.changes_made == 1
