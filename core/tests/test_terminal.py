"""Tests for the terminal runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synapse.terminal.runner import TerminalRunner, CommandResult


class TestTerminalRunner:
    """Tests for TerminalRunner."""

    def test_runner_creation(self, tmp_path):
        runner = TerminalRunner(tmp_path)
        assert runner._root == tmp_path

    def test_run_simple_command(self, tmp_path):
        runner = TerminalRunner(tmp_path, timeout=10)
        result = runner.run("echo hello")
        assert result.success
        assert "hello" in result.stdout

    def test_run_failing_command(self, tmp_path):
        runner = TerminalRunner(tmp_path, timeout=10)
        result = runner.run("exit 1")
        assert not result.success
        assert result.return_code == 1

    def test_command_history(self, tmp_path):
        runner = TerminalRunner(tmp_path, timeout=10)
        runner.run("echo a")
        runner.run("echo b")

        history = runner.get_history()
        assert len(history) == 2

    def test_command_timeout(self, tmp_path):
        runner = TerminalRunner(tmp_path, timeout=1)
        # Use ping to simulate a long-running command on Windows
        result = runner.run("ping -n 10 127.0.0.1", timeout=1)
        assert result.timed_out or not result.success  # Either timeout or killed

    def test_command_result_to_dict(self, tmp_path):
        runner = TerminalRunner(tmp_path, timeout=10)
        result = runner.run("echo test")
        d = result.to_dict()
        assert "command" in d
        assert "return_code" in d
        assert d["success"] is True
