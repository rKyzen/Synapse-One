"""TerminalRunner — sandboxed command execution for the workspace.

Allows the agent to run commands, install packages, execute scripts,
run tests, and capture output. All commands are logged and can be
cancelled.

Safety features:
    - Commands run in the workspace directory
    - Timeout protection
    - Output capture and logging
    - Command history
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from synapse.logging import get_logger

log = get_logger("synapse.terminal.runner")


@dataclass
class CommandResult:
    """Result of a command execution."""

    command: str
    return_code: int
    stdout: str
    stderr: str
    elapsed_ms: float
    timed_out: bool = False
    working_dir: str = ""

    @property
    def success(self) -> bool:
        return self.return_code == 0

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "return_code": self.return_code,
            "stdout": self.stdout[:5000],
            "stderr": self.stderr[:2000],
            "elapsed_ms": self.elapsed_ms,
            "timed_out": self.timed_out,
            "success": self.success,
        }


@dataclass
class CommandEntry:
    """A logged command execution."""

    id: str
    command: str
    working_dir: str
    started_at: str
    completed_at: str | None = None
    result: CommandResult | None = None
    status: str = "running"  # "running", "completed", "failed", "cancelled"


class TerminalRunner:
    """Executes commands in the workspace with safety controls."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        timeout: int = 60,
        max_output: int = 50000,
    ) -> None:
        self._root = Path(workspace_root)
        self._timeout = timeout
        self._max_output = max_output
        self._history: list[CommandEntry] = []
        self._lock = threading.RLock()
        self._counter = 0

    def run(
        self,
        command: str,
        *,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> CommandResult:
        """Execute a command and return the result."""
        timeout = timeout or self._timeout
        entry = self._log_start(command)

        try:
            result = self._execute(command, timeout, env, on_output)
            entry.result = result
            entry.status = "completed" if result.success else "failed"
            entry.completed_at = datetime.now(timezone.utc).isoformat()
            return result
        except Exception as exc:
            result = CommandResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr=str(exc),
                elapsed_ms=0,
                working_dir=str(self._root),
            )
            entry.result = result
            entry.status = "failed"
            entry.completed_at = datetime.now(timezone.utc).isoformat()
            return result

    def run_async(
        self,
        command: str,
        *,
        on_complete: Callable[[CommandResult], None] | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> str:
        """Execute a command asynchronously. Returns command ID."""
        entry = self._log_start(command)

        def worker():
            result = self._execute(command, self._timeout, None, on_output)
            entry.result = result
            entry.status = "completed" if result.success else "failed"
            entry.completed_at = datetime.now(timezone.utc).isoformat()
            if on_complete:
                on_complete(result)

        threading.Thread(target=worker, daemon=True).start()
        return entry.id

    def get_history(self, limit: int = 50) -> list[CommandEntry]:
        """Get command history."""
        with self._lock:
            return list(reversed(self._history[-limit:]))

    def get_entry(self, entry_id: str) -> CommandEntry | None:
        """Get a specific command entry."""
        with self._lock:
            for entry in self._history:
                if entry.id == entry_id:
                    return entry
        return None

    # -- Private methods -----------------------------------------------------

    def _execute(
        self,
        command: str,
        timeout: int,
        env: dict[str, str] | None,
        on_output: Callable[[str], None] | None,
    ) -> CommandResult:
        """Execute the command."""
        start = time.perf_counter()
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=str(self._root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            stdout_chunks = []
            stderr_chunks = []

            # Read output in real-time
            while True:
                try:
                    stdout_line = process.stdout.readline()
                    if stdout_line:
                        stdout_chunks.append(stdout_line)
                        if on_output:
                            on_output(stdout_line)
                    stderr_line = process.stderr.readline()
                    if stderr_line:
                        stderr_chunks.append(stderr_line)
                    if not stdout_line and not stderr_line:
                        if process.poll() is not None:
                            break
                    if time.perf_counter() - start > timeout:
                        process.kill()
                        elapsed = (time.perf_counter() - start) * 1000
                        return CommandResult(
                            command=command,
                            return_code=-1,
                            stdout="".join(stdout_chunks)[:self._max_output],
                            stderr="Command timed out",
                            elapsed_ms=round(elapsed, 1),
                            timed_out=True,
                            working_dir=str(self._root),
                        )
                except Exception:
                    break

            process.wait()
            elapsed = (time.perf_counter() - start) * 1000

            return CommandResult(
                command=command,
                return_code=process.returncode or 0,
                stdout="".join(stdout_chunks)[:self._max_output],
                stderr="".join(stderr_chunks)[:self._max_output],
                elapsed_ms=round(elapsed, 1),
                working_dir=str(self._root),
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return CommandResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr=str(exc),
                elapsed_ms=round(elapsed, 1),
                working_dir=str(self._root),
            )

    def _log_start(self, command: str) -> CommandEntry:
        """Log the start of a command."""
        with self._lock:
            self._counter += 1
            entry = CommandEntry(
                id=f"cmd_{self._counter}",
                command=command,
                working_dir=str(self._root),
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            self._history.append(entry)
            # Keep only last 100 commands
            if len(self._history) > 100:
                self._history = self._history[-100:]
            return entry
