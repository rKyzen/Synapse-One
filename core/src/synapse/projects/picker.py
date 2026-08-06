"""Native OS folder picker for project workspaces.

The web UI runs in a browser and cannot open a native directory dialog, so
the local server asks the OS on the user's behalf (Synapse is a local-first
app: server and user share the machine).

Platforms:
- Windows — ``FolderBrowserDialog`` via PowerShell.
- Linux — ``zenity``/``kdialog`` when available.
- macOS — ``osascript``.
- Fallback — ``None`` so callers can prompt for a manual path.

The user picks the FOLDER that will become the project workspace — the UI
derives the project name from it (``create(name, parent_dir=<folder>.parent)``)
so the only question asked is *where* to create the project.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from synapse.logging import get_logger

log = get_logger("synapse.projects.picker")


def pick_folder(initial_dir: str | None = None) -> str | None:
    """Open a native directory picker; returns the chosen absolute path or
    ``None`` when cancelled/unsupported. Never raises."""
    try:
        if sys.platform == "win32":
            path = _pick_windows(initial_dir)
        elif shutil.which("zenity"):
            path = _pick_zenity(initial_dir)
        elif shutil.which("kdialog"):
            path = _pick_kdialog(initial_dir)
        elif sys.platform == "darwin":
            path = _pick_macos(initial_dir)
        else:
            path = None
    except Exception:  # noqa: BLE001 - a picker failure must never crash the server
        log.warning("folder_picker_failed")
        return None
    if not path:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        return None
    log.info("folder_picked", path=str(resolved))
    return str(resolved)


def _pick_windows(initial_dir: str | None) -> str | None:
    """System.Windows.Forms.FolderBrowserDialog driven through PowerShell."""
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
        "$d.Description = 'Choose the folder for the new project'; "
    )
    if initial_dir:
        script += f"$d.SelectedPath = '{initial_dir}'; "
    script += (
        "$r = $d.ShowDialog(); "
        "if ($r -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $d.SelectedPath }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        log.debug("folder_picker_ps_failed", stderr=(result.stderr or "")[:200])
        return None
    out = (result.stdout or "").strip()
    return out.splitlines()[0] if out else None


def _pick_zenity(initial_dir: str | None) -> str | None:
    cmd = ["zenity", "--file-selection", "--directory", "--title=Choose the folder for the new project"]
    if initial_dir:
        cmd += ["--filename", str(Path(initial_dir) / "")]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    out = (result.stdout or "").strip()
    return out.splitlines()[0] if out else None


def _pick_kdialog(initial_dir: str | None) -> str | None:
    cmd = ["kdialog", "--getexistingdirectory", initial_dir or str(Path.home())]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    out = (result.stdout or "").strip()
    return out.splitlines()[0] if out else None


def _pick_macos(initial_dir: str | None) -> str | None:
    script = 'POSIX path of (choose folder with prompt "Choose the folder for the new project")'
    if initial_dir:
        script = f'set f to POSIX file "{initial_dir}"\n' + script
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = (result.stdout or "").strip()
    return out.splitlines()[0] if out else None
