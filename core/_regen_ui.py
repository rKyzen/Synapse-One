"""Regenerate web_ui.py with debug logging and connection check."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("_gen_ui", "src/synapse/api/_gen_ui.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

page = mod.PAGE

header = '''"""Web UI - utilitarian monochrome (Phase 5).

Served by FastAPI at GET /ui. Black/white/transparency, Codex-style.
"""

from __future__ import annotations

PAGE = %r


def render_page() -> str:
    return PAGE
''' % page

Path("src/synapse/api/web_ui.py").write_text(header, encoding="utf-8")
print(f"Written: {len(header)} chars")
