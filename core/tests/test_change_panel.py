"""Tests for the change panel system."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synapse.workspace.change_panel import ChangePanel, ChangeType


class TestChangePanel:
    """Tests for ChangePanel."""

    def test_panel_creation(self):
        panel = ChangePanel(session_id="test")
        assert panel._session_id == "test"

    def test_record_created(self):
        panel = ChangePanel()
        panel.record_created("new.py", size=100)

        summary = panel.get_summary()
        assert len(summary.created) == 1
        assert summary.created[0].path == "new.py"

    def test_record_modified(self):
        panel = ChangePanel()
        panel.record_modified("existing.py", size_before=50, size_after=100)

        summary = panel.get_summary()
        assert len(summary.modified) == 1
        assert summary.modified[0].size_after == 100

    def test_record_deleted(self):
        panel = ChangePanel()
        panel.record_deleted("old.py")

        summary = panel.get_summary()
        assert len(summary.deleted) == 1

    def test_record_renamed(self):
        panel = ChangePanel()
        panel.record_renamed("old_name.py", "new_name.py")

        summary = panel.get_summary()
        assert len(summary.renamed) == 1
        assert summary.renamed[0].old_path == "old_name.py"

    def test_get_recent(self):
        panel = ChangePanel()
        for i in range(5):
            panel.record_created(f"file{i}.py")

        recent = panel.get_recent(limit=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0].path == "file4.py"

    def test_clear(self):
        panel = ChangePanel()
        panel.record_created("test.py")
        assert len(panel.get_summary().changes) == 1

        panel.clear()
        assert len(panel.get_summary().changes) == 0

    def test_to_dict(self):
        panel = ChangePanel()
        panel.record_created("a.py")
        panel.record_modified("b.py")

        d = panel.to_dict()
        assert d["total"] == 2
        assert d["created"] == 1
        assert d["modified"] == 1

    def test_to_markdown(self):
        panel = ChangePanel()
        panel.record_created("new.py")
        panel.record_deleted("old.py")

        summary = panel.get_summary()
        md = summary.to_markdown()
        assert "Created" in md
        assert "Deleted" in md
        assert "new.py" in md


class TestChangeSummary:
    """Tests for ChangeSummary."""

    def test_summary_properties(self):
        from synapse.workspace.change_panel import ChangeSummary, FileChange

        changes = [
            FileChange(path="a.py", change_type=ChangeType.CREATED, timestamp="t1"),
            FileChange(path="b.py", change_type=ChangeType.MODIFIED, timestamp="t2"),
        ]
        summary = ChangeSummary(changes=changes)
        assert len(summary.created) == 1
        assert len(summary.modified) == 1
        assert len(summary.deleted) == 0
