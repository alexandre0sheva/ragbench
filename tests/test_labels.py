"""The issue labels are one list (`.github/labels.yml`): the sync script reads it, and the labels the templates and CONTRIBUTING.md name exist in it."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sync_labels", ROOT / "scripts" / "sync_labels.py")
sync_labels = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules["sync_labels"] = sync_labels
spec.loader.exec_module(sync_labels)  # type: ignore[union-attr]


def _names() -> set[str]:
    return {command[3] for command in sync_labels.commands()}


def test_every_label_becomes_a_create_command_with_a_colour_and_a_description():
    commands = sync_labels.commands()
    assert len(commands) == len(_names()), "label names must be unique"
    for command in commands:
        assert command[:3] == ["gh", "label", "create"]
        assert re.fullmatch(r"[0-9a-f]{6}", command[command.index("--color") + 1])
        assert command[command.index("--description") + 1]


def test_the_issue_templates_and_contributing_only_name_labels_that_exist():
    names = _names()
    for template in (ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.md"):
        label = re.search(r"^labels: (.+)$", template.read_text(encoding="utf-8"), re.M)
        assert label and label.group(1).strip() in names, template.name
    section = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8").split("## Issue labels")[1].split("## Commit messages")[0]
    table = [line for line in section.splitlines() if line.startswith("| ") and not line.startswith(("| Label", "| ---"))]
    named = {name for line in table for name in re.findall(r"`([^`]+)`", line.split("|")[1])}
    assert named == names
