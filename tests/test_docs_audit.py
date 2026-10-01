"""The documentation has no broken links and no duplicated knowledge, and the audit that says so can actually find what it looks for."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_docs", ROOT / "scripts" / "audit_docs.py")
audit = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules["audit_docs"] = audit  # a dataclass in the script looks its module up
spec.loader.exec_module(audit)  # type: ignore[union-attr]


def test_the_documentation_passes_its_own_audit():
    findings = audit.audit()
    assert not findings, "\n".join(str(f) for f in findings)


def test_every_document_is_audited():
    names = {path.relative_to(ROOT).as_posix() for path in audit.documents()}
    assert {"README.md", "CONTRIBUTING.md", "docs/methodology.md", "docs/choosing-an-architecture.md", "docs/extending.md"} <= names
    assert "docs/github-setup.md" not in names  # merged into CONTRIBUTING.md


def _doc(tmp_path: Path, text: str, name: str = "doc.md") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_the_audit_flags_a_broken_link_and_a_missing_anchor(tmp_path):
    other = _doc(tmp_path, "# Real heading\n\n## Providers & model refs\n", "other.md")
    path = _doc(tmp_path, "[gone](missing.md) [ok](other.md#real-heading) [bad](other.md#nope) [slug](other.md#providers--model-refs) [web](https://x.y) [self](#top)\n\n# Top\n")
    found = audit.check_links(path)
    assert sorted(f.rule for f in found) == ["anchor", "link"] and other.exists()
    assert not [f for f in found if "providers" in f.text or "real-heading" in f.text or "#top" in f.text]


def test_links_inside_code_blocks_are_ignored(tmp_path):
    assert audit.check_links(_doc(tmp_path, "```\n[x](missing.md)\n```\n")) == []


def test_the_audit_flags_an_option_table_outside_its_owners(tmp_path):
    table = "| Option | Type | Default |\n| --- | --- | --- |\n| `a` | int | 1 |\n"
    assert [f.rule for f in audit.check_option_tables(_doc(tmp_path, table))] == ["option-table"]
    assert audit.check_option_tables(_doc(tmp_path, "| Field | Meaning |\n| --- | --- |\n")) == []


def test_the_audit_flags_a_system_described_outside_systems_md(tmp_path):
    names = {"bm25", "vector"}
    assert [f.rule for f in audit.check_system_rows(_doc(tmp_path, "| `bm25` | Lexical search | cheap |\n"), names)] == ["system-row"]
    assert audit.check_system_rows(_doc(tmp_path, "| If you have a small corpus | `bm25` | because |\n"), names) == []


def test_the_audit_flags_a_metric_definition_but_not_a_mention(tmp_path):
    assert [f.rule for f in audit.check_metric_definitions(_doc(tmp_path, "MRR is the mean reciprocal rank of the first relevant document.\n"))] == ["metric-definition"]
    assert [f.rule for f in audit.check_metric_definitions(_doc(tmp_path, "Faithfulness measures whether the answer sticks to the context.\n"))] == ["metric-definition"]
    assert audit.check_metric_definitions(_doc(tmp_path, "Look at recall and MRR in the leaderboard.\n")) == []
    assert audit.check_metric_definitions(_doc(tmp_path, "```\nMRR is the mean reciprocal rank\n```\n")) == []


@pytest.mark.parametrize("text,flagged", [("We compare all eight systems.", True), ("Twelve tools ship with it.", True), ("The two systems are compared.", False), ("Chunkers: see below.", False)])
def test_the_audit_flags_stale_counts_only_for_larger_numbers(tmp_path, text, flagged):
    assert bool(audit.check_counts(_doc(tmp_path, text + "\n"))) is flagged


def test_the_audit_flags_models_that_are_no_longer_in_the_price_table(tmp_path):
    assert [f.rule for f in audit.check_removed_models(_doc(tmp_path, "Use gpt-5.4-nano here.\n"))] == ["removed-model"]
    assert audit.check_removed_models(_doc(tmp_path, "Use gpt-6-luna here.\n")) == []


def test_every_markdown_file_the_readme_links_to_is_in_the_docs_index():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for path in (ROOT / "docs").glob("*.md"):
        if path.name not in {"release-checklist.md"}:
            assert f"docs/{path.name}" in readme, f"README's documentation index does not list docs/{path.name}"
