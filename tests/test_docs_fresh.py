from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_docs", ROOT / "scripts" / "generate_docs.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_generated_docs_are_up_to_date():
    stale = [str(path.relative_to(ROOT)) for path in _load_generator().stale_files()]
    assert not stale, f"Run `python scripts/generate_docs.py` to refresh: {stale}"


def test_systems_doc_covers_every_builtin_system_and_its_options():
    from ragbench.rag_systems import all_specs

    text = (ROOT / "docs" / "systems.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for spec in all_specs():
        assert f"## `{spec.type}`" in text and f"| `{spec.type}` |" in readme
        for option in spec.options.model_fields:
            assert f"| `{option}` |" in text, f"{spec.type}.{option} missing from docs/systems.md"


def test_readme_table_markers_are_present_exactly_once():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.count("<!-- systems:start -->") == 1 and readme.count("<!-- systems:end -->") == 1


def test_leaderboard_best_for_comes_from_the_spec():
    from ragbench.reporting.markdown_report import _best_for

    assert _best_for("bm25") == "Cheap baseline and exact-term matching"
    assert _best_for("not_registered") == "Custom comparison"
