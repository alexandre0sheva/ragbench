"""`ragbench harness show` and `ragbench harness diff`."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from cli_support import invoke

from ragbench.harness.prompts import ANSWER_PROMPTS


def _config(root: Path, name: str, systems: list[dict]) -> Path:
    path = root / f"{name}.yaml"
    config = {"run": {"name": name}, "dataset": {"documents_path": "docs", "questions_path": "questions.jsonl"}, "systems": systems}
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_show_prints_a_system_types_default_harness_with_its_knobs():
    result = invoke("harness", "show", "hybrid_rerank")
    assert result.exit_code == 0, result.output
    shown = yaml.safe_load(result.output)
    assert {key: shown[key] for key in ANSWER_PROMPTS} == ANSWER_PROMPTS
    assert (shown["top_k"], shown["reranker"], shown["vector_weight"], shown["chunk_size"]) == (None, "local_relevance", 1.0, 500)
    assert "top_k: null  # knob, 1 to 50, unset: the run's evaluation.context_k applies" in result.output
    assert "reranker: local_relevance  # knob\n" in result.output
    assert result.output.startswith("# hybrid_rerank (type hybrid_rerank), fingerprint ")


def test_show_marks_what_a_configs_patch_changed(tmp_path):
    config = _config(tmp_path, "a", [{"type": "bm25", "name": "lexical", "harness": {"citation_style": "Cite as (doc_id).", "top_k": 8}}])
    result = invoke("harness", "show", "lexical", "--config", str(config))
    assert result.exit_code == 0, result.output
    assert "citation_style: Cite as (doc_id).  # patched" in result.output
    assert "top_k: 8  # knob, 1 to 50, patched" in result.output
    as_json = json.loads(invoke("harness", "show", "lexical", "--config", str(config), "--json").output)
    assert as_json["patched"] == ["citation_style", "top_k"] and as_json["harness"]["top_k"] == 8 and len(as_json["fingerprint"]) == 16


def test_show_suggests_a_close_name():
    result = invoke("harness", "show", "hybird")
    assert result.exit_code == 2 and "Did you mean 'hybrid'" in result.output


def test_diff_lists_field_changes_and_systems_on_one_side(tmp_path):
    base = [{"type": "bm25", "name": "lexical"}, {"type": "vector", "name": "dense"}]
    a = _config(tmp_path, "a", base)
    b = _config(tmp_path, "b", [{"type": "bm25", "name": "lexical", "harness": {"answer_instructions": "Answer in one sentence.", "top_k": 3}}, {"type": "hyde", "name": "probe"}])
    result = invoke("harness", "diff", str(a), str(b))
    assert result.exit_code == 0, result.output
    assert "lexical (type bm25)\n  answer_instructions:\n    - Be concise but complete.\n    + Answer in one sentence.\n  top_k: unset -> 3\n" in result.output
    assert "dense: only in a.yaml" in result.output and "probe: only in b.yaml" in result.output
    changes = json.loads(invoke("harness", "diff", str(a), str(b), "--json").output)
    assert changes[0] == {"system": "lexical", "changes": {"answer_instructions": ["Be concise but complete.", "Answer in one sentence."], "top_k": [None, 3]}}


def test_diff_of_identical_harnesses_says_so(tmp_path):
    a = _config(tmp_path, "a", [{"type": "bm25", "name": "lexical"}])
    result = invoke("harness", "diff", str(a), str(a))
    assert result.exit_code == 0 and "No harness differences." in result.output


def test_show_explains_systems_without_a_default_harness(tmp_path):
    required = invoke("harness", "show", "adaptive")
    assert required.exit_code == 2 and "has required options" in required.output and "--config" in required.output
    config = _config(tmp_path, "a", [{"type": "adaptive", "name": "router", "retrieval": {"routes": {"default": {"type": "bm25"}}}}])
    routed = invoke("harness", "show", "router", "--config", str(config))
    assert routed.exit_code == 2 and "takes no `harness:`" in routed.output
