from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from ragbench.config.loader import load_config
from ragbench.config.schema import ChunkerConfig, ExperimentConfig, SystemConfig
from ragbench.documents.chunkers import FixedCharacterChunker, MarkdownAwareChunker, TokenChunker, create_chunker
from ragbench.rag_systems import SYSTEMS, create_rag_system
from ragbench.rag_systems.options import HybridOptions, ParentDocOptions


def _experiment(systems: list[dict]) -> dict:
    return {"run": {"name": "t"}, "dataset": {"documents_path": "d", "questions_path": "q"}, "systems": systems}


def _message(excinfo) -> str:
    return " ".join(str(excinfo.value).split())


def test_typo_in_retrieval_option_fails_with_a_suggestion():
    with pytest.raises(ValidationError) as excinfo:
        SystemConfig(type="rerank", name="r", retrieval={"candidate_top": 10})

    message = _message(excinfo)
    assert "System 'r' (type rerank)" in message
    assert "unknown option 'retrieval.candidate_top'" in message
    assert "Did you mean 'candidate_top_k'?" in message
    assert "valid `retrieval` options:" in message and "reranker" in message


def test_unknown_option_without_a_close_match_just_lists_valid_options():
    with pytest.raises(ValidationError) as excinfo:
        SystemConfig(type="bm25", retrieval={"zzz": 1})

    message = _message(excinfo)
    assert "unknown option 'retrieval.zzz'" in message and "Did you mean" not in message and "top_k" in message


def test_wrong_value_types_and_ranges_are_reported_with_their_path():
    with pytest.raises(ValidationError, match=r"retrieval\.rrf_k"):
        SystemConfig(type="hybrid", retrieval={"rrf_k": 0})
    with pytest.raises(ValidationError, match=r"retrieval\.vector_store"):
        SystemConfig(type="vector", retrieval={"vector_store": "chrome"})
    with pytest.raises(ValidationError, match=r"retrieval\.parent_score_aggregation"):
        SystemConfig(type="parent_doc", retrieval={"parent_score_aggregation": "median"})


def test_reranker_name_is_checked_against_the_registry():
    with pytest.raises(ValidationError) as excinfo:
        SystemConfig(type="rerank", retrieval={"reranker": "local_relevence"})

    message = _message(excinfo)
    assert "retrieval.reranker" in message and "Did you mean 'local_relevance'" in message
    SystemConfig(type="rerank", retrieval={"reranker": "tfidf"})  # aliases are accepted


def test_chunker_section_is_typed_too():
    with pytest.raises(ValidationError, match=r"chunker\.chunk_sizes"):
        SystemConfig(type="vector", chunker={"chunk_sizes": 10})
    with pytest.raises(ValidationError, match="Did you mean 'markdown'"):
        SystemConfig(type="vector", chunker={"type": "markdwn"})
    with pytest.raises(ValidationError, match="chunk_overlap must be smaller"):
        SystemConfig(type="vector", chunker={"chunk_size": 50, "chunk_overlap": 50})
    # parent_doc has its own chunker shape and does not accept the flat one.
    SystemConfig(type="parent_doc", chunker={"parent_chunk_size": 500, "child_chunk_size": 100})
    with pytest.raises(ValidationError, match=r"chunker\.chunk_size"):
        SystemConfig(type="parent_doc", chunker={"chunk_size": 500})


def test_llm_features_only_where_supported():
    SystemConfig(type="llm_heavy", llm_features={"enable_llm_rerank": True})
    with pytest.raises(ValidationError, match=r"llm_features\.enable_llm_rerank_x"):
        SystemConfig(type="llm_heavy", llm_features={"enable_llm_rerank_x": True})
    with pytest.raises(ValidationError, match="does not use `llm_features`"):
        SystemConfig(type="bm25", llm_features={"enable_query_rewrite": True})


def test_unknown_top_level_system_keys_are_rejected():
    with pytest.raises(ValidationError, match="retreival"):
        SystemConfig.model_validate({"type": "bm25", "retreival": {"top_k": 3}})


def test_unregistered_or_spec_less_types_are_not_validated_at_system_level():
    SystemConfig(type="some_custom_plugin", retrieval={"anything": 1}, chunker={"whatever": 2})


def test_experiment_config_rejects_unknown_types_and_duplicate_names_up_front():
    with pytest.raises(ValidationError) as excinfo:
        ExperimentConfig.model_validate(_experiment([{"type": "bm25"}, {"type": "hybrd"}]))
    assert "systems[1]" in _message(excinfo) and "Did you mean 'hybrid'" in _message(excinfo)
    with pytest.raises(ValidationError, match="duplicate system name 'bm25'"):
        ExperimentConfig.model_validate(_experiment([{"type": "bm25"}, {"type": "bm25"}]))
    ExperimentConfig.model_validate(_experiment([{"type": "bm25", "name": "a"}, {"type": "bm25", "name": "b"}]))


def test_config_loader_reports_the_offending_file_content(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(_experiment([{"type": "hybrid", "name": "h", "retrieval": {"bm25_topk": 5}}])), encoding="utf-8")

    with pytest.raises(ValidationError, match="Did you mean 'bm25_top_k'"):
        load_config(path)


def test_every_shipped_config_still_validates():
    from pathlib import Path

    for path in sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml")):
        assert load_config(path).systems, path.name


def test_option_defaults_and_helpers():
    hybrid = HybridOptions()
    assert (hybrid.bm25_top_k, hybrid.vector_top_k, hybrid.rrf_k, hybrid.multi_query) == (20, 20, 60, False)
    assert hybrid.configured_top_k is None and hybrid.resolve_top_k(None) == 5 and hybrid.resolve_top_k(10) == 10
    assert HybridOptions(final_top_k=3, top_k=7).configured_top_k == 3  # final_top_k wins
    assert HybridOptions(top_k=7).configured_top_k == 7
    assert ParentDocOptions(parent_score_aggregation="SUM").parent_score_aggregation == "sum"
    assert ParentDocOptions().configured_top_k == 4


def test_chunker_defaults_come_from_the_chunker_class():
    assert isinstance(create_chunker(None), TokenChunker)
    token = create_chunker({"type": "token"})
    assert (token.chunk_size, token.chunk_overlap) == (500, 80)
    fixed = create_chunker(ChunkerConfig(type="fixed"))
    assert isinstance(fixed, FixedCharacterChunker) and (fixed.chunk_size, fixed.chunk_overlap) == (1200, 150)
    markdown = create_chunker({"type": "md", "chunk_size": 100, "chunk_overlap": 0})
    assert isinstance(markdown, MarkdownAwareChunker) and (markdown.chunk_size, markdown.chunk_overlap) == (100, 0)  # 0 overlap is honored


def test_every_builtin_system_declares_a_complete_spec():
    assert len(SYSTEMS.names()) >= 8
    for name in SYSTEMS.names():
        spec = SYSTEMS.get(name).spec
        assert spec is not None, f"{name} has no SystemSpec"
        assert spec.type == name and spec.title and spec.summary and spec.best_for
        assert spec.cost_profile in {"free", "low", "medium", "high"} and spec.latency_profile in {"fast", "medium", "slow"}


def test_systems_read_defaults_from_their_option_models_and_no_system_builds_its_own_components():
    import inspect

    import ragbench.rag_systems as systems_pkg

    for name in SYSTEMS.names():
        cls = SYSTEMS.get(name)
        source = inspect.getsource(inspect.getmodule(cls))  # type: ignore[arg-type]
        assert "create_embedding_model(" not in source and "VectorStore(" not in source and "create_chunker(" not in source, name
        assert ".retrieval.get(" not in source and "cfg.get(" not in source, f"{name} bypasses its typed options"
    assert systems_pkg.create_rag_system(SystemConfig(type="bm25"), force_mock=True).name == "bm25"


def test_create_rag_system_unknown_type_is_a_value_error_with_suggestion():
    with pytest.raises(ValueError, match="Did you mean 'hybrid'"):
        create_rag_system(SystemConfig(type="hybrd"), force_mock=True)


def test_cli_shows_a_short_actionable_message_for_a_bad_config(tmp_path):
    from typer.testing import CliRunner

    from ragbench.cli import app

    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(_experiment([{"type": "rerank", "name": "r", "retrieval": {"candidate_top": 10}}])), encoding="utf-8")

    result = CliRunner().invoke(app, ["run", "--config", str(path), "--mock"])

    output = " ".join(result.output.split())
    assert result.exit_code == 2
    assert "Did you mean 'candidate_top_k'?" in output and "Traceback" not in result.output
    missing = CliRunner().invoke(app, ["run", "--config", str(tmp_path / "missing.yaml"), "--mock"])
    assert missing.exit_code == 2 and "Config file not found" in missing.output
