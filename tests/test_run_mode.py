from __future__ import annotations

import json

import pytest
import yaml

from ragbench.evaluation.evaluator import run_benchmark
from ragbench.models import embeddings, llms
from ragbench.models.errors import ModelInitError
from ragbench.models.providers import openai as openai_provider
from ragbench.reporting.html_report import write_html_report
from ragbench.reporting.markdown_report import write_leaderboard


def _config(tmp_path, extra: dict | None = None):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc_001.md").write_text("# Product\n\nHarborShield AI handles marine cargo submissions.\n", encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        '{"id":"q1","question":"What handles marine cargo submissions?","reference_answer":"HarborShield AI.","expected_keywords":["HarborShield AI"],"relevant_doc_ids":["doc_001"]}\n',
        encoding="utf-8",
    )
    config = {
        "run": {"name": "mode_run", "output_dir": str(tmp_path / "results")},
        "dataset": {"documents_path": str(docs), "questions_path": str(questions)},
        "systems": [{"type": "bm25", "name": "bm25_mock", "retrieval": {"top_k": 2}}],
        "evaluation": {"k_values": [1, 3, 5], "max_questions": 1},
        **(extra or {}),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_run_summary_records_mode_models_and_pricing(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    out = run_benchmark(_config(tmp_path, {"pricing": {"my-model": {"input": 1.0, "output": 2.0}}}), force_mock=True)

    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "mock"
    assert "mock-llm" in summary["models_used"]
    assert summary["unknown_priced_models"] == []
    assert summary["pricing_as_of"]
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "Mock run" in html
    assert "Mock run" in (out / "leaderboard.md").read_text(encoding="utf-8")


def test_pricing_overrides_do_not_leak_between_runs(tmp_path, monkeypatch):
    from ragbench.models import cost

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    run_benchmark(_config(tmp_path, {"pricing": {"gpt-6-luna": {"input": 99.0, "output": 0.0}}}), force_mock=True)

    assert cost.estimate_model_cost("gpt-6-luna", 1_000_000, 0) == 0.10


def test_reports_warn_about_unknown_priced_models(tmp_path):
    rows = [{"system": "a", "system_type": "bm25", "answer_score": 4.0}]
    notices = ["Cost under-reported: no price registered for model-x"]

    write_leaderboard(tmp_path / "leaderboard.md", rows, notices=notices)
    write_html_report(
        tmp_path / "report.html",
        run_id="r",
        summary_rows=rows,
        category_rows=[],
        cost_rows=[],
        failure_rows=[],
        config_text="",
        run_meta={"num_systems": 1, "num_questions": 1, "run_wall_time_ms": 1.0, "mode": "live", "unknown_priced_models": ["model-x"]},
    )

    assert "model-x" in (tmp_path / "leaderboard.md").read_text(encoding="utf-8")
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "model-x" in html and "Mock run" not in html


def test_live_key_with_failing_client_raises_instead_of_silently_mocking(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def boom(*_args, **_kwargs):
        raise RuntimeError("client init failed")

    monkeypatch.setattr(openai_provider, "OpenAILLM", boom)
    monkeypatch.setattr(openai_provider, "OpenAIEmbeddingModel", boom)

    with pytest.raises(ModelInitError):
        llms.create_llm("gpt-6-luna")
    with pytest.raises(ModelInitError):
        embeddings.create_embedding_model("text-embedding-3-small")

    # Explicit opt-outs keep working.
    assert llms.create_llm("gpt-6-luna", force_mock=True).model_name == "mock-llm"
    assert llms.create_llm("gpt-6-luna", strict=False).model_name == "mock-llm"
    assert embeddings.create_embedding_model("x", strict=False).model_name == "hashing-embedding"


def test_no_key_means_mock_without_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")  # empty counts as set, so the repo's real .env key is never loaded

    assert llms.create_llm("gpt-6-luna").model_name == "mock-llm"
    assert embeddings.create_embedding_model("text-embedding-3-small").model_name == "hashing-embedding"
