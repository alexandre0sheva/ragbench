import yaml

from ragbench.evaluation.evaluator import run_benchmark


def test_mock_mode_can_run_without_openai_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc_001.md").write_text("# Product\n\nHarborShield AI handles marine cargo submissions.\n", encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        '{"id":"q1","question":"What handles marine cargo submissions?","reference_answer":"HarborShield AI handles marine cargo submissions.","expected_keywords":["HarborShield AI"],"relevant_doc_ids":["doc_001"],"category":"direct_fact","difficulty":"easy","answer_type":"single_fact"}\n',
        encoding="utf-8",
    )
    config = {
        "run": {"name": "test_run", "output_dir": str(tmp_path / "results")},
        "dataset": {"documents_path": str(docs), "questions_path": str(questions)},
        "systems": [
            {
                "type": "vector",
                "name": "vector_mock",
                "chunker": {"type": "token", "chunk_size": 50, "chunk_overlap": 0},
                "retrieval": {"top_k": 2},
                "models": {"embedding": "text-embedding-3-small", "generator": "gpt-5.4-nano"},
            }
        ],
        "evaluation": {"k_values": [1, 3, 5], "judge_enabled": True, "judge_model": "gpt-5.4-nano", "max_questions": 1},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    output_dir = run_benchmark(config_path, force_mock=True)

    assert (output_dir / "leaderboard.md").exists()
    assert (output_dir / "report.html").exists()
    assert (output_dir / "per_question_results.jsonl").exists()
