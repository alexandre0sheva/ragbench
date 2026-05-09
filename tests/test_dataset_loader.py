import json

from ragbench.datasets.loader import load_dataset


def test_dataset_loader_loads_questions_and_derives_qrels(tmp_path):
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "Who?",
                "reference_answer": "A.",
                "expected_keywords": ["A"],
                "relevant_doc_ids": ["doc_001"],
                "category": "direct_fact",
                "difficulty": "easy",
                "answer_type": "single_fact",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_dataset(questions)

    assert len(dataset.questions) == 1
    assert dataset.questions[0].id == "q1"
    assert dataset.qrels["q1"]["doc_001"] == 1

