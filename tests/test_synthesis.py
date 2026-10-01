"""`generate-questions`: the category mix, validation of unanswerable questions, dedupe, determinism, mock mode and the CLI."""

from __future__ import annotations

import itertools
import json
import threading
from pathlib import Path

import pytest
import yaml
from scripted_llm import ScriptedLLM
from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.synthesis import (
    CATEGORIES,
    DEFAULT_MIX,
    MOCK_GENERATOR,
    allocate,
    estimate_generation_cost,
    generate_questions,
    parse_mix,
    stratified_order,
)
from ragbench.datasets.templates import NOT_IN_DOCUMENTS
from ragbench.datasets.validation import validate_dataset
from ragbench.documents.schema import Document
from ragbench.evaluation.budget import BudgetGuard
from ragbench.models import cost as pricing
from ragbench.models.llms import MockLLM
from ragbench.models.prompts import PARAPHRASE_QUESTION_MARKER, WRITE_MULTIHOP_MARKER, WRITE_QUESTION_MARKER, WRITE_UNANSWERABLE_MARKER
from ragbench.utils.jsonl import write_jsonl

THEMES = ("harbor", "claims", "billing")


def _corpus(count: int = 12) -> list[Document]:
    """Documents in three themes (so BM25 finds related pairs), of growing length, most with a number in them."""
    documents = []
    for number in range(count):
        theme = THEMES[number % 3]
        sentences = [f"Product{number} handles {theme} workflows for insurers and brokers in every region."]
        sentences += [
            f"Product{number} feature{part} offers capability{part} for {theme} teams at ${100 + number + part} per month with quota{number}{part} limits."
            for part in range(1 + number)
        ]
        text = f"# Product{number} {theme} guide\n\n" + " ".join(sentences)
        documents.append(Document(doc_id=f"doc_{number + 1:03d}", path=f"doc_{number + 1:03d}.md", title=f"Product{number} {theme} guide", text=text))
    return documents


def _scripted(entities: list[str] | None = None, *, same_question: str | None = None, price: float = 0.0, junk_first: int = 0) -> ScriptedLLM:
    """Replies that are valid for every prompt kind and unique per call (so nothing is deduplicated), unless told otherwise."""
    numbers, lock, junk = itertools.count(1), threading.Lock(), [junk_first]
    queued = list(entities or [])

    def respond(prompt: str) -> str:
        with lock:
            n = next(numbers)
            if junk[0] > 0:
                junk[0] -= 1
                return "this is not json"
            entity = queued.pop(0) if queued else f"Imaginary{n}"
        question = same_question or f"Which detail number{n} describes feature{n} of product{n}?"
        if WRITE_MULTIHOP_MARKER in prompt:
            return json.dumps({"question": question, "reference_answer": f"combined answer {n}", "expected_keywords": [f"k{n}"]})
        if PARAPHRASE_QUESTION_MARKER in prompt:
            return json.dumps({"question": f"Could you explain what unique{n} wording means for matter{n}?"})
        if WRITE_UNANSWERABLE_MARKER in prompt:
            return json.dumps({"question": f"What is the price of {entity} widget{n}?", "entity": entity})
        assert WRITE_QUESTION_MARKER in prompt
        return json.dumps({"question": question, "reference_answer": f"answer {n}", "expected_keywords": [f"k{n}"]})

    return ScriptedLLM(respond, price_per_call=price)


@pytest.fixture(autouse=True)
def _clean_pricing():
    yield
    pricing.clear_pricing_overrides()
    pricing.reset_unknown_priced_models()


# -- mix and allocation ----------------------------------------------------------------------------


def test_parse_mix_normalizes_shares_and_rejects_nonsense():
    assert parse_mix(None) == DEFAULT_MIX
    mix = parse_mix("single_hop=2,numeric=2")
    assert mix == {"single_hop": 0.5, "numeric": 0.5}
    with pytest.raises(ValueError, match="Available"):
        parse_mix("hardest=1")
    with pytest.raises(ValueError, match="category=share"):
        parse_mix("single_hop")
    with pytest.raises(ValueError, match="more than 0"):
        parse_mix("single_hop=0")


@pytest.mark.parametrize("n", [1, 7, 20, 33, 100])
def test_allocation_sums_to_n_and_stays_within_one_of_each_share(n):
    counts = allocate(n, DEFAULT_MIX)
    assert sum(counts.values()) == n and set(counts) == set(DEFAULT_MIX)
    for category, share in DEFAULT_MIX.items():
        assert abs(counts[category] - n * share) < 1


def test_the_category_mix_is_honoured_within_one_question():
    n = 40
    result = generate_questions(_corpus(), n=n, llm=_scripted(), seed=3)
    assert len(result.rows) == n and result.warnings == []
    seen = {category: sum(row["category"] == category for row in result.rows) for category in CATEGORIES}
    for category, share in DEFAULT_MIX.items():
        assert abs(seen[category] - n * share) <= 1, (category, seen)
    assert seen == result.made == result.requested


# -- validation ------------------------------------------------------------------------------------


def test_an_unanswerable_candidate_whose_entity_occurs_in_the_corpus_is_rejected():
    corpus = _corpus()
    # "Product3" and "billing" are in the corpus; the model's first three proposals name things that are there.
    result = generate_questions(corpus, n=20, mix={"unanswerable": 1.0}, llm=_scripted(["Product3", "BILLING", "harbor"]), seed=0)
    assert result.rejected["entity_in_corpus"] == 3
    assert len(result.rows) == 20 and all(row["category"] == "unanswerable" for row in result.rows)
    text = " ".join(document.text.lower() for document in corpus)
    assert all(row["metadata"]["entity"].lower() not in text for row in result.rows)
    first = result.rows[0]
    assert first["answerable"] is False and first["relevant_doc_ids"] == [] and first["reference_answer"] == NOT_IN_DOCUMENTS and first["answer_type"] == "unanswerable"


def test_an_unanswerable_candidate_without_an_entity_is_unusable():
    llm = ScriptedLLM(lambda prompt: json.dumps({"question": "What is the price of the thing nobody mentions?"}))
    result = generate_questions(_corpus(), n=4, mix={"unanswerable": 1.0}, llm=llm)
    assert result.rows == [] and result.rejected["unusable"] > 0 and any("unanswerable" in warning for warning in result.warnings)


def test_duplicate_questions_are_rejected_and_the_shortfall_is_reported():
    result = generate_questions(_corpus(), n=5, mix={"single_hop": 1.0}, llm=_scripted(same_question="Which feature does the product offer to every customer?"))
    assert len(result.rows) == 1 and result.rejected["duplicate"] >= 4
    assert any("Only 1 of 5 single_hop" in warning for warning in result.warnings)


def test_unusable_replies_are_counted_and_do_not_stop_the_run():
    llm = _scripted(junk_first=3)
    result = generate_questions(_corpus(), n=6, mix={"single_hop": 1.0}, llm=llm)
    assert len(result.rows) == 6 and result.rejected["unusable"] == 3


def test_a_category_that_cannot_be_written_is_skipped_with_a_warning_not_padded_with_another():
    documents = [Document(doc_id="doc_001", path="a", title="A", text="Words only, with no digits anywhere in this short document body.")]
    result = generate_questions(documents, n=4, mix={"numeric": 0.5, "single_hop": 0.5}, llm=_scripted())
    assert {row["category"] for row in result.rows} == {"single_hop"} and len(result.rows) == 2
    assert any("Only 0 of 2 numeric" in warning for warning in result.warnings) and result.rejected["no_suitable_document"] > 0


def test_numeric_questions_come_from_documents_that_contain_numbers():
    documents = [
        Document(doc_id="doc_001", path="a", title="A", text="No numerals in this text at all, only plain words and sentences."),
        Document(doc_id="doc_002", path="b", title="B", text="The plan costs $42 per month for each seat on the team."),
    ]
    result = generate_questions(documents, n=2, mix={"numeric": 1.0}, llm=_scripted())
    assert {tuple(row["relevant_doc_ids"]) for row in result.rows} == {("doc_002",)}


# -- structure, metadata, determinism --------------------------------------------------------------


def test_rows_are_valid_flagged_for_review_and_load_as_a_clean_dataset(tmp_path):
    corpus = _corpus()
    result = generate_questions(corpus, n=30, llm=_scripted(), seed=1)
    assert [row["id"] for row in result.rows] == [f"q_{number:03d}" for number in range(1, 31)]
    for row in result.rows:
        assert row["metadata"]["synthetic"] is True and row["metadata"]["needs_review"] is True and row["metadata"]["generator"] == "scripted-model"
        assert "mock" not in row["metadata"]
        if row["category"] == "multi_hop":
            assert len(set(row["relevant_doc_ids"])) == 2
        elif row["category"] != "unanswerable":
            assert len(row["relevant_doc_ids"]) == 1 and row["reference_answer"]
        if row["category"] == "paraphrase":
            assert row["metadata"]["paraphrase_of"] != row["question"]
    path = tmp_path / "questions.jsonl"
    write_jsonl(path, result.rows)
    dataset = load_dataset(path)
    assert not dataset.label_free and validate_dataset(corpus, dataset) == []
    assert sum(not q.is_answerable for q in dataset.questions) == result.made["unanswerable"]


def test_multi_hop_pairs_are_bm25_neighbours_and_not_repeated():
    result = generate_questions(_corpus(), n=6, mix={"multi_hop": 1.0}, llm=_scripted())
    pairs = [frozenset(row["relevant_doc_ids"]) for row in result.rows]
    assert len(pairs) == 6 and len(set(pairs)) == 6
    # Documents of the same theme share vocabulary, so every pair is within one theme (doc numbers 1, 4, 7... are `harbor`, and so on).
    for pair in pairs:
        assert len({(int(doc.split("_")[1]) - 1) % 3 for doc in pair}) == 1


def test_the_same_seed_gives_the_same_questions_and_another_seed_differs():
    first = generate_questions(_corpus(), n=20, llm=_scripted(), seed=7)
    again = generate_questions(_corpus(), n=20, llm=_scripted(), seed=7)
    other = generate_questions(_corpus(), n=20, llm=_scripted(), seed=8)
    assert first.rows == again.rows
    assert [row["relevant_doc_ids"] for row in first.rows] != [row["relevant_doc_ids"] for row in other.rows]


def test_documents_are_sampled_stratified_by_length():
    import random

    corpus = _corpus(9)  # lengths grow with the number: three short, three medium, three long
    order = stratified_order(corpus, random.Random(0))
    assert {document.doc_id for document in order} == {document.doc_id for document in corpus}
    ranked = sorted(corpus, key=lambda d: len(d.text))
    bands = [{d.doc_id for d in ranked[i : i + 3]} for i in (0, 3, 6)]
    for start in (0, 3, 6):  # every window of three consecutive picks takes one document from each band
        assert [sum(d.doc_id in band for d in order[start : start + 3]) for band in bands] == [1, 1, 1]


def test_a_questions_prompt_lists_those_already_asked_about_that_document_so_replies_differ():
    llm = _scripted()
    generate_questions(_corpus(3), n=6, mix={"single_hop": 1.0}, llm=llm)
    assert any("Already asked" in prompt for prompt in llm.prompts) and not llm.prompts[0].count("Already asked")


# -- cost and budget -------------------------------------------------------------------------------


def test_the_spending_cap_stops_generation_and_keeps_what_was_written():
    llm = _scripted(price=0.5)
    result = generate_questions(_corpus(), n=20, llm=llm, budget=BudgetGuard(1.0))
    assert result.stopped_by_budget and 0 < len(result.rows) < 20 and llm.calls == 2
    assert result.cost_usd == pytest.approx(1.0) and any("spending cap" in warning for warning in result.warnings)


def test_the_cost_estimate_scales_with_the_work_and_uses_the_price_table():
    pricing.register_pricing({"fake-gen": {"input": 1.0, "output": 2.0}})
    corpus = _corpus()
    small = estimate_generation_cost(corpus, 20, DEFAULT_MIX, "fake-gen")
    large = estimate_generation_cost(corpus, 200, DEFAULT_MIX, "fake-gen")
    assert 0 < small < large and large == pytest.approx(10 * small, rel=0.15)
    assert estimate_generation_cost(corpus, 20, DEFAULT_MIX, "model-with-no-price") == 0
    assert estimate_generation_cost(corpus, 20, {"paraphrase": 1.0}, "fake-gen") > estimate_generation_cost(corpus, 20, {"single_hop": 1.0}, "fake-gen")


# -- mock mode -------------------------------------------------------------------------------------


def test_mock_mode_is_template_based_deterministic_and_flagged():
    corpus = _corpus()
    result = generate_questions(corpus, n=30, seed=2)
    again = generate_questions(corpus, n=30, seed=2)
    assert result.mock and result.generator == MOCK_GENERATOR and result.calls == 0 and result.cost_usd == 0
    assert result.rows == again.rows and len(result.rows) == 30
    assert all(row["metadata"]["mock"] is True and row["metadata"]["generator"] == MOCK_GENERATOR and row["metadata"]["needs_review"] for row in result.rows)
    for category, share in DEFAULT_MIX.items():
        assert abs(sum(row["category"] == category for row in result.rows) - 30 * share) <= 1
    assert any(row["question"].startswith("What does the document say about") for row in result.rows)


def test_the_mock_model_answers_synthesis_prompts_with_the_same_templates():
    """Offline runs through `MockLLM` (and the fake OpenAI server behind it) produce exactly what mock mode does."""
    corpus = _corpus()
    templated = generate_questions(corpus, n=20, seed=5)
    through_model = generate_questions(corpus, n=20, seed=5, llm=MockLLM())
    assert [row["question"] for row in through_model.rows] == [row["question"] for row in templated.rows]
    assert through_model.calls > 0


def test_nothing_to_write_about_is_an_error():
    with pytest.raises(ValueError, match="no documents"):
        generate_questions([], n=5)
    with pytest.raises(ValueError, match="at least 1"):
        generate_questions(_corpus(), n=0)


# -- the CLI ---------------------------------------------------------------------------------------


def _docs_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for document in _corpus(6):
        (root / f"{document.doc_id}.md").write_text(document.text, encoding="utf-8")
    return root


def test_cli_mock_run_writes_a_dataset_that_inspect_dataset_flags_for_review(tmp_path):
    docs, out = _docs_dir(tmp_path / "docs"), tmp_path / "questions.jsonl"
    runner = CliRunner()
    result = runner.invoke(app, ["generate-questions", "--docs", str(docs), "--out", str(out), "--n", "15", "--mock", "--seed", "4"])
    assert result.exit_code == 0, result.output
    assert "Mock mode" in result.output and "needs_review" in result.output
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 15 and all(row["metadata"]["needs_review"] for row in rows)

    inspected = runner.invoke(app, ["inspect-dataset", "--docs", str(docs), "--questions", str(out), "--no-estimate"])
    assert inspected.exit_code == 0, inspected.output
    assert "needs_review" in inspected.output and "mock templates" in inspected.output


def test_cli_refuses_to_overwrite_questions_without_force_and_rejects_a_bad_mix(tmp_path):
    docs, out = _docs_dir(tmp_path / "docs"), tmp_path / "questions.jsonl"
    out.write_text("reviewed\n")
    runner = CliRunner()
    refused = runner.invoke(app, ["generate-questions", "--docs", str(docs), "--out", str(out), "--mock"])
    assert refused.exit_code == 2 and "--force" in refused.output and out.read_text() == "reviewed\n"
    bad = runner.invoke(app, ["generate-questions", "--docs", str(docs), "--out", str(tmp_path / "new.jsonl"), "--mix", "hardest=1", "--mock"])
    assert bad.exit_code == 2 and "Available" in bad.output
    assert runner.invoke(app, ["generate-questions", "--docs", str(docs), "--out", str(out), "--mock", "--force"]).exit_code == 0


def test_cli_live_run_asks_for_confirmation_and_refuses_when_nobody_can_answer(tmp_path, monkeypatch):
    """With a (fake) key the command goes down the live path; an estimate over the threshold must stop it before any call is made."""
    docs = _docs_dir(tmp_path / "docs")
    config = {
        "run": {"name": "x", "output_dir": str(tmp_path / "results")},
        "dataset": {"documents_path": str(docs), "questions_path": str(tmp_path / "q.jsonl")},
        "systems": [{"type": "bm25", "name": "bm25"}],
        "pricing": {"gpt-6-luna": {"input": 5_000_000.0, "output": 5_000_000.0}},  # dollars per million tokens: any estimate is far above $1
    }
    (tmp_path / "x.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    monkeypatch.setenv("CI", "1")
    result = CliRunner().invoke(app, ["generate-questions", "--docs", str(docs), "--out", str(tmp_path / "out.jsonl"), "--config", str(tmp_path / "x.yaml"), "--n", "10"])
    assert result.exit_code == 2 and "confirmation threshold" in result.output and "--yes" in result.output
    assert not (tmp_path / "out.jsonl").exists()
