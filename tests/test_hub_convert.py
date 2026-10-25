"""The dataset hub's converters and stratified subsets, on tiny committed fixtures (no network, no downloads)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragbench.datasets.hub.convert import convert_galileo, convert_hotpotqa, convert_multihop_rag
from ragbench.datasets.hub.subset import SubsetError, stratified_subset

FIXTURES = Path(__file__).parent / "fixtures" / "hub"


def _json(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _by_id(rows):
    return {row["id"]: row for row in rows}


# -- MultiHop-RAG ---------------------------------------------------------------------------------


def test_multihop_documents_keep_title_source_date_and_body():
    converted = convert_multihop_rag(_json("multihop_corpus.json"), _json("multihop_queries.json"))

    assert list(converted.documents) == [f"doc_news_{n:04d}" for n in range(1, 6)]
    first = converted.documents["doc_news_0001"]
    assert first.title == "Harbor Bank settles lawsuit"
    assert "The Daily Ledger" in first.text and "2023-09-26" in first.text and "Dana Reyes" in first.text and "agreed on Tuesday to pay $40 million" in first.text
    assert "None" not in converted.documents["doc_news_0002"].text, "a missing author is left out, not printed"
    assert converted.contexts == {}, "the corpus is shared by all questions"


def test_multihop_questions_carry_the_type_as_category_and_the_relevant_documents():
    questions = _by_id(convert_multihop_rag(_json("multihop_corpus.json"), _json("multihop_queries.json")).questions)

    first = questions["mh_0001"]
    assert first["category"] == "inference_query" and first["reference_answer"] == "Nova Systems"
    assert first["relevant_doc_ids"] == ["doc_news_0002", "doc_news_0003"] and first["answerable"] is True


def test_a_null_query_is_unanswerable_with_no_relevant_documents():
    null = _by_id(convert_multihop_rag(_json("multihop_corpus.json"), _json("multihop_queries.json")).questions)["mh_0005"]

    assert null["answerable"] is False and "relevant_doc_ids" not in null
    assert null["reference_answer"] == "Insufficient information." and null["category"] == "null_query"


def test_evidence_is_matched_by_url_then_title_and_kept_only_when_the_quote_is_in_the_article():
    converted = convert_multihop_rag(_json("multihop_corpus.json"), _json("multihop_queries.json"))
    questions = _by_id(converted.questions)

    assert questions["mh_0001"]["evidence"] == [
        {"doc_id": "doc_news_0002", "quote": "Nova Systems unveiled the Aurora chip on Monday."},
        {"doc_id": "doc_news_0003", "quote": "Regulators opened an antitrust probe into Nova Systems on Tuesday."},
    ]
    third = questions["mh_0003"]  # one url differs only by a query string; one fact is a paraphrase
    assert third["relevant_doc_ids"] == ["doc_news_0001", "doc_news_0003"]
    assert third["evidence"] == [{"doc_id": "doc_news_0003", "quote": "Regulators opened an antitrust probe into Nova Systems on Tuesday."}]


def test_evidence_that_matches_no_article_is_reported_and_leaves_the_question_unlabeled():
    converted = convert_multihop_rag(_json("multihop_corpus.json"), _json("multihop_queries.json"))
    lost = _by_id(converted.questions)["mh_0006"]

    assert "relevant_doc_ids" not in lost and lost["answerable"] is True
    assert any("mh_0006" in warning and "corpus" in warning for warning in converted.warnings)


def test_multihop_qrels_have_one_row_per_relevant_document():
    converted = convert_multihop_rag(_json("multihop_corpus.json"), _json("multihop_queries.json"))

    assert {"query_id": "mh_0002", "doc_id": "doc_news_0001", "relevance": 1} in converted.qrels
    assert len([row for row in converted.qrels if row["query_id"] == "mh_0001"]) == 2


# -- HotpotQA -------------------------------------------------------------------------------------


def test_hotpotqa_paragraphs_are_deduplicated_by_title_and_text():
    converted = convert_hotpotqa(_json("hotpotqa_rows.json"))

    titles = sorted(doc.title for doc in converted.documents.values())
    assert titles == ["Ed Wood", "Scott Derrickson", "Sinister (film)", "Tyler Bates", "Tyler Bates"], "two different Tyler Bates paragraphs, one Ed Wood"
    assert all(doc_id.startswith("doc_hp_") for doc_id in converted.documents)


def test_hotpotqa_labels_come_from_the_supporting_facts():
    converted = convert_hotpotqa(_json("hotpotqa_rows.json"))
    questions = _by_id(converted.questions)
    first = questions["hp001"]

    assert first["category"] == "comparison" and first["difficulty"] == "easy" and first["reference_answer"] == "yes"
    relevant = {converted.documents[doc_id].title for doc_id in first["relevant_doc_ids"]}
    assert relevant == {"Ed Wood", "Scott Derrickson"}
    assert {item["quote"] for item in first["evidence"]} == {"Edward Davis Wood Jr. was an American filmmaker.", "Scott Derrickson is an American director."}
    second = questions["hp002"]
    assert {item["quote"] for item in second["evidence"]} == {"Its music was composed by Christopher Young.", "He scored Doctor Strange."}
    assert {converted.documents[d].text for d in second["relevant_doc_ids"]} >= {"Tyler Bates is a composer. He scored Doctor Strange."}


def test_hotpotqa_contexts_list_every_paragraph_shown_with_a_question():
    converted = convert_hotpotqa(_json("hotpotqa_rows.json"))

    assert len(converted.contexts["hp001"]) == 3 and set(converted.contexts["hp001"]) <= set(converted.documents)
    assert set(converted.contexts["hp001"]) >= set(_by_id(converted.questions)["hp001"]["relevant_doc_ids"])


# -- Galileo RAGBench -----------------------------------------------------------------------------


def test_galileo_documents_are_shared_by_text_and_labels_follow_the_relevant_keys():
    converted = convert_galileo(_json("galileo_rows.json"), "covidqa")
    questions = _by_id(converted.questions)

    assert len(converted.documents) == 3
    assert [converted.documents[d].text for d in questions["gl001"]["relevant_doc_ids"]] == ["Heat above 60 C kills the virus within minutes. It survives longer in the cold."]
    assert questions["gl001"]["category"] == "covidqa"
    assert converted.contexts["gl001"] == converted.contexts["gl003"] and len(converted.contexts["gl001"]) == 2


def test_galileo_with_no_relevant_sentence_is_unanswerable_and_a_reference_needs_an_adherent_response():
    questions = _by_id(convert_galileo(_json("galileo_rows.json"), "covidqa").questions)

    assert questions["gl002"]["answerable"] is False and "relevant_doc_ids" not in questions["gl002"]
    assert questions["gl001"]["reference_answer"] == "Temperatures above 60 C kill it."
    assert "reference_answer" not in questions["gl003"], "a response the annotators found unsupported is not a reference"
    assert questions["gl001"]["metadata"]["reference_is_model_response"] is True


def test_galileo_duplicate_question_ids_are_made_unique_with_a_warning():
    rows = _json("galileo_rows.json")
    rows[1]["id"] = rows[0]["id"]

    converted = convert_galileo(rows, "covidqa")

    assert len({row["id"] for row in converted.questions}) == 3
    assert any("duplicate" in warning.lower() for warning in converted.warnings)


# -- stratified subset ----------------------------------------------------------------------------


def _rows():
    rows = []
    for category, answerable, count in (("a", True, 48), ("a", False, 12), ("b", True, 32), ("b", False, 8)):
        rows += [{"id": f"{category}{int(answerable)}_{i:02d}", "category": category, "answerable": answerable} for i in range(count)]
    return rows


def test_a_subset_keeps_every_stratum_in_proportion_and_the_original_order():
    rows = _rows()

    chosen = stratified_subset(rows, 30, seed=1)

    assert len(chosen) == 30
    assert [r["id"] for r in chosen] == sorted((r["id"] for r in chosen), key=[x["id"] for x in rows].index)
    counts = {(c, a): sum(r["category"] == c and r["answerable"] == a for r in chosen) for c in "ab" for a in (True, False)}
    ideal = {("a", True): 14.4, ("a", False): 3.6, ("b", True): 9.6, ("b", False): 2.4}
    assert all(abs(counts[key] - ideal[key]) <= 1 for key in ideal), counts


def test_a_subset_always_keeps_a_rare_stratum():
    rows = _rows() + [{"id": "rare_0", "category": "rare", "answerable": True}]

    chosen = stratified_subset(rows, 10, seed=3)

    assert "rare_0" in {r["id"] for r in chosen} and len(chosen) == 10


def test_the_same_seed_gives_the_same_subset_and_another_seed_another():
    rows = _rows()

    assert stratified_subset(rows, 30, seed=1) == stratified_subset(rows, 30, seed=1)
    assert stratified_subset(rows, 30, seed=1) != stratified_subset(rows, 30, seed=2)


def test_a_subset_as_large_as_the_dataset_is_the_dataset():
    rows = _rows()
    assert stratified_subset(rows, 100, seed=1) == rows
    assert stratified_subset(rows, 500, seed=1) == rows


def test_a_subset_smaller_than_the_number_of_strata_is_refused():
    with pytest.raises(SubsetError, match="4 strata"):
        stratified_subset(_rows(), 3, seed=1)
