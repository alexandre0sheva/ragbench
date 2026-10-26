"""`late_interaction` (ColBERT-style MaxSim): the scoring, both modes, the checkpoint layouts the loader reads, and the offline encoder.

No test downloads a model: mock runs use deterministic hashing token vectors, and the real loader is exercised on fixture folders."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document, RetrievedChunk
from ragbench.models.colbert import (
    HashingTokenEncoder,
    frame_document,
    frame_query,
    load_token_encoder,
    marker_token_id,
    maxsim,
    read_settings,
)
from ragbench.models.cost import CostBreakdown
from ragbench.rag_systems import create_rag_system

CHUNKER = {"type": "word", "chunk_size": 40, "chunk_overlap": 0}


def _docs() -> list[Document]:
    return [
        Document(doc_id="doc_a", path="a.md", title="a", text="The Vela pump runs at forty litres per minute."),
        Document(doc_id="doc_b", path="b.md", title="b", text="The cafeteria opens at noon on weekdays."),
        Document(doc_id="doc_c", path="c.md", title="c", text="Vela pump warranty: two years, parts and labour."),
        Document(doc_id="doc_d", path="d.md", title="d", text="Parking is free on Sundays and holidays."),
    ]


def _system(**retrieval):
    system = create_rag_system(SystemConfig(type="late_interaction", chunker=CHUNKER, retrieval={"vector_store": "numpy", **retrieval}), force_mock=True)
    system.ingest(_docs())
    return system


# --- MaxSim -------------------------------------------------------------------------------------------------------------------


def test_maxsim_sums_each_query_vectors_best_match():
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    document = np.array([[0.6, 0.8], [1.0, 0.0], [0.0, -1.0]])

    assert maxsim(query, document) == pytest.approx(1.0 + 0.8)


def test_the_hashing_encoder_is_deterministic_and_matches_identical_tokens_exactly():
    encoder = HashingTokenEncoder()
    [doc] = encoder.encode_documents(["Vela pump"])

    assert np.allclose(encoder.encode_query("vela pump"), HashingTokenEncoder().encode_query("vela pump"))
    assert maxsim(encoder.encode_query("vela pump"), doc) == pytest.approx(2.0)
    assert encoder.document_vector_counts(["Vela pump runs", "x"]) == [3, 1]


# --- modes --------------------------------------------------------------------------------------------------------------------


def test_rerank_mode_reorders_a_planted_candidate_list_by_maxsim(monkeypatch):
    system = _system(mode="rerank", candidate_k=3)
    planted = [
        RetrievedChunk(chunk_id=f"p{i}", doc_id=f"doc_{i}", text=text, score=1.0 / i, rank=i)
        for i, text in enumerate(["cafeteria noon weekdays", "parking free sundays", "vela pump warranty years"], start=1)
    ]
    monkeypatch.setattr(system, "_first_stage", lambda question, k: (planted, CostBreakdown()))
    question = "vela pump warranty"

    result = system.fetch_context(question, top_k=3)

    encoder = HashingTokenEncoder()
    expected = sorted(planted, key=lambda c: -maxsim(encoder.encode_query(question), encoder.encode_documents([c.text])[0]))
    assert [c.chunk_id for c in result.chunks] == [c.chunk_id for c in expected] and result.chunks[0].chunk_id == "p3"
    assert [c.rank for c in result.chunks] == [1, 2, 3] and result.chunks[0].metadata["first_stage_rank"] == 3


def test_full_mode_is_exact_maxsim_over_every_chunk():
    system = _system(mode="full")
    question = "How fast does the Vela pump run?"

    result = system.fetch_context(question, top_k=4)

    encoder = HashingTokenEncoder()
    brute = sorted(system.chunks, key=lambda c: -maxsim(encoder.encode_query(question), encoder.encode_documents([c.text])[0]))
    assert [c.chunk_id for c in result.chunks] == [c.chunk_id for c in brute]
    assert [c.score for c in result.chunks] == sorted((c.score for c in result.chunks), reverse=True)
    assert [step.name for step in system.answer_question(question).steps] == ["maxsim_search", "answer"]


def test_full_mode_refuses_a_corpus_too_large_and_suggests_rerank():
    system = create_rag_system(SystemConfig(type="late_interaction", chunker=CHUNKER, retrieval={"mode": "full", "max_index_vectors": 10}), force_mock=True)

    with pytest.raises(ValueError, match="mode: rerank"):
        system.ingest(_docs())


def test_full_mode_needs_no_embedding_model_at_ingestion():
    system = _system(mode="full")

    assert system.ingest(_docs()).cost.embedding_cost == 0 and system.ingest(_docs()).cost.embedding_input_tokens == 0


# --- the encoder ----------------------------------------------------------------------------------------------------------------


def test_mock_runs_never_load_a_model():
    assert isinstance(load_token_encoder("answerdotai/answerai-colbert-small-v1", force_mock=True), HashingTokenEncoder)


def test_without_the_colbert_extra_a_live_build_names_the_extra(monkeypatch):
    # As if not installed (and nothing is downloaded). Torch too: importing it for real next to faiss aborts a macOS process (two OpenMP runtimes).
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(ImportError, match=r"ragbench\[colbert\]"):
        load_token_encoder("some/colbert-model", force_mock=False)


def test_a_stanford_colbert_checkpoint_is_read_from_its_artifact_metadata(tmp_path: Path):
    (tmp_path / "artifact.metadata").write_text(
        json.dumps({"query_token_id": "[unused0]", "doc_token_id": "[unused1]", "query_maxlen": 32, "doc_maxlen": 300, "attend_to_mask_tokens": False, "mask_punctuation": True})
    )

    settings = read_settings(tmp_path)

    assert (settings.query_marker, settings.document_marker, settings.query_length, settings.document_length) == ("[unused0]", "[unused1]", 32, 300)
    assert settings.projection_file == "model.safetensors" and "." in settings.skiplist and not settings.attend_to_expansion
    assert settings.query_prompt == settings.document_prompt == ""


def test_a_pylate_checkpoint_is_read_from_its_sentence_transformers_config(tmp_path: Path):
    (tmp_path / "1_Dense").mkdir()
    (tmp_path / "config_sentence_transformers.json").write_text(
        json.dumps(
            {
                "prompts": {"query": "search_query: ", "document": "search_document: "},
                "query_prefix": "[Q] ",
                "document_prefix": "[D] ",
                "query_length": 39,
                "document_length": 519,
                "attend_to_expansion_tokens": False,
                "skiplist_words": ["!", "."],
            }
        )
    )

    settings = read_settings(tmp_path)

    assert (settings.query_marker, settings.document_marker, settings.query_prompt, settings.document_prompt) == ("[Q]", "[D]", "search_query: ", "search_document: ")
    assert (settings.query_length, settings.document_length, settings.skiplist, settings.projection_file) == (39, 519, ("!", "."), "1_Dense/model.safetensors")


def test_a_folder_that_is_no_colbert_checkpoint_is_refused(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}")

    with pytest.raises(ValueError, match="ColBERT"):
        read_settings(tmp_path)


def test_a_query_gets_its_marker_after_the_first_token_and_is_padded_with_masks_it_does_not_attend_to():
    ids, attention = frame_query([101, 7, 8, 102], marker_id=1, length=8, mask_id=103, attend_to_expansion=False)

    assert ids == [101, 1, 7, 8, 102, 103, 103, 103]
    assert attention == [1, 1, 1, 1, 1, 0, 0, 0]
    assert frame_query([101, 7, 102], marker_id=1, length=5, mask_id=103, attend_to_expansion=True)[1] == [1, 1, 1, 1, 1]


def test_a_document_gets_its_marker_and_drops_skiplist_tokens_from_its_vectors():
    ids, keep = frame_document([101, 7, 999, 8, 102], marker_id=2, length=6, skip_ids={999})

    assert ids == [101, 2, 7, 999, 8, 102]
    assert keep == [True, True, True, False, True, True]
    assert frame_document([101, 7, 8, 9, 10, 102], marker_id=2, length=4, skip_ids=set())[0] == [101, 2, 7, 102], "cut to length, the end token kept"


@pytest.mark.parametrize("retrieval", [{"mode": "exact"}, {"candidate_k": 0}, {"first_stage": "vector"}])
def test_bad_options_fail_when_the_system_is_built(retrieval):
    with pytest.raises(ValidationError):
        create_rag_system(SystemConfig(type="late_interaction", retrieval=retrieval), force_mock=True)


def test_a_live_run_imports_the_model_libraries_up_front_and_a_mock_run_does_not():
    from ragbench.config.schema import ExperimentConfig
    from ragbench.evaluation.warm_up import modules_to_warm_up

    config = ExperimentConfig.model_validate(
        {"run": {"name": "x"}, "dataset": {"documents_path": "docs", "questions_path": "q.jsonl"}, "systems": [{"type": "late_interaction"}, {"type": "bm25"}]}
    )

    assert {"torch", "transformers"} <= set(modules_to_warm_up(config, "live"))
    assert not {"torch", "transformers"} & set(modules_to_warm_up(config, "mock"))


class _Vocab:
    unk_token_id = 100

    def convert_tokens_to_ids(self, token):
        return {"[unused0]": 1, "[Q]": 50368}.get(token, self.unk_token_id)


def test_a_marker_the_tokenizer_does_not_know_is_refused_instead_of_becoming_the_unknown_token():
    assert marker_token_id(_Vocab(), "[unused0]") == 1

    with pytest.raises(ValueError, match=r"\[D\]"):
        marker_token_id(_Vocab(), "[D]")


def test_full_mode_tokenizes_each_chunk_once(monkeypatch):
    import ragbench.models.colbert as colbert

    calls: list[str] = []
    real = colbert.tokenize
    monkeypatch.setattr(colbert, "tokenize", lambda text: calls.append(text) or real(text))
    system = create_rag_system(SystemConfig(type="late_interaction", chunker=CHUNKER, retrieval={"mode": "full"}), force_mock=True)

    system.ingest(_docs())

    assert len(calls) == len(system.chunks), "the size guard and the encoding share one tokenization"
