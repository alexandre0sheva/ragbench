"""`compression` (contextual compression): retrieve deeper, cut every candidate down to what bears on the question, pack to a token budget."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLMResult, MockLLM
from ragbench.models.prompts import COMPRESS_PASSAGES_MARKER
from ragbench.rag_systems import create_rag_system
from ragbench.utils.text import estimate_tokens, normalize_text, truncate

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"
CHUNKER = {"type": "word", "chunk_size": 60, "chunk_overlap": 0}


class RecordingLLM(MockLLM):
    """The mock's answers with every call recorded; `override(prompt)` may answer instead."""

    def __init__(self, override=None):
        super().__init__()
        self.override = override
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def generate(self, messages, **kwargs) -> LLMResult:
        prompt = "\n".join(str(m.get("content") or "") for m in messages)
        with self._lock:
            self.calls.append(prompt)
        text = self.override(prompt) if self.override else None
        if text is None:
            return super().generate(messages, **kwargs)
        cost = CostBreakdown(llm_prompt_tokens=10, llm_completion_tokens=5, llm_cost=0.001)
        return LLMResult(text=text, model="recording", prompt_tokens=10, completion_tokens=5, cost=cost, finish_reason="stop")


def _docs() -> list[Document]:
    return [
        Document(doc_id="doc_a", path="a.md", title="a", text="The Vela pump runs at 40 litres per minute. Its housing is painted blue. The warranty lasts two years."),
        Document(doc_id="doc_b", path="b.md", title="b", text="The cafeteria opens at noon. Parking is free on Sundays. The Vela pump weighs nine kilograms."),
        Document(doc_id="doc_c", path="c.md", title="c", text="Office plants are watered on Mondays. The lift is serviced every spring."),
    ]


def _system(llm=None, **retrieval):
    system = create_rag_system(SystemConfig(type="compression", chunker=CHUNKER, retrieval={"vector_store": "numpy", **retrieval}), force_mock=True)
    if llm is not None:
        system.llm = llm
    system.ingest(_docs())
    return system


def _verbatim(text: str, docs: list[Document]) -> bool:
    return all(any(normalize_text(piece) in normalize_text(d.text) for d in docs) for piece in text.split(" … "))


# --- llm_extract --------------------------------------------------------------------------------------------------------------


def test_llm_extract_keeps_only_quotes_found_word_for_word_in_their_chunk():
    def override(prompt: str):
        if COMPRESS_PASSAGES_MARKER not in prompt:
            return None
        passages = {int(n): body for n, body in re.findall(r"^\[(\d+)\] (.*)$", prompt, flags=re.M)}
        quotes = []
        for number, body in passages.items():
            if "Vela pump runs" in body:
                quotes.append({"id": number, "quotes": ["The Vela pump runs at 40 litres per minute.", "The pump is made in Norway."]})
        return json.dumps({"passages": quotes})

    llm = RecordingLLM(override)
    system = _system(llm, candidate_k=3)

    answer = system.answer_question("How fast does the Vela pump run?")

    block = answer.metadata["compression"]
    assert block["unverified_quotes"] == 1, "the invented sentence is dropped and counted"
    assert list(answer.metadata["context_texts"].values()) == ["The Vela pump runs at 40 litres per minute."]
    assert answer.metadata["context_chunk_ids"] == list(answer.metadata["context_texts"])
    generation = llm.calls[-1]
    assert "40 litres per minute" in generation and "painted blue" not in generation and "made in Norway" not in generation


def test_a_passage_id_written_as_a_string_still_counts():
    def override(prompt: str):
        if COMPRESS_PASSAGES_MARKER not in prompt:
            return None
        number = next(n for n, body in re.findall(r"^\[(\d+)\] (.*)$", prompt, flags=re.M) if "Vela pump runs" in body)
        return json.dumps({"passages": [{"id": number, "quotes": ["The Vela pump runs at 40 litres per minute."]}]})

    answer = _system(RecordingLLM(override), candidate_k=3).answer_question("How fast does the Vela pump run?")

    assert list(answer.metadata["context_texts"].values()) == ["The Vela pump runs at 40 litres per minute."]


def test_llm_extract_is_one_call_for_all_candidates():
    llm = RecordingLLM()
    system = _system(llm, candidate_k=3)

    system.answer_question("How fast does the Vela pump run?")

    assert sum(COMPRESS_PASSAGES_MARKER in call for call in llm.calls) == 1 and len(llm.calls) == 2


def test_the_mock_compression_is_smaller_than_its_input_and_fits_the_budget():
    system = _system(candidate_k=3, context_token_budget=40)

    answer = system.answer_question("How much does the Vela pump weigh?")

    block = answer.metadata["compression"]
    assert 0 < block["tokens_after"] < block["tokens_before"]
    assert sum(estimate_tokens(text) for text in answer.metadata["context_texts"].values()) <= 40
    assert _verbatim(" … ".join(answer.metadata["context_texts"].values()), _docs())


# --- embedding_filter and decision ----------------------------------------------------------------------------------------------


def test_embedding_filter_calls_no_llm_besides_the_answer_and_keeps_verbatim_sentences():
    llm = RecordingLLM()
    system = _system(llm, compressor="embedding_filter", candidate_k=3, similarity_threshold=0.2)

    answer = system.answer_question("How much does the Vela pump weigh?")

    assert len(llm.calls) == 1
    texts = list(answer.metadata["context_texts"].values())
    assert texts and _verbatim(" … ".join(texts), _docs())
    assert any("nine kilograms" in text for text in texts) and not any("cafeteria" in text for text in texts)
    assert [step.name for step in answer.steps][-2:] == ["compress", "answer"]


def test_the_decision_compressor_keeps_the_sentences_the_decision_model_accepts():
    system = _system(compressor="decision", candidate_k=3)

    answer = system.answer_question("How much does the Vela pump weigh?")

    texts = list(answer.metadata["context_texts"].values())
    assert any("nine kilograms" in text for text in texts) and not any("Parking" in text for text in texts)
    compress = next(step for step in answer.steps if step.name == "compress")
    assert compress.kind == "grade" and compress.cost.decision_input_tokens > 0


def test_when_nothing_survives_the_generator_gets_no_context():
    llm = RecordingLLM(lambda prompt: json.dumps({"passages": []}) if COMPRESS_PASSAGES_MARKER in prompt else None)
    system = _system(llm, candidate_k=3)

    answer = system.answer_question("How fast does the Vela pump run?")

    assert answer.metadata["context_chunk_ids"] == [] and answer.metadata["compression"]["tokens_after"] == 0
    assert "could not find" in answer.answer.lower()


@pytest.mark.parametrize("retrieval", [{"compressor": "summarize"}, {"candidate_k": 0}, {"similarity_threshold": 2}])
def test_bad_options_fail_when_the_system_is_built(retrieval):
    with pytest.raises(ValidationError):
        create_rag_system(SystemConfig(type="compression", retrieval=retrieval), force_mock=True)


# --- in a run ---------------------------------------------------------------------------------------------------------------------


def test_in_a_run_the_judge_reads_the_compressed_text_and_the_summary_reports_tokens_before_and_after(tmp_path, monkeypatch):
    from ragbench.evaluation import question_runner

    judged: list[tuple[str, str]] = []
    real = question_runner.context_recall_doc

    def spy(chunks, relevant):  # sees exactly the chunks the judge is handed
        judged.extend((chunk.chunk_id, chunk.text) for chunk in chunks)
        return real(chunks, relevant)

    monkeypatch.setattr(question_runner, "context_recall_doc", spy)
    chunker = {"type": "word", "chunk_size": 120, "chunk_overlap": 20}
    config = {
        "run": {"name": "compression", "output_dir": str(tmp_path / "results")},
        "dataset": {"documents_path": str(DEMO / "docs"), "questions_path": str(DEMO / "questions.jsonl"), "qrels_path": str(DEMO / "qrels.jsonl")},
        "systems": [
            {"type": "hybrid", "name": "hybrid", "chunker": chunker},
            {"type": "compression", "name": "compressed", "chunker": chunker, "retrieval": {"candidate_k": 10}},
        ],
        "evaluation": {"max_questions": 6, "max_workers": 1, "latency_probe_questions": 0},
        "cache": {"enabled": False},
    }
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    out = run_benchmark(path, force_mock=True)

    summary = pd.read_csv(out / "metrics_summary.csv").set_index("system")
    assert summary.loc["compressed", "avg_context_tokens_after"] < summary.loc["compressed", "avg_context_tokens_before"]
    assert pd.isna(summary.loc["hybrid", "avg_context_tokens_before"])
    rows = [json.loads(line) for line in (out / "per_question_results.jsonl").read_text().splitlines()]
    compressed = [row for row in rows if row["system"] == "compressed"]
    assert all(row["compression"]["tokens_after"] <= row["compression"]["tokens_before"] for row in compressed)
    for row in compressed:  # the stored previews (judge-audit sheet, report explorer) show what the generator read
        texts = row["compression"]["context_texts"]
        previews = {c["chunk_id"]: c["text_preview"] for c in row["retrieved_contexts"] if c["in_context"]}
        assert set(previews) == set(texts) and all(previews[i] == truncate(texts[i], 260) for i in texts)
    given = {chunk_id: text for row in compressed for chunk_id, text in row["compression"]["context_texts"].items()}
    seen = {chunk_id: text for chunk_id, text in judged if chunk_id in given}
    assert seen and seen == {chunk_id: given[chunk_id] for chunk_id in seen}, "the judge reads the compressed text, not the whole chunk"


def test_an_unusable_extraction_reply_falls_back_to_the_uncompressed_candidates():
    llm = RecordingLLM(lambda prompt: "Sorry, I cannot help with that." if COMPRESS_PASSAGES_MARKER in prompt else None)

    answer = _system(llm, candidate_k=3).answer_question("How fast does the Vela pump run?")

    block = answer.metadata["compression"]
    assert block["fallback"] is True and block["kept"] == 3 and block["tokens_after"] == block["tokens_before"]
    originals = {chunk.chunk_id: chunk.text for chunk in answer.retrieval_result.chunks}
    assert answer.metadata["context_texts"] == {chunk_id: originals[chunk_id] for chunk_id in answer.metadata["context_texts"]}
