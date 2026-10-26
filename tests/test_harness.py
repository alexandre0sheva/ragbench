"""The editable harness: default prompts render byte-for-byte as before, patches inherit what they omit, and edits reach the model."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from dataset_support import write_tiny_dataset
from pydantic import ValidationError

from ragbench.agents import prompts as agent_prompts
from ragbench.config.loader import load_config
from ragbench.config.schema import ExperimentConfig, SystemConfig
from ragbench.datasets.leakage import LeakageError
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.schema import Question
from ragbench.datasets.splits import SPLITS_NAME, load_splits, make_splits, write_splits
from ragbench.documents.schema import Document
from ragbench.evaluation import checkpoint
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.harness import FewShotExample, PromptHarness
from ragbench.harness.leakage import assert_few_shot_from_train
from ragbench.harness.prompts import ANSWER_PROMPTS, NO_RETRIEVAL_PROMPTS, REDUCE_PROMPTS, TOOL_AGENT_PROMPTS
from ragbench.rag_systems import create_rag_system

# The prompts as they were written out before the harness existed (0.3.x), kept here verbatim.
OLD_ANSWER = """You are answering questions using only the provided context.

Rules:
- Use only the provided context.
- If the answer is not available in the context, say: "I could not find the answer in the provided documents."
- Be concise but complete.
- Cite source document IDs in square brackets when possible, for example [doc_014].
"""
OLD_NO_RETRIEVAL = """You are answering questions from your own knowledge only. No documents are provided.

Rules:
- Answer only what you actually know; do not invent names, numbers, dates, or identifiers.
- If you do not know, say: "I could not find the answer from my own knowledge."
- Be concise but complete.
"""
OLD_REDUCE = """You are answering questions about a document collection from points that were drawn from reports on groups of related entities.

Rules:
- Use only the points provided; points with a higher score matter more.
- If the points do not answer the question, say: "I could not find the answer in the provided documents."
- Be concise but complete.
- Cite source document IDs in square brackets when possible, for example [doc_014].
"""
OLD_TOOL_AGENT = (
    "You are a research agent with tools, answering a question about a collection of documents.\n"
    "Use the tools to find the facts you need, then answer. Rules:\n"
    "- Base the answer only on what the tools returned; never guess or use outside knowledge.\n"
    "- If a tool can compute something exactly (arithmetic, dates), use it instead of working it out yourself.\n"
    "- Cite the source document ID in square brackets, for example [doc_014], after each fact taken from a document.\n"
    '- If the documents do not contain the answer, say: "I could not find the answer in the provided documents."\n'
    "- When you have enough, reply with the final answer as plain text and no tool call. Be concise but complete."
)
GREP_GUIDANCE = (
    "There is no search index. Use `corpus_grep` to find where names, identifiers and figures occur, `read_document` to read around a hit, "
    "and `list_documents` to see what exists."
)
OLD_REACT_BLOCK = (
    "\n\nYou cannot call functions directly. Reply with exactly one JSON object and nothing else each turn.\n"
    'To use a tool: {"action": "tool", "tool": "<tool name>", "args": {<arguments>}}\n'
    'To give the final answer: {"action": "answer", "answer": "<your answer with [doc_id] citations>"}\n'
    "After each tool call you will receive its result as an observation.\n\nAvailable tools:\n"
)


@pytest.mark.parametrize(("defaults", "old"), [(ANSWER_PROMPTS, OLD_ANSWER), (NO_RETRIEVAL_PROMPTS, OLD_NO_RETRIEVAL), (REDUCE_PROMPTS, OLD_REDUCE)])
def test_the_default_answer_prompts_render_as_before(defaults, old):
    assert PromptHarness.resolve(defaults).answer_system() == old


def test_the_default_tool_agent_prompt_renders_as_before_in_both_protocols():
    harness = PromptHarness.resolve(TOOL_AGENT_PROMPTS)
    assert agent_prompts.tool_agent_system(harness) == OLD_TOOL_AGENT
    assert agent_prompts.tool_agent_system(harness, "- search: find passages") == OLD_TOOL_AGENT + OLD_REACT_BLOCK + "- search: find passages"
    grep = PromptHarness.resolve({**TOOL_AGENT_PROMPTS, "tool_guidance": GREP_GUIDANCE})
    assert agent_prompts.tool_agent_system(grep) == OLD_TOOL_AGENT + "\n" + GREP_GUIDANCE


def test_a_patch_replaces_only_the_fields_it_names():
    harness = PromptHarness.resolve(ANSWER_PROMPTS, {"system_prompt": "You answer questions about ship insurance.", "citation_style": "Cite as (doc_id)."})
    assert harness.answer_system() == (
        "You answer questions about ship insurance.\n\nRules:\n- Use only the provided context.\n"
        '- If the answer is not available in the context, say: "I could not find the answer in the provided documents."\n'
        "- Be concise but complete.\n- Cite as (doc_id).\n"
    )


def test_an_empty_field_drops_its_rule():
    harness = PromptHarness.resolve(ANSWER_PROMPTS, {"refusal_policy": "", "citation_style": ""})
    assert harness.answer_system() == "You are answering questions using only the provided context.\n\nRules:\n- Use only the provided context.\n- Be concise but complete.\n"
    bare = PromptHarness.resolve(ANSWER_PROMPTS, {"grounding": "", "refusal_policy": "", "answer_instructions": "", "citation_style": ""})
    assert bare.answer_system() == "You are answering questions using only the provided context.\n"
    agent = PromptHarness.resolve(TOOL_AGENT_PROMPTS, {"answer_instructions": ""})
    assert agent_prompts.tool_agent_system(agent).endswith("- When you have enough, reply with the final answer as plain text and no tool call.")


def test_few_shot_examples_close_the_system_message_without_looking_like_the_question():
    examples = [FewShotExample(question="Who insures the Aurora?", answer="HarborShield [doc_001]."), FewShotExample(question="When?", answer="In 2024.")]
    harness = PromptHarness.resolve(ANSWER_PROMPTS, {"few_shot": examples})
    text = harness.answer_system()
    assert text == OLD_ANSWER + (
        "\nExamples of good answers (they show the form of an answer; they are not evidence):\n"
        "Q: Who insures the Aurora?\nA: HarborShield [doc_001].\n\nQ: When?\nA: In 2024.\n"
    )
    assert "Question:" not in text and "Context:" not in text
    assert agent_prompts.tool_agent_system(PromptHarness.resolve(TOOL_AGENT_PROMPTS, {"few_shot": examples})).endswith("Q: When?\nA: In 2024.\n")


def test_a_field_the_defaults_do_not_have_is_refused():
    with pytest.raises(ValueError, match="query_rewrite_prompt"):
        PromptHarness.resolve(ANSWER_PROMPTS, {"query_rewrite_prompt": "Rewrite it."})


# --- the `harness:` block of a system config ------------------------------------------------------------------------------------------


def _system(**fields) -> SystemConfig:
    return SystemConfig.model_validate(fields)


def test_a_misspelt_harness_field_gets_a_did_you_mean():
    with pytest.raises(ValidationError, match=r"unknown option 'harness\.sytem_prompt'\. Did you mean 'system_prompt'\?"):
        _system(type="bm25", harness={"sytem_prompt": "You are terse."})


def test_a_prompt_the_system_does_not_send_is_refused_with_the_ones_it_does():
    with pytest.raises(ValidationError, match=r"System 'bm25' \(type bm25\): harness: this system has no `query_rewrite_prompt` \(its harness prompts: system_prompt, grounding"):
        _system(type="bm25", harness={"query_rewrite_prompt": "Rewrite it."})
    with pytest.raises(ValidationError, match="does not take a `harness:`"):
        _system(type="python_callable", retrieval={"path": "tests.fixtures:answer"}, harness={"system_prompt": "Hi."})


def test_knobs_set_the_option_they_name_and_win_over_the_section():
    hybrid = _system(type="hybrid", retrieval={"final_top_k": 5}, harness={"top_k": 8, "vector_weight": 0.5, "chunk_size": 300})
    assert (hybrid.retrieval["final_top_k"], hybrid.retrieval["vector_weight"], hybrid.chunker["chunk_size"]) == (8, 0.5, 300)
    assert _system(type="bm25", harness={"top_k": 3}).retrieval == {"top_k": 3}
    assert _system(type="iterative", harness={"max_hops": 2}).retrieval["max_hops"] == 2
    assert _system(type="grep_agent", harness={"max_steps": 4}).retrieval["max_steps"] == 4


def test_knobs_are_bounded_and_must_exist_on_the_system():
    with pytest.raises(ValidationError, match=r"harness\.top_k: Input should be less than or equal to 50"):
        _system(type="bm25", harness={"top_k": 500})
    with pytest.raises(ValidationError, match=r"harness: this system has no knob `max_hops` \(its knobs: top_k, chunk_size\)"):
        _system(type="bm25", harness={"max_hops": 2})
    with pytest.raises(ValidationError, match=r"harness\.reranker: Unknown reranker 'magic'\. Available: "):
        _system(type="hybrid_rerank", harness={"reranker": "magic"})


# --- edits reach the model ------------------------------------------------------------------------------------------------------------

DOCS = [
    Document(doc_id="doc_a", path="a.md", title="a", text="Zorblax turbines produce clean energy for coastal towns. Zorblax turbines are quiet."),
    Document(doc_id="doc_b", path="b.md", title="b", text="Quuxwell batteries store energy overnight for mountain villages. Quuxwell cells last long."),
]
QUESTION = "Which turbines produce clean energy for coastal towns?"
EDITED = "You write search queries about renewable energy hardware."


def _ask(fields: dict, question: str = QUESTION) -> tuple[list[list[dict]], object]:
    """Answer QUESTION with a mock system built from `fields`; every message list its model received, and the answer."""
    system = create_rag_system(_system(**fields), force_mock=True)
    calls: list[list[dict]] = []
    generate = system.llm.generate

    def recording(messages, **kwargs):
        calls.append(messages)
        return generate(messages, **kwargs)

    system.llm.generate = recording  # type: ignore[method-assign]
    system.ingest(DOCS)
    return calls, system.answer_question(question, top_k=2, context_k=2)


def _system_messages(calls: list[list[dict]]) -> list[str]:
    return [message["content"] for messages in calls for message in messages if message["role"] == "system"]


def test_an_edited_answer_prompt_and_few_shot_examples_reach_the_generator():
    calls, _ = _ask({"type": "bm25", "harness": {"answer_instructions": "Answer in one sentence.", "few_shot": [{"question": "Q1?", "answer": "A1."}]}})
    assert _system_messages(calls) == [
        PromptHarness.resolve(ANSWER_PROMPTS, {"answer_instructions": "Answer in one sentence.", "few_shot": [FewShotExample(question="Q1?", answer="A1.")]}).answer_system()
    ]


@pytest.mark.parametrize("system_type", ["hyde", "llm_heavy", "rag_fusion", "decompose", "corrective", "iterative"])
def test_an_edited_query_rewrite_prompt_reaches_the_query_writing_call(system_type):
    # `corrective` writes a new query only when the first search found too little.
    calls, _ = _ask({"type": system_type, "harness": {"query_rewrite_prompt": EDITED}}, "Who founded the Plinth harbour fleet?" if system_type == "corrective" else QUESTION)
    assert EDITED in _system_messages(calls)


def test_a_tool_agent_with_an_edited_opening_still_runs_its_tools_in_mock():
    calls, answer = _ask({"type": "agent_search", "harness": {"system_prompt": "You are a careful analyst.", "tool_guidance": "Search twice."}})
    assert _system_messages(calls)[0].startswith("You are a careful analyst.\nUse the tools to find the facts you need, then answer. Rules:\n")
    assert _system_messages(calls)[0].endswith("\nSearch twice.")
    assert answer.retrieval_result.chunks, "the mock agent called `search`"


# --- few-shot examples come from the train split ---------------------------------------------------------------------------------------


def _few_shot_config(root: Path, examples: list[dict], split: str | None) -> Path:
    dataset = write_tiny_dataset(root)
    if split is not None:
        questions = load_dataset(Path(dataset["questions_path"])).questions
        write_splits(Path(dataset["questions_path"]).parent / SPLITS_NAME, make_splits(questions, train=0.5, dev=0.25, seed=0))
        dataset["split"] = split
    config = {
        "run": {"name": "few_shot", "output_dir": str(root / "results")},
        "dataset": dataset,
        "systems": [{"type": "bm25", "name": "bm25", "harness": {"few_shot": examples}}],
        "evaluation": {"max_workers": 1, "latency_probe_questions": 0},
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _split_question(root: Path, name: str) -> Question:
    splits = load_splits(root / SPLITS_NAME)
    return next(question for question in load_dataset(root / "questions.jsonl").questions if question.id in splits.ids(name))


def test_few_shot_examples_from_the_train_split_are_allowed(tmp_path):
    config = _few_shot_config(tmp_path, [], "test")
    train = _split_question(tmp_path, "train")
    raw = yaml.safe_load(config.read_text())
    raw["systems"][0]["harness"]["few_shot"] = [{"question": train.question, "answer": train.reference_answer or "See the documents."}]
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert (run_benchmark(config, force_mock=True) / "metrics_summary.csv").exists()


@pytest.mark.parametrize("split_name", ["dev", "test"])
def test_a_few_shot_example_from_a_held_out_split_stops_the_run_before_it_starts(tmp_path, split_name):
    config = _few_shot_config(tmp_path, [], "test")
    held_out = _split_question(tmp_path, split_name)
    raw = yaml.safe_load(config.read_text())
    raw["systems"][0]["harness"]["few_shot"] = [{"question": held_out.question, "answer": "Whatever the documents say."}]
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(LeakageError, match=rf"bm25\.harness\.few_shot\[0\]: question text of {held_out.id}"):
        run_benchmark(config, force_mock=True)
    assert not (tmp_path / "results").exists() or not any((tmp_path / "results").rglob("per_question_results.jsonl"))


def test_without_splits_no_question_of_the_dataset_may_be_a_few_shot_example(tmp_path):
    question = load_dataset(Path(write_tiny_dataset(tmp_path)["questions_path"])).questions[0]
    with pytest.raises(LeakageError, match="few_shot"):
        run_benchmark(_few_shot_config(tmp_path, [{"question": question.question, "answer": "It depends."}], None), force_mock=True)
    invented = _few_shot_config(tmp_path, [{"question": "What colour is a ripe banana in most supermarkets?", "answer": "Yellow."}], None)
    assert (run_benchmark(invented, force_mock=True) / "metrics_summary.csv").exists()


def test_a_system_without_a_harness_keeps_the_checkpoint_hash_it_had_before_the_harness_existed(monkeypatch):
    monkeypatch.setattr(checkpoint, "__version__", "0.0.0")
    config = ExperimentConfig.model_validate(
        {
            "run": {"name": "pinned"},
            "dataset": {"documents_path": "docs", "questions_path": "questions.jsonl"},
            "systems": [{"type": "bm25", "name": "bm25", "retrieval": {"top_k": 3}}, {"type": "hybrid", "name": "hybrid"}],
        }
    )
    # Computed with the code before Task 19 (0.4.0 development), with the version pinned as here: old checkpoints stay valid.
    assert [checkpoint.context_hash(system, config, "digest", "mock") for system in config.systems] == [
        "4b468d8441cb8b2d4ecf9fc6e95ecbee111b1de7762354df3f90010963284551",
        "21e401bd29b2cc4f60b0b1ccbe5d79fd9a288c59cfbc69ea1aed34c9990988ad",
    ]


def test_a_mistake_in_another_section_is_reported_as_usual_next_to_a_harness():
    with pytest.raises(ValidationError, match=r"unknown option 'retrieval\.top_kk'\. Did you mean 'top_k'\?"):
        _system(type="bm25", retrieval={"top_kk": 3}, harness={"top_k": 4})


def test_few_shot_examples_given_as_objects_are_checked_too(tmp_path):
    config = load_config(_few_shot_config(tmp_path, [], "test"))
    held_out = _split_question(tmp_path, "test")
    system = config.systems[0].model_copy(update={"harness": {"few_shot": [FewShotExample(question=held_out.question, answer="Unknown.")]}})
    with pytest.raises(LeakageError, match=r"bm25\.harness\.few_shot\[0\]"):
        assert_few_shot_from_train(config.model_copy(update={"systems": [system]}))
