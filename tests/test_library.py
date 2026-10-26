"""The experience library: what an optimization learned on one collection, read by later work on similar ones, never holding held-out text."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from cli_support import invoke
from optimize_support import edit_config, register_tunable, scripted_reply, write_project
from scripted_llm import ScriptedLLM

from ragbench.datasets.leakage import LeakageError, LeakageGuard
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.profile import profile_dataset
from ragbench.datasets.schema import Question
from ragbench.documents.loaders import load_documents
from ragbench.optimize.library import MIN_SIMILARITY, Entry, Library, LibraryError, features_of, fingerprint_of, similarity
from ragbench.optimize.loop import OptimizeSettings, optimize
from ragbench.optimize.proposer import Proposer
from ragbench.utils.jsonl import write_jsonl

GOOD = {"INV": "GOOD", "DOM": "Name the item.", "COMPOSE": "GOOD"}
FEATURES = {
    "language": "en",
    "documents": 40,
    "tokens": 12_000,
    "doc_tokens_p50": 300.0,
    "question_words_p50": 10.0,
    "label_free": False,
    "answerable_ratio": 1.0,
    "categories": ["direct_fact"],
}


@pytest.fixture(autouse=True)
def _tunable(monkeypatch):
    register_tunable(monkeypatch)


@pytest.fixture
def library() -> Library:
    """The library the autouse conftest fixture points `RAGBENCH_LIBRARY` at."""
    return Library(Path(os.environ["RAGBENCH_LIBRARY"]))


def entry(created_at: str = "2026-01-01T00:00:00.000000+00:00", **changes) -> Entry:
    fields = {
        "created_at": created_at,
        "fingerprint": fingerprint_of(FEATURES),
        "features": FEATURES,
        "mock": True,
        "system": "tuned",
        "system_type": "hybrid_rerank",
        "harness": {"top_k": 10},
        "changed": {"top_k": {"from": None, "to": 10}},
        "gain": {"mean_diff": 0.4, "ci_lo": 0.1, "ci_hi": 0.7, "p_holm": 0.01, "n": 50},
        "lesson": "A larger `top_k` helped on short, factual collections.",
        **changes,
    }
    return Entry(**fields)


def project(root: Path) -> Path:
    """`write_project` into `root`, created if need be; the config is reused when `root` already holds one."""
    root.mkdir(parents=True, exist_ok=True)
    return root / "config.yaml" if (root / "config.yaml").exists() else write_project(root)


def optimize_in(root: Path, edits: dict[str, str], *, name: str = "opt", **settings):
    config = project(root)
    llm = ScriptedLLM(lambda prompt: scripted_reply(prompt, edits))
    result = optimize(config, OptimizeSettings(system="tuned", k=1, **settings), proposer=Proposer(llm), out_dir=root / name, force_mock=True)
    return result, llm


# -- similarity and the key -------------------------------------------------------------------------------------------------------------------


def test_the_same_corpus_shape_gives_the_same_key_and_full_similarity():
    assert fingerprint_of(FEATURES) == fingerprint_of({**FEATURES, "tokens": 12_400, "documents": 41})
    assert similarity(FEATURES, FEATURES) == 1.0


def test_a_different_language_and_size_is_a_different_key_and_not_similar():
    other = {**FEATURES, "language": "de", "tokens": 4_000_000, "doc_tokens_p50": 9_000.0, "documents": 8_000, "label_free": True, "categories": ["multi_hop"]}
    assert fingerprint_of(other) != fingerprint_of(FEATURES)
    assert similarity(FEATURES, other) < MIN_SIMILARITY
    near = {**FEATURES, "tokens": 15_000, "documents": 50}
    assert similarity(FEATURES, near) > similarity(FEATURES, other)
    assert similarity(FEATURES, near) >= MIN_SIMILARITY


# -- the store --------------------------------------------------------------------------------------------------------------------------------


def test_entries_are_appended_in_time_order_and_only_older_ones_are_read(library):
    first = library.append(entry("2026-01-01T00:00:00.000000+00:00", lesson="first"))
    library.append(entry("2026-03-01T00:00:00.000000+00:00", lesson="third"))
    library.append(entry("2026-02-01T00:00:00.000000+00:00", lesson="second"))

    assert [e.lesson for e in library.entries()] == ["first", "second", "third"]
    assert [e.lesson for e in library.entries(before=datetime(2026, 2, 15, tzinfo=UTC))] == ["first", "second"]
    assert library.entries(before=datetime(2026, 1, 1, tzinfo=UTC)) == []  # an entry made at the very moment is not older
    assert first.id and first.id != library.entries()[1].id
    assert len((library.path / "entries.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_append_stamps_the_time_when_the_entry_has_none(library):
    stored = library.append(entry(created_at=""))
    assert datetime.fromisoformat(stored.created_at) <= datetime.now(UTC)


def test_mock_entries_are_invisible_to_live_reads_and_the_other_way_round(library):
    library.append(entry(mock=True, lesson="mock"))
    library.append(entry(mock=False, lesson="live"))
    assert [e.lesson for e in library.entries(mock=False)] == ["live"]
    assert [e.lesson for e in library.entries(mock=True)] == ["mock"]
    assert len(library.entries()) == 2


def test_nearest_ranks_by_similarity_and_leaves_out_what_is_not_similar(library):
    library.append(entry(lesson="near", features={**FEATURES, "tokens": 15_000}))
    library.append(entry(lesson="same", features=FEATURES))
    library.append(entry(lesson="far", features={**FEATURES, "language": "de", "tokens": 9_000_000, "documents": 20_000, "label_free": True}))

    matches = library.nearest(FEATURES, mock=True)

    assert [m.entry.lesson for m in matches] == ["same", "near"]
    assert matches[0].similarity == 1.0 and matches[0].similarity > matches[1].similarity >= MIN_SIMILARITY
    assert [m.entry.lesson for m in library.nearest(FEATURES, mock=True, limit=1)] == ["same"]


def test_nearest_reads_only_older_entries(library):
    library.append(entry("2026-01-01T00:00:00.000000+00:00", lesson="old"))
    library.append(entry("2026-06-01T00:00:00.000000+00:00", lesson="newer"))
    matches = library.nearest(FEATURES, before=datetime(2026, 3, 1, tzinfo=UTC), mock=True)
    assert [m.entry.lesson for m in matches] == ["old"]


def test_an_unreadable_line_is_reported_with_its_position(library):
    library.append(entry())
    with (library.path / "entries.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    with pytest.raises(LibraryError, match=r"entries\.jsonl line 2"):
        library.entries()


def test_clear_removes_one_collections_entries_or_all(library):
    other = {**FEATURES, "language": "fr", "tokens": 2_000_000}
    library.append(entry(lesson="a"))
    library.append(entry(lesson="b", features=other, fingerprint=fingerprint_of(other)))
    assert library.clear(fingerprint_of(other)) == 1
    assert [e.lesson for e in library.entries()] == ["a"]
    assert library.clear() == 1 and library.entries() == []


def test_a_guarded_append_refuses_held_out_text_and_writes_nothing(library):
    guard = LeakageGuard([Question(id="item_omega_003", question="What is the code of item omega 3?", reference_answer="code3x")])
    with pytest.raises(LeakageError):
        library.append(entry(lesson="Answer questions like: What is the code of item omega 3?"), guard=guard)
    assert library.entries() == []


# -- the optimizer writes and reads it ---------------------------------------------------------------------------------------------------------


def test_an_adopted_harness_is_recorded_with_its_verified_gain_and_one_lesson(tmp_path, library):
    result, _ = optimize_in(tmp_path / "a", GOOD)

    assert result.verdict == "improved"
    (stored,) = library.entries()
    assert stored.mock and stored.system == "tuned" and stored.system_type == "fake_tunable"
    assert stored.harness == {"answer_instructions": "GOOD"}
    assert stored.changed["answer_instructions"]["to"] == "GOOD"
    assert stored.gain["mean_diff"] > 0 and stored.gain["ci_lo"] < stored.gain["mean_diff"] < stored.gain["ci_hi"] and stored.gain["p_holm"] < 0.05
    assert "answer_instructions" in stored.lesson and "scripted" in stored.lesson  # the field, and the proposer's reason from train evidence
    assert stored.fingerprint == fingerprint_of(stored.features)
    assert f"{stored.id}" in (result.out_dir / "optimization.md").read_text(encoding="utf-8")


def test_a_run_without_a_reliable_gain_records_nothing(tmp_path, library):
    result, _ = optimize_in(tmp_path, {"INV": "MAGIC", "DOM": "MAGIC", "COMPOSE": "MAGIC"})
    assert result.verdict == "no_reliable_gain"
    assert library.entries() == []


def test_the_second_collection_reads_the_first_ones_lesson(tmp_path, library):
    _, first_llm = optimize_in(tmp_path / "a", GOOD)
    (stored,) = library.entries()
    assert not any(stored.lesson in prompt for prompt in first_llm.prompts)  # the first run had nothing to read

    result, second_llm = optimize_in(tmp_path / "b", GOOD)

    assert second_llm.prompts and all(stored.lesson in prompt for prompt in second_llm.prompts)
    assert (result.out_dir / "evidence" / "base-t0" / "library" / f"{stored.id}.md").is_file()
    assert any("library" in note.lower() and stored.id in note for note in result.notes)
    assert len(library.entries()) == 2  # and the second run added its own


def test_a_run_reads_only_entries_older_than_its_start(tmp_path, library):
    library.append(entry("2999-01-01T00:00:00.000000+00:00", lesson="FROM THE FUTURE", features=_features_of_project(tmp_path / "b")))
    library.append(entry("2000-01-01T00:00:00.000000+00:00", lesson="FROM THE PAST", features=_features_of_project(tmp_path / "b")))

    _, llm = optimize_in(tmp_path / "b", GOOD)

    assert all("FROM THE PAST" in prompt and "FROM THE FUTURE" not in prompt for prompt in llm.prompts)


def test_a_resumed_run_keeps_the_start_time_of_the_first_attempt(tmp_path, library):
    optimize_in(tmp_path / "a", GOOD)
    state = json.loads((tmp_path / "a" / "opt" / "state.json").read_text(encoding="utf-8"))
    assert datetime.fromisoformat(state["started_at"]) < datetime.fromisoformat(library.entries()[0].created_at)
    optimize_in(tmp_path / "a", GOOD)  # the same folder again: a resume
    assert json.loads((tmp_path / "a" / "opt" / "state.json").read_text(encoding="utf-8"))["started_at"] == state["started_at"]
    assert len(library.entries()) == 1  # the finished optimization is not recorded twice


def test_nothing_from_the_test_split_is_written(tmp_path, library):
    optimize_in(tmp_path / "a", GOOD)
    written = "\n".join(path.read_text(encoding="utf-8") for path in library.path.rglob("*") if path.is_file())
    assert written and "omega" not in written  # the test questions are "item omega N"
    dataset = load_dataset(tmp_path / "a" / "questions.jsonl", None)
    held_out = LeakageGuard(q for q in dataset.questions if "omega" in q.id)
    assert held_out.scan_text(written) == []


def test_an_entry_that_quotes_a_test_question_is_not_written(tmp_path, library, monkeypatch):
    """Whatever put the text there, the library refuses it: the optimizer hands every entry to the guard over dev and test."""
    from ragbench.optimize import loop

    real = loop.build_entry
    monkeypatch.setattr(loop, "build_entry", lambda **kw: replace(real(**kw), lesson="Try phrasing like: What is the code of item omega 3?"))

    result, _ = optimize_in(tmp_path / "a", GOOD)

    assert result.verdict == "improved" and library.entries() == []
    assert any("nothing written" in note.lower() and "Held-out content" in note for note in result.notes)


def test_the_features_describe_the_train_questions_only(tmp_path, library):
    root = tmp_path / "a"
    project(root)
    rows = [json.loads(line) for line in (root / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    write_jsonl(root / "questions.jsonl", [{**row, "category": "only_in_test"} if "omega" in row["id"] else row for row in rows])

    optimize_in(root, GOOD)

    (stored,) = library.entries()
    assert stored.features["categories"] == ["direct_fact"]  # the test questions' category is not in it


def test_a_mock_run_never_writes_to_the_default_location(tmp_path, monkeypatch):
    monkeypatch.delenv("RAGBENCH_LIBRARY")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    optimize_in(tmp_path / "a", GOOD)
    assert not (tmp_path / "home" / ".ragbench").exists()


def test_the_config_names_the_library_and_can_switch_it_off(tmp_path, library):
    root = tmp_path / "a"
    config = project(root)
    chosen = tmp_path / "chosen"
    edit_config(config, lambda raw: {**raw, "library": {"path": str(chosen)}})
    optimize_in(root, GOOD)
    assert len(Library(chosen).entries()) == 1 and library.entries() == []

    edit_config(config, lambda raw: {**raw, "library": {"path": str(chosen), "enabled": False}})
    optimize_in(root, GOOD, name="off")
    assert len(Library(chosen).entries()) == 1


def test_no_library_skips_reading_and_writing(tmp_path, library):
    seeded = library.append(entry(features=_features_of_project(tmp_path / "a"), lesson="SEEDED LESSON"))
    result, llm = optimize_in(tmp_path / "a", GOOD, library=False)
    assert result.verdict == "improved" and library.entries() == [seeded]
    assert not any("SEEDED LESSON" in prompt for prompt in llm.prompts)


def _features_of_project(root: Path) -> dict:
    """The features `optimize` computes for the project written by `write_project` (train questions of the splits)."""
    project(root)
    from ragbench.datasets.splits import apply_split, load_splits

    dataset = load_dataset(root / "questions.jsonl", None)
    train = apply_split(dataset, load_splits(root / "splits.json"), "train")
    return features_of(profile_dataset(load_documents(root / "docs"), train))


# -- ragbench library -------------------------------------------------------------------------------------------------------------------------


def test_list_show_export_and_clear(library):
    first = library.append(entry("2026-01-01T00:00:00.000000+00:00", lesson="alpha lesson"))
    library.append(entry("2026-02-01T00:00:00.000000+00:00", lesson="beta lesson"))

    listed = invoke("library", "list")
    assert listed.exit_code == 0, listed.output
    assert first.id in listed.output and "hybrid_rerank" in listed.output

    as_json = json.loads(invoke("library", "list", "--json").output)
    assert [row["lesson"] for row in as_json] == ["alpha lesson", "beta lesson"]

    shown = invoke("library", "show", first.id[:6])
    assert shown.exit_code == 0 and "alpha lesson" in shown.output and "+0.40" in shown.output

    exported = invoke("library", "export")
    assert exported.exit_code == 0
    assert [json.loads(line)["lesson"] for line in exported.stdout.splitlines()] == ["alpha lesson", "beta lesson"]

    refused = invoke("library", "clear")
    assert refused.exit_code == 2 and len(library.entries()) == 2  # nothing to ask, no --yes
    cleared = invoke("library", "clear", "--yes")
    assert cleared.exit_code == 0 and "2" in cleared.output and library.entries() == []


def test_show_suggests_nothing_for_an_unknown_id_and_list_says_when_empty(library):
    assert "empty" in invoke("library", "list").output.lower()
    result = invoke("library", "show", "nope")
    assert result.exit_code == 2 and "nope" in result.output


def test_the_path_option_and_the_config_choose_the_library(tmp_path):
    other = Library(tmp_path / "other")
    other.append(entry(lesson="elsewhere"))
    assert "elsewhere" in invoke("library", "list", "--path", str(other.path)).output
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump({"run": {"name": "c"}, "dataset": {"documents_path": "d", "questions_path": "q.jsonl"}, "systems": [{"type": "bm25"}], "library": {"path": str(other.path)}}), encoding="utf-8")
    assert "elsewhere" in invoke("library", "list", "--config", str(config)).output


# -- auto -------------------------------------------------------------------------------------------------------------------------------------


def _auto_inputs(root: Path) -> tuple[Path, Path]:
    docs = root / "docs"
    docs.mkdir(parents=True)
    for i in range(1, 7):
        (docs / f"doc_{i:03d}.md").write_text(f"# Topic {i}\n\nThe unit {i} costs {i * 10} dollars per month and ships in quarter {i}.", encoding="utf-8")
    questions = root / "questions.jsonl"
    write_jsonl(questions, [{"id": f"q{i}", "question": f"How much does unit {i} cost per month?", "relevant_doc_ids": [f"doc_{i:03d}"]} for i in range(1, 7)])
    return docs, questions


def _auto_features(docs: Path, questions: Path) -> dict:
    return features_of(profile_dataset(load_documents(docs), load_dataset(questions, None)))


def test_auto_chooses_the_preset_that_holds_the_system_a_similar_collection_adopted(tmp_path, library):
    docs, questions = _auto_inputs(tmp_path)
    features = _auto_features(docs, questions)
    library.append(entry(features=features, fingerprint=fingerprint_of(features), system_type="decompose", lesson="Split multi-part questions."))

    result = invoke("auto", "--docs", str(docs), "--questions", str(questions), "--mock", "--output-dir", str(tmp_path / "out"))

    assert result.exit_code == 0, result.output
    (run_dir,) = [p for p in (tmp_path / "out").iterdir() if p.is_dir()]
    assert json.loads((run_dir / "auto.json").read_text(encoding="utf-8"))["settings"]["preset"] == "thorough"  # `decompose` is only in thorough
    matches = json.loads((run_dir / "library.json").read_text(encoding="utf-8"))
    assert matches["preset"] == "thorough" and matches["matches"][0]["lesson"] == "Split multi-part questions."
    assert "thorough" in result.output and "decompose" in result.output


def test_an_explicit_preset_is_never_replaced_by_the_library(tmp_path, library):
    docs, questions = _auto_inputs(tmp_path)
    features = _auto_features(docs, questions)
    library.append(entry(features=features, fingerprint=fingerprint_of(features), system_type="decompose"))

    result = invoke("auto", "--docs", str(docs), "--questions", str(questions), "--preset", "quick", "--mock", "--output-dir", str(tmp_path / "out"))

    assert result.exit_code == 0, result.output
    (run_dir,) = [p for p in (tmp_path / "out").iterdir() if p.is_dir()]
    assert json.loads((run_dir / "auto.json").read_text(encoding="utf-8"))["settings"]["preset"] == "quick"


def test_without_a_similar_entry_auto_keeps_the_standard_preset(tmp_path, library):
    docs, questions = _auto_inputs(tmp_path)
    far = {**FEATURES, "language": "de", "tokens": 9_000_000, "documents": 20_000, "label_free": True}
    library.append(entry(features=far, fingerprint=fingerprint_of(far), system_type="decompose"))

    result = invoke("auto", "--docs", str(docs), "--questions", str(questions), "--mock", "--output-dir", str(tmp_path / "out"))

    assert result.exit_code == 0, result.output
    (run_dir,) = [p for p in (tmp_path / "out").iterdir() if p.is_dir()]
    assert json.loads((run_dir / "auto.json").read_text(encoding="utf-8"))["settings"]["preset"] == "standard"
    assert not (run_dir / "library.json").exists()
