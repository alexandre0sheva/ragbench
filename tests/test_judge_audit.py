from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.evaluation.agreement import krippendorff_alpha
from ragbench.evaluation.answer_judge import SCORE_AXES
from ragbench.evaluation.judge_audit import KEY_FILE, SHEET_FILE, AuditError, score_audit, write_audit_sheet

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"
# Judge answer scores of the fixture run: two systems x six questions, spread over the low (<2), middle and high (>=4) bands.
SCORES = {"alpha": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "beta": [5.0, 5.0, 4.5, 4.0, 1.5, 3.0]}


def _row(system: str, index: int, score: float, panel: dict[str, float] | None = None) -> dict:
    metadata: dict = {"judge": "gpt-6-luna", "fallback": False}
    if panel is not None:
        metadata = {"judge": "panel", "fallback": False, "panel_scores": {ref: {**{a: s for a in SCORE_AXES}, "answer_score": s} for ref, s in panel.items()}}
    return {
        "system": system,
        "question_id": f"q{index}",
        "question": f"Question {index}?",
        "reference_answer": f"Answer {index}.",
        "answer": f"{system} says {index}.",
        "retrieved_contexts": [
            {"doc_id": f"doc_{index}", "text_preview": f"Context of {index}.", "in_context": True},
            {"doc_id": "doc_x", "text_preview": "Not shown to the generator.", "in_context": False},
        ],
        "answer_judge": {**{axis: score for axis in SCORE_AXES}, "answer_score": score, "metadata": metadata},
        "error": None,
    }


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "fixture_run"
    run.mkdir()
    rows = [_row(system, index, score) for system, scores in SCORES.items() for index, score in enumerate(scores)]
    rows.append({**_row("beta", 9, 0.0), "error": {"type": "Boom", "message": "x"}})  # a failed question is never audited
    (run / "per_question_results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (run / "run_summary.json").write_text(json.dumps({"run_id": "fixture_run"}), encoding="utf-8")
    return run


def _read_sheet(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fill(run: Path, score_for, target: Path | None = None) -> Path:
    """Fill the sheet as a human would: `score_for(system, question_id, axis)` gives each cell (None leaves it blank)."""
    key = json.loads((run / KEY_FILE).read_text(encoding="utf-8"))["items"]
    rows = _read_sheet(run / SHEET_FILE)
    for row in rows:
        item = key[row["audit_id"]]
        for axis in SCORE_AXES:
            value = score_for(item["system"], item["question_id"], axis)
            row[axis] = "" if value is None else str(value)
    target = target or run / "filled.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return target


def _judge_score(system: str, question_id: str) -> float:
    return SCORES[system][int(question_id[1:])]


# -- the sheet ------------------------------------------------------------------------------------


def test_the_sheet_is_a_blind_stratified_sample(run_dir):
    sheet = write_audit_sheet(run_dir, sample=6, seed=0)
    rows = _read_sheet(sheet.sheet_path)
    assert len(rows) == 6 == sheet.n
    assert list(rows[0]) == ["audit_id", "question", "reference_answer", "context", "answer", *SCORE_AXES, "notes"]
    assert all(row[axis] == "" for row in rows for axis in SCORE_AXES)  # nothing to anchor on: no judge score, no system name
    assert "Context of" in rows[0]["context"] and "Not shown" not in rows[0]["context"]  # only what the generator was given
    key = json.loads(sheet.key_path.read_text(encoding="utf-8"))["items"]
    picked = [key[row["audit_id"]] for row in rows]
    assert {item["system"] for item in picked} == {"alpha", "beta"}
    bands = {("low" if s < 2 else "high" if s >= 4 else "mid") for s in (_judge_score(i["system"], i["question_id"]) for i in picked)}
    assert bands == {"low", "mid", "high"}
    assert all(i["question_id"] != "q9" for i in picked)


def test_the_same_seed_draws_the_same_sheet(run_dir, tmp_path):
    first = write_audit_sheet(run_dir, sample=5, seed=3).sheet_path.read_text(encoding="utf-8")
    again = write_audit_sheet(run_dir, sample=5, seed=3, overwrite=True).sheet_path.read_text(encoding="utf-8")
    assert first == again


def test_a_sample_larger_than_the_run_takes_every_answer(run_dir):
    assert write_audit_sheet(run_dir, sample=50).n == 12


def test_an_existing_sheet_is_never_overwritten_by_accident(run_dir):
    write_audit_sheet(run_dir, sample=4)
    with pytest.raises(AuditError, match="already"):
        write_audit_sheet(run_dir, sample=4)


# -- scoring the labels ---------------------------------------------------------------------------


def test_a_human_who_agrees_with_the_judge_gives_alpha_one(run_dir):
    write_audit_sheet(run_dir, sample=12)
    result = score_audit(run_dir, _fill(run_dir, lambda system, qid, axis: _judge_score(system, qid)))
    assert result["n_labeled"] == 12
    assert result["axes"]["answer_score"] == {"n": 12, "alpha": 1.0, "mean_abs_diff": 0.0, "bias": 0.0}
    assert result["verdict_agreement"] == 1.0
    assert (run_dir / "judge_audit.md").exists() and json.loads((run_dir / "judge_audit.json").read_text())["n_labeled"] == 12


def test_a_judge_that_is_lenient_on_one_system_shows_up_as_its_bias(run_dir):
    write_audit_sheet(run_dir, sample=12)
    harsh = {"alpha": 1.0, "beta": 0.0}  # the human scores alpha one point below the judge
    result = score_audit(run_dir, _fill(run_dir, lambda system, qid, axis: max(0.0, _judge_score(system, qid) - harsh[system])))
    expected_pairs = [[s, max(0.0, s - harsh[system])] for system, scores in SCORES.items() for s in scores]
    assert result["axes"]["correctness"]["alpha"] == pytest.approx(krippendorff_alpha(expected_pairs))
    assert result["by_system"]["alpha"]["bias"] == pytest.approx(5 / 6)  # 0.0 cannot go lower
    assert result["by_system"]["beta"]["bias"] == 0.0
    assert result["axes"]["answer_score"]["bias"] == pytest.approx(5 / 12)


def test_blank_cells_are_not_labels_and_answer_score_needs_all_five_axes(run_dir):
    write_audit_sheet(run_dir, sample=12)
    result = score_audit(run_dir, _fill(run_dir, lambda system, qid, axis: _judge_score(system, qid) if axis == "correctness" or system == "alpha" else None))
    assert result["axes"]["correctness"]["n"] == 12 and result["axes"]["faithfulness"]["n"] == 6 and result["axes"]["answer_score"]["n"] == 6


@pytest.mark.parametrize("bad", ["7", "abc"])
def test_a_label_outside_the_scale_is_refused(run_dir, bad):
    write_audit_sheet(run_dir, sample=4)
    filled = _fill(run_dir, lambda *_: 3)
    text = filled.read_text(encoding="utf-8").replace(",3,", f",{bad},", 1)
    filled.write_text(text, encoding="utf-8")
    with pytest.raises(AuditError, match="0 to 5"):
        score_audit(run_dir, filled)


def test_labels_for_an_answer_that_is_not_in_the_sheet_are_refused(run_dir):
    write_audit_sheet(run_dir, sample=4)
    filled = _fill(run_dir, lambda *_: 3)
    rows = _read_sheet(filled)
    rows[0]["audit_id"] = "A999"
    with filled.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(AuditError, match="A999"):
        score_audit(run_dir, filled)


def test_a_panel_run_also_reports_each_judge_against_the_human(tmp_path):
    run = tmp_path / "panel_run"
    run.mkdir()
    rows = [_row("alpha", i, s, panel={"judge-a": s, "judge-b": 5.0 - s}) for i, s in enumerate([0.0, 1.0, 4.0, 5.0])]
    (run / "per_question_results.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    write_audit_sheet(run, sample=4)
    by_judge = score_audit(run, _fill(run, lambda system, qid, axis: [0.0, 1.0, 4.0, 5.0][int(qid[1:])]))["by_judge"]
    assert by_judge["judge-a"]["alpha"] == 1.0 and by_judge["judge-b"]["alpha"] < 0 and by_judge["judge-a"]["n"] == 4


# -- the command ----------------------------------------------------------------------------------


def test_judge_audit_round_trip_through_the_cli_links_the_audit_from_the_report(tmp_path):
    from ragbench.evaluation.evaluator import run_benchmark

    config = {
        "run": {"name": "audit", "output_dir": str(tmp_path / "results")},
        "dataset": {"documents_path": str(DEMO / "docs"), "questions_path": str(DEMO / "questions.jsonl"), "qrels_path": str(DEMO / "qrels.jsonl")},
        "systems": [{"type": "bm25", "name": "bm25"}],
        "evaluation": {"max_questions": 6, "max_workers": 1},
    }
    path = tmp_path / "audit.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    out = run_benchmark(path, force_mock=True)
    runner = CliRunner()

    sheet = runner.invoke(app, ["judge-audit", str(out), "--sample", "4"])
    assert sheet.exit_code == 0, sheet.output
    assert SHEET_FILE in sheet.output and len(_read_sheet(out / SHEET_FILE)) == 4
    assert runner.invoke(app, ["judge-audit", str(out), "--sample", "4"]).exit_code == 2  # the sheet is there already

    filled = _fill(out, lambda *_: 4, target=tmp_path / "filled.csv")
    scored = runner.invoke(app, ["judge-audit", str(out), "--labels", str(filled)])
    assert scored.exit_code == 0, scored.output
    assert "judge_audit.md" in scored.output and (out / "judge_audit.md").exists()
    assert "judge_audit.md" in (out / "report.html").read_text(encoding="utf-8")
