"""The evidence tree a harness proposer reads: what a system did on the train questions of a finished run, written as files (Harness²'s
`context.py`).

    MISSION.md      what to change and how a proposal is judged useful (never names the measurement)
    task.md         the documents and the visible questions, from the dataset profile
    harness/        each system's effective harness (`<name>.yaml`) and the fields the target may change (`fields.md`)
    rollouts/       one file per train question: the target's trace, context and answer (and, with `labels`, the judge's grade)
    contrasts/      per other system, the questions where exactly one of the two was right and where their traces part
    failures.md     the target's failures (judged with `labels`, observable otherwise: errors, refusals, nothing retrieved)
    library/        lessons from earlier work (filled by the experience library)
    manifest.json   what was built from what (read by the proposer, not shown to it)

Feedback modes: `labels` shows the judge's grades and the reference answers of the train questions; `contrast` shows behaviour only
(which questions two systems disagree on and where their traces part, never who was right: the Harness² setting); `none` shows the
rollouts alone. Only the train (or dev) questions of the run's splits are read. Every passage taken from the run is checked by the
leakage guard and withheld when it holds dev or test text (a document can quote a held-out answer); the finished tree is checked again
as a whole and deleted if anything slipped through.
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import yaml

from ragbench.config.schema import ExperimentConfig, SystemConfig
from ragbench.datasets.leakage import LeakageError, LeakageGuard
from ragbench.datasets.loader import configured_splits_path, load_dataset
from ragbench.datasets.profile import DatasetProfile, profile_dataset
from ragbench.datasets.schema import Dataset
from ragbench.datasets.splits import SplitError, Splits, apply_split, load_splits
from ragbench.documents.loaders import load_dataset_documents
from ragbench.errors import UsageError
from ragbench.evaluation.contrast import build_contrasts
from ragbench.harness.apply import effective_harness, knob_targets
from ragbench.harness.schema import HarnessPatch
from ragbench.utils.jsonl import read_jsonl
from ragbench.utils.text import truncate

Feedback = Literal["labels", "contrast", "none"]
FEEDBACK_MODES: tuple[str, ...] = ("labels", "contrast", "none")
READABLE_SPLITS = ("train", "dev")
SPLIT_PARTS = ("train", "dev", "test")
MANIFEST = "manifest.json"
WITHHELD = "[withheld: held-out text]"
# Harness² keeps the measurement out of what the method reads, so it optimizes the task and not the grader: the framing files never use these.
FORBIDDEN_WORDS = ("benchmark", "score", "rubric")
# What a `labels` tree writes and a `contrast` / `none` tree must not hold (`assert_no_verdicts`).
VERDICT_MARKERS = ("Judge grade:", "Reference answer:", "Failure type:", "Right: ", "Judge's note:")
CONTEXT_CHARS = 400
STEP_CHARS = 300
ANSWER_CHARS = 1200


class EvidenceError(UsageError):
    """The evidence cannot be built as asked (no split, unknown system, no train questions in the run)."""


# -- the run ----------------------------------------------------------------------------------------------------------------------------------


def run_config(run_dir: Path) -> ExperimentConfig:
    path = run_dir / "config.yaml"
    if not (run_dir / "per_question_results.jsonl").is_file() or not path.is_file():
        raise EvidenceError(f"{run_dir} is not a finished run directory (it needs per_question_results.jsonl and config.yaml).")
    return ExperimentConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def run_splits(run_dir: Path, config: ExperimentConfig) -> Splits:
    """The splits the run's dataset uses; refused when they changed since a split run (its questions may have moved between parts)."""
    try:
        splits = load_splits(configured_splits_path(config.dataset))
    except SplitError as exc:
        raise EvidenceError(f"Evidence is built from the train split, and this run's dataset has none: {exc}", hint="Create one with `ragbench split`.") from exc
    manifest_path = run_dir / "run_manifest.json"
    recorded = (json.loads(manifest_path.read_text(encoding="utf-8")).get("split") if manifest_path.is_file() else None) or {}
    if recorded.get("hash") and recorded["hash"] != splits.digest:
        raise EvidenceError("The splits file changed since the run was made, so its train questions are not the run's. Re-run on the current split.")
    return splits


def evidence_guard(dataset: Dataset, splits: Splits, split: str) -> LeakageGuard:
    """The guard over every question outside `split`: a method reading the train split may see neither dev nor test."""
    return LeakageGuard.for_splits(dataset, splits, held_out=tuple(part for part in SPLIT_PARTS if part != split))


def guard_for_run(run_dir: Path, split: Literal["train", "dev"] = "train") -> LeakageGuard:
    config = run_config(run_dir)
    return evidence_guard(load_dataset(config.dataset.questions_path, config.dataset.qrels_path), run_splits(run_dir, config), split)


def _pick_system(config: ExperimentConfig, rows: list[dict[str, Any]], name: str | None) -> SystemConfig:
    systems = {system.resolved_name: system for system in config.systems}
    names = [system for system in dict.fromkeys(str(row["system"]) for row in rows) if system in systems]
    if name is None:
        if len(names) != 1:
            raise EvidenceError(f"Name the system to build evidence for: the run has {', '.join(names)}.")
        name = names[0]
    if name not in names:
        close = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise EvidenceError(f"The run has no system named '{name}'.{hint} (Systems: {', '.join(names)})")
    return systems[name]


# -- writing ----------------------------------------------------------------------------------------------------------------------------------


class _Writer:
    """Writes the tree, passing every passage taken from the run through the guard: a passage holding held-out text is withheld."""

    def __init__(self, root: Path, guard: LeakageGuard):
        self.root = root
        self.guard = guard
        self.withheld = 0

    def clean(self, text: Any, limit: int | None = None) -> str:
        value = "" if text is None else str(text)
        value = truncate(value, limit) if limit else value
        if value and self.guard.scan_text(value):
            self.withheld += 1
            return WITHHELD
        return value

    def write(self, name: str, text: str) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value) or "_"


def _grade(row: dict[str, Any]) -> float | None:
    value = (row.get("answer_judge") or {}).get("answer_score")
    return float(value) if isinstance(value, int | float) else None


def _mission(system: SystemConfig, feedback: str, others: list[str]) -> str:
    seen = {
        "labels": "what the system did on each question, its answer, the grade a judge gave the answer (0 to 5) and the reference answer",
        "contrast": "what the system did on each question and its answer; no grades and no reference answers",
        "none": "what the system did on each question and its answer; no grades, no reference answers and no comparisons",
    }[feedback]
    contrasts = {
        "labels": "the questions where exactly one of this system and another was right, which one, and the step where their traces part",
        "contrast": "the questions where this system and another reached different outcomes, and the step where their traces part. Which one was "
        "right is not shown: read both and decide which behaviour to keep",
        "none": "empty in this mode",
    }[feedback]
    other_files = f" `harness/<name>.yaml` holds the harness of each other system in the run ({', '.join(others)}) for comparison." if others else ""
    return f"""# Mission

Improve how the system `{system.resolved_name}` (type `{system.type}`) answers questions about the documents described in `task.md`, by
changing its harness: the prompt texts and settings listed in `harness/fields.md`. The model behind it stays the same.

Each request asks for one change of a given kind:

- **INV**: a general change that would help this kind of system on any document collection. It must not mention names, terms or topics
  of this collection.
- **DOM**: a change made for this collection: its vocabulary, its kinds of questions, the shape of its documents.

A good change fixes a pattern that recurs across many questions, not one question. Prefer a small, clear change over many edits at once.

## What is here

- `task.md`: the documents and the questions you can see.
- `harness/{system.resolved_name}.yaml`: the system's harness now; `harness/fields.md`: what may be changed, with limits.{other_files}
- `rollouts/`: one file per question: {seen}.
- `contrasts/`: {contrasts}.
- `failures.md`: {"the questions the system got wrong, by kind of failure" if feedback == "labels" else "what went visibly wrong: errors, refusals, nothing retrieved"}.
- `library/`: lessons from earlier work on similar collections.
"""


def _task(profile: DatasetProfile, split: str) -> str:
    documents, questions, qrels = profile.documents, profile.questions, profile.qrels

    def counts(values: dict[str, int]) -> str:
        return ", ".join(f"`{name}` {count}" for name, count in sorted(values.items(), key=lambda item: (-item[1], item[0])))

    lines = [
        "# The task",
        "",
        "A question-answering system answers questions about a collection of documents. It finds passages and writes an answer from them.",
        "",
        "## The documents",
        "",
        f"- {documents.count} documents, about {documents.total_tokens:,} tokens in all "
        f"(per document: median {documents.tokens.p50:,.0f}, 95th percentile {documents.tokens.p95:,.0f}, longest {documents.tokens.max:,.0f}).",
        f"- Language: {documents.language}.",
        "",
        f"## The questions ({split} part, the only questions you see)",
        "",
        f"- {questions.count} questions, median {questions.words.p50:.0f} words.",
        f"- Kinds: {counts(questions.categories)}.",
    ]
    if set(questions.difficulties) != {"unknown"}:
        lines.append(f"- Difficulty: {counts(questions.difficulties)}.")
    if set(questions.answer_types) != {"unknown"}:
        lines.append(f"- Expected answer types: {counts(questions.answer_types)}.")
    lines.append(f"- {questions.unanswerable} of them have no answer in the documents; the right reply to those says so.")
    if not qrels.label_free:
        lines.append(f"- {qrels.labeled_questions} name the documents that hold their answer.")
    return "\n".join(lines)


def field_names(system: SystemConfig, current: dict[str, Any]) -> list[str]:
    """The harness fields `system` may change: its prompt texts, `few_shot` (with a system prompt) and its knobs."""
    import ragbench.rag_systems  # noqa: F401  (registers the built-in systems)
    from ragbench.registry import SYSTEMS

    spec = getattr(SYSTEMS.mapping.get(system.type), "spec", None)
    names = [name for name in current if name != "few_shot"]
    if "system_prompt" in current:
        names.append("few_shot")
    return names + [knob for knob in (knob_targets(spec) if spec is not None else {}) if knob not in names]


def _fields(names: list[str]) -> str:
    lines = [
        "# What may be changed",
        "",
        "A change lists only the fields it sets; every other field keeps its current value. An empty text drops that rule from the prompt.",
        "",
    ]
    for name in names:
        description = HarnessPatch.model_fields[name].description or ""
        lines.append(f"- `{name}`: {description}")
    return "\n".join(lines)


def _step_lines(writer: _Writer, steps: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for index, step in enumerate(steps, start=1):
        latency = step.get("latency_ms")
        took = f" ({latency:.0f} ms)" if isinstance(latency, int | float) else ""
        lines.append(f"{index}. {step.get('kind')} `{step.get('name')}`{took}")
        if step.get("input_preview"):
            lines.append(f"   - in: {writer.clean(step['input_preview'], STEP_CHARS)}")
        if step.get("output_preview"):
            lines.append(f"   - out: {writer.clean(step['output_preview'], STEP_CHARS)}")
        docs = (step.get("metadata") or {}).get("doc_ids")
        if isinstance(docs, list) and docs:
            lines.append(f"   - documents: {', '.join(str(doc) for doc in docs[:20])}")
    return lines


def _rollout(writer: _Writer, question_id: str, trials: list[dict[str, Any]], feedback: str) -> str:
    first = trials[0]
    lines = [f"# Question {question_id}", "", f"Question: {writer.clean(first.get('question'))}", f"Kind: `{first.get('category')}`"]
    if feedback == "labels":
        lines.append(f"Reference answer: {writer.clean(first.get('reference_answer'))}")
    for number, row in enumerate(trials, start=1):
        lines += ["", f"## Attempt {number}" if len(trials) > 1 else "## What the system did", ""]
        if row.get("error"):
            lines.append(f"Error: {writer.clean(row['error'], STEP_CHARS)}")
        lines += _step_lines(writer, row.get("steps") or [])
        contexts = [context for context in row.get("retrieved_contexts") or [] if context.get("in_context", True)]
        if contexts:
            lines += ["", "Passages the answer was written from:"]
            lines += [f"- [{context.get('doc_id')}] {writer.clean(context.get('text_preview'), CONTEXT_CHARS)}" for context in contexts]
        lines += ["", "Answer:", "", writer.clean(row.get("answer"), ANSWER_CHARS) or "(empty)"]
        if feedback == "labels":
            grade = _grade(row)
            lines += ["", f"Judge grade: {grade:.2f} / 5" if grade is not None else "Judge grade: none (not graded)"]
            lines.append(f"Failure type: {row.get('failure_type') or 'unknown'}")
            reasoning = (row.get("answer_judge") or {}).get("reasoning")
            if reasoning:
                lines.append(f"Judge's note: {writer.clean(reasoning, STEP_CHARS)}")
    return "\n".join(lines)


def _retrieved_nothing(row: dict[str, Any]) -> bool:
    """A retrieval step ran and no passage reached the answer (a system that never retrieves is not flagged)."""
    return any(step.get("kind") == "retrieve" for step in row.get("steps") or []) and not row.get("retrieved_contexts")


def _failures(writer: _Writer, by_question: dict[str, list[dict[str, Any]]], feedback: str) -> tuple[str, list[str]]:
    """`failures.md` and the questions it lists, in order."""
    listed: list[str] = []
    entries: list[str] = []
    kinds: Counter[str] = Counter()
    for question_id, trials in by_question.items():
        if feedback == "labels":
            problems = sorted({str(row.get("failure_type")) for row in trials if row.get("failure_type") not in (None, "no_failure")})
        else:
            problems = sorted(
                {
                    "error" if row.get("error") else "refused" if row.get("refused") else "nothing retrieved" if _retrieved_nothing(row) else ""
                    for row in trials
                }
                - {""}
            )
        if problems:
            listed.append(question_id)
            kinds.update(problems)
            entries.append(f"- [{question_id}](rollouts/{_safe_name(question_id)}.md) (`{trials[0].get('category')}`): {', '.join(problems)}")
    title = "# Failures" if feedback == "labels" else "# What went visibly wrong"
    if not entries:
        return f"{title}\n\nNothing on the questions shown.", listed
    summary = ", ".join(f"{kind} {count}" for kind, count in kinds.most_common())
    return "\n".join([title, "", f"{len(entries)} of {len(by_question)} questions: {summary}.", "", *entries]), listed


def _contrast_file(writer: _Writer, target: str, other: str, records: list[dict[str, Any]], summary: dict[str, Any], feedback: str) -> str:
    lines = [f"# This system (`{target}`) and `{other}`", ""]
    if feedback == "labels":
        lines.append(f"{len(records)} questions where exactly one was right: `{target}` {summary['a_only']}, `{other}` {summary['b_only']}.")
    else:
        lines.append(f"{len(records)} questions where the two reached different outcomes. Which one was right is not shown.")
    stages = Counter(record["diverges_at"]["stage"] for record in records if record["diverges_at"])
    if stages:
        lines.append(f"Where the traces first part: {', '.join(f'{stage} {count}' for stage, count in stages.most_common())}.")
    for record in records:
        lines += ["", f"## {record['question_id']} (`{record.get('category')}`)", "", f"Question: {writer.clean(record.get('question'))}"]
        if feedback == "labels":
            lines.append(f"Right: `{record['winner']}` (grade {record['scores'][record['winner']]:.2f} against {record['scores'][record['loser']]:.2f})")
            lines.append(f"Failure type of `{record['loser']}`: {record['loser_failure_type']}")
        divergence = record.get("diverges_at")
        if divergence:
            lines.append(f"The traces part at step {divergence['index'] + 1} ({divergence['stage']}): {divergence['reason'].replace('_', ' ')}.")
        lines += ["", f"This system (`{target}`): {writer.clean(record['answers'][target])}", "", f"`{other}`: {writer.clean(record['answers'][other])}"]
        steps = record.get("steps") or []
        if steps:
            lines += ["", f"| step | `{target}` | `{other}` | same |", "| --- | --- | --- | --- |"]
            for step in steps:
                a = f"{step['a']}: {writer.clean(step['a_output'])}" if step["a"] else "-"
                b = f"{step['b']}: {writer.clean(step['b_output'])}" if step["b"] else "-"
                lines.append(f"| {step['index'] + 1} | {a.replace('|', '/')} | {b.replace('|', '/')} | {'yes' if step['same'] else 'no'} |")
    return "\n".join(lines)


def _is_evidence(folder: Path) -> bool:
    try:
        manifest = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(manifest, dict) and {"system", "split", "feedback", "rollouts"} <= manifest.keys()


def _prepare(out_dir: Path) -> None:
    if out_dir.exists():
        if not _is_evidence(out_dir):
            raise EvidenceError(f"{out_dir} exists and is not an evidence folder (no {MANIFEST}); choose another `out_dir`.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)


def build_evidence(run_dir: Path, split: Literal["train", "dev"] = "train", *, system: str | None = None, feedback: Feedback = "contrast", out_dir: Path | None = None) -> Path:
    """Write the evidence tree of `system` (required when the run has several) from the `split` questions of a finished run; returns its folder.

    The default folder is `<run>/evidence/<system>-<feedback>`; an earlier tree there is replaced. Raises `LeakageError` (and leaves no
    folder) when held-out text reached the tree in a way no single passage shows, such as a harness prompt that holds a test question.
    """
    if split not in READABLE_SPLITS:
        raise EvidenceError(f"Evidence may be built from the train or dev split only, not '{split}': a method never reads the test split.")
    if feedback not in FEEDBACK_MODES:
        raise EvidenceError(f"Unknown feedback mode '{feedback}' (modes: {', '.join(FEEDBACK_MODES)}).")
    config = run_config(run_dir)
    splits = run_splits(run_dir, config)
    dataset = load_dataset(config.dataset.questions_path, config.dataset.qrels_path)
    guard = evidence_guard(dataset, splits, split)
    visible = splits.ids(split)
    rows = read_jsonl(run_dir / "per_question_results.jsonl")
    target = _pick_system(config, rows, system)
    name = target.resolved_name
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["system"] == name and str(row["question_id"]) in visible:
            by_question[str(row["question_id"])].append(row)
    if not by_question:
        raise EvidenceError(f"The run has no {split} questions for `{name}`: run it on `dataset.split: {split}` (or `all`) first.")

    current = effective_harness(target)
    fields = field_names(target, current)
    if not fields:
        raise EvidenceError(f"`{name}` (type {target.type}) has no harness RAGBench can change: its prompts and settings live outside RAGBench.")

    out = out_dir or run_dir / "evidence" / f"{_safe_name(name)}-{feedback}"
    _prepare(out)
    writer = _Writer(out, guard)
    try:
        others = [system.resolved_name for system in config.systems if system.resolved_name != name and any(row["system"] == system.resolved_name for row in rows)]
        writer.write("MISSION.md", _mission(target, feedback, others))
        visible_dataset = apply_split(dataset, splits, split)
        writer.write("task.md", _task(profile_dataset(load_dataset_documents(config.dataset, warnings=[]), visible_dataset), split))
        writer.write("harness/fields.md", _fields(fields))
        for system_config in config.systems:
            if system_config.resolved_name in (name, *others):
                writer.write(f"harness/{_safe_name(system_config.resolved_name)}.yaml", yaml.safe_dump(effective_harness(system_config), sort_keys=False, allow_unicode=True, width=120))
        for question_id, trials in by_question.items():
            writer.write(f"rollouts/{_safe_name(question_id)}.md", _rollout(writer, question_id, trials, feedback))
        failures, failed = _failures(writer, by_question, feedback)
        writer.write("failures.md", failures)
        contrasted: list[str] = []
        if feedback == "none" or not others:
            writer.write("contrasts/README.md", "# Contrasts\n\n" + ("None in this mode." if feedback == "none" else "Only one system ran: nothing to compare with."))
        else:
            contrasts = build_contrasts(run_dir, [(name, other) for other in others], question_ids=list(by_question))
            for summary in contrasts.summaries:
                other = summary["pair"][1]
                records = [record for record in contrasts.records if record["pair"] == summary["pair"]]
                contrasted += [record["question_id"] for record in records]
                writer.write(f"contrasts/{_safe_name(other)}.md", _contrast_file(writer, name, other, records, summary, feedback))
        writer.write("library/README.md", "# Library\n\nNo lessons from earlier work on similar collections yet.")
        focus = list(dict.fromkeys([*failed, *contrasted]))
        manifest = {
            "version": 1,
            "run_dir": str(run_dir),
            "system": name,
            "system_type": target.type,
            "split": split,
            "split_hash": splits.digest,
            "feedback": feedback,
            "questions": len(by_question),
            "rollouts": len(by_question),
            "contrast_questions": len(set(contrasted)),
            "withheld_passages": writer.withheld,
            "focus": focus,
            "fields": fields,
            "current_harness": current,
        }
        writer.write(MANIFEST, json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
        guard.assert_path_clean(out)
        if feedback != "labels":
            assert_no_verdicts(out)
    except (LeakageError, AssertionError):
        shutil.rmtree(out, ignore_errors=True)
        raise
    return out


def read_manifest(evidence: Path) -> dict[str, Any]:
    path = evidence / MANIFEST
    if not path.is_file():
        raise EvidenceError(f"{evidence} is not an evidence folder (no {MANIFEST}): build it with `build_evidence`.")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def assert_no_verdicts(evidence: Path) -> None:
    """The tripwire of the label-free modes (Harness²'s `assert_no_grader_verdicts`): no file may hold a grade, a reference answer, a failure
    type or a winner. Raises AssertionError naming the file."""
    for path in sorted(evidence.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        found = [marker for marker in VERDICT_MARKERS if marker in text]
        if found:
            raise AssertionError(f"{path.relative_to(evidence).as_posix()} holds a judge's verdict ({', '.join(found)}) in a label-free evidence tree.")
