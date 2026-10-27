"""Publish a finished run as a benchmark: copy its small result files into docs/benchmarks/, rebuild its reports there, render the pages.

    python scripts/publish_benchmark.py multihop-rag --run results/all_solutions_multihop \\
        [--harnesses results/all_solutions_harnesses_multihop] [--models results/all_solutions_models_multihop] \\
        [--optimize results/optimize_hybrid_rerank_multihop]

writes `docs/benchmarks/<name>-run/` (and `<name>-harnesses-run/`, `<name>-models-run/`, `<name>-optimize/`), then
`python scripts/check_results.py --write`. Raw per-question rows (questions, answers, passages) are not copied: `ragbench report`
rebuilds the leaderboard, the recommendation and report.html from what is, and `patterns.json` keeps the per-question aggregates the
page quotes (counts and means only, no question or answer text).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_results  # noqa: E402

RUN_FILES = (
    "config.yaml", "metrics_summary.csv", "significance.csv", "stats.json", "run_summary.json", "run_manifest.json", "recommendation.json",
    "recommendation.md", "winner.yaml", "pareto.json", "stages.json", "system_runtime.csv", "tool_usage.csv", "routes.csv", "contrast_summary.json",
)
OPTIMIZE_FILES = ("optimization.md", "ledger.json", "optimized.yaml")
KEPT = ("corrections.json",)  # recorded by hand after the run (with their source), so a re-publish keeps them, like the screenshots


def copy(source: Path, dest: Path, names: tuple[str, ...]) -> list[str]:
    if not (source / names[0]).is_file():
        raise SystemExit(f"{source} has no {names[0]}: not a finished run")
    if dest.exists():  # a re-publish replaces the run's files but keeps the report screenshots and corrections recorded by hand
        for path in dest.iterdir():
            if path.suffix != ".png" and path.name not in KEPT:
                shutil.rmtree(path) if path.is_dir() else path.unlink()
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in names:
        if (source / name).is_file():
            shutil.copyfile(source / name, dest / name)
            copied.append(name)
    return copied


def copy_dataset_record(run_dir: Path, source: Path) -> None:
    """The dataset's `hub.json` (source, pinned revision, license, subset seed) next to the run, when the dataset came from `ragbench datasets pull`."""
    import yaml

    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    documents = Path((config.get("dataset") or {}).get("documents_path") or "")
    record = (documents if documents.is_absolute() else ROOT / documents).parent / "hub.json"
    if record.is_file():
        shutil.copyfile(record, run_dir / "hub.json")


GOOD_ANSWER = 4.0  # an answer score at or above this is a good answer, for counting the questions no system answers well


def write_patterns(source: Path, dest: Path) -> None:
    """`patterns.json`: what the page says about the questions, computed from the raw rows that are not published.

    The questions per category; each system's mean answer score on the answerable questions; how many questions no system answers
    well; the mean answer score by how much of the evidence the top 5 held (answerable questions, retrieving systems); the failure
    types over every answer.
    """
    raw = source / "per_question_results.jsonl"
    if not raw.is_file():
        return
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    categories = {row["question_id"]: row["category"] for row in rows}
    scored = [row for row in rows if (row.get("answer_judge") or {}).get("answer_score") is not None]
    answerable: dict[str, list[float]] = defaultdict(list)
    per_question: dict[str, list[float]] = defaultdict(list)
    buckets: dict[str, list[float]] = {"none": [], "some": [], "all": []}
    for row in scored:
        value = row["answer_judge"]["answer_score"]
        per_question[row["question_id"]].append(value)
        if row["answerable"]:
            answerable[row["system"]].append(value)
            recall = (row.get("retrieval_metrics") or {}).get("recall@5")
            if recall is not None:
                buckets["none" if recall == 0 else "all" if recall == 1 else "some"].append(value)
    unsolved = [question for question, values in per_question.items() if max(values) < GOOD_ANSWER]
    patterns = {
        "questions": dict(sorted(Counter(categories.values()).items())),
        "answerable_score": {system: mean(values) for system, values in sorted(answerable.items())},
        "hard": {
            "threshold": GOOD_ANSWER,
            "none": len(unsolved),
            "none_by_category": dict(sorted(Counter(categories[question] for question in unsolved).items())),
        },
        "recall_buckets": [{"recall": name, "answers": len(values), "answer_score": mean(values) if values else None} for name, values in buckets.items()],
        "failures": dict(Counter(row["failure_type"] for row in rows).most_common()),
    }
    (dest / "patterns.json").write_text(json.dumps(patterns, indent=2) + "\n", encoding="utf-8")


def rebuild(run_dir: Path) -> None:
    from ragbench.reporting.regenerate import regenerate_reports

    _, notes = regenerate_reports(run_dir)
    for note in notes:
        print(f"  note: {note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="the benchmark's name: docs/benchmarks/<name>.md, scripts/benchmark_templates/<name>.md.j2")
    parser.add_argument("--run", type=Path, required=True, help="the main run directory")
    parser.add_argument("--harnesses", type=Path, help="a run directory of systems run on fewer questions (the agent harnesses)")
    parser.add_argument("--models", type=Path, help="a model-comparison run directory")
    parser.add_argument("--optimize", type=Path, help="a `ragbench optimize` output folder")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    benchmarks = args.root / check_results.BENCHMARKS
    runs = ((args.run, f"{args.name}-run"), (args.harnesses, f"{args.name}-harnesses-run"), (args.models, f"{args.name}-models-run"))
    for source, folder in runs:
        if source is not None:
            copied = copy(source, benchmarks / folder, RUN_FILES)
            copy_dataset_record(benchmarks / folder, source)
            write_patterns(source, benchmarks / folder)
            rebuild(benchmarks / folder)
            print(f"{folder}: {', '.join(copied)} + rebuilt reports")
    if args.optimize is not None:
        copy(args.optimize, benchmarks / f"{args.name}-optimize", OPTIMIZE_FILES)
        print(f"{args.name}-optimize: {', '.join(OPTIMIZE_FILES)}")
    return check_results.main(["--write", "--root", str(args.root)])


if __name__ == "__main__":
    raise SystemExit(main())
