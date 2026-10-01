"""`auto`: from a folder of documents to a decision in one command."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml
from rich.markup import escape

from ragbench.cli.common import console, fail, load_optional_config, money
from ragbench.cli.dataset import run_generation
from ragbench.cli.run import execute_benchmark
from ragbench.config.presets import PRESET_NAMES
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.validation import validate_dataset
from ragbench.documents.loaders import DocumentLoadError, load_documents
from ragbench.errors import RagbenchError
from ragbench.models.errors import MissingExtraError
from ragbench.utils.env import load_project_env
from ragbench.utils.text import estimate_tokens

commands = typer.Typer()


@commands.command()
def auto(
    docs: Path | None = typer.Option(None, "--docs", help="Your documents: a folder, or one file."),
    questions: Path | None = typer.Option(None, "--questions", help="Your questions JSONL. Without it, questions are written from your documents (flagged needs_review)."),
    qrels: Path | None = typer.Option(None, "--qrels", help="Optional qrels JSONL for --questions."),
    preset: str = typer.Option("standard", "--preset", help=f"Which systems to compare: {', '.join(PRESET_NAMES)}."),
    profile: str = typer.Option("balanced", "--profile", help="What the recommendation optimizes: balanced, max_quality, cheapest_acceptable or lowest_latency."),
    n_questions: int = typer.Option(50, "--n-questions", min=1, help="How many questions to write when you have none."),
    seed: int = typer.Option(0, "--seed", help="Seed for writing questions."),
    max_cost: float | None = typer.Option(None, "--max-cost", min=0, help="Total dollars to spend (writing questions plus the run). The run stops once it is reached."),
    model: str | None = typer.Option(None, "--model", help="Model ref that writes the questions (default: the default generator model)."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Take models, providers, pricing, evaluation and selection settings from this config (its systems and dataset are not used)."),
    output_dir: Path = typer.Option(Path("results"), "--output-dir", help="Where the run directory is created."),
    resume: Path | None = typer.Option(None, "--resume", help="Continue an earlier auto run in this directory: finished systems are kept, the rest are run."),
    mock: bool = typer.Option(False, "--mock", help="Force local mock mode: nothing is paid for and the scores only validate the pipeline."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Do not read or write the persistent disk cache."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before spending more than evaluation.cost_confirm_threshold_usd."),
    open_report: bool = typer.Option(False, "--open", help="Open report.html in your browser when done."),
) -> None:
    """From a folder of documents to a decision: profile them, write questions if you have none, run the preset, and recommend a system.

    Writes everything to one run directory (questions, results, recommendation.md, winner.yaml, report.html). If it stops (budget, a crash, Ctrl-C),
    `--resume RUN_DIR` continues without paying again for the systems that finished.
    """
    import webbrowser

    from ragbench.datasets.profile import profile_dataset
    from ragbench.datasets.synthesis import parse_mix
    from ragbench.models.defaults import DEFAULT_GENERATOR_MODEL
    from ragbench.utils.jsonl import write_jsonl
    from ragbench.workflows import auto as workflow

    load_project_env(resume or docs)
    try:
        if resume is not None:
            given = [flag for flag, value in (("--docs", docs), ("--questions", questions), ("--qrels", qrels), ("--config", config)) if value is not None]
            if given:
                raise workflow.AutoError(f"{', '.join(given)} come from the run being resumed; drop them (only --max-cost, --mock, --yes, --no-cache and --open can change).")
            settings, generation_cost = workflow.read_state(resume)
            run_dir = resume
            if max_cost is not None:
                settings.max_cost = max_cost
            settings.mock = settings.mock or mock
        else:
            if docs is None:
                raise workflow.AutoError("Pass --docs (a folder of documents) to start, or --resume RUN_DIR to continue a run.")
            workflow.check_choices(preset, profile)
            settings = workflow.AutoSettings(
                docs=str(docs.resolve()),
                questions=str(questions.resolve()) if questions else None,
                qrels=str(qrels.resolve()) if qrels else None,
                preset=preset,
                profile=profile,
                n_questions=n_questions,
                seed=seed,
                max_cost=max_cost,
                model=model,
                config=str(config.resolve()) if config else None,
                mock=mock,
            )
            generation_cost = 0.0
            run_dir = None  # created below, once the inputs are known to be readable
        documents = load_documents(Path(settings.docs))
        loaded = load_optional_config(Path(settings.config)) if settings.config else None
        if settings.questions is not None and not Path(settings.questions).is_file():
            raise workflow.AutoError(f"Questions file not found: {settings.questions}")
        shares = parse_mix(None)
    except (workflow.AutoError, ValueError, FileNotFoundError, DocumentLoadError, MissingExtraError) as exc:
        raise fail(str(exc)) from exc
    if run_dir is None:
        run_dir = workflow.new_run_dir(output_dir)
    workflow.write_state(run_dir, settings, generation_cost)
    console.print(f"Run directory: [bold]{run_dir}[/bold]" + ("  [dim](resuming)[/dim]" if resume else ""), soft_wrap=True)
    total_tokens = sum(estimate_tokens(document.text) for document in documents)
    console.print(f"{len(documents)} documents, ~{total_tokens:,} tokens")

    # 1. Questions: yours, ones an earlier attempt already wrote, or new synthetic ones.
    if settings.questions is not None:
        questions_path = Path(settings.questions)
    else:
        questions_path = run_dir / workflow.QUESTIONS_NAME
        if questions_path.exists():
            console.print(f"[dim]Using the questions already written to {questions_path}.[/dim]")
        else:
            result, live, _ = run_generation(
                documents,
                n=settings.n_questions,
                shares=shares,
                ref=settings.model or DEFAULT_GENERATOR_MODEL,
                seed=settings.seed,
                loaded=loaded,
                mock=settings.mock,
                max_cost=settings.max_cost,
                no_cache=no_cache,
                yes=yes,
            )
            if not result.rows:
                raise fail("No question could be written. " + " ".join(result.warnings))
            write_jsonl(questions_path, result.rows)
            generation_cost = result.cost_usd
            workflow.write_state(run_dir, settings, generation_cost)
            console.print(f"Wrote {len(result.rows)} questions to {questions_path} ({'live' if live else 'mock templates'}, {money(result.cost_usd)}); read them: every one is flagged `needs_review`.")
            for warning in result.warnings:
                console.print(f"  [yellow]•[/yellow] {escape(warning)}", soft_wrap=True)
            if result.stopped_by_budget:
                raise fail(f"Stopped at --max-cost while writing questions. Raise it and run `ragbench auto --resume {run_dir}`.", code=1)

    # 2. What the data looks like, and what is likely to distort the comparison.
    try:
        dataset = load_dataset(questions_path, Path(settings.qrels) if settings.qrels else None)
    except (FileNotFoundError, ValueError) as exc:
        raise fail(str(exc)) from exc
    profile_result = profile_dataset(documents, dataset)
    console.print(
        f"{profile_result.questions.count} questions ({profile_result.questions.answerable} answerable) · {profile_result.qrels.labeled_questions} with relevance labels"
        + (" · [yellow]label-free: retrieval metrics will be blank[/yellow]" if dataset.label_free else "")
    )
    for warning in [*validate_dataset(documents, dataset), *profile_result.warnings][:6]:
        console.print(f"  [yellow]•[/yellow] {escape(warning)}", soft_wrap=True)

    # 3. The run: estimate, confirm, run every system, recommend. Systems an earlier attempt finished are not run again.
    try:
        existing = run_dir / "config.yaml"
        if resume is not None and existing.exists():
            raw = yaml.safe_load(existing.read_text(encoding="utf-8")) or {}
            if max_cost is not None:
                raw["evaluation"] = {**(raw.get("evaluation") or {}), "max_cost_usd": workflow.remaining_budget(settings.max_cost, generation_cost)}
        else:
            raw = workflow.build_config(settings, run_dir, questions_path, max_cost_usd=workflow.remaining_budget(settings.max_cost, generation_cost))
    except (workflow.AutoError, ValueError, FileNotFoundError) as exc:
        raise fail(str(exc)) from exc
    try:
        finished = execute_benchmark(existing, settings.mock, None, "Auto run complete.", use_cache=not no_cache, raw_config=raw, yes=yes, run_dir=run_dir)
    except (typer.Exit, RagbenchError) as stopped:
        console.print(f"[bold]Resume with:[/bold] ragbench auto --resume {run_dir}" + (" --max-cost <more>" if settings.max_cost else ""), soft_wrap=True)
        raise stopped

    # 4. The deliverables.
    files = workflow.winner_files(finished)
    recommendation_path = finished / "recommendation.json"
    winner = json.loads(recommendation_path.read_text(encoding="utf-8")).get("winner") if recommendation_path.exists() else None
    console.print(f"\n[bold]{'Deploy ' + escape(winner) if winner else 'No system meets your constraints'}[/bold]")
    for label, key in (("Runnable config", "winner"), ("Report", "report"), ("Reasons", "recommendation")):
        if key in files:
            console.print(f"  {label}: {files[key]}", soft_wrap=True)
    if open_report and "report" in files:
        webbrowser.open(files["report"].resolve().as_uri())
