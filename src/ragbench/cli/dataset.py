"""Commands that work on data: the demo set, profiling, generating questions, labeling by pooling, scaffolding and importing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from ragbench.cli.common import confirm_spend, console, fail, load_optional_config, money
from ragbench.cli.run import print_estimate
from ragbench.config.presets import PRESET_NAMES
from ragbench.config.schema import ExperimentConfig
from ragbench.datasets.demo_generator import write_demo_dataset
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.validation import validate_dataset
from ragbench.documents.loaders import DocumentLoadError, load_documents
from ragbench.evaluation.budget import BudgetGuard
from ragbench.models.errors import MissingExtraError, RagbenchModelError
from ragbench.utils.env import load_project_env

commands = typer.Typer()


@commands.command()
def demo(
    output: Path = typer.Option(Path("data/demo"), "--output", "-o", help="Directory where the demo dataset is written."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite demo files that differ from the bundled copy (by default they are kept and reported)."),
) -> None:
    """Create or verify the bundled demo dataset."""
    stats = write_demo_dataset(output, overwrite=overwrite)
    console.print(f"[green]Demo dataset ready at {output}[/green]")
    console.print(f"Documents: {stats['documents']} | Questions: {stats['questions']} | Qrels: {stats['qrels']}")
    if stats["modified"]:
        console.print(f"[yellow]{stats['modified']} file(s) differ from the bundled copy and were kept; use --overwrite to restore them.[/yellow]")


def _distribution(values: Any, unit: str = "") -> str:
    return f"min {values.min:,.0f} · median {values.p50:,.0f} · p95 {values.p95:,.0f} · max {values.max:,.0f}{unit}"


def _print_profile(profile: Any, label_free: bool) -> None:
    from ragbench.datasets.profile import DatasetProfile

    assert isinstance(profile, DatasetProfile)
    table = Table(title="Dataset Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    documents, questions, qrels = profile.documents, profile.questions, profile.qrels
    table.add_row("Documents", str(documents.count))
    table.add_row("Corpus size", f"~{documents.total_tokens:,} tokens")
    table.add_row("Tokens per document", _distribution(documents.tokens))
    language = f"{documents.language} ({documents.language_confidence:.0%})" if documents.language != "unknown" else "unknown"
    table.add_row("Language (guess)", language)
    table.add_row("Questions", str(questions.count))
    table.add_row("Words per question", _distribution(questions.words))
    table.add_row("Answerable questions", f"{questions.answerable} ({questions.answerable_ratio:.0%})")
    table.add_row("Not-in-context questions", str(questions.unanswerable))
    table.add_row("Qrel rows", str(qrels.qrel_rows))
    console.print(table)

    coverage = Table(title="Qrels coverage")
    coverage.add_column("Metric")
    coverage.add_column("Value", justify="right")
    coverage.add_row("Questions with relevance labels", f"{qrels.labeled_questions} of {questions.count} ({qrels.labeled_share:.0%})")
    coverage.add_row("Answerable, no labels (retrieval not scored)", str(qrels.unlabeled_answerable))
    coverage.add_row("Documents some question points to", f"{qrels.documents_referenced} of {documents.count} ({qrels.corpus_coverage:.0%})")
    console.print(coverage)
    if label_free:
        console.print("[yellow]Label-free dataset: retrieval metrics will be blank; answers are judged against the reference answer or the retrieved context.[/yellow]")

    for title, column, counts in (
        ("Categories", "Category", questions.categories),
        ("Difficulty", "Difficulty", questions.difficulties),
        ("Answer types", "Answer type", questions.answer_types),
    ):
        if title != "Categories" and set(counts) <= {"unknown"}:
            continue
        balance = Table(title=title)
        balance.add_column(column)
        balance.add_column("Count", justify="right")
        for name, count in counts.items():
            balance.add_row(name, str(count))
        console.print(balance)

    if profile.chunk_sizes:
        console.print(f"[bold]Suggested chunk sizes[/bold] (tokens): {', '.join(map(str, profile.chunk_sizes))}. {escape(profile.chunk_size_note)}")
        console.print("[dim]Compare them with a sweep: `sweep: {chunker.chunk_size: [...]}` (docs/configuration.md#sweeps).[/dim]")


@commands.command("inspect-dataset")
def inspect_dataset(
    docs: Path = typer.Option(..., "--docs", help="Document folder."),
    questions: Path = typer.Option(..., "--questions", help="Questions JSONL file."),
    qrels: Path | None = typer.Option(None, "--qrels", help="Optional qrels JSONL file."),
    include: list[str] | None = typer.Option(None, "--include", help="Only load files matching this glob (repeatable)."),
    exclude: list[str] | None = typer.Option(None, "--exclude", help="Skip files matching this glob (repeatable)."),
    on_error: str = typer.Option("raise", "--on-error", help="`raise` (fail on a file that cannot be loaded) or `skip` (warn and go on)."),
    estimate_cost: bool = typer.Option(True, "--estimate/--no-estimate", help="Also project the cost of the `standard` preset (runs each system once on the mock models; a few seconds)."),
) -> None:
    """Profile a dataset: sizes and balance, label coverage, suggested chunk sizes, projected cost and likely problems."""
    from ragbench.config.schema import DatasetConfig
    from ragbench.datasets.profile import profile_dataset, projected_standard_cost

    if on_error not in ("raise", "skip"):
        raise fail("--on-error must be 'raise' or 'skip'.")
    load_warnings: list[str] = []
    try:
        documents = load_documents(docs, include=include, exclude=exclude, on_error=on_error, warnings=load_warnings)  # type: ignore[arg-type]
    except (DocumentLoadError, ValueError, FileNotFoundError, MissingExtraError) as exc:
        raise fail(str(exc)) from exc
    try:
        dataset = load_dataset(questions, qrels)
    except (FileNotFoundError, ValueError) as exc:  # ValidationError is a ValueError
        raise fail(str(exc)) from exc
    profile = profile_dataset(documents, dataset)
    _print_profile(profile, dataset.label_free)
    if load_warnings:
        console.print(f"[yellow]{len(load_warnings)} document loading warning(s):[/yellow]")
        for message in load_warnings:
            console.print(f"  [yellow]•[/yellow] {escape(message)}", soft_wrap=True)
    warnings = [*validate_dataset(documents, dataset), *profile.warnings]
    if warnings:
        console.print(f"[yellow]Found {len(warnings)} dataset issue(s):[/yellow]")
        for warning in warnings:
            console.print(f"  [yellow]•[/yellow] {escape(warning)}", soft_wrap=True)
    else:
        console.print("[green]No dataset issues found.[/green]")
    if estimate_cost:
        try:
            with console.status("Projecting the cost of the standard preset…"):
                projected = projected_standard_cost(
                    DatasetConfig(documents_path=docs, questions_path=questions, qrels_path=qrels, include=include, exclude=exclude, on_error=on_error)
                )
        except Exception as exc:  # noqa: BLE001 (advice only: a failure here must not hide the profile above)
            console.print(f"[yellow]Could not project the cost: {escape(str(exc))}[/yellow]")
        else:
            console.print("\n[bold]Projected cost[/bold] of `--preset standard` with the default models:")
            print_estimate(projected)


def run_generation(
    documents: list[Any],
    *,
    n: int,
    shares: dict[str, float],
    ref: str,
    seed: int,
    loaded: ExperimentConfig | None,
    mock: bool,
    max_cost: float | None,
    no_cache: bool,
    yes: bool,
) -> tuple[Any, bool, float]:
    """Write synthetic questions: with the model `ref` (estimate, confirm, cache) or, in mock mode / without a key, from templates.

    Returns the `SynthesisResult`, whether it ran live, and what the cache avoided. Shared by `generate-questions` and `auto`.
    """
    from ragbench.cache import activate_cache, open_cache_runtime
    from ragbench.config.schema import CacheConfig, EvaluationConfig
    from ragbench.datasets.synthesis import allocate, estimate_generation_cost, generate_questions
    from ragbench.models import cost as pricing
    from ragbench.models.llms import create_llm
    from ragbench.models.refs import parse_model_ref, provider_reachable
    from ragbench.runtime import RuntimeContext, activate_runtime

    try:
        live = not mock and provider_reachable(parse_model_ref(ref)[0])
    except ValueError as exc:
        raise fail(str(exc)) from exc
    providers = loaded.providers if loaded else {}
    pricing.clear_pricing_overrides()
    pricing.reset_unknown_priced_models()
    pricing.register_pricing({name: price.model_dump() for name, price in loaded.pricing.items()} if loaded else {})
    cache_runtime = None
    try:
        if live:
            from ragbench.datasets.synthesis import allocate

            estimate = estimate_generation_cost(documents, n, shares, ref)
            if ref in pricing.unknown_priced_models() or estimate == 0:
                console.print(f"[yellow]No price is registered for {escape(ref)}: the cost is shown as $0. Add one under `pricing:` (pass --config).[/yellow]")
            confirm_spend(estimate, f"writing {n} questions ({', '.join(f'{count} {name}' for name, count in allocate(n, shares).items())})", yes, (loaded.evaluation if loaded else EvaluationConfig()).cost_confirm_threshold_usd)
            cache_config = loaded.cache if loaded else CacheConfig()
            cache_runtime = open_cache_runtime(cache_config) if cache_config.enabled and not no_cache else None
            with activate_cache(cache_runtime), activate_runtime(RuntimeContext(providers=providers)):
                try:
                    llm = create_llm(ref, providers=providers)
                    with console.status(f"Writing {n} questions with {ref}…"):
                        result = generate_questions(documents, n=n, mix=shares, llm=llm, seed=seed, budget=BudgetGuard(max_cost))
                except (RagbenchModelError, MissingExtraError) as exc:
                    raise fail(f"The model failed: {exc}") from exc
        else:
            console.print("[yellow]Mock mode: template questions, no model called. They validate the pipeline and say nothing about real quality.[/yellow]")
            result = generate_questions(documents, n=n, mix=shares, llm=None, seed=seed)
        saved = cache_runtime.disk.stats()["saved_cost_usd"] if cache_runtime is not None else 0.0
    finally:
        if cache_runtime is not None:
            cache_runtime.disk.close()
        pricing.clear_pricing_overrides()
    return result, live, saved


@commands.command("generate-questions")
def generate_questions_command(
    docs: Path = typer.Option(..., "--docs", help="Your documents: a folder, or one file."),
    out: Path = typer.Option(..., "--out", help="Questions JSONL to write, e.g. questions.jsonl."),
    n: int = typer.Option(100, "--n", min=1, help="How many questions to write."),
    mix: str | None = typer.Option(None, "--mix", help="Category shares, e.g. single_hop=0.4,multi_hop=0.2,paraphrase=0.15,numeric=0.1,unanswerable=0.15 (the default)."),
    model: str | None = typer.Option(None, "--model", help="Model ref that writes the questions (default: the default generator model)."),
    seed: int = typer.Option(0, "--seed", help="Same seed, same documents sampled, same questions."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Read `providers:`, `pricing:` and `cache:` from this config (needed for openai_compatible: models)."),
    mock: bool = typer.Option(False, "--mock", help="Write template questions with no model (pipeline validation only). Also used when the model's API key is missing."),
    max_cost: float | None = typer.Option(None, "--max-cost", min=0, help="Stop writing once this many dollars were charged (partial results are kept)."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Do not read or write the persistent disk cache."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before a live run whose estimated cost is above the confirmation threshold."),
    force: bool = typer.Option(False, "--force", help="Replace --out if it exists (it may hold questions you reviewed)."),
) -> None:
    """Write questions for documents that have none: single-hop, multi-hop, paraphrased, numeric and unanswerable, each flagged `needs_review`."""
    from ragbench.datasets.synthesis import MOCK_GENERATOR, parse_mix
    from ragbench.models.defaults import DEFAULT_GENERATOR_MODEL
    from ragbench.utils.jsonl import write_jsonl

    load_project_env(docs)
    if out.exists() and not force:
        raise fail(f"{out} already exists. Use --force to replace it (reviewed questions would be lost), or pick another --out.")
    try:
        shares = parse_mix(mix)
        documents = load_documents(docs)
        loaded = load_optional_config(config)
        ref = model or DEFAULT_GENERATOR_MODEL
    except (ValueError, FileNotFoundError, DocumentLoadError, MissingExtraError) as exc:
        raise fail(str(exc)) from exc
    result, live, saved = run_generation(documents, n=n, shares=shares, ref=ref, seed=seed, loaded=loaded, mock=mock, max_cost=max_cost, no_cache=no_cache, yes=yes)
    if not result.rows:
        raise fail("No question could be written. " + " ".join(result.warnings))
    write_jsonl(out, result.rows)

    table = Table(title=f"Questions written to {out}", title_justify="left")
    table.add_column("Category", style="bold")
    table.add_column("Asked for", justify="right")
    table.add_column("Written", justify="right")
    for name, wanted in result.requested.items():
        table.add_row(name, str(wanted), str(result.made.get(name, 0)))
    table.add_section()
    table.add_row("Total", str(sum(result.requested.values())), str(len(result.rows)))
    console.print(table)
    if live:
        spent = f"{money(result.cost_usd)} charged at standalone prices over {result.calls} model calls" + (f"; the cache avoided {money(saved)}" if saved else "")
        console.print(f"[dim]Generator {escape(result.generator)} · {spent}.[/dim]")
    else:
        console.print(f"[dim]Generator {MOCK_GENERATOR} (mock).[/dim]")
    for warning in result.warnings:
        console.print(f"  [yellow]•[/yellow] {escape(warning)}", soft_wrap=True)
    console.print(
        f"[bold]Review them:[/bold] every row is flagged `metadata.needs_review`. Read the file, fix or delete bad questions, then check it with "
        f"`ragbench inspect-dataset --docs {docs} --questions {out}`.",
        soft_wrap=True,
    )
    if result.stopped_by_budget:
        raise fail(f"Stopped at --max-cost {money(max_cost or 0)}: {len(result.rows)} of {n} questions were written.", code=1)


@commands.command("label")
def label_command(
    run: Path = typer.Option(..., "--run", help="A finished run directory, e.g. results/<run>."),
    top_k: int = typer.Option(10, "--top-k", min=1, help="Pool the first K distinct documents each system retrieved, per question."),
    judge_model: str | None = typer.Option(None, "--judge-model", help="Model ref that grades relevance (default: the run's judge model)."),
    out: Path | None = typer.Option(None, "--out", help="Directory for the outputs (default: the run directory)."),
    apply: bool = typer.Option(False, "--apply", help="Also write qrels.merged.jsonl: your labels plus the proposed grades for documents they do not mention."),
    mock: bool = typer.Option(False, "--mock", help="Grade with a word-overlap stand-in, no model (pipeline validation only). Also used when the model's API key is missing."),
    max_cost: float | None = typer.Option(None, "--max-cost", min=0, help="Stop grading once this many dollars were charged (partial proposals are written)."),
    max_workers: int = typer.Option(4, "--max-workers", min=1, help="Documents graded at the same time."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Do not read or write the persistent disk cache."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before a live run whose estimated cost is above the confirmation threshold."),
    force: bool = typer.Option(False, "--force", help="With --apply: replace an existing qrels.merged.jsonl."),
) -> None:
    """Propose relevance labels for a run by pooling: every system's top documents, graded 0-3 by an LLM. Never changes your qrels."""
    from ragbench.cache import activate_cache, open_cache_runtime
    from ragbench.datasets.pooling import (
        MERGED_NAME,
        LabelError,
        build_pool,
        estimate_grading_cost,
        grade_pool,
        load_run,
        write_label_outputs,
    )
    from ragbench.models import cost as pricing
    from ragbench.models.llms import create_llm
    from ragbench.models.refs import parse_model_ref, provider_reachable
    from ragbench.runtime import RuntimeContext, activate_runtime

    load_project_env(run)
    destination = out or run
    if apply and (destination / MERGED_NAME).exists() and not force:
        raise fail(f"{destination / MERGED_NAME} already exists. Use --force to replace it, or pick another --out.")
    try:
        data = load_run(run)
        plan = build_pool(data, top_k)
        ref = judge_model or data.config.evaluation.judge_model
        live = not mock and provider_reachable(parse_model_ref(ref)[0])
    except (LabelError, ValueError, ValidationError) as exc:
        raise fail(str(exc)) from exc
    pricing.clear_pricing_overrides()
    pricing.reset_unknown_priced_models()
    pricing.register_pricing({name: price.model_dump() for name, price in data.config.pricing.items()})
    console.print(f"{len(plan.questions)} questions · {plan.pairs} (question, document) pairs pooled from {len({s for r in data.rankings.values() for s in r})} systems at top {top_k}")
    if plan.skipped_unanswerable:
        console.print(f"[dim]{plan.skipped_unanswerable} unanswerable question(s) skipped.[/dim]")
    cache_runtime = None
    try:
        if live:
            estimate = estimate_grading_cost(data, plan, ref)
            if estimate == 0:
                console.print(f"[yellow]No price is registered for {escape(ref)}: the cost is shown as $0. Add one under `pricing:` in the run's config.[/yellow]")
            confirm_spend(estimate, f"grading {plan.pairs} documents with {ref}", yes, data.config.evaluation.cost_confirm_threshold_usd)
            cache_runtime = open_cache_runtime(data.config.cache) if data.config.cache.enabled and not no_cache else None
        else:
            console.print("[yellow]Mock grader: word overlap, no model called. The proposals only validate the pipeline; do not use them as labels.[/yellow]")
        progress = Progress(SpinnerColumn(), TextColumn("Grading"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console, transient=True)
        with activate_cache(cache_runtime), activate_runtime(RuntimeContext(providers=data.config.providers)), progress:
            task = progress.add_task("grade", total=plan.pairs)
            try:
                llm = create_llm(ref, force_mock=not live, providers=data.config.providers)
                graded = grade_pool(data, plan, llm, workers=max_workers, budget=BudgetGuard(max_cost) if max_cost is not None else None, progress=lambda done, total: progress.update(task, completed=done))
            except (RagbenchModelError, MissingExtraError) as exc:
                raise fail(f"The grader failed: {exc}") from exc
        saved = cache_runtime.disk.stats()["saved_cost_usd"] if cache_runtime is not None else 0.0
    finally:
        if cache_runtime is not None:
            cache_runtime.disk.close()
        pricing.clear_pricing_overrides()
    files = write_label_outputs(destination, data, plan, graded, top_k=top_k, judge_model=ref if live else "mock grader", mock=not live, apply=apply)

    distribution = {grade: sum(1 for judgments in graded.grades.values() for j in judgments.values() if j.grade == grade) for grade in range(4)}
    console.print(f"[green]Graded {sum(distribution.values())} pairs[/green] (0: {distribution[0]} · 1: {distribution[1]} · 2: {distribution[2]} · 3: {distribution[3]})")
    if live:
        console.print(f"[dim]{money(graded.cost_usd)} charged at standalone prices over {graded.calls} calls" + (f"; the cache avoided {money(saved)}" if saved else "") + ".[/dim]")
    for pair in graded.ungraded[:5]:
        console.print(f"  [yellow]•[/yellow] {pair[0]}/{pair[1]}: no valid grade; nothing is proposed for it", soft_wrap=True)
    console.print(f"Wrote [bold]{files.proposed}[/bold] and [bold]{files.review}[/bold]" + (f" and [bold]{files.merged}[/bold]" if files.merged else ""))
    if files.merged:
        console.print(f"Use the merged labels: set `dataset.qrels_path: {files.merged}` in your config. Your own files were not touched.")
    else:
        console.print("[dim]Read the review, then add --apply to write a merged qrels file (your labels stay authoritative).[/dim]")
    if graded.stopped_by_budget:
        raise fail(f"Stopped at --max-cost {money(max_cost or 0)}: some pairs were not graded.", code=1)


@commands.command()
def init(
    directory: Path = typer.Argument(Path("."), help="Where to write ragbench.yaml (and questions.jsonl if you have none yet)."),
    docs: Path = typer.Option(..., "--docs", help="Your documents: a folder, or one file."),
    questions: Path | None = typer.Option(None, "--questions", help="Your questions JSONL. Without it a questions.jsonl template is created in DIRECTORY."),
    preset: str = typer.Option("standard", "--preset", help=f"Which systems to compare: {', '.join(PRESET_NAMES)}."),
    force: bool = typer.Option(False, "--force", help="Replace an existing ragbench.yaml (questions files are never replaced)."),
) -> None:
    """Start a benchmark of your own documents: write a ready-to-run config and a questions file to edit."""
    from ragbench.datasets.scaffold import ScaffoldError, scaffold_project

    try:
        result = scaffold_project(directory, docs, questions, preset=preset, force=force)
    except ScaffoldError as exc:
        raise fail(str(exc)) from exc
    console.print(f"[green]Wrote {result.config_path}[/green] ({result.n_documents} documents, preset `{result.preset}`)")
    if result.created_questions:
        console.print(f"[green]Wrote {result.questions_path}[/green]: placeholder questions, so the project runs; replace them with real ones.")
    console.print("\n[bold]Next steps[/bold]")
    steps = []
    if result.created_questions:
        steps.append(f"Put your real questions in {result.questions_path} (docs/dataset-format.md), or bring existing ones with `ragbench import`.")
    steps += [
        f"ragbench inspect-dataset --docs {docs} --questions {result.questions_path}   # check the data and see the projected cost",
        f"ragbench run --config {result.config_path} --mock   # a free pipeline check",
        f"ragbench run --config {result.config_path}   # the real run (asks before spending more than the confirmation threshold)",
    ]
    for number, step in enumerate(steps, start=1):
        console.print(f"  {number}. {escape(step)}", soft_wrap=True)
    console.print("[dim]Paths in the config are relative to the directory you run ragbench from.[/dim]")


@commands.command("import")
def import_command(
    fmt: str = typer.Option(..., "--format", help="csv (question,answer,doc_ids), beir (corpus.jsonl + queries.jsonl + qrels/), or qa-md (Markdown `Q:` / `A:` pairs)."),
    source: Path = typer.Option(..., "--input", help="The file (csv, qa-md) or folder (beir, qa-md) to import."),
    output: Path = typer.Option(..., "--output", help="Directory to write questions.jsonl (and qrels.jsonl, docs/ for beir) into."),
    docs: Path | None = typer.Option(None, "--docs", help="Your documents folder, to check the imported doc ids against (csv, qa-md)."),
    split: str = typer.Option("test", "--split", help="beir: which qrels/<split>.tsv to use."),
    force: bool = typer.Option(False, "--force", help="Replace the files of an earlier import into --output (for beir, its docs/ folder too)."),
) -> None:
    """Convert a CSV, BEIR dataset or Markdown Q/A file into RAGBench's questions (and qrels, documents)."""
    from ragbench.datasets.importers import ImportDatasetError, import_dataset

    try:
        result = import_dataset(fmt, source, output, docs=docs, split=split, force=force)
    except (ImportDatasetError, DocumentLoadError, MissingExtraError, FileNotFoundError, ValueError) as exc:
        raise fail(str(exc)) from exc
    console.print(f"[green]Imported {result.n_questions} questions[/green] to {result.questions_path}")
    if result.docs_dir is not None:
        console.print(f"Documents: {result.n_documents} in {result.docs_dir}" + (f" · qrels: {result.n_qrels} rows in {result.qrels_path}" if result.qrels_path else ""))
    for warning in result.warnings[:10]:
        console.print(f"  [yellow]•[/yellow] {escape(warning)}", soft_wrap=True)
    if len(result.warnings) > 10:
        console.print(f"  [yellow]… and {len(result.warnings) - 10} more.[/yellow]")
    documents = result.docs_dir or docs or Path("YOUR_DOCS")
    console.print(f"\nNext: [bold]ragbench init {output} --docs {documents} --questions {result.questions_path}[/bold]", soft_wrap=True)
