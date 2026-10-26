"""`ragbench optimize`: improve one system's harness on the train split and keep the change only if the test split confirms it."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer
from rich.markup import escape
from rich.table import Table

from ragbench.cli.common import console, fail, in_ci, is_interactive, json_output, load_experiment, money, print_json
from ragbench.config.schema import ExperimentConfig
from ragbench.errors import CommandFailed
from ragbench.evaluation.estimate import confirmation_decision
from ragbench.models.defaults import DEFAULT_GENERATOR_MODEL
from ragbench.models.llms import MockLLM
from ragbench.models.refs import answering_model, resolve_run_mode
from ragbench.optimize.estimate import OptimizeEstimate, estimate_optimize
from ragbench.optimize.loop import OptimizeSettings, optimize
from ragbench.optimize.proposer import Proposer
from ragbench.utils.env import load_project_env

commands = typer.Typer()

MODES = ("parallel", "sequential")
FEEDBACK = ("contrast", "labels", "none")
SELECTIONS = ("auto", "score", "arena")


def _choice(value: str, allowed: tuple[str, ...], option: str) -> str:
    if value not in allowed:
        raise fail(f"{option} must be one of {', '.join(allowed)}, not '{value}'.")
    return value


def proposer_model(config: ExperimentConfig, system: str, given: str | None) -> str:
    """`--proposer`, else the model that writes the system's answers, else the default generator."""
    if given:
        return given
    target = next((s for s in config.systems if s.resolved_name == system), None)
    return (answering_model(target) if target is not None else None) or DEFAULT_GENERATOR_MODEL


def print_optimize_estimate(estimate: OptimizeEstimate) -> None:
    table = Table(title=f"Optimizing {estimate.system}: estimate", title_justify="left")
    table.add_column("Stage", style="bold")
    table.add_column("What", overflow="fold")
    table.add_column("Cost", justify="right")
    q = estimate.questions
    runs = 3 * estimate.k + 1 if estimate.mode == "parallel" else 4 * estimate.k
    table.add_row("Train probes", f"{4 * estimate.k} probes of {q['train']} questions ({runs} runs, each indexing the corpus)", money(estimate.train_usd))
    table.add_row("Dev selection", f"base + finalists on {q['dev']} questions" if q["dev"] else "no dev questions: picked on train", money(estimate.dev_usd))
    table.add_row("Test verification", f"base and pick on {q['test']} questions", money(estimate.test_usd))
    table.add_row("Proposer", f"{3 * estimate.k} requests to {estimate.proposer_model}", money(estimate.proposer_usd))
    table.add_section()
    table.add_row("Total", "", f"[bold]{money(estimate.total_usd)}[/bold]")
    console.print(table)
    console.print(
        f"[dim]Per question {money(estimate.per_question_usd)} (answer and judge), indexing {money(estimate.ingestion_usd)} per run; measured on the mock models "
        "and priced with the configured ones, assuming no cache hits (a live run re-uses unchanged calls for free). Upper bound for the proposer: a full evidence "
        "budget per request; retries are not counted.[/dim]"
    )
    for warning in estimate.measured.warnings:
        console.print(f"[yellow]• {escape(warning)}[/yellow]")


@commands.command("optimize")
def optimize_command(
    config: Path = typer.Option(..., "--config", "-c", help="Experiment config holding the system and a dataset with `splits.json` (`ragbench split`)."),
    system: str = typer.Option(..., "--system", help="Name of the system whose harness to optimize."),
    k: int = typer.Option(2, "--k", min=1, help="Width (parallel) or depth (sequential): 4k train probes and 3k proposer requests."),
    mode: str = typer.Option("parallel", "--mode", help="parallel: k rollouts, k INV + k DOM candidates, k compositions. sequential: k rounds, carrying the best forward."),
    feedback: str = typer.Option("contrast", "--feedback", help="What the proposer sees: contrast (behaviour only), labels (train grades and references) or none."),
    proposer: str | None = typer.Option(None, "--proposer", help="Model ref of the proposer. Default: the system's own generator."),
    finalists: int = typer.Option(2, "--finalists", min=1, help="Candidates that beat the base on train and go on to dev."),
    select: str = typer.Option("auto", "--select", help="How dev picks: score, arena (pairwise, label-free), or auto (arena on a label-free dataset)."),
    test_trials: int = typer.Option(1, "--test-trials", min=1, help="Times each test question is answered by the base and the pick."),
    max_cost_usd: float | None = typer.Option(None, "--max-cost-usd", min=0, help="Spending cap for the whole optimization (default: `evaluation.max_cost_usd`). Required for a live run."),
    out: Path | None = typer.Option(None, "--out", help="Output folder (default: results/optimize_<system>_<time>). Re-running into the same folder resumes."),
    mock: bool = typer.Option(False, "--mock", help="Mock models and the scripted proposer: the whole loop offline, at no cost."),
    max_workers: int | None = typer.Option(None, "--max-workers", min=1, help="Questions answered at the same time in each probe."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before a live optimization whose estimate is above `evaluation.cost_confirm_threshold_usd`."),
    as_json: bool = typer.Option(False, "--json", help="Print the outcome as JSON on stdout."),
) -> None:
    """Propose → Probe → Compose: improve a system's harness (prompts and knobs) on train, pick on dev, verify on test.

    The proposer reads evidence from train runs only (never dev or test); a change is adopted only when the test split shows a
    significant gain over the base (paired bootstrap, Holm). Writes `optimized.yaml`, `optimization.md` and `ledger.json`.
    """
    settings = OptimizeSettings(
        system=system,
        k=k,
        mode=_choice(mode, MODES, "--mode"),  # type: ignore[arg-type]
        feedback=_choice(feedback, FEEDBACK, "--feedback"),  # type: ignore[arg-type]
        finalists=finalists,
        select=_choice(select, SELECTIONS, "--select"),  # type: ignore[arg-type]
        test_trials=test_trials,
    )
    with json_output(as_json):
        load_project_env(config)
        loaded = load_experiment(config, None)
        settings.max_cost_usd = max_cost_usd if max_cost_usd is not None else loaded.evaluation.max_cost_usd
        live = resolve_run_mode(loaded, mock) == "live"
        model = proposer_model(loaded, system, proposer)
        chosen = Proposer.from_ref(model, force_mock=not live)
        if live:
            if settings.max_cost_usd is None:
                raise fail("A live optimization needs a spending cap: pass --max-cost-usd (or set `evaluation.max_cost_usd`). Check the cost first with `ragbench estimate --optimize`.")
            if isinstance(chosen.llm, MockLLM):
                raise fail(f"The proposer model {model} cannot be reached (is its API key set?). Pass --mock to run offline, or another --proposer.")
            _confirm(loaded, settings, model, yes)
        folder = out or loaded.run.output_dir / f"optimize_{system}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            result = optimize(config, settings, proposer=chosen, out_dir=folder, force_mock=not live, max_workers=max_workers, progress=lambda line: console.print(f"[dim]{escape(line)}[/dim]"))
        except ValueError as exc:
            raise fail(str(exc)) from exc
        console.print(f"[bold]{result.verdict.replace('_', ' ')}[/bold] · {result.ledger.probes} probes, {result.ledger.proposer_calls} proposer requests, charged {money(result.ledger.budget.spent)}")
        for note in result.notes:
            console.print(f"[dim]• {escape(note)}[/dim]")
        console.print(f"Results: [bold]{result.out_dir}[/bold] (optimization.md, optimized.yaml, ledger.json)", soft_wrap=True)
        if as_json:
            print_json(
                {
                    "verdict": result.verdict,
                    "pick": result.pick,
                    "adopted_harness": (result.adopted.raw.get("harness") if result.adopted else None),
                    "verification": result.verification,
                    "probes": result.ledger.probes,
                    "proposer_calls": result.ledger.proposer_calls,
                    "spent_usd": result.ledger.budget.spent,
                    "out_dir": str(result.out_dir),
                }
            )
        if result.verdict == "stopped":
            raise CommandFailed(f"Stopped early: {result.ledger.stopped}. The base harness is kept; fix the cause (or raise the cap) and re-run into {result.out_dir} to resume.")


def _confirm(config: ExperimentConfig, settings: OptimizeSettings, model: str, yes: bool) -> None:
    with console.status("Estimating the cost…"):
        estimate = estimate_optimize(config, settings, model)
    print_optimize_estimate(estimate)
    cap = settings.max_cost_usd
    if cap is not None and estimate.total_usd > cap:
        console.print(f"[yellow]The estimate is above the {money(cap)} cap: the loop will stop early and keep the base.[/yellow]")
    decision = confirmation_decision(estimate.total_usd, config.evaluation.cost_confirm_threshold_usd, yes=yes, interactive=is_interactive(), ci=in_ci())
    if decision == "refuse":
        raise fail(f"The estimated cost {money(estimate.total_usd)} is above the confirmation threshold and there is no one to ask. Re-run with --yes to go ahead.")
    if decision == "ask" and not typer.confirm(f"Optimize for about {money(estimate.total_usd)} (cap {money(cap or 0)})?", default=False):
        raise fail("Cancelled. Nothing was spent.", code=1)
