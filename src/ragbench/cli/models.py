"""`ragbench models`: the models RAGBench can price, and refreshing OpenRouter's prices and capabilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.markup import escape
from rich.table import Table

from ragbench.cli.common import console, fail, json_output, print_json
from ragbench.models import cost as pricing
from ragbench.models import openrouter_catalog as catalog

models_app = typer.Typer(help="List the models RAGBench has prices for, and sync OpenRouter's catalog (prices, tools, reasoning).")

PROVIDERS = ("all", "openai", "anthropic", "openrouter")


def _row(ref: str, price: dict[str, Any] | None, **extra: Any) -> dict[str, Any]:
    tiers = (price or {}).get("tiers") or []
    above = [f">{t['max_prompt_tokens']:,}: {t2['input']:g}/{t2['output']:g}" for t, t2 in zip(tiers, tiers[1:], strict=False) if "max_prompt_tokens" in t]
    return {
        "ref": ref,
        "input": None if price is None else price["input"],
        "output": None if price is None else price["output"],
        "long_prompt": "; ".join(above) or None,
        **extra,
    }


def _table_rows(provider: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, price in pricing.MODEL_PRICING_USD_PER_1M.items():
        if model in ("mock-llm", "hashing-embedding"):
            continue
        family = "anthropic" if model.startswith("claude-") else "openai"
        if provider in ("all", family):
            rows.append(_row(model if family == "openai" else f"anthropic:{model}", price, source=f"price table ({pricing.PRICING_AS_OF})"))
    if provider in ("all", "openrouter"):
        loaded = catalog.load_catalog()
        for model_id, info in loaded.models.items():
            rows.append(
                _row(
                    f"openrouter:{model_id}",
                    info.price,
                    source=f"OpenRouter catalog ({loaded.fetched_at})",
                    open_weights=info.open_weights,
                    tools=info.supports_tools,
                    json=info.supports_json,
                    lowest_effort=info.lowest_effort(),
                )
            )
    return rows


@models_app.command("list")
def list_models(
    provider: str = typer.Option("all", "--provider", "-p", help=f"Only one provider: {', '.join(PROVIDERS)}."),
    search: str | None = typer.Option(None, "--search", "-s", help="Only refs containing this text (case-insensitive)."),
    open_weights: bool = typer.Option(False, "--open-weights", help="Only OpenRouter models with published weights (a Hugging Face id)."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON on stdout instead of a table."),
) -> None:
    """Models with a price, USD per 1M tokens: the built-in table (OpenAI, Anthropic) and the OpenRouter catalog. Offline."""
    if provider not in PROVIDERS:
        raise fail(f"--provider must be one of {', '.join(PROVIDERS)}, not {provider!r}")
    with json_output(as_json):
        rows = _table_rows(provider)
        if search:
            rows = [row for row in rows if search.lower() in row["ref"].lower()]
        if open_weights:
            rows = [row for row in rows if row.get("open_weights")]
        if as_json:
            print_json(rows)
            return
        table = Table(title="Model prices (USD per 1M tokens)", title_justify="left")
        for column, justify in (("Model ref", "left"), ("Input", "right"), ("Output", "right"), ("Long prompts", "left"), ("Open weights", "left"), ("Tools", "left"), ("Reasoning", "left")):
            table.add_column(column, justify=justify, overflow="fold")  # type: ignore[arg-type]

        def flag(value: Any) -> str:
            return "—" if value is None else ("yes" if value else "no")

        for row in rows:
            table.add_row(
                escape(row["ref"]),
                "variable" if row["input"] is None else f"{row['input']:g}",
                "variable" if row["output"] is None else f"{row['output']:g}",
                row["long_prompt"] or "",
                flag(row.get("open_weights")),
                flag(row.get("tools")),
                row.get("lowest_effort") or ("—" if "lowest_effort" not in row else "always on"),
            )
        console.print(table)
        loaded = catalog.load_catalog()
        where = "the bundled snapshot" if loaded.path == catalog.BUNDLED_CATALOG else str(loaded.path)
        console.print(
            f"[dim]OpenRouter catalog: {len(loaded.models)} models from {escape(where)}, fetched {loaded.fetched_at or 'never'}. "
            "Refresh it with `ragbench models sync`. `Reasoning` is the effort RAGBench sends by default.[/dim]"
        )


@models_app.command("sync")
def sync_models(
    from_file: Path | None = typer.Option(None, "--from-file", help="Read a saved `GET /api/v1/models` response instead of downloading it."),
    output: Path | None = typer.Option(None, "--output", "-o", help=f"Where to write the catalog. Default: ${catalog.CATALOG_ENV}, else ~/.cache/ragbench/."),
) -> None:
    """Download OpenRouter's model list (public, no key, free) and save prices, tool / JSON support and reasoning switches as a dated file."""
    if from_file is not None:
        try:
            response = json.loads(from_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise fail(f"Could not read {from_file}: {exc}") from None
    else:
        try:
            with console.status(f"Fetching {catalog.MODELS_URL}…"):
                response = catalog.fetch_models()
        except Exception as exc:  # noqa: BLE001 (network errors of every kind become one message)
            raise fail(f"Could not fetch {catalog.MODELS_URL}: {type(exc).__name__}: {exc}", code=1) from None
    try:
        written = catalog.write_catalog(response, output)
    except ValueError as exc:
        raise fail(str(exc)) from None
    priced = sum(1 for info in written.models.values() if info.price is not None)
    console.print(f"Saved {len(written.models)} models ({priced} with a fixed price) to {escape(str(written.path))}, dated {written.fetched_at}.")
    if output is not None and output != catalog.synced_catalog_path():
        console.print(f"[yellow]RAGBench reads the catalog from {escape(str(catalog.synced_catalog_path()))}; set {catalog.CATALOG_ENV} to use this file.[/yellow]")
