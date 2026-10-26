"""`ragbench harness show` and `ragbench harness diff`: what a system's harness is, and how two configs' harnesses differ."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

import typer
import yaml
from pydantic import ValidationError

from ragbench.cli.common import fail, json_output, load_experiment, print_json
from ragbench.config.schema import SystemConfig
from ragbench.harness.apply import effective_harness, harness_fingerprint
from ragbench.harness.schema import KNOBS, HarnessPatch

harness_app = typer.Typer(help="Show a system's editable harness (prompts and knobs) and compare the harnesses of two configs. Offline.")


def _range(knob: str) -> str:
    """`1 to 50` from the knob's declared bounds; empty for an unbounded knob (`reranker`)."""
    bounds = {type(item).__name__: getattr(item, type(item).__name__.lower(), None) for item in HarnessPatch.model_fields[knob].metadata}
    low, high = bounds.get("Ge"), bounds.get("Le")
    return f"{low:g} to {high:g}" if low is not None and high is not None else ""


def _systems(path: Path) -> dict[str, SystemConfig]:
    return {system.resolved_name: system for system in load_experiment(path, None).systems}


def _closest(name: str, known: list[str], what: str) -> Exception:
    close = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
    hint = f" Did you mean '{close[0]}'?" if close else ""
    return fail(f"No {what} named '{name}'.{hint} ({what.capitalize()}s: {', '.join(known)})")


@harness_app.command("show")
def show(
    system: str = typer.Argument(..., help="A system type (its defaults), or with --config the name of a system in that config."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config whose system to show, with its `harness:` patch applied."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON on stdout instead of YAML."),
) -> None:
    """Print a system's harness as YAML: every prompt text it sends and its knobs, with what the config's patch changed marked."""
    import ragbench.rag_systems  # noqa: F401  (registers the built-in systems)
    from ragbench.registry import SYSTEMS

    with json_output(as_json):
        if config is not None:
            systems = _systems(config)
            if system not in systems:
                raise _closest(system, list(systems), "system")
            chosen = systems[system]
        elif system in SYSTEMS:
            try:
                chosen = SystemConfig(type=system)
            except ValidationError:
                raise fail(f"System type '{system}' has required options, so it has no default harness; show one from a config with --config.") from None
        else:
            raise _closest(system, SYSTEMS.names(), "system type")
        harness = effective_harness(chosen)
        if not harness:
            raise fail(f"System '{system}' (type {chosen.type}) takes no `harness:`: an external system's prompts live outside RAGBench, an adaptive system's routes each have their own.")
        patched = list(chosen.harness)
        fingerprint = harness_fingerprint(chosen)
        if as_json:
            print_json({"system": chosen.resolved_name, "type": chosen.type, "fingerprint": fingerprint, "patched": patched, "harness": harness})
            return
        lines = [f"# {chosen.resolved_name} (type {chosen.type}), fingerprint {fingerprint}"]
        for key, value in harness.items():
            notes = [f"knob, {_range(key)}" if _range(key) else "knob"] if key in KNOBS else []
            notes += ["unset: the run's evaluation.context_k applies"] if key == "top_k" and value is None else []
            notes += ["patched"] if key in patched else []
            text = yaml.safe_dump({key: value}, sort_keys=False, allow_unicode=True, width=1_000_000).rstrip("\n").split("\n")
            text[0] += f"  # {', '.join(notes)}" if notes else ""
            lines += text
        typer.echo("\n".join(lines))


def _changes(a: SystemConfig, b: SystemConfig) -> dict[str, list[Any]]:
    left, right = effective_harness(a), effective_harness(b)
    return {key: [left.get(key), right.get(key)] for key in dict.fromkeys([*left, *right]) if left.get(key) != right.get(key)}


def _describe(key: str, old: Any, new: Any) -> list[str]:
    if isinstance(old, str) and isinstance(new, str):
        lines = difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=1)
        return [f"  {key}:", *(f"    {line[0]} {line[1:]}" for line in list(lines)[2:] if not line.startswith("@@"))]
    return [f"  {key}: {'unset' if old is None else old} -> {'unset' if new is None else new}"]


@harness_app.command("diff")
def diff(
    first: Path = typer.Argument(..., help="A config file."),
    second: Path = typer.Argument(..., help="Another config file (for example the optimized one)."),
    system: str | None = typer.Option(None, "--system", "-s", help="Only this system (by name)."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON on stdout instead of text."),
) -> None:
    """Compare the harnesses of the systems two configs share (by name), field by field; list the systems only one of them has."""
    with json_output(as_json):
        left, right = _systems(first), _systems(second)
        names = [name for name in dict.fromkeys([*left, *right]) if system is None or name == system]
        if system is not None and not names:
            raise _closest(system, list(dict.fromkeys([*left, *right])), "system")
        report: list[dict[str, Any]] = []
        for name in names:
            if name not in right or name not in left:
                report.append({"system": name, "only_in": (first if name in left else second).name})
            elif changes := _changes(left[name], right[name]):
                report.append({"system": name, "changes": changes})
        if as_json:
            print_json(report)
            return
        if not report:
            typer.echo("No harness differences.")
            return
        lines: list[str] = []
        for entry in report:
            if "only_in" in entry:
                lines.append(f"{entry['system']}: only in {entry['only_in']}")
                continue
            a, b = left[entry["system"]], right[entry["system"]]
            lines.append(f"{entry['system']} (type {a.type})" if a.type == b.type else f"{entry['system']} (type {a.type} -> {b.type})")
            for key, (old, new) in entry["changes"].items():
                lines += _describe(key, old, new)
        typer.echo("\n".join(lines))
