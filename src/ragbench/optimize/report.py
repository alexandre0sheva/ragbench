"""The optimization's outputs: `optimized.yaml` (the config with the adopted harness, or unchanged), `optimization.md` (lineage, changes,
probe scores, candidate diversity, verdict) and `ledger.json`."""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

import yaml

from ragbench.harness.apply import effective_harness
from ragbench.optimize.loop import Candidate, OptimizeResult

VERDICTS = {
    "improved": "improved: the pick beat the base on the test split, significantly",
    "no_reliable_gain": "no reliable gain: the base harness is kept",
    "stopped": "stopped early: the base harness is kept",
}
COLLAPSE_BELOW = 0.1  # candidate diversity under this: the proposals barely differ (Harness²'s collapse check)


def _lines(candidate: Candidate) -> set[str]:
    return set(yaml.safe_dump(effective_harness(candidate.config), sort_keys=True, allow_unicode=True, width=10_000).splitlines())


def diversity(candidates: list[Candidate]) -> float | None:
    """1 minus the mean pairwise line-Jaccard similarity of the distinct candidates' harnesses (1 = nothing shared, 0 = all the same)."""
    distinct = list({c.fingerprint: c for c in candidates}.values())
    if len(distinct) < 2:
        return None
    similarities = []
    for a, b in itertools.combinations([_lines(c) for c in distinct], 2):
        similarities.append(len(a & b) / len(a | b) if a | b else 1.0)
    return 1 - sum(similarities) / len(similarities)


def _number(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _changes(result: OptimizeResult) -> list[str]:
    by_label = {c.label: c for c in [result.base, *result.candidates]}
    lines: list[str] = []
    for candidate in result.candidates:
        if candidate.proposal is None or candidate.proposal.inherited:
            continue
        parent = by_label.get(candidate.parent or "", result.base)
        before, after = effective_harness(parent.config), effective_harness(candidate.config)
        lines += ["", f"### `{candidate.label}` ({candidate.kind}, from `{parent.label}`)", ""]
        if candidate.proposal.rationale:
            lines += [f"> {candidate.proposal.rationale}", ""]
        lines.append("```diff")
        for name in candidate.proposal.patch:
            lines += [f"- {name}: {before.get(name)!r}", f"+ {name}: {after.get(name)!r}"]
        lines.append("```")
    return lines or ["", "No candidate changed the harness."]


def render_report(result: OptimizeResult) -> str:
    settings, ledger = result.settings, result.ledger
    lines = [
        f"# Optimization of `{settings.system}`",
        "",
        f"{settings.mode} mode, k = {settings.k}, feedback `{settings.feedback}`, {settings.finalists} finalist(s), selection `{settings.select}`.",
        "",
        "## Verdict",
        "",
        f"**{VERDICTS[result.verdict]}.**",
    ]
    verification = result.verification
    if verification:
        lines += [
            "",
            f"On test ({verification['n']} questions, {verification['trials']} trial(s)): `{verification['pick']}` {_number(verification['pick_mean'])} against the base's "
            f"{_number(verification['base_mean'])}, difference {verification['mean_diff']:+.3f} (95% CI {verification['ci_lo']:+.3f} to {verification['ci_hi']:+.3f}), "
            f"p = {verification['p']:.4f}, Holm-adjusted over {verification['comparisons']} comparison(s) p = {verification['p_holm']:.4f}; "
            f"wins / ties / losses {verification['wins']} / {verification['ties']} / {verification['losses']}.",
        ]
    if result.adopted is not None:
        lines += ["", f"`optimized.yaml` holds `{settings.system}` with the harness of `{result.adopted.label}`."]
    else:
        lines += ["", f"`optimized.yaml` holds `{settings.system}` unchanged."]
    lines += [f"- {note}" for note in ([""] if result.notes else []) + result.notes if note]
    lines += ["", "## Candidates", "", "| candidate | kind | from | changes | train | dev | note |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for candidate in [result.base, *result.candidates]:
        proposal = candidate.proposal
        changes = ", ".join(f"`{name}`" for name in proposal.patch) if proposal and proposal.patch else "-"
        note = "inherited: " + (proposal.error or "") if proposal and proposal.inherited else ""
        if candidate.label == result.pick:
            note = (note + " pick").strip()
        lines.append(f"| `{candidate.label}` | {candidate.kind} | {candidate.parent or '-'} | {changes} | {_number(candidate.train_mean)} | {_number(candidate.dev_value)} | {note} |")
    lines += ["", "Train and dev columns are mean answer scores (0 to 5; dev is arena points when picked by arena). The base's train value averages its rollouts."]
    lines += ["", "## Changes", *_changes(result)]
    value = diversity([c for c in result.candidates if c.train is not None])
    lines += ["", "## Diversity", ""]
    if value is None:
        lines.append("Fewer than two distinct candidates: nothing to compare.")
    else:
        lines.append(f"Candidate diversity {value:.2f} (1 minus the mean pairwise line-Jaccard similarity of the candidates' harnesses).")
        if value < COLLAPSE_BELOW:
            lines.append("The candidates barely differ: the proposals collapsed onto one idea.")
    lines += [
        "",
        "## Ledger",
        "",
        f"{ledger.probes} of {ledger.expected['probes']} train probes and {ledger.proposer_calls} of {ledger.expected['proposer_calls']} proposer stages; "
        f"charged ${ledger.budget.spent:.4f}" + (f" of a ${ledger.budget.cap_usd:.4f} cap" if ledger.budget.cap_usd is not None else "") + f", real spend ${ledger.real_spend_usd:.4f}. "
        "Every step is in `ledger.json`.",
    ]
    return "\n".join(lines) + "\n"


def optimized_config(result: OptimizeResult, raw: dict[str, Any], target_index: int) -> dict[str, Any]:
    config = copy.deepcopy(raw)
    adopted = result.adopted
    if adopted is not None:
        config["systems"][target_index]["harness"] = copy.deepcopy(adopted.raw.get("harness") or {})
    return config


def write_outputs(result: OptimizeResult, raw: dict[str, Any], target_index: int) -> list[Path]:
    out = result.out_dir
    header = f"# Written by `ragbench optimize` for `{result.settings.system}`: {VERDICTS[result.verdict]}. See optimization.md.\n"
    (out / "optimized.yaml").write_text(header + yaml.safe_dump(optimized_config(result, raw, target_index), sort_keys=False, allow_unicode=True), encoding="utf-8")
    (out / "optimization.md").write_text(render_report(result), encoding="utf-8")
    result.ledger.write(out / "ledger.json")
    return [out / "optimized.yaml", out / "optimization.md", out / "ledger.json"]
