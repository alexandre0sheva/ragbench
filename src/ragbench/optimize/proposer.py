"""The harness proposer: reads an evidence tree and proposes one change to a system's harness, of a requested kind.

**INV** asks for a general change (one that would help on any document collection), **DOM** for one made for this collection (Harness²'s
two proposal kinds). The model gets the evidence files inline (an API model cannot open a folder), framing files first and rollouts last,
within `max_evidence_chars`; the whole request passes the leakage guard before it is sent. The reply is a patch with omitted-is-inherited
semantics. A reply that cannot be used (not JSON, an unknown field, a value out of range, a field the system does not have, no change, or
held-out text) is sent back once with the reason; a second bad reply leaves the harness as it was (`Proposal.inherited`).

Any chat model works (`Proposer.from_ref`); in mock mode `MockLLM` answers with a scripted edit (`models/mock_proposer.py`), which reads
the `Kind:`, `Proposal number:`, `Fields you may change:` and `Current harness (JSON):` lines of the request, so keep them as they are.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from ragbench.config.schema import SystemConfig
from ragbench.datasets.leakage import LeakageGuard
from ragbench.harness.apply import effective_harness
from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLM, create_llm
from ragbench.models.prompts import PROPOSE_HARNESS_MARKER
from ragbench.optimize.evidence import MANIFEST, field_names, guard_for_run, read_manifest

logger = logging.getLogger(__name__)

ProposalKind = Literal["INV", "DOM"]
MAX_EVIDENCE_CHARS = 60_000  # about 15k tokens of evidence per request

PROPOSER_SYSTEM_PROMPT = f"""{PROPOSE_HARNESS_MARKER}: the prompt texts and settings that shape how it finds passages and writes answers.

The evidence of how the system behaves follows as files, each headed by `=== path ===`. Start with MISSION.md. Look for patterns that recur
across questions (in the rollouts, the contrasts and the failures), find their cause in the harness, and change what would fix them.

Rules:
- Change only the fields listed as fields you may change, within their limits. List only the fields you change; every other field keeps its value.
- A text field replaces the whole current text: write the complete new text, not an addition to it.
- A few-shot example may only use a question from the rollouts.
- Reply with one JSON object and nothing else."""

KIND_REQUESTS: dict[str, str] = {
    "INV": "a general change that would help this kind of system on any document collection. Do not mention names, terms or topics of this one.",
    "DOM": "a change made for this document collection: its vocabulary, its kinds of questions, the shape of its documents.",
}
RETRY_PROMPT = "That change cannot be used: {error}. Reply again with one JSON object of the same form."


@dataclass
class Proposal:
    kind: ProposalKind
    patch: dict[str, Any]  # the fields the proposal changes; {} when the current harness is inherited
    harness: dict[str, Any]  # the system's whole `harness:` block with the patch applied (the old block when inherited)
    rationale: str = ""
    attempts: int = 0  # model calls made, the retry included
    error: str | None = None  # why the last reply was refused (always set when inherited)
    cost: CostBreakdown = field(default_factory=CostBreakdown)

    @property
    def inherited(self) -> bool:
        return not self.patch

    def apply(self, system: SystemConfig) -> SystemConfig:
        """`system` with this proposal's harness (the same object when inherited)."""
        return system if self.inherited else _with_harness(system, self.harness)


def _with_harness(system: SystemConfig, harness: dict[str, Any]) -> SystemConfig:
    """A validated copy of `system` with `harness` as its `harness:` block. Raises ValueError with the config check's message."""
    try:
        return SystemConfig.model_validate({**system.model_dump(), "harness": harness})
    except ValidationError as exc:
        raise ValueError("; ".join(str(error["msg"]).removeprefix("Value error, ") for error in exc.errors())) from None


def _parse(text: str) -> tuple[dict[str, Any], str]:
    """The `harness` object and the rationale of a reply. Raises ValueError with what is wrong."""
    body = text.strip()
    if body.startswith("```"):
        body = body.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        raise ValueError('the reply is not one JSON object of the form {"rationale": "...", "harness": {...}}') from None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("harness"), dict):
        raise ValueError('the reply needs a `harness` object holding the fields to change: {"rationale": "...", "harness": {"<field>": <value>}}')
    return parsed["harness"], str(parsed.get("rationale") or "")


class Proposer:
    MAX_ATTEMPTS = 2  # the first reply and one retry

    def __init__(self, llm: LLM, *, temperature: float = 0.7, max_evidence_chars: int = MAX_EVIDENCE_CHARS, guard: LeakageGuard | None = None):
        self.llm = llm
        self.temperature = temperature
        self.max_evidence_chars = max_evidence_chars
        self.guard = guard  # None: built from the evidence's run and split

    @classmethod
    def from_ref(cls, ref: str | None, *, force_mock: bool = False, **kwargs: Any) -> Proposer:
        """A proposer on a model ref (`deepseek/...` through OpenRouter, `anthropic:claude-sonnet-5-5`, ...); the mock in mock mode."""
        return cls(create_llm(ref, force_mock=force_mock), **kwargs)

    def propose(self, evidence: Path, system: SystemConfig, kind: ProposalKind, *, variant: int = 0) -> Proposal:
        """One proposal of `kind` for `system` from the evidence tree at `evidence`. `variant` numbers the proposals made from one tree."""
        manifest = read_manifest(evidence)
        guard = self.guard or guard_for_run(Path(manifest["run_dir"]), manifest["split"])
        current = effective_harness(system)
        request = f"{render_evidence(evidence, manifest, self.max_evidence_chars)}\n\n{_request(kind, variant, field_names(system, current), current)}"
        guard.assert_text_clean(request, "the proposer's request")
        messages = [{"role": "system", "content": PROPOSER_SYSTEM_PROMPT}, {"role": "user", "content": request}]
        proposal = Proposal(kind, {}, dict(system.harness))
        for _ in range(self.MAX_ATTEMPTS):
            result = self.llm.generate(messages, json_mode=True, temperature=self.temperature)
            proposal.attempts += 1
            proposal.cost = proposal.cost.plus(CostBreakdown(llm_prompt_tokens=result.prompt_tokens, llm_completion_tokens=result.completion_tokens, llm_cost=result.cost.total_cost))
            try:
                patch, rationale = _parse(result.text)
                harness = self._check(system, current, patch, guard)
            except ValueError as exc:
                proposal.error = str(exc)
                messages = [*messages, {"role": "assistant", "content": result.text}, {"role": "user", "content": RETRY_PROMPT.format(error=proposal.error)}]
                continue
            proposal.patch, proposal.harness, proposal.rationale, proposal.error = patch, harness, rationale, None
            return proposal
        logger.warning("The %s proposal for %s was not usable twice (%s); the current harness is kept.", kind, system.resolved_name, proposal.error)
        return proposal

    @staticmethod
    def _check(system: SystemConfig, current: dict[str, Any], patch: dict[str, Any], guard: LeakageGuard) -> dict[str, Any]:
        """The system's new `harness:` block, or ValueError saying why `patch` cannot be used."""
        patch = {name: value for name, value in patch.items() if value is not None}
        if not patch:
            raise ValueError("the change changes nothing: `harness` lists no field")
        if guard.scan_text(json.dumps(patch, ensure_ascii=False)):
            # Name neither the question nor its id: the reason itself would show the model held-out content.
            raise ValueError("it holds the text of a question that is not among the questions you were shown; use only questions from the rollouts")
        harness = {**system.harness, **patch}
        candidate = _with_harness(system, harness)
        if effective_harness(candidate) == current:
            raise ValueError("the change changes nothing: every field it lists already has that value")
        return harness


def _request(kind: ProposalKind, variant: int, fields: list[str], current: dict[str, Any]) -> str:
    return "\n".join(
        [
            "=== request ===",
            f"Kind: {kind}: {KIND_REQUESTS[kind]}",
            f"Proposal number: {variant + 1} (each proposal from this evidence should try a different idea)",
            f"Fields you may change: {', '.join(fields)}",
            f"Current harness (JSON): {json.dumps(current, ensure_ascii=False, default=str)}",
            'Reply with one JSON object and nothing else: {"rationale": "the pattern you saw and why this change fixes it, in two or three sentences", '
            '"harness": {"<field>": <new value>, ...}}',
        ]
    )


def render_evidence(evidence: Path, manifest: dict[str, Any], max_chars: int = MAX_EVIDENCE_CHARS) -> str:
    """The evidence files as one text, `=== path ===` before each: the framing files always, then failures, contrasts, the library and the
    rollouts (those of failed and contrasted questions first) while they fit in `max_chars`."""
    system_file = f"harness/{manifest['system']}.yaml"
    framing = ["MISSION.md", "task.md", "harness/fields.md", system_file]
    harness = sorted(path.relative_to(evidence).as_posix() for path in (evidence / "harness").glob("*.yaml"))
    rollouts = sorted((evidence / "rollouts").glob("*.md"))
    focus = {question_id: index for index, question_id in enumerate(manifest.get("focus") or [])}
    ordered_rollouts = sorted(rollouts, key=lambda path: (focus.get(_question_of(path), len(focus)), path.name))
    rest = [
        *(name for name in harness if name != system_file),
        "failures.md",
        *sorted(path.relative_to(evidence).as_posix() for path in (evidence / "contrasts").glob("*.md")),
        *sorted(path.relative_to(evidence).as_posix() for path in (evidence / "library").glob("*.md")),
        *(path.relative_to(evidence).as_posix() for path in ordered_rollouts),
    ]
    parts = [_section(evidence, name) for name in framing if (evidence / name).is_file()]
    used = sum(len(part) for part in parts)
    left_out = 0
    for name in rest:
        if name == MANIFEST or not (evidence / name).is_file():
            continue
        part = _section(evidence, name)
        if used + len(part) > max_chars:
            left_out += 1
            continue
        parts.append(part)
        used += len(part)
    if left_out:
        parts.append(f"({left_out} more file{'s' if left_out != 1 else ''} not shown: the evidence is longer than one request holds.)")
    return "\n\n".join(parts)


def _question_of(path: Path) -> str:
    """The question id a rollout file is about (its first line is `# Question <id>`)."""
    first = path.read_text(encoding="utf-8").split("\n", 1)[0]
    return first.removeprefix("# Question ").strip()


def _section(evidence: Path, name: str) -> str:
    return f"=== {name} ===\n{(evidence / name).read_text(encoding='utf-8').rstrip()}"
