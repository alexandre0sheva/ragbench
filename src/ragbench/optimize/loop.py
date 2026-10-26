"""`ragbench optimize`: Propose → Probe → Compose over one system's harness (Harness²'s recursion), picked on dev, verified on test.

**Parallel** (width): the base harness is run `k` times on train (`k` trials of one run); from each rollout's evidence the proposer writes
one INV and one DOM candidate; every candidate is probed on train; from evidence that contrasts each candidate with the base, the proposer
writes `k` compositions, which are probed too. **Sequential** (depth): `k` rounds of base → INV + DOM → composition, each round starting
from the best harness on train so far. Both make `4k` train probes and `3k` proposer stages (`ledger.py`).

Then the candidates that beat the base on train (the best `finalists`) and the base are run on **dev**, and the best of them on dev (by
answer score, or by pairwise arena points on a label-free dataset) is the pick. Without dev questions the pick is the best on train. The
pick and the base are then run on **test** and compared question by question (paired bootstrap; Holm over the comparisons made on test,
one here): only a significant gain is "improved"; otherwise the base is kept. The method never reads dev or test: the proposer sees
evidence from train runs only, and only the numbers of dev and test runs are used.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from ragbench.config.loader import load_config_dict
from ragbench.config.schema import ExperimentConfig, SystemConfig
from ragbench.config.sweep import expand_sweeps
from ragbench.datasets.leakage import LeakageError
from ragbench.datasets.loader import configured_splits_path, load_dataset
from ragbench.datasets.profile import profile_dataset
from ragbench.datasets.splits import Splits, apply_split, load_splits
from ragbench.documents.loaders import load_dataset_documents
from ragbench.evaluation.stats import ALPHA, grouped_paired_bootstrap, holm_adjust
from ragbench.harness.apply import effective_harness, harness_fingerprint
from ragbench.models.cost import CostBreakdown
from ragbench.optimize.evidence import Feedback, build_evidence, evidence_guard, field_names
from ragbench.optimize.ledger import Entry, Ledger
from ragbench.optimize.library import Library, LibraryError, build_entry, features_of, lesson_markdown, resolve_library
from ragbench.optimize.probe import ProbeFailed, ProbeResult, ProbeStopped, probe_config, run_probe
from ragbench.optimize.proposer import Proposal, ProposalKind, Proposer
from ragbench.utils.jsonl import read_jsonl, write_jsonl

Mode = Literal["parallel", "sequential"]
Select = Literal["auto", "score", "arena"]
Verdict = Literal["improved", "no_reliable_gain", "stopped"]
STATE = "state.json"


@dataclass
class OptimizeSettings:
    system: str
    k: int = 2
    mode: Mode = "parallel"
    feedback: Feedback = "contrast"
    finalists: int = 2  # candidates taken from train to dev
    select: Select = "auto"  # auto: arena points on a label-free dataset, answer score otherwise
    max_cost_usd: float | None = None
    test_trials: int = 1
    library: bool = True  # read lessons from the experience library and, on an adopted harness, write one (`library.enabled` in the config also applies)

    def identity(self) -> dict[str, Any]:
        """What a resumed attempt must share with the first (the cap may change between attempts)."""
        return {"system": self.system, "k": self.k, "mode": self.mode, "feedback": self.feedback, "finalists": self.finalists, "select": self.select, "test_trials": self.test_trials}


@dataclass
class Candidate:
    label: str
    kind: str  # base | INV | DOM | COMPOSE
    parent: str | None
    raw: dict[str, Any]  # the system's raw config with this candidate's `harness:`
    proposal: Proposal | None = None
    train: ProbeResult | None = None
    dev_value: float | None = None

    @property
    def config(self) -> SystemConfig:
        return SystemConfig.model_validate({**self.raw, "name": self.label})

    @property
    def fingerprint(self) -> str:
        return harness_fingerprint(self.config)

    @property
    def train_mean(self) -> float | None:
        return self.train.mean if self.train is not None else None


@dataclass
class OptimizeResult:
    verdict: Verdict
    out_dir: Path
    base: Candidate
    candidates: list[Candidate]
    ledger: Ledger
    settings: OptimizeSettings
    pick: str | None = None
    verification: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    selection: dict[str, Any] = field(default_factory=dict)

    @property
    def adopted(self) -> Candidate | None:
        return next((c for c in self.candidates if c.label == self.pick), None) if self.verdict == "improved" else None


class _Stop(Exception):
    """The spending cap was reached, or a probe failed: the loop ends and the base is kept."""


def _target(config: ExperimentConfig, raw: dict[str, Any], name: str) -> tuple[int, dict[str, Any]]:
    names = [system.resolved_name for system in config.systems]
    if name not in names:
        close = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise ValueError(f"The config has no system named '{name}'.{hint} (Systems: {', '.join(names)})")
    index = names.index(name)
    system = {key: value for key, value in raw["systems"][index].items() if key != "name"}
    return index, system


class _Optimizer:
    def __init__(
        self,
        config_path: Path,
        settings: OptimizeSettings,
        proposer: Proposer,
        out_dir: Path,
        *,
        force_mock: bool,
        max_workers: int | None,
        raw_config: dict[str, Any] | None,
        progress: Callable[[str], None],
    ):
        if settings.k < 1 or settings.finalists < 1 or settings.test_trials < 1:
            raise ValueError("k, finalists and test trials must each be at least 1.")
        self.config_path = config_path
        self.settings = settings
        self.proposer = proposer
        self.out = out_dir
        self.force_mock = force_mock
        self.max_workers = max_workers
        self.progress = progress
        self.raw = expand_sweeps(copy.deepcopy(raw_config)) if raw_config is not None else load_config_dict(config_path)
        self.config = ExperimentConfig.model_validate(self.raw)
        self.target_index, self.target_raw = _target(self.config, self.raw, settings.system)
        target = self.config.systems[self.target_index]
        if not field_names(target, effective_harness(target)):
            raise ValueError(f"`{settings.system}` (type {target.type}) has no harness RAGBench can change: its prompts and settings live outside RAGBench.")
        self.splits: Splits = load_splits(configured_splits_path(self.config.dataset))
        if not self.splits.train or not self.splits.test:
            raise ValueError("Optimizing needs train and test questions in the splits file (`ragbench split --train 0.5 --dev 0.1`).")
        dataset = load_dataset(self.config.dataset.questions_path, self.config.dataset.qrels_path)
        self.label_free = dataset.label_free
        self.ledger = Ledger(settings.k, settings.mode, settings.max_cost_usd)
        self.probed: dict[str, ProbeResult] = {}  # harness fingerprint -> its one-trial train probe
        self.candidates: list[Candidate] = []
        self.notes: list[str] = []
        self.started_at = datetime.now(UTC)
        self.library: Library | None = None
        self.library_named = False
        self.features: dict[str, Any] = {}
        self.lessons: list[tuple[str, str]] = []
        if settings.library and self.config.library.enabled:
            self.library, self.library_named = resolve_library(self.config.library.path)
            # What the library sees of this collection is its corpus and its train questions: dev and test questions never describe it.
            train = apply_split(dataset, self.splits, "train")
            self.features = features_of(profile_dataset(load_dataset_documents(self.config.dataset, warnings=[]), train))
            self.guard = evidence_guard(dataset, self.splits, "train")

    # -- state and resume ---------------------------------------------------------------------------------------------------------------

    def _check_state(self) -> None:
        payload = json.dumps({key: value for key, value in self.raw.items() if key != "run"}, sort_keys=True, default=str)
        state = {**self.settings.identity(), "config_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest()}
        path = self.out / STATE
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            started = previous.pop("started_at", None)  # when the first attempt began: a resume reads the library as of then, never what that attempt wrote
            if previous != state:
                changed = ", ".join(key for key in state if previous.get(key) != state[key])
                raise ValueError(f"{self.out} holds an optimization made with other settings ({changed}); choose another output folder.")
            if started:
                self.started_at = datetime.fromisoformat(started)
        self.out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**state, "started_at": self.started_at.isoformat(timespec="microseconds")}, indent=2) + "\n", encoding="utf-8")

    # -- probes -------------------------------------------------------------------------------------------------------------------------

    def _guard_budget(self, what: str) -> None:
        if self.ledger.budget.exhausted:
            self.ledger.stopped = f"the spending cap of ${self.ledger.budget.cap_usd:.4f} was reached before {what}"
            raise _Stop

    def _run(self, step: str, label: str, systems: dict[str, dict[str, Any]], split: str, directory: str, *, trials: int = 1, arena: bool = False, rollouts: int = 1):
        self._guard_budget(f"{step} `{label}`")
        self.progress(f"{step}: {label} on {split}" + (f" ({trials} trials)" if trials > 1 else ""))
        try:
            probe = run_probe(
                self.config_path, self.raw, systems, split, self.out / "probes" / directory,
                trials=trials, cap_usd=self.ledger.remaining, arena=arena, force_mock=self.force_mock, max_workers=self.max_workers,
            )  # fmt: skip
        except ProbeStopped as exc:
            self.ledger.add(Entry(step, label, exc.spent_usd, rollouts=0, detail={"split": split, "stopped": True}))
            self.ledger.stopped = f"the spending cap of ${self.ledger.budget.cap_usd:.4f} was reached during {step} `{label}`"
            raise _Stop from exc
        except ProbeFailed as exc:
            self.ledger.add(Entry(step, label, exc.spent_usd, rollouts=0, detail={"split": split, "failed": str(exc)}))
            self.ledger.stopped = f"{step} `{label}` failed ({exc}); its run folder is {self.out / 'probes' / directory}"
            raise _Stop from exc
        questions = len(next(iter(probe.results.values())).scores) if probe.results else 0
        detail = {"split": split, "questions": questions, "run_dir": str(probe.run_dir), "means": {name: result.mean for name, result in probe.results.items()}}
        self.ledger.add(Entry(step, label, probe.cost_usd, rollouts=rollouts, restored=probe.restored, detail=detail), probe.real_spend_usd)
        return probe

    def _train_probe(self, candidate: Candidate, step: str = "probe") -> None:
        """Probe a candidate on train, or read back the probe of the same harness made earlier in this attempt."""
        fingerprint = candidate.fingerprint
        detail = {"split": "train", "harness": candidate.raw.get("harness") or {}, "fingerprint": fingerprint}
        if fingerprint in self.probed:
            candidate.train = self.probed[fingerprint]
            self.ledger.add(Entry(step, candidate.label, reused=True, detail={**detail, "same_as": candidate.train.label, "mean": candidate.train.mean}))
            return
        probe = self._run(step, candidate.label, {candidate.label: candidate.raw}, "train", f"train-{candidate.label}")
        candidate.train = self.probed[fingerprint] = probe.results[candidate.label]
        self.ledger.entries[-1].detail.update(detail)

    # -- proposals ----------------------------------------------------------------------------------------------------------------------

    def _proposal(self, label: str, kind: ProposalKind, evidence: Path, parent: Candidate, variant: int, candidates: dict[str, dict[str, Any]] | None = None, *, evidence_trial: int | None = None) -> Candidate:
        step = "compose" if kind == "COMPOSE" else "propose"
        path = self.out / "proposals" / f"{label}.json"
        restored = path.is_file()
        if restored:
            saved = json.loads(path.read_text(encoding="utf-8"))
            proposal = Proposal(saved["kind"], saved["patch"], saved["harness"], saved["rationale"], saved["attempts"], saved["error"], CostBreakdown.model_validate(saved["cost"]))
        elif kind == "COMPOSE" and not candidates:
            proposal = Proposal(kind, {}, dict(parent.raw.get("harness") or {}), error="no candidate changed the harness: nothing to compose")
        else:
            self._guard_budget(f"the {kind} proposal `{label}`")
            self.progress(f"{step}: {label} ({kind})")
            proposal = self.proposer.propose(evidence, parent.config, kind, variant=variant, candidates=candidates)
        if not restored:
            path.parent.mkdir(parents=True, exist_ok=True)
            fields = {"kind": proposal.kind, "patch": proposal.patch, "harness": proposal.harness, "rationale": proposal.rationale, "attempts": proposal.attempts, "error": proposal.error}
            path.write_text(json.dumps({**fields, "cost": proposal.cost.model_dump()}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        detail = {
            "kind": kind, "parent": parent.label, "patch": proposal.patch, "inherited": proposal.inherited, "attempts": proposal.attempts,
            "error": proposal.error, "rationale": proposal.rationale, "evidence": str(evidence), "evidence_trial": evidence_trial,
        }  # fmt: skip
        self.ledger.add(Entry(step, label, proposal.cost.total_cost, restored=restored, detail=detail))
        raw = parent.raw if proposal.inherited else {**parent.raw, "harness": proposal.harness}
        candidate = Candidate(label, kind, parent.label, copy.deepcopy(raw), proposal)
        self.candidates.append(candidate)
        return candidate

    def _evidence(self, source: ProbeResult, name: str, trial: int | None = None) -> Path:
        return build_evidence(source.run_dir, system=source.label, feedback=self.settings.feedback, out_dir=self.out / "evidence" / name, trial=trial, library=self.lessons)

    def _pool_evidence(self, base: Candidate, candidates: list[Candidate], name: str) -> tuple[Path, dict[str, dict[str, Any]]]:
        """Evidence whose contrasts compare each candidate that changed the harness with `base`: a run folder holding their train rows."""
        changed = {c.label: c for c in candidates if c.proposal is not None and not c.proposal.inherited and c.fingerprint != base.fingerprint}
        pool = self.out / "pools" / name
        shutil.rmtree(pool, ignore_errors=True)
        pool.mkdir(parents=True)
        assert base.train is not None
        rows: list[dict[str, Any]] = []
        for label, result in ((base.label, base.train), *((c.label, c.train) for c in changed.values() if c.train is not None)):
            for row in read_jsonl(result.run_dir / "per_question_results.jsonl"):
                if row["system"] == result.label and int(row.get("trial", 0)) == 0:
                    rows.append({**{key: value for key, value in row.items() if key != "trial"}, "system": label})
        write_jsonl(pool / "per_question_results.jsonl", rows)
        systems = {base.label: base.raw, **{label: c.raw for label, c in changed.items()}}
        (pool / "config.yaml").write_text(yaml.safe_dump(probe_config(self.raw, systems, "train", trials=1, cap_usd=None), sort_keys=False), encoding="utf-8")
        shutil.copy(base.train.run_dir / "run_manifest.json", pool / "run_manifest.json")
        evidence = self._evidence(ProbeResult(base.label, "train", pool), name)
        return evidence, {label: c.proposal.patch for label, c in changed.items() if c.proposal is not None}

    # -- the two modes ------------------------------------------------------------------------------------------------------------------

    def parallel(self) -> Candidate:
        k = self.settings.k
        base = Candidate("base", "base", None, copy.deepcopy(self.target_raw))
        probe = self._run("base", base.label, {base.label: base.raw}, "train", "train-base", trials=k, rollouts=k)
        base.train = probe.results[base.label]
        self.ledger.entries[-1].detail.update({"harness": base.raw.get("harness") or {}, "fingerprint": base.fingerprint})
        self.probed[base.fingerprint] = base.train.trial(0)
        proposed: list[Candidate] = []
        for i in range(k):
            evidence = self._evidence(base.train, f"base-t{i}", trial=i)
            for kind in ("INV", "DOM"):
                proposed.append(self._proposal(f"p{i}-{kind.lower()}", kind, evidence, base, i, evidence_trial=i))
        for candidate in proposed:
            self._train_probe(candidate)
        evidence, patches = self._pool_evidence(base, proposed, "compose")
        compositions = [self._proposal(f"c{j}", "COMPOSE", evidence, base, j, patches) for j in range(k)]
        for candidate in compositions:
            self._train_probe(candidate)
        return base

    def sequential(self) -> Candidate:
        base = current = Candidate("r0-base", "base", None, copy.deepcopy(self.target_raw))
        for r in range(self.settings.k):
            if r:
                current = Candidate(f"r{r}-base", "base", current.label, copy.deepcopy(current.raw))
            self._train_probe(current, step="base")
            assert current.train is not None
            evidence = self._evidence(current.train, f"r{r}-base", trial=0)
            inv = self._proposal(f"r{r}-inv", "INV", evidence, current, r, evidence_trial=0)
            dom = self._proposal(f"r{r}-dom", "DOM", evidence, current, r, evidence_trial=0)
            self._train_probe(inv)
            self._train_probe(dom)
            pool, patches = self._pool_evidence(current, [inv, dom], f"r{r}")
            composition = self._proposal(f"r{r}-comp", "COMPOSE", pool, current, r, patches)
            self._train_probe(composition)
            contenders = [current, inv, dom, composition]
            best = max(contenders, key=lambda c: (c.train_mean if c.train_mean is not None else float("-inf"), -contenders.index(c)))
            if best is not current:
                self.notes.append(f"Round {r + 1} carries `{best.label}` forward ({best.train_mean:.3f} on train against {current.train_mean or 0:.3f}).")
            current = best
        return base

    # -- selection and verification -----------------------------------------------------------------------------------------------------

    def select(self, base: Candidate) -> tuple[Candidate | None, dict[str, Any]]:
        base_mean = base.train_mean if base.train_mean is not None else float("-inf")
        seen = {base.fingerprint}
        pool: list[Candidate] = []
        for candidate in self.candidates:
            if candidate.kind != "base" and candidate.train_mean is not None and candidate.fingerprint not in seen:
                seen.add(candidate.fingerprint)
                pool.append(candidate)
        better = sorted((c for c in pool if (c.train_mean or 0) > base_mean), key=lambda c: -(c.train_mean or 0))[: self.settings.finalists]
        info: dict[str, Any] = {"base_train": base.train_mean, "finalists": [c.label for c in better]}
        if not better:
            self.notes.append("No candidate beat the base on train, so nothing was run on dev or test.")
            return None, info
        if not self.splits.dev:
            self.notes.append(f"The splits have no dev questions, so `{better[0].label}` was picked on train (the best there); the test split still decides.")
            info["picked_on"] = "train"
            return better[0], info
        arena = self.settings.select == "arena" or (self.settings.select == "auto" and self.label_free)
        systems = {base.label: base.raw, **{c.label: c.raw for c in better}}
        probe = self._run("select", "dev", systems, "dev", "dev-select", arena=arena)
        values = {label: (result.arena_score if arena else result.mean) for label, result in probe.results.items()}
        for candidate in better:
            candidate.dev_value = values.get(candidate.label)
        info.update({"picked_on": "dev", "by": "arena" if arena else "score", "dev": values})
        ranked = [c for c in better if c.dev_value is not None]
        best = max(ranked, key=lambda c: c.dev_value or 0, default=None)
        base_dev = values.get(base.label)
        if best is None or base_dev is None or (best.dev_value or 0) <= base_dev:
            self.notes.append("No finalist beat the base on dev, so nothing was run on test.")
            return None, info
        return best, info

    def verify(self, base: Candidate, pick: Candidate) -> dict[str, Any]:
        trials = self.settings.test_trials
        probe = self._run("verify", "test", {base.label: base.raw, pick.label: pick.raw}, "test", "test-verify", trials=trials)
        base_scores, pick_scores = probe.results[base.label].scores, probe.results[pick.label].scores
        questions = sorted(set(base_scores) | set(pick_scores))
        result = grouped_paired_bootstrap([pick_scores.get(q, []) for q in questions], [base_scores.get(q, []) for q in questions], seed=self.config.evaluation.stats.seed)
        p_holm = holm_adjust([result.p_two_sided])[0]
        return {
            "pick": pick.label, "base_mean": probe.results[base.label].mean, "pick_mean": probe.results[pick.label].mean,
            "mean_diff": result.mean_diff, "ci_lo": result.ci_lo, "ci_hi": result.ci_hi, "p": result.p_two_sided, "p_holm": p_holm,
            "comparisons": 1, "wins": result.wins, "ties": result.ties, "losses": result.losses, "n": result.n, "trials": trials,
            "improved": bool(result.mean_diff > 0 and p_holm < ALPHA),
        }  # fmt: skip

    # -- the experience library ---------------------------------------------------------------------------------------------------------

    def read_library(self) -> None:
        """The lessons of the nearest earlier entries (older than this run's start, and of the same kind: mock or live), for every evidence tree."""
        if self.library is None:
            return
        try:
            matches = self.library.nearest(self.features, before=self.started_at, mock=self.force_mock)
        except LibraryError as exc:
            self.notes.append(f"Library: not read ({exc}). Continuing without it.")
            return
        self.lessons = [(match.entry.id, lesson_markdown(match.entry, match.similarity)) for match in matches]
        if self.lessons:
            self.notes.append(f"Library: the proposer read {len(self.lessons)} lesson(s) from earlier work on similar collections ({', '.join(match.entry.id for match in matches)}).")

    def write_library(self, result: OptimizeResult) -> None:
        """Keep an adopted harness, its verified gain and a lesson. Nothing else is worth a lesson: a gain the test split did not confirm is not one."""
        adopted = result.adopted
        if self.library is None or adopted is None or result.verification is None:
            return
        if self.force_mock and not self.library_named:
            self.notes.append("Library: nothing written, because a mock run only writes to a library you name (`library.path` or RAGBENCH_LIBRARY).")
            return
        entry = build_entry(
            features=self.features,
            system=self.settings.system,
            system_type=adopted.config.type,
            base_harness=effective_harness(result.base.config),
            adopted_harness=effective_harness(adopted.config),
            patch=dict(adopted.raw.get("harness") or {}),
            rationale=adopted.proposal.rationale if adopted.proposal is not None else "",
            verification=result.verification,
            mock=self.force_mock,
            run=str(self.out),
        )
        try:
            if any(old.run == entry.run and old.harness == entry.harness for old in self.library.entries()):
                self.notes.append("Library: this optimization is already recorded.")  # a resumed run that finished before
                return
            stored = self.library.append(entry, guard=self.guard)
        except (LeakageError, LibraryError) as exc:
            self.notes.append(f"Library: nothing written ({exc})")
            return
        self.notes.append(f"Library: recorded lesson {stored.id} in {self.library.path}.")

    def run(self) -> OptimizeResult:
        self._check_state()
        self.read_library()
        base: Candidate | None = None
        pick: Candidate | None = None
        verification: dict[str, Any] | None = None
        selection: dict[str, Any] = {}
        try:
            base = self.parallel() if self.settings.mode == "parallel" else self.sequential()
            self.ledger.assert_identity()
            pick, selection = self.select(base)
            if pick is not None:
                verification = self.verify(base, pick)
        except _Stop:
            self.notes.append(f"Stopped early: {self.ledger.stopped}. The base harness is kept.")
        if base is None:
            label = "base" if self.settings.mode == "parallel" else "r0-base"
            base = next((c for c in self.candidates if c.label == label), Candidate(label, "base", None, copy.deepcopy(self.target_raw)))
        verdict: Verdict = "stopped" if self.ledger.stopped else "improved" if verification and verification["improved"] else "no_reliable_gain"
        return OptimizeResult(verdict, self.out, base, self.candidates, self.ledger, self.settings, pick.label if pick else None, verification, self.notes, selection)


def optimize(
    config_path: Path,
    settings: OptimizeSettings,
    *,
    proposer: Proposer,
    out_dir: Path,
    force_mock: bool = False,
    max_workers: int | None = None,
    raw_config: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> OptimizeResult:
    """Optimize `settings.system`'s harness and write `optimized.yaml`, `optimization.md` and `ledger.json` to `out_dir`."""
    from ragbench.optimize.report import write_outputs

    optimizer = _Optimizer(config_path, settings, proposer, out_dir, force_mock=force_mock, max_workers=max_workers, raw_config=raw_config, progress=progress or (lambda _: None))
    result = optimizer.run()
    optimizer.write_library(result)
    write_outputs(result, optimizer.raw, optimizer.target_index)
    return result
