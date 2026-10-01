"""The recommendation engine: from a finished run directory to "deploy this one".

Reads what the run wrote (`metrics_summary.csv`, `per_question_results.jsonl`, `cost_breakdown.csv`, `run_summary.json`,
`stats.json`, and the `config.yaml` copy), so it works on any past run and can be re-asked with other constraints.

The rule (docs/methodology.md#selection): drop the systems that break a constraint; find the best answer quality among
the rest; every system whose quality is *statistically indistinguishable* from it (paired bootstrap, Holm-adjusted) ties
with it; among the tied, deploy the one that is best on cost and latency by the profile's weights, then the simplest.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import yaml

from ragbench.evaluation.stats import ALPHA, holm_adjust, paired_bootstrap, pareto_front
from ragbench.selection import explain
from ragbench.selection.constraints import Constraints, SystemFacts, Violation, check_constraints
from ragbench.selection.scoring import Profile, Weights, efficiency_penalties, resolve_profile, weighted_scores

DEFAULT_N_BOOT = 2000
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"}  # noqa: S104 (a host name to recognize, not an address to bind)


@dataclass
class ScoredSystem:
    """One feasible system with the numbers the decision rests on."""

    system: str
    score: float  # weighted, min-max normalized across the feasible systems (0-1): for ranking and display
    quality: float | None
    cost_per_question: float | None
    latency_ms_p95: float | None
    ingestion_cost: float
    faithfulness: float | None
    llm_calls: float  # mean model calls per question (the simplicity tie-break)
    steps: float  # mean traced steps per question
    tied: bool = False  # statistically indistinguishable from the best quality
    pareto: bool = False


@dataclass
class Recommendation:
    winner: str | None
    ranking: list[ScoredSystem]
    tied_with_winner: list[str]
    pareto: list[str]
    by_category: dict[str, str]
    rationale: list[str]
    infeasible: dict[str, list[str]]  # system -> the constraints it violates
    profile: str = "balanced"
    weights: Weights = field(default_factory=Weights)
    constraints: Constraints = field(default_factory=Constraints)
    quality_metric: str = "answer_score"
    category_scores: dict[str, dict[str, float]] = field(default_factory=dict)  # category -> system -> mean quality (feasible systems)
    closest_miss: str | None = None  # when nothing is feasible: the system that comes closest
    mode: str = "live"
    n_questions: int = 0
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        def clean(value: Any) -> Any:
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value

        return {
            "winner": self.winner,
            "profile": self.profile,
            "weights": self.weights.model_dump(),
            "constraints": self.constraints.model_dump(),
            "quality_metric": self.quality_metric,
            "mode": self.mode,
            "n_questions": self.n_questions,
            "run_id": self.run_id,
            "tied_with_winner": self.tied_with_winner,
            "pareto": self.pareto,
            "by_category": self.by_category,
            "category_scores": self.category_scores,
            "ranking": [{key: clean(value) for key, value in vars(row).items()} for row in self.ranking],
            "infeasible": self.infeasible,
            "closest_miss": self.closest_miss,
            "rationale": self.rationale,
        }


@dataclass
class _Run:
    summary: dict[str, dict[str, Any]]  # system -> metrics_summary row (NaN -> None), in the run's order
    rows: dict[str, dict[str, dict[str, Any]]]  # system -> question id -> per_question row (successful questions only)
    ingestion: dict[str, float]
    mode: str
    primary_k: int
    n_boot: int
    seed: int
    run_id: str
    raw_config: dict[str, Any] | None


def _load_run(run_dir: Path) -> _Run:
    summary_path, rows_path = run_dir / "metrics_summary.csv", run_dir / "per_question_results.jsonl"
    for path in (summary_path, rows_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: {run_dir} is not a finished RAGBench run directory")
    frame = pd.read_csv(summary_path)
    summary = {
        str(record["system"]): {key: (None if pd.isna(value) else value) for key, value in record.items()} for record in frame.to_dict("records")
    }
    rows: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in summary}
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["error"] is None and row["system"] in rows:
            rows[row["system"]][row["question_id"]] = row
    ingestion: dict[str, float] = {name: 0.0 for name in summary}
    costs_path = run_dir / "cost_breakdown.csv"
    if costs_path.exists():
        costs = pd.read_csv(costs_path)
        if "stage" in costs.columns:
            for system, spent in costs[costs["stage"] == "ingestion"].groupby("system")["total_cost"].sum().items():
                ingestion[str(system)] = float(spent)
    run_summary = _read_json(run_dir / "run_summary.json")
    stats = _read_json(run_dir / "stats.json")
    config_path = run_dir / "config.yaml"
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
    return _Run(
        summary=summary,
        rows=rows,
        ingestion=ingestion,
        mode=str(run_summary.get("mode", "live")),
        primary_k=int(run_summary.get("primary_k", 5)),
        n_boot=int(stats.get("n_boot", DEFAULT_N_BOOT)),
        seed=int(stats.get("seed", 0)),
        run_id=str(run_summary.get("run_id", run_dir.name)),
        raw_config=raw_config if isinstance(raw_config, dict) else None,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# -- quality ------------------------------------------------------------------------------------


def _quality_candidates(primary_k: int) -> list[tuple[str, str, Any]]:
    """(metric, label, extractor over a per-question row): the metric a run's quality is judged by, best first."""
    return [
        ("answer_score", "answer score", lambda row: row["answer_judge"]["answer_score"]),
        ("token_f1", "token F1", lambda row: (row.get("answer_metrics") or {}).get("token_f1")),
        (f"recall@{primary_k}", f"recall@{primary_k}", lambda row: (row.get("retrieval_metrics") or {}).get(f"recall@{primary_k}")),
    ]


def _vectors(run: _Run, extract: Any) -> dict[str, dict[str, float]]:
    found: dict[str, dict[str, float]] = {}
    for system, rows in run.rows.items():
        values = {question_id: extract(row) for question_id, row in rows.items()}
        found[system] = {question_id: float(value) for question_id, value in values.items() if value is not None and not math.isnan(float(value))}
    return found


def _choose_quality(run: _Run) -> tuple[str, str, dict[str, dict[str, float]]]:
    """The first metric every system has; failing that, the first one any system has (the others are then not comparable)."""
    candidates = [(metric, label, _vectors(run, extract)) for metric, label, extract in _quality_candidates(run.primary_k)]
    for metric, label, vectors in candidates:
        if all(vectors[system] for system in run.summary):
            return metric, label, vectors
    for metric, label, vectors in candidates:
        if any(vectors[system] for system in run.summary):
            return metric, label, vectors
    return candidates[0]


def _cost_or_inf(value: float | None) -> float:
    return math.inf if value is None else value


def _mean(values: dict[str, float]) -> float | None:
    return float(np.mean(list(values.values()))) if values else None


# -- constraints that need the run's config -----------------------------------------------------


def _is_local(ref: str, providers: dict[str, Any]) -> bool:
    from ragbench.models.refs import parse_model_ref, split_endpoint

    provider, model = parse_model_ref(ref)
    if provider == "local":
        return True
    if provider == "openai_compatible":
        endpoint = providers.get(split_endpoint(model)[0])
        host = (urlparse(endpoint.base_url).hostname or "") if endpoint is not None else ""
        return host in LOOPBACK_HOSTS or host.endswith(".localhost") or host.startswith("127.")
    return False


def _locality(config: Any, system_name: str) -> tuple[list[str] | None, list[str] | None]:
    """(models that do not run on this machine, tools that reach the network) for one system and its routes; None = unknown."""
    from ragbench.config.schema import _route_configs
    from ragbench.models.refs import system_model_refs
    from ragbench.tools.registry import resolve_tools

    system = next((s for s in config.systems if s.resolved_name == system_name), None)
    if system is None:
        return None, None
    non_local: list[str] = []
    network: list[str] = []
    try:
        for member in [system, *(cfg for _, cfg in _route_configs(system))]:
            refs = system_model_refs(member)
            non_local += [ref for ref in [*refs.llm, *refs.embedding] if ref not in non_local and not _is_local(ref, config.providers)]
            network += [tool.spec.name for tool in resolve_tools(member.tools, config.tools) if tool.side_effects == "network"]
    except Exception:  # noqa: BLE001 (a config that cannot be inspected is reported as unknown, not as a crash)
        return None, None
    return non_local, network


def _config_or_none(raw: dict[str, Any] | None) -> Any:
    if raw is None:
        return None
    from ragbench.config.schema import ExperimentConfig

    try:
        return ExperimentConfig.model_validate(raw)
    except Exception:  # noqa: BLE001
        return None


# -- the engine ---------------------------------------------------------------------------------


def recommend(
    run_dir: Path,
    *,
    constraints: Constraints | None = None,
    weights: Weights | None = None,
    profile: str | None = None,
    alpha: float = ALPHA,
) -> Recommendation:
    """Which system of the run to deploy, why, and what is statistically a tie. See the module docstring for the rule."""
    run_dir = Path(run_dir)
    run = _load_run(run_dir)
    constraints = constraints or Constraints()
    profile_name = profile or "balanced"
    chosen: Profile = resolve_profile(profile_name)
    weights = weights or chosen.weights
    metric, label, vectors = _choose_quality(run)
    n_questions = max((len(rows) for rows in run.rows.values()), default=0)

    config = _config_or_none(run.raw_config) if (constraints.require_local_models or constraints.require_no_network) else None
    infeasible: dict[str, list[str]] = {}
    violations: dict[str, list[Violation]] = {}
    for system, row in run.summary.items():
        non_local, network = _locality(config, system) if config is not None else (None, None)
        facts = SystemFacts(
            system=system,
            cost_per_question=row.get("avg_cost_per_question"),
            latency_ms_p95=row.get("latency_ms_p95"),
            faithfulness=row.get("faithfulness"),
            answer_score=row.get("answer_score"),
            ingestion_cost=run.ingestion.get(system, 0.0),
            non_local_models=non_local,
            network_tools=network,
        )
        found = check_constraints(constraints, facts)
        if not vectors[system]:
            found.append(Violation("quality", f"no {label} was measured for it, so it cannot be compared", math.inf))
        if found:
            violations[system] = found
            infeasible[system] = [violation.message for violation in found]
    feasible = [system for system in run.summary if system not in violations]

    base = Recommendation(
        winner=None,
        ranking=[],
        tied_with_winner=[],
        pareto=[],
        by_category={},
        rationale=[],
        infeasible=infeasible,
        profile=profile_name,
        weights=weights,
        constraints=constraints,
        quality_metric=metric,
        mode=run.mode,
        n_questions=n_questions,
        run_id=run.run_id,
    )
    if not feasible:
        base.closest_miss = _closest_miss(violations)
        base.rationale = explain.explain_no_winner(infeasible, base.closest_miss, run.mode, label)
        return base

    quality = {system: _mean(vectors[system]) for system in feasible}
    cost = {system: run.summary[system].get("avg_cost_per_question") for system in feasible}
    latency = {system: run.summary[system].get("latency_ms_p95") for system in feasible}
    scores = weighted_scores(quality, cost, latency, weights)
    steps = {system: _step_counts(run.rows[system]) for system in feasible}

    leader = min(feasible, key=lambda system: (-(quality[system] or 0.0), _cost_or_inf(cost[system]), system))
    ties = _quality_ties(run, vectors, feasible, leader, merge=chosen.merge_ties, alpha=alpha)
    tied = [leader, *[system for system in feasible if system != leader and ties[system].tied]]
    penalties = efficiency_penalties({s: cost[s] for s in tied}, {s: latency[s] for s in tied}, weights)

    def tie_key(system: str) -> tuple[float, float, float, float, str]:
        return (round(penalties[system], 9), _cost_or_inf(cost[system]), steps[system][0], steps[system][1], system)

    tied.sort(key=tie_key)
    winner = tied[0]

    points = [{"system": s, "quality": quality[s], "cost": cost[s], "latency": run.summary[s].get("avg_latency_ms")} for s in feasible]
    front = pareto_front(points, maximize=["quality"], minimize=["cost", "latency"])

    def scored(system: str) -> ScoredSystem:
        return ScoredSystem(
            system=system,
            score=scores[system],
            quality=quality[system],
            cost_per_question=cost[system],
            latency_ms_p95=latency[system],
            ingestion_cost=run.ingestion.get(system, 0.0),
            faithfulness=run.summary[system].get("faithfulness"),
            llm_calls=steps[system][0],
            steps=steps[system][1],
            tied=system in tied,
            pareto=system in front,
        )

    rest = sorted((s for s in feasible if s not in tied), key=lambda s: (-scores[s], _cost_or_inf(cost[s]), s))
    ranking = [scored(system) for system in [*tied, *rest]]
    categories = _by_category(run, vectors, feasible, cost)

    cheapest = min((s for s in feasible if cost[s] is not None), key=lambda s: (cost[s], s), default=None)
    versus = [other for other in dict.fromkeys([leader, cheapest]) if other is not None and other != winner]
    comparisons = [_compare(run, vectors, winner, other, cost, alpha) for other in versus]

    base.winner = winner
    base.ranking = ranking
    base.tied_with_winner = [system for system in tied if system != winner]
    base.pareto = front
    base.by_category = {category: scores_for[0] for category, scores_for in categories.items()}
    base.category_scores = {category: dict(by_system) for category, (_, by_system) in categories.items()}
    base.rationale = explain.explain_winner(
        ranking=ranking,
        leader=leader,
        ties=ties,
        tied=tied,
        comparisons=comparisons,
        label=label,
        profile_name=profile_name,
        profile=chosen,
        weights=weights,
        infeasible=infeasible,
        mode=run.mode,
        n_questions=n_questions,
        constraints=constraints,
        alpha=alpha,
    )
    return base


def _step_counts(rows: dict[str, dict[str, Any]]) -> tuple[float, float]:
    """(mean model calls, mean traced steps) per question: how much machinery a system runs. Calls are the steps that used tokens."""
    if not rows:
        return 0.0, 0.0
    llm = [sum(1 for step in row["steps"] if step.get("prompt_tokens", 0) or step.get("completion_tokens", 0)) for row in rows.values()]
    return float(np.mean(llm)), float(np.mean([len(row["steps"]) for row in rows.values()]))


def _quality_ties(
    run: _Run, vectors: dict[str, dict[str, float]], feasible: list[str], leader: str, *, merge: bool, alpha: float
) -> dict[str, explain.TieTest]:
    """Does each system's quality differ from the leader's? Paired on shared questions, Holm-adjusted across the systems tested."""
    others = [system for system in feasible if system != leader]
    results: dict[str, explain.TieTest] = {}
    if not merge:  # ties are not merged: only exactly equal quality counts
        leader_mean = _mean(vectors[leader])
        return {system: explain.TieTest(0.0, 1.0, tied=_mean(vectors[system]) == leader_mean) for system in others}
    tests: dict[str, Any] = {}
    for system in others:
        shared = [q for q in vectors[leader] if q in vectors[system]]
        if shared:
            tests[system] = paired_bootstrap(
                np.array([vectors[leader][q] for q in shared]), np.array([vectors[system][q] for q in shared]), n_boot=run.n_boot, seed=run.seed
            )
    adjusted = dict(zip(tests, holm_adjust([test.p_two_sided for test in tests.values()]), strict=True))
    for system in others:
        test = tests.get(system)
        if test is None:  # nothing measured on both: no evidence of a tie
            results[system] = explain.TieTest(math.nan, math.nan, tied=False)
            continue
        worse = adjusted[system] < alpha and test.mean_diff > 0  # the leader is significantly better
        results[system] = explain.TieTest(test.mean_diff, adjusted[system], tied=not worse)
    return results


def _compare(run: _Run, vectors: dict[str, dict[str, float]], winner: str, other: str, cost: dict[str, float | None], alpha: float) -> explain.Comparison:
    shared = [q for q in vectors[winner] if q in vectors[other]]
    paired = paired_bootstrap(
        np.array([vectors[winner][q] for q in shared]), np.array([vectors[other][q] for q in shared]), n_boot=run.n_boot, seed=run.seed, alpha=alpha
    )
    winner_cost, other_cost = cost[winner], cost[other]
    ratio = winner_cost / other_cost if winner_cost is not None and other_cost else None
    significant = bool(paired.n) and (paired.ci_lo > 0 or paired.ci_hi < 0)
    return explain.Comparison(winner, other, paired.mean_diff, paired.ci_lo, paired.ci_hi, significant, ratio, paired.n)


def _by_category(
    run: _Run, vectors: dict[str, dict[str, float]], feasible: list[str], cost: dict[str, float | None]
) -> dict[str, tuple[str, dict[str, float]]]:
    """Per question category: the best feasible system by mean quality (the cheaper on an exact tie) and everyone's mean. Descriptive only."""
    by_category: dict[str, dict[str, list[float]]] = {}
    for system in feasible:
        for question_id, value in vectors[system].items():
            category = str(run.rows[system][question_id].get("category") or "unknown")
            by_category.setdefault(category, {}).setdefault(system, []).append(value)
    result: dict[str, tuple[str, dict[str, float]]] = {}
    for category in sorted(by_category):
        means = {system: float(np.mean(values)) for system, values in by_category[category].items()}
        best = min(means, key=lambda s: (-means[s], _cost_or_inf(cost[s]), s))
        result[category] = (best, means)
    return result


def _closest_miss(violations: dict[str, list[Violation]]) -> str | None:
    """The system that breaks the fewest constraints, and by the least (as a ratio of its worst violation)."""
    if not violations:
        return None
    return min(violations, key=lambda system: (len(violations[system]), max(v.severity for v in violations[system]), system))


def selection_of_run(run_dir: Path) -> Any:
    """The `selection:` section the run was made with (the defaults when its config cannot be read)."""
    from ragbench.config.schema import SelectionConfig

    raw = _load_run_config(Path(run_dir))
    try:
        return SelectionConfig.model_validate((raw or {}).get("selection") or {})
    except Exception:  # noqa: BLE001
        return SelectionConfig()


def _load_run_config(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "config.yaml"
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    return loaded if isinstance(loaded, dict) else None


# -- artifacts ----------------------------------------------------------------------------------


def winner_yaml_text(run_dir: Path, winner: str) -> str:
    """A config that runs only the winner: its system block, plus everything the system needs from the run's `config.yaml`.

    The `dataset:` block is the benchmark's: point it at your production documents before deploying.
    """
    raw = yaml.safe_load((Path(run_dir) / "config.yaml").read_text(encoding="utf-8")) or {}
    systems = [s for s in raw.get("systems", []) if (s.get("name") or s.get("type")) == winner]
    if not systems:
        raise ValueError(f"system '{winner}' is not in {Path(run_dir) / 'config.yaml'}")
    run_block = {**raw.get("run", {}), "name": f"{raw.get('run', {}).get('name', 'run')}_winner"}
    evaluation = {key: value for key, value in (raw.get("evaluation") or {}).items() if key != "stats"}
    stats = {key: value for key, value in ((raw.get("evaluation") or {}).get("stats") or {}).items() if key != "baseline"}  # the baseline is not here any more
    if stats:
        evaluation["stats"] = stats
    out: dict[str, Any] = {"run": run_block, "dataset": raw.get("dataset", {}), "systems": systems}
    for key in ("providers", "tools", "pricing", "limits", "cache"):
        if key in raw:
            out[key] = raw[key]
    if evaluation:
        out["evaluation"] = evaluation
    header = (
        f"# The winner of {Path(run_dir).name}: '{winner}'. Run it with `ragbench run --config <this file>`.\n"
        "# `dataset:` is the benchmark's data; replace it with your production documents before deploying.\n"
    )
    return header + yaml.safe_dump(out, sort_keys=False)


def write_recommendation(run_dir: Path, recommendation: Recommendation) -> dict[str, Path]:
    """Write `recommendation.json`, `recommendation.md` and (when there is a winner) `winner.yaml` into `run_dir`."""
    run_dir = Path(run_dir)
    paths = {"json": run_dir / "recommendation.json", "markdown": run_dir / "recommendation.md"}
    paths["json"].write_text(json.dumps(recommendation.to_dict(), indent=2), encoding="utf-8")
    paths["markdown"].write_text(explain.render_markdown(recommendation), encoding="utf-8")
    if recommendation.winner is not None:
        paths["winner"] = run_dir / "winner.yaml"
        paths["winner"].write_text(winner_yaml_text(run_dir, recommendation.winner), encoding="utf-8")
    return paths
