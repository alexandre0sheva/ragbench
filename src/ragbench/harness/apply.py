"""Applying a system's `harness:` patch when its config loads: knobs are written into `retrieval:` / `chunker:`, prompt fields are checked
against the prompts the system actually sends."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from ragbench.harness.prompts import PromptHarness
from ragbench.harness.schema import KNOBS, HarnessPatch

if TYPE_CHECKING:
    from ragbench.config.schema import SystemConfig
    from ragbench.rag_systems.spec import SystemSpec


def knob_targets(spec: SystemSpec) -> dict[str, tuple[str, str]]:
    """The knobs a system has, each as `(section, option)` it sets."""
    targets: dict[str, tuple[str, str]] = {}
    for knob, (section, fields) in KNOBS.items():
        model = spec.options if section == "retrieval" else spec.chunker
        found = next((name for name in fields if model is not None and name in model.model_fields), None)
        if found is not None:
            targets[knob] = (section, found)
    return targets


def apply_harness(config: SystemConfig, label: str) -> None:
    """Check `config.harness` against the system and write its knobs into the sections (in place). Raises ValueError with `label`."""
    import ragbench.rag_systems  # noqa: F401  (registers the built-in systems)
    from ragbench.registry import SYSTEMS

    patch = HarnessPatch.model_validate(config.harness)
    cls = SYSTEMS.mapping.get(config.type)
    spec = getattr(cls, "spec", None)
    if cls is None:
        return  # an unknown type is reported by ExperimentConfig
    if spec is not None and spec.external:
        raise ValueError(f"{label} does not take a `harness:`: its prompts live outside RAGBench")
    knobs = patch.knobs()
    if knobs:
        targets = knob_targets(spec) if spec is not None else {}
        missing = [knob for knob in knobs if knob not in targets]
        if missing:
            raise ValueError(f"{label}: harness: this system has no knob {', '.join(f'`{knob}`' for knob in missing)} (its knobs: {', '.join(targets) or 'none'})")
        sections: dict[str, dict[str, Any]] = {"retrieval": dict(config.retrieval), "chunker": dict(config.chunker)}
        for knob, value in knobs.items():
            section, option = targets[knob]
            sections[section][option] = value
        config.retrieval, config.chunker = sections["retrieval"], sections["chunker"]
    options_model = spec.options if spec is not None else None
    try:
        options = options_model.model_validate(config.retrieval) if options_model is not None else None
    except ValidationError as exc:
        _raise_knob_errors(label, exc, knob_targets(spec) if spec is not None else {}, knobs)
        return  # a mistake the harness did not make: the section check reports it
    try:
        PromptHarness.resolve(cls.default_harness(options), patch)
    except ValueError as exc:
        raise ValueError(f"{label}: harness: {exc}") from None


def _raise_knob_errors(label: str, exc: ValidationError, targets: dict[str, tuple[str, str]], knobs: dict[str, object]) -> None:
    """Report an option a knob set under the knob's name; return when none of the errors is a knob's (the section check reports them)."""
    by_option = {option: knob for knob, (section, option) in targets.items() if section == "retrieval" and knob in knobs}
    problems = [f"harness.{by_option[str(error['loc'][0])]}: {str(error['msg']).removeprefix('Value error, ')}" for error in exc.errors() if error["loc"] and str(error["loc"][0]) in by_option]
    if problems:
        raise ValueError(f"{label}: " + "; ".join(problems)) from None


def effective_harness(config: SystemConfig) -> dict[str, Any]:
    """Everything a system's harness decides, after its patch: each prompt it sends, the few-shot examples, and the value of each knob it has.
    `{}` for a system with no harness (an external one, an unknown type)."""
    import ragbench.rag_systems  # noqa: F401  (registers the built-in systems)
    from ragbench.registry import SYSTEMS

    cls = SYSTEMS.mapping.get(config.type)
    spec = getattr(cls, "spec", None)
    if cls is None or (spec is not None and spec.external):
        return {}
    options = spec.options.model_validate(config.retrieval) if spec is not None else None
    harness = PromptHarness.resolve(cls.default_harness(options), HarnessPatch.model_validate(config.harness))
    found: dict[str, Any] = dict(harness.fields())
    if harness.few_shot:
        found["few_shot"] = [example.model_dump() for example in harness.few_shot]
    if spec is None:
        return found
    from ragbench.registry import CHUNKERS

    sections: dict[str, Any] = {"retrieval": spec.options.model_validate(config.retrieval), "chunker": spec.chunker.model_validate(config.chunker) if spec.chunker is not None else None}
    for knob, (section, option) in knob_targets(spec).items():
        # `top_k` as the system reads it (`final_top_k` or its `top_k` alias); None = unset, the run's `evaluation.context_k` applies.
        value = sections["retrieval"].configured_top_k if knob == "top_k" else getattr(sections[section], option)
        if knob == "chunk_size" and value is None:  # unset: the chunker's own default
            value = CHUNKERS.get(sections["chunker"].type).default_chunk_size
        found[knob] = value
    return found


def harness_fingerprint(config: SystemConfig) -> str:
    """A short, stable hash of `effective_harness(config)`: what identifies a harness in checkpoints and optimizer runs."""
    payload = json.dumps(effective_harness(config), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
