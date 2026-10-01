"""Config sweeps: one system block with a `sweep:` becomes one system per combination of the listed values.

```yaml
systems:
  - type: hybrid_rerank
    name: hr
    sweep:
      chunker.chunk_size: [300, 500]
      retrieval.reranker: [local_relevance, cross_encoder]
      tools: [[], [calculator]]          # tool sets are an axis too
```

Variants are named `hr[chunk_size=300,reranker=cross_encoder]` and come in a fixed order (the first axis varies slowest), so the
same config always yields the same systems. Expansion happens before validation (see `load_config` and `ExperimentConfig`), so every
variant is checked like a hand-written system, and the run directory's `config.yaml` records the expanded systems.
"""

from __future__ import annotations

import copy
import difflib
import itertools
import json
from collections.abc import Sequence
from typing import Any

MAX_VARIANTS = 100  # one sweep may not expand to more systems than this: a typo in a list should not queue a thousand runs


def expand_sweeps(raw: dict[str, Any]) -> dict[str, Any]:
    """`raw` with every `sweep:` system replaced by its variants. Returns `raw` itself when there is nothing to expand; never mutates it."""
    systems = raw.get("systems")
    if not isinstance(systems, list) or not any(isinstance(system, dict) and "sweep" in system for system in systems):
        return raw
    expanded: list[Any] = []
    for index, system in enumerate(systems):
        if isinstance(system, dict) and "sweep" in system:
            expanded.extend(_variants(index, system))
        else:
            expanded.append(system)
    return {**raw, "systems": expanded}


def valid_axes(system_type: str) -> list[str]:
    """The `sweep:` keys a system type accepts: its chunker, retrieval and llm_features options, its models, and `tools` when it uses them."""
    import ragbench.rag_systems  # noqa: F401  (registers the built-in systems)
    from ragbench.registry import SYSTEMS

    spec = getattr(SYSTEMS.get(system_type), "spec", None)  # raises with a did-you-mean hint for an unknown type
    axes: list[str] = []
    if spec is not None:
        for section, model in (("chunker", spec.chunker), ("retrieval", spec.options), ("llm_features", spec.llm_features)):
            if model is not None:
                axes.extend(f"{section}.{name}" for name in model.model_fields)
        if spec.supports_tools:
            axes.append("tools")
    axes.extend(["models.generator", "models.embedding"])
    return sorted(axes)


def axis_labels(axes: Sequence[str]) -> list[str]:
    """How each axis appears in a variant's name: its last path segment, or the whole path when two axes would end the same way."""
    last = [axis.rsplit(".", 1)[-1] for axis in axes]
    return [axis if last.count(label) > 1 else label for axis, label in zip(axes, last, strict=True)]


def _label(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "+".join(_label(item.get("name") if isinstance(item, dict) else item) for item in value) or "none"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _set_path(system: dict[str, Any], axis: str, value: Any) -> None:
    section, _, option = axis.partition(".")
    if not option:  # `tools`
        system[axis] = copy.deepcopy(value)
        return
    target = system.get(section)
    if not isinstance(target, dict):
        target = system[section] = {}
    target[option] = copy.deepcopy(value)


def _variants(index: int, system: dict[str, Any]) -> list[dict[str, Any]]:
    base_name = system.get("name") or system.get("type")
    where = f"systems[{index}] ('{base_name}', type {system.get('type')})"
    sweep = system["sweep"]
    if not isinstance(sweep, dict):
        raise ValueError(f"{where}: `sweep` must be a mapping of axis -> list of values")
    if not sweep:
        raise ValueError(f"{where}: `sweep` needs at least one axis")
    valid = valid_axes(str(system.get("type")))
    for axis, values in sweep.items():
        if axis not in valid:
            close = difflib.get_close_matches(str(axis), valid, n=1, cutoff=0.6)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            nested = " Nested paths into an option's value are not supported." if str(axis).count(".") > 1 else ""
            raise ValueError(f"{where}: `sweep` axis '{axis}' is not valid.{hint}{nested} Valid axes: {', '.join(valid)}")
        if not isinstance(values, list):
            raise ValueError(f"{where}: sweep axis '{axis}' must be a list of values (got {type(values).__name__})")
        if not values:
            raise ValueError(f"{where}: sweep axis '{axis}' needs at least one value")
    axes = list(sweep)
    count = 1
    for axis in axes:
        count *= len(sweep[axis])
    if count > MAX_VARIANTS:
        raise ValueError(f"{where}: this sweep expands to {count} systems, more than the limit of {MAX_VARIANTS}. Split it or shorten an axis.")
    labels = axis_labels(axes)
    template = {key: value for key, value in system.items() if key != "sweep"}
    variants: list[dict[str, Any]] = []
    for combination in itertools.product(*(sweep[axis] for axis in axes)):
        variant = copy.deepcopy(template)
        for axis, value in zip(axes, combination, strict=True):
            _set_path(variant, axis, value)
        parts = ",".join(f"{label}={_label(value)}" for label, value in zip(labels, combination, strict=True))
        variant["name"] = f"{base_name}[{parts}]"
        variants.append(variant)
    return variants


# -- choosing systems by name (`--systems`) ---------------------------------------------------------


def split_system_names(values: Sequence[str]) -> list[str]:
    """Names from `--systems` values: each value may hold several, separated by commas that are not inside a variant's `[...]`."""
    names: list[str] = []
    for value in values:
        depth, current = 0, ""
        for char in value:
            depth += (char == "[") - (char == "]")
            if char == "," and depth <= 0:
                names.append(current.strip())
                current = ""
            else:
                current += char
        names.append(current.strip())
    return [name for name in names if name]


def _matches(name: str, request: str) -> bool:
    return name == request or name.startswith(request + "[")


def _system_names(raw: dict[str, Any], wanted: Sequence[str], flag: str) -> tuple[list[dict[str, Any]], list[str]]:
    """The config's systems with their names, after checking that every requested name (or sweep base name) exists."""
    systems = raw.get("systems") or []
    names = [str(system.get("name") or system.get("type")) for system in systems]
    for request in wanted:
        if not any(_matches(name, request) for name in names):
            close = difflib.get_close_matches(request, names, n=1, cutoff=0.6)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise ValueError(f"{flag}: no system named '{request}'.{hint} Available: {', '.join(names)}")
    return systems, names


def select_systems(raw: dict[str, Any], wanted: Sequence[str]) -> dict[str, Any]:
    """`raw` keeping only the named systems (config order). A sweep's base name (`hr`) selects all of its variants (`hr[...]`)."""
    systems, names = _system_names(raw, wanted, "--only")
    return {**raw, "systems": [system for system, name in zip(systems, names, strict=True) if any(_matches(name, request) for request in wanted)]}


def skip_systems(raw: dict[str, Any], unwanted: Sequence[str]) -> dict[str, Any]:
    """`raw` without the named systems (a sweep's base name drops all of its variants). Skipping every system is an error."""
    systems, names = _system_names(raw, unwanted, "--skip")
    kept = [system for system, name in zip(systems, names, strict=True) if not any(_matches(name, request) for request in unwanted)]
    if not kept:
        raise ValueError(f"--skip: that leaves no system to run (skipped: {', '.join(unwanted)}).")
    return {**raw, "systems": kept}
