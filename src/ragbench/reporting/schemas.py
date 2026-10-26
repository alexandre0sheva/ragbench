"""The versioned JSON Schemas of a run's machine-readable files.

`run_summary.json` carries `schema_version`; `metrics_summary.csv` is versioned with it (a CSV has nowhere to put one without adding a column).
A change that could break a reader (a renamed or removed field, a changed meaning or unit) gets a new version and a new schema file; adding
an optional field does not.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

SCHEMA_VERSION = 1
SCHEMAS = ("run_summary", "metrics_summary")
SUPPORTED_VERSIONS = (1,)


def load_schema(name: str, version: int = SCHEMA_VERSION) -> dict[str, Any]:
    """The JSON Schema `name` (`run_summary` or `metrics_summary`) at `version`."""
    if name not in SCHEMAS:
        raise ValueError(f"No schema '{name}'. Schemas: {', '.join(SCHEMAS)}.")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"No version {version} of {name}. Versions: {', '.join(map(str, SUPPORTED_VERSIONS))}.")
    text = (resources.files("ragbench.schemas") / f"{name}.v{version}.schema.json").read_text(encoding="utf-8")
    loaded: dict[str, Any] = json.loads(text)
    return loaded
