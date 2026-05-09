from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ragbench.config.schema import ExperimentConfig


def load_config(path: Path) -> ExperimentConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ExperimentConfig.model_validate(raw)


def load_config_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

