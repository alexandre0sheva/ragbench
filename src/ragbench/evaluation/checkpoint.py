"""Per-system checkpoints: what lets a run be resumed (`ragbench auto --resume RUN_DIR`) without paying for finished systems again.

When a run is given its own directory, every system that finishes cleanly (all questions answered, none failed) is written to
`checkpoints/<system>.json` at once. A resumed run loads the checkpoints whose *context hash* still matches (the system's config,
every setting that changes its results, the dataset, the run mode) and runs only the rest. A system with failed questions is never
checkpointed: re-running it is the point of resuming.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragbench import __version__
from ragbench.config.schema import ExperimentConfig, SystemConfig

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = "checkpoints"
FORMAT_VERSION = 1
# Evaluation settings that change how fast or how safely a run goes, never what a system answers or scores.
NEUTRAL_EVALUATION_KEYS = frozenset(
    {"max_workers", "system_workers", "ingest_workers", "latency_probe_questions", "max_cost_usd", "cost_confirm_threshold_usd", "max_error_rate", "stats"}
)


def context_hash(system: SystemConfig, config: ExperimentConfig, dataset_digest: str, mode: str) -> str:
    """Hash of everything the system's results depend on. Change any of it and the checkpoint is stale."""
    evaluation = {key: value for key, value in config.evaluation.model_dump(mode="json").items() if key not in NEUTRAL_EVALUATION_KEYS}
    payload = {
        "version": __version__,
        "system": system.model_dump(mode="json"),
        "evaluation": evaluation,
        "pricing": {name: price.model_dump(mode="json") for name, price in sorted(config.pricing.items())},
        "tools": config.tools.model_dump(mode="json"),
        "providers": {name: provider.model_dump(mode="json") for name, provider in sorted(config.providers.items())},
        "dataset": dataset_digest,
        "mode": mode,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _plain(value: Any) -> Any:
    """JSON encoder hook: numpy scalars become Python numbers (anything else is a bug worth seeing, so it raises)."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


@dataclass
class Checkpoint:
    system: str
    system_type: str
    ingestion_row: dict[str, Any] | None
    results: list[dict[str, Any]]
    runtime_row: dict[str, Any]
    models_used: list[str] = field(default_factory=list)
    generator_model: tuple[str, str] | None = None  # (provider, model), for the self-preference check
    probe: dict[str, Any] | None = None  # the latency probe's measurements, once it has run


class CheckpointStore:
    """The `checkpoints/` folder of one run directory. Safe to call from the system worker threads."""

    def __init__(self, run_dir: Path):
        self.directory = run_dir / CHECKPOINT_DIR
        self._lock = threading.Lock()

    def path_for(self, system: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", system).strip("_")[:60] or "system"
        return self.directory / f"{slug}-{hashlib.sha256(system.encode('utf-8')).hexdigest()[:8]}.json"

    def save(self, checkpoint: Checkpoint, hash_: str) -> None:
        payload = {
            "format": FORMAT_VERSION,
            "context_hash": hash_,
            "system": checkpoint.system,
            "system_type": checkpoint.system_type,
            "ingestion_row": checkpoint.ingestion_row,
            "results": checkpoint.results,
            "runtime_row": checkpoint.runtime_row,
            "models_used": checkpoint.models_used,
            "generator_model": list(checkpoint.generator_model) if checkpoint.generator_model else None,
            "probe": checkpoint.probe,
        }
        self._write(checkpoint.system, payload)

    def _write(self, system: str, payload: dict[str, Any]) -> None:
        path = self.path_for(system)
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, default=_plain), encoding="utf-8")
            os.replace(temporary, path)  # a crash mid-write leaves the old file or none, never half of one

    def load(self, system: str, hash_: str) -> Checkpoint | None:
        """The checkpoint of `system` if it exists, is readable and was made in the same context; None otherwise (the system is re-run)."""
        path = self.path_for(system)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["format"] != FORMAT_VERSION or payload["context_hash"] != hash_ or payload["system"] != system:
                return None
            generator = payload.get("generator_model")
            return Checkpoint(
                system=payload["system"],
                system_type=payload["system_type"],
                ingestion_row=payload["ingestion_row"],
                results=payload["results"],
                runtime_row=payload["runtime_row"],
                models_used=list(payload.get("models_used", [])),
                generator_model=(str(generator[0]), str(generator[1])) if generator else None,
                probe=payload.get("probe"),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Ignoring unreadable checkpoint %s: %s", path.name, exc)
            return None

    def save_probe(self, system: str, probe: dict[str, Any]) -> None:
        """Add the latency probe's measurements to an existing checkpoint (the probe runs after every system finished)."""
        path = self.path_for(system)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        payload["probe"] = probe
        self._write(system, payload)
