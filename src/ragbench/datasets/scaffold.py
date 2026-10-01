"""`ragbench init`: a starter project for your own documents (a config with a preset's systems, and a questions file to edit)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ragbench.config.presets import apply_preset, preset_systems
from ragbench.config.schema import ExperimentConfig
from ragbench.config.sweep import expand_sweeps
from ragbench.documents.loaders import DocumentLoadError, load_documents
from ragbench.documents.schema import Document
from ragbench.models.errors import MissingExtraError
from ragbench.utils.jsonl import write_jsonl

CONFIG_NAME = "ragbench.yaml"
QUESTIONS_NAME = "questions.jsonl"
TEMPLATE_QUESTIONS = 3
DEFAULT_MAX_COST_USD = 5.0  # a safety net for a first live run; the config says how to remove it


class ScaffoldError(ValueError):
    """`init` cannot proceed (bad input, or it would overwrite a file); nothing has been written."""


@dataclass
class ScaffoldResult:
    config_path: Path
    questions_path: Path
    created_questions: bool
    n_documents: int
    preset: str


def _template_questions(documents: list[Document]) -> list[dict[str, Any]]:
    """Placeholder questions, one per first documents, so the project runs before the real questions exist. They are marked `template` and label-free."""
    rows = []
    for number, document in enumerate(sorted(documents, key=lambda d: d.doc_id)[:TEMPLATE_QUESTIONS], start=1):
        title = re.sub(r"\s+", " ", document.title).strip()[:80] or document.doc_id
        rows.append({"id": f"q_{number:03d}", "question": f"What are the key points of \"{title}\"?", "answerable": True, "metadata": {"template": True}})
    return rows


def _config_text(config: dict[str, Any], preset: str) -> str:
    def dump(section: dict[str, Any]) -> str:
        return yaml.safe_dump(section, sort_keys=False, default_flow_style=False, allow_unicode=True)

    return (
        "# Written by `ragbench init`. Edit freely: docs/configuration.md describes every key.\n"
        "# Paths are relative to the directory you run ragbench from.\n"
        f"{dump({'run': config['run'], 'dataset': config['dataset']})}\n"
        f"# The systems to compare (the `{preset}` preset). `ragbench list-systems` shows the rest.\n"
        f"{dump({'systems': config['systems']})}\n"
        "# `max_cost_usd` stops a live run once it has been charged this much; remove it to run without a cap.\n"
        f"{dump({'evaluation': config['evaluation'], 'selection': config['selection']})}"
    )


def scaffold_project(directory: Path, docs: Path, questions: Path | None = None, *, preset: str = "standard", force: bool = False) -> ScaffoldResult:
    """Write `ragbench.yaml` (and `questions.jsonl` when no questions file exists) into `directory`.

    `docs` is checked by loading it, so a wrong path or an unreadable corpus fails here, before anything is written. An existing config is only
    replaced with `force`; questions files are never overwritten.
    """
    config_path = directory / CONFIG_NAME
    if config_path.exists() and not force:
        raise ScaffoldError(f"{config_path} already exists. Use --force to replace it, or pick another directory.")
    try:
        preset_systems(preset)  # validates the name
        documents = load_documents(docs)
    except (FileNotFoundError, ValueError, DocumentLoadError, MissingExtraError) as exc:
        raise ScaffoldError(str(exc)) from exc
    if questions is not None and not questions.is_file():
        raise ScaffoldError(f"Questions file not found: {questions}")
    questions_path = questions if questions is not None else directory / QUESTIONS_NAME
    create_questions = not questions_path.exists()

    name = re.sub(r"[^A-Za-z0-9_-]+", "_", directory.resolve().name).strip("_") or "ragbench"
    stub = {"run": {"name": name, "output_dir": (directory / "results").as_posix()}}
    config = apply_preset(stub, preset, docs=docs.as_posix(), questions=questions_path.as_posix())
    config["evaluation"] = {"judge_enabled": True, "max_workers": 4, "max_cost_usd": DEFAULT_MAX_COST_USD}
    config["selection"] = {"profile": "balanced"}
    ExperimentConfig.model_validate(expand_sweeps(config))  # what is written must load

    directory.mkdir(parents=True, exist_ok=True)
    if create_questions:
        write_jsonl(questions_path, _template_questions(documents))
    config_path.write_text(_config_text(config, preset), encoding="utf-8")
    return ScaffoldResult(config_path, questions_path, create_questions, len(documents), preset)
