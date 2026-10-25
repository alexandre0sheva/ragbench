"""Reads a hub dataset's downloaded files and converts them."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ragbench.datasets.hub.convert import Converted, convert_galileo, convert_hotpotqa, convert_multihop_rag
from ragbench.datasets.hub.registry import HubDataset
from ragbench.utils.extras import require_extra

GALILEO_COLUMNS = ["id", "question", "documents", "response", "adherence_score", "all_relevant_sentence_keys", "dataset_name"]
HOTPOTQA_COLUMNS = ["id", "question", "answer", "type", "level", "supporting_facts", "context"]


def _parquet(path: Path, columns: list[str]) -> list[dict[str, Any]]:
    require_extra("pyarrow", "Reading this dataset (a Parquet file)", "hub", package="pyarrow")
    import pyarrow.parquet as parquet

    return list(parquet.read_table(path, columns=columns).to_pylist())


def _multihop_rag(entry: HubDataset, paths: dict[str, Path]) -> Converted:
    corpus = json.loads(paths["corpus.json"].read_text(encoding="utf-8"))
    queries = json.loads(paths["MultiHopRAG.json"].read_text(encoding="utf-8"))
    return convert_multihop_rag(corpus, queries)


def _hotpotqa(entry: HubDataset, paths: dict[str, Path]) -> Converted:
    return convert_hotpotqa(_parquet(paths["validation.parquet"], HOTPOTQA_COLUMNS))


def _galileo(entry: HubDataset, paths: dict[str, Path]) -> Converted:
    return convert_galileo(_parquet(paths["test.parquet"], GALILEO_COLUMNS), entry.options["subset"])


READERS: dict[str, Callable[[HubDataset, dict[str, Path]], Converted]] = {"multihop_rag": _multihop_rag, "hotpotqa": _hotpotqa, "galileo": _galileo}
