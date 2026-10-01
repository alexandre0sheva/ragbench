"""Ready-made system lists for a first comparison on any dataset (`--preset quick|standard|thorough|agentic`).

A preset is plain data: it picks systems, never the dataset or the models (those come from the config you pass, or the defaults).
The lists follow the cost ladder in docs/systems.md, from three cheap baselines up to every retrieval system plus a chunk-size
sweep, and a separate agentic set (loops, tools, routing) with a reference to measure it against.
"""

from __future__ import annotations

import copy
import difflib
from pathlib import Path
from typing import Any

PRESET_NAMES = ("quick", "standard", "thorough", "agentic")

_QUICK: list[dict[str, Any]] = [
    {"type": "bm25", "name": "bm25"},
    {"type": "vector", "name": "vector"},
    {"type": "hybrid_rerank", "name": "hybrid_rerank"},
]
_STANDARD_EXTRA: list[dict[str, Any]] = [
    {"type": "hybrid", "name": "hybrid"},
    {"type": "rerank", "name": "rerank"},
    {"type": "parent_doc", "name": "parent_doc"},
    {"type": "hyde", "name": "hyde"},
    {"type": "contextual", "name": "contextual"},
]
_THOROUGH_EXTRA: list[dict[str, Any]] = [
    {"type": "hierarchical", "name": "hierarchical"},
    {"type": "sentence_window", "name": "sentence_window"},
    {"type": "rag_fusion", "name": "rag_fusion"},
    {"type": "decompose", "name": "decompose"},
    {"type": "llm_heavy", "name": "llm_heavy"},
    # The chunk size is the axis that moves retrieval quality most; 500 words is the default the plain `hybrid_rerank` above already covers.
    {"type": "hybrid_rerank", "name": "hybrid_rerank", "sweep": {"chunker.chunk_size": [250, 1000]}},
]
_AGENT_CHUNKER = {"type": "word", "chunk_size": 250, "chunk_overlap": 40}
_AGENT_RETRIEVAL = {"retriever": "hybrid", "top_k": 5}
_AGENTIC: list[dict[str, Any]] = [
    {"type": "hybrid_rerank", "name": "hybrid_rerank"},  # the non-agentic reference every agent has to beat to be worth its cost
    {"type": "corrective", "name": "corrective", "chunker": _AGENT_CHUNKER, "retrieval": {**_AGENT_RETRIEVAL, "max_rounds": 2, "max_cost_usd": 0.02}},
    {"type": "iterative", "name": "iterative", "chunker": _AGENT_CHUNKER, "retrieval": {**_AGENT_RETRIEVAL, "max_hops": 3, "max_cost_usd": 0.02}},
    {"type": "agent_search", "name": "agent_search_plain", "chunker": _AGENT_CHUNKER, "retrieval": {**_AGENT_RETRIEVAL, "max_steps": 5, "max_cost_usd": 0.05}, "tools": []},
    {
        "type": "agent_search",
        "name": "agent_search_tools",
        "chunker": _AGENT_CHUNKER,
        "retrieval": {**_AGENT_RETRIEVAL, "max_steps": 5, "max_cost_usd": 0.05},
        "tools": ["calculator", "date_calc", "corpus_grep"],
    },
    {"type": "grep_agent", "name": "grep_agent", "retrieval": {"max_steps": 6, "max_cost_usd": 0.05, "top_k": 5}},
    {
        "type": "adaptive",
        "name": "adaptive",
        "retrieval": {
            "router": "heuristic",
            "routes": {
                "default": {"type": "hybrid_rerank", "chunker": _AGENT_CHUNKER, "retrieval": {"final_top_k": 5}},
                "lexical": {"type": "bm25", "chunker": _AGENT_CHUNKER, "retrieval": {"top_k": 5}},
                "multi_hop": {"type": "decompose", "chunker": _AGENT_CHUNKER, "retrieval": dict(_AGENT_RETRIEVAL)},
                "computation": {
                    "type": "agent_search",
                    "chunker": _AGENT_CHUNKER,
                    "retrieval": {**_AGENT_RETRIEVAL, "max_steps": 5, "max_cost_usd": 0.05},
                    "tools": ["calculator", "date_calc"],
                },
            },
        },
    },
]


def preset_systems(name: str) -> list[dict[str, Any]]:
    """The system blocks of a preset (fresh copies; `thorough` contains a `sweep:`)."""
    lists = {
        "quick": _QUICK,
        "standard": [*_QUICK, *_STANDARD_EXTRA],
        "thorough": [*_QUICK, *_STANDARD_EXTRA, *_THOROUGH_EXTRA],
        "agentic": _AGENTIC,
    }
    if name not in lists:
        close = difflib.get_close_matches(name, PRESET_NAMES, n=1)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise ValueError(f"Unknown preset '{name}'.{hint} Available: {', '.join(PRESET_NAMES)}")
    return copy.deepcopy(lists[name])


def apply_preset(
    raw: dict[str, Any] | None,
    preset: str,
    *,
    docs: Path | str | None = None,
    questions: Path | str | None = None,
    qrels: Path | str | None = None,
) -> dict[str, Any]:
    """A config running the preset's systems. `raw` (an existing config) keeps everything but its `systems`; the dataset comes from
    the flags where given, else from `raw`. Raises `ValueError` when there is no dataset or the preset is unknown."""
    systems = preset_systems(preset)
    config: dict[str, Any] = copy.deepcopy(raw) if raw else {"run": {"name": f"preset_{preset}", "output_dir": "results"}}
    config.setdefault("run", {"name": f"preset_{preset}", "output_dir": "results"})
    if docs is not None or questions is not None or qrels is not None:
        given: dict[str, Any] = dict(config.get("dataset") or {})
        for key, value in (("documents_path", docs), ("questions_path", questions), ("qrels_path", qrels)):
            if value is not None:
                given[key] = str(value)
        config["dataset"] = given
    dataset = config.get("dataset") or {}
    if not dataset.get("documents_path") or not dataset.get("questions_path"):
        raise ValueError("--preset needs a dataset: pass --docs and --questions (and optionally --qrels), or --config with a `dataset:` section")
    config["systems"] = systems
    return config
