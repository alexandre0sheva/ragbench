"""Measure what a run would send to paid models, using the offline mock models (`ragbench estimate`).

The mock LLM and the hashing embedder report every call to the active recorder, if there is one: how many calls, and the prompt,
completion and embedding tokens. Prices are applied afterwards (see `evaluation/estimate.py`), so the same measurement can be
priced with any model's rates. Like the cache and runtime contexts this is a plain module global rather than a ContextVar,
because ingestion and question workers run on threads that would not inherit it.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

_lock = threading.Lock()
_active: UsageRecorder | None = None


@dataclass
class UsageRecorder:
    llm_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    embed_calls: int = 0
    embed_tokens: int = 0


@contextmanager
def record_usage() -> Iterator[UsageRecorder]:
    """Collect the usage of every mock model call made inside the block. Blocks do not nest: the inner one gets its own recorder."""
    global _active
    recorder = UsageRecorder()
    with _lock:
        previous, _active = _active, recorder
    try:
        yield recorder
    finally:
        with _lock:
            _active = previous


def note_llm(prompt_tokens: int, completion_tokens: int) -> None:
    with _lock:
        if _active is not None:
            _active.llm_calls += 1
            _active.llm_prompt_tokens += prompt_tokens
            _active.llm_completion_tokens += completion_tokens


def note_embedding(tokens: int) -> None:
    with _lock:
        if _active is not None:
            _active.embed_calls += 1
            _active.embed_tokens += tokens


def recording() -> bool:
    return _active is not None
