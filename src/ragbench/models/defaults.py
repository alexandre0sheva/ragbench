"""Default model refs, in one place. Re-check them (and `models/cost.py`) whenever the price table is refreshed."""

from __future__ import annotations

# Newest cheap/fast OpenAI generation (was gpt-5.4-nano). Used for generation and as the default judge.
DEFAULT_GENERATOR_MODEL = "gpt-6-luna"
DEFAULT_JUDGE_MODEL = DEFAULT_GENERATOR_MODEL
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
