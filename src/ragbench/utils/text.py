from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_DEFAULT_ENCODING = "o200k_base"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def truncate(text: str, max_chars: int = 240) -> str:
    clean = normalize_text(text)
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


@lru_cache(maxsize=64)
def _get_encoding(model: str | None):
    """Resolve (once per model) the tiktoken encoding used for estimates; None means use the heuristic."""
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding(_DEFAULT_ENCODING)
    except KeyError:
        pass  # Unknown model name: newer OpenAI models all use the o200k family.
    except Exception:
        return None
    try:
        return tiktoken.get_encoding(_DEFAULT_ENCODING)
    except Exception:
        return None


def estimate_tokens(text: str, model: str | None = None) -> int:
    encoding = _get_encoding(model)
    if encoding is not None:
        try:
            return len(encoding.encode(text, disallowed_special=()))
        except Exception:
            pass
    return max(1, int(len(text.split()) * 1.3))
