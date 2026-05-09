from __future__ import annotations

import re
from collections.abc import Iterable

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


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


def estimate_tokens(text: str, model: str | None = None) -> int:
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model or "gpt-5.4-nano")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text.split()) * 1.3))
