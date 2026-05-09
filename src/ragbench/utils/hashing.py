from __future__ import annotations

import hashlib


def stable_hash(text: str, length: int = 12) -> str:
    """Return a short deterministic hash for IDs and cache keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]

