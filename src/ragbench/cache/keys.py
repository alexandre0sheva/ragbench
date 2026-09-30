from __future__ import annotations

import hashlib
import json
from typing import Any

# Bump when the stored payload format changes, so old entries are simply never hit again.
SCHEMA_VERSION = 1


def cache_key(namespace: str, **parts: Any) -> str:
    """Stable sha256 over the canonical JSON of the namespace and everything that determines the result."""
    payload = json.dumps({"v": SCHEMA_VERSION, "ns": namespace, **parts}, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
