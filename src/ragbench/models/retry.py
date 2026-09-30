from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

from ragbench.models.errors import RateLimitError, TransientModelError

T = TypeVar("T")


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    sleep: Callable[[float], None] | None = None,
    jitter: Callable[[], float] = random.random,
) -> T:
    """Call `fn`, retrying `RateLimitError` / `TransientModelError` with exponential backoff.

    Delay before retry n (1-based) is `base_delay * 2**(n-1)` plus up to 25% jitter, capped at
    `max_delay`. A provider `Retry-After` hint takes precedence when it asks for longer. Other
    exceptions, including `PermanentModelError`, propagate immediately. `sleep` defaults to
    `time.sleep` resolved at call time so tests can patch it.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except (RateLimitError, TransientModelError) as exc:
            if attempt >= max_attempts:
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1) * (1 + 0.25 * jitter()))
            retry_after = getattr(exc, "retry_after", None)
            if retry_after is not None:
                delay = max(delay, retry_after)
            (sleep or time.sleep)(delay)
