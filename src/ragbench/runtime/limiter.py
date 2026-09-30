from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager


class RateLimiter:
    """Shared throttle for one provider: concurrent requests, requests per minute, and tokens per minute.

    `acquire()` first waits until the sliding one-minute windows have room (sleeping *outside* any lock, so other
    threads keep making progress), then takes a concurrency slot for the duration of the request. A request larger
    than the whole token budget is let through when nothing else is in the window instead of waiting forever. Any
    limit left as `None` is not enforced. `clock`/`sleep` are injectable so tests need no real time.
    """

    def __init__(
        self,
        *,
        max_concurrent: int | None = None,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._slots = threading.BoundedSemaphore(max_concurrent) if max_concurrent else None
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._request_times: deque[float] = deque()
        self._token_events: deque[tuple[float, int]] = deque()

    @property
    def limits_tokens(self) -> bool:
        """Whether callers need to estimate a request's tokens (an avoidable cost when it is not limited)."""
        return self.tokens_per_minute is not None

    def _wait_needed(self, now: float, est_tokens: int) -> float:
        """Seconds until the request fits in both windows (0 when it can go now). Called with the lock held."""
        horizon = now - 60.0
        while self._request_times and self._request_times[0] <= horizon:
            self._request_times.popleft()
        while self._token_events and self._token_events[0][0] <= horizon:
            self._token_events.popleft()
        wait = 0.0
        if self.requests_per_minute and len(self._request_times) >= self.requests_per_minute:
            wait = max(wait, self._request_times[0] + 60.0 - now)
        if self.tokens_per_minute and self._token_events:
            used = sum(tokens for _, tokens in self._token_events)
            if used + est_tokens > self.tokens_per_minute:
                # Wait for enough of the oldest requests to age out to make room.
                freed, need = 0, used + est_tokens - self.tokens_per_minute
                for stamp, tokens in self._token_events:
                    freed += tokens
                    wait = max(wait, stamp + 60.0 - now)
                    if freed >= need:
                        break
        return wait

    def _reserve(self, est_tokens: int) -> None:
        while True:
            with self._lock:
                now = self._clock()
                wait = self._wait_needed(now, est_tokens)
                if wait <= 0:
                    self._request_times.append(now)
                    if self.tokens_per_minute:
                        self._token_events.append((now, est_tokens))
                    return
            self._sleep(wait)

    @contextmanager
    def acquire(self, est_tokens: int = 0) -> Iterator[None]:
        if self.requests_per_minute or self.tokens_per_minute:
            self._reserve(est_tokens)
        if self._slots is not None:
            self._slots.acquire()
        try:
            yield
        finally:
            if self._slots is not None:
                self._slots.release()
