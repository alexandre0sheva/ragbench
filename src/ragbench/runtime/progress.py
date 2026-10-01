from __future__ import annotations

import threading


class ProgressListener:
    """No-op progress hooks; the CLI subclasses this to drive a live display."""

    def run_started(self, num_systems: int, num_questions: int) -> None: ...

    def system_started(self, name: str, index: int, total: int) -> None: ...

    def ingestion_finished(self, name: str, num_chunks: int, latency_ms: float) -> None: ...

    def question_finished(self, name: str, done: int, total: int) -> None: ...

    def system_finished(self, name: str, wall_time_ms: float) -> None: ...

    def latency_probe(self, name: str, done: int, total: int) -> None: ...

    def system_restored(self, name: str, num_questions: int) -> None: ...


class SerializedProgress(ProgressListener):
    """Forwards every hook to `inner` under one lock, so listeners never have to be thread-safe themselves.

    Systems and questions run on worker threads; without this a listener could be entered by several at once.
    """

    def __init__(self, inner: ProgressListener):
        self._inner = inner
        self._lock = threading.Lock()

    def run_started(self, num_systems: int, num_questions: int) -> None:
        with self._lock:
            self._inner.run_started(num_systems, num_questions)

    def system_started(self, name: str, index: int, total: int) -> None:
        with self._lock:
            self._inner.system_started(name, index, total)

    def ingestion_finished(self, name: str, num_chunks: int, latency_ms: float) -> None:
        with self._lock:
            self._inner.ingestion_finished(name, num_chunks, latency_ms)

    def question_finished(self, name: str, done: int, total: int) -> None:
        with self._lock:
            self._inner.question_finished(name, done, total)

    def system_finished(self, name: str, wall_time_ms: float) -> None:
        with self._lock:
            self._inner.system_finished(name, wall_time_ms)

    def latency_probe(self, name: str, done: int, total: int) -> None:
        with self._lock:
            self._inner.latency_probe(name, done, total)

    def system_restored(self, name: str, num_questions: int) -> None:
        with self._lock:
            self._inner.system_restored(name, num_questions)
