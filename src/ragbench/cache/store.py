from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BATCH = 500  # stays under SQLite's bound-variable limit on every build


class DiskCache:
    """A small persistent key/value cache on SQLite (WAL), safe to share between threads.

    Keys live in a namespace (`llm`, `embeddings`). One connection guarded by a lock serves every thread; each
    operation is a single short statement, so contention is negligible next to an API call. A database that
    cannot be opened or read is treated as a cache miss: one warning is logged and the cache switches itself off
    for the rest of the run instead of failing the benchmark.

    Hit/miss/saved-cost counters are per instance (per run); entries persist on disk.
    """

    def __init__(self, path: Path, *, enabled: bool = True, ttl_days: int | None = None):
        self.path = Path(path)
        self.enabled = enabled
        self.ttl_days = ttl_days
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._broken = False
        self._hits: Counter[str] = Counter()
        self._misses: Counter[str] = Counter()
        self._saved: defaultdict[str, float] = defaultdict(float)

    # -- connection -----------------------------------------------------------------------------

    def _connection(self) -> sqlite3.Connection | None:
        if not self.enabled or self._broken:
            return None
        if self._conn is None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False, isolation_level=None)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS entries ("
                    "namespace TEXT NOT NULL, key TEXT NOT NULL, value BLOB NOT NULL, meta TEXT, created REAL NOT NULL, "
                    "PRIMARY KEY (namespace, key))"
                )
                self._conn = conn
            except (sqlite3.Error, OSError) as exc:
                self._fail(exc)
                return None
        return self._conn

    def _fail(self, exc: Exception) -> None:
        if not self._broken:
            logger.warning("Disk cache at %s is unusable (%s); continuing without it.", self.path, exc)
        self._broken = True
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    def _cutoff(self) -> float | None:
        return None if self.ttl_days is None else time.time() - self.ttl_days * 86_400

    # -- reads ----------------------------------------------------------------------------------

    def get(self, namespace: str, key: str) -> bytes | None:
        found = self.get_many(namespace, [key])
        return found.get(key)

    def get_many(self, namespace: str, keys: Iterable[str]) -> dict[str, bytes]:
        keys = list(dict.fromkeys(keys))
        found: dict[str, bytes] = {}
        with self._lock:
            conn = self._connection()
            if conn is not None:
                cutoff = self._cutoff()
                try:
                    for start in range(0, len(keys), _BATCH):
                        batch = keys[start : start + _BATCH]
                        marks = ",".join("?" * len(batch))
                        sql = f"SELECT key, value FROM entries WHERE namespace = ? AND key IN ({marks})"
                        args: list[Any] = [namespace, *batch]
                        if cutoff is not None:
                            sql += " AND created >= ?"
                            args.append(cutoff)
                        found.update({key: bytes(value) for key, value in conn.execute(sql, args)})
                except sqlite3.Error as exc:
                    self._fail(exc)
                    found = {}
            self._hits[namespace] += len(found)
            self._misses[namespace] += len(keys) - len(found)
        return found

    # -- writes ---------------------------------------------------------------------------------

    def put(self, namespace: str, key: str, value: bytes, *, meta: dict[str, Any] | None = None) -> None:
        self.put_many(namespace, [(key, value)], meta=meta)

    def put_many(self, namespace: str, items: Iterable[tuple[str, bytes]], *, meta: dict[str, Any] | None = None) -> None:
        rows = [(namespace, key, sqlite3.Binary(value), json.dumps(meta) if meta else None, time.time()) for key, value in items]
        if not rows:
            return
        with self._lock:
            conn = self._connection()
            if conn is None:
                return
            try:
                conn.execute("BEGIN")
                conn.executemany("INSERT OR REPLACE INTO entries (namespace, key, value, meta, created) VALUES (?, ?, ?, ?, ?)", rows)
                conn.execute("COMMIT")
            except sqlite3.Error as exc:
                self._fail(exc)

    def clear(self, namespace: str | None = None) -> int:
        """Delete entries (one namespace or all); returns how many were removed."""
        with self._lock:
            conn = self._connection() if self.path.exists() else None
            if conn is None:
                return 0
            try:
                cursor = conn.execute("DELETE FROM entries" + (" WHERE namespace = ?" if namespace else ""), [namespace] if namespace else [])
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                return cursor.rowcount
            except sqlite3.Error as exc:
                self._fail(exc)
                return 0

    # -- accounting -----------------------------------------------------------------------------

    def record_saved(self, namespace: str, usd: float) -> None:
        """Note API spend avoided by a hit (computed by the caller at the current price)."""
        with self._lock:
            self._saved[namespace] += usd

    def stats(self) -> dict[str, Any]:
        with self._lock:
            namespaces = sorted({*self._hits, *self._misses, *self._saved})
            by_namespace = {
                ns: {"hits": self._hits[ns], "misses": self._misses[ns], "saved_cost_usd": self._saved[ns]} for ns in namespaces
            }
            size = sum(p.stat().st_size for p in (self.path, Path(f"{self.path}-wal")) if p.exists())
            return {
                "hits": sum(self._hits.values()),
                "misses": sum(self._misses.values()),
                "saved_cost_usd": sum(self._saved.values()),
                "size_bytes": size,
                "by_namespace": by_namespace,
            }

    def describe(self) -> dict[str, dict[str, int]]:
        """Entries and stored bytes per namespace (empty when the file does not exist; never creates it)."""
        if not self.path.exists():
            return {}
        with self._lock:
            conn = self._connection()
            if conn is None:
                return {}
            try:
                rows = conn.execute("SELECT namespace, COUNT(*), COALESCE(SUM(LENGTH(value)), 0) FROM entries GROUP BY namespace ORDER BY namespace")
                return {ns: {"entries": count, "bytes": size} for ns, count, size in rows}
            except sqlite3.Error as exc:
                self._fail(exc)
                return {}
