"""Tests for the server-side Arrow result cache.

The dataset is immutable per-load — the same wire-scatter SQL fired by
the viewer after every gesture release returns the same bytes. The
cache short-circuits these redundant DuckDB scans. Any DDL/DML must
invalidate so subsequent reads see the new state.
"""

from __future__ import annotations

import pytest

from embedding_atlas.server import _ArrowResultCache, _READONLY_RE


def test_cache_hit_returns_same_bytes():
    cache = _ArrowResultCache()
    sql = "SELECT 1"
    body = b"some-arrow-ipc-bytes"
    cache.put(sql, body)
    assert cache.get(sql) is body
    assert cache.hits == 1
    assert cache.misses == 0


def test_cache_miss_returns_none():
    cache = _ArrowResultCache()
    assert cache.get("SELECT 1") is None
    assert cache.misses == 1
    assert cache.hits == 0


def test_lru_evicts_when_over_budget():
    cache = _ArrowResultCache(max_bytes=10)
    cache.put("a", b"01234")  # 5 bytes
    cache.put("b", b"56789")  # 5 bytes — at budget
    cache.put("c", b"AB")  # 2 bytes — must evict 'a'
    assert cache.get("a") is None
    assert cache.get("b") == b"56789"
    assert cache.get("c") == b"AB"


def test_oversized_response_skipped():
    cache = _ArrowResultCache(max_bytes=10)
    cache.put("huge", b"x" * 100)
    # Should refuse to cache the oversized item rather than evict everything.
    assert cache.get("huge") is None


def test_lru_recency_on_get():
    cache = _ArrowResultCache(max_bytes=10)
    cache.put("a", b"01234")
    cache.put("b", b"56789")
    cache.get("a")  # touch a → most recent
    cache.put("c", b"ZZ")  # eviction must hit 'b' now, not 'a'
    assert cache.get("a") == b"01234"
    assert cache.get("b") is None


def test_invalidate_clears():
    cache = _ArrowResultCache()
    cache.put("a", b"x")
    cache.put("b", b"y")
    cache.invalidate()
    assert cache.get("a") is None
    assert cache.get("b") is None


@pytest.mark.parametrize(
    "sql,readonly",
    [
        ("SELECT * FROM t", True),
        ("  SELECT 1", True),
        ("\n\tSELECT count(*) FROM t", True),
        ("WITH x AS (SELECT 1) SELECT * FROM x", True),
        ("PRAGMA table_info(t)", True),
        ("DESCRIBE t", True),
        ("SUMMARIZE t", True),
        ("ALTER TABLE t ADD COLUMN c INT", False),
        ("UPDATE t SET c = 1", False),
        ("INSERT INTO t VALUES (1)", False),
        ("CREATE TABLE u AS SELECT 1", False),
        ("DROP TABLE t", False),
    ],
)
def test_readonly_pattern(sql, readonly):
    assert (_READONLY_RE.match(sql) is not None) == readonly


def test_thread_safe_concurrent_access():
    """Two threads slamming the cache shouldn't corrupt the byte total."""
    import threading

    cache = _ArrowResultCache(max_bytes=1000)
    payload = b"x" * 50

    def worker(prefix: str):
        for i in range(200):
            cache.put(f"{prefix}-{i}", payload)
            cache.get(f"{prefix}-{i // 2}")

    threads = [threading.Thread(target=worker, args=(p,)) for p in ("A", "B", "C", "D")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Bytes accounting must still be within budget.
    assert cache._size <= cache.max_bytes
