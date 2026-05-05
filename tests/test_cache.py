"""Tests for api/cache.py — TTL behavior, get/set, invalidation."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from cachetools import TTLCache


def _memory_only_cache(monkeypatch) -> TTLCache:
    import api.cache as cache_mod

    monkeypatch.setattr(cache_mod, "_DISK_CACHE_ENABLED", False)
    return TTLCache(maxsize=16, ttl=60)


def test_set_and_get_cached():
    from api.cache import get_cached, set_cached, short_cache

    key = f"test_key_{time.time()}"
    set_cached(short_cache, key, {"data": 42})
    result = get_cached(short_cache, key)
    assert result["data"] == 42
    assert "_meta" in result
    assert "fetched_at" in result["_meta"]
    assert result["_meta"]["cache_ttl"] == 300
    assert result["_meta"]["stale"] is False


def test_get_missing_key_returns_none():
    from api.cache import get_cached, short_cache

    result = get_cached(short_cache, "nonexistent_key_12345")
    assert result is None


def test_invalidate_all_clears_caches():
    from api.cache import get_cached, invalidate_all, long_cache, set_cached, short_cache

    set_cached(short_cache, "test_inv_short", "a")
    set_cached(long_cache, "test_inv_long", "b")

    invalidate_all()

    assert get_cached(short_cache, "test_inv_short") is None
    assert get_cached(long_cache, "test_inv_long") is None


def test_delete_cached():
    from api.cache import delete_cached, get_cached, set_cached, short_cache

    key = f"test_delete_{time.time()}"
    set_cached(short_cache, key, "value")
    assert get_cached(short_cache, key) == "value"

    delete_cached(short_cache, key)
    # After delete, in-memory should be gone
    # (disk may still have it, but in-memory is authoritative for this test)
    from api.cache import _lock

    with _lock:
        assert short_cache.get(key) is None


def test_cache_stores_complex_data():
    from api.cache import get_cached, set_cached, short_cache

    key = f"test_complex_{time.time()}"
    data = {
        "nested": {"list": [1, 2, 3]},
        "string": "hello",
        "number": 42.5,
    }
    set_cached(short_cache, key, data)
    result = get_cached(short_cache, key)
    assert result["nested"] == {"list": [1, 2, 3]}
    assert result["string"] == "hello"
    assert result["number"] == 42.5
    assert "_meta" in result


def test_get_or_set_cached_hit_does_not_call_loader(monkeypatch):
    from api.cache import get_or_set_cached, set_cached

    cache = _memory_only_cache(monkeypatch)
    set_cached(cache, "hit", {"value": 1})

    def loader():
        raise AssertionError("loader should not run on cache hit")

    assert get_or_set_cached(cache, "hit", loader)["value"] == 1


def test_get_or_set_cached_concurrent_misses_call_loader_once(monkeypatch):
    from api.cache import get_or_set_cached

    cache = _memory_only_cache(monkeypatch)
    calls = 0
    lock = threading.Lock()

    def loader():
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.05)
        return {"value": 7}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _idx: get_or_set_cached(cache, "miss", loader), range(4)))

    assert calls == 1
    assert [r["value"] for r in results] == [7, 7, 7, 7]


def test_get_or_set_cached_concurrent_force_refresh_calls_loader_once(monkeypatch):
    from api.cache import get_or_set_cached, set_cached

    cache = _memory_only_cache(monkeypatch)
    set_cached(cache, "refresh", {"value": "old"})
    calls = 0
    lock = threading.Lock()

    def loader():
        nonlocal calls
        with lock:
            calls += 1
            value = calls
        time.sleep(0.05)
        return {"value": value}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _idx: get_or_set_cached(cache, "refresh", loader, force_refresh=True), range(4)))

    assert calls == 1
    assert [r["value"] for r in results] == [1, 1, 1, 1]


def test_get_or_set_cached_loader_exception_propagates_and_retries(monkeypatch):
    from api.cache import get_or_set_cached

    cache = _memory_only_cache(monkeypatch)
    calls = 0
    started = threading.Event()

    def failing_loader():
        nonlocal calls
        calls += 1
        started.set()
        time.sleep(0.05)
        raise RuntimeError("boom")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(get_or_set_cached, cache, "error", failing_loader)
        started.wait(timeout=1)
        second = pool.submit(get_or_set_cached, cache, "error", failing_loader)
        with pytest.raises(RuntimeError, match="boom"):
            first.result()
        with pytest.raises(RuntimeError, match="boom"):
            second.result()

    assert calls == 1
    assert get_or_set_cached(cache, "error", lambda: {"value": "retry"})["value"] == "retry"


def test_get_or_set_cached_timeout_rechecks_cache_before_second_loader(monkeypatch):
    from api.cache import get_or_set_cached, set_cached

    cache = _memory_only_cache(monkeypatch)
    owner_started = threading.Event()
    owner_release = threading.Event()
    fallback_calls = 0

    def owner_loader():
        owner_started.set()
        owner_release.wait(timeout=1)
        return {"value": "owner"}

    def waiter_loader():
        nonlocal fallback_calls
        fallback_calls += 1
        return {"value": "fallback"}

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(get_or_set_cached, cache, "timeout", owner_loader)
        owner_started.wait(timeout=1)
        waiter = pool.submit(
            get_or_set_cached,
            cache,
            "timeout",
            waiter_loader,
            wait_timeout_s=0.05,
        )
        time.sleep(0.01)
        set_cached(cache, "timeout", {"value": "published"})

        assert waiter.result(timeout=1)["value"] == "published"
        assert fallback_calls == 0

        owner_release.set()
        assert owner.result(timeout=1)["value"] == "owner"
