"""Tests for api/cache.py — TTL behavior, get/set, invalidation."""

import json
import time

import pytest


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
