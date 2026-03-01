"""
TTL caching for FastAPI route handlers.

Mirrors Streamlit's @st.cache_data(ttl=...) behaviour:
- short_cache: 300s  — live market data (prices, breadth, signals)
- long_cache:  3600s — slow/external scrapes (country dashboard, central banks, industry)
"""

import threading
from cachetools import TTLCache

short_cache: TTLCache = TTLCache(maxsize=256, ttl=300)
long_cache: TTLCache = TTLCache(maxsize=128, ttl=3600)
_lock = threading.Lock()


def get_cached(cache: TTLCache, key: str):
    with _lock:
        return cache.get(key)


def set_cached(cache: TTLCache, key: str, value) -> None:
    with _lock:
        cache[key] = value


def invalidate_all() -> None:
    """Clear both caches (used by /api/cache/clear endpoint)."""
    with _lock:
        short_cache.clear()
        long_cache.clear()
