"""
TTL caching for FastAPI route handlers.

TTL cache behaviour:
- short_cache: 300s  — live market data (prices, breadth, signals)
- long_cache:  3600s — slow/external scrapes (country dashboard, central banks, industry)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

from cachetools import TTLCache

logger = logging.getLogger("uvicorn.error")

short_cache: TTLCache = TTLCache(maxsize=32, ttl=300)
long_cache: TTLCache = TTLCache(maxsize=32, ttl=3600)
_lock = threading.Lock()

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DISK_CACHE_ROOT = Path(os.getenv("API_DISK_CACHE_DIR") or (_REPO_ROOT / "data_cache" / "api_cache"))
_DISK_CACHE_ENABLED = os.getenv("API_DISK_CACHE_DISABLE", "").strip().lower() not in ("1", "true", "yes")
logger.info(
    "api cache init: disk_cache=%s root=%s short_ttl=%ss long_ttl=%ss",
    "enabled" if _DISK_CACHE_ENABLED else "disabled",
    str(_DISK_CACHE_ROOT),
    getattr(short_cache, "ttl", "n/a"),
    getattr(long_cache, "ttl", "n/a"),
)

if _DISK_CACHE_ENABLED:
    try:
        (_DISK_CACHE_ROOT / "short").mkdir(parents=True, exist_ok=True)
        (_DISK_CACHE_ROOT / "long").mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.warning(
            "api cache init: failed to create disk cache dirs at %s; disabling disk cache",
            str(_DISK_CACHE_ROOT),
            exc_info=True,
        )
        _DISK_CACHE_ENABLED = False


def _disk_cache_path(cache: TTLCache, key: str) -> Path:
    name = "short" if cache is short_cache else "long"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return _DISK_CACHE_ROOT / name / f"{h}.json"


def _disk_get(cache: TTLCache, key: str):
    if not _DISK_CACHE_ENABLED:
        return None
    path = _disk_cache_path(cache, key)
    try:
        if not path.exists():
            return None
        ttl = getattr(cache, "ttl", None)
        if isinstance(ttl, (int, float)) and ttl > 0:
            age = time.time() - path.stat().st_mtime
            if age > float(ttl):
                return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("value")
    except Exception:
        logger.debug("api cache disk read failed (key=%s path=%s)", key, str(path), exc_info=True)
        return None


def _disk_set(cache: TTLCache, key: str, value) -> None:
    if not _DISK_CACHE_ENABLED:
        return
    path = _disk_cache_path(cache, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"key": key, "value": value}), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        # Disk cache is best-effort; ignore failures.
        logger.debug("api cache disk write failed (key=%s path=%s)", key, str(path), exc_info=True)
        return


def get_cached(cache: TTLCache, key: str):
    with _lock:
        v = cache.get(key)
        if v is not None:
            return v
        # Disk cache is a read-through fallback only — do NOT reload into
        # the in-memory TTLCache.  Re-populating the in-memory cache after
        # TTL eviction defeats eviction and causes unbounded memory growth.
        return _disk_get(cache, key)


def set_cached(cache: TTLCache, key: str, value) -> None:
    with _lock:
        cache[key] = value
        _disk_set(cache, key, value)


def delete_cached(cache: TTLCache, key: str) -> None:
    """Delete a single cache entry from memory and disk (best-effort)."""
    with _lock:
        try:
            cache.pop(key, None)
        except Exception:
            pass
        if _DISK_CACHE_ENABLED:
            try:
                _disk_cache_path(cache, key).unlink(missing_ok=True)
            except Exception:
                pass
    logger.info("api cache delete (key=%s)", key)


def invalidate_all() -> None:
    """Clear both caches (used by /api/cache/clear endpoint)."""
    with _lock:
        short_cache.clear()
        long_cache.clear()
    logger.info("api cache invalidate_all: memory cleared")
    try:
        if _DISK_CACHE_ROOT.exists():
            for sub in ("short", "long"):
                d = _DISK_CACHE_ROOT / sub
                if d.exists():
                    for p in d.glob("*.json"):
                        try:
                            p.unlink()
                        except Exception:
                            pass
    except Exception:
        logger.debug("api cache invalidate_all: disk cleanup failed", exc_info=True)
