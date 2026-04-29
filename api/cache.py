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
from datetime import datetime
from pathlib import Path

from cachetools import TTLCache

logger = logging.getLogger("uvicorn.error")

short_cache: TTLCache = TTLCache(maxsize=128, ttl=300)
long_cache: TTLCache = TTLCache(maxsize=128, ttl=3600)
_lock = threading.Lock()

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DISK_CACHE_ENABLED = os.getenv("API_DISK_CACHE_DISABLE", "").strip().lower() not in ("1", "true", "yes")


def _candidate_disk_cache_roots() -> list[Path]:
    env_path = (os.getenv("API_DISK_CACHE_DIR") or "").strip()
    roots: list[Path] = []
    if env_path:
        roots.append(Path(env_path).expanduser())
    roots.append(_REPO_ROOT / "data_cache" / "api_cache")
    roots.append(Path(os.getenv("TMPDIR") or "/tmp") / "market-scripts" / "data_cache" / "api_cache")
    deduped: list[Path] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return deduped


def _initialize_disk_cache_root(candidates: list[Path]) -> tuple[Path, bool]:
    for root in candidates:
        try:
            (root / "short").mkdir(parents=True, exist_ok=True)
            (root / "long").mkdir(parents=True, exist_ok=True)
            return root, True
        except Exception:
            logger.warning(
                "api cache init: failed to create disk cache dirs at %s; trying next fallback",
                str(root),
                exc_info=True,
            )
    return candidates[0], False


_DISK_CACHE_ROOTS = _candidate_disk_cache_roots()
_DISK_CACHE_ROOT = _DISK_CACHE_ROOTS[0]
if _DISK_CACHE_ENABLED:
    _DISK_CACHE_ROOT, _DISK_CACHE_ENABLED = _initialize_disk_cache_root(_DISK_CACHE_ROOTS)

logger.info(
    "api cache init: disk_cache=%s root=%s short_ttl=%ss long_ttl=%ss",
    "enabled" if _DISK_CACHE_ENABLED else "disabled",
    str(_DISK_CACHE_ROOT),
    getattr(short_cache, "ttl", "n/a"),
    getattr(long_cache, "ttl", "n/a"),
)


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


def _stamp_age(v, cache: TTLCache) -> None:
    """Compute data_age_seconds and stale flag from _meta.fetched_at."""
    if not isinstance(v, dict):
        return
    meta = v.get("_meta")
    if not meta or "fetched_at" not in meta:
        return
    try:
        age = (datetime.now() - datetime.fromisoformat(meta["fetched_at"])).total_seconds()
        meta["data_age_seconds"] = round(age)
        ttl = meta.get("cache_ttl") or getattr(cache, "ttl", 600)
        meta["stale"] = age > 2 * ttl
    except Exception:
        pass


def get_cached(cache: TTLCache, key: str):
    with _lock:
        v = cache.get(key)
        if v is not None:
            _stamp_age(v, cache)
            return v
        # Disk cache is a read-through fallback only — do NOT reload into
        # the in-memory TTLCache.  Re-populating the in-memory cache after
        # TTL eviction defeats eviction and causes unbounded memory growth.
        v = _disk_get(cache, key)
        if v is not None:
            _stamp_age(v, cache)
        return v


def set_cached(cache: TTLCache, key: str, value) -> None:
    if isinstance(value, dict):
        value["_meta"] = {
            "fetched_at": datetime.now().isoformat(),
            "cache_ttl": getattr(cache, "ttl", None),
        }
    with _lock:
        cache[key] = value
        _disk_set(cache, key, value)


def stamp_fresh(result: dict) -> dict:
    """Add _meta to uncached endpoint responses (always fresh)."""
    result["_meta"] = {
        "fetched_at": datetime.now().isoformat(),
        "cache_ttl": None,
        "data_age_seconds": 0,
        "stale": False,
    }
    return result


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

    # Downstream module disk caches that survive between server restarts.
    try:
        from equities.market_technicals.market_breadth import invalidate_disk_cache as _invalidate_breadth

        _invalidate_breadth()
    except Exception:
        logger.debug("api cache invalidate_all: breadth disk cleanup failed", exc_info=True)
