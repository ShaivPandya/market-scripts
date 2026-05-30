"""
TTL caching for FastAPI route handlers.

TTL cache behaviour:
- short_cache: 300s  — live market data (prices, breadth, signals)
- long_cache:  3600s — slow/external scrapes (central banks, industry)
- daily_cache: 86400s — low-frequency macro/reference data (country dashboard)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from cachetools import TTLCache  # type: ignore[import-untyped]

logger = logging.getLogger("uvicorn.error")

short_cache: TTLCache = TTLCache(maxsize=128, ttl=300)
long_cache: TTLCache = TTLCache(maxsize=128, ttl=3600)
daily_cache: TTLCache = TTLCache(maxsize=128, ttl=24 * 60 * 60)
_lock = threading.Lock()
_singleflight_lock = threading.Lock()
_MISSING = object()


@dataclass(slots=True)
class _SingleFlightState:
    event: threading.Event
    value: object = _MISSING
    error: Exception | None = None


_singleflight_by_key: dict[str, _SingleFlightState] = {}

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DISK_CACHE_ENABLED = os.getenv("API_DISK_CACHE_DISABLE", "").strip().lower() not in ("1", "true", "yes")
_GCS_CACHE_PREFIX = (os.getenv("API_GCS_CACHE_PREFIX") or "live/cache/api_cache").strip("/")


def _candidate_disk_cache_roots() -> list[Path]:
    env_path = (os.getenv("API_DISK_CACHE_DIR") or "").strip()
    roots: list[Path] = []
    if env_path:
        roots.append(Path(env_path).expanduser())
    roots.append(_REPO_ROOT / "data_cache" / "api_cache")
    roots.append(Path(os.getenv("TMPDIR") or "/tmp") / "talisman" / "data_cache" / "api_cache")
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
            (root / "daily").mkdir(parents=True, exist_ok=True)
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
    "api cache init: disk_cache=%s root=%s short_ttl=%ss long_ttl=%ss daily_ttl=%ss",
    "enabled" if _DISK_CACHE_ENABLED else "disabled",
    str(_DISK_CACHE_ROOT),
    getattr(short_cache, "ttl", "n/a"),
    getattr(long_cache, "ttl", "n/a"),
    getattr(daily_cache, "ttl", "n/a"),
)


def _disk_cache_name(cache: TTLCache) -> str:
    if cache is short_cache:
        return "short"
    if cache is daily_cache:
        return "daily"
    return "long"


def _disk_cache_path(cache: TTLCache, key: str) -> Path:
    name = _disk_cache_name(cache)
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return _DISK_CACHE_ROOT / name / f"{h}.json"


def _gcs_cache_enabled() -> bool:
    if _DISK_CACHE_ENABLED:
        return False
    try:
        from api.state_storage import use_gcs_state

        return bool(use_gcs_state())
    except Exception:
        return False


def _gcs_cache_key(cache: TTLCache, key: str) -> str:
    name = _disk_cache_name(cache)
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"{_GCS_CACHE_PREFIX}/{name}/{h}.json"


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except Exception:
        return None
    return parsed


def _timestamp_age_seconds(value: datetime) -> float:
    now = datetime.now(value.tzinfo) if value.tzinfo is not None else datetime.now()
    return (now - value).total_seconds()


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


def _gcs_get(cache: TTLCache, key: str):
    if not _gcs_cache_enabled():
        return None
    gcs_key = _gcs_cache_key(cache, key)
    try:
        from api.state_storage import object_updated, read_text

        payload = json.loads(read_text(_DISK_CACHE_ROOT / ".state-cache-placeholder", gcs_key, encoding="utf-8"))
        ttl = getattr(cache, "ttl", None)
        if isinstance(ttl, (int, float)) and ttl > 0:
            created_at = _parse_timestamp(payload.get("created_at"))
            if created_at is None:
                created_at = object_updated(_DISK_CACHE_ROOT / ".state-cache-placeholder", gcs_key)
            if created_at is not None and _timestamp_age_seconds(created_at) > float(ttl):
                return None
        return payload.get("value")
    except Exception:
        logger.debug("api cache state read failed (key=%s gcs_key=%s)", key, gcs_key, exc_info=True)
        return None


def _gcs_set(cache: TTLCache, key: str, value) -> None:
    if not _gcs_cache_enabled():
        return
    gcs_key = _gcs_cache_key(cache, key)
    try:
        from api.state_storage import write_text

        write_text(
            _DISK_CACHE_ROOT / ".state-cache-placeholder",
            gcs_key,
            json.dumps({"key": key, "created_at": datetime.now().isoformat(), "value": value}, allow_nan=False),
            encoding="utf-8",
            content_type="application/json; charset=utf-8",
        )
    except Exception:
        logger.debug("api cache state write failed (key=%s gcs_key=%s)", key, gcs_key, exc_info=True)


def _gcs_delete(cache: TTLCache, key: str) -> None:
    if not _gcs_cache_enabled():
        return
    gcs_key = _gcs_cache_key(cache, key)
    try:
        from api.state_storage import delete_file

        delete_file(_DISK_CACHE_ROOT / ".state-cache-placeholder", gcs_key)
    except Exception:
        logger.debug("api cache state delete failed (key=%s gcs_key=%s)", key, gcs_key, exc_info=True)


def _gcs_invalidate_all() -> None:
    if not _gcs_cache_enabled():
        return
    try:
        from api.state_storage import delete_file, list_keys

        for key in list_keys(f"{_GCS_CACHE_PREFIX}/"):
            delete_file(_DISK_CACHE_ROOT / ".state-cache-placeholder", key)
    except Exception:
        logger.debug("api cache state invalidate_all failed", exc_info=True)


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
        if v is None:
            v = _gcs_get(cache, key)
        if v is not None:
            _stamp_age(v, cache)
        return v


def set_cached(cache: TTLCache, key: str, value) -> None:
    if isinstance(value, dict):
        raw_meta = value.get("_meta")
        existing_meta: dict = raw_meta if isinstance(raw_meta, dict) else {}
        value["_meta"] = {
            **existing_meta,
            "fetched_at": datetime.now().isoformat(),
            "cache_ttl": getattr(cache, "ttl", None),
        }
    with _lock:
        cache[key] = value
        _disk_set(cache, key, value)
        _gcs_set(cache, key, value)


def _singleflight_key(cache: TTLCache, key: str) -> str:
    return f"{id(cache)}::{key}"


def _get_or_set_cached_with_status(
    cache: TTLCache,
    key: str,
    loader: Callable[[], Any],
    *,
    force_refresh: bool = False,
    wait_timeout_s: float = 120,
) -> tuple[Any, str]:
    if not force_refresh:
        cached = get_cached(cache, key)
        if cached is not None:
            return cached, "hit"

    flight_key = _singleflight_key(cache, key)
    with _singleflight_lock:
        state = _singleflight_by_key.get(flight_key)
        if state is None:
            state = _SingleFlightState(event=threading.Event())
            _singleflight_by_key[flight_key] = state
            owner = True
        else:
            owner = False

    if owner:
        status = "refresh" if force_refresh else "miss_fetch"
        try:
            value = loader()
            set_cached(cache, key, value)
            state.value = value
            return value, status
        except Exception as exc:
            state.error = exc
            raise
        finally:
            state.event.set()
            with _singleflight_lock:
                _singleflight_by_key.pop(flight_key, None)

    waited = state.event.wait(timeout=wait_timeout_s)
    if waited:
        if state.error is not None:
            raise state.error
        if state.value is not _MISSING:
            if force_refresh:
                return state.value, "refresh"
            return state.value, "miss_wait"

    cached_after = get_cached(cache, key)
    if cached_after is not None:
        return cached_after, "refresh" if force_refresh else "miss_wait"

    logger.warning(
        "api cache singleflight wait timeout; running fallback loader key=%s force_refresh=%s",
        key,
        force_refresh,
    )
    value = loader()
    set_cached(cache, key, value)
    return value, "refresh" if force_refresh else "miss_refetch"


def get_or_set_cached(
    cache: TTLCache,
    key: str,
    loader: Callable[[], Any],
    *,
    force_refresh: bool = False,
    wait_timeout_s: float = 120,
):
    value, _status = _get_or_set_cached_with_status(
        cache,
        key,
        loader,
        force_refresh=force_refresh,
        wait_timeout_s=wait_timeout_s,
    )
    return value


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
        _gcs_delete(cache, key)
    logger.info("api cache delete (key=%s)", key)


def invalidate_all() -> None:
    """Clear both caches (used by /api/cache/clear endpoint)."""
    with _lock:
        short_cache.clear()
        long_cache.clear()
        daily_cache.clear()
    logger.info("api cache invalidate_all: memory cleared")
    try:
        if _DISK_CACHE_ROOT.exists():
            for sub in ("short", "long", "daily"):
                d = _DISK_CACHE_ROOT / sub
                if d.exists():
                    for p in d.glob("*.json"):
                        try:
                            p.unlink()
                        except Exception:
                            pass
    except Exception:
        logger.debug("api cache invalidate_all: disk cleanup failed", exc_info=True)

    _gcs_invalidate_all()

    # Downstream module disk caches that survive between server restarts.
    try:
        from equities.market_technicals.market_breadth import invalidate_disk_cache as _invalidate_breadth

        _invalidate_breadth()
    except Exception:
        logger.debug("api cache invalidate_all: breadth disk cleanup failed", exc_info=True)

    try:
        from api.signal_aggregator import invalidate_sp500_price_cache

        invalidate_sp500_price_cache()
    except Exception:
        logger.debug("api cache invalidate_all: signal aggregator price cache cleanup failed", exc_info=True)

    try:
        from api.job_queue import clear_memory_jobs

        clear_memory_jobs()
    except Exception:
        logger.debug("api cache invalidate_all: async job cleanup failed", exc_info=True)

    try:
        from api.job_events import clear_memory_events

        clear_memory_events()
    except Exception:
        logger.debug("api cache invalidate_all: async job event cleanup failed", exc_info=True)
