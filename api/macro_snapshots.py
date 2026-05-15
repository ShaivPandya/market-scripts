"""Snapshot helpers for slow-moving macro dashboards."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

from cachetools import TTLCache

from api.cache import get_or_set_cached
from api.exceptions import DataFetchError
from api.snapshot_keys import (
    SNAPSHOT_ECONOMIC_GROWTH,
    SNAPSHOT_HOUSING,
    SNAPSHOT_LABOR_MARKET,
    SNAPSHOT_SCHEMA_VERSION,
)
from api.snapshot_store import (
    attach_snapshot_meta,
    get_snapshot_response,
    write_snapshot_failure,
    write_snapshot_success,
)
from ontology.sources.source_registry import attach_source_registry_metadata, source_registry_metadata_for_snapshot

logger = logging.getLogger("api.macro_snapshots")


def payload_as_of(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("as_of", "as_of_date", "latest_date", "date", "timestamp"):
        value = payload.get(key)
        if value is not None:
            return str(value)[:32]
    latest = payload.get("latest")
    if isinstance(latest, dict):
        dates = [str(row.get("date")) for row in latest.values() if isinstance(row, dict) and row.get("date")]
        if dates:
            return max(dates)[:32]
    return None


def _snapshot_is_stale(snapshot: dict[str, Any]) -> bool:
    meta = snapshot.get("_meta")
    snapshot_meta = meta.get("snapshot") if isinstance(meta, dict) else None
    if not isinstance(snapshot_meta, dict):
        return True
    if snapshot_meta.get("refresh_status") not in (None, "ok"):
        return True
    return bool(snapshot_meta.get("stale"))


def _load_and_write_snapshot(
    *,
    snapshot_key: str,
    load_payload: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    payload = load_payload()
    payload_for_write = attach_source_registry_metadata(payload, snapshot_key=snapshot_key)
    record = write_snapshot_success(
        snapshot_key,
        payload_for_write,
        as_of_date=payload_as_of(payload_for_write),
        version=SNAPSHOT_SCHEMA_VERSION,
    )
    return attach_snapshot_meta(payload_for_write, record)


def get_snapshot_backed_response(
    *,
    snapshot_key: str,
    cache: TTLCache,
    cache_key: str,
    source: str,
    load_payload: Callable[[], dict[str, Any]],
    force_refresh: bool = False,
) -> dict[str, Any]:
    snapshot = get_snapshot_response(snapshot_key)
    if snapshot is not None and not force_refresh and not _snapshot_is_stale(snapshot):
        return snapshot

    try:
        if force_refresh:
            return _load_and_write_snapshot(snapshot_key=snapshot_key, load_payload=load_payload)
        return cast(
            dict[str, Any],
            get_or_set_cached(
                cache,
                cache_key,
                lambda: _load_and_write_snapshot(snapshot_key=snapshot_key, load_payload=load_payload),
            ),
        )
    except DataFetchError:
        if snapshot is not None:
            return snapshot
        raise
    except Exception as exc:
        if snapshot is not None:
            return snapshot
        raise DataFetchError(source=source, detail=str(exc)) from exc


def refresh_macro_snapshots(_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute and persist daily macro snapshots independently."""
    from api.routers.economic_growth import load_economic_growth_payload
    from api.routers.housing import load_housing_payload
    from api.routers.labor_market import load_labor_market_payload

    specs: tuple[tuple[str, str, Callable[[], dict[str, Any]]], ...] = (
        ("labor_market", SNAPSHOT_LABOR_MARKET, load_labor_market_payload),
        ("housing", SNAPSHOT_HOUSING, load_housing_payload),
        ("economic_growth", SNAPSHOT_ECONOMIC_GROWTH, load_economic_growth_payload),
    )

    results: list[dict[str, Any]] = []
    for name, snapshot_key, loader in specs:
        registry = source_registry_metadata_for_snapshot(snapshot_key)
        try:
            payload = loader()
            payload_for_write = attach_source_registry_metadata(payload, snapshot_key=snapshot_key)
            record = write_snapshot_success(
                snapshot_key,
                payload_for_write,
                as_of_date=payload_as_of(payload_for_write),
                version=SNAPSHOT_SCHEMA_VERSION,
            )
            results.append(
                {
                    "snapshot_key": snapshot_key,
                    "module": name,
                    "status": "ok",
                    "as_of": record.as_of_date,
                    "source_registry": registry,
                }
            )
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            failure_record = write_snapshot_failure(snapshot_key, message, version=SNAPSHOT_SCHEMA_VERSION)
            logger.warning("macro snapshot refresh failed for %s", name, exc_info=True)
            results.append(
                {
                    "snapshot_key": snapshot_key,
                    "module": name,
                    "status": "error",
                    "error": message,
                    "as_of": failure_record.as_of_date if failure_record else None,
                    "source_registry": registry,
                }
            )

    return {"snapshots": results}
