"""Daily refresh job for expensive market/regime snapshots."""

from __future__ import annotations

import logging
from typing import Any

from api.serializers import serialize_value
from api.snapshot_keys import (
    SNAPSHOT_LIQUIDITY,
    SNAPSHOT_MARKET_BREADTH,
    SNAPSHOT_MOMENTUM,
    SNAPSHOT_SCHEMA_VERSION,
    SNAPSHOT_SECTOR_METRICS,
    SNAPSHOT_SIGNAL_AGGREGATOR,
    SNAPSHOT_TOP50_BREADTH,
    SNAPSHOT_VIX_TERM_STRUCTURE,
)
from api.snapshot_store import write_snapshot_failure, write_snapshot_success

logger = logging.getLogger("api.market_snapshots")

_MODULE_SNAPSHOT_KEYS = {
    "market_breadth": SNAPSHOT_MARKET_BREADTH,
    "top50_breadth": SNAPSHOT_TOP50_BREADTH,
    "sector_metrics": SNAPSHOT_SECTOR_METRICS,
    "liquidity": SNAPSHOT_LIQUIDITY,
    "vix_term_structure": SNAPSHOT_VIX_TERM_STRUCTURE,
    "momentum": SNAPSHOT_MOMENTUM,
}


def _payload_as_of(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("as_of", "as_of_date", "latest_date", "date", "timestamp"):
        value = payload.get(key)
        if value is not None:
            return str(value)[:32]
    latest = payload.get("latest_df")
    if isinstance(latest, list) and latest and isinstance(latest[0], dict):
        value = latest[0].get("Date") or latest[0].get("date")
        if value is not None:
            return str(value)[:32]
    return None


def _module_status(signal_payload: dict[str, Any], module_name: str) -> tuple[str, str | None]:
    status = signal_payload.get("module_status")
    if not isinstance(status, dict):
        return "error", "missing module_status"
    state = status.get(module_name)
    if not isinstance(state, dict):
        return "error", "missing module status"
    if state.get("status") == "ok":
        return "ok", None
    return "error", str(state.get("detail") or "module refresh failed")


def refresh_market_snapshots(_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute and persist daily market/regime snapshots.

    The signal aggregator already performs one shared S&P 500 constituent price
    download and passes that frame into breadth, top-50 breadth, and sector
    metrics. Running this job after the close turns those expensive request-path
    operations into snapshot reads.
    """
    from api.signal_aggregator import build_signal_aggregator

    try:
        signal = build_signal_aggregator(
            lookback_weeks=520,
            include_raw_modules=True,
        )
    except Exception as exc:
        message = str(exc) or exc.__class__.__name__
        write_snapshot_failure(SNAPSHOT_SIGNAL_AGGREGATOR, message, version=SNAPSHOT_SCHEMA_VERSION)
        logger.warning("market snapshot refresh failed before signal payload", exc_info=True)
        raise

    signal_payload = serialize_value(signal)
    raw_modules = signal_payload.get("raw_modules")
    if not isinstance(raw_modules, dict):
        raw_modules = {}

    results: list[dict[str, Any]] = []
    for module_name, snapshot_key in _MODULE_SNAPSHOT_KEYS.items():
        status, error = _module_status(signal_payload, module_name)
        payload = raw_modules.get(module_name)
        if status == "ok" and isinstance(payload, dict):
            record = write_snapshot_success(
                snapshot_key,
                payload,
                as_of_date=_payload_as_of(payload),
                version=SNAPSHOT_SCHEMA_VERSION,
            )
            results.append({"snapshot_key": snapshot_key, "status": "ok", "as_of": record.as_of_date})
        else:
            failure_record = write_snapshot_failure(
                snapshot_key, error or "missing module payload", version=SNAPSHOT_SCHEMA_VERSION
            )
            results.append(
                {
                    "snapshot_key": snapshot_key,
                    "status": "error",
                    "error": error or "missing module payload",
                    "as_of": failure_record.as_of_date if failure_record else None,
                }
            )

    signal_record = write_snapshot_success(
        SNAPSHOT_SIGNAL_AGGREGATOR,
        signal_payload,
        as_of_date=str(signal_payload.get("as_of")) if signal_payload.get("as_of") else None,
        version=SNAPSHOT_SCHEMA_VERSION,
    )
    results.append({"snapshot_key": SNAPSHOT_SIGNAL_AGGREGATOR, "status": "ok", "as_of": signal_record.as_of_date})

    return {"snapshots": results}
