"""Helpers for serving signal-aggregator snapshots with request-time slicing."""

from __future__ import annotations

import copy
from typing import Any

from api.signal_aggregator import _build_episodes, build_signal_aggregator_from_payloads
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
from api.snapshot_store import get_snapshot_response

_MODULE_SNAPSHOT_KEYS = {
    "market_breadth": SNAPSHOT_MARKET_BREADTH,
    "top50_breadth": SNAPSHOT_TOP50_BREADTH,
    "sector_metrics": SNAPSHOT_SECTOR_METRICS,
    "liquidity": SNAPSHOT_LIQUIDITY,
    "vix_term_structure": SNAPSHOT_VIX_TERM_STRUCTURE,
    "momentum": SNAPSHOT_MOMENTUM,
}


def _history_score(row: dict[str, Any]) -> float | None:
    try:
        value = row.get("score")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _slice_history(data: dict[str, Any], lookback_weeks: int) -> None:
    history = data.get("history")
    if not isinstance(history, dict):
        return
    series = history.get("series")
    if not isinstance(series, list):
        return
    lookback = max(26, min(int(lookback_weeks), 520))
    rows = [row for row in series if isinstance(row, dict)][-lookback:]
    history["lookback_weeks"] = lookback
    history["series"] = rows
    history["episodes"] = _build_episodes(rows)

    regime = data.get("regime")
    if not isinstance(regime, dict):
        return
    try:
        regime_score = regime.get("score")
        if regime_score is None:
            return
        composite = float(regime_score)
    except (TypeError, ValueError):
        return
    scores = [score for score in (_history_score(row) for row in rows) if score is not None]
    if scores:
        regime["history_percentile"] = round(
            (sum(1 for score in scores if score <= composite) / len(scores)) * 100.0, 2
        )


def get_signal_aggregator_snapshot_response(
    *,
    lookback_weeks: int,
    include_raw_modules: bool,
) -> dict[str, Any] | None:
    snapshot = get_snapshot_response(SNAPSHOT_SIGNAL_AGGREGATOR)
    if snapshot is None:
        return None
    data = copy.deepcopy(snapshot)
    if not include_raw_modules:
        data.pop("raw_modules", None)
    _slice_history(data, lookback_weeks)
    return data


def _snapshot_meta(payload: dict[str, Any]) -> dict[str, Any]:
    meta = payload.get("_meta")
    if isinstance(meta, dict) and isinstance(meta.get("snapshot"), dict):
        return meta["snapshot"]
    return {}


def _payload_without_meta(payload: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(payload)
    data.pop("_meta", None)
    return data


def _latest_string(values: list[Any]) -> str | None:
    clean = [str(v) for v in values if v is not None and str(v)]
    return max(clean) if clean else None


def _fallback_snapshot_meta(
    module_payloads: dict[str, dict[str, Any]],
    module_status: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metas = [_snapshot_meta(payload) for payload in module_payloads.values()]
    errors = [
        f"{module}: {state.get('detail')}"
        for module, state in module_status.items()
        if state.get("status") != "ok" and state.get("detail")
    ]
    ages = [m.get("data_age_seconds") for m in metas if isinstance(m.get("data_age_seconds"), int | float)]
    return {
        "key": SNAPSHOT_SIGNAL_AGGREGATOR,
        "source": "module_snapshots",
        "as_of": _latest_string([m.get("as_of") for m in metas]),
        "fetched_at": _latest_string([m.get("fetched_at") for m in metas]),
        "data_age_seconds": max(ages) if ages else None,
        "stale": any(bool(m.get("stale")) for m in metas),
        "refresh_status": "ok" if not errors else "degraded",
        "error": "; ".join(errors[:3]) if errors else None,
        "version": SNAPSHOT_SCHEMA_VERSION,
    }


def get_signal_aggregator_module_snapshot_response(
    *,
    lookback_weeks: int,
    include_raw_modules: bool,
) -> dict[str, Any] | None:
    """Synthesize the current aggregator from module snapshots when the aggregate snapshot is absent."""
    raw: dict[str, Any] = {}
    module_status: dict[str, dict[str, Any]] = {}
    module_payloads: dict[str, dict[str, Any]] = {}

    for module_name, snapshot_key in _MODULE_SNAPSHOT_KEYS.items():
        payload = get_snapshot_response(snapshot_key)
        if payload is None:
            raw[module_name] = None
            module_status[module_name] = {"status": "error", "detail": f"Snapshot unavailable: {snapshot_key}"}
            continue

        module_payloads[module_name] = payload
        raw[module_name] = _payload_without_meta(payload)
        meta = _snapshot_meta(payload)
        refresh_status = str(meta.get("refresh_status") or "ok")
        if refresh_status != "ok":
            module_status[module_name] = {
                "status": "error",
                "detail": str(meta.get("error") or f"snapshot refresh {refresh_status}"),
            }
        else:
            module_status[module_name] = {"status": "ok"}

    if not module_payloads:
        return None

    try:
        data = build_signal_aggregator_from_payloads(
            raw,
            module_status,
            lookback_weeks=lookback_weeks,
            include_raw_modules=include_raw_modules,
            include_history=False,
        )
    except Exception:
        return None

    meta = data.get("_meta") if isinstance(data.get("_meta"), dict) else {}
    meta["snapshot"] = _fallback_snapshot_meta(module_payloads, module_status)
    data["_meta"] = meta
    return data


def get_signal_aggregator_snapshot_or_module_response(
    *,
    lookback_weeks: int,
    include_raw_modules: bool,
) -> dict[str, Any] | None:
    snapshot = get_signal_aggregator_snapshot_response(
        lookback_weeks=lookback_weeks,
        include_raw_modules=include_raw_modules,
    )
    if snapshot is not None:
        return snapshot
    return get_signal_aggregator_module_snapshot_response(
        lookback_weeks=lookback_weeks,
        include_raw_modules=include_raw_modules,
    )
