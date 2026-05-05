from fastapi import APIRouter, Query

from api.cache import get_or_set_cached, short_cache
from api.exceptions import DataFetchError, SnapshotUnavailableError
from api.serializers import serialize_value
from api.signal_aggregator import (
    DEFAULT_LOOKBACK_WEEKS,
    DEFAULT_POSITIONING_INSTRUMENTS,
    build_signal_aggregator,
)
from api.signal_snapshot import get_signal_aggregator_snapshot_or_module_response
from api.snapshot_keys import SNAPSHOT_SIGNAL_AGGREGATOR
from api.snapshot_store import snapshots_required

router = APIRouter()


def _normalize_positioning_instruments(value: str) -> str:
    aliases = [part.strip().upper() for part in (value or "").split(",") if part.strip()]
    return ",".join(aliases) or DEFAULT_POSITIONING_INSTRUMENTS


def _snapshot_missing_liquidity(snapshot: dict) -> bool:
    module_status = snapshot.get("module_status")
    if isinstance(module_status, dict):
        liquidity_status = module_status.get("liquidity")
        if isinstance(liquidity_status, dict):
            if liquidity_status.get("status") == "ok":
                return False
            return "Snapshot unavailable" in str(liquidity_status.get("detail") or "")

    meta = snapshot.get("_meta")
    snapshot_meta = meta.get("snapshot") if isinstance(meta, dict) else None
    if not isinstance(snapshot_meta, dict):
        return False
    error = str(snapshot_meta.get("error") or "")
    return "liquidity" in error and "Snapshot unavailable" in error


@router.get("/signal-aggregator")
def get_signal_aggregator(
    lookback_weeks: int = Query(DEFAULT_LOOKBACK_WEEKS, ge=26, le=520),
    positioning_instruments: str = Query(DEFAULT_POSITIONING_INSTRUMENTS),
    include_raw_modules: bool = Query(False),
    force_refresh: bool = Query(False),
):
    normalized_instruments = _normalize_positioning_instruments(positioning_instruments)
    key = f"signal_aggregator:{lookback_weeks}:positioning={normalized_instruments}:include_raw={include_raw_modules}"

    def loader():
        require_snapshots = snapshots_required()
        if not force_refresh:
            snapshot = get_signal_aggregator_snapshot_or_module_response(
                lookback_weeks=lookback_weeks,
                include_raw_modules=include_raw_modules,
            )
            if snapshot is not None and (require_snapshots or not _snapshot_missing_liquidity(snapshot)):
                return snapshot

        if require_snapshots and not force_refresh:
            raise SnapshotUnavailableError(SNAPSHOT_SIGNAL_AGGREGATOR)

        try:
            data = build_signal_aggregator(
                lookback_weeks=lookback_weeks,
                positioning_instruments=normalized_instruments,
                include_raw_modules=include_raw_modules,
            )
        except Exception as e:
            raise DataFetchError(source="signal_aggregator", detail=str(e)) from e

        return serialize_value(data)

    return get_or_set_cached(short_cache, key, loader, force_refresh=force_refresh)
