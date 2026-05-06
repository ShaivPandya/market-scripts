"""Typed ontology materialization for signal aggregator computed snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from api.snapshot_keys import (
    DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
    SNAPSHOT_LIQUIDITY,
    SNAPSHOT_MARKET_BREADTH,
    SNAPSHOT_MOMENTUM,
    SNAPSHOT_SECTOR_METRICS,
    SNAPSHOT_SIGNAL_AGGREGATOR,
    SNAPSHOT_VIX_TERM_STRUCTURE,
)
from ontology.command_service import OPERATIONAL_ONTOLOGY_RUN_ID
from ontology.object_service import OntologyObjectService
from ontology.schemas.identity import (
    computed_snapshot_ref_id,
    market_regime_snapshot_id,
    object_version_ref_id,
    signal_factor_score_id,
)

_FACTOR_SNAPSHOT_KEYS = {
    "vix": SNAPSHOT_VIX_TERM_STRUCTURE,
    "breadth": SNAPSHOT_MARKET_BREADTH,
    "liquidity": SNAPSHOT_LIQUIDITY,
    "sector": SNAPSHOT_SECTOR_METRICS,
    "momentum": SNAPSHOT_MOMENTUM,
}


def materialize_signal_aggregator_snapshot(
    *,
    snapshot_key: str,
    snapshot_version_id: str | None,
    payload: Mapping[str, Any] | None,
    as_of_date: str | None,
    fetched_at: str | None,
    status: str = "ok",
    quality: str = "ok",
    error: str | None = None,
    object_service: OntologyObjectService | None = None,
    provenance_id: str | None = None,
) -> list[dict[str, Any]]:
    """Write typed regime objects for the current signal aggregator payload."""

    if snapshot_key != SNAPSHOT_SIGNAL_AGGREGATOR or not isinstance(payload, Mapping):
        return []

    service = object_service or OntologyObjectService()
    valid_from = as_of_date or fetched_at or _now_fallback()
    source_snapshot_id = snapshot_version_id or _hash_value(payload)
    regime_key = f"{snapshot_key}:{source_snapshot_id}"
    provenance = provenance_id or f"pv:computed_snapshot:{_hash_text(regime_key, length=16)}"
    actor = {"actor_type": "system", "actor_id": "market_snapshot_refresh"}
    rows: list[dict[str, Any]] = []

    computed_ref = service.write_object(
        "ComputedSnapshotRef",
        snapshot_key,
        {
            "snapshot_key": snapshot_key,
            "snapshot_id": source_snapshot_id,
            "status": status,
            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
        },
        valid_from,
        actor=actor,
        provenance=provenance,
        input_hash=source_snapshot_id,
    )
    rows.append(computed_ref)

    regime = _dict(payload.get("regime"))
    module_status = _dict(payload.get("module_status"))
    market_row = service.write_object(
        "MarketRegimeSnapshot",
        regime_key,
        {
            "snapshot_id": regime_key,
            "snapshot_key": snapshot_key,
            "regime_label": regime.get("label") or "unknown",
            "score": _optional_float(regime.get("score")),
            "confidence": _optional_float(regime.get("confidence")),
            "history_percentile": _optional_float(regime.get("history_percentile")),
            "as_of_date": as_of_date or _optional_text(payload.get("as_of")),
            "fetched_at": fetched_at,
            "status": status,
            "quality": quality,
            "stale_after_hours": int(DEFAULT_SNAPSHOT_MAX_AGE_SECONDS / 3600),
            "source_status": module_status,
            "error": error,
            "weights": _dict(payload.get("weights")),
            "module_status": module_status,
            "failed_modules": _strings(payload.get("failed_modules")),
            "snapshot_payload_hash": _hash_value(payload),
            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
        },
        valid_from,
        actor=actor,
        provenance=provenance,
        input_hash=_hash_value(payload),
    )
    rows.append(market_row)
    market_uid = market_regime_snapshot_id(regime_key)

    version_id = _version_id(market_row)
    if version_id:
        ref_key = f"{market_uid}:{version_id}"
        version_ref = service.write_object(
            "ObjectVersionRef",
            ref_key,
            {
                "ref_id": ref_key,
                "object_uid": market_uid,
                "object_type": "MarketRegimeSnapshot",
                "version_id": version_id,
                "valid_from": valid_from,
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            },
            valid_from,
            actor=actor,
            provenance=provenance,
            input_hash=_hash_value(payload),
        )
        rows.append(version_ref)
        rows.append(
            service.write_relation(
                computed_snapshot_ref_id(snapshot_key),
                object_version_ref_id(ref_key),
                "computed_snapshot_materializes_object_version",
                {
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                    "object_uid": market_uid,
                    "object_type": "MarketRegimeSnapshot",
                    "version_id": version_id,
                },
                valid_from,
                actor=actor,
                provenance=provenance,
                input_hash=_hash_value(payload),
            )
        )

    rows.extend(
        _write_factor_scores(
            service=service,
            market_uid=market_uid,
            regime_key=regime_key,
            factors=payload.get("factors"),
            valid_from=valid_from,
            actor=actor,
            provenance=provenance,
        )
    )
    rows.extend(
        _write_forward_outlook(
            service=service,
            market_uid=market_uid,
            regime_key=regime_key,
            outlook=payload.get("forward_outlook"),
            as_of_date=as_of_date or _optional_text(payload.get("as_of")),
            valid_from=valid_from,
            actor=actor,
            provenance=provenance,
        )
    )
    rows.extend(
        _write_episodes(
            service=service,
            market_uid=market_uid,
            regime_key=regime_key,
            episodes=_dict(payload.get("history")).get("episodes"),
            valid_from=valid_from,
            actor=actor,
            provenance=provenance,
        )
    )
    return rows


def _write_factor_scores(
    *,
    service: OntologyObjectService,
    market_uid: str,
    regime_key: str,
    factors: Any,
    valid_from: str,
    actor: Mapping[str, Any],
    provenance: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(factors, list):
        return rows
    for index, item in enumerate(factors):
        if not isinstance(item, Mapping):
            continue
        factor_key = str(item.get("key") or item.get("factor_key") or f"factor_{index}").strip() or f"factor_{index}"
        factor_id = f"{regime_key}:factor:{factor_key}"
        factor_row = service.write_object(
            "SignalFactorScore",
            factor_id,
            {
                "factor_score_id": factor_id,
                "snapshot_id": regime_key,
                "factor_key": factor_key,
                "factor_name": str(item.get("name") or factor_key),
                "status": str(item.get("status") or "unknown"),
                "score": _optional_float(item.get("score")),
                "weight": _optional_float(item.get("weight")),
                "contribution": _optional_float(item.get("contribution")),
                "highlights": item.get("highlights"),
                "source_snapshot_key": _FACTOR_SNAPSHOT_KEYS.get(factor_key),
                "as_of_date": _optional_text(item.get("as_of") or item.get("as_of_date")),
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            },
            valid_from,
            actor=actor,
            provenance=provenance,
            input_hash=_hash_value(item),
        )
        rows.append(factor_row)
        factor_uid = signal_factor_score_id(factor_id)
        rows.append(
            service.write_relation(
                market_uid,
                factor_uid,
                "market_regime_has_factor_score",
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                valid_from,
                actor=actor,
                provenance=provenance,
                input_hash=_hash_value(item),
            )
        )
        source_snapshot_key = _FACTOR_SNAPSHOT_KEYS.get(factor_key)
        if source_snapshot_key:
            rows.append(
                service.write_object(
                    "ComputedSnapshotRef",
                    source_snapshot_key,
                    {
                        "snapshot_key": source_snapshot_key,
                        "status": str(item.get("status") or "unknown"),
                        "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                    },
                    valid_from,
                    actor=actor,
                    provenance=provenance,
                    input_hash=_hash_value(item),
                )
            )
            rows.append(
                service.write_relation(
                    factor_uid,
                    computed_snapshot_ref_id(source_snapshot_key),
                    "factor_score_uses_computed_snapshot",
                    {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                    valid_from,
                    actor=actor,
                    provenance=provenance,
                    input_hash=_hash_value(item),
                )
            )
    return rows


def _write_forward_outlook(
    *,
    service: OntologyObjectService,
    market_uid: str,
    regime_key: str,
    outlook: Any,
    as_of_date: str | None,
    valid_from: str,
    actor: Mapping[str, Any],
    provenance: str,
) -> list[dict[str, Any]]:
    if not isinstance(outlook, Mapping) or not outlook:
        return []
    outlook_key = f"{regime_key}:forward_outlook"
    row = service.write_object(
        "ForwardOutlook",
        outlook_key,
        {
            "outlook_id": outlook_key,
            "snapshot_id": regime_key,
            "label": str(outlook.get("label") or "unknown"),
            "detail": outlook.get("detail"),
            "basis": outlook.get("basis"),
            "as_of_date": as_of_date,
            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
        },
        valid_from,
        actor=actor,
        provenance=provenance,
        input_hash=_hash_value(outlook),
    )
    return [
        row,
        service.write_relation(
            market_uid,
            row["object_uid"],
            "market_regime_has_forward_outlook",
            {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
            valid_from,
            actor=actor,
            provenance=provenance,
            input_hash=_hash_value(outlook),
        ),
    ]


def _write_episodes(
    *,
    service: OntologyObjectService,
    market_uid: str,
    regime_key: str,
    episodes: Any,
    valid_from: str,
    actor: Mapping[str, Any],
    provenance: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(episodes, list):
        return rows
    for index, item in enumerate(episodes):
        if not isinstance(item, Mapping):
            continue
        episode_key = f"{regime_key}:episode:{index}:{item.get('start_date')}:{item.get('regime')}"
        row = service.write_object(
            "RegimeEpisode",
            episode_key,
            {
                "episode_id": episode_key,
                "snapshot_id": regime_key,
                "regime": str(item.get("regime") or item.get("label") or "unknown"),
                "start_date": item.get("start_date"),
                "end_date": item.get("end_date"),
                "weeks": _optional_int(item.get("weeks")),
                "avg_score": _optional_float(item.get("avg_score")),
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            },
            valid_from,
            actor=actor,
            provenance=provenance,
            input_hash=_hash_value(item),
        )
        rows.append(row)
        rows.append(
            service.write_relation(
                market_uid,
                row["object_uid"],
                "market_regime_has_episode",
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                valid_from,
                actor=actor,
                provenance=provenance,
                input_hash=_hash_value(item),
            )
        )
    return rows


def _version_id(row: Mapping[str, Any]) -> str | None:
    meta = row.get("_meta")
    if isinstance(meta, Mapping):
        temporal = meta.get("temporal")
        if isinstance(temporal, Mapping) and temporal.get("version_id"):
            return str(temporal["version_id"])
    value = row.get("version_id")
    return str(value) if value else None


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _hash_value(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _hash_text(value: str, *, length: int = 32) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _now_fallback() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
