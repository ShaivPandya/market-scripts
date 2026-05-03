from __future__ import annotations

from typing import Any, cast

from ontology.schemas.base import expected_risk_level
from ontology.schemas.identity import (
    asset_id,
    canonical_ticker,
    catalyst_id,
    evaluation_id,
    macro_indicator_id,
    position_id,
    sector_id,
    signal_id,
    slug,
    thesis_id,
)

ALLOWED_SIGNAL_DIRECTIONS = {"deteriorating", "stable", "improving", "neutral", "unknown"}


def adapt_node_payload(
    *,
    node_id: str,
    node_type: str,
    label: str,
    properties: dict[str, Any],
    run_id: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    props = dict(properties or {})
    if props.get("schema_version") == 1:
        return node_id, label, props

    if node_type == "Position":
        ticker = _ticker_from(node_id, label, props, prefix="position")
        score = _score(props.get("risk_score"), default=0.0)
        payload = {
            "schema_version": 1,
            "ticker": ticker,
            "asset": _lower(props.get("asset"), default="unknown"),
            "direction": _lower(props.get("direction"), default="unknown"),
            "timeframe": _text(props.get("timeframe"), default="unknown"),
            "latest_price": _optional_float(props.get("latest_price")),
            "as_of": _optional_text(props.get("as_of")),
            "risk_score": score,
            "risk_level": expected_risk_level(score),
            "volatility_cluster": _score(props.get("volatility_cluster"), default=0.0),
            "breadth_stress": _score(props.get("breadth_stress"), default=0.0),
            "sector_stress": _score(props.get("sector_stress"), default=0.0),
            "macro_regime": _score(props.get("macro_regime"), default=0.0),
            "ontology_run_id": _text(props.get("ontology_run_id") or run_id, default="legacy"),
        }
        return position_id(ticker), ticker, payload

    if node_type == "Asset":
        ticker = _ticker_from(node_id, label, props, prefix="asset")
        payload = {
            "schema_version": 1,
            "ticker": ticker,
            "asset": _lower(props.get("asset"), default="unknown"),
            "name": _optional_text(props.get("name")),
            "currency": _optional_text(props.get("currency")),
            "exchange": _optional_text(props.get("exchange")),
        }
        return asset_id(ticker), ticker, payload

    if node_type == "Sector":
        name = _text(props.get("name") or label, default="Unknown Equity")
        payload = {
            "schema_version": 1,
            "name": name,
            "sector_source": _text(props.get("sector_source") or props.get("source"), default="legacy"),
        }
        return sector_id(name), name, payload

    if node_type == "MacroIndicator":
        indicator_key = _indicator_key_from(node_id, label, props)
        name = _text(props.get("name") or label, default=indicator_key.replace("_", " ").title())
        payload = {
            "schema_version": 1,
            "indicator_key": indicator_key,
            "name": name,
            "source": _text(props.get("source"), default=indicator_key),
            "as_of": _text(props.get("as_of") or run_id, default="legacy"),
            "ontology_run_id": _text(props.get("ontology_run_id") or run_id, default="legacy"),
        }
        return macro_indicator_id(indicator_key), name, payload

    if node_type == "Signal":
        source = _text(props.get("source") or _signal_source_from_id(node_id), default="legacy")
        name = _text(props.get("name") or label, default="Signal")
        direction = str(props.get("direction") or "").strip().lower()
        payload = {
            "schema_version": 1,
            "signal_key": slug(f"{source}:{name}"),
            "name": name,
            "source": source,
            "value": props.get("value"),
            "threshold": _text(props.get("threshold"), default="unknown"),
            "direction": direction if direction in ALLOWED_SIGNAL_DIRECTIONS else "unknown",
            "raw_signal": props.get("raw_signal"),
            "component": _optional_text(props.get("component")),
            "sector": _optional_text(props.get("sector")),
            "ontology_run_id": _text(props.get("ontology_run_id") or run_id, default="legacy"),
        }
        return signal_id(source, name), name, payload

    if node_type == "Thesis":
        ticker = _ticker_from(node_id, label, props, prefix="thesis")
        status = str(props.get("status") or "active").strip()
        if status not in {"active", "under_review", "invalidated"}:
            status = "active"
        payload = {
            "schema_version": 1,
            "ticker": ticker,
            "status": status,
            "created_at": _text(props.get("created_at") or run_id, default="legacy"),
            "updated_at": _text(props.get("updated_at") or run_id, default="legacy"),
            "ontology_run_id": _text(props.get("ontology_run_id") or run_id, default="legacy"),
        }
        return thesis_id(ticker), f"Thesis: {ticker}", payload

    if node_type == "Evaluation":
        ticker = _ticker_from(node_id, label, props, prefix="evaluation")
        evaluated_at = _text(props.get("evaluated_at") or _suffix_after(node_id, 2), default="latest")
        payload = {
            "schema_version": 1,
            "ticker": ticker,
            "evaluated_at": evaluated_at,
            "thesis_status": _text(props.get("thesis_status"), default="unknown"),
            "technical_read": _text(props.get("technical_read"), default="unknown"),
            "fundamental_read": _text(props.get("fundamental_read"), default="unknown"),
            "action": _text(props.get("action"), default="unknown"),
            "confidence": _text(props.get("confidence"), default="unknown"),
            "risk_flag": _optional_text(props.get("risk_flag")),
            "key_developments": _list_text(props.get("key_developments")),
            "ontology_run_id": _text(props.get("ontology_run_id") or run_id, default="legacy"),
        }
        return evaluation_id(ticker, evaluated_at), f"Eval: {ticker}", payload

    if node_type == "Catalyst":
        ticker = _ticker_from(node_id, label, props, prefix="catalyst")
        name = _text(props.get("name") or label, default="Catalyst")
        description = _text(props.get("description") or name, default=name)
        payload = {
            "schema_version": 1,
            "ticker": ticker,
            "name": name,
            "description": description,
            "source": _text(props.get("source"), default="thesis_markdown"),
            "category": _optional_text(props.get("category")),
            "target_date": _optional_text(props.get("target_date")),
            "status": _optional_text(props.get("status")),
            "ontology_run_id": _text(props.get("ontology_run_id") or run_id, default="legacy"),
        }
        return catalyst_id(ticker, name, description), name, payload

    raise ValueError(f"Unsupported ontology node type: {node_type}")


def adapt_edge_payload(
    *,
    relation_type: str,
    properties: dict[str, Any],
    run_id: str | None = None,
) -> dict[str, Any]:
    props = dict(properties or {})
    if props.get("schema_version") == 1:
        return props
    if relation_type == "exposed_to_signal":
        direction = str(props.get("direction") or "").strip().lower()
        return {
            "schema_version": 1,
            "component": _text(props.get("component"), default="unknown"),
            "source": _text(props.get("source"), default="unknown"),
            "name": _text(props.get("name"), default="Signal"),
            "value": props.get("value"),
            "threshold": _text(props.get("threshold"), default="unknown"),
            "direction": direction if direction in ALLOWED_SIGNAL_DIRECTIONS else "unknown",
            "contribution": _score(props.get("contribution"), default=0.0),
            "ontology_run_id": _text(props.get("ontology_run_id") or run_id, default="legacy"),
        }
    return {
        "schema_version": 1,
        "ontology_run_id": _text(props.get("ontology_run_id") or run_id, default="legacy"),
        "source": _text(props.get("source"), default="legacy")
        if relation_type == "belongs_to_sector"
        else _optional_text(props.get("source")),
    }


def _ticker_from(node_id: str, label: str, props: dict[str, Any], *, prefix: str) -> str:
    raw = props.get("ticker")
    if raw is None and node_id.startswith(f"{prefix}:"):
        raw = node_id.split(":", 2)[1]
    if raw is None:
        raw = label
    return canonical_ticker(raw)


def _indicator_key_from(node_id: str, label: str, props: dict[str, Any]) -> str:
    raw = props.get("indicator_key")
    if raw is None and node_id.startswith("macro_indicator:"):
        raw = node_id.split(":", 1)[1]
    if raw is None:
        raw = label
    return slug(raw)


def _signal_source_from_id(node_id: str) -> str | None:
    if not node_id.startswith("signal:"):
        return None
    parts = node_id.split(":")
    return parts[1] if len(parts) > 2 else None


def _suffix_after(text: str, count: int) -> str | None:
    parts = text.split(":", count)
    return parts[count] if len(parts) > count else None


def _text(value: object, *, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _lower(value: object, *, default: str) -> str:
    return _text(value, default=default).lower()


def _score(value: object, *, default: float) -> float:
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _optional_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _list_text(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
