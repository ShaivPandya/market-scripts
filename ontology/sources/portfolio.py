from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ontology.runtime_read_service import OntologyRuntimeReadService
from ontology.sources.base import (
    as_dict,
    build_source_result,
    clean_str,
    iso_string,
    latest_series_value,
    schema_issue,
    series_point_count,
    status_for_drift,
    unknown_fields,
)
from ontology.sources.dtos import PortfolioMetadata, PortfolioPosition, PortfolioSnapshot


class PortfolioAdapter:
    source_name = "portfolio"
    source_version = "1"
    required = True
    raw_module = "portfolio.portfolio_dashboard"
    raw_function = "get_data"

    def __init__(self, *, timeframe: str):
        self.timeframe = timeframe
        self.parameters = {"timeframe": timeframe, "all_timeframes": False}

    def fetch(self) -> dict[str, Any]:
        positions = OntologyRuntimeReadService().positions(include_hedges=True)
        timestamp = datetime.now(UTC).isoformat()
        metadata = {}
        position_order = []
        for row in positions:
            ticker = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
            if not ticker:
                continue
            position_order.append(ticker)
            metadata[ticker] = dict(row)
        return {
            "positions": {ticker: [] for ticker in position_order},
            "metadata": metadata,
            "timeframe": self.timeframe,
            "timestamp": timestamp,
            "position_order": position_order,
            "analytics": {"source": "ontology"},
        }

    def normalize(self, raw: Any):
        if not isinstance(raw, dict):
            return build_source_result(
                self,
                raw,
                None,
                status="error",
                quality="missing",
                as_of=None,
                detail="portfolio payload is not a dict",
            )

        expected = {"positions", "metadata", "timeframe", "timestamp", "position_order", "analytics", "warning"}
        drift = unknown_fields(raw, expected)

        metadata_raw = raw.get("metadata")
        positions_raw = raw.get("positions")
        if not isinstance(metadata_raw, dict):
            drift.append(schema_issue("warning", "$.metadata", "dict[ticker, metadata]", metadata_raw, "defaulted"))
            metadata_raw = {}
        if not isinstance(positions_raw, dict):
            drift.append(
                schema_issue("warning", "$.positions", "dict[ticker, price series]", positions_raw, "defaulted")
            )
            positions_raw = {}

        position_order = _string_list(raw.get("position_order"))
        tickers = position_order or list(metadata_raw.keys()) or list(positions_raw.keys())
        positions: dict[str, PortfolioPosition] = {}
        for ticker_obj in tickers:
            ticker = str(ticker_obj).strip().upper()
            if not ticker:
                continue
            meta_raw = as_dict(metadata_raw.get(ticker) or metadata_raw.get(str(ticker_obj)))
            metadata = PortfolioMetadata(
                ticker=ticker,
                asset=str(meta_raw.get("asset") or "unknown").strip().lower(),
                direction=str(meta_raw.get("direction") or "unknown").strip().lower(),
                instrument_type=str(meta_raw.get("instrument_type") or "security").strip().lower(),
                price_symbol=clean_str(meta_raw.get("price_symbol")) or ticker,
                quantity=_float_or_none(
                    meta_raw.get("quantity") if meta_raw.get("quantity") is not None else meta_raw.get("shares")
                ),
                contract_multiplier=_float_or_none(meta_raw.get("contract_multiplier")) or 1.0,
                fx_base_currency=clean_str(meta_raw.get("fx_base_currency")),
                fx_quote_currency=clean_str(meta_raw.get("fx_quote_currency")),
                raw=dict(meta_raw),
            )
            series = _get_mapping_value(positions_raw, ticker, str(ticker_obj))
            positions[ticker] = PortfolioPosition(
                ticker=ticker,
                asset=metadata.asset,
                direction=metadata.direction,
                latest_price=latest_series_value(series),
                series_points=series_point_count(series),
                as_of=iso_string(raw.get("timestamp")),
                metadata=metadata,
                instrument_type=metadata.instrument_type,
                price_symbol=metadata.price_symbol,
                quantity=metadata.quantity,
                contract_multiplier=metadata.contract_multiplier,
                fx_base_currency=metadata.fx_base_currency,
                fx_quote_currency=metadata.fx_quote_currency,
            )

        snapshot = PortfolioSnapshot(
            positions=positions,
            timeframe=str(clean_str(raw.get("timeframe")) or self.timeframe),
            timestamp=iso_string(raw.get("timestamp")),
            position_order=list(positions.keys()),
            analytics=as_dict(raw.get("analytics")),
        )
        coverage = {
            "positions": len(snapshot.positions),
            "priced_positions": sum(1 for item in snapshot.positions.values() if item.latest_price is not None),
            "metadata_count": len(metadata_raw),
        }

        if not snapshot.positions:
            return build_source_result(
                self,
                raw,
                snapshot,
                status="error",
                quality="missing",
                as_of=snapshot.timestamp,
                schema_drift=drift,
                detail=str(raw.get("warning") or "no portfolio positions found"),
                coverage=coverage,
            )

        status, quality = status_for_drift(base_status="ok", base_quality="ok", drift=drift)
        return build_source_result(
            self,
            raw,
            snapshot,
            status=status,
            quality=quality,
            as_of=snapshot.timestamp,
            schema_drift=drift,
            detail=clean_str(raw.get("warning")),
            coverage=coverage,
        )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip().upper() for item in value if str(item).strip()]


def _get_mapping_value(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out
