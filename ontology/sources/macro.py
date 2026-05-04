from __future__ import annotations

import os
from typing import Any

from ontology.sources.base import (
    SourceQuality,
    SourceStatus,
    as_dict,
    as_rows,
    build_source_result,
    clean_str,
    iso_string,
    schema_issue,
    status_for_drift,
    to_float,
    unknown_fields,
)
from ontology.sources.dtos import (
    EconomicGrowthSnapshot,
    LaborIndicator,
    LaborMarketSnapshot,
    PositioningRow,
    PositioningSnapshot,
    SentimentSnapshot,
)


class SentimentAdapter:
    source_name = "sentiment"
    source_version = "1"
    required = False
    raw_module = "macro.sentiment.sentiment"
    raw_function = "get_put_call/get_surveys/get_volatility"
    parameters: dict[str, Any] = {"put_call_lookback_days": 180, "volatility_lookback_days": 365}

    def fetch(self) -> dict[str, Any]:
        from macro.sentiment.sentiment import get_put_call, get_surveys, get_volatility

        return {
            "put_call": get_put_call(lookback_days=180),
            "surveys": get_surveys(),
            "volatility": get_volatility(lookback_days=365),
        }

    def normalize(self, raw: Any):
        if not isinstance(raw, dict):
            return build_source_result(
                self, raw, None, status="error", quality="missing", as_of=None, detail="payload is not a dict"
            )

        expected = {"put_call", "surveys", "volatility"}
        drift = unknown_fields(raw, expected)
        volatility = as_rows(raw.get("volatility"))
        latest_vvix = to_float(volatility[-1].get("vvix")) if volatility else None
        surveys = as_dict(raw.get("surveys"))
        errors = as_dict(surveys.get("errors"))
        snapshot = SentimentSnapshot(
            put_call=as_dict(raw.get("put_call")),
            surveys=surveys,
            volatility=volatility,
            latest_vvix=latest_vvix,
        )
        status, quality = status_for_drift(
            base_status="partial" if errors else "ok",
            base_quality="degraded" if errors else "ok",
            drift=drift,
        )
        return build_source_result(
            self,
            raw,
            snapshot,
            status=status,
            quality=quality,
            as_of=volatility[-1].get("date") if volatility else None,
            schema_drift=drift,
            detail="; ".join(f"{key}: {value}" for key, value in errors.items()) if errors else None,
            coverage={"volatility_rows": len(volatility), "survey_errors": len(errors)},
        )


class PositioningAdapter:
    source_name = "positioning_summary"
    source_version = "1"
    required = False
    raw_module = "macro.positioning.positioning"
    raw_function = "fetch_multiple_instruments"

    def __init__(
        self,
        *,
        instruments: str = "SP500,NASDAQ,RUSSELL,US10Y,EUR",
        start: str = "2015-01-01",
        end: str | None = None,
        groups: str | None = None,
        z_window: int = 0,
        force_threshold: float = 2.0,
    ):
        self.parameters = {
            "instruments": instruments,
            "start": start,
            "end": end,
            "groups": groups,
            "z_window": z_window,
            "force_threshold": force_threshold,
        }

    def fetch(self) -> list[dict[str, Any]]:
        from macro.positioning.positioning import DATASETS, DEFAULT_DOMAIN, fetch_multiple_instruments

        instrument_list = [i.strip() for i in str(self.parameters["instruments"]).split(",") if i.strip()]
        end_value = self.parameters.get("end")
        groups_value = self.parameters.get("groups")
        z_window_value = self.parameters.get("z_window")
        force_threshold_value = self.parameters.get("force_threshold")
        return fetch_multiple_instruments(
            domain=DEFAULT_DOMAIN,
            dataset_id=DATASETS.get("tff_futures_only", "tff_futures_only"),
            app_token=os.environ.get("SODA_APP_TOKEN") or None,
            instruments=instrument_list,
            start=str(self.parameters["start"]),
            end=str(end_value) if end_value is not None else None,
            groups=str(groups_value) if groups_value else None,
            z_window=int(z_window_value) if z_window_value is not None else 0,
            force_threshold=float(force_threshold_value) if force_threshold_value is not None else 2.0,
        )

    def normalize(self, raw: Any):
        raw_rows = as_rows(raw.get("rows")) if isinstance(raw, dict) and "rows" in raw else as_rows(raw)
        rows: list[PositioningRow] = []
        for row in raw_rows:
            instrument = clean_str(row.get("instrument"))
            if not instrument:
                continue
            rows.append(
                PositioningRow(
                    instrument=instrument,
                    report_date=clean_str(row.get("report_date")),
                    lf_net=to_float(row.get("lf_net")),
                    lf_net_pct_oi=to_float(row.get("lf_net_pct_oi")),
                    lf_z=to_float(row.get("lf_z")),
                    lf_deleveraging_z=to_float(row.get("lf_deleveraging_z")),
                    lf_forced=clean_str(row.get("lf_forced")),
                    raw=dict(row),
                )
            )

        snapshot = PositioningSnapshot(rows=rows)
        status: SourceStatus = "ok" if rows else "partial"
        quality: SourceQuality = "ok" if rows else "missing"
        return build_source_result(
            self,
            raw,
            snapshot,
            status=status,
            quality=quality,
            as_of=max((row.report_date for row in rows if row.report_date), default=None),
            detail=None if rows else "no positioning rows",
            coverage={"rows": len(rows)},
        )


class EconomicGrowthAdapter:
    source_name = "economic_growth"
    source_version = "1"
    required = False
    raw_module = "macro.economic_growth.economic_growth"
    raw_function = "get_data"
    parameters: dict[str, Any] = {}

    def fetch(self) -> dict[str, Any]:
        from macro.economic_growth.economic_growth import get_data

        return get_data()

    def normalize(self, raw: Any):
        if not isinstance(raw, dict):
            return build_source_result(
                self, raw, None, status="error", quality="missing", as_of=None, detail="payload is not a dict"
            )

        raw = _normalize_currency_payload(dict(raw))
        expected = {
            "commodities",
            "equities",
            "equity_relative_returns",
            "currencies",
            "equity_periods",
            "currency_periods",
            "crb_available",
            "crb_filename",
            "crb_uploaded_at",
            "crb_latest_date",
            "crb_latest_value",
            "crb_rows",
            "timestamp",
            "benchmarks",
        }
        drift = unknown_fields(raw, expected)
        for key in ("commodities", "equities", "currencies"):
            if key not in raw:
                drift.append(schema_issue("warning", f"$.{key}", "dict", None, "defaulted"))

        snapshot = EconomicGrowthSnapshot(
            commodities=as_dict(raw.get("commodities")),
            equities=as_dict(raw.get("equities")),
            equity_relative_returns=as_dict(raw.get("equity_relative_returns")),
            currencies=as_dict(raw.get("currencies")),
            timestamp=iso_string(raw.get("timestamp")),
            crb_metadata={
                "available": raw.get("crb_available"),
                "filename": raw.get("crb_filename"),
                "uploaded_at": raw.get("crb_uploaded_at"),
                "latest_date": raw.get("crb_latest_date"),
                "latest_value": raw.get("crb_latest_value"),
                "rows": raw.get("crb_rows"),
            },
        )
        status, quality = status_for_drift(base_status="ok", base_quality="ok", drift=drift)
        return build_source_result(
            self,
            raw,
            snapshot,
            status=status,
            quality=quality,
            as_of=snapshot.timestamp or clean_str(raw.get("crb_latest_date")),
            schema_drift=drift,
            coverage={
                "commodities": len(snapshot.commodities),
                "equities": len(snapshot.equities),
                "currencies": len(snapshot.currencies),
            },
        )


class LaborMarketAdapter:
    source_name = "labor_market"
    source_version = "1"
    required = False
    raw_module = "macro.labor_market.labor_market"
    raw_function = "get_data"
    parameters: dict[str, Any] = {}

    def fetch(self) -> dict[str, Any]:
        from macro.labor_market.labor_market import get_data

        return get_data()

    def normalize(self, raw: Any):
        if not isinstance(raw, dict):
            return build_source_result(
                self, raw, None, status="error", quality="missing", as_of=None, detail="payload is not a dict"
            )

        expected = {"series", "latest", "timestamp"}
        drift = unknown_fields(raw, expected)
        latest_raw = raw.get("latest")
        if not isinstance(latest_raw, dict):
            drift.append(schema_issue("warning", "$.latest", "dict[indicator, latest]", latest_raw, "defaulted"))
            latest_raw = {}

        labels: dict[str, str] = {}
        units: dict[str, str] = {}
        series = as_dict(raw.get("series"))
        for key, value in series.items():
            if isinstance(value, dict):
                if value.get("label") is not None:
                    labels[str(key)] = str(value.get("label"))
                if value.get("unit") is not None:
                    units[str(key)] = str(value.get("unit"))

        latest: dict[str, LaborIndicator] = {}
        for key, info in latest_raw.items():
            bucket = as_dict(info)
            latest[str(key)] = LaborIndicator(
                key=str(key),
                value=to_float(bucket.get("value")),
                date=clean_str(bucket.get("date")),
                change=to_float(bucket.get("change")),
                label=labels.get(str(key)),
                unit=units.get(str(key)),
            )

        initial_claims = latest.get("initial_claims")
        snapshot = LaborMarketSnapshot(
            latest=latest,
            timestamp=iso_string(raw.get("timestamp")),
            series_labels=labels,
            series_units=units,
            initial_claims_change=initial_claims.change if initial_claims else None,
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
            coverage={"latest_indicators": len(latest), "series": len(series)},
        )


def _normalize_currency_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required_periods = ["1-mo", "3-mo", "6-mo", "1-yr"]
    periods = payload.get("currency_periods")
    normalized_periods = [p for p in periods if isinstance(p, str)] if isinstance(periods, list) else []
    for period in required_periods:
        if period not in normalized_periods:
            normalized_periods.append(period)
    payload["currency_periods"] = normalized_periods

    currencies = payload.get("currencies")
    if isinstance(currencies, dict):
        for returns in currencies.values():
            if isinstance(returns, dict):
                for period in normalized_periods:
                    returns.setdefault(period, None)
    return payload
