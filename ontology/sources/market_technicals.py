from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from ontology.sources.base import (
    as_dict,
    build_source_result,
    clean_str,
    first_row,
    schema_issue,
    status_for_drift,
    to_float,
    to_int,
    unknown_fields,
)
from ontology.sources.dtos import MarketBreadthSnapshot, Top50BreadthSnapshot, VixTermStructureSnapshot


class MarketBreadthAdapter:
    source_name = "market_breadth"
    source_version = "1"
    required = True
    raw_module = "equities.market_technicals.market_breadth"
    raw_function = "get_data"
    parameters: dict[str, Any] = {}

    def fetch(self) -> dict[str, Any]:
        from equities.market_technicals.market_breadth import get_data

        return get_data()

    def normalize(self, raw: Any):
        if not isinstance(raw, dict):
            return build_source_result(
                self, raw, None, status="error", quality="missing", as_of=None, detail="payload is not a dict"
            )

        expected = {
            "above_200dma",
            "above_20dma",
            "at_20day_high",
            "at_20day_low",
            "at_52wk_high",
            "at_52wk_low",
            "at_24wk_high",
            "at_24wk_low",
            "total_analyzed",
            "pct_above_200dma",
            "pct_above_20dma",
            "pct_at_20day_high",
            "pct_at_20day_low",
            "pct_at_52wk_high",
            "pct_at_52wk_low",
            "pct_at_24wk_high",
            "pct_at_24wk_low",
            "as_of_date",
            "failed_tickers",
            "tickers",
        }
        drift = unknown_fields(raw, expected)
        for key in ("pct_above_200dma", "pct_above_20dma", "pct_at_20day_low", "pct_at_52wk_low"):
            if key not in raw:
                drift.append(schema_issue("warning", f"$.{key}", "number", None, "risk scoring will use fallback"))

        failed = raw.get("failed_tickers")
        failed_count = len(failed) if isinstance(failed, list) else 0
        snapshot = MarketBreadthSnapshot(
            total_analyzed=to_int(raw.get("total_analyzed")),
            pct_above_200dma=to_float(raw.get("pct_above_200dma")),
            pct_above_20dma=to_float(raw.get("pct_above_20dma")),
            pct_at_20day_low=to_float(raw.get("pct_at_20day_low")),
            pct_at_52wk_low=to_float(raw.get("pct_at_52wk_low")),
            as_of_date=clean_str(raw.get("as_of_date")),
            failed_ticker_count=failed_count,
        )
        base_quality = "degraded" if failed_count else "ok"
        status, quality = status_for_drift(base_status="ok", base_quality=base_quality, drift=drift)
        return build_source_result(
            self,
            raw,
            snapshot,
            status=status,
            quality=quality,
            as_of=snapshot.as_of_date,
            schema_drift=drift,
            coverage={"total_analyzed": snapshot.total_analyzed, "failed_tickers": failed_count},
        )


class Top50BreadthAdapter:
    source_name = "top50_breadth"
    source_version = "1"
    required = True
    raw_module = "equities.market_technicals.top50_breadth"
    raw_function = "get_data"
    parameters: dict[str, Any] = {}

    def fetch(self) -> dict[str, Any]:
        from equities.market_technicals.top50_breadth import get_data

        return get_data()

    def normalize(self, raw: Any):
        if not isinstance(raw, dict):
            return build_source_result(
                self, raw, None, status="error", quality="missing", as_of=None, detail="payload is not a dict"
            )

        expected = {
            "pct_below_50dma",
            "pct_3plus_dist",
            "pct_broke_20low",
            "tickers_below_50dma",
            "tickers_3plus_dist",
            "tickers_broke_20low",
            "universe_size",
            "raw_df",
        }
        drift = unknown_fields(raw, expected)
        for key in ("pct_below_50dma", "pct_3plus_dist", "pct_broke_20low"):
            if key not in raw:
                drift.append(schema_issue("warning", f"$.{key}", "number", None, "risk scoring will use fallback"))

        snapshot = Top50BreadthSnapshot(
            pct_below_50dma=to_float(raw.get("pct_below_50dma")),
            pct_3plus_dist=to_float(raw.get("pct_3plus_dist")),
            pct_broke_20low=to_float(raw.get("pct_broke_20low")),
            universe_size=to_int(raw.get("universe_size")),
        )
        status, quality = status_for_drift(base_status="ok", base_quality="ok", drift=drift)
        return build_source_result(
            self,
            raw,
            snapshot,
            status=status,
            quality=quality,
            as_of=None,
            schema_drift=drift,
            coverage={"universe_size": snapshot.universe_size},
            fingerprint_payload={k: v for k, v in raw.items() if k != "raw_df"},
        )


class VixTermStructureAdapter:
    source_name = "vix_term_structure"
    source_version = "1"
    required = True
    raw_module = "equities.market_technicals.vix_term_structure"
    raw_function = "get_data"

    def __init__(self):
        start = (date.today() - timedelta(days=400)).isoformat()
        self.parameters = {"tail": 252, "signals_count": 20, "start": start}

    def fetch(self) -> dict[str, Any]:
        from equities.market_technicals.vix_term_structure import get_data

        return get_data(**self.parameters)

    def normalize(self, raw: Any):
        if not isinstance(raw, dict):
            return build_source_result(
                self, raw, None, status="error", quality="missing", as_of=None, detail="payload is not a dict"
            )

        expected = {"latest_df", "recent_df", "hits_df"}
        drift = unknown_fields(raw, expected)
        latest = first_row(raw.get("latest_df"))
        if not latest:
            drift.append(schema_issue("warning", "$.latest_df[0]", "latest VIX row", None, "defaulted to neutral"))

        snapshot = VixTermStructureSnapshot(
            date=clean_str(latest.get("Date")),
            vix=to_float(latest.get("VIX")),
            vix3m=to_float(latest.get("VIX3M")),
            ratio=to_float(latest.get("Ratio")),
            signal=str(clean_str(latest.get("Signal")) or "Neutral"),
            used_ticker=clean_str(latest.get("UsedTicker")),
        )
        if not latest:
            return build_source_result(
                self,
                raw,
                snapshot,
                status="partial",
                quality="missing",
                as_of=None,
                schema_drift=drift,
                detail="missing latest VIX term-structure row",
                coverage={"latest_rows": 0},
            )

        status, quality = status_for_drift(base_status="ok", base_quality="ok", drift=drift)
        return build_source_result(
            self,
            raw,
            snapshot,
            status=status,
            quality=quality,
            as_of=snapshot.date,
            schema_drift=drift,
            coverage={"latest_rows": 1},
        )
