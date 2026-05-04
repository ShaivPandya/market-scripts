from __future__ import annotations

from typing import Any

from equities.sector_metrics.payload import sector_metric_rows
from ontology.sources.base import (
    build_source_result,
    clean_str,
    iso_string,
    schema_issue,
    status_for_drift,
    to_float,
    unknown_fields,
)
from ontology.sources.dtos import SectorMetricRow, SectorMetricsSnapshot


class SectorMetricsAdapter:
    source_name = "sector_metrics"
    source_version = "1"
    required = True
    raw_module = "equities.sector_metrics.sector_metrics"
    raw_function = "get_data"
    parameters: dict[str, Any] = {}

    def fetch(self) -> dict[str, Any]:
        from equities.sector_metrics.sector_metrics import get_data

        return get_data()

    def normalize(self, raw: Any):
        if not isinstance(raw, dict):
            return build_source_result(
                self, raw, None, status="error", quality="missing", as_of=None, detail="payload is not a dict"
            )

        expected = {"weights_df", "d_1m", "d_3m", "d_6m", "timestamp"}
        drift = unknown_fields(raw, expected)
        raw_rows = sector_metric_rows(raw.get("weights_df"))
        expected_row_fields = {
            "Sector",
            "index",
            "Weight_Now",
            "Chg_1M_pp",
            "Chg_3M_pp",
            "Chg_6M_pp",
            "RelPerf_3M_pp",
            "RelPerf_12M_pp",
            "Pct_Above_200DMA",
        }
        rows: list[SectorMetricRow] = []
        for idx, row in enumerate(raw_rows):
            if idx == 0:
                drift.extend(unknown_fields(row, expected_row_fields, path="$.weights_df[0]"))
            sector = clean_str(row.get("Sector") or row.get("index"))
            if not sector:
                drift.append(
                    schema_issue("warning", f"$.weights_df[{idx}].Sector", "non-empty string", None, "skipped")
                )
                continue
            rows.append(
                SectorMetricRow(
                    sector=sector,
                    weight_now=to_float(row.get("Weight_Now")),
                    chg_1m_pp=to_float(row.get("Chg_1M_pp")),
                    chg_3m_pp=to_float(row.get("Chg_3M_pp")),
                    chg_6m_pp=to_float(row.get("Chg_6M_pp")),
                    relperf_3m_pp=to_float(row.get("RelPerf_3M_pp")),
                    relperf_12m_pp=to_float(row.get("RelPerf_12M_pp")),
                    pct_above_200dma=to_float(row.get("Pct_Above_200DMA")),
                )
            )

        snapshot = SectorMetricsSnapshot(
            rows=rows,
            timestamp=iso_string(raw.get("timestamp")),
            d_1m=clean_str(raw.get("d_1m")),
            d_3m=clean_str(raw.get("d_3m")),
            d_6m=clean_str(raw.get("d_6m")),
        )
        if not rows:
            return build_source_result(
                self,
                raw,
                snapshot,
                status="partial",
                quality="missing",
                as_of=snapshot.timestamp,
                schema_drift=drift,
                detail="no usable sector metric rows",
                coverage={"rows": 0},
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
            coverage={"rows": len(rows)},
        )
