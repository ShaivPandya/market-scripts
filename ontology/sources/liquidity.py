from __future__ import annotations

from typing import Any

from ontology.sources.base import (
    as_dict,
    as_rows,
    build_source_result,
    clean_str,
    schema_issue,
    status_for_drift,
    to_float,
    unknown_fields,
)
from ontology.sources.dtos import LiquiditySnapshot


class LiquidityAdapter:
    source_name = "liquidity"
    source_version = "1"
    required = True
    raw_module = "macro.liquidity.liquidity"
    raw_function = "get_snapshot"
    parameters: dict[str, Any] = {}

    def fetch(self) -> dict[str, Any]:
        from macro.liquidity.liquidity import get_snapshot

        return get_snapshot()

    def normalize(self, raw: Any):
        if not isinstance(raw, dict):
            return build_source_result(
                self, raw, None, status="error", quality="missing", as_of=None, detail="payload is not a dict"
            )

        expected = {
            "composite_score",
            "regime",
            "regime_color",
            "latest_date",
            "regional_scores",
            "components",
            "changes",
            "df_weekly",
            "composite_series",
        }
        drift = unknown_fields(raw, expected)
        if "regime" not in raw:
            drift.append(schema_issue("warning", "$.regime", "liquidity regime string", None, "defaulted to normal"))

        snapshot = LiquiditySnapshot(
            composite_score=to_float(raw.get("composite_score")),
            regime=str(clean_str(raw.get("regime")) or "normal").lower(),
            latest_date=clean_str(raw.get("latest_date")),
            regional_scores=as_dict(raw.get("regional_scores")),
            components=as_rows(raw.get("components")),
            changes=as_dict(raw.get("changes")),
        )
        status, quality = status_for_drift(base_status="ok", base_quality="ok", drift=drift)
        fingerprint_payload = {k: v for k, v in raw.items() if k not in {"df_weekly", "composite_series"}}
        return build_source_result(
            self,
            raw,
            snapshot,
            status=status,
            quality=quality,
            as_of=snapshot.latest_date,
            schema_drift=drift,
            coverage={"components": len(snapshot.components), "regions": len(snapshot.regional_scores)},
            fingerprint_payload=fingerprint_payload,
        )
