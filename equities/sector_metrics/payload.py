"""JSON payload helpers for sector metrics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

SECTOR_ORDER: tuple[str, ...] = (
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
)


def sector_metric_rows(value: Any) -> list[dict[str, Any]]:
    """Return sector metric rows with a stable ``Sector`` field.

    Older cached snapshots serialized the DataFrame without its sector index,
    leaving rows with numeric data but no sector label. The computation emits
    rows in ``SECTOR_ORDER``, so we can repair those legacy payloads by ordinal.
    """
    records: list[dict[str, Any]] = []

    if isinstance(value, list):
        records = [{str(k): v for k, v in row.items()} for row in value if isinstance(row, dict)]
    elif hasattr(value, "reset_index") and hasattr(value, "to_dict"):
        try:
            frame = value.reset_index()
            raw_records = frame.to_dict(orient="records")
        except Exception:
            raw_records = []
        if isinstance(raw_records, list):
            records = [{str(k): v for k, v in row.items()} for row in raw_records if isinstance(row, dict)]
    elif hasattr(value, "to_dict"):
        try:
            raw_records = value.to_dict(orient="records")
        except Exception:
            raw_records = []
        if isinstance(raw_records, list):
            records = [{str(k): v for k, v in row.items()} for row in raw_records if isinstance(row, dict)]

    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(records):
        out = dict(row)
        sector = _clean_label(out.get("Sector")) or _clean_label(out.get("index"))
        if not sector and idx < len(SECTOR_ORDER):
            sector = SECTOR_ORDER[idx]
        if sector:
            out["Sector"] = sector
        normalized.append(out)
    return normalized


def normalize_sector_metrics_payload(payload: Any) -> dict[str, Any]:
    """Normalize a sector metrics response for API/snapshot JSON."""
    if not isinstance(payload, dict):
        return {}
    out = deepcopy(payload)
    rows = sector_metric_rows(out.get("weights_df"))
    if rows:
        out["weights_df"] = rows
    return out


def _clean_label(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text or text.isdigit():
        return None
    return text
