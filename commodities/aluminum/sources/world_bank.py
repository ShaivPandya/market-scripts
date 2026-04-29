"""World Bank Pink Sheet aluminum price source."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from commodities.aluminum.config import WORLD_BANK_MONTHLY_XLS_URL, WORLD_BANK_RAW_XLS
from utils.retry import requests_get

log = logging.getLogger(__name__)

_SOURCE = "world_bank_pink_sheet"


def download_world_bank_pink_sheet(
    *,
    refresh: bool = False,
    url: str = WORLD_BANK_MONTHLY_XLS_URL,
    raw_path: Path = WORLD_BANK_RAW_XLS,
) -> Path:
    """Download/cache the World Bank monthly commodity XLS workbook."""
    if raw_path.exists() and not refresh:
        return raw_path

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests_get(url, timeout=60)
    response.raise_for_status()
    raw_path.write_bytes(response.content)
    log.info("Cached World Bank Pink Sheet XLS to %s", raw_path)
    return raw_path


def _month_end(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, datetime | pd.Timestamp):
        ts = pd.Timestamp(value)
        return ts + pd.offsets.MonthEnd(0)

    if isinstance(value, int | float) and not isinstance(value, bool):
        # Excel serial dates are usually above 20,000 for modern monthly data.
        if 20000 <= float(value) <= 90000:
            ts = pd.to_datetime(value, unit="D", origin="1899-12-30", errors="coerce")
            if not pd.isna(ts):
                return pd.Timestamp(ts) + pd.offsets.MonthEnd(0)
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None

    compact = text.replace(" ", "")
    match = re.match(r"^(\d{4})M(\d{1,2})$", compact, flags=re.IGNORECASE)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)

    match = re.match(r"^(\d{4})[-_/](\d{1,2})$", compact)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed) + pd.offsets.MonthEnd(0)


def _numeric(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        cleaned = str(value).replace(",", "").strip()
        out = float(cleaned)
    except (TypeError, ValueError):
        return None
    if pd.isna(out) or out <= 0:
        return None
    return out


def _find_header_row(raw: pd.DataFrame, aluminum_row_idx: int) -> tuple[int, dict[int, pd.Timestamp]] | None:
    best: tuple[int, dict[int, pd.Timestamp]] | None = None
    best_count = 0
    first_row = max(0, aluminum_row_idx - 30)

    for row_idx in range(first_row, aluminum_row_idx):
        date_by_col: dict[int, pd.Timestamp] = {}
        for col_idx, value in raw.iloc[row_idx].items():
            date = _month_end(value)
            if date is not None:
                date_by_col[int(col_idx)] = date
        if len(date_by_col) > best_count:
            best = (row_idx, date_by_col)
            best_count = len(date_by_col)

    if best is None or best_count < 3:
        return None
    return best


def _parse_sheet_with_commodity_columns(raw: pd.DataFrame) -> pd.DataFrame | None:
    """Parse Pink Sheet layout where commodities are columns and dates are rows."""
    for header_idx in range(len(raw)):
        header = raw.iloc[header_idx]
        aluminum_cols = [int(col_idx) for col_idx, value in header.items() if "alum" in str(value).strip().lower()]
        if not aluminum_cols:
            continue

        for aluminum_col in aluminum_cols:
            records: list[dict[str, Any]] = []
            for row_idx in range(header_idx + 1, len(raw)):
                date = _month_end(raw.iloc[row_idx, 0])
                if date is None:
                    continue
                value = _numeric(raw.iloc[row_idx, aluminum_col])
                if value is None:
                    continue
                records.append(
                    {
                        "date": date,
                        "aluminum_price_usd_tonne": value,
                        "source": _SOURCE,
                    }
                )
            if len(records) >= 3:
                return pd.DataFrame(records)
    return None


def parse_world_bank_pink_sheet(path: str | Path) -> pd.DataFrame:
    """Parse monthly aluminum prices from a World Bank Pink Sheet workbook.

    The Pink Sheet layout has changed over time. This parser searches all sheets
    for an aluminum row and pairs it with the nearest preceding row containing
    monthly date headers instead of depending on a fixed sheet name or row offset.
    """
    workbook = pd.ExcelFile(path)
    parsed_frames: list[pd.DataFrame] = []

    for sheet_name in workbook.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
        if raw.empty:
            continue

        by_column = _parse_sheet_with_commodity_columns(raw)
        if by_column is not None:
            parsed_frames.append(by_column)
            continue

        for row_idx in range(len(raw)):
            row = raw.iloc[row_idx]
            row_text = " ".join(str(v).lower() for v in row.dropna().tolist())
            if "alum" not in row_text:
                continue

            header = _find_header_row(raw, row_idx)
            if header is None:
                continue
            _, date_by_col = header

            records: list[dict[str, Any]] = []
            for col_idx, dt in date_by_col.items():
                value = _numeric(row.iloc[col_idx]) if col_idx < len(row) else None
                if value is not None:
                    records.append(
                        {
                            "date": dt,
                            "aluminum_price_usd_tonne": value,
                            "source": _SOURCE,
                        }
                    )

            if len(records) >= 3:
                parsed_frames.append(pd.DataFrame(records))

    if not parsed_frames:
        raise RuntimeError(f"Could not parse aluminum prices from World Bank workbook: {path}")

    out = pd.concat(parsed_frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce") + pd.offsets.MonthEnd(0)
    out["aluminum_price_usd_tonne"] = pd.to_numeric(out["aluminum_price_usd_tonne"], errors="coerce")
    out = out.dropna(subset=["date", "aluminum_price_usd_tonne"])
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"Parsed World Bank workbook but found no valid aluminum price rows: {path}")
    return out[["date", "aluminum_price_usd_tonne", "source"]]


def fetch_world_bank_aluminum_prices(*, refresh: bool = False) -> pd.DataFrame:
    """Fetch/cache and normalize World Bank aluminum monthly prices."""
    path = download_world_bank_pink_sheet(refresh=refresh)
    return parse_world_bank_pink_sheet(path)
