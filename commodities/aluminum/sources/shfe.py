"""Optional SHFE public HTML source for aluminum inventory/futures data."""

from __future__ import annotations

import logging
import re
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from commodities.aluminum.config import RAW_DIR, SHFE_ALUMINUM_FUTURES_URL, SHFE_WEEKLY_DATA_URL
from utils.retry import requests_get

log = logging.getLogger(__name__)

_EMPTY_COLUMNS = ["date", "contract_or_product", "inventory_tonnes", "source"]


def empty_shfe_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_COLUMNS)


def fetch_shfe_html(url: str, *, cache_name: str, refresh: bool = False, raw_dir: Path = RAW_DIR) -> str:
    """Fetch a SHFE page and save a raw HTML snapshot."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / cache_name
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")

    response = requests_get(
        url,
        headers={"User-Agent": "Talisman aluminum research backtest"},
        timeout=45,
    )
    response.raise_for_status()
    text: str = response.text
    path.write_text(text, encoding="utf-8")
    return text


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in out.columns]
    return out


def _first_matching_column(columns: list[str], patterns: tuple[str, ...]) -> str | None:
    for col in columns:
        if any(pattern in col for pattern in patterns):
            return col
    return None


def _html_date(html: str) -> pd.Timestamp | None:
    match = re.search(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})", html)
    if not match:
        return None
    return pd.Timestamp(year=int(match.group(1)), month=int(match.group(2)), day=int(match.group(3)))


def _looks_like_aluminum(value: Any) -> bool:
    text = str(value).strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", text)
    return compact in {"al", "alu", "aluminum", "aluminium"} or "aluminum" in compact or "aluminium" in compact


def parse_shfe_aluminum_inventory_html(html: str) -> pd.DataFrame:
    """Parse available aluminum inventory rows from SHFE HTML.

    Parsing is intentionally permissive because public SHFE table structures can
    change. Fetching is separate so tests can exercise this parser with snippets.
    """
    try:
        tables = pd.read_html(StringIO(html))
    except Exception as exc:
        log.warning("SHFE HTML table parsing failed: %s", exc)
        return empty_shfe_frame()

    fallback_date = _html_date(html)
    rows: list[dict[str, Any]] = []

    for table in tables:
        if table.empty:
            continue
        df = _clean_columns(table)
        columns = list(df.columns)
        date_col = _first_matching_column(columns, ("date", "day"))
        product_col = _first_matching_column(columns, ("product", "commodity", "variety", "contract", "symbol"))
        inventory_col = _first_matching_column(columns, ("inventory", "stock", "warehouse", "warrant"))

        if inventory_col is None:
            continue

        for _, row in df.iterrows():
            product = row.get(product_col) if product_col is not None else "Aluminum"
            if product_col is not None and not _looks_like_aluminum(product):
                continue

            date_value = row.get(date_col) if date_col is not None else fallback_date
            dt = pd.to_datetime(date_value, errors="coerce")
            if pd.isna(dt):
                continue
            inventory = pd.to_numeric(str(row.get(inventory_col)).replace(",", ""), errors="coerce")
            if pd.isna(inventory):
                continue
            rows.append(
                {
                    "date": pd.Timestamp(dt),
                    "contract_or_product": str(product).strip() or "Aluminum",
                    "inventory_tonnes": float(inventory),
                    "source": "shfe_public_html",
                }
            )

    if not rows:
        return empty_shfe_frame()
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "inventory_tonnes"]).sort_values("date").reset_index(drop=True)
    return out[_EMPTY_COLUMNS]


def fetch_shfe_aluminum_inventory(*, refresh: bool = False) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for url, cache_name in (
        (SHFE_WEEKLY_DATA_URL, "shfe_weekly_data.html"),
        (SHFE_ALUMINUM_FUTURES_URL, "shfe_aluminum_futures.html"),
    ):
        try:
            html = fetch_shfe_html(url, cache_name=cache_name, refresh=refresh)
            parsed = parse_shfe_aluminum_inventory_html(html)
            if not parsed.empty:
                frames.append(parsed)
        except Exception as exc:
            log.warning("SHFE fetch failed for %s; continuing without that page: %s", url, exc)

    if not frames:
        return empty_shfe_frame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("date").drop_duplicates(["date", "contract_or_product"], keep="last").reset_index(drop=True)
