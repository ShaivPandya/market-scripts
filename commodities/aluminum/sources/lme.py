"""Licensed/local LME XML adapter.

This module does not scrape LME websites. It only reads local XML files or an
explicit licensed XML endpoint configured by environment variables.
"""

from __future__ import annotations

import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd

from commodities.aluminum.config import LME_XML_DIR, RAW_DIR
from load_env import load_env
from utils.retry import requests_get

log = logging.getLogger(__name__)

_PRICE_COLUMNS = ["date", "lme_aluminum_cash", "lme_aluminum_3m", "source"]
_STOCK_COLUMNS = ["date", "warehouse_location", "stock_tonnes", "cancelled_tonnes", "source"]


def parse_lme_stocks_excel(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, header=None)
    except Exception as exc:
        log.warning("Failed to read Excel file %s: %s", path, exc)
        return pd.DataFrame(columns=_STOCK_COLUMNS)

    header_row_idx = None
    for i in range(min(20, len(df))):
        row_vals = [str(x).strip() for x in df.iloc[i].values if pd.notna(x)]
        if "BusinessDate" in row_vals and "AH" in row_vals:
            header_row_idx = i
            break

    if header_row_idx is None:
        return pd.DataFrame(columns=_STOCK_COLUMNS)

    df.columns = df.iloc[header_row_idx]
    df = df.iloc[header_row_idx + 1 :].copy()

    if "BusinessDate" not in df.columns or "AH" not in df.columns:
        return pd.DataFrame(columns=_STOCK_COLUMNS)

    df = df[["BusinessDate", "AH"]].dropna(subset=["BusinessDate", "AH"])
    df["BusinessDate"] = pd.to_datetime(df["BusinessDate"], errors="coerce")
    df = df.dropna(subset=["BusinessDate"])

    out = pd.DataFrame(
        {
            "date": df["BusinessDate"],
            "warehouse_location": "Global",
            "stock_tonnes": pd.to_numeric(df["AH"], errors="coerce"),
            "cancelled_tonnes": 0.0,
            "source": "lme_public_excel",
        }
    )
    out = out.dropna(subset=["stock_tonnes"])
    return out[_STOCK_COLUMNS]


def empty_lme_prices_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_PRICE_COLUMNS)


def empty_lme_stocks_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_STOCK_COLUMNS)


def _norm_key(value: str) -> str:
    value = value.split("}", 1)[-1]
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _flatten_element(element: ET.Element) -> dict[str, str]:
    values: dict[str, str] = {}
    for child in element.iter():
        text = (child.text or "").strip()
        if not text:
            continue
        key = _norm_key(child.tag)
        if key:
            values[key] = text
    return values


def _candidate_records(root: ET.Element) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for element in root.iter():
        flat = _flatten_element(element)
        keys = set(flat)
        has_date = bool(keys & {"date", "businessdate", "promptdate", "valuationdate", "pricedate"})
        has_price = bool(keys & {"cash", "cashprice", "cashsettlement", "lmealuminumcash", "threemonth", "3m"})
        has_stock = bool(keys & {"stock", "stocks", "stocktonnes", "onwarrant", "cancelled", "cancelledtonnes"})
        if has_date and (has_price or has_stock):
            records.append(flat)
    return records


def _get_first(record: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _float(value: Any) -> float | None:
    if value is None:
        return None
    out = pd.to_numeric(str(value).replace(",", ""), errors="coerce")
    if pd.isna(out):
        return None
    return float(out)


def _date(record: dict[str, str]) -> pd.Timestamp | None:
    value = _get_first(record, ("date", "businessdate", "promptdate", "valuationdate", "pricedate"))
    if value is None:
        return None
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return None
    return pd.Timestamp(dt)


def _is_aluminum_record(record: dict[str, str]) -> bool:
    metal = _get_first(record, ("metal", "commodity", "product", "symbol", "contract"))
    if metal is None:
        return True
    text = metal.strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", text)
    return compact in {"al", "alu", "aluminum", "aluminium"} or "alum" in compact


def parse_lme_price_xml(xml_text: str) -> pd.DataFrame:
    root = ET.fromstring(xml_text)
    rows: list[dict[str, Any]] = []
    for record in _candidate_records(root):
        if not _is_aluminum_record(record):
            continue
        dt = _date(record)
        cash = _float(_get_first(record, ("lmealuminumcash", "cash", "cashprice", "cashsettlement")))
        three_m = _float(_get_first(record, ("lmealuminum3m", "3m", "threemonth", "threemonthprice")))
        if dt is None or (cash is None and three_m is None):
            continue
        rows.append({"date": dt, "lme_aluminum_cash": cash, "lme_aluminum_3m": three_m, "source": "lme_xml"})

    if not rows:
        return empty_lme_prices_frame()
    out = pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return out[_PRICE_COLUMNS]


def parse_lme_stock_xml(xml_text: str) -> pd.DataFrame:
    root = ET.fromstring(xml_text)
    rows: list[dict[str, Any]] = []
    for record in _candidate_records(root):
        if not _is_aluminum_record(record):
            continue
        dt = _date(record)
        stock = _float(_get_first(record, ("stocktonnes", "stock", "stocks", "onwarrant")))
        cancelled = _float(_get_first(record, ("cancelledtonnes", "cancelled", "cancelledwarrants")))
        location = _get_first(record, ("warehouselocation", "location", "warehouse", "city")) or "unknown"
        if dt is None or stock is None:
            continue
        rows.append(
            {
                "date": dt,
                "warehouse_location": location,
                "stock_tonnes": stock,
                "cancelled_tonnes": 0.0 if cancelled is None else cancelled,
                "source": "lme_xml",
            }
        )

    if not rows:
        return empty_lme_stocks_frame()
    out = (
        pd.DataFrame(rows)
        .sort_values(["date", "warehouse_location"])
        .drop_duplicates(["date", "warehouse_location", "stock_tonnes", "cancelled_tonnes"], keep="last")
        .reset_index(drop=True)
    )
    return out[_STOCK_COLUMNS]


def _download_licensed_xml_if_configured() -> list[Path]:
    load_env()
    url = os.environ.get("LME_XML_URL")
    username = os.environ.get("LME_USERNAME")
    password = os.environ.get("LME_PASSWORD")
    if not (url and username and password):
        return []

    try:
        response = requests_get(url, auth=(username, password), timeout=60)
        response.raise_for_status()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        path = RAW_DIR / "lme_licensed_endpoint.xml"
        path.write_text(response.text, encoding="utf-8")
        return [path]
    except Exception as exc:
        log.warning("Configured LME XML endpoint failed; continuing with local files only: %s", exc)
        return []


def load_lme_xml_data(*, xml_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalize LME XML price and stock files if available."""
    directory = Path(xml_dir) if xml_dir is not None else LME_XML_DIR
    files = sorted(directory.glob("*.xml")) if directory.exists() else []
    files.extend(_download_licensed_xml_if_configured())
    excel_files = sorted(RAW_DIR.glob("Stocks*.xlsx")) if RAW_DIR.exists() else []

    if not files and not excel_files:
        log.warning("No LME XML files, Excel files, or configured licensed endpoint found; skipping optional LME data")
        return empty_lme_prices_frame(), empty_lme_stocks_frame()

    price_frames: list[pd.DataFrame] = []
    stock_frames: list[pd.DataFrame] = []
    for path in excel_files:
        try:
            stocks = parse_lme_stocks_excel(path)
            if not stocks.empty:
                stock_frames.append(stocks)
        except Exception as exc:
            log.warning("Failed to parse LME Excel file %s: %s", path, exc)

    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
            prices = parse_lme_price_xml(text)
            stocks = parse_lme_stock_xml(text)
            if not prices.empty:
                price_frames.append(prices)
            if not stocks.empty:
                stock_frames.append(stocks)
        except Exception as exc:
            log.warning("Failed to parse LME XML file %s: %s", path, exc)

    prices_out = (
        pd.concat(price_frames, ignore_index=True).sort_values("date").drop_duplicates("date", keep="last")
        if price_frames
        else empty_lme_prices_frame()
    )
    stocks_out = pd.concat(stock_frames, ignore_index=True) if stock_frames else empty_lme_stocks_frame()
    return prices_out.reset_index(drop=True), stocks_out.reset_index(drop=True)
