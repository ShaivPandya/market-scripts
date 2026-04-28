"""Optional EIA electricity proxy source for aluminum research."""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from commodities.aluminum.config import EIA_API_BASE_URL, EIA_POWER_PROXY_KEY, EIA_RETAIL_SALES_ROUTE
from load_env import load_env
from utils.retry import requests_get

log = logging.getLogger(__name__)

_EMPTY_COLUMNS = ["date", "eia_series_id_or_route", "value", "unit", "source"]


def empty_eia_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_COLUMNS)


class EIAClient:
    """Small EIA API v2 client."""

    def __init__(self, api_key: str, base_url: str = EIA_API_BASE_URL) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def get(self, route: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{route.strip('/')}/data/"
        query = dict(params or {})
        query["api_key"] = self.api_key
        response = requests_get(url, params=query, timeout=45)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("EIA response was not a JSON object")
        return payload


def normalize_eia_power_proxy_response(
    payload: dict[str, Any],
    *,
    route_key: str = EIA_POWER_PROXY_KEY,
    value_field: str = "price",
) -> pd.DataFrame:
    data = payload.get("response", {}).get("data", [])
    if not isinstance(data, list) or not data:
        return empty_eia_frame()

    rows: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        period = item.get("period")
        value = pd.to_numeric(item.get(value_field), errors="coerce")
        if pd.isna(value):
            continue
        date = pd.to_datetime(period, errors="coerce")
        if pd.isna(date):
            continue
        rows.append(
            {
                "date": pd.Timestamp(date) + pd.offsets.MonthEnd(0),
                "eia_series_id_or_route": route_key,
                "value": float(value),
                "unit": item.get(f"{value_field}-units") or item.get("unit") or "unknown",
                "source": "eia_api_v2",
            }
        )

    if not rows:
        return empty_eia_frame()
    out = pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return out[_EMPTY_COLUMNS]


def fetch_eia_power_proxy(
    *,
    api_key: str | None = None,
    route: str | None = None,
    stateid: str | None = None,
    sectorid: str | None = None,
    data_field: str | None = None,
) -> pd.DataFrame:
    """Fetch a configurable EIA electricity price proxy when EIA_API_KEY exists.

    The default is U.S. industrial retail electricity price. It is an optional
    power-cost proxy, not a direct aluminum smelter cost model.
    """
    load_env()
    resolved_key = api_key or os.environ.get("EIA_API_KEY")
    if not resolved_key:
        log.warning("EIA_API_KEY is not set; skipping optional EIA power proxy")
        return empty_eia_frame()

    resolved_route = route or os.environ.get("EIA_ALUMINUM_POWER_ROUTE") or EIA_RETAIL_SALES_ROUTE
    resolved_stateid = stateid or os.environ.get("EIA_ALUMINUM_POWER_STATEID") or "US"
    resolved_sectorid = sectorid or os.environ.get("EIA_ALUMINUM_POWER_SECTORID") or "IND"
    resolved_data_field = data_field or os.environ.get("EIA_ALUMINUM_POWER_DATA_FIELD") or "price"
    route_key = f"{resolved_route}:{resolved_stateid}:{resolved_sectorid}:{resolved_data_field}"

    params = {
        "frequency": "monthly",
        "data[0]": resolved_data_field,
        "facets[stateid][]": resolved_stateid,
        "facets[sectorid][]": resolved_sectorid,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000,
    }

    try:
        payload = EIAClient(resolved_key).get(resolved_route, params=params)
        return normalize_eia_power_proxy_response(payload, route_key=route_key, value_field=resolved_data_field)
    except Exception as exc:
        log.warning("EIA power proxy fetch failed; continuing without it: %s", exc)
        return empty_eia_frame()
