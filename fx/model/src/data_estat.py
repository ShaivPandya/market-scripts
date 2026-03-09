"""e-Stat (Statistics Bureau of Japan) CPI fetcher for the FX model pipeline."""

import os
import re
from pathlib import Path

import pandas as pd

from utils.retry import requests_get

ESTAT_CPI_STATS_DATA_ID = "0003427113"
_ESTAT_BASE_URL = "https://api.e-stat.go.jp/rest/3.0/app/json"


class EStatError(RuntimeError):
    pass


def _check_status(resp_json: dict, root_key: str) -> None:
    try:
        result = resp_json[root_key]["RESULT"]
        status = str(result.get("STATUS", ""))
        if status != "0":
            raise EStatError(f"e-Stat API error (status={status}): {result.get('ERROR_MSG', 'unknown error')}")
    except (KeyError, TypeError):
        pass


def _parse_month_from_name(name: str) -> "pd.Timestamp | None":
    """Parse a pd.Timestamp from a human-readable e-Stat time label."""
    if not name:
        return None
    s = str(name).strip()

    # YYYY-MM, YYYY/MM, YYYY.MM
    m = re.search(r"(?<!\d)(\d{4})\s*[-/.]\s*(\d{1,2})(?!\d)", s)
    if not m:
        # Japanese-style: e.g. "2024年1月" or "2024.1月"
        m = re.search(r"(?<!\d)(\d{4}).{0,8}?(\d{1,2})\s*月", s)
    if not m:
        # YYYYMNN, e.g. "2024M01"
        m = re.search(r"(?<!\d)(\d{4})\s*M\s*(\d{1,2})(?!\d)", s, flags=re.IGNORECASE)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return pd.Timestamp(year=year, month=month, day=1)
    return None


def _parse_month_from_code(code: str) -> "pd.Timestamp | None":
    """Parse a pd.Timestamp from an e-Stat time code (e.g. '202401', '2024000101')."""
    if not code:
        return None
    s = str(code).strip()
    digits = re.sub(r"\D", "", s)
    if len(digits) < 6 or not digits[:4].isdigit():
        return None

    year = int(digits[:4])
    month_candidates: list[int] = []

    # Standard YYYYMM
    mm = digits[4:6]
    if mm.isdigit():
        m = int(mm)
        if 1 <= m <= 12:
            month_candidates.append(m)

    # Some e-Stat codes: YYYY00MM.. (e.g. 2024000101)
    if len(digits) >= 8 and digits[4:6] == "00":
        mm2 = digits[6:8]
        if mm2.isdigit():
            m = int(mm2)
            if 1 <= m <= 12:
                month_candidates.append(m)

    if not month_candidates:
        return None
    return pd.Timestamp(year=year, month=month_candidates[0], day=1)


def fetch_estat_cpi(
    stats_data_id: str = ESTAT_CPI_STATS_DATA_ID,
    *,
    start: str,
    cache_dir: Path,
    refresh: bool = False,
    timeout: int = 30,
) -> pd.Series:
    """Fetch Japan CPI index (price level) from e-Stat with disk caching.

    Returns a pd.Series with DatetimeIndex of monthly CPI index values,
    compatible with the price-level format expected by the FX model pipeline.
    The series covers from `start` onward.

    Requires the ESTAT_APP_ID environment variable.
    """
    app_id = os.environ.get("ESTAT_APP_ID", "")
    if not app_id:
        raise EStatError("Missing ESTAT_APP_ID environment variable")

    cache_path = cache_dir / f"estat_{stats_data_id}.csv"
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, parse_dates=["date"])
        s = pd.to_numeric(df["value"], errors="coerce")
        series = pd.Series(s.values, index=df["date"], name=f"ESTAT_{stats_data_id}").sort_index()
        return series[series.index >= pd.Timestamp(start)]

    # ── Step 1: fetch metadata to resolve classification codes ────────────────
    meta_resp = requests_get(
        f"{_ESTAT_BASE_URL}/getStatsData",
        params={
            "appId": app_id,
            "statsDataId": stats_data_id,
            "metaGetFlg": "Y",
            "cntGetFlg": "N",
            "limit": 1,
        },
        timeout=timeout,
    )
    meta_resp.raise_for_status()
    meta = meta_resp.json()
    _check_status(meta, "GET_STATS_DATA")

    try:
        class_objs = meta["GET_STATS_DATA"]["STATISTICAL_DATA"]["CLASS_INF"]["CLASS_OBJ"]
    except (KeyError, TypeError) as e:
        raise EStatError(f"e-Stat CPI: unexpected meta structure: {e!r}") from e

    if isinstance(class_objs, dict):
        class_objs = [class_objs]

    def _dim_items(dim_id: str) -> list:
        for obj in class_objs:
            if obj.get("@id") != dim_id:
                continue
            items = obj.get("CLASS", [])
            if isinstance(items, dict):
                items = [items]
            return [i for i in items if isinstance(i, dict)]
        return []

    def _find_code(dim_id: str, keywords: list) -> "str | None":
        for item in _dim_items(dim_id):
            if any(kw in item.get("@name", "") for kw in keywords):
                return item.get("@code")
        return None

    tab_code = _find_code("tab", ["指数"])  # index level (not MoM/YoY %)
    cat01_code = _find_code("cat01", ["総合"])  # all items
    area_code = _find_code("area", ["全国"])  # all Japan

    if not tab_code or not cat01_code or not area_code:
        raise EStatError(
            f"e-Stat CPI: could not resolve classification codes (tab={tab_code}, cat01={cat01_code}, area={area_code})"
        )

    # Build time code -> label map for date parsing
    time_items = _dim_items("time")
    time_code_to_name = {
        str(item.get("@code")): str(item.get("@name", "")) for item in time_items if item.get("@code") is not None
    }

    def _parse_estat_month(time_code: str) -> "pd.Timestamp | None":
        name = time_code_to_name.get(str(time_code), "")
        dt = _parse_month_from_name(name)
        if dt is not None:
            return dt
        return _parse_month_from_code(time_code)

    # Find the earliest time code >= start so we don't over-fetch
    observation_start = pd.Timestamp(start).replace(day=1)
    cd_time_from = None
    parsed_time_codes: list[tuple] = []
    for item in time_items:
        code = str(item.get("@code", "")).strip()
        if not code:
            continue
        dt = _parse_estat_month(code)
        if dt is None:
            continue
        parsed_time_codes.append((dt, code))

    if parsed_time_codes:
        parsed_time_codes.sort(key=lambda x: x[0])
        for dt, code in parsed_time_codes:
            if dt >= observation_start:
                cd_time_from = code
                break

    # ── Step 2: fetch filtered time series ───────────────────────────────────
    params: dict = {
        "appId": app_id,
        "statsDataId": stats_data_id,
        "cdTab": tab_code,
        "cdCat01": cat01_code,
        "cdArea": area_code,
        "metaGetFlg": "N",
        "cntGetFlg": "N",
        "limit": 1000,
    }
    if cd_time_from:
        params["cdTimeFrom"] = cd_time_from

    data_resp = requests_get(f"{_ESTAT_BASE_URL}/getStatsData", params=params, timeout=timeout)
    data_resp.raise_for_status()
    data = data_resp.json()
    _check_status(data, "GET_STATS_DATA")

    try:
        values_list = data["GET_STATS_DATA"]["STATISTICAL_DATA"]["DATA_INF"]["VALUE"]
    except (KeyError, TypeError) as e:
        raise EStatError(f"e-Stat CPI: unexpected data structure: {e!r}") from e

    if isinstance(values_list, dict):
        values_list = [values_list]

    if not values_list:
        raise EStatError("e-Stat CPI: no values returned")

    dates, values = [], []
    for item in values_list:
        time_str = item.get("@time", "")
        val_str = str(item.get("$", "")).strip()
        if not time_str or not val_str or val_str == "-":
            continue
        try:
            val = float(val_str)
        except ValueError:
            continue
        dt = _parse_estat_month(time_str)
        if dt is None or dt < observation_start:
            continue
        dates.append(dt)
        values.append(val)

    if not dates:
        raise EStatError("e-Stat CPI: no data points could be parsed")

    series = pd.Series(values, index=pd.to_datetime(dates), name=f"ESTAT_{stats_data_id}").sort_index()
    series = series[~series.index.duplicated(keep="last")]

    pd.DataFrame({"date": series.index, "value": series.values}).to_csv(cache_path, index=False)
    return series[series.index >= pd.Timestamp(start)]
