import pandas as pd
import requests
from pathlib import Path


class StatCanError(RuntimeError):
    pass


def _statcan_wds_post(method: str, payload: dict, timeout: int = 20) -> dict:
    """Minimal Statistics Canada Web Data Service (WDS) client."""
    url = f"https://www150.statcan.gc.ca/t1/wds/rest/{method}"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, list):
        if not data:
            raise StatCanError("Statistics Canada WDS returned an empty list response")
        if not all(isinstance(item, dict) for item in data):
            raise StatCanError("Statistics Canada WDS returned an unexpected list response")
        for item in data:
            status = item.get("status")
            if status and status != "SUCCESS":
                message = item.get("message") or item.get("object") or item
                raise StatCanError(f"Statistics Canada WDS error: {message}")
        if len(data) == 1:
            data = data[0]
        else:
            data = {"status": "SUCCESS", "object": [item.get("object") for item in data]}

    status = data.get("status")
    if status and status != "SUCCESS":
        message = data.get("message") or data.get("object") or data
        raise StatCanError(f"Statistics Canada WDS error: {message}")

    return data


def _fetch_statcan_vector_latest_n(*, vector_id: int, latest_n: int, timeout: int = 30) -> pd.Series:
    """Fetch the latest N observations for a StatCan vector. Returns pd.Series with DatetimeIndex."""
    vector_id_int = int(vector_id)
    payload = [{"vectorId": vector_id_int, "latestN": int(latest_n)}]
    data = _statcan_wds_post("getDataFromVectorsAndLatestNPeriods", payload=payload, timeout=timeout)

    obj = data.get("object")
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list) or not obj:
        raise StatCanError("Statistics Canada WDS returned no data object")

    def _matches_vector(b: dict) -> bool:
        raw = b.get("vectorId") or b.get("vector_id") or b.get("vector") or ""
        return str(raw).lstrip("v") == str(vector_id_int)

    block = obj[0] if len(obj) == 1 else next((b for b in obj if _matches_vector(b)), obj[0])
    points = (
        block.get("vectorDataPoint")
        or block.get("vectorDataPoints")
        or block.get("dataPoints")
        or []
    )
    if not isinstance(points, list) or not points:
        raise StatCanError("Statistics Canada WDS returned no datapoints")

    dates, values = [], []
    for p in points:
        ref = p.get("refPer") or p.get("refper") or p.get("ref_period")
        val = p.get("value")
        if ref is None or val in (None, ""):
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        dt = pd.to_datetime(ref, errors="coerce")
        if pd.isna(dt) and isinstance(ref, str):
            if ref.isdigit() and len(ref) == 6:
                dt = pd.to_datetime(ref + "01", format="%Y%m%d", errors="coerce")
            elif ref.isdigit() and len(ref) == 8:
                dt = pd.to_datetime(ref, format="%Y%m%d", errors="coerce")
        if pd.isna(dt):
            continue
        dates.append(dt)
        values.append(num)

    if not dates:
        raise StatCanError("Statistics Canada WDS datapoints could not be parsed")

    series = pd.Series(values, index=pd.to_datetime(dates), name=f"STATCAN_v{vector_id_int}").sort_index()
    return series[~series.index.duplicated(keep="last")]


def fetch_statcan_cpi(vector_id: int, start: str, cache_dir: Path, refresh: bool = False) -> pd.Series:
    """Fetch Statistics Canada CPI series (price level index) with disk caching.

    Returns a pd.Series with DatetimeIndex, compatible with the FRED series format
    expected by the FX model pipeline (price level, not YoY %).
    """
    cache_path = cache_dir / f"statcan_{vector_id}.csv"
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, parse_dates=["date"])
        s = pd.to_numeric(df["value"], errors="coerce")
        return pd.Series(s.values, index=df["date"], name=f"STATCAN_v{vector_id}").sort_index()

    # Compute how many monthly observations to request (from start to now + buffer)
    n_months = max(700, (pd.Timestamp.now() - pd.Timestamp(start)).days // 28 + 12)
    series = _fetch_statcan_vector_latest_n(vector_id=int(vector_id), latest_n=n_months)
    series = series[series.index >= pd.Timestamp(start)]

    if series.empty:
        raise StatCanError(f"No observations returned for StatCan v{vector_id} from {start}")

    pd.DataFrame({"date": series.index, "value": series.values}).to_csv(cache_path, index=False)
    return series
