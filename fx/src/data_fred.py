import os
from pathlib import Path
import requests
import pandas as pd

FRED_ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"

class FredError(RuntimeError):
    pass

def _fred_api_key():
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise FredError("Missing FRED_API_KEY. Set it in your environment or in a .env file.")
    return key

def fetch_fred_series(series_id: str, start: str, cache_dir: Path, refresh: bool = False) -> pd.Series:
    # Fetch a FRED series as a pandas Series with DatetimeIndex. Cache raw observations to CSV.
    cache_path = cache_dir / f"fred_{series_id}.csv"
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, parse_dates=["date"])
        s = pd.to_numeric(df["value"], errors="coerce")
        return pd.Series(s.values, index=df["date"], name=series_id).sort_index()

    params = {
        "series_id": series_id,
        "api_key": _fred_api_key(),
        "file_type": "json",
        "observation_start": start,
    }
    r = requests.get(FRED_ENDPOINT, params=params, timeout=60)
    if r.status_code != 200:
        raise FredError(f"FRED request failed ({r.status_code}): {r.text[:2000]}")

    j = r.json()
    obs = pd.DataFrame(j.get("observations", []))
    if obs.empty:
        raise FredError(f"No observations returned for {series_id}")

    obs = obs[["date", "value"]].copy()
    obs.to_csv(cache_path, index=False)

    obs["date"] = pd.to_datetime(obs["date"])
    s = pd.to_numeric(obs["value"].replace(".", pd.NA), errors="coerce")
    return pd.Series(s.values, index=obs["date"], name=series_id).sort_index()
