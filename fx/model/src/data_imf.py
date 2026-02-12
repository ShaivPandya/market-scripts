from pathlib import Path
import requests
import pandas as pd

IMF_BASE = "https://www.imf.org/external/datamapper/api/v1"

class ImfError(RuntimeError):
    pass

def fetch_imf_datamapper_indicator(indicator: str, iso3: str, cache_dir: Path, refresh: bool = False) -> pd.Series:
    # Fetch an IMF DataMapper indicator for an ISO3 country code. Cache extracted annual series to CSV.
    cache_path = cache_dir / f"imf_{indicator}_{iso3}.csv"
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, parse_dates=["date"])
        s = pd.to_numeric(df["value"], errors="coerce")
        return pd.Series(s.values, index=df["date"], name=f"{indicator}_{iso3}").sort_index()

    url = f"{IMF_BASE}/{indicator}/{iso3}"
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise ImfError(f"IMF DataMapper request failed ({r.status_code}): {r.text[:2000]}")

    j = r.json()
    values = (j.get("values", {})
                .get(indicator, {})
                .get(iso3, {}))

    if not values:
        raise ImfError(f"No values returned for {indicator}/{iso3}")

    s = pd.Series(values, dtype="float64")
    idx = pd.to_datetime(s.index.astype(str) + "-12-31")  # annual -> year-end date index

    out = pd.DataFrame({"date": idx, "value": s.values})
    out.to_csv(cache_path, index=False)

    return pd.Series(s.values, index=idx, name=f"{indicator}_{iso3}").sort_index()
