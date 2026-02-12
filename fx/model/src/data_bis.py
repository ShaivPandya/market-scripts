from pathlib import Path
import pandas as pd

class BisError(RuntimeError):
    pass

def fetch_bis_ws_eer_m(series_key: str, cache_dir: Path, refresh: bool = False) -> pd.Series:
    # Fetch BIS WS_EER_M monthly effective exchange rate series using pandaSDMX.
    # series_key example: 'M.R.B.CA' (monthly, REER, broad basket, Canada).
    cache_path = cache_dir / f"bis_WS_EER_M_{series_key.replace('.','_')}.csv"
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, parse_dates=["date"])
        s = pd.to_numeric(df["value"], errors="coerce")
        return pd.Series(s.values, index=df["date"], name=f"BIS_WS_EER_M_{series_key}").sort_index()

    try:
        from pandasdmx import Request
    except Exception as e:
        raise BisError("pandaSDMX is not installed or could not be imported. Install from requirements.txt") from e

    req = Request("BIS")

    try:
        msg = req.data("WS_EER_M", key=series_key, params={"detail": "dataonly"})
    except Exception as e:
        raise BisError(f"BIS SDMX query failed for key={series_key}: {e}") from e

    try:
        data = msg.to_pandas()
    except Exception as e:
        raise BisError(f"Failed to convert BIS response to pandas for key={series_key}: {e}") from e

    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]

    s = data.dropna()
    if isinstance(s.index, pd.PeriodIndex):
        idx = s.index.to_timestamp(how="end")
    else:
        idx = pd.to_datetime(s.index)
        idx = idx.to_period("M").to_timestamp(how="end")

    out = pd.DataFrame({"date": idx, "value": pd.to_numeric(s.values, errors="coerce")}).dropna()
    out.to_csv(cache_path, index=False)
    return pd.Series(out["value"].values, index=out["date"], name=f"BIS_WS_EER_M_{series_key}").sort_index()
