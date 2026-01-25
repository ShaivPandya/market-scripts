import numpy as np
import pandas as pd

def _as_price_index(s: pd.Series, *, base: float = 100.0) -> pd.Series:
    """
    Ensure a CPI-like series is a strictly-positive price level index.

    Some FRED/OECD series are delivered as percent changes (can be 0/negative),
    which makes log transforms invalid. In that case, we build an implied index
    via cumulative compounding.
    """
    s = pd.to_numeric(s, errors="coerce")
    non_na = s.dropna()
    if non_na.empty:
        return s

    looks_like_level = (non_na > 0).all() and (float(non_na.median()) > 10.0)
    if looks_like_level:
        return s

    growth = 1.0 + (s / 100.0)
    growth = growth.where(growth > 0)  # guard against <= -100% observations
    return base * growth.fillna(1.0).cumprod()

def to_monthly(s: pd.Series, how: str = "last") -> pd.Series:
    s = s.dropna()
    if how == "mean":
        return s.resample("M").mean()
    return s.resample("M").last()

def build_monthly_panel(series: dict) -> pd.DataFrame:
    monthly = {}
    for name, s in series.items():
        if name in ("wti", "vix"):
            monthly[name] = to_monthly(s, how="mean")
        else:
            monthly[name] = to_monthly(s, how="last")
    df = pd.concat(monthly.values(), axis=1)
    df.columns = list(monthly.keys())
    return df.sort_index()

def compute_features(df: pd.DataFrame, window_z: int = 120) -> pd.DataFrame:
    d = df.copy()

    d["logS"] = np.log(d["usdcad"])
    # Bilateral real exchange rate (PPP-style): q = s + p_us - p_ca
    cpi_ca_level = _as_price_index(d["cpi_ca"])
    d["cpi_ca_level"] = cpi_ca_level
    d["rer"] = d["logS"] + np.log(d["cpi_us"]) - np.log(cpi_ca_level)

    mu = d["rer"].rolling(window_z).mean()
    sd = d["rer"].rolling(window_z).std()
    d["rer_z"] = (d["rer"] - mu) / sd

    d["wti_real"] = np.log(d["wti"]) - np.log(d["cpi_us"])
    d["oil_mom12"] = d["wti_real"].diff(12)

    d["carry"] = (d["r_ca"] - d["r_us"]) / 100.0
    d["mom12"] = d["logS"].diff(12)

    d["ret1"] = d["logS"].diff(1)
    d["rv12"] = d["ret1"].rolling(12).std() * np.sqrt(12)

    d["carry_to_vol"] = d["carry"] / d["rv12"]

    return d

def implied_spot_reference_points(df: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    d = df.copy()
    rer = d["rer"]
    rer_med = rer.rolling(window).median()
    rer_p25 = rer.rolling(window).quantile(0.25)
    rer_p75 = rer.rolling(window).quantile(0.75)

    cpi_ca_level = d["cpi_ca_level"] if "cpi_ca_level" in d.columns else _as_price_index(d["cpi_ca"])
    price_diff = np.log(d["cpi_us"]) - np.log(cpi_ca_level)

    s_med = rer_med - price_diff
    s_25  = rer_p25 - price_diff
    s_75  = rer_p75 - price_diff

    out = pd.DataFrame(index=d.index)
    out["spot"] = d["usdcad"]
    out["spot_implied_median"] = np.exp(s_med)
    out["spot_implied_p25"] = np.exp(s_25)
    out["spot_implied_p75"] = np.exp(s_75)
    return out
