"""Feature engineering for FX models.

Uses generic column names:
- spot: exchange rate (quote per base)
- cpi_base: CPI for base currency
- cpi_quote: CPI for quote currency
- r_base: short rate for base currency
- r_quote: short rate for quote currency

RER = log(spot) + log(cpi_base) - log(cpi_quote)
Carry = (r_quote - r_base) / 100
"""
import numpy as np
import pandas as pd


def _as_price_index(s: pd.Series, *, base: float = 100.0) -> pd.Series:
    """Ensure a CPI-like series is a strictly-positive price level index.

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
    """Resample a series to monthly frequency."""
    s = s.dropna()
    if how == "mean":
        return s.resample("ME").mean()
    return s.resample("ME").last()


def build_monthly_panel(series: dict) -> pd.DataFrame:
    """Build a monthly panel from a dict of series."""
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
    """Compute model features from monthly panel.

    Expects columns: spot, cpi_base, cpi_quote, r_base, r_quote, wti, vix
    """
    d = df.copy()

    # Log spot
    d["logS"] = np.log(d["spot"])

    # Bilateral real exchange rate (PPP-style):
    # RER = log(spot) + log(CPI_base) - log(CPI_quote)
    cpi_base_level = _as_price_index(d["cpi_base"])
    cpi_quote_level = _as_price_index(d["cpi_quote"])
    d["cpi_base_level"] = cpi_base_level
    d["cpi_quote_level"] = cpi_quote_level
    d["rer"] = d["logS"] + np.log(cpi_base_level) - np.log(cpi_quote_level)

    # RER z-score (rolling)
    mu = d["rer"].rolling(window_z).mean()
    sd = d["rer"].rolling(window_z).std()
    d["rer_z"] = (d["rer"] - mu) / sd

    # Real oil price and momentum
    d["wti_real"] = np.log(d["wti"]) - np.log(cpi_base_level)
    d["oil_mom12"] = d["wti_real"].diff(12)

    # Carry: (r_quote - r_base) / 100
    # Positive carry = going long base currency earns positive carry
    d["carry"] = (d["r_quote"] - d["r_base"]) / 100.0

    # Spot momentum
    d["mom12"] = d["logS"].diff(12)

    # Realized volatility
    d["ret1"] = d["logS"].diff(1)
    d["rv12"] = d["ret1"].rolling(12).std() * np.sqrt(12)

    # Carry-to-vol ratio
    d["carry_to_vol"] = d["carry"] / d["rv12"]

    return d


def implied_spot_reference_points(df: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    """Compute implied spot levels if RER reverted to rolling quantiles."""
    d = df.copy()
    rer = d["rer"]
    rer_med = rer.rolling(window).median()
    rer_p25 = rer.rolling(window).quantile(0.25)
    rer_p75 = rer.rolling(window).quantile(0.75)

    # Get CPI levels (already computed in compute_features)
    cpi_base_level = d["cpi_base_level"] if "cpi_base_level" in d.columns else _as_price_index(d["cpi_base"])
    cpi_quote_level = d["cpi_quote_level"] if "cpi_quote_level" in d.columns else _as_price_index(d["cpi_quote"])

    # price_diff = log(CPI_base) - log(CPI_quote)
    price_diff = np.log(cpi_base_level) - np.log(cpi_quote_level)

    # Implied log spot: log(S) = RER - price_diff
    s_med = rer_med - price_diff
    s_25 = rer_p25 - price_diff
    s_75 = rer_p75 - price_diff

    out = pd.DataFrame(index=d.index)
    out["spot"] = d["spot"]
    out["spot_implied_median"] = np.exp(s_med)
    out["spot_implied_p25"] = np.exp(s_25)
    out["spot_implied_p75"] = np.exp(s_75)
    return out
