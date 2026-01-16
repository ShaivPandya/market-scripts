# Realized volatility (20d/60d) for your tickers using Yahoo Finance via yfinance.
# Computes close-to-close log-return vol (daily and annualized) and the "defense" vol
# used for sizing: sigma = max(sigma_20, sigma_60).
#
# USAGE:
#   1. Install dependencies:
#      pip install yfinance numpy pandas
#
#   2. Run with tickers from a file (one ticker per line):
#      python3 realized_volatility.py --tickers-file tickers.txt
#
# OUTPUT:
#   - Prints realized volatility metrics to console
#   - Saves results to realized_vols_latest.csv

import argparse
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

# pip install yfinance
import yfinance as yf

TRADING_DAYS = 252  # standard for equities/ETFs; use consistently across the book

# --- 1) Define your instruments ---
def load_tickers_from_file(filepath: str) -> list[str]:
    """Load tickers from a text file, one per line, ignoring empty lines."""
    tickers = []
    with open(filepath, 'r') as f:
        for line in f:
            ticker = line.strip()
            if ticker:
                tickers.append(ticker)
    return tickers

def get_tickers(ticker_file: str) -> list[str]:
    """Load tickers from the provided file."""
    return load_tickers_from_file(ticker_file)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Calculate realized volatility for tickers')
parser.add_argument('--tickers-file', type=str, required=True, help='Path to file containing tickers (one per line)')
args = parser.parse_args()

tickers = get_tickers(args.tickers_file)

# --- 2) Download adjusted close prices ---
end = date.today()
start = end - timedelta(days=365)  # enough buffer to compute rolling 60d vols

px = yf.download(
    tickers=tickers,
    start=start.isoformat(),
    end=end.isoformat(),
    auto_adjust=True,     # produces adjusted prices; better for equities/ETFs
    progress=False
)

# yfinance returns either a single DataFrame (single ticker) or a column MultiIndex
if isinstance(px.columns, pd.MultiIndex):
    # prefer "Close" when auto_adjust=True (it's already adjusted); fall back if needed
    if ("Close" in px.columns.get_level_values(0)):
        prices = px["Close"].copy()
    else:
        # older yfinance sometimes uses different keys; try to recover
        prices = px.xs(px.columns.levels[0][0], axis=1, level=0).copy()
else:
    prices = px.copy()
    prices.columns = [tickers[0]]

# --- 3) Build synthetic JPYUSD from USDJPY ---
# JPYUSD = 1 / USDJPY. Log-return(JPYUSD) = -Log-return(USDJPY).
if "USDJPY=X" in prices.columns:
    prices["JPYUSD_SYNTH"] = 1.0 / prices["USDJPY=X"]

# --- 4) Compute log returns ---
rets = np.log(prices / prices.shift(1))

# For clarity, keep both USDJPY and JPYUSD series; you can drop USDJPY if you want.
# If you want "long JPYUSD", use JPYUSD_SYNTH in sizing, not USDJPY=X.
use_cols = [c for c in rets.columns if c != "USDJPY=X"]
rets_use = rets[use_cols].dropna(how="all")

# --- 5) Rolling realized vol (daily and annualized) ---
def realized_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling realized vol of log returns."""
    return returns.rolling(window).std()

vol20_d = realized_vol(rets_use, 20)
vol60_d = realized_vol(rets_use, 60)

# "Defense" vol input for sizing: conservative of 20d and 60d
vol_def_d = pd.concat([vol20_d, vol60_d], axis=1, keys=["v20", "v60"])
vol_def_d = vol_def_d.T.groupby(level=1).max().T  # max across v20/v60 for each ticker

# Annualize
vol20_a = vol20_d * math.sqrt(TRADING_DAYS)
vol60_a = vol60_d * math.sqrt(TRADING_DAYS)
voldef_a = vol_def_d * math.sqrt(TRADING_DAYS)

# --- 6) Fetch 5Y monthly beta from Yahoo Finance ---
def get_beta(ticker: str) -> float:
    """Fetch 5Y monthly beta from Yahoo Finance ticker info."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("beta", np.nan)
    except Exception:
        return np.nan

beta_dict = {ticker: get_beta(ticker) for ticker in rets_use.columns}

# --- 7) Output the latest available vols ---
# Use last valid value for each ticker (handles exchange holidays)
latest = pd.DataFrame({
    "vol20_daily": vol20_d.apply(lambda col: col.dropna().iloc[-1] if col.notna().any() else np.nan),
    "vol60_daily": vol60_d.apply(lambda col: col.dropna().iloc[-1] if col.notna().any() else np.nan),
    "vol_def_daily(max20,60)": vol_def_d.apply(lambda col: col.dropna().iloc[-1] if col.notna().any() else np.nan),
    "vol20_annual": vol20_a.apply(lambda col: col.dropna().iloc[-1] if col.notna().any() else np.nan),
    "vol60_annual": vol60_a.apply(lambda col: col.dropna().iloc[-1] if col.notna().any() else np.nan),
    "vol_def_annual(max20,60)": voldef_a.apply(lambda col: col.dropna().iloc[-1] if col.notna().any() else np.nan),
    "beta_5Y": pd.Series(beta_dict),
}).sort_index()

# Optional: drop tickers with no data
latest = latest.dropna(how="all")

pd.set_option("display.float_format", lambda x: f"{x:0.4f}")
print(latest)

# # --- 8) (Optional) Save to CSV ---
# latest.to_csv("realized_vols_latest.csv")
# print("\nSaved: realized_vols_latest.csv")
