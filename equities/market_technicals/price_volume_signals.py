"""
Detects:
1) downside volume spike: down day + volume at/near a rolling max.
2) top signals:
   A) New high on low volume
   B) High-volume churn: several high-volume days with little/no upside progress in closes

Applies to S&P 500, Nasdaq Composite, Russell 2000.
NOTE: Index tickers often have unreliable/zero volume. This script tries index first,
then falls back to liquid ETF proxies (SPY, QQQ, IWM) if index volume is missing.

Install:
  pip install yfinance pandas numpy

Run:
  python3 price_volume_signals.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Missing dependency yfinance. Install with: pip install yfinance") from e


# ANSI color codes
class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'


def colorize_retpct(value: float) -> str:
    """Colorize RetPct value: green if positive, red if negative."""
    if pd.isna(value):
        return str(value)
    formatted = f"{value:.2f}"
    if value > 0:
        return f"{Color.GREEN}{formatted}{Color.RESET}"
    elif value < 0:
        return f"{Color.RED}{formatted}{Color.RESET}"
    else:
        return formatted


# ----------------------------
# Configuration (tune as needed)
# ----------------------------

START_DATE = "2000-01-01"

# downside + volume extreme
LOOKBACK_DAYS = 252          # rolling "record" window
MIN_DOWN_PCT = 0.0           # require down day (<= 0). Set e.g. -2.0 for -2% threshold.

# new high on low volume
NEW_HIGH_LOOKBACK = 252      # define "new high" vs rolling max close
LOWVOL_LOOKBACK = 50         # baseline for volume
LOWVOL_MAX_RATIO = 0.80      # volume <= 80% of rolling avg volume => "low volume" high

# high-volume churn / distribution-like action
CHURN_WINDOW = 5             # "several days"
CHURN_MIN_HIVOL_DAYS = 3     # within window, at least this many high-volume days
HIVOL_LOOKBACK = 50
HIVOL_MIN_RATIO = 1.20       # day volume >= 120% of rolling avg => "high volume"
MAX_UPSIDE_PROGRESS_PCT = 1.0  # within the window, total close-to-close progress <= +1%


# ----------------------------
# Data fetching utilities
# ----------------------------

SYMBOLS = {
    "S&P 500":    {"primary": "^GSPC", "fallback": "SPY"},
    "Nasdaq":     {"primary": "^IXIC", "fallback": "QQQ"},
    "Russell 2000":{"primary": "^RUT", "fallback": "IWM"},
}

def download_ohlcv(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if df.empty:
        return df
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)  # standardize just in case
    # Expected: Open, High, Low, Close, Adj Close, Volume
    return df

def ensure_volume(df: pd.DataFrame) -> bool:
    """Return True if volume appears usable (not all 0/NaN)."""
    if df.empty or "Volume" not in df.columns:
        return False
    v = df["Volume"].replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return False
    return not np.all(v.values == 0)


# ----------------------------
# Signal logic
# ----------------------------

def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close", "Volume"])

    # Returns
    df["PrevClose"] = df["Close"].shift(1)
    df["RetPct"] = (df["Close"] / df["PrevClose"] - 1.0) * 100.0

    # Rolling volume stats
    df["VolAvg50"] = df["Volume"].rolling(LOWVOL_LOOKBACK).mean()
    df["VolAvgHV"] = df["Volume"].rolling(HIVOL_LOOKBACK).mean()
    df["VolMax"] = df["Volume"].rolling(LOOKBACK_DAYS).max()

    # Rolling highs
    df["RollHighClose"] = df["Close"].rolling(NEW_HIGH_LOOKBACK).max()

    # 1) down day + rolling max volume (or equal within float tolerance)
    df["DownsideRecordVol"] = (
        (df["RetPct"] <= MIN_DOWN_PCT) &
        (df["Volume"] >= df["VolMax"])
    )

    # 2A) new high on low volume
    # "new high" means close equals rolling max close (including today)
    # require enough lookback data and non-null vol avg
    df["NewHigh_LowVol"] = (
        (df["Close"] >= df["RollHighClose"]) &
        (df["Volume"] <= LOWVOL_MAX_RATIO * df["VolAvg50"])
    )

    # 2B) churn: several high-volume days with little upside progress
    df["HiVolDay"] = df["Volume"] >= HIVOL_MIN_RATIO * df["VolAvgHV"]

    # Count high-volume days in rolling window
    df["HiVolCountWin"] = df["HiVolDay"].rolling(CHURN_WINDOW).sum()

    # Window progress: % change from close N days ago to today
    df["WinProgressPct"] = (df["Close"] / df["Close"].shift(CHURN_WINDOW - 1) - 1.0) * 100.0

    df["HiVol_Churn"] = (
        (df["HiVolCountWin"] >= CHURN_MIN_HIVOL_DAYS) &
        (df["WinProgressPct"] <= MAX_UPSIDE_PROGRESS_PCT)
    )

    return df


def summarize_latest(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{
            "Market": label,
            "Date": None,
            "DownsideRecordVol": None,
            "NewHigh_LowVol": None,
            "HiVol_Churn": None,
            "Close": None,
            "RetPct": None,
            "Volume": None,
        }])

    last = df.iloc[-1]
    return pd.DataFrame([{
        "Market": label,
        "Date": df.index[-1].date().isoformat(),
        "DownsideRecordVol": bool(last.get("DownsideRecordVol", False)),
        "NewHigh_LowVol": bool(last.get("NewHigh_LowVol", False)),
        "HiVol_Churn": bool(last.get("HiVol_Churn", False)),
        "Close": float(last["Close"]),
        "RetPct": float(last.get("RetPct", np.nan)),
        "Volume": float(last["Volume"]),
    }])


def main():
    all_latest = []
    all_hits = []

    for name, tickers in SYMBOLS.items():
        primary = tickers["primary"]
        fallback = tickers["fallback"]

        df = download_ohlcv(primary, START_DATE)
        used = primary

        # Fallback if index volume is unusable
        if not ensure_volume(df):
            df_fb = download_ohlcv(fallback, START_DATE)
            if ensure_volume(df_fb):
                df = df_fb
                used = fallback

        if df.empty:
            print(f"{name}: no data for {primary} or {fallback}")
            continue

        sig = add_signals(df)
        sig["UsedTicker"] = used
        sig["MarketName"] = name

        # latest snapshot
        latest = summarize_latest(sig, f"{name} ({used})")
        all_latest.append(latest)

        # collect all historical hit dates for each signal
        hits = sig.loc[
            sig["DownsideRecordVol"] | sig["NewHigh_LowVol"] | sig["HiVol_Churn"],
            ["Close", "RetPct", "Volume", "DownsideRecordVol", "NewHigh_LowVol", "HiVol_Churn", "UsedTicker", "MarketName"]
        ].copy()
        hits.index = pd.to_datetime(hits.index)
        hits["Date"] = hits.index.date.astype(str)
        all_hits.append(hits.reset_index(drop=True))

    if all_latest:
        latest_df = pd.concat(all_latest, ignore_index=True)
        # Format numeric columns to 2 decimal places
        latest_df["Close"] = latest_df["Close"].round(2)

        print("\n=== Latest Signals ===")
        # Print header
        print(f"{'Market':<20} {'Date':<12} {'DownsideRecordVol':<18} {'NewHigh_LowVol':<16} {'HiVol_Churn':<12} {'Close':>10} {'RetPct':>8} {'Volume':>14}")

        # Print each row with colored RetPct
        for _, row in latest_df.iterrows():
            colored_retpct = colorize_retpct(row['RetPct'])
            # ANSI codes add 11 chars (escape sequences), so pad accordingly
            padding = 19 if '\033[' in colored_retpct else 8

            print(f"{row['Market']:<20} {row['Date']:<12} {str(row['DownsideRecordVol']):<18} "
                  f"{str(row['NewHigh_LowVol']):<16} {str(row['HiVol_Churn']):<12} "
                  f"{row['Close']:>10.2f} {colored_retpct:>{padding}} {row['Volume']:>14.1f}")

    if all_hits:
        hits_df = pd.concat(all_hits, ignore_index=True)
        hits_df = hits_df.sort_values("Date", ascending=False)
        # Format numeric columns to 2 decimal places
        hits_df["Close"] = hits_df["Close"].round(2)

        # Show most recent signals grouped by index
        print("\n=== Most Recent Signal Dates (by Index) ===")
        cols = ["Date", "Close", "RetPct", "Volume",
                "DownsideRecordVol", "NewHigh_LowVol", "HiVol_Churn"]

        for market_name in hits_df["MarketName"].unique():
            market_hits = hits_df[hits_df["MarketName"] == market_name]
            if not market_hits.empty:
                used_ticker = market_hits.iloc[0]["UsedTicker"]

                # Create display dataframe with colored RetPct
                display_hits = market_hits[cols].head(10).copy()
                display_hits["RetPct"] = display_hits["RetPct"].apply(colorize_retpct)

                print(f"\n{market_name} ({used_ticker}) - Last 10 signals:")
                print(display_hits.to_string(index=False))


if __name__ == "__main__":
    main()
