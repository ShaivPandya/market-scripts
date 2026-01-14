"""
python3 breadth.py --from-file sp500_top50_6mo.csv
"""

from __future__ import annotations
import argparse
from typing import List, Dict, Any
import pandas as pd
import yfinance as yf


def fetch_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close", "Volume"])


def analyze_ticker(df: pd.DataFrame) -> Dict[str, Any]:
    out = {
        "below_50dma": False,
        "dist_days_last20": 0,
        "has_3plus_dist_days": False,
        "broke_prior20_low_last_week": False,
        "rows": int(df.shape[0]),
    }
    if df.shape[0] < 30:  # lowered requirement from 55 to 30
        return out

    sma50_close = df["Close"].rolling(50, min_periods=1).mean()
    sma50_vol = df["Volume"].rolling(50, min_periods=1).mean()

    last_close = df["Close"].iloc[-1]
    last_sma50 = sma50_close.iloc[-1]
    if pd.notna(last_sma50):
        out["below_50dma"] = bool(last_close < last_sma50)

    down_vs_prior = df["Close"] < df["Close"].shift(1)
    dist_day = down_vs_prior & (df["Volume"] > sma50_vol)
    out["dist_days_last20"] = int(dist_day.tail(20).sum())
    out["has_3plus_dist_days"] = bool(out["dist_days_last20"] >= 3)

    if df.shape[0] >= 25:
        prior20_low = df["Low"].iloc[-25:-5].min()
        last5_close = df["Close"].iloc[-5:]
        out["broke_prior20_low_last_week"] = bool((last5_close < prior20_low).any())

    return out


def compute_metrics(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            hist = fetch_history(t, period=period)
            if hist.empty:
                rows.append({
                    "ticker": t.upper(),
                    "rows": 0,
                    "below_50dma": None,
                    "dist_days_last20": None,
                    "has_3plus_dist_days": None,
                    "broke_prior20_low_last_week": None,
                    "error": "no data",
                })
                continue
            res = analyze_ticker(hist)
            rows.append({
                "ticker": t.upper(),
                "rows": res["rows"],
                "below_50dma": res["below_50dma"],
                "dist_days_last20": res["dist_days_last20"],
                "has_3plus_dist_days": res["has_3plus_dist_days"],
                "broke_prior20_low_last_week": res["broke_prior20_low_last_week"],
            })
        except Exception as e:
            rows.append({
                "ticker": t.upper(),
                "rows": 0,
                "below_50dma": None,
                "dist_days_last20": None,
                "has_3plus_dist_days": None,
                "broke_prior20_low_last_week": None,
                "error": str(e),
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> None:
    valid = df[df["rows"] >= 30].copy()
    n = len(valid)
    if n == 0:
        print("No tickers with sufficient history.")
        return

    pct_below_50dma = 100 * valid["below_50dma"].mean()
    pct_3plus_dist = 100 * valid["has_3plus_dist_days"].mean()
    pct_broke_20low = 100 * valid["broke_prior20_low_last_week"].mean()

    print(f"Universe size (sufficient data): {n}")
    print(f"1) % below 50-DMA: {pct_below_50dma:.2f}%")
    print(f"2) % with ≥3 distribution days (last 20): {pct_3plus_dist:.2f}%")
    print(f"3) % that closed below prior 20-day low in last 5 days: {pct_broke_20low:.2f}%\n")

    print("Distribution days are defined as days where the stock closed lower but volume was above the 50-day average volume.\n");

    print("Tickers below 50-DMA:")
    print(", ".join(valid.loc[valid["below_50dma"], "ticker"]) or "(none)")
    print("Tickers with ≥3 distribution days (last 20):")
    print(", ".join(valid.loc[valid["has_3plus_dist_days"], "ticker"]) or "(none)")
    print("Tickers that broke prior 20-day low in last 5 days:")
    print(", ".join(valid.loc[valid["broke_prior20_low_last_week"], "ticker"]) or "(none)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute breadth metrics for tickers.")
    p.add_argument("tickers", nargs="*", help="Ticker symbols, space-separated.")
    p.add_argument("--from-file", help="Path to CSV file with ticker column.")
    p.add_argument("--period", default="2y",
                   help="History period (e.g., 6mo, 1y, 2y). Default 2y.")
    return p.parse_args()


def main():
    args = parse_args()
    tickers: List[str] = []
    if args.from_file:
        csv_df = pd.read_csv(args.from_file)
        if "ticker" not in csv_df.columns:
            print("Error: CSV file must have a 'ticker' column.")
            return
        tickers.extend(csv_df["ticker"].dropna().astype(str).tolist())
    tickers.extend(args.tickers)
    tickers = [t.upper() for t in tickers if t.strip()]
    if not tickers:
        print("Provide tickers via args or --from-file.")
        return
    df = compute_metrics(tickers, period=args.period)
    summarize(df)


if __name__ == "__main__":
    main()
