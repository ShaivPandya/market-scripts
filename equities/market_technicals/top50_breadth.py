"""
Compute breadth metrics for top 50 S&P 500 performers (by 6-month return).

Metrics calculated:
  1. % below 50-day moving average
  2. % with 3+ distribution days in last 20 sessions
  3. % that closed below prior 20-day low in last 5 days

Distribution days: down days with above-average volume (50-day avg).

Usage:
  python3 top50_breadth.py
"""

from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd
import yfinance as yf
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
except ImportError:
    Console = None

CONSOLE = Console() if Console else None

from get_top50 import main as generate_top50


def print_header() -> None:
    if CONSOLE:
        title = Text("Top 50 Breadth", style="bold cyan")
        subtitle = Text("Top performers | Leadership signals", style="dim")
        body = Text.assemble(title, "\n", subtitle)
        CONSOLE.print(Panel.fit(body, box=box.ASCII, padding=(1, 4), style="cyan"))
        return
    print("=" * 60)
    print("TOP 50 BREADTH")
    print("=" * 60)


def format_ticker_list(series):
    tickers = ", ".join(series) or "(none)"
    if tickers == "(none)":
        return Text(tickers, style="dim")
    return tickers


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


def _no_data_row(ticker: str, error: str = "no data") -> Dict[str, Any]:
    return {
        "ticker": ticker.upper(),
        "rows": 0,
        "below_50dma": None,
        "dist_days_last20": None,
        "has_3plus_dist_days": None,
        "broke_prior20_low_last_week": None,
        "error": error,
    }


def compute_metrics(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    if raw is None or raw.empty:
        return pd.DataFrame([_no_data_row(t) for t in tickers])

    rows = []
    is_multi = isinstance(raw.columns, pd.MultiIndex)

    for t in tickers:
        try:
            if is_multi:
                ticker_df = raw.xs(t, level="Ticker", axis=1).dropna(subset=["Close", "Volume"])
            else:
                # Single ticker — columns are already flat
                ticker_df = raw.dropna(subset=["Close", "Volume"])

            if ticker_df.empty:
                rows.append(_no_data_row(t))
                continue

            res = analyze_ticker(ticker_df)
            rows.append({
                "ticker": t.upper(),
                "rows": res["rows"],
                "below_50dma": res["below_50dma"],
                "dist_days_last20": res["dist_days_last20"],
                "has_3plus_dist_days": res["has_3plus_dist_days"],
                "broke_prior20_low_last_week": res["broke_prior20_low_last_week"],
            })
        except Exception as e:
            rows.append(_no_data_row(t, error=str(e)))

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> None:
    valid = df[df["rows"] >= 30].copy()
    n = len(valid)
    if n == 0:
        if CONSOLE:
            CONSOLE.print("No tickers with sufficient history.", style="yellow")
        else:
            print("No tickers with sufficient history.")
        return

    pct_below_50dma = 100 * valid["below_50dma"].mean()
    pct_3plus_dist = 100 * valid["has_3plus_dist_days"].mean()
    pct_broke_20low = 100 * valid["broke_prior20_low_last_week"].mean()

    if CONSOLE:
        summary = Table(title="Top 50 Breadth Summary", box=box.ASCII)
        summary.add_column("Metric")
        summary.add_column("Value", justify="right")
        summary.add_row("Universe size (sufficient data)", str(n))
        summary.add_row("% below 50-DMA", f"{pct_below_50dma:.2f}%")
        summary.add_row("% with >=3 distribution days (last 20)", f"{pct_3plus_dist:.2f}%")
        summary.add_row("% that closed below prior 20-day low in last 5 days", f"{pct_broke_20low:.2f}%")
        CONSOLE.print(summary)
        CONSOLE.print(
            "Distribution days are defined as days where the stock closed lower "
            "but volume was above the 50-day average volume.",
            style="dim",
        )

        tickers_table = Table(title="Affected Tickers", box=box.ASCII)
        tickers_table.add_column("Signal")
        tickers_table.add_column("Tickers")
        tickers_table.add_row(
            "Below 50-DMA",
            format_ticker_list(valid.loc[valid["below_50dma"], "ticker"]),
        )
        tickers_table.add_row(
            ">=3 distribution days (last 20)",
            format_ticker_list(valid.loc[valid["has_3plus_dist_days"], "ticker"]),
        )
        tickers_table.add_row(
            "Broke prior 20-day low in last 5 days",
            format_ticker_list(valid.loc[valid["broke_prior20_low_last_week"], "ticker"]),
        )
        CONSOLE.print(tickers_table)
    else:
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


def get_summary_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract summary metrics from the computed DataFrame for GUI consumption.

    Returns dict with:
      - pct_below_50dma, pct_3plus_dist, pct_broke_20low: percentages
      - tickers_below_50dma, tickers_3plus_dist, tickers_broke_20low: lists
      - universe_size: count of valid tickers
      - raw_df: the full DataFrame for detailed views
    """
    valid = df[df["rows"] >= 30].copy()
    n = len(valid)

    if n == 0:
        return {
            "pct_below_50dma": None,
            "pct_3plus_dist": None,
            "pct_broke_20low": None,
            "tickers_below_50dma": [],
            "tickers_3plus_dist": [],
            "tickers_broke_20low": [],
            "universe_size": 0,
            "raw_df": df,
        }

    return {
        "pct_below_50dma": 100 * valid["below_50dma"].mean(),
        "pct_3plus_dist": 100 * valid["has_3plus_dist_days"].mean(),
        "pct_broke_20low": 100 * valid["broke_prior20_low_last_week"].mean(),
        "tickers_below_50dma": valid.loc[valid["below_50dma"], "ticker"].tolist(),
        "tickers_3plus_dist": valid.loc[valid["has_3plus_dist_days"], "ticker"].tolist(),
        "tickers_broke_20low": valid.loc[valid["broke_prior20_low_last_week"], "ticker"].tolist(),
        "universe_size": n,
        "raw_df": df,
    }


def get_data(period: str = "2y") -> Dict[str, Any]:
    """
    Fetch top50 breadth data for GUI consumption.

    Returns dict with summary metrics and raw DataFrame.
    """
    generate_top50()
    script_dir = Path(__file__).parent
    csv_path = script_dir / "sp500_top50_6mo.csv"
    csv_df = pd.read_csv(csv_path)
    tickers = csv_df["ticker"].dropna().astype(str).str.upper().tolist()
    df = compute_metrics(tickers, period=period)
    return get_summary_metrics(df)


def main():
    print_header()
    print("Generating top 50 S&P 500 performers...")
    generate_top50()
    script_dir = Path(__file__).parent
    csv_path = script_dir / "sp500_top50_6mo.csv"
    csv_df = pd.read_csv(csv_path)
    tickers = csv_df["ticker"].dropna().astype(str).str.upper().tolist()
    df = compute_metrics(tickers, period="2y")
    summarize(df)


if __name__ == "__main__":
    main()
