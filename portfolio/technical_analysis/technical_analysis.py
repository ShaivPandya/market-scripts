#!/usr/bin/env python3
"""
Technical Analysis – moving-average & rate-of-change dashboard for a single ticker.

CLI usage:
    python technical_analysis.py AAPL

GUI usage (called from gui/app.py):
    from technical_analysis import get_data
    result = get_data("AAPL")
"""

import sys
from datetime import datetime

import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_daily(ticker: str, years: int = 9) -> pd.Series:
    """Download daily close prices (enough history for 200-week MA over 5Y)."""
    raw = yf.download(
        ticker,
        period=f"{years}y",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")
    close = raw["Close"].squeeze().dropna()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    close.name = "Close"
    return close


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def _moving_averages(close: pd.Series) -> pd.DataFrame:
    """Compute daily, weekly, and monthly SMAs reindexed to daily."""
    df = close.to_frame("Close")

    # Daily
    df["100D SMA"] = close.rolling(100, min_periods=100).mean()
    df["150D SMA"] = close.rolling(150, min_periods=150).mean()
    df["200D SMA"] = close.rolling(200, min_periods=200).mean()

    # Weekly
    weekly = close.resample("W-FRI").last().dropna()
    w40 = weekly.rolling(40, min_periods=40).mean()
    w200 = weekly.rolling(200, min_periods=200).mean()
    df["40W SMA"] = w40.reindex(df.index, method="ffill")
    df["200W SMA"] = w200.reindex(df.index, method="ffill")

    # Monthly
    monthly = close.resample("ME").last().dropna()
    m10 = monthly.rolling(10, min_periods=10).mean()
    m20 = monthly.rolling(20, min_periods=20).mean()
    df["10M SMA"] = m10.reindex(df.index, method="ffill")
    df["20M SMA"] = m20.reindex(df.index, method="ffill")

    return df


# ---------------------------------------------------------------------------
# Rate of change
# ---------------------------------------------------------------------------

def _rate_of_change(close: pd.Series) -> pd.DataFrame:
    """1-month, 3-month, and 12-month ROC (%)."""
    roc = pd.DataFrame(index=close.index)
    roc["1M ROC"] = (close / close.shift(21) - 1) * 100
    roc["3M ROC"] = (close / close.shift(63) - 1) * 100
    roc["12M ROC"] = (close / close.shift(252) - 1) * 100
    return roc


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _build_summary(price_df: pd.DataFrame, roc_df: pd.DataFrame) -> list[dict]:
    """Generate a list of signal rows for the summary table."""
    latest = price_df.dropna(subset=["Close"]).iloc[-1]
    latest_roc = roc_df.dropna(how="all").iloc[-1]
    close = latest["Close"]

    rows: list[dict] = []

    # MA signals
    for col in ["100D SMA", "150D SMA", "200D SMA", "40W SMA", "200W SMA", "10M SMA", "20M SMA"]:
        ma_val = latest.get(col)
        if pd.notna(ma_val):
            above = close >= ma_val
            rows.append({
                "Indicator": f"Price vs {col}",
                "Value": f"{ma_val:,.2f}",
                "Signal": "Above" if above else "Below",
                "Bias": "Bullish" if above else "Bearish",
            })

    # ROC signals
    for col in ["1M ROC", "3M ROC", "12M ROC"]:
        val = latest_roc.get(col)
        if pd.notna(val):
            rows.append({
                "Indicator": col,
                "Value": f"{val:+.2f}%",
                "Signal": "Positive" if val >= 0 else "Negative",
                "Bias": "Bullish" if val >= 0 else "Bearish",
            })

    return rows


# ---------------------------------------------------------------------------
# Public API (used by GUI)
# ---------------------------------------------------------------------------

LOOKBACK_OPTIONS = {
    "3M": pd.DateOffset(months=3),
    "1Y": pd.DateOffset(years=1),
    "2Y": pd.DateOffset(years=2),
    "5Y": pd.DateOffset(years=5),
}


def get_data(ticker: str, lookback: str = "2Y") -> dict:
    """Fetch and compute all technical analysis data for *ticker*."""
    try:
        close = _fetch_daily(ticker)
        price_df = _moving_averages(close)
        roc_df = _rate_of_change(close)
        summary = _build_summary(price_df, roc_df)

        # Trim display to selected lookback
        offset = LOOKBACK_OPTIONS.get(lookback, pd.DateOffset(years=2))
        cutoff = price_df.index.max() - offset
        price_df = price_df.loc[price_df.index >= cutoff]
        roc_df = roc_df.loc[roc_df.index >= cutoff]

        return {
            "ticker": ticker.upper(),
            "price_data": price_df,
            "roc_data": roc_df,
            "summary": summary,
            "timestamp": datetime.now(),
        }
    except Exception as e:
        import traceback
        return {"error": f"{e}\n\n{traceback.format_exc()}"}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli(ticker: str) -> None:
    """Render charts with matplotlib and print summary with rich."""
    import matplotlib.pyplot as plt
    from rich.console import Console
    from rich.table import Table

    result = get_data(ticker)
    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    price_df = result["price_data"]
    roc_df = result["roc_data"]
    summary = result["summary"]
    ticker = result["ticker"]

    # -- Charts --
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), height_ratios=[3, 1],
        sharex=True, gridspec_kw={"hspace": 0.08},
    )

    # Price + MAs
    ax1.plot(price_df.index, price_df["Close"], label="Close", linewidth=1.4, color="white")
    ma_colors = {
        "100D SMA": "#FB923C",
        "150D SMA": "#38BDF8",
        "200D SMA": "#FF6B6B",
        "40W SMA": "#4ECDC4",
        "200W SMA": "#FFE66D",
        "10M SMA": "#A78BFA",
        "20M SMA": "#F472B6",
    }
    for col, color in ma_colors.items():
        valid = price_df[col].dropna()
        if not valid.empty:
            ax1.plot(valid.index, valid.values, label=col, linewidth=1, color=color, alpha=0.85)

    ax1.set_title(f"{ticker} – Technical Analysis", fontsize=14, fontweight="bold", color="white")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)

    # ROC
    roc_colors = {"1M ROC": "#FF6B6B", "3M ROC": "#4ECDC4", "12M ROC": "#FFE66D"}
    for col, color in roc_colors.items():
        valid = roc_df[col].dropna()
        if not valid.empty:
            ax2.plot(valid.index, valid.values, label=col, linewidth=1, color=color, alpha=0.85)
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("ROC (%)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.patch.set_facecolor("#1e1e1e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#555")

    plt.tight_layout()
    plt.show()

    # -- Summary table --
    console = Console()
    table = Table(title=f"{ticker} Signal Summary", show_lines=True)
    table.add_column("Indicator", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Signal")
    table.add_column("Bias")

    for row in summary:
        bias_style = "green" if row["Bias"] == "Bullish" else "red"
        table.add_row(
            row["Indicator"],
            row["Value"],
            row["Signal"],
            f"[{bias_style}]{row['Bias']}[/{bias_style}]",
        )

    bullish_count = sum(1 for r in summary if r["Bias"] == "Bullish")
    total = len(summary)
    overall = "Bullish" if bullish_count > total / 2 else "Bearish" if bullish_count < total / 2 else "Neutral"
    overall_color = {"Bullish": "green", "Bearish": "red", "Neutral": "yellow"}[overall]

    console.print(table)
    console.print(
        f"\nOverall: [{overall_color}]{overall}[/{overall_color}] "
        f"({bullish_count}/{total} bullish signals)\n"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python technical_analysis.py <TICKER>")
        sys.exit(1)
    _cli(sys.argv[1])
