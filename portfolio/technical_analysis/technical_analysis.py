#!/usr/bin/env python3
"""
Technical Analysis – moving-average & rate-of-change dashboard for a single ticker.

CLI usage:
    python technical_analysis.py AAPL

GUI usage (called from gui/app.py):
    from technical_analysis import get_data
    result = get_data("AAPL")
"""

import logging
import sys
from datetime import datetime

import pandas as pd

from utils.retry import yf_download, yf_ticker_info

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def _fetch_daily(ticker: str, years: int = 9) -> pd.Series:
    """Download daily close prices (enough history for 200-week MA over 5Y)."""
    raw = yf_download(
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
            rows.append(
                {
                    "Indicator": f"Price vs {col}",
                    "Value": f"{ma_val:,.2f}",
                    "Signal": "Above" if above else "Below",
                    "Bias": "Bullish" if above else "Bearish",
                }
            )

    # ROC signals
    for col in ["1M ROC", "3M ROC", "12M ROC"]:
        val = latest_roc.get(col)
        if pd.notna(val):
            rows.append(
                {
                    "Indicator": col,
                    "Value": f"{val:+.2f}%",
                    "Signal": "Positive" if val >= 0 else "Negative",
                    "Bias": "Bullish" if val >= 0 else "Bearish",
                }
            )

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

RATIO_METHOD_PRICE = "price_ratio"
RATIO_METHODS = {RATIO_METHOD_PRICE}


def _fetch_name(ticker: str) -> str:
    """Return the long name for *ticker*, falling back to the ticker symbol."""
    try:
        info = yf_ticker_info(ticker)
        return info.get("longName") or info.get("shortName") or ticker.upper()
    except Exception:
        return ticker.upper()


def _fetch_pair_daily(symbol_a: str, symbol_b: str, years: int = 10) -> pd.DataFrame:
    """Download daily close prices for a two-symbol ratio series."""
    symbols = [symbol_a.upper(), symbol_b.upper()]
    raw = yf_download(
        symbols,
        period=f"{years}y",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise ValueError(f"No data returned for symbols '{symbols[0]}' and '{symbols[1]}'")

    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("Unexpected Yahoo response while fetching paired symbols.")

    level0 = set(raw.columns.get_level_values(0))
    if "Close" in level0:
        close = raw["Close"].copy()
    elif "Adj Close" in level0:
        close = raw["Adj Close"].copy()
    else:
        raise ValueError("Yahoo response did not include Close prices.")

    close.columns = [str(col).strip().upper() for col in close.columns]
    missing = [symbol for symbol in symbols if symbol not in close.columns]
    if missing:
        raise ValueError(f"Missing price history for symbol(s): {missing}")

    close = close[symbols].dropna(how="all")
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    return close


def _parse_optional_date(value: str | None, field_name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid {field_name}: '{value}'. Use YYYY-MM-DD.")
    return pd.Timestamp(parsed).normalize()


def _compute_price_ratio(close_df: pd.DataFrame, symbol_a: str, symbol_b: str) -> pd.Series:
    safe_denominator = close_df[symbol_b].replace(0, pd.NA)
    ratio = close_df[symbol_a] / safe_denominator
    return ratio.rename("Ratio")


def get_data(ticker: str, lookback: str = "2Y") -> dict:
    """Fetch and compute all technical analysis data for *ticker*."""
    try:
        close = _fetch_daily(ticker)
        price_df = _moving_averages(close)
        roc_df = _rate_of_change(close)
        summary = _build_summary(price_df, roc_df)
        name = _fetch_name(ticker)

        # Trim display to selected lookback
        offset = LOOKBACK_OPTIONS.get(lookback, pd.DateOffset(years=2))
        cutoff = price_df.index.max() - offset
        price_df = price_df.loc[price_df.index >= cutoff]
        roc_df = roc_df.loc[roc_df.index >= cutoff]

        return {
            "ticker": ticker.upper(),
            "name": name,
            "price_data": price_df,
            "roc_data": roc_df,
            "summary": summary,
            "timestamp": datetime.now(),
        }
    except Exception as e:
        import traceback

        return {"error": f"{e}\n\n{traceback.format_exc()}"}


def get_ratio_data(
    symbol_a: str,
    symbol_b: str,
    start_date: str | None = None,
    end_date: str | None = None,
    method: str = RATIO_METHOD_PRICE,
) -> dict:
    """Fetch and compute ratio data for two symbols."""
    try:
        a = str(symbol_a).strip().upper()
        b = str(symbol_b).strip().upper()
        if not a or not b:
            raise ValueError("Both symbol_a and symbol_b are required.")
        if a == b:
            raise ValueError("symbol_a and symbol_b must be different.")

        method_normalized = str(method).strip().lower() or RATIO_METHOD_PRICE
        if method_normalized not in RATIO_METHODS:
            supported = ", ".join(sorted(RATIO_METHODS))
            raise ValueError(f"Unsupported method '{method}'. Supported methods: {supported}.")

        start_ts = _parse_optional_date(start_date, "start_date")
        end_ts = _parse_optional_date(end_date, "end_date")
        if start_ts is not None and end_ts is not None and start_ts > end_ts:
            raise ValueError("start_date must be less than or equal to end_date.")

        close_df = _fetch_pair_daily(a, b, years=10)
        if start_ts is not None:
            close_df = close_df.loc[close_df.index >= start_ts]
        if end_ts is not None:
            close_df = close_df.loc[close_df.index <= end_ts]

        close_df = close_df.dropna(subset=[a, b])
        if close_df.empty:
            raise ValueError("No overlapping price history found in the selected date range.")

        ratio_series = _compute_price_ratio(close_df, a, b)
        ratio_df = pd.DataFrame(
            {
                "Price A": close_df[a],
                "Price B": close_df[b],
                "Ratio": ratio_series,
            }
        ).dropna(subset=["Ratio"])
        if ratio_df.empty:
            raise ValueError("Ratio series is empty after filtering.")

        start_ratio = float(ratio_df["Ratio"].iloc[0])
        end_ratio = float(ratio_df["Ratio"].iloc[-1])
        ratio_change_pct = ((end_ratio / start_ratio) - 1.0) if start_ratio != 0 else None
        historical_avg = float(ratio_df["Ratio"].mean())
        historical_median = float(ratio_df["Ratio"].median())
        current_vs_historical_avg_pct = ((end_ratio / historical_avg) - 1.0) if historical_avg != 0 else None
        if abs(end_ratio - historical_avg) < 1e-12:
            historical_position = "at"
        elif end_ratio > historical_avg:
            historical_position = "above"
        else:
            historical_position = "below"

        return {
            "symbol_a": a,
            "symbol_b": b,
            "name_a": _fetch_name(a),
            "name_b": _fetch_name(b),
            "method": method_normalized,
            "ratio_label": f"{a}/{b}",
            "ratio_data": ratio_df,
            "stats": {
                "start_ratio": start_ratio,
                "end_ratio": end_ratio,
                "change_pct": ratio_change_pct,
                "historical_avg": historical_avg,
                "historical_median": historical_median,
                "current_vs_historical_avg_pct": current_vs_historical_avg_pct,
                "historical_position": historical_position,
                "min_ratio": float(ratio_df["Ratio"].min()),
                "max_ratio": float(ratio_df["Ratio"].max()),
                "start_date": ratio_df.index.min().date().isoformat(),
                "end_date": ratio_df.index.max().date().isoformat(),
                "observations": int(len(ratio_df)),
            },
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
        LOGGER.error("Error: %s", result["error"])
        sys.exit(1)

    price_df = result["price_data"]
    roc_df = result["roc_data"]
    summary = result["summary"]
    ticker = result["ticker"]
    name = result.get("name", ticker)

    # -- Charts --
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        height_ratios=[3, 1],
        sharex=True,
        gridspec_kw={"hspace": 0.08},
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

    ax1.set_title(f"{ticker} – {name} – Technical Analysis", fontsize=14, fontweight="bold", color="white")
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
        f"\nOverall: [{overall_color}]{overall}[/{overall_color}] ({bullish_count}/{total} bullish signals)\n"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    LOGGER.info("Starting script execution: %s", __file__)
    if len(sys.argv) < 2:
        print("Usage: python technical_analysis.py <TICKER>")
        sys.exit(1)
    _cli(sys.argv[1])
