#!/usr/bin/env python3
"""
VIX term structure signal (3M VIX / 1M VIX).

Idea:
  - High ratio (e.g., >= 1.25): investors more concerned about later volatility.
  - Low ratio (< 1.0): near-term fear; can coincide with tactical lows.

Dependencies:
  pip install pandas numpy yfinance

Usage:
  python3 vix_term_structure.py
  python3 vix_term_structure.py --start 2010-01-01 --low 1.0 --high 1.25 --tail 5 --signals 10
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
except ImportError:
    Console = None

CONSOLE = Console() if Console else None

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Missing dependency yfinance. Install with: pip install yfinance") from e


VIX_TICKER = "^VIX"
VIX3M_TICKER = "^VIX3M"
VIX3M_FALLBACK = "^VXV"  # 3-month VIX proxy if VIX3M is unavailable

DEFAULT_START = "2006-01-01"
DEFAULT_LOW = 1.0
DEFAULT_HIGH = 1.25


def print_header() -> None:
    if CONSOLE:
        title = Text("VIX Term Structure", style="bold cyan")
        subtitle = Text("3M VIX / 1M VIX ratio", style="dim")
        body = Text.assemble(title, "\n", subtitle)
        CONSOLE.print(Panel.fit(body, box=box.ASCII, padding=(1, 4), style="cyan"))
        return
    print("=" * 60)
    print("VIX TERM STRUCTURE")
    print("=" * 60)


def download_close(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    if "Close" not in df.columns:
        return pd.DataFrame()
    out = df[["Close"]].copy()
    out.columns = [ticker]
    return out


def load_term_structure(start: str) -> tuple[pd.DataFrame, str]:
    vix = download_close(VIX_TICKER, start)
    if vix.empty or vix[VIX_TICKER].dropna().empty:
        raise RuntimeError(f"No data for {VIX_TICKER}")

    vix3m = download_close(VIX3M_TICKER, start)
    used_vix3m = VIX3M_TICKER
    if vix3m.empty or vix3m[VIX3M_TICKER].dropna().empty:
        vix3m = download_close(VIX3M_FALLBACK, start)
        used_vix3m = VIX3M_FALLBACK

    if vix3m.empty or vix3m[used_vix3m].dropna().empty:
        raise RuntimeError(f"No data for {VIX3M_TICKER} or {VIX3M_FALLBACK}")

    data = pd.concat([vix, vix3m], axis=1, join="inner")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data.columns = ["VIX", "VIX3M"]
    data["Ratio"] = data["VIX3M"] / data["VIX"]
    return data, used_vix3m


def add_signals(data: pd.DataFrame, low: float, high: float) -> pd.DataFrame:
    out = data.copy()
    conditions = [out["Ratio"] >= high, out["Ratio"] < low]
    choices = ["Complacency", "Fear"]
    out["Signal"] = np.select(conditions, choices, default="Neutral")
    return out


def format_signal_text(value: str) -> Text:
    if value == "Fear":
        return Text(value, style="red")
    if value == "Complacency":
        return Text(value, style="yellow")
    if value == "Neutral":
        return Text(value, style="dim")
    return Text(str(value))


def render_latest(latest: pd.Series, used_vix3m: str) -> None:
    table = Table(title="Latest Snapshot", box=box.ASCII)
    table.add_column("Date")
    table.add_column("VIX", justify="right")
    table.add_column(f"3M ({used_vix3m})", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("Signal")

    table.add_row(
        latest.name.date().isoformat(),
        f"{latest['VIX']:.2f}",
        f"{latest['VIX3M']:.2f}",
        f"{latest['Ratio']:.2f}",
        format_signal_text(str(latest["Signal"])),
    )
    CONSOLE.print()
    CONSOLE.print(table)


def render_recent(data: pd.DataFrame, used_vix3m: str, title: str) -> None:
    table = Table(title=title, box=box.ASCII)
    table.add_column("Date")
    table.add_column("VIX", justify="right")
    table.add_column(f"3M ({used_vix3m})", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("Signal")

    for dt, row in data.iterrows():
        table.add_row(
            dt.date().isoformat(),
            f"{row['VIX']:.2f}",
            f"{row['VIX3M']:.2f}",
            f"{row['Ratio']:.2f}",
            format_signal_text(str(row["Signal"])),
        )
    CONSOLE.print()
    CONSOLE.print(table)


def print_latest(latest: pd.Series, used_vix3m: str) -> None:
    print("\nLatest Snapshot")
    print(f"Date:   {latest.name.date().isoformat()}")
    print(f"VIX:    {latest['VIX']:.2f}")
    print(f"3M({used_vix3m}): {latest['VIX3M']:.2f}")
    print(f"Ratio:  {latest['Ratio']:.2f}")
    print(f"Signal: {latest['Signal']}")


def print_table(data: pd.DataFrame, used_vix3m: str, title: str) -> None:
    if data.empty:
        print(f"\n{title}: none")
        return
    print(f"\n{title}")
    header = f"{'Date':<12} {'VIX':>7} {f'3M({used_vix3m})':>10} {'Ratio':>8} {'Signal':>12}"
    print(header)
    print("-" * len(header))
    for dt, row in data.iterrows():
        print(
            f"{dt.date().isoformat():<12} "
            f"{row['VIX']:>7.2f} "
            f"{row['VIX3M']:>10.2f} "
            f"{row['Ratio']:>8.2f} "
            f"{row['Signal']:>12}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VIX term structure: 3M VIX / 1M VIX ratio"
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--low", type=float, default=DEFAULT_LOW, help="Low threshold (fear)")
    parser.add_argument("--high", type=float, default=DEFAULT_HIGH, help="High threshold (complacency)")
    parser.add_argument("--tail", type=int, default=5, help="Recent sessions to show (0 to skip)")
    parser.add_argument("--signals", type=int, default=10, help="Most recent signal hits to show (0 to skip)")
    args = parser.parse_args()

    print_header()
    data, used_vix3m = load_term_structure(args.start)
    signals = add_signals(data, args.low, args.high)

    latest = signals.iloc[-1]

    if CONSOLE:
        render_latest(latest, used_vix3m)
    else:
        print_latest(latest, used_vix3m)

    if args.tail > 0:
        recent = signals.tail(args.tail)
        title = f"Recent Ratios (last {len(recent)})"
        if CONSOLE:
            render_recent(recent, used_vix3m, title)
        else:
            print_table(recent, used_vix3m, title)

    if args.signals > 0:
        hits = signals[signals["Signal"] != "Neutral"].copy()
        hits = hits.sort_index(ascending=False).head(args.signals)
        title = f"Most Recent Signal Hits (last {len(hits)})"
        if CONSOLE:
            render_recent(hits, used_vix3m, title)
        else:
            print_table(hits, used_vix3m, title)


if __name__ == "__main__":
    main()
