#!/usr/bin/env python3
"""
Rate of Change (ROC) Analysis for Stock Momentum

DESCRIPTION:
    This script analyzes stock price momentum using rate-of-change calculations.
    It computes:
    1. First derivative: Long-term momentum (default 52-week lookback)
    2. Second derivative: Change in momentum (default 4-week lookback)
    3. Smoothed second derivative using EMA to reduce noise

    The script visualizes price action alongside momentum indicators and detects
    crossover points where the second derivative crosses its smoothed EMA, which
    can signal potential trend changes or acceleration/deceleration in momentum.

USAGE:
    python3 rate_of_change.py <TICKER> [OPTIONS]

    Examples:
        python3 rate_of_change.py AAPL
        python3 rate_of_change.py AAPL --mom 52 --deriv 4 --smooth 13
        python3 rate_of_change.py TSLA --years 5 --mom 26

    Required:
        ticker              Stock ticker symbol (e.g., AAPL, TSLA, MSFT)

    Optional:
        --years INT         Years of historical data to fetch (default: 10)
        --mom INT           Momentum lookback period in weeks (default: 52)
        --deriv INT         Second derivative lookback in weeks (default: 4)
        --smooth INT        EMA smoothing period in weeks (default: 13)

DEPENDENCIES:
    pip install yfinance pandas matplotlib numpy

OUTPUT:
    - Three-panel chart showing price, momentum, and change in momentum
    - Console table listing all crossover dates with price and direction
"""
import argparse, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError:
    sys.stderr.write("pip install yfinance pandas matplotlib\n"); sys.exit(1)

def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def main():
    p = argparse.ArgumentParser(description="Price, 52w momentum, and 2nd-derivative-of-momentum")
    p.add_argument("ticker")
    p.add_argument("--years", type=int, default=10, help="history to fetch (default 10)")
    p.add_argument("--mom", type=int, default=52, help="momentum lookback in weeks (default 52)")
    p.add_argument("--deriv", type=int, default=4, help="second-derivative lookback in weeks (default 4≈1 month)")
    p.add_argument("--smooth", type=int, default=13, help="EMA smoothing in weeks for 2nd deriv (default 13)")
    args = p.parse_args()

    # weekly closes
    df = yf.download(args.ticker.upper(), period=f"{args.years}y", interval="1d",
                     auto_adjust=True, progress=False)
    if df.empty:
        sys.stderr.write("no data\n"); sys.exit(2)
    close_w = df["Close"].resample("W-FRI").last().dropna()

    # log price
    p_log = np.log(close_w)

    # first derivative: long-horizon momentum (52w by default)
    r1 = p_log - p_log.shift(args.mom)

    # second derivative: change in that momentum over a short horizon (4w by default)
    r2 = r1 - r1.shift(args.deriv)

    # smooth to cut noise
    r2s = ema(r2, args.smooth)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(12, 9),
                                        gridspec_kw={"height_ratios":[3,1,1]})
    ax0.plot(close_w.index, close_w, lw=1.2)
    ax0.set_title(f"{args.ticker.upper()} — Price, {args.mom}w Momentum, and ΔMomentum ({args.deriv}w, EMA {args.smooth})")
    ax0.set_ylabel("Price")

    ax1.plot(r1.index, r1, lw=1.0); ax1.axhline(0, lw=0.8); ax1.set_ylabel("Momentum")

    ax2.plot(r2.index, r2, lw=0.6, alpha=0.4, label="raw ΔMomentum")
    ax2.plot(r2s.index, r2s, lw=1.2, label=f"EMA({args.smooth})")
    ax2.axhline(0, lw=0.8); ax2.set_ylabel("ΔMomentum"); ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")

    # Detect crossovers: r2 crossing r2s
    # Align all series to the same index and ensure they're valid
    if not r2.empty and not r2s.empty and not close_w.empty:
        combined = pd.DataFrame(index=r2.index)
        combined["r2"] = r2
        combined["r2s"] = r2s
        combined["price"] = close_w
        combined = combined.dropna()
    else:
        combined = pd.DataFrame()

    if len(combined) > 1:
        # Calculate the difference and detect sign changes
        diff = combined["r2"] - combined["r2s"]
        crosses = (diff.shift(1) * diff < 0)  # Sign change indicates a cross

        cross_dates = combined[crosses].index
        if len(cross_dates) > 0:
            print(f"\n{'='*70}")
            print(f"ΔMomentum (r2) crosses EMA({args.smooth}) - {args.ticker.upper()}")
            print(f"{'='*70}")
            print(f"{'Date':<12} {'Price':>10} {'Direction':<15}")
            print(f"{'-'*70}")

            for date in cross_dates:
                price = combined.loc[date, "price"]
                r2_val = combined.loc[date, "r2"]
                r2s_val = combined.loc[date, "r2s"]

                # Determine if crossing up or down
                direction = "Cross Above" if r2_val > r2s_val else "Cross Below"

                print(f"{date.strftime('%Y-%m-%d'):<12} ${price:>9.2f} {direction:<15}")

            print(f"{'='*70}\n")

    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
