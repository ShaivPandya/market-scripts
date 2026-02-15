#!/usr/bin/env python3
"""
Momentum ROC Analysis

Computes:
1) 20-day avg of 63-day ROC (%) - absolute price
2) 20-day avg of 63-day ROC (%) - volume
3) 42-day ROC (%) of relative price to benchmark
4) 10-day avg of ROC (%) of relative price to benchmark

Usage:
    python3 momentum.py AAPL --benchmark SPY
    python3 momentum.py --tickers-file tickers.txt --benchmark SPY
    python3 momentum.py --tickers-file us_mega_cap --benchmark QQQ --years 10
    python3 momentum.py AAPL
    python3 momentum.py --list-universes
    python3 momentum.py                    # Uses portfolio.csv by default

Requirements:
    pip install yfinance pandas
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from equities.common import load_universe, list_universes

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[38;2;52;199;89m"
RESET = "\033[0m"
PORTFOLIO_CSV = Path(__file__).resolve().parents[2] / "portfolio.csv"


def colorize(value: float, threshold: float, below_is_red: bool = True) -> str:
    """Return colored string based on threshold comparison."""
    if below_is_red:
        color = RED if value < threshold else GREEN
    else:
        color = GREEN if value < threshold else RED
    return f"{color}{value:.6f}{RESET}"


def fetch_prices_yfinance(ticker: str, years: int = 5) -> tuple[pd.Series, pd.Series]:
    """Fetch price and volume series for a single ticker.

    Returns:
        Tuple of (price_series, volume_series).
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("Missing dependency: yfinance. Install with: pip install yfinance")

    end = datetime.now(UTC).date() + timedelta(days=1)
    start = end - timedelta(days=365 * years)

    df = yf.download(
        ticker,
        start=str(start),
        end=str(end),
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=True,
    )

    if df is None or df.empty:
        raise RuntimeError(f"No data returned for ticker '{ticker}'.")

    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    s = df[col].dropna().copy()
    v = df["Volume"].dropna().copy()

    if isinstance(s, pd.DataFrame):
        s = s.squeeze()
    if isinstance(v, pd.DataFrame):
        v = v.squeeze()

    s.index = pd.to_datetime(s.index).tz_localize(None)
    v.index = pd.to_datetime(v.index).tz_localize(None)
    return s, v


def fetch_prices_batch(tickers: list[str], years: int = 5) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Download multiple tickers in a single API call (much faster than sequential).

    Returns:
        Tuple of (prices_map, volumes_map) where each maps ticker -> Series.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("Missing dependency: yfinance. Install with: pip install yfinance")

    if not tickers:
        return {}, {}

    end = datetime.now(UTC).date() + timedelta(days=1)
    start = end - timedelta(days=365 * years)

    df = yf.download(
        tickers,
        start=str(start),
        end=str(end),
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=True,
    )

    if df is None or df.empty:
        return {}, {}

    prices = {}
    volumes = {}
    # Handle single vs multiple ticker response format
    if len(tickers) == 1:
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        s = df[col].dropna().copy()
        v = df["Volume"].dropna().copy()
        if isinstance(s, pd.DataFrame):
            s = s.squeeze()
        if isinstance(v, pd.DataFrame):
            v = v.squeeze()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        v.index = pd.to_datetime(v.index).tz_localize(None)
        prices[tickers[0]] = s
        volumes[tickers[0]] = v
    else:
        # Multi-ticker: columns are MultiIndex (metric, ticker)
        col = "Adj Close" if "Adj Close" in df.columns.get_level_values(0) else "Close"
        for ticker in tickers:
            try:
                s = df[col][ticker].dropna().copy()
                v = df["Volume"][ticker].dropna().copy()
                if isinstance(s, pd.DataFrame):
                    s = s.squeeze()
                if isinstance(v, pd.DataFrame):
                    v = v.squeeze()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                v.index = pd.to_datetime(v.index).tz_localize(None)
                if len(s) > 0:
                    prices[ticker] = s
                if len(v) > 0:
                    volumes[ticker] = v
            except (KeyError, TypeError):
                continue

    return prices, volumes


def fetch_metadata_batch(tickers: list[str], max_workers: int = 10) -> dict[str, tuple]:
    """Fetch metadata for multiple tickers in parallel using threads."""
    results = {}

    def fetch_one(ticker: str) -> tuple[str, tuple]:
        try:
            return ticker, fetch_ticker_metadata(ticker)
        except Exception:
            return ticker, (None, None, False)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, metadata = future.result()
            results[ticker] = metadata

    return results


def determine_benchmark(metadata: tuple[float | None, str | None, bool]) -> str:
    """Determine benchmark from pre-fetched metadata (no API call)."""
    market_cap, sector, is_etf = metadata

    if is_etf:
        return "SPY"

    if market_cap is not None and market_cap <= 20_000_000_000:
        return "IWM"

    if sector and "technology" in sector.lower():
        return "QQQ"

    return "SPY"


def fetch_ticker_metadata(ticker: str) -> tuple[float | None, str | None, bool]:
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("Missing dependency: yfinance. Install with: pip install yfinance")

    yf_ticker = yf.Ticker(ticker)
    info = yf_ticker.get_info()
    if not info:
        info = yf_ticker.info

    market_cap = info.get("marketCap")
    sector = info.get("sector")
    quote_type = str(info.get("quoteType", "")).lower()
    is_etf = quote_type == "etf"
    return market_cap, sector, is_etf


def select_benchmark_ticker(ticker: str) -> str:
    try:
        market_cap, sector, is_etf = fetch_ticker_metadata(ticker)
    except Exception as e:
        print(f"Warning: failed to fetch metadata for {ticker}: {e}. Defaulting to SPY.", file=sys.stderr)
        return "SPY"

    if is_etf:
        return "SPY"

    if market_cap is not None and market_cap <= 20_000_000_000:
        return "IWM"

    if sector and "technology" in sector.lower():
        return "QQQ"

    return "SPY"


def analyze_ticker(
    ticker: str,
    benchmark_prices: pd.Series,
    years: int,
    ticker_prices: pd.Series | None = None,
    ticker_volume: pd.Series | None = None,
) -> dict | None:
    """Analyze a single ticker and return results dict, or None on error.

    Args:
        ticker: Ticker symbol
        benchmark_prices: Pre-fetched benchmark price series
        years: Years of history (used only if ticker_prices not provided)
        ticker_prices: Pre-fetched ticker prices (if None, will fetch)
        ticker_volume: Pre-fetched ticker volume (if None, will fetch)
    """
    if ticker_prices is None:
        try:
            ticker_prices, ticker_volume = fetch_prices_yfinance(ticker, years=years)
        except Exception as e:
            print(f"Error fetching {ticker}: {e}", file=sys.stderr)
            return None

    # Align on common dates
    combined = pd.DataFrame({"ticker": ticker_prices, "benchmark": benchmark_prices}).dropna()

    # Need enough data: 63 + 20 = 83 days minimum
    min_points = 63 + 20
    if len(combined) < min_points:
        print(
            f"Not enough data for {ticker}: need at least {min_points} trading days, got {len(combined)}.",
            file=sys.stderr,
        )
        return None

    prices = combined["ticker"]

    # 1. 20-day avg of 63-day ROC (%) - absolute price
    roc63 = (prices / prices.shift(63) - 1.0) * 100.0
    avg20_roc63 = roc63.rolling(window=20, min_periods=20).mean()

    # 2. 20-day avg of 63-day ROC (%) - volume
    avg20_vol_roc63 = None
    if ticker_volume is not None:
        vol = ticker_volume.reindex(combined.index)
        vol = vol.where(vol > 0)
        if vol.notna().sum() >= min_points:
            vol_roc63 = (vol / vol.shift(63) - 1.0) * 100.0
            avg20_vol_roc63_series = vol_roc63.rolling(window=20, min_periods=20).mean()
            valid = avg20_vol_roc63_series.dropna()
            if not valid.empty:
                avg20_vol_roc63 = float(valid.iloc[-1])

    # Relative price calculations
    relative_price = combined["ticker"] / combined["benchmark"]

    # 3. 42-day ROC (%) of relative price
    rel_roc42 = (relative_price / relative_price.shift(42) - 1.0) * 100.0

    # 4. 10-day avg of ROC (%) of relative price
    avg10_rel_roc = rel_roc42.rolling(window=10, min_periods=10).mean()

    return {
        "ticker": ticker,
        "date": combined.index[-1],
        "close": float(prices.iloc[-1]),
        "avg20_roc63": float(avg20_roc63.iloc[-1]),
        "avg20_vol_roc63": avg20_vol_roc63,
        "rel_roc42": float(rel_roc42.iloc[-1]),
        "avg10_rel_roc": float(avg10_rel_roc.iloc[-1]),
    }


def get_data(universe: str = None, years: int = 5) -> dict:
    """Fetch momentum data for GUI consumption.

    Args:
        universe: Universe name or path to file. Defaults to portfolio.csv.
        years: Years of history to download.

    Returns:
        dict with keys:
            results: list of dicts with ticker momentum metrics
            count: number of tickers analyzed
            date: latest data date
            error: error message if failed
    """
    try:
        # Load tickers
        if universe:
            tickers = load_universe(universe)
        else:
            portfolio_path = PORTFOLIO_CSV
            if not portfolio_path.exists():
                return {"error": f"portfolio.csv not found at {portfolio_path}"}
            portfolio_df = pd.read_csv(portfolio_path)
            if "ticker" not in portfolio_df.columns:
                return {"error": "portfolio.csv must have a 'ticker' column"}
            tickers = [t.strip().upper() for t in portfolio_df["ticker"].dropna()]
            if not tickers:
                return {"error": "No tickers found in portfolio.csv"}

        # Fetch all metadata in parallel to determine benchmarks
        metadata_map = fetch_metadata_batch(tickers)
        ticker_to_benchmark = {t: determine_benchmark(metadata_map.get(t, (None, None, False))) for t in tickers}

        # Collect all unique symbols to download (tickers + benchmarks)
        all_benchmarks = set(ticker_to_benchmark.values())
        all_symbols = list(set(tickers) | all_benchmarks)

        # Batch download all prices at once
        prices_map, volumes_map = fetch_prices_batch(all_symbols, years=years)

        # Analyze each ticker using pre-fetched data
        results = []
        for ticker in tickers:
            ticker_prices = prices_map.get(ticker)
            if ticker_prices is None:
                continue

            benchmark_ticker = ticker_to_benchmark[ticker]
            benchmark_prices = prices_map.get(benchmark_ticker)
            if benchmark_prices is None:
                continue

            result = analyze_ticker(ticker, benchmark_prices, years, ticker_prices=ticker_prices, ticker_volume=volumes_map.get(ticker))
            if result:
                result["benchmark"] = benchmark_ticker
                results.append(result)

        if not results:
            return {"error": "No valid results"}

        return {
            "results": results,
            "count": len(results),
            "date": results[0]["date"].date() if results else None,
        }
    except Exception as e:
        return {"error": str(e)}


def main() -> int:
    p = argparse.ArgumentParser(description="Momentum ROC analysis")
    p.add_argument("ticker", nargs="?", help="Single ticker symbol (e.g., AAPL)")
    p.add_argument("--tickers-file", type=str, help="Universe name or path to file containing tickers")
    p.add_argument(
        "--benchmark",
        help="Benchmark ticker (e.g., SPY). If omitted, auto-selects IWM for <= $20B market cap, QQQ for Technology, else SPY.",
    )
    p.add_argument("--years", type=int, default=5, help="Years of history to download (default: 5)")
    p.add_argument("--list-universes", action="store_true", help="List available universe files and exit")
    args = p.parse_args()

    if args.list_universes:
        universes = list_universes()
        print("Available universes:", ", ".join(universes) if universes else "(none)")
        return 0

    # Determine tickers to process
    if args.ticker and args.tickers_file:
        print("Error: specify either a single ticker or --tickers-file, not both", file=sys.stderr)
        return 1
    elif args.ticker:
        tickers = [args.ticker.strip().upper()]
    elif args.tickers_file:
        tickers = load_universe(args.tickers_file)
    else:
        # Default to reading from portfolio.csv
        portfolio_path = PORTFOLIO_CSV
        if not portfolio_path.exists():
            print(f"Error: portfolio.csv not found at {portfolio_path}", file=sys.stderr)
            return 1
        try:
            portfolio_df = pd.read_csv(portfolio_path)
            if "ticker" not in portfolio_df.columns:
                print("Error: portfolio.csv must have a 'ticker' column", file=sys.stderr)
                return 1
            tickers = [t.strip().upper() for t in portfolio_df["ticker"].dropna()]
            if not tickers:
                print("Error: no tickers found in portfolio.csv", file=sys.stderr)
                return 1
            print(f"Using {len(tickers)} tickers from portfolio.csv")
        except Exception as e:
            print(f"Error reading portfolio.csv: {e}", file=sys.stderr)
            return 1

    benchmark_override = args.benchmark.strip().upper() if args.benchmark else None

    # Determine benchmarks for all tickers
    if benchmark_override:
        ticker_to_benchmark = {t: benchmark_override for t in tickers}
    else:
        print(f"Fetching metadata for {len(tickers)} tickers...")
        metadata_map = fetch_metadata_batch(tickers)
        ticker_to_benchmark = {t: determine_benchmark(metadata_map.get(t, (None, None, False))) for t in tickers}

    # Collect all unique symbols to download
    all_benchmarks = set(ticker_to_benchmark.values())
    all_symbols = list(set(tickers) | all_benchmarks)

    # Batch download all prices at once
    print(f"Downloading prices for {len(all_symbols)} symbols...")
    prices_map, volumes_map = fetch_prices_batch(all_symbols, years=args.years)

    # Process each ticker using pre-fetched data
    results = []
    for ticker in tickers:
        ticker_prices = prices_map.get(ticker)
        if ticker_prices is None:
            print(f"No data for {ticker}", file=sys.stderr)
            continue

        benchmark_ticker = ticker_to_benchmark[ticker]
        benchmark_prices = prices_map.get(benchmark_ticker)
        if benchmark_prices is None:
            print(f"No data for benchmark {benchmark_ticker}", file=sys.stderr)
            continue

        result = analyze_ticker(ticker, benchmark_prices, args.years, ticker_prices=ticker_prices, ticker_volume=volumes_map.get(ticker))
        if result:
            result["benchmark"] = benchmark_ticker
            results.append(result)

    if not results:
        print("No valid results.", file=sys.stderr)
        return 1

    # Print results
    if benchmark_override:
        print(f"Benchmark: {benchmark_override}")
        print(f"As of: {results[0]['date'].date().isoformat()}")
        print()

    for r in results:
        if not benchmark_override:
            print(f"Benchmark: {r['benchmark']}")
            print(f"As of: {r['date'].date().isoformat()}")
        print(f"Ticker: {r['ticker']:<6}  Close: {r['close']:.4f}")
        print(f"  20-day avg of 63-day ROC (%):        {colorize(r['avg20_roc63'], 1.5)}")
        vol_str = colorize(r['avg20_vol_roc63'], 0) if r.get('avg20_vol_roc63') is not None else "N/A"
        print(f"  20-day avg of 63-day vol ROC (%):    {vol_str}")
        print(f"  42-day ROC of relative price (%):   {colorize(r['rel_roc42'], 0)}")
        print(f"  10-day avg of relative ROC (%):     {colorize(r['avg10_rel_roc'], 0)}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
