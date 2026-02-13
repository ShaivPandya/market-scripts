#!/usr/bin/env python3
"""
Economic Growth Dashboard - Live Data Version
Fetches real-time data from Yahoo Finance and reads CRB data from XLS file.
Green = Outperforming S&P 500 (for equities), Red = Underperforming S&P 500

Requirements:
    pip install yfinance rich pandas openpyxl

Usage:
    python economic_growth.py [--crb-file path/to/crb.xlsx]
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import warnings

import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

console = Console()

CRB_INDEX_NAME = "CRB Industrial Spot Index"
DEFAULT_YF_PERIOD = "2y"
DEFAULT_CRB_PATH = Path(__file__).resolve().with_name("crb.xlsx")

# ============================================================================
# TICKER DEFINITIONS
# ============================================================================

COMMODITIES = {
    "Copper": "HG=F",                    # COMEX Copper Futures
    "GS Commodity Index": "GSG",         # iShares S&P GSCI Commodity ETF
    CRB_INDEX_NAME: None,                # Read from XLS file
}

EQUITIES = {
    "S&P 500": "SPY",                    # SPDR S&P 500 ETF
    "Russell 2000": "IWM",               # iShares Russell 2000 ETF
    "S&P 600": "IJR",                    # iShares Core S&P Small-Cap ETF
    "DJ Transport": "IYT",               # iShares Transportation Average ETF
    "KBW Banks": "KBWB",                 # Invesco KBW Bank ETF
    "US Retail": "XRT",                  # SPDR S&P Retail ETF
    "US Staples": "XLP",                 # Consumer Staples Select Sector SPDR
    "US Utilities": "XLU",               # Utilities Select Sector SPDR
    "STOXX 600": "^STOXX",               # STOXX Europe 600 Index
    "Europe Banks": "EXV1.DE",           # iShares STOXX Europe 600 Banks UCITS ETF
    "MSCI Korea": "EWY",                 # iShares MSCI South Korea ETF
}

CURRENCIES = {
    "AUD/JPY": "AUDJPY=X",
    "CAD/JPY": "CADJPY=X",
}

# Time periods in calendar days
EQUITY_PERIODS = {
    "1-mo": 30,
    "3-mo": 91,
    "6-mo": 182,
    "1-yr": 365,
}

CURRENCY_PERIODS = {
    "1-mo": 30,
    "3-mo": 91,
    "6-mo": 182,
}

# ============================================================================
# CRB FILE READING FUNCTIONS
# ============================================================================

def read_crb_from_xls(xls_path):
    """
    Read CRB Industrial Spot Index data from Excel file (.xlsx or .xls).
    Returns a DataFrame with date and value columns.
    """
    try:
        df = pd.read_excel(
            xls_path,
            engine="openpyxl",
            header=None,
            skiprows=5,
            usecols=[0, 1],
            names=["date", "value"],
        )

        # Parse and clean data
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"])

        if not df.empty:
            return df.sort_values("date", ignore_index=True)

        console.print("[yellow]Warning: No valid data found in CRB file[/yellow]")
        return None

    except Exception as exc:
        console.print(f"[yellow]Warning: Could not read CRB file: {exc}[/yellow]")
        console.print(f"[yellow]Make sure the file is in .xlsx format and openpyxl is installed[/yellow]")
        return None

def calculate_crb_returns(df, periods):
    """Calculate returns from CRB DataFrame for given periods."""
    if df is None or df.empty:
        return {period: None for period in periods}

    date_index = pd.DatetimeIndex(df["date"])
    values = df["value"].to_numpy()
    latest = values[-1]
    latest_date = date_index[-1]

    returns = {}
    for period_name, days in periods.items():
        target_date = latest_date - timedelta(days=days)
        past_idx = date_index.searchsorted(target_date, side="right") - 1

        if past_idx >= 0:
            past_value = values[past_idx]
            if past_value == 0:
                returns[period_name] = None
                continue
            ret = ((latest - past_value) / past_value) * 100
            returns[period_name] = round(float(ret), 1)
        else:
            returns[period_name] = None

    return returns

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

def download_close_series(ticker_dict, period=DEFAULT_YF_PERIOD):
    """Download close-price series for name->ticker mappings in one yfinance call."""
    valid_tickers = {name: ticker for name, ticker in ticker_dict.items() if ticker}
    if not valid_tickers:
        return {}

    tickers = list(valid_tickers.values())
    try:
        raw = yf.download(
            tickers=tickers if len(tickers) > 1 else tickers[0],
            period=period,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception:
        return {}

    if raw is None or raw.empty:
        return {}

    series_by_name = {}
    is_multi = isinstance(raw.columns, pd.MultiIndex)

    if is_multi:
        available = set(raw.columns.get_level_values(0))
        for name, ticker in valid_tickers.items():
            if ticker not in available:
                continue

            close = raw[ticker].get("Close")
            if close is None:
                continue

            close = close.dropna()
            if not close.empty:
                series_by_name[name] = close.sort_index()
        return series_by_name

    # Single ticker fallback: yfinance returns non-MultiIndex columns.
    close = raw.get("Close")
    if close is None:
        return {}
    close = close.dropna()
    if close.empty:
        return {}

    only_name = next(iter(valid_tickers))
    series_by_name[only_name] = close.sort_index()
    return series_by_name

def _normalize_reference_time(reference_time, index):
    """Align a timestamp with a DatetimeIndex timezone."""
    if index.tz is None:
        if reference_time.tzinfo is None:
            return reference_time
        return reference_time.tz_localize(None)

    if reference_time.tzinfo is None:
        return reference_time.tz_localize(index.tz)
    return reference_time.tz_convert(index.tz)

def calculate_return(close_series, days, reference_time=None):
    """Calculate return over a calendar-day lookback from a close-price series."""
    if close_series is None:
        return None

    close_series = close_series.dropna()
    if close_series.size < 2:
        return None

    if not isinstance(close_series.index, pd.DatetimeIndex):
        return None

    reference_ts = pd.Timestamp.now() if reference_time is None else pd.Timestamp(reference_time)
    reference_ts = _normalize_reference_time(reference_ts, close_series.index)
    target_date = reference_ts - timedelta(days=days)

    past_idx = close_series.index.searchsorted(target_date, side="right") - 1
    past_price = close_series.iloc[past_idx] if past_idx >= 0 else close_series.iloc[0]
    current_price = close_series.iloc[-1]

    if past_price == 0:
        return None

    return float((current_price - past_price) / past_price * 100)

def fetch_all_returns(ticker_dict, periods, category_name, crb_returns=None):
    """Fetch returns for all tickers in a category."""
    results = {}
    reference_time = pd.Timestamp.now()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Fetching {category_name}...", total=len(ticker_dict))
        progress.update(task, description=f"[cyan]Downloading {category_name} quotes...")
        close_series_map = download_close_series(ticker_dict, period=DEFAULT_YF_PERIOD)

        for name, ticker in ticker_dict.items():
            progress.update(task, description=f"[cyan]Calculating {name}...")

            # Handle CRB Industrial Spot Index specially
            if name == CRB_INDEX_NAME and crb_returns is not None:
                results[name] = dict(crb_returns)
                progress.advance(task)
                continue

            if ticker is None:
                results[name] = {period: None for period in periods}
                progress.advance(task)
                continue

            close_series = close_series_map.get(name)
            returns = {}
            for period_name, days in periods.items():
                ret = calculate_return(close_series, days, reference_time=reference_time)
                returns[period_name] = round(ret, 1) if ret is not None else None

            results[name] = returns
            progress.advance(task)

    return results

# ============================================================================
# TABLE CREATION FUNCTIONS
# ============================================================================

def format_return(value, benchmark=None, is_benchmark=False):
    """Format return value with color coding."""
    if value is None:
        return Text("N/A", style="dim")
    
    text = f"{value:+.1f}%"
    
    if benchmark is None or is_benchmark:
        if value >= 0:
            return Text(text, style="green")
        else:
            return Text(text, style="red")
    else:
        if value > benchmark:
            return Text(text, style="bold green")
        elif value < benchmark:
            return Text(text, style="bold red")
        else:
            return Text(text, style="yellow")

def create_commodities_table(results, periods):
    """Create commodities performance table."""
    table = Table(
        title="📊 Commodities Performance",
        show_header=True,
        header_style="bold cyan",
        title_style="bold white",
        border_style="blue"
    )
    table.add_column("Name", style="bold white", min_width=22)
    
    for period in periods.keys():
        table.add_column(period, justify="right", min_width=8)
    
    for name, returns in results.items():
        row = [name]
        for period in periods.keys():
            row.append(format_return(returns.get(period)))
        table.add_row(*row)
    
    return table

def create_equities_table(results, periods):
    """Create equities table with benchmark coloring.
    US equities vs S&P 500, Europe Banks vs STOXX 600.
    """
    table = Table(
        title="📈 Equities Performance (vs Benchmark)",
        show_header=True,
        header_style="bold cyan",
        title_style="bold white",
        border_style="blue"
    )
    table.add_column("Name", style="bold white", min_width=20)
    
    for period in periods.keys():
        table.add_column(period, justify="right", min_width=8)
    
    sp500_returns = results.get("S&P 500", {})
    stoxx600_returns = results.get("STOXX 600", {})
    
    for name, returns in results.items():
        row = [name]
        is_benchmark = (name == "S&P 500" or name == "STOXX 600")
        
        # Europe Banks compared to STOXX 600, others to S&P 500
        if name == "Europe Banks":
            benchmark_returns = stoxx600_returns
        else:
            benchmark_returns = sp500_returns
        
        for period in periods.keys():
            value = returns.get(period)
            benchmark = benchmark_returns.get(period)
            row.append(format_return(value, benchmark, is_benchmark=is_benchmark))
        
        table.add_row(*row)
    
    return table

def create_currencies_table(results, periods):
    """Create currencies performance table."""
    table = Table(
        title="💱 Currency Pair Performance",
        show_header=True,
        header_style="bold cyan",
        title_style="bold white",
        border_style="blue"
    )
    table.add_column("Pair", style="bold white", min_width=12)
    
    for period in periods.keys():
        table.add_column(period, justify="right", min_width=8)
    
    for name, returns in results.items():
        row = [name]
        for period in periods.keys():
            row.append(format_return(returns.get(period)))
        table.add_row(*row)
    
    return table

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def print_header():
    """Print dashboard header."""
    header = Panel(
        f"[bold white]MARKET PERFORMANCE DASHBOARD[/bold white]\n"
        f"[dim]Live Data from Yahoo Finance + CRB from Moody's Analytics[/dim]\n"
        f"[dim]Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        border_style="bold blue",
        padding=(1, 2)
    )
    console.print(header)

def print_legend():
    """Print color legend."""
    console.print("\n[bold]📋 Legend (Equities Table):[/bold]")
    console.print("  [bold green]Green[/bold green] = Outperforming benchmark for that period")
    console.print("  [bold red]Red[/bold red] = Underperforming benchmark for that period")
    console.print("  [dim]Benchmark: S&P 500 for US equities, STOXX 600 for Europe Banks[/dim]")
    console.print("  [green]Benchmark rows[/green] = Colored by positive/negative only")

def print_data_sources(crb_file_used):
    """Print data source information."""
    console.print("\n" + "─" * 60)
    console.print("[dim]Data Sources:[/dim]")
    console.print("[dim]  • Equities/Currencies: Yahoo Finance via yfinance[/dim]")
    if crb_file_used:
        console.print("[dim]  • CRB Industrial Spot Index: Moody's Analytics (Barchart.com)[/dim]")
    console.print("[dim]  • STOXX 600 (^STOXX) = STOXX Europe 600 Index[/dim]")
    console.print("[dim]  • Europe Banks (EXV1.DE) = iShares STOXX Europe 600 Banks UCITS ETF[/dim]")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Market Performance Dashboard')
    parser.add_argument(
        "--crb-file",
        type=Path,
        default=DEFAULT_CRB_PATH,
        help="Path to CRB Excel file from Moody's Analytics (.xls or .xlsx)",
    )
    args = parser.parse_args()
    crb_path = Path(args.crb_file).expanduser()
    
    console.print()
    print_header()
    console.print()
    
    # Try to read CRB data from file
    crb_returns = None
    crb_file_used = False
    
    if crb_path.exists():
        console.print(f"[bold yellow]Reading CRB data from {crb_path}...[/bold yellow]")
        crb_df = read_crb_from_xls(crb_path)
        if crb_df is not None:
            crb_returns = calculate_crb_returns(crb_df, EQUITY_PERIODS)
            crb_file_used = True
            console.print(f"[green]✓ Loaded {len(crb_df)} data points (latest: {crb_df['date'].iloc[-1].date()})[/green]")
        else:
            console.print("[yellow]Could not read CRB file, will show N/A[/yellow]")
    else:
        console.print(f"[yellow]CRB file not found: {crb_path}[/yellow]")
    
    console.print()
    console.print("[bold yellow]Fetching market data...[/bold yellow]\n")
    
    commodities_results = fetch_all_returns(COMMODITIES, EQUITY_PERIODS, "Commodities", crb_returns)
    equities_results = fetch_all_returns(EQUITIES, EQUITY_PERIODS, "Equities")
    currency_results = fetch_all_returns(CURRENCIES, CURRENCY_PERIODS, "Currencies")
    
    # Display tables
    console.print("\n")
    console.print(create_commodities_table(commodities_results, EQUITY_PERIODS))
    
    console.print("\n")
    console.print(create_equities_table(equities_results, EQUITY_PERIODS))
    
    console.print("\n")
    console.print(create_currencies_table(currency_results, CURRENCY_PERIODS))
    
    print_legend()
    print_data_sources(crb_file_used)
    console.print()

def get_data(crb_file: str = None) -> dict:
    """
    Fetch market dashboard data for GUI consumption.

    Args:
        crb_file: Optional path to CRB Excel file

    Returns dict with:
      - commodities: dict of ticker -> {period -> return%}
      - equities: dict of ticker -> {period -> return%}
      - currencies: dict of pair -> {period -> return%}
      - crb_available: bool
      - timestamp: datetime of data fetch
      - benchmarks: dict mapping tickers to their benchmark
    """
    crb_path = Path(crb_file).expanduser() if crb_file else DEFAULT_CRB_PATH

    crb_returns = None
    crb_available = False

    if crb_path.exists():
        crb_df = read_crb_from_xls(crb_path)
        if crb_df is not None:
            crb_returns = calculate_crb_returns(crb_df, EQUITY_PERIODS)
            crb_available = True

    commodities_results = fetch_all_returns(COMMODITIES, EQUITY_PERIODS, "Commodities", crb_returns)
    equities_results = fetch_all_returns(EQUITIES, EQUITY_PERIODS, "Equities")
    currency_results = fetch_all_returns(CURRENCIES, CURRENCY_PERIODS, "Currencies")

    return {
        "commodities": commodities_results,
        "equities": equities_results,
        "currencies": currency_results,
        "crb_available": crb_available,
        "timestamp": datetime.now(),
        "benchmarks": {
            "default": "S&P 500",
            "Europe Banks": "STOXX 600",
        },
        "equity_periods": list(EQUITY_PERIODS.keys()),
        "currency_periods": list(CURRENCY_PERIODS.keys()),
    }


if __name__ == "__main__":
    main()
