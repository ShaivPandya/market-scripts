#!/usr/bin/env python3
"""
Market Performance Dashboard - Live Data Version
Fetches real-time data from Yahoo Finance and reads CRB data from XLS file.
Green = Outperforming S&P 500 (for equities), Red = Underperforming S&P 500

Requirements:
    pip install yfinance rich pandas openpyxl

Usage:
    python market_dashboard_live.py [--crb-file path/to/crb.xls]
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import os
import argparse
import warnings

warnings.filterwarnings('ignore')

console = Console()

# ============================================================================
# TICKER DEFINITIONS
# ============================================================================

COMMODITIES = {
    "Copper": "HG=F",                    # COMEX Copper Futures
    "GS Commodity Index": "GSG",         # iShares S&P GSCI Commodity ETF
    "CRB Spot Index": None,              # Read from XLS file
}

EQUITIES = {
    "S&P 500": "SPY",                    # SPDR S&P 500 ETF
    "Russell 2000": "IWM",               # iShares Russell 2000 ETF
    "S&P 600": "IJR",                    # iShares Core S&P Small-Cap ETF
    "US Staples": "XLP",                 # Consumer Staples Select Sector SPDR
    "DJ Transport": "IYT",               # iShares Transportation Average ETF
    "KBW Banks": "KBWB",                 # Invesco KBW Bank ETF
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
    Read CRB Spot Index data from Excel file (.xlsx or .xls).
    Returns a DataFrame with date and value columns.
    """
    try:
        # Read Excel file (works for both .xlsx and .xls)
        df = pd.read_excel(xls_path, engine='openpyxl', header=None, skiprows=5)

        # Extract first two columns as date and value
        df = df.iloc[:, :2]
        df.columns = ['date', 'value']

        # Parse and clean data
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['date', 'value'])

        if len(df) > 0:
            return df.sort_values('date')

        console.print("[yellow]Warning: No valid data found in CRB file[/yellow]")
        return None

    except Exception as e:
        console.print(f"[yellow]Warning: Could not read CRB file: {e}[/yellow]")
        console.print(f"[yellow]Make sure the file is in .xlsx format and openpyxl is installed[/yellow]")
        return None

def calculate_crb_returns(df, periods):
    """Calculate returns from CRB DataFrame for given periods."""
    if df is None or len(df) == 0:
        return {period: None for period in periods}
    
    latest = df['value'].iloc[-1]
    latest_date = df['date'].iloc[-1]
    
    returns = {}
    for period_name, days in periods.items():
        target_date = latest_date - timedelta(days=days)
        past_df = df[df['date'] <= target_date]
        
        if len(past_df) > 0:
            past_value = past_df['value'].iloc[-1]
            ret = ((latest - past_value) / past_value) * 100
            returns[period_name] = round(ret, 1)
        else:
            returns[period_name] = None
    
    return returns

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

def get_historical_data(ticker, period="2y"):
    """Fetch historical data for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return None
        return hist
    except Exception as e:
        return None

def calculate_return(hist, days):
    """Calculate return over specified number of calendar days."""
    if hist is None or len(hist) < 2:
        return None
    
    try:
        current_price = hist['Close'].iloc[-1]
        target_date = datetime.now() - timedelta(days=days)
        
        if hist.index.tz is not None:
            target_date = target_date.replace(tzinfo=hist.index.tz)
        
        hist_before = hist[hist.index <= pd.Timestamp(target_date)]
        
        if len(hist_before) == 0:
            past_price = hist['Close'].iloc[0]
        else:
            past_price = hist_before['Close'].iloc[-1]
        
        if past_price == 0:
            return None
        
        return ((current_price - past_price) / past_price) * 100
    except Exception as e:
        return None

def fetch_all_returns(ticker_dict, periods, category_name, crb_returns=None):
    """Fetch returns for all tickers in a category."""
    results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Fetching {category_name}...", total=len(ticker_dict))
        
        for name, ticker in ticker_dict.items():
            progress.update(task, description=f"[cyan]Fetching {name}...")
            
            # Handle CRB Spot Index specially
            if name == "CRB Spot Index" and crb_returns is not None:
                results[name] = crb_returns
                progress.advance(task)
                continue
            
            if ticker is None:
                results[name] = {period: None for period in periods}
                progress.advance(task)
                continue
            
            hist = get_historical_data(ticker, period="2y")
            
            returns = {}
            for period_name, days in periods.items():
                ret = calculate_return(hist, days)
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
        console.print("[dim]  • CRB Spot Index: Moody's Analytics (Barchart.com)[/dim]")
    console.print("[dim]  • STOXX 600 (^STOXX) = STOXX Europe 600 Index[/dim]")
    console.print("[dim]  • Europe Banks (EXV1.DE) = iShares STOXX Europe 600 Banks UCITS ETF[/dim]")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    # Default CRB file path is in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_crb_path = os.path.join(script_dir, 'crb.xlsx')

    parser = argparse.ArgumentParser(description='Market Performance Dashboard')
    parser.add_argument('--crb-file', type=str, default=default_crb_path,
                        help='Path to CRB Excel file from Moody\'s Analytics (.xls or .xlsx)')
    args = parser.parse_args()
    
    console.print()
    print_header()
    console.print()
    
    # Try to read CRB data from file
    crb_returns = None
    crb_file_used = False
    
    if os.path.exists(args.crb_file):
        console.print(f"[bold yellow]Reading CRB data from {args.crb_file}...[/bold yellow]")
        crb_df = read_crb_from_xls(args.crb_file)
        if crb_df is not None:
            crb_returns = calculate_crb_returns(crb_df, EQUITY_PERIODS)
            crb_file_used = True
            console.print(f"[green]✓ Loaded {len(crb_df)} data points (latest: {crb_df['date'].iloc[-1].date()})[/green]")
        else:
            console.print("[yellow]Could not read CRB file, will show N/A[/yellow]")
    else:
        console.print(f"[yellow]CRB file not found: {args.crb_file}[/yellow]")
    
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

if __name__ == "__main__":
    main()
