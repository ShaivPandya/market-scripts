"""
Government Bond Yields Tracker

Fetches government bond yields (2-year, 10-year, 30-year) for multiple countries
and calculates the change in yields over various time periods (1 month, 3 months, 6 months, 1 year).

Countries covered:
- United States
- United Kingdom
- Germany
- Japan
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import sys
import os
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    print("Warning: fredapi not installed. Install with: pip install fredapi", file=sys.stderr)
    print("FRED API key also required. Get one at: https://fred.stlouisfed.org/docs/api/api_key.html", file=sys.stderr)

# FRED API configuration
# Set your FRED API key as an environment variable: export FRED_API_KEY='your_key_here'
import os
FRED_API_KEY = os.environ.get('FRED_API_KEY', None)

# FRED series IDs for US Treasury yields
FRED_SERIES = {
    "United States": {
        "2-Year": "DGS2",   # 2-Year Treasury Constant Maturity Rate
        "10-Year": "DGS10", # 10-Year Treasury Constant Maturity Rate
        "30-Year": "DGS30"  # 30-Year Treasury Constant Maturity Rate
    }
}

# Data files for other countries (stored locally)
# File format: "Download Data - BOND_BX_XTUP_TMBMK{COUNTRY_CODE}-{MATURITY}.csv"
BOND_DATA_FILES = {
    "United Kingdom": {
        "2-Year": "Download Data - BOND_BX_XTUP_TMBMKGB-02Y.csv",
        "10-Year": "Download Data - BOND_BX_XTUP_TMBMKGB-10Y.csv"
    },
    "Germany": {
        "2-Year": "Download Data - BOND_BX_XTUP_TMBMKDE-02Y.csv",
        "10-Year": "Download Data - BOND_BX_XTUP_TMBMKDE-10Y.csv"
    },
    "Japan": {
        "2-Year": "Download Data - BOND_BX_XTUP_TMBMKJP-02Y.csv",
        "10-Year": "Download Data - BOND_BX_XTUP_TMBMKJP-10Y.csv"
    }
}

# Data directory path
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def get_fred_data(series_id: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch historical bond yield data from FRED.

    Args:
        series_id: FRED series ID
        start_date: Start date for historical data
        end_date: End date for historical data

    Returns:
        DataFrame with historical yield data
    """
    if not FRED_AVAILABLE or not FRED_API_KEY:
        print(f"FRED API not available for {series_id}", file=sys.stderr)
        return pd.DataFrame()

    try:
        fred = Fred(api_key=FRED_API_KEY)
        data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)

        # Convert to DataFrame format similar to yfinance
        df = pd.DataFrame({'Close': data})
        return df
    except Exception as e:
        print(f"Error fetching FRED data for {series_id}: {str(e)}", file=sys.stderr)
        return pd.DataFrame()


def load_bond_data_from_csv(filename: str, end_date: datetime) -> pd.DataFrame:
    """
    Load historical bond yield data from a CSV file.

    Args:
        filename: CSV filename in the data directory
        end_date: End date for historical data

    Returns:
        DataFrame with historical yield data (all available data up to end_date)
    """
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        print(f"Error: Data file not found: {filepath}", file=sys.stderr)
        return pd.DataFrame()

    try:
        # Read CSV file
        df = pd.read_csv(filepath)

        # Convert Date column to datetime (format: MM/DD/YYYY)
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df = df.set_index('Date')

        # Sort by date (ascending order) - CSV files may have dates in descending order
        df = df.sort_index()

        # Remove percentage signs and convert Close column to float
        if 'Close' in df.columns:
            df['Close'] = df['Close'].str.rstrip('%').astype('float')
        elif 'Yield' in df.columns:
            df['Close'] = df['Yield'].str.rstrip('%').astype('float')

        # Only filter by end date - we want all historical data available
        # for calculating changes over various time periods
        df = df[df.index <= end_date]

        return df
    except Exception as e:
        print(f"Error loading data from {filename}: {str(e)}", file=sys.stderr)
        return pd.DataFrame()


def get_current_yield_from_csv(filename: str) -> float:
    """
    Get the current yield from a CSV file.

    Args:
        filename: CSV filename in the data directory

    Returns:
        Current yield value (most recent date)
    """
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        print(f"Error: Data file not found: {filepath}", file=sys.stderr)
        return None

    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df = df.sort_values('Date')

        # Get the most recent yield (remove % sign and convert to float)
        if 'Close' in df.columns and not df.empty:
            close_value = df['Close'].iloc[-1]
            if isinstance(close_value, str):
                return float(close_value.rstrip('%'))
            return float(close_value)
        elif 'Yield' in df.columns and not df.empty:
            yield_value = df['Yield'].iloc[-1]
            if isinstance(yield_value, str):
                return float(yield_value.rstrip('%'))
            return float(yield_value)

        return None
    except Exception as e:
        print(f"Error loading current yield from {filename}: {str(e)}", file=sys.stderr)
        return None


def calculate_yield_changes(current_yield: float, historical_data: pd.DataFrame,
                           periods: List[Tuple[str, int]]) -> Dict[str, float]:
    """
    Calculate yield changes over specified time periods.

    Args:
        current_yield: Current bond yield
        historical_data: Historical yield data
        periods: List of (period_name, days_back) tuples

    Returns:
        Dictionary with yield changes for each period
    """
    changes = {}
    today = datetime.now()

    for period_name, days_back in periods:
        target_date = today - timedelta(days=days_back)

        # Find the closest available date
        if not historical_data.empty:
            historical_data.index = pd.to_datetime(historical_data.index)

            # Convert target_date to timezone-aware if the index is timezone-aware
            if historical_data.index.tz is not None:
                # Make target_date timezone-aware using the same timezone as the index
                target_date = pd.Timestamp(target_date).tz_localize(historical_data.index.tz)

            closest_data = historical_data[historical_data.index <= target_date]

            if not closest_data.empty:
                historical_yield = closest_data.iloc[-1]['Close']
                change = current_yield - historical_yield
                changes[period_name] = round(change, 4)
            else:
                changes[period_name] = None
        else:
            changes[period_name] = None

    return changes


def get_current_yield_fred(series_id: str) -> float:
    """
    Get the current yield from FRED.

    Args:
        series_id: FRED series ID

    Returns:
        Current yield value
    """
    if not FRED_AVAILABLE or not FRED_API_KEY:
        return None

    try:
        fred = Fred(api_key=FRED_API_KEY)
        # Get the most recent observation
        data = fred.get_series(series_id)
        if not data.empty:
            return data.iloc[-1]
        return None
    except Exception as e:
        print(f"Error fetching current FRED yield for {series_id}: {str(e)}", file=sys.stderr)
        return None


def format_yield_change(change: float) -> str:
    """Format yield change with appropriate sign and color coding.

    Green: Negative change (yields trending lower)
    Red: Positive change (yields trending higher)
    """
    if change is None:
        return "N/A"

    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

    sign = "+" if change >= 0 else ""
    formatted_value = f"{sign}{change:.4f}"

    # Green for negative (yields down), red for positive (yields up)
    if change < 0:
        return f"{GREEN}{formatted_value}{RESET}"
    elif change > 0:
        return f"{RED}{formatted_value}{RESET}"
    else:
        return formatted_value  # No color for zero change


def main():
    """Main function to fetch and display government bond yields."""

    print("=" * 100)
    print("GOVERNMENT BOND YIELDS TRACKER")
    print(f"Data as of: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    print()

    if not FRED_API_KEY:
        print("\nWARNING: FRED_API_KEY not set. US Treasury data will not be available.")
        print("To get US data, set environment variable: export FRED_API_KEY='your_key_here'")
        print("Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html\n")

    # Define time periods for comparison (name, days back)
    periods = [
        ("1 Month", 30),
        ("3 Months", 90),
        ("6 Months", 180),
        ("1 Year", 365)
    ]

    # Fetch data going back 1 year + buffer
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)

    # Process United States (using FRED)
    if FRED_AVAILABLE and FRED_API_KEY:
        print(f"\n{'=' * 100}")
        print("UNITED STATES")
        print(f"{'=' * 100}")

        for maturity, series_id in FRED_SERIES["United States"].items():
            print(f"\n{maturity} Treasury ({series_id}):")
            print("-" * 100)

            # Get current yield from FRED
            current_yield = get_current_yield_fred(series_id)

            if current_yield is None:
                print(f"  Current Yield: N/A (Data not available)")
                continue

            print(f"  Current Yield: {current_yield:.4f}%")

            # Get historical data from FRED
            historical_data = get_fred_data(series_id, start_date, end_date)

            # Calculate changes
            changes = calculate_yield_changes(current_yield, historical_data, periods)

            print(f"\n  Changes:")
            for period_name, _ in periods:
                change = changes.get(period_name)
                formatted_change = format_yield_change(change)
                print(f"    {period_name:12s}: {formatted_change:>10s} bps" if change is not None else f"    {period_name:12s}: N/A")

        print()

    # Process other countries (using local CSV files)
    for country, bonds in BOND_DATA_FILES.items():
        print(f"\n{'=' * 100}")
        print(f"{country.upper()}")
        print(f"{'=' * 100}")

        for maturity, filename in bonds.items():
            print(f"\n{maturity} Bond ({filename}):")
            print("-" * 100)

            # Get current yield from CSV
            current_yield = get_current_yield_from_csv(filename)

            if current_yield is None:
                print(f"  Current Yield: N/A (Data not available)")
                continue

            print(f"  Current Yield: {current_yield:.4f}%")

            # Get historical data from CSV
            historical_data = load_bond_data_from_csv(filename, end_date)

            # Calculate changes
            changes = calculate_yield_changes(current_yield, historical_data, periods)

            print(f"\n  Changes:")
            for period_name, _ in periods:
                change = changes.get(period_name)
                formatted_change = format_yield_change(change)
                print(f"    {period_name:12s}: {formatted_change:>10s} bps" if change is not None else f"    {period_name:12s}: N/A")

        print()

    print("=" * 100)
    print("\nNotes:")
    print("- Yields are expressed as percentages")
    print("- Changes are expressed in basis points (bps)")
    print("- US data source: FRED (Federal Reserve Economic Data)")
    print("- Other countries data source: Local CSV files in the data/ directory")
    print("- CSV files should have 'Date' and 'Yield' columns")
    print("=" * 100)


def export_to_csv(filename: str = "government_bond_yields.csv"):
    """
    Export bond yield data to a CSV file.

    Args:
        filename: Output CSV filename
    """
    periods = [
        ("1 Month", 30),
        ("3 Months", 90),
        ("6 Months", 180),
        ("1 Year", 365)
    ]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)

    results = []

    # Export US data from FRED
    if FRED_AVAILABLE and FRED_API_KEY:
        for maturity, series_id in FRED_SERIES["United States"].items():
            current_yield = get_current_yield_fred(series_id)

            if current_yield is None:
                continue

            historical_data = get_fred_data(series_id, start_date, end_date)
            changes = calculate_yield_changes(current_yield, historical_data, periods)

            row = {
                "Country": "United States",
                "Maturity": maturity,
                "Ticker": series_id,
                "Current Yield (%)": f"{current_yield:.4f}",
                "1 Month Change (bps)": changes.get("1 Month"),
                "3 Months Change (bps)": changes.get("3 Months"),
                "6 Months Change (bps)": changes.get("6 Months"),
                "1 Year Change (bps)": changes.get("1 Year"),
            }
            results.append(row)

    # Export other countries from local CSV files
    for country, bonds in BOND_DATA_FILES.items():
        for maturity, filename in bonds.items():
            current_yield = get_current_yield_from_csv(filename)

            if current_yield is None:
                continue

            historical_data = load_bond_data_from_csv(filename, end_date)
            changes = calculate_yield_changes(current_yield, historical_data, periods)

            row = {
                "Country": country,
                "Maturity": maturity,
                "Data File": filename,
                "Current Yield (%)": f"{current_yield:.4f}",
                "1 Month Change (bps)": changes.get("1 Month"),
                "3 Months Change (bps)": changes.get("3 Months"),
                "6 Months Change (bps)": changes.get("6 Months"),
                "1 Year Change (bps)": changes.get("1 Year"),
            }
            results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\nData exported to {filename}")


if __name__ == "__main__":
    # Check if export flag is provided
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        main()
        export_to_csv()
    else:
        main()
