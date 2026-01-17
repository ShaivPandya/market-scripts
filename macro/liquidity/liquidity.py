#!/usr/bin/env python3
"""
Liquidity Dashboard
Fetches macro liquidity inputs from FRED, builds a composite liquidity score
tilted toward trend (4w changes in net liquidity/reserves and global CB assets),
plus net liquidity level for regime context, classifies regimes, and renders a
Rich terminal dashboard. Optional charts are available with --plot.

Run:
  python macro/liquidity/liquidity.py
  python macro/liquidity/liquidity.py --plot
"""
import argparse
import os
import sys
from pathlib import Path

# Load environment variables from .env file
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from load_env import load_env
load_env()

import pandas as pd
from fredapi import Fred

try:
    import sdmx
    SDMX_AVAILABLE = True
except ImportError:
    SDMX_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
except ImportError:
    Console = None

SERIES = {
    # H.4.1 plumbing
    "fed_assets_wed_level": "WALCL",    # Total assets (less eliminations), weekly as-of Wed
    "reserve_balances_wavg": "WRESBAL", # Reserve balances, week avg
    "tga_wavg": "WTREGEN",              # Treasury General Account, week avg
    "on_rrp": "RRPONTSYD",              # ON RRP (Treasuries), daily aggregated

    # Japan central bank
    "boj_assets": "JPNASSETS",          # BoJ total assets, monthly (100 million JPY)
    "jpn_m3_yoy": "JPNMABMM301GYSAM",   # M3 YoY growth (monthly, OECD)
    "jpn_credit_private": "QJPPAM770A", # Credit to private sector (quarterly, BIS)

    # Credit / conditions
    "ig_oas": "BAMLC0A0CM",             # ICE BofA US Corporate OAS
    "hy_oas": "BAMLH0A0HYM2",           # ICE BofA HY OAS
    "nfci": "NFCI",

    # Broad money and activity
    "m2": "M2SL",                       # M2 Money Stock, monthly SA
    "gdp": "GDP",                       # Nominal GDP, quarterly SAAR
}

# ECB SDMX series (fetched separately via sdmx1 library)
ECB_SERIES = {
    "ecb_excess_liquidity": "D.U2.C.EXLIQ.U2.EUR",
    "ecb_net_liquidity_effect": "D.U2.C.NLIQ.U2.EUR",
}

SERIES_META = {
    # FRED units differ across series; align on millions for H.4.1 when needed.
    "on_rrp": {"scale": 1000.0},  # RRPONTSYD is typically in billions; convert to millions
    "boj_assets": {"scale": 100.0},  # JPNASSETS is in 100m JPY; convert to millions
}

# Regional component definitions (weights sum to 1.0 within each region)
US_COMPONENTS = [
    {"key": "net_liquidity_change_4w", "label": "Net Liquidity (4w change)", "polarity": 1, "weight": 0.25, "value_kind": "billions"},
    {"key": "net_liquidity", "label": "Net Liquidity (level)", "polarity": 1, "weight": 0.20, "value_kind": "billions"},
    {"key": "reserves_change_4w", "label": "Reserve Balances (4w change)", "polarity": 1, "weight": 0.20, "value_kind": "billions"},
    {"key": "ig_oas", "label": "IG OAS", "polarity": -1, "weight": 0.15, "value_kind": "percent"},
    {"key": "hy_oas", "label": "HY OAS", "polarity": -1, "weight": 0.10, "value_kind": "percent"},
    {"key": "nfci", "label": "NFCI", "polarity": -1, "weight": 0.05, "value_kind": "index"},
    {"key": "m2_gdp", "label": "M2 / GDP", "polarity": 1, "weight": 0.05, "value_kind": "ratio"},
]

EUROPE_COMPONENTS = [
    {"key": "ecb_excess_liquidity", "label": "Excess Liquidity", "polarity": 1, "weight": 0.60, "value_kind": "billions"},
    {"key": "ecb_net_liquidity_effect", "label": "Net Liquidity Effect", "polarity": 1, "weight": 0.40, "value_kind": "billions"},
]

JAPAN_COMPONENTS = [
    {"key": "boj_assets_yoy", "label": "BoJ Assets YoY", "polarity": 1, "weight": 0.35, "value_kind": "percent"},
    {"key": "jpn_m3_yoy", "label": "M3 YoY", "polarity": 1, "weight": 0.35, "value_kind": "percent"},
    {"key": "jpn_credit_yoy", "label": "Credit to Private YoY", "polarity": 1, "weight": 0.30, "value_kind": "percent"},
]

# Regional weights for aggregate composite score
REGIONAL_WEIGHTS = {
    "us": 0.60,
    "europe": 0.30,
    "japan": 0.10,
}

ALL_REGIONS = [
    ("US", US_COMPONENTS),
    ("Europe", EUROPE_COMPONENTS),
    ("Japan", JAPAN_COMPONENTS),
]

CHANGE_WINDOWS = {
    "1w": 7,
    "1m": 30,
    "3m": 90,
}

Z_WINDOW_WEEKS = 104
MOMENTUM_WINDOW_WEEKS = 4


def get_fred_client():
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("Missing FRED_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)
    return Fred(api_key=api_key)


def fetch_fred_series(fred):
    data = {}
    errors = []
    for name, sid in SERIES.items():
        try:
            series = fred.get_series(sid)
            if series is None or series.empty:
                errors.append(f"{name} ({sid}) returned empty series")
                continue
            data[name] = series
        except Exception as exc:
            errors.append(f"{name} ({sid}) error: {exc}")
    if errors:
        for err in errors:
            print(f"Data fetch error: {err}", file=sys.stderr)
        sys.exit(1)
    df = pd.concat(data, axis=1)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def apply_scales(df):
    df = df.copy()
    for name, meta in SERIES_META.items():
        if name in df.columns:
            df[name] = df[name] * meta.get("scale", 1.0)
    return df


def fetch_ecb_series():
    """
    Fetch ECB excess liquidity and net liquidity effect from ECB SDMX API.
    Returns a DataFrame with daily data, or None if unavailable.
    """
    if not SDMX_AVAILABLE:
        print("sdmx1 library not installed. Skipping ECB data.", file=sys.stderr)
        return None

    try:
        ecb_client = sdmx.Client("ECB")
    except Exception as exc:
        print(f"Could not connect to ECB SDMX API: {exc}", file=sys.stderr)
        return None

    data = {}
    errors = []

    for name, key in ECB_SERIES.items():
        try:
            response = ecb_client.data("ILM", key=key)
            series = sdmx.to_pandas(response)

            if isinstance(series, pd.DataFrame):
                # Flatten multi-index if present
                series = series.iloc[:, 0] if series.shape[1] == 1 else series.stack()

            if series is None or series.empty:
                errors.append(f"{name} ({key}) returned empty series")
                continue

            # Handle multi-index (e.g., tuples) from SDMX response
            if isinstance(series.index, pd.MultiIndex):
                # ECB SDMX data typically has (TIME_PERIOD, ..., CURRENCY) structure
                # We want the TIME_PERIOD which is usually the first level
                time_level = None
                for i, level_name in enumerate(series.index.names):
                    if level_name and 'TIME' in str(level_name).upper():
                        time_level = i
                        break

                if time_level is not None:
                    series.index = series.index.get_level_values(time_level)
                else:
                    # Default to first level if TIME not found
                    series.index = series.index.get_level_values(0)

            # Convert index to datetime
            # Handle various SDMX time period formats
            if not isinstance(series.index, pd.DatetimeIndex):
                try:
                    series.index = pd.to_datetime(series.index)
                except (TypeError, ValueError):
                    # If direct conversion fails, try converting to string first
                    series.index = pd.to_datetime(series.index.astype(str))

            data[name] = series
        except Exception as exc:
            errors.append(f"{name} ({key}) error: {exc}")

    if errors:
        for err in errors:
            print(f"ECB data fetch warning: {err}", file=sys.stderr)

    if not data:
        return None

    df = pd.concat(data, axis=1)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def build_weekly_panel(df, week_ending="W-WED"):
    df_weekly = df.sort_index().resample(week_ending).last()
    return df_weekly.ffill()


def align_series_to_weekly(series, week_ending="W-WED", target_index=None):
    weekly = series.sort_index().resample(week_ending).last()
    weekly = weekly.ffill()
    if target_index is not None:
        weekly = weekly.reindex(target_index).ffill()
    return weekly


def add_derived_series(df_weekly):
    """Add US-specific derived series."""
    df = df_weekly.copy()
    df["reserves"] = df["reserve_balances_wavg"]
    df["net_liquidity"] = df["fed_assets_wed_level"] - df["tga_wavg"] - df["on_rrp"]
    df["m2_gdp"] = df["m2"] / df["gdp"]
    df["net_liquidity_change_4w"] = df["net_liquidity"].diff(MOMENTUM_WINDOW_WEEKS)
    df["reserves_change_4w"] = df["reserves"].diff(MOMENTUM_WINDOW_WEEKS)
    return df


def add_japan_derived_series(df_weekly, df_raw, week_ending="W-WED"):
    """Add Japan-specific derived series."""
    df = df_weekly.copy()

    # BoJ Assets YoY (12-month pct_change on monthly data)
    boj_assets = df_raw["boj_assets"].ffill()
    boj_yoy = boj_assets.pct_change(12, fill_method=None) * 100
    df["boj_assets_yoy"] = align_series_to_weekly(
        boj_yoy, week_ending=week_ending, target_index=df.index
    )

    # M3 YoY - already provided as YoY from FRED, just align to weekly
    df["jpn_m3_yoy"] = align_series_to_weekly(
        df_raw["jpn_m3_yoy"], week_ending=week_ending, target_index=df.index
    )

    # Credit to private sector YoY (4-quarter pct_change on quarterly data)
    credit_private = df_raw["jpn_credit_private"].ffill()
    credit_yoy = credit_private.pct_change(4, fill_method=None) * 100
    df["jpn_credit_yoy"] = align_series_to_weekly(
        credit_yoy, week_ending=week_ending, target_index=df.index
    )

    return df


def add_europe_derived_series(df_weekly, df_ecb, week_ending="W-WED"):
    """Add Europe-specific derived series from ECB SDMX data."""
    df = df_weekly.copy()

    if df_ecb is None:
        # Graceful degradation: set to NaN if ECB data unavailable
        df["ecb_excess_liquidity"] = pd.NA
        df["ecb_net_liquidity_effect"] = pd.NA
        return df

    # ECB data is daily - resample to weekly (week-ending Wednesday)
    for col in ["ecb_excess_liquidity", "ecb_net_liquidity_effect"]:
        if col in df_ecb.columns:
            weekly = df_ecb[col].resample(week_ending).last().ffill()
            df[col] = weekly.reindex(df.index).ffill()
        else:
            df[col] = pd.NA

    return df


def rolling_zscore(series, window):
    """
    Calculate rolling z-score with adaptive window.
    If the full window isn't available, uses the maximum available data
    with a minimum of 13 weeks (1 quarter) for statistical validity.
    """
    # Use a minimum of 13 weeks for statistical validity
    min_periods = min(13, window)

    # Calculate available data points
    available = series.notna().sum()

    # Adjust window to available data, but respect minimum
    effective_window = min(window, max(available, min_periods))

    mean = series.rolling(window=effective_window, min_periods=min_periods).mean()
    std = series.rolling(window=effective_window, min_periods=min_periods).std()
    std = std.replace(0, pd.NA)
    return (series - mean) / std


def compute_regional_scores(df):
    """
    Compute separate scores for each region and an aggregate composite.
    Returns: (composite, regional_scores, all_z_scores, all_contributions)
    """
    regional_scores = {}
    all_z_scores = {}
    all_contributions = {}

    regions = {
        "us": US_COMPONENTS,
        "europe": EUROPE_COMPONENTS,
        "japan": JAPAN_COMPONENTS,
    }

    for region_name, components in regions.items():
        contributions = {}

        for comp in components:
            key = comp["key"]
            if key not in df.columns:
                continue

            series = df[key]
            if series.isna().all():
                continue

            z = rolling_zscore(series, Z_WINDOW_WEEKS) * comp["polarity"]
            all_z_scores[key] = z
            contributions[key] = z * comp["weight"]
            all_contributions[key] = contributions[key]

        if contributions:
            regional_score = pd.concat(contributions.values(), axis=1).sum(axis=1)
        else:
            regional_score = pd.Series(index=df.index, dtype=float)

        regional_scores[region_name] = regional_score

    # Compute aggregate composite from regional scores
    composite_parts = []
    for region_name, weight in REGIONAL_WEIGHTS.items():
        if region_name in regional_scores and not regional_scores[region_name].empty:
            composite_parts.append(regional_scores[region_name] * weight)

    if composite_parts:
        composite = pd.concat(composite_parts, axis=1).sum(axis=1)
    else:
        composite = pd.Series(index=df.index, dtype=float)

    return composite, regional_scores, all_z_scores, all_contributions


def classify_regime(score):
    if score > 1.0:
        return "ample", "green"
    if score >= -0.5:
        return "normal", "cyan"
    if score >= -1.5:
        return "tight", "yellow"
    return "stress", "red"


def get_value_asof(series, date):
    series = series.dropna()
    if series.empty:
        return None
    asof = series.loc[:date]
    if asof.empty:
        return None
    return asof.iloc[-1]


def change_over_days(series, date, days):
    current = get_value_asof(series, date)
    if current is None:
        return None
    past = get_value_asof(series, date - pd.Timedelta(days=days))
    if past is None:
        return None
    return current - past


def format_value(value, kind, signed=False):
    if value is None or pd.isna(value):
        return "N/A"
    sign = "+" if signed and value >= 0 else ""
    if kind == "billions":
        return f"{sign}{value / 1000.0:.2f}B"
    if kind == "percent":
        return f"{sign}{value:.2f}%"
    if kind == "ratio":
        return f"{sign}{value:.3f}"
    if kind == "index":
        return f"{sign}{value:.2f}"
    if kind == "score":
        return f"{value:+.2f}"
    return f"{sign}{value:.2f}"


def format_score(value):
    if value is None or pd.isna(value):
        return Text("N/A", style="dim")
    text = f"{value:+.2f}"
    if value >= 1:
        style = "green"
    elif value <= -1:
        style = "red"
    else:
        style = "yellow"
    return Text(text, style=style)


def format_signal(value):
    if value is None or pd.isna(value):
        return Text("n/a", style="dim")
    if value >= 0:
        return Text("supportive", style="green")
    return Text("tightening", style="red")


def build_regional_scores_panel(regional_scores, latest_date):
    """Build a panel showing regional liquidity scores."""
    table = Table(
        title="Regional Liquidity Scores",
        show_header=True,
        header_style="bold cyan",
        title_style="bold white",
        border_style="blue",
    )
    table.add_column("Region", style="bold white")
    table.add_column("Score", justify="right")
    table.add_column("Regime", justify="right")
    table.add_column("Weight", justify="right")

    region_labels = {
        "us": "United States",
        "europe": "Europe",
        "japan": "Japan",
    }

    for region_key, label in region_labels.items():
        score_series = regional_scores.get(region_key)
        if score_series is None or score_series.empty:
            score = None
        else:
            score = get_value_asof(score_series, latest_date)

        if score is None or pd.isna(score):
            score_text = Text("N/A", style="dim")
            regime_text = Text("N/A", style="dim")
        else:
            regime, color = classify_regime(score)
            score_text = Text(f"{score:+.2f}", style=f"bold {color}")
            regime_text = Text(regime.upper(), style=f"bold {color}")

        weight = REGIONAL_WEIGHTS.get(region_key, 0)
        table.add_row(label, score_text, regime_text, f"{weight * 100:.0f}%")

    return table


def build_components_table(df, latest_date, z_scores, contributions):
    """Build components table grouped by region."""
    table = Table(
        title="Liquidity Components",
        show_header=True,
        header_style="bold cyan",
        title_style="bold white",
        border_style="blue",
    )
    table.add_column("Component", style="bold white")
    table.add_column("Value", justify="right")
    table.add_column("Z", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Contribution", justify="right")
    table.add_column("Signal", justify="right")

    for region_name, components in ALL_REGIONS:
        # Add region header row
        table.add_row(
            Text(f"── {region_name} ──", style="bold magenta"),
            "", "", "", "", ""
        )

        for comp in components:
            key = comp["key"]
            if key not in df.columns:
                continue

            value = get_value_asof(df[key], latest_date)
            z_value = get_value_asof(z_scores.get(key, pd.Series()), latest_date)
            contribution = get_value_asof(contributions.get(key, pd.Series()), latest_date)

            table.add_row(
                comp["label"],
                format_value(value, comp["value_kind"]),
                format_score(z_value),
                f"{comp['weight'] * 100:.0f}%",
                format_score(contribution),
                format_signal(z_value),
            )

    return table


def build_changes_table(changes, latest_date):
    table = Table(
        title="Historical Changes",
        show_header=True,
        header_style="bold cyan",
        title_style="bold white",
        border_style="blue",
    )
    table.add_column("Series", style="bold white")
    for label in CHANGE_WINDOWS.keys():
        table.add_column(label, justify="right")

    for item in changes:
        row = [item["label"]]
        polarity = item.get("polarity", 1)
        for _, days in CHANGE_WINDOWS.items():
            delta = change_over_days(item["series"], latest_date, days)
            if delta is None or pd.isna(delta):
                row.append(Text("N/A", style="dim"))
            else:
                text = format_value(delta, item["value_kind"], signed=True)
                effective_delta = delta * polarity
                style = "green" if effective_delta > 0 else "red" if effective_delta < 0 else "yellow"
                row.append(Text(text, style=style))
        table.add_row(*row)
    return table


def render_dashboard(df, composite, regional_scores, z_scores, contributions):
    if Console is None:
        print("rich is not installed. Install with: pip install rich", file=sys.stderr)
        sys.exit(1)

    composite_clean = composite.dropna()
    if composite_clean.empty:
        print("Not enough data to compute composite score.", file=sys.stderr)
        sys.exit(1)

    latest_date = composite_clean.index[-1]
    latest_score = composite_clean.iloc[-1]
    regime, color = classify_regime(latest_score)

    console = Console()

    header_text = Text.assemble(
        ("Liquidity Dashboard", "bold white"),
        ("\n", ""),
        (f"Last update: {latest_date.date().isoformat()}", "dim"),
    )
    header = Panel(header_text, border_style="blue")

    composite_text = Text.assemble(
        ("Composite Score: ", "bold white"),
        (f"{latest_score:+.2f}", f"bold {color}"),
        ("  Regime: ", "bold white"),
        (regime.upper(), f"bold {color}"),
    )
    composite_panel = Panel(composite_text, title="Aggregate Liquidity Regime", border_style=color)

    regional_panel = build_regional_scores_panel(regional_scores, latest_date)
    components_table = build_components_table(df, latest_date, z_scores, contributions)

    change_items = [
        {"label": "Composite Score", "series": composite, "value_kind": "score"},
        {"label": "Net Liquidity", "series": df["net_liquidity"], "value_kind": "billions"},
        {"label": "Reserve Balances", "series": df["reserves"], "value_kind": "billions"},
        {"label": "IG OAS", "series": df["ig_oas"], "value_kind": "percent", "polarity": -1},
        {"label": "HY OAS", "series": df["hy_oas"], "value_kind": "percent", "polarity": -1},
        {"label": "NFCI", "series": df["nfci"], "value_kind": "index", "polarity": -1},
        {"label": "M2 / GDP", "series": df["m2_gdp"], "value_kind": "ratio"},
        {"label": "BoJ Assets (JPY)", "series": df["boj_assets"], "value_kind": "billions"},
    ]
    changes_table = build_changes_table(change_items, latest_date)

    console.print(header)
    console.print(composite_panel)
    console.print(regional_panel)
    console.print(components_table)
    console.print(changes_table)


def plot_charts(df, composite):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Install with: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    composite_clean = composite.dropna()
    if composite_clean.empty:
        print("Not enough data to plot composite score.", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    ax = axes[0]
    ax.plot(composite_clean.index, composite_clean.values, color="black", label="Composite")
    ax.axhspan(1.0, composite_clean.max(), color="green", alpha=0.1)
    ax.axhspan(-0.5, 1.0, color="cyan", alpha=0.1)
    ax.axhspan(-1.5, -0.5, color="yellow", alpha=0.1)
    ax.axhspan(composite_clean.min(), -1.5, color="red", alpha=0.1)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Composite Liquidity Score")
    ax.set_ylabel("Z-score")

    ax = axes[1]
    ax.plot(df.index, df["net_liquidity"] / 1000.0, color="blue", label="Net Liquidity (B)")
    ax.set_title("Net Liquidity")
    ax.set_ylabel("Billions")

    ax = axes[2]
    ax.plot(df.index, df["ig_oas"], color="purple", label="IG OAS")
    ax.plot(df.index, df["hy_oas"], color="orange", label="HY OAS")
    ax.set_title("Credit Spreads")
    ax.set_ylabel("Percent")
    ax.legend(loc="upper left")

    ax = axes[3]
    ax.plot(df.index, df["nfci"], color="brown", label="NFCI")
    ax2 = ax.twinx()
    ax2.plot(df.index, df["m2_gdp"], color="green", label="M2 / GDP")
    ax.set_title("Financial Conditions and Money")
    ax.set_ylabel("NFCI")
    ax2.set_ylabel("M2 / GDP")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Liquidity monitoring dashboard.")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib charts")
    parser.add_argument("--no-ecb", action="store_true", help="Skip ECB SDMX data fetch")
    args = parser.parse_args()

    # Fetch FRED data (includes Japan series)
    fred = get_fred_client()
    df = fetch_fred_series(fred)
    df = apply_scales(df)

    # Fetch ECB SDMX data (optional, graceful degradation)
    df_ecb = None
    if not args.no_ecb:
        df_ecb = fetch_ecb_series()

    # Build weekly panel and add derived series
    week_ending = "W-WED"
    df_weekly = build_weekly_panel(df, week_ending=week_ending)
    df_weekly = add_derived_series(df_weekly)
    df_weekly = add_japan_derived_series(df_weekly, df, week_ending=week_ending)
    df_weekly = add_europe_derived_series(df_weekly, df_ecb, week_ending=week_ending)

    # Compute regional and composite scores
    composite, regional_scores, z_scores, contributions = compute_regional_scores(df_weekly)

    render_dashboard(df_weekly, composite, regional_scores, z_scores, contributions)
    if args.plot:
        plot_charts(df_weekly, composite)


if __name__ == "__main__":
    main()
