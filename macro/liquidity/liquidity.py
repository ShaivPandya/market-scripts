#!/usr/bin/env python3
"""
Liquidity Dashboard
Fetches macro liquidity inputs from FRED, builds a composite liquidity score
tilted toward trend (4w changes in net liquidity/reserves and global CB assets),
classifies regimes, and renders a Rich terminal dashboard. Optional charts are
available with --plot.

Run:
  export FRED_API_KEY=your_key
  python macro/liquidity/liquidity.py
  python macro/liquidity/liquidity.py --plot
"""
import argparse
import os
import sys

import pandas as pd
from fredapi import Fred

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

    # Global central bank balance sheets
    "ecb_assets": "ECBASSETSW",         # ECB total assets, weekly (millions of EUR)
    "boj_assets": "JPNASSETS",          # BoJ total assets, monthly (100 million JPY)

    # Credit / conditions
    "ig_oas": "BAMLC0A0CM",             # ICE BofA US Corporate OAS
    "hy_oas": "BAMLH0A0HYM2",           # ICE BofA HY OAS
    "nfci": "NFCI",

    # Broad money and activity
    "m2": "M2SL",                       # M2 Money Stock, monthly SA
    "gdp": "GDP",                       # Nominal GDP, quarterly SAAR
}

SERIES_META = {
    # FRED units differ across series; align on millions for H.4.1 when needed.
    "on_rrp": {"scale": 1000.0},  # RRPONTSYD is typically in billions; convert to millions
    "boj_assets": {"scale": 100.0},  # JPNASSETS is in 100m JPY; convert to millions
}

COMPONENTS = [
    {"key": "net_liquidity_change_4w", "label": "Net Liquidity (4w change)", "polarity": 1, "weight": 0.35, "value_kind": "billions"},
    {"key": "reserves_change_4w", "label": "Reserve Balances (4w change)", "polarity": 1, "weight": 0.15, "value_kind": "billions"},
    {"key": "ecb_assets_change_4w", "label": "ECB Assets (4w change)", "polarity": 1, "weight": 0.10, "value_kind": "billions"},
    {"key": "boj_assets_change_4w", "label": "BoJ Assets (4w change)", "polarity": 1, "weight": 0.10, "value_kind": "billions"},
    {"key": "ig_oas", "label": "IG OAS", "polarity": -1, "weight": 0.10, "value_kind": "percent"},
    {"key": "hy_oas", "label": "HY OAS", "polarity": -1, "weight": 0.10, "value_kind": "percent"},
    {"key": "nfci", "label": "NFCI", "polarity": -1, "weight": 0.05, "value_kind": "index"},
    {"key": "m2_gdp", "label": "M2 / GDP", "polarity": 1, "weight": 0.05, "value_kind": "ratio"},
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


def build_weekly_panel(df, week_ending="W-WED"):
    df_weekly = df.sort_index().resample(week_ending).last()
    return df_weekly.ffill()


def add_derived_series(df_weekly):
    df = df_weekly.copy()
    df["reserves"] = df["reserve_balances_wavg"]
    df["net_liquidity"] = df["fed_assets_wed_level"] - df["tga_wavg"] - df["on_rrp"]
    df["m2_gdp"] = df["m2"] / df["gdp"]
    df["net_liquidity_change_4w"] = df["net_liquidity"].diff(MOMENTUM_WINDOW_WEEKS)
    df["reserves_change_4w"] = df["reserves"].diff(MOMENTUM_WINDOW_WEEKS)
    df["ecb_assets_change_4w"] = df["ecb_assets"].diff(MOMENTUM_WINDOW_WEEKS)
    df["boj_assets_change_4w"] = df["boj_assets"].diff(MOMENTUM_WINDOW_WEEKS)
    return df


def rolling_zscore(series, window):
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    std = std.replace(0, pd.NA)
    return (series - mean) / std


def compute_component_scores(df):
    z_scores = {}
    contributions = {}
    for comp in COMPONENTS:
        series = df[comp["key"]]
        z = rolling_zscore(series, Z_WINDOW_WEEKS) * comp["polarity"]
        z_scores[comp["key"]] = z
        contributions[comp["key"]] = z * comp["weight"]
    composite = pd.concat(contributions.values(), axis=1).sum(axis=1)
    return composite, z_scores, contributions


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


def build_components_table(df, latest_date, z_scores, contributions):
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

    for comp in COMPONENTS:
        key = comp["key"]
        value = get_value_asof(df[key], latest_date)
        z_value = get_value_asof(z_scores[key], latest_date)
        contribution = get_value_asof(contributions[key], latest_date)
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


def render_dashboard(df, composite, z_scores, contributions):
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
    composite_panel = Panel(composite_text, title="Liquidity Regime", border_style=color)

    components_table = build_components_table(df, latest_date, z_scores, contributions)

    change_items = [
        {"label": "Composite Score", "series": composite, "value_kind": "score"},
        {"label": "Net Liquidity", "series": df["net_liquidity"], "value_kind": "billions"},
        {"label": "Reserve Balances", "series": df["reserves"], "value_kind": "billions"},
        {"label": "IG OAS", "series": df["ig_oas"], "value_kind": "percent", "polarity": -1},
        {"label": "HY OAS", "series": df["hy_oas"], "value_kind": "percent", "polarity": -1},
        {"label": "NFCI", "series": df["nfci"], "value_kind": "index", "polarity": -1},
        {"label": "M2 / GDP", "series": df["m2_gdp"], "value_kind": "ratio"},
        {"label": "ECB Assets (EUR)", "series": df["ecb_assets"], "value_kind": "billions"},
        {"label": "BoJ Assets (JPY)", "series": df["boj_assets"], "value_kind": "billions"},
    ]
    changes_table = build_changes_table(change_items, latest_date)

    console.print(header)
    console.print(composite_panel)
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
    args = parser.parse_args()

    fred = get_fred_client()
    df = fetch_fred_series(fred)
    df = apply_scales(df)
    df_weekly = build_weekly_panel(df)
    df_weekly = add_derived_series(df_weekly)

    composite, z_scores, contributions = compute_component_scores(df_weekly)

    render_dashboard(df_weekly, composite, z_scores, contributions)
    if args.plot:
        plot_charts(df_weekly, composite)


if __name__ == "__main__":
    main()
