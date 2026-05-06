#!/usr/bin/env python3
"""
Portfolio Dashboard

Fetches closing-price time series for portfolio positions from Yahoo Finance.
Reads holdings from the runtime portfolio state (ticker, asset class, direction).
Displays a grid of line charts with Daily/Weekly/Monthly toggles (GUI),
or prints summary tables in the terminal.

Terminal:
  python portfolio/portfolio_dashboard.py
"""

import logging
import warnings
from datetime import datetime

import pandas as pd

from ontology.runtime_read_service import get_positions, get_positions_df
from portfolio.portfolio_analytics import compute_analytics
from utils.retry import yf_download

LOGGER = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


def _load_portfolio() -> pd.DataFrame:
    """Load portfolio positions from the runtime read adapter."""
    return get_positions_df()


def _build_globals(df: pd.DataFrame) -> tuple[dict, list, dict]:
    positions = {row.ticker: getattr(row, "price_symbol", None) or row.ticker for row in df.itertuples()}
    order = list(df.ticker)
    meta = {
        row.ticker: {
            "asset": row.asset,
            "direction": row.direction,
            "instrument_type": getattr(row, "instrument_type", "security"),
            "price_symbol": getattr(row, "price_symbol", row.ticker),
            "quantity": getattr(row, "quantity", getattr(row, "shares", None)),
            "shares": getattr(row, "quantity", getattr(row, "shares", None)),
            "contract_multiplier": getattr(row, "contract_multiplier", 1.0),
            "currency": getattr(row, "currency", None),
            "country": getattr(row, "country", None),
            "exchange": getattr(row, "exchange", None),
            "base_currency": getattr(row, "base_currency", "USD"),
            "fx_rate_to_base": getattr(row, "fx_rate_to_base", None),
            "fx_rate_as_of": getattr(row, "fx_rate_as_of", None),
            "cost_basis_base": getattr(row, "cost_basis_base", None),
            "notional_base": getattr(row, "notional_base", None),
            "valuation_status": getattr(row, "valuation_status", None),
            "role": getattr(row, "role", "position"),
        }
        for row in df.itertuples()
    }
    return positions, order, meta


_portfolio_df = _load_portfolio()
POSITIONS, POSITION_ORDER, POSITION_META = _build_globals(_portfolio_df)


def reload_portfolio() -> None:
    """Re-read positions from the database and update module-level globals."""
    global _portfolio_df, POSITIONS, POSITION_ORDER, POSITION_META
    _portfolio_df = _load_portfolio()
    POSITIONS, POSITION_ORDER, POSITION_META = _build_globals(_portfolio_df)


# -- Timeframe configs: name -> yfinance (period, interval) ──────────────────
TIMEFRAMES = {
    "This Week": {"period": "5d", "interval": "15m"},
    "Daily": {"period": "90d", "interval": "1d"},
    "Weekly": {"period": "2y", "interval": "1wk"},
    "Monthly": {"period": "5y", "interval": "1mo"},
}


# -- Data fetching ────────────────────────────────────────────────────────────


def _empty_payload(timeframe: str, warning: str | None = None) -> dict:
    """Return a valid dashboard payload when live prices are unavailable."""
    positions: dict[str, pd.Series] = {}
    payload = {
        "positions": positions,
        "metadata": POSITION_META,
        "timeframe": timeframe,
        "timestamp": datetime.now(),
        "position_order": POSITION_ORDER,
        "analytics": compute_analytics(positions, get_positions()),
    }
    if warning:
        payload["warning"] = warning
    return payload


def fetch_portfolio_data(timeframe: str = "Daily") -> dict:
    """
    Fetch closing-price time series for all portfolio positions.

    Returns dict with:
        positions – dict[ticker] -> pd.Series (Close prices)
        metadata  – dict[ticker] -> {asset, direction}
        timeframe – str
        timestamp – datetime
        error     – str (only on failure)
    """
    tf = TIMEFRAMES.get(timeframe)
    if tf is None:
        return {"error": f"Invalid timeframe: {timeframe}"}

    tickers = list(dict.fromkeys(POSITIONS.values()))
    if not tickers:
        return _empty_payload(timeframe, "No portfolio positions configured.")

    try:
        raw = yf_download(
            tickers=tickers,
            period=tf["period"],
            interval=tf["interval"],
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as e:
        LOGGER.warning("Portfolio yfinance download failed for %s: %s", timeframe, e)
        return _empty_payload(timeframe, f"yfinance download failed: {e}")

    if raw is None or raw.empty:
        LOGGER.warning("Portfolio yfinance download returned no rows for %s", timeframe)
        return _empty_payload(timeframe, "No data returned from yfinance.")

    is_multi = isinstance(raw.columns, pd.MultiIndex)
    positions = {}

    for ticker in POSITION_ORDER:
        price_symbol = POSITIONS.get(ticker, ticker)
        try:
            if is_multi:
                if price_symbol not in raw.columns.get_level_values(0):
                    continue
                series = raw[price_symbol]["Close"].dropna()
            else:
                series = raw["Close"].dropna()

            if series.empty:
                continue

            if hasattr(series.index, "tz") and series.index.tz is not None:
                series.index = series.index.tz_localize(None)

            positions[ticker] = series
        except Exception:
            continue

    holdings = get_positions()
    analytics = compute_analytics(positions, holdings)
    metadata = {ticker: dict(meta) for ticker, meta in POSITION_META.items()}
    per_position = analytics.get("per_position", {}) if isinstance(analytics, dict) else {}
    for ticker, metrics in per_position.items():
        if ticker in metadata and isinstance(metrics, dict):
            metadata[ticker]["current_notional"] = metrics.get("current_notional")
            metadata[ticker]["cost_notional"] = metrics.get("cost_notional")

    warnings_out: list[str] = []
    if any(str(row.get("instrument_type") or "").lower() == "future" for row in holdings):
        warnings_out.append("Continuous futures use front/active contract pricing; roll P&L is not modeled.")

    payload = {
        "positions": positions,
        "metadata": metadata,
        "timeframe": timeframe,
        "timestamp": datetime.now(),
        "position_order": POSITION_ORDER,
        "analytics": analytics,
    }
    if warnings_out:
        payload["warning"] = "; ".join(warnings_out)
    return payload


def fetch_all_timeframes_data() -> dict:
    """Fetch portfolio data for all supported timeframes."""
    results = {}
    analytics = None
    warnings_by_timeframe = {}
    for tf_name in TIMEFRAMES:
        data = fetch_portfolio_data(timeframe=tf_name)
        if "error" in data and data["error"]:
            return data
        results[tf_name] = data
        if data.get("warning"):
            warnings_by_timeframe[tf_name] = data["warning"]
        # Use Weekly (2y) analytics for top-level — good 52-week coverage
        if tf_name == "Weekly":
            analytics = data.get("analytics")
    payload = {
        "timeframes": results,
        "timestamp": datetime.now(),
        "analytics": analytics,
    }
    if warnings_by_timeframe:
        unique_warnings = list(dict.fromkeys(warnings_by_timeframe.values()))
        payload["warning"] = "; ".join(unique_warnings)
    return payload


def get_data(timeframe: str = "Daily", all_timeframes: bool = False) -> dict:
    """GUI-facing entry point."""
    reload_portfolio()
    if all_timeframes:
        return fetch_all_timeframes_data()
    return fetch_portfolio_data(timeframe=timeframe)


def format_price(value: float) -> str:
    """Format price based on magnitude for clean display."""
    if abs(value) >= 100:
        return f"{value:,.2f}"
    return f"{value:.4f}"


# -- Terminal output ──────────────────────────────────────────────────────────


def print_terminal():
    """Print Portfolio dashboard results for all timeframes."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    header = Panel(
        "[bold white]PORTFOLIO DASHBOARD[/bold white]\n"
        f"[dim]Data from Yahoo Finance[/dim]\n"
        f"[dim]Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        border_style="bold blue",
        padding=(1, 2),
    )
    console.print(header)

    for tf_name in TIMEFRAMES:
        console.print(f"\n[bold yellow]Fetching {tf_name} data...[/bold yellow]")
        data = fetch_portfolio_data(timeframe=tf_name)

        if "error" in data:
            console.print(f"[red]Error: {data['error']}[/red]")
            continue

        positions = data.get("positions", {})
        if not positions:
            console.print("[yellow]No data returned[/yellow]")
            continue

        analytics = data.get("analytics", {})
        per_pos = analytics.get("per_position", {})

        table = Table(
            title=f"Portfolio Dashboard — {tf_name}",
            show_header=True,
            header_style="bold cyan",
            title_style="bold white",
            border_style="blue",
        )
        table.add_column("Ticker", style="bold white", min_width=8)
        table.add_column("Dir", min_width=5)
        table.add_column("Asset", min_width=7)
        table.add_column("Price", justify="right", min_width=10)
        table.add_column("Cost Basis", justify="right", min_width=10)
        table.add_column("PnL %", justify="right", min_width=8)
        table.add_column("52w DD", justify="right", min_width=8)
        table.add_column("Wk Ret%", justify="right", min_width=8)
        table.add_column("Wk Attr%", justify="right", min_width=8)

        for ticker in POSITION_ORDER:
            series = positions.get(ticker)
            meta = POSITION_META.get(ticker, {})
            direction = meta.get("direction", "").upper()
            asset = meta.get("asset", "")
            a = per_pos.get(ticker, {})

            dir_style = "green" if direction == "LONG" else "red"

            if series is None or series.empty:
                table.add_row(ticker, f"[{dir_style}]{direction}[/{dir_style}]", asset, "N/A", "", "", "", "", "")
                continue

            latest = series.iloc[-1]
            cb = a.get("cost_basis")
            pnl = a.get("unrealized_pnl_pct")
            dd = a.get("drawdown_from_52w_pct")
            wk = a.get("weekly_return_pct")
            wk_attr = a.get("weekly_contribution_pct")

            pnl_str = f"[{'green' if pnl >= 0 else 'red'}]{pnl:+.1f}%[/]" if pnl is not None else "—"
            dd_str = f"[{'green' if dd == 0 else 'red'}]{dd:+.1f}%[/]" if dd is not None else "—"
            wk_str = f"[{'green' if wk >= 0 else 'red'}]{wk:+.1f}%[/]" if wk is not None else "—"
            attr_str = f"[{'green' if wk_attr >= 0 else 'red'}]{wk_attr:+.2f}%[/]" if wk_attr is not None else "—"

            table.add_row(
                ticker,
                f"[{dir_style}]{direction}[/{dir_style}]",
                asset,
                format_price(latest),
                format_price(cb) if cb is not None else "—",
                pnl_str,
                dd_str,
                wk_str,
                attr_str,
            )

        # Summary row
        port = analytics.get("portfolio", {})
        total_pnl = port.get("total_unrealized_pnl_pct")
        wk_port = port.get("weekly_portfolio_return_pct")
        n_prof = port.get("positions_profitable", 0)
        n_loss = port.get("positions_losing", 0)

        summary_parts = []
        if total_pnl is not None:
            c = "green" if total_pnl >= 0 else "red"
            summary_parts.append(f"Avg PnL: [{c}]{total_pnl:+.1f}%[/{c}]")
        summary_parts.append(f"W/L: {n_prof}/{n_loss}")
        if wk_port is not None:
            c = "green" if wk_port >= 0 else "red"
            summary_parts.append(f"Wk Return: [{c}]{wk_port:+.2f}%[/{c}]")

        if summary_parts:
            table.add_section()
            table.add_row("PORTFOLIO", "", "", "", "", *["" for _ in range(4)])
            console.print(table)
            console.print(f"  {'  |  '.join(summary_parts)}")
        else:
            console.print(table)

    console.print()


def main():
    print_terminal()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    LOGGER.info("Starting script execution: %s", __file__)
    main()
