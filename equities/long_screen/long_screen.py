#!/usr/bin/env python3
"""
Long Screen: Identify potential long candidates from a stock universe.

Screening criteria (all enabled filters must be met to pass):
  1. P/B ratio below threshold (undervalued)
  2. Gross profit OR operating profit (profitable companies)
  3. (Optional) Min YoY revenue growth across last 3 quarters
  4. (Optional) Min YoY EPS growth across last 3 quarters
  5. (Optional) Net equity issuance in the bottom quartile (buyback-heavy) among Phase 1 passers
  6. (Optional) Price-based filters: 52w return, drawdown, positive momentum

Execution is phased:
  Phase 1 — parallel yfinance fetch, filter by P/B + profitability + growth
  Phase 2 — sequential SEC EDGAR calls ONLY for Phase 1 passers (buyback filter)
  Phase 3 — batch price download for price/momentum filters
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from equities.short_screen.short_screen import (
    PHASE1_WORKERS,
    YF_BATCH_DELAY,
    YF_CHUNK_SIZE,
    _build_result_row,
    fetch_sec_issuance,
    fetch_yf_data,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1: Screen one ticker (inverted criteria)
# ---------------------------------------------------------------------------


def screen_ticker_long(
    ticker: str,
    pb_threshold: float | None,
    profit_type: str | None,
    check_revenue: bool = False,
    min_revenue_growth: float = 5.0,
    check_eps: bool = False,
    min_eps_growth: float = 5.0,
) -> tuple[bool, dict]:
    """
    Apply Phase 1 criteria using yfinance data (inverted from short screen).

    Returns:
        (passes: bool, data: dict)
    """
    data = fetch_yf_data(ticker)

    if "error" in data:
        return False, data

    pb = data.get("price_to_book", np.nan)
    gross = data.get("gross_profit", np.nan)
    operating = data.get("operating_income", np.nan)

    # P/B must be positive AND below threshold (undervalued)
    if pb_threshold is not None:
        pb_ok = (not (isinstance(pb, float) and np.isnan(pb))) and pb > 0 and pb < pb_threshold
    else:
        pb_ok = True

    # Profitability check (inverted: require profit, not loss)
    if profit_type is None:
        profit_ok = True
    elif profit_type == "Gross Profit":
        profit_ok = (not (isinstance(gross, float) and np.isnan(gross))) and (gross > 0)
    else:
        profit_ok = (not (isinstance(operating, float) and np.isnan(operating))) and (operating > 0)

    # Revenue growth filter: avg YoY growth must be >= threshold
    if check_revenue:
        avg = data.get("rev_yoy_avg", np.nan)
        rev_ok = not (isinstance(avg, float) and np.isnan(avg)) and avg >= min_revenue_growth
    else:
        rev_ok = True

    # EPS growth filter: avg YoY growth must be >= threshold
    if check_eps:
        avg = data.get("eps_yoy_avg", np.nan)
        eps_ok = not (isinstance(avg, float) and np.isnan(avg)) and avg >= min_eps_growth
    else:
        eps_ok = True

    return (pb_ok and profit_ok and rev_ok and eps_ok), data


# ---------------------------------------------------------------------------
# Phase 3: Price-based filters (positive momentum)
# ---------------------------------------------------------------------------


def _apply_price_filters_long(
    passers: list[dict],
    *,
    check_52w_positive: bool,
    check_min_drawdown: bool,
    min_drawdown_pct: float,
    check_max_drawdown: bool,
    max_drawdown_pct: float,
    check_3m_pos_momentum: bool,
    check_2m_pos_rel_momentum: bool,
    benchmark_ticker: str,
) -> tuple[list[dict], dict[str, dict]]:
    """
    Apply optional price-based filters to Phase 1/2 passers.

    Momentum filters are inverted from the short screen:
    - 3m positive momentum: keep stocks with return_3m > 0
    - 2m positive relative momentum: keep stocks outperforming benchmark
    """
    from utils.retry import yf_download

    passer_tickers = [d["ticker"] for d in passers]

    download_tickers = list(passer_tickers)
    need_benchmark = check_2m_pos_rel_momentum and benchmark_ticker
    if need_benchmark and benchmark_ticker not in download_tickers:
        download_tickers.append(benchmark_ticker)

    chunks = [download_tickers[i : i + YF_CHUNK_SIZE] for i in range(0, len(download_tickers), YF_CHUNK_SIZE)]
    all_dfs: list[pd.DataFrame] = []
    for i, chunk in enumerate(chunks):
        try:
            df = yf_download(
                chunk,
                period="1y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if df is not None and not df.empty:
                all_dfs.append(df)
        except Exception:
            LOGGER.warning("Price filter batch %d/%d failed", i + 1, len(chunks), exc_info=True)
        if i < len(chunks) - 1:
            time.sleep(YF_BATCH_DELAY)

    if not all_dfs:
        LOGGER.warning("All price filter downloads failed; skipping price filters")
        return passers, {}

    if len(all_dfs) == 1:
        raw = all_dfs[0]
    else:
        parts: dict[str, list[pd.DataFrame]] = {}
        for df in all_dfs:
            if isinstance(df.columns, pd.MultiIndex):
                for level in df.columns.get_level_values(0).unique():
                    parts.setdefault(str(level), []).append(df[level])
            else:
                for col in df.columns:
                    parts.setdefault(str(col), []).append(df[[col]])
        raw = pd.concat({k: pd.concat(v, axis=1) for k, v in parts.items() if v}, axis=1)

    def _get_close(df: pd.DataFrame, ticker: str) -> pd.Series | None:
        if df is None or df.empty:
            return None
        try:
            if isinstance(df.columns, pd.MultiIndex):
                if ticker in df.columns.get_level_values(1):
                    s = df[("Close", ticker)].dropna()
                else:
                    return None
            else:
                s = df["Close"].dropna()
            return s if len(s) > 0 else None
        except (KeyError, TypeError):
            return None

    bench_close: pd.Series | None = None
    if need_benchmark:
        bench_close = _get_close(raw, benchmark_ticker)

    filtered: list[dict] = []
    metrics: dict[str, dict] = {}

    for data in passers:
        tk = data["ticker"]
        close = _get_close(raw, tk)
        if close is None or len(close) < 10:
            continue

        current = float(close.iloc[-1])
        m: dict[str, float | None] = {}

        # 52-week return
        ret_52w: float | None = None
        if len(close) >= 200:
            price_52w = float(close.iloc[0])
            ret_52w = (current / price_52w - 1) * 100
        m["return_52w"] = ret_52w

        # Drawdown from 52-week high
        peak = float(close.max())
        dd_pct = (current - peak) / peak * 100 if peak > 0 else None
        m["drawdown_pct"] = dd_pct

        # 3-month return (~63 trading days)
        ret_3m: float | None = None
        if len(close) >= 63:
            price_3m = float(close.iloc[-63])
            ret_3m = (current / price_3m - 1) * 100
        m["return_3m"] = ret_3m

        # 2-month relative return (~42 trading days)
        rel_ret_2m: float | None = None
        if len(close) >= 42 and bench_close is not None and len(bench_close) >= 42:
            price_2m = float(close.iloc[-42])
            stock_ret = (current / price_2m - 1) * 100
            bench_current = float(bench_close.iloc[-1])
            bench_2m = float(bench_close.iloc[-42])
            bench_ret = (bench_current / bench_2m - 1) * 100
            rel_ret_2m = stock_ret - bench_ret
        m["rel_return_2m"] = rel_ret_2m

        # Apply filters
        passes = True

        if check_52w_positive:
            if ret_52w is None or ret_52w <= 0:
                passes = False

        if check_min_drawdown and passes:
            if dd_pct is None or abs(dd_pct) < min_drawdown_pct:
                passes = False

        if check_max_drawdown and passes:
            if dd_pct is None or abs(dd_pct) > max_drawdown_pct:
                passes = False

        # INVERTED: require positive 3m momentum (short screen requires negative)
        if check_3m_pos_momentum and passes:
            if ret_3m is None or ret_3m <= 0:
                passes = False

        # INVERTED: require positive relative momentum (short screen requires negative)
        if check_2m_pos_rel_momentum and passes:
            if rel_ret_2m is None or rel_ret_2m <= 0:
                passes = False

        if passes:
            filtered.append(data)
            metrics[tk] = m

    return filtered, metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def get_data(
    tickers: list[str],
    pb_threshold: float | None = 1.5,
    profit_type: str | None = "Gross Profit",
    check_issuance: bool = False,
    check_revenue: bool = False,
    min_revenue_growth: float = 5.0,
    check_eps: bool = False,
    min_eps_growth: float = 5.0,
    check_52w_positive: bool = False,
    check_min_drawdown: bool = False,
    min_drawdown_pct: float = 25.0,
    check_max_drawdown: bool = False,
    max_drawdown_pct: float = 60.0,
    check_3m_pos_momentum: bool = False,
    check_2m_pos_rel_momentum: bool = False,
    benchmark_ticker: str = "IWM",
    progress_callback=None,
) -> dict:
    """
    Run the long screen over the provided ticker universe.

    Returns on success:
        {
            "results_df":          pd.DataFrame
            "failed_tickers":      List[str]
            "phase1_count":        int
            "phase1_pass_count":   int
            "phase3_pass_count":   int  (if price filters enabled)
            "final_count":         int
        }

    Returns on hard failure:
        {"error": str}
    """
    import yfinance as yf

    universe = tickers

    if not universe:
        return {"error": "No tickers provided"}

    total = len(universe)

    # Pre-warm yfinance session
    try:
        yf.Ticker(universe[0]).fast_info.last_price  # noqa: B018
    except Exception:
        LOGGER.debug("yfinance session pre-warm failed", exc_info=True)

    # ------------------------------------------------------------------
    # Phase 1: Batched parallel yfinance fetch + P/B + profit filter
    # ------------------------------------------------------------------
    phase1_pass_data: list[dict] = []
    failed_tickers: list[str] = []
    done_count = 0
    batches = [universe[i : i + YF_CHUNK_SIZE] for i in range(0, total, YF_CHUNK_SIZE)]

    with ThreadPoolExecutor(max_workers=PHASE1_WORKERS) as pool:
        for batch_idx, batch in enumerate(batches):
            futures = {
                pool.submit(
                    screen_ticker_long,
                    tk,
                    pb_threshold,
                    profit_type,
                    check_revenue=check_revenue,
                    min_revenue_growth=min_revenue_growth,
                    check_eps=check_eps,
                    min_eps_growth=min_eps_growth,
                ): tk
                for tk in batch
            }
            for future in as_completed(futures):
                tk = futures[future]
                try:
                    passes, data = future.result()
                    if passes:
                        phase1_pass_data.append(data)
                    elif "error" in data:
                        failed_tickers.append(tk)
                except Exception:
                    failed_tickers.append(tk)

                done_count += 1
                if progress_callback and (done_count % 25 == 0 or done_count == total):
                    progress_callback(done_count, total)

            if batch_idx < len(batches) - 1:
                time.sleep(YF_BATCH_DELAY)

    phase1_pass_count = len(phase1_pass_data)

    if not phase1_pass_data:
        return {
            "results_df": pd.DataFrame(),
            "failed_tickers": failed_tickers,
            "phase1_count": total,
            "phase1_pass_count": 0,
            "final_count": 0,
        }

    # ------------------------------------------------------------------
    # Phase 2 (optional): SEC EDGAR issuance — bottom quartile (buybacks)
    # ------------------------------------------------------------------
    issuance_info: dict[str, dict] = {}

    if not check_issuance:
        phase2_pass_data = list(phase1_pass_data)
    else:
        phase2_pass_data = []
        issuance_records: list[dict] = []
        for data in phase1_pass_data:
            sec = fetch_sec_issuance(data["ticker"])
            if "error" in sec:
                continue

            net = sec.get("net_issuance", np.nan)
            mktcap = data.get("market_cap", np.nan)

            if (
                (isinstance(net, float) and np.isnan(net))
                or (isinstance(mktcap, float) and np.isnan(mktcap))
                or mktcap <= 0
            ):
                continue

            issuance_records.append({"data": data, "net": net, "pct": net / mktcap})

        if issuance_records:
            net_values = [r["net"] for r in issuance_records]
            # INVERTED: bottom quartile (low/negative net issuance = buybacks)
            cutoff = float(np.percentile(net_values, 25))
            for rec in issuance_records:
                if rec["net"] <= cutoff:
                    phase2_pass_data.append(rec["data"])
                    issuance_info[rec["data"]["ticker"]] = {
                        "net": rec["net"],
                        "pct": rec["pct"],
                    }

    # ------------------------------------------------------------------
    # Phase 3 (optional): Price-based filters
    # ------------------------------------------------------------------
    any_price_filter = (
        check_52w_positive
        or check_min_drawdown
        or check_max_drawdown
        or check_3m_pos_momentum
        or check_2m_pos_rel_momentum
    )

    price_metrics: dict[str, dict] = {}
    phase3_pass_count: int | None = None

    if any_price_filter and phase2_pass_data:
        phase2_pass_data, price_metrics = _apply_price_filters_long(
            phase2_pass_data,
            check_52w_positive=check_52w_positive,
            check_min_drawdown=check_min_drawdown,
            min_drawdown_pct=min_drawdown_pct,
            check_max_drawdown=check_max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            check_3m_pos_momentum=check_3m_pos_momentum,
            check_2m_pos_rel_momentum=check_2m_pos_rel_momentum,
            benchmark_ticker=benchmark_ticker,
        )
        phase3_pass_count = len(phase2_pass_data)

    # ------------------------------------------------------------------
    # Build final result rows
    # ------------------------------------------------------------------
    final_rows: list[dict] = []
    for data in phase2_pass_data:
        tk = data["ticker"]
        pm = price_metrics.get(tk) if any_price_filter else None
        row = _build_result_row(data, price_metrics=pm)
        if tk in issuance_info:
            row["Net Issuance ($M)"] = round(issuance_info[tk]["net"] / 1e6, 1)
            row["Issuance % Mkt Cap"] = round(issuance_info[tk]["pct"] * 100, 1)
        final_rows.append(row)

    results_df = pd.DataFrame(final_rows)

    if not results_df.empty:
        # Sort by P/B ascending (cheapest first)
        results_df = results_df.sort_values("P/B Ratio", ascending=True).reset_index(drop=True)

    result = {
        "results_df": results_df,
        "failed_tickers": failed_tickers,
        "phase1_count": total,
        "phase1_pass_count": phase1_pass_count,
        "final_count": len(results_df),
    }
    if phase3_pass_count is not None:
        result["phase3_pass_count"] = phase3_pass_count
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    from equities.common import load_universe

    parser = argparse.ArgumentParser(description="Long Screen")
    parser.add_argument(
        "universe", nargs="?", default="russell2000", help="Universe: sp500, russell2000, sp400, xlk, etc."
    )
    parser.add_argument("--pb", type=float, default=1.5, help="P/B threshold (default 1.5)")
    parser.add_argument(
        "--profit",
        choices=["gross", "operating"],
        default="gross",
        help="Profit type: gross (default) or operating",
    )
    parser.add_argument(
        "--issuance",
        action="store_true",
        help="Keep only bottom-quartile net equity issuers (buyback-heavy) among screened stocks",
    )
    parser.add_argument(
        "--check-revenue", action="store_true", help="Filter by min YoY revenue growth (avg of last 3 quarters)"
    )
    parser.add_argument("--min-rev-growth", type=float, default=5.0, help="Min YoY revenue growth %% (default 5)")
    parser.add_argument("--check-eps", action="store_true", help="Filter by min avg YoY EPS growth (last 3 quarters)")
    parser.add_argument("--min-eps-growth", type=float, default=5.0, help="Min avg YoY EPS growth %% (default 5)")
    args = parser.parse_args()

    tickers = load_universe(args.universe)
    if not tickers:
        print(f"ERROR: Failed to load universe '{args.universe}'")
        return

    profit_type = "Gross Profit" if args.profit == "gross" else "Operating Profit"

    def cb(done, total):
        print(f"\rPhase 1: {done}/{total}", end="", flush=True)

    print(
        f"Running long screen: {args.universe} ({len(tickers)} tickers), P/B < {args.pb}, {profit_type}"
        + (", buyback-heavy" if args.issuance else "")
    )

    result = get_data(
        tickers=tickers,
        pb_threshold=args.pb,
        profit_type=profit_type,
        check_issuance=args.issuance,
        check_revenue=args.check_revenue,
        min_revenue_growth=args.min_rev_growth,
        check_eps=args.check_eps,
        min_eps_growth=args.min_eps_growth,
        progress_callback=cb,
    )
    print()

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    print(f"\nUniverse: {result['phase1_count']} tickers")
    print(f"Phase 1 pass: {result['phase1_pass_count']}")
    if "phase3_pass_count" in result:
        print(f"Phase 3 pass: {result['phase3_pass_count']}")
    print(f"Final candidates: {result['final_count']}")
    print(f"Data errors: {len(result['failed_tickers'])}")

    df = result["results_df"]
    if df.empty:
        print("No candidates found.")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
