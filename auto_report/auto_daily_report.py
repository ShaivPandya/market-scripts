#!/usr/bin/env python3
"""
Automated daily portfolio risk report.

Collects per-position risk metrics, calls Claude for risk analysis,
runs the portfolio sizer deterministically, computes share adjustments
vs open_positions.csv, and creates a GitHub Issue.

Run:
    python auto_report/auto_daily_report.py --force   # bypass weekday-afternoon gate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# sys.path — same pattern as auto_weekly_report.py
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

_PATHS = [
    PROJECT_ROOT,
    PROJECT_ROOT / "portfolio",
    PROJECT_ROOT / "portfolio" / "portfolio_optimizer",
    PROJECT_ROOT / "portfolio" / "technical_analysis",
    PROJECT_ROOT / "portfolio" / "momentum" / "price_momentum",
    PROJECT_ROOT / "portfolio" / "momentum" / "fundamental_momentum",
    PROJECT_ROOT / "equities" / "quality",
    PROJECT_ROOT / "equities",
]
for _p in reversed(_PATHS):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from auto_report.shared import (  # noqa: E402
    call_claude,
    create_github_issue,
    load_prompt_file,
    serialize_bundle,
    strip_llm_meta,
    write_bundle,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("auto_daily_report")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ET = ZoneInfo("America/New_York")
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "daily"
HISTORY_DIR = OUTPUT_DIR / "history"
PROMPTS_DIR = SCRIPT_DIR / "prompts"
PORTFOLIO_CSV = PROJECT_ROOT / "portfolio" / "portfolio.csv"
OPEN_POSITIONS_CSV = PROJECT_ROOT / "portfolio" / "open_positions.csv"
DAILY_SUMMARY_SEPARATOR = "<!-- DAILY_SUMMARY_JSON -->"


# ---------------------------------------------------------------------------
# Schedule gate
# ---------------------------------------------------------------------------


def _is_weekday_morning_et() -> bool:
    """True if it's a weekday at 09:xx ET."""
    now_et = datetime.now(ET)
    return now_et.weekday() < 5 and now_et.hour == 9


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_portfolio():
    """Load portfolio.csv and return a pandas DataFrame."""
    import pandas as pd

    df = pd.read_csv(PORTFOLIO_CSV)
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df["direction"] = df["direction"].fillna("").str.strip().str.lower()
    df["conviction"] = df["conviction"].fillna(3).astype(int)
    return df


def load_open_positions():
    """Load open_positions.csv (IBKR export). Returns empty DataFrame if missing."""
    import pandas as pd

    if not OPEN_POSITIONS_CSV.exists():
        log.warning("open_positions.csv not found at %s", OPEN_POSITIONS_CSV)
        return pd.DataFrame()
    df = pd.read_csv(OPEN_POSITIONS_CSV)
    df.columns = df.columns.str.strip()
    return df


# ---------------------------------------------------------------------------
# Risk data collection
# ---------------------------------------------------------------------------


def collect_risk_data(portfolio_df) -> dict:
    """Collect per-position risk metrics from all risk modules."""
    import pandas as pd

    results = {}
    tickers = list(portfolio_df["ticker"])
    asset_map = dict(zip(portfolio_df["ticker"], portfolio_df["asset"]))

    # 1. Technical analysis (per-ticker MA signals + ROC)
    try:
        from technical_analysis import get_data as get_ta_data

        t0 = time.perf_counter()
        ta_results = {}
        for ticker in tickers:
            try:
                ta = get_ta_data(ticker, lookback="2Y")
                ta_results[ticker] = ta.get("summary", ta)
            except Exception as exc:
                ta_results[ticker] = {"error": str(exc)}
        log.info("technical analysis fetched in %.2fs", time.perf_counter() - t0)
        results["technical_analysis"] = ta_results
    except Exception as e:
        log.warning("technical analysis fetch failed: %s", e, exc_info=True)
        results["technical_analysis"] = {"error": str(e)}

    # 2. Price momentum (batch — uses portfolio.csv)
    try:
        from momentum import get_data as get_momentum_data

        t0 = time.perf_counter()
        momentum = get_momentum_data()
        log.info("price momentum fetched in %.2fs", time.perf_counter() - t0)
        results["price_momentum"] = momentum
    except Exception as e:
        log.warning("price momentum fetch failed: %s", e, exc_info=True)
        results["price_momentum"] = {"error": str(e)}

    # 3. Portfolio risk metrics (volatility, drawdown, beta)
    try:
        from portfolio_analyzer import (
            MARKET_TICKER_LONG,
            MARKET_TICKER_SHORT,
            compute_beta_frame,
            compute_defense_volatility,
            compute_distressed_metrics,
            compute_severe_drawdown_flags,
            download_prices,
            fetch_currencies,
            get_required_fx_tickers,
            to_usd_price,
        )

        t0 = time.perf_counter()
        market_tickers = [MARKET_TICKER_LONG, MARKET_TICKER_SHORT]
        all_tickers = list(set(tickers + market_tickers))
        ticker_currencies = fetch_currencies(all_tickers)
        fx_tickers = get_required_fx_tickers(ticker_currencies)
        prices_all = download_prices(all_tickers, fx_tickers)

        usd_prices = pd.DataFrame(index=prices_all.index)
        for t in all_tickers:
            local_px = prices_all[t]
            ccy = ticker_currencies.get(t, "USD")
            usd_prices[t] = to_usd_price(local_px, ccy, prices_all)
        usd_prices = usd_prices.ffill()
        rets = usd_prices.pct_change(fill_method=None).dropna(how="all")

        # Defense volatility
        defense_vol = compute_defense_volatility(usd_prices, tickers)
        results["defense_volatility"] = defense_vol.to_dict()

        # Severe drawdown flags
        equity_tickers = [
            t for t in tickers if asset_map.get(t, "").lower() == "equity"
        ]
        severe_dd = compute_severe_drawdown_flags(usd_prices, equity_tickers)
        results["severe_drawdown_flags"] = severe_dd

        # Distressed metrics
        distressed_tickers = list(
            portfolio_df.loc[
                portfolio_df["distressed"]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"]),
                "ticker",
            ]
        )
        if distressed_tickers:
            distressed_metrics = compute_distressed_metrics(
                prices_all, distressed_tickers
            )
            results["distressed_metrics"] = distressed_metrics.to_dict(orient="index")
        else:
            results["distressed_metrics"] = {}

        # Beta frame
        valid_tickers = [t for t in tickers if t in rets.columns]
        beta_frame, _, _ = compute_beta_frame(rets, valid_tickers)
        results["beta_frame"] = beta_frame.to_dict(orient="index")

        log.info(
            "portfolio risk metrics fetched in %.2fs", time.perf_counter() - t0
        )
    except Exception as e:
        log.warning("portfolio risk metrics fetch failed: %s", e, exc_info=True)
        results["portfolio_risk"] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Portfolio sizer (deterministic, no AI)
# ---------------------------------------------------------------------------


def run_sizer(portfolio_df, book: float) -> dict:
    """Run portfolio sizer and return full result dict."""
    from portfolio_sizer import size_portfolio

    positions = [
        {"ticker": row["ticker"], "conviction": int(row["conviction"])}
        for _, row in portfolio_df.iterrows()
        if row["direction"] in ("long", "short")
    ]
    return size_portfolio(positions=positions, book=book, target_leverage=2.0)


# ---------------------------------------------------------------------------
# Share adjustment computation (deterministic)
# ---------------------------------------------------------------------------


def compute_adjustments(sizer_result: dict, open_positions_df) -> "pd.DataFrame":
    """Compare sizer target shares to current open positions.

    Returns DataFrame with columns: ticker, direction, target_shares,
    current_shares, delta, action, price, dollar_value.
    """
    import pandas as pd

    weights_df = sizer_result.get("weights_df")
    hedges_df = sizer_result.get("hedges_df")

    if weights_df is None:
        return pd.DataFrame()

    # Build target map from sizer output
    target: dict[str, dict] = {}
    for _, row in weights_df.iterrows():
        target[row["ticker"]] = {
            "target_shares": int(row.get("shares", 0)),
            "price": float(row.get("price", 0)),
            "direction": str(row.get("direction", "")),
        }
    if hedges_df is not None:
        for _, row in hedges_df.iterrows():
            target[row["ticker"]] = {
                "target_shares": int(row.get("shares", 0)),
                "price": float(row.get("price", 0)),
                "direction": "hedge",
            }

    # Build current holdings map from open_positions.csv
    current: dict[str, int] = {}
    if not open_positions_df.empty:
        for _, row in open_positions_df.iterrows():
            symbol = str(row.get("Symbol", "")).strip().upper()
            qty = int(row.get("Quantity", 0))
            side = str(row.get("Side", "")).strip().lower()
            if side == "short":
                qty = -abs(qty)
            current[symbol] = qty

    all_tickers = sorted(set(list(target.keys()) + list(current.keys())))

    rows = []
    for ticker in all_tickers:
        t = target.get(ticker, {})
        target_shares = t.get("target_shares", 0)
        price = t.get("price", 0)
        direction = t.get("direction", "closed")
        current_shares = current.get(ticker, 0)
        delta = target_shares - current_shares

        if ticker in current and ticker not in target:
            action = "CLOSE"
            target_shares = 0
            delta = -current_shares
        elif delta > 0:
            action = "BUY"
        elif delta < 0:
            action = "SELL"
        else:
            action = "HOLD"

        rows.append(
            {
                "ticker": ticker,
                "direction": direction,
                "target_shares": target_shares,
                "current_shares": current_shares,
                "delta": delta,
                "action": action,
                "price": round(price, 2),
                "dollar_value": round(delta * price, 2),
            }
        )

    return pd.DataFrame(rows)


def format_adjustments_markdown(adj_df) -> str:
    """Format the adjustments DataFrame as a Markdown table."""
    if adj_df.empty:
        return "## Share Adjustments\n\n_No adjustments computed._\n"

    dir_map = {"long": "L", "short": "S", "hedge": "H", "closed": "X", "": "?"}
    header = (
        "## Share Adjustments\n\n"
        "| Ticker | Dir | Target | Current | Delta | Action | $ Value |\n"
        "|--------|-----|-------:|--------:|------:|--------|--------:|\n"
    )
    lines = []
    for _, row in adj_df.iterrows():
        d = dir_map.get(str(row["direction"]).lower(), "?")
        lines.append(
            f"| {row['ticker']} | {d} | {row['target_shares']:,} | "
            f"{row['current_shares']:,} | {row['delta']:+,} | "
            f"{row['action']} | ${row['dollar_value']:+,.0f} |"
        )
    return header + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Risk summary tables (deterministic, pre-AI)
# ---------------------------------------------------------------------------


def build_risk_summary_markdown(risk_data: dict, sizer_result: dict) -> str:
    """Build deterministic Markdown tables from risk data and sizer output."""
    sections: list[str] = []

    # --- Sizer exposures ---
    exposures = sizer_result.get("exposures")
    if isinstance(exposures, dict):
        exposure_labels = {
            "equity_gross": "Equity Gross",
            "equity_net": "Equity Net",
            "commodity_gross": "Commodity Gross",
            "commodity_net": "Commodity Net",
            "total_gross": "Total Gross",
            "total_net": "Total Net",
            "hedge_gross": "Hedge Gross",
        }
        sections.append("## Portfolio Exposures\n")
        for k, v in exposures.items():
            label = exposure_labels.get(k, k.replace("_", " ").title())
            try:
                sections.append(f"- **{label}**: {float(v) * 100:.1f}%")
            except (TypeError, ValueError):
                sections.append(f"- **{label}**: {v}")
        sections.append("")

    # --- Constraints utilization ---
    constraints = sizer_result.get("constraints")
    if isinstance(constraints, dict):
        sections.append("## Constraints Utilization\n")
        sections.append("| Constraint | Limit | Current | Utilization |")
        sections.append("|---|---:|---:|---:|")
        for name, info in constraints.items():
            if not isinstance(info, dict):
                continue
            util_pct = float(info.get("utilization", 0)) * 100
            sections.append(
                f"| {name} | {info.get('limit', 'N/A')} | "
                f"{float(info.get('current', 0)):.4f} | {util_pct:.1f}% |"
            )
        sections.append("")

    # --- Beta summary ---
    beta_keys = [
        "net_beta_spy",
        "net_beta_iwm",
        "post_hedge_beta_spy",
        "post_hedge_beta_iwm",
    ]
    beta_lines = []
    for key in beta_keys:
        val = sizer_result.get(key)
        if val is not None:
            try:
                beta_lines.append(f"- **{key}**: {float(val):.4f}")
            except (TypeError, ValueError):
                pass
    if beta_lines:
        sections.append("## Beta Summary\n")
        sections.extend(beta_lines)
        sections.append("")

    # --- Hedge direction warnings ---
    warnings = sizer_result.get("hedge_direction_issues", [])
    if warnings:
        sections.append("### Hedge Warnings\n")
        for w in warnings:
            sections.append(f"- {w}")
        sections.append("")

    # --- Per-position risk table ---
    weights_df = sizer_result.get("weights_df")
    if weights_df is not None:
        sections.append("## Position Details\n")
        sections.append(
            "| Ticker | Dir | Conv | Weight | Beta SPY | Beta IWM | Vol | Shares | $ Weight |"
        )
        sections.append(
            "|--------|-----|-----:|-------:|---------:|---------:|----:|-------:|---------:|"
        )
        for _, row in weights_df.iterrows():
            sections.append(
                f"| {row['ticker']} | "
                f"{str(row.get('direction', '')).lower()[:1].upper()} | "
                f"{row.get('conviction', '')} | {row['weight'] * 100:.2f}% | "
                f"{row.get('beta_spy', 0):.3f} | {row.get('beta_iwm', 0):.3f} | "
                f"{row.get('realized_vol', 0) * 100:.2f}% | "
                f"{row.get('shares', 0):,} | ${row.get('dollar_weight', 0):,.0f} |"
            )
        sections.append("")

    # --- Hedges table ---
    hedges_df = sizer_result.get("hedges_df")
    if hedges_df is not None and not hedges_df.empty:
        sections.append("## Hedges\n")
        sections.append("| Ticker | Dir | Weight | Shares | $ Weight |")
        sections.append("|--------|-----|-------:|-------:|---------:|")
        for _, row in hedges_df.iterrows():
            sections.append(
                f"| {row['ticker']} | "
                f"{str(row.get('direction', '')).lower()[:1].upper()} | "
                f"{row.get('weight', 0):.4f} | "
                f"{row.get('shares', 0):,} | ${row.get('dollar_weight', 0):,.0f} |"
            )
        sections.append("")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_daily_user_message(
    bundle: dict,
    risk_summary_md: str,
    adjustments_md: str,
) -> str:
    """Build the user message for the daily risk report Claude call."""
    import copy

    prompt_bundle = copy.deepcopy(bundle)

    # Drop heavy serialized DataFrames that are already in the Markdown tables
    for key in ["weights_df", "hedges_df", "max_scaled"]:
        sizer = prompt_bundle.get("sizer_summary", {})
        if isinstance(sizer, dict):
            sizer.pop(key, None)

    bundle_json = json.dumps(prompt_bundle, indent=2, default=str)

    return f"""Here is today's portfolio risk data bundle:

```json
{bundle_json}
```

{risk_summary_md}

{adjustments_md}

Analyze this portfolio and produce a daily risk report with these sections:

1. **Risk Summary** -- max 5 bullets on the most important risks/vulnerabilities today
2. **Position-Level Flags** -- for each position with an actionable signal (deteriorating technicals, momentum divergence, severe drawdown, distressed gating changes, high beta exposure), describe the concern and severity (low/medium/high)
3. **Portfolio-Level Risks** -- beta neutrality status, gross leverage vs limits, concentration, correlation risks
4. **Actionable Alerts** -- positions where the share adjustment is large or where risk metrics warrant immediate attention
5. **Watchlist** -- 3-5 specific triggers that would warrant intraday or next-day action

Constraints:
- Cite specific metrics from the data (vol, beta, drawdown %, ROC, MA signals).
- Be direct and concise. No filler.
- Focus on what changed or what is abnormal.
- The share adjustments table is deterministic -- do not second-guess the sizer, but flag if any adjustment is unusually large.

After the report, output the separator `{DAILY_SUMMARY_SEPARATOR}` on its own line, then a JSON block:
```json
{{
  "risk_level": "<low|moderate|elevated|high|critical>",
  "top_risks": ["<top 3-5 risks>"],
  "positions_flagged": ["<tickers with actionable flags>"],
  "largest_adjustments": ["<tickers with biggest delta>"]
}}
```

End immediately after the JSON. No assistant meta text."""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _fallback_daily_summary() -> dict:
    return {
        "risk_level": "unknown",
        "top_risks": [],
        "positions_flagged": [],
        "largest_adjustments": [],
        "parse_error": True,
    }


def parse_daily_response(text: str) -> tuple[str, dict]:
    """Parse Claude response into (report_md, summary_dict)."""
    if DAILY_SUMMARY_SEPARATOR in text:
        parts = text.split(DAILY_SUMMARY_SEPARATOR, 1)
        report_md = parts[0].strip()
        json_part = parts[1].strip()
        if json_part.startswith("```"):
            json_part = (
                json_part.split("\n", 1)[1] if "\n" in json_part else json_part[3:]
            )
        if json_part.endswith("```"):
            json_part = json_part[:-3]
        json_part = json_part.strip()
        try:
            summary = json.loads(json_part)
        except json.JSONDecodeError:
            log.warning("Failed to parse daily summary JSON")
            summary = _fallback_daily_summary()
    else:
        log.warning("No daily summary separator found")
        report_md = text.strip()
        summary = _fallback_daily_summary()
    return report_md, summary


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def write_daily_outputs(
    report_md: str,
    summary: dict,
    bundle: dict,
    adjustments_df,
    output_dir: Path,
    today: str,
):
    """Write all daily report outputs and archive."""
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "report.md").write_text(report_md, encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_bundle(bundle, output_dir / "daily_bundle.json")
    if not adjustments_df.empty:
        adjustments_df.to_csv(output_dir / "adjustments.csv", index=False)
    log.info("Wrote daily outputs to %s", output_dir)

    # Archive
    archive_dir = output_dir / "history" / today
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "report.md").write_text(report_md, encoding="utf-8")
    (archive_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_bundle(bundle, archive_dir / "daily_bundle.json")
    if not adjustments_df.empty:
        adjustments_df.to_csv(archive_dir / "adjustments.csv", index=False)
    log.info("Archived daily to %s", archive_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Automated daily portfolio risk report"
    )
    parser.add_argument(
        "--force", action="store_true", help="Bypass weekday-afternoon gate"
    )
    parser.add_argument(
        "--book",
        type=float,
        default=None,
        help="Book size in USD (default: sum of abs(PositionValue) from open_positions.csv)",
    )
    args = parser.parse_args()

    if (
        not args.force
        and not os.environ.get("FORCE_RUN")
        and not _is_weekday_morning_et()
    ):
        log.info("Not weekday 09:xx ET — exiting (use --force to override)")
        sys.exit(0)

    today_str = datetime.now(ET).strftime("%Y-%m-%d")
    log.info("=== Daily risk report run starting (%s) ===", today_str)

    # 1. Load portfolio + open positions
    portfolio_df = load_portfolio()
    open_positions_df = load_open_positions()
    log.info(
        "Loaded %d portfolio positions, %d open positions",
        len(portfolio_df),
        len(open_positions_df),
    )

    # 2. Determine book size
    if args.book:
        book = args.book
    elif not open_positions_df.empty and "PositionValue" in open_positions_df.columns:
        book = float(open_positions_df["PositionValue"].abs().sum())
        log.info("Book size from open_positions.csv: $%,.2f", book)
    else:
        book = 100_000.0
        log.info("Using default book size: $%,.2f", book)

    # 3. Collect risk data
    log.info("Collecting risk data for %d positions...", len(portfolio_df))
    t_collect = time.perf_counter()
    risk_data = collect_risk_data(portfolio_df)
    log.info("Risk data collected in %.2fs", time.perf_counter() - t_collect)

    # 4. Run sizer (deterministic)
    log.info("Running portfolio sizer (book=$%,.2f)...", book)
    t_sizer = time.perf_counter()
    sizer_result = run_sizer(portfolio_df, book)
    log.info("Sizer completed in %.2fs", time.perf_counter() - t_sizer)

    if sizer_result.get("error"):
        log.error("Sizer failed: %s", sizer_result["error"])

    # 5. Compute share adjustments (deterministic)
    adjustments_df = compute_adjustments(sizer_result, open_positions_df)

    # 6. Build deterministic Markdown sections
    risk_summary_md = build_risk_summary_markdown(risk_data, sizer_result)
    adjustments_md = format_adjustments_markdown(adjustments_df)

    # 7. Serialize bundle (exclude heavy DataFrames)
    bundle = serialize_bundle(
        {
            "risk_data": risk_data,
            "sizer_summary": {
                k: v
                for k, v in sizer_result.items()
                if k not in ("weights_df", "hedges_df", "max_scaled")
            },
        }
    )

    # 8. Load prompt
    daily_system_md = load_prompt_file(
        PROMPTS_DIR / "daily_system.md", "prompts/daily_system.md"
    )

    # 9. Call Claude
    user_msg = _build_daily_user_message(bundle, risk_summary_md, adjustments_md)
    report_md = None
    summary = None

    try:
        response_text, citations = call_claude(
            system_msg=daily_system_md,
            user_msg=user_msg,
            allowed_domains=None,  # no web search for daily
        )
        report_md, summary = parse_daily_response(response_text)
        report_md = strip_llm_meta(report_md)

        # Compose final report: deterministic tables + AI analysis
        report_md = (
            f"# Daily Portfolio Risk Report — {today_str}\n\n"
            f"{risk_summary_md}\n\n"
            f"{adjustments_md}\n\n"
            f"---\n\n"
            f"## AI Risk Analysis\n\n"
            f"{report_md}"
        )

        if citations:
            sources_lines = ["\n\n---\n\n## Sources\n"]
            for title, url in citations:
                sources_lines.append(f"- [{title}]({url})")
            report_md += "\n".join(sources_lines)
    except Exception as e:
        log.error("Claude call failed: %s", e, exc_info=True)
        report_md = (
            f"# Daily Portfolio Risk Report — {today_str}\n\n"
            f"{risk_summary_md}\n\n"
            f"{adjustments_md}\n\n"
            f"---\n\n"
            f"**Error**: Claude analysis failed.\n\n```\n{e}\n```"
        )
        summary = _fallback_daily_summary()
        summary["error"] = str(e)

    # 10. Write outputs + archive
    write_daily_outputs(
        report_md, summary, bundle, adjustments_df, OUTPUT_DIR, today_str
    )

    # 11. Create GitHub Issue
    issue_title = f"Daily Risk Report — {today_str}"
    try:
        create_github_issue(issue_title, report_md)
    except Exception as e:
        log.error("GitHub Issue creation failed: %s", e, exc_info=True)

    log.info("=== Daily risk report run complete ===")


if __name__ == "__main__":
    main()
