#!/usr/bin/env python3
"""
Automated daily portfolio report — commentary plus recommendations.

Pass 1: Market analysis using all 12 data sources + news search → stance + leverage.
Pass 2: Portfolio risk commentary with stance-driven target leverage.
Pass 3: Decision recommendations with data-quality gates and approval persistence.

Run:
    python -m auto_report.auto_daily_report --force   # bypass weekday-morning gate
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
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from auto_report.auto_weekly_report import (
    DAILY_NEWS_DIGEST_DAYS,
    RULES_TEXT,
    _prepare_prompt_bundle,
    build_performance_markdown,
)

# Import market data collection + formatting from weekly module
from auto_report.auto_weekly_report import (  # noqa: E402
    collect_data as collect_market_data,
)
from auto_report.recommendations import (  # noqa: E402
    RECOMMENDATIONS_SEPARATOR,
    assess_report_data_quality,
    build_recommendations_user_message,
    evaluate_due_recommendations,
    fallback_recommendations_payload,
    format_recommendations_markdown,
    parse_recommendations_response,
    persist_recommendations,
    repair_recommendations_response,
    stable_hash,
)
from auto_report.shared import (  # noqa: E402
    call_report_llm,
    create_github_issue,
    load_prompt_file,
    serialize_bundle,
    strip_llm_meta,
    write_bundle,
)
from llm_utils import MODEL_HIGH  # noqa: E402
from portfolio.position_groups import normalize_group_name

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

# Separators for parsing LLM responses
PASS1_SUMMARY_SEPARATOR = "<!-- PASS1_STANCE_JSON -->"
DAILY_SUMMARY_SEPARATOR = "<!-- DAILY_SUMMARY_JSON -->"

# Stance → leverage mapping (hybrid: fixed base ± 0.25 adjustment)
STANCE_LEVERAGE_MAP = {
    "Aggressively Offensive": {"base": 2.5, "low": 2.25, "high": 2.75},
    "Offensive": {"base": 2.0, "low": 1.75, "high": 2.25},
    "Neutral / Watchful": {"base": 1.5, "low": 1.25, "high": 1.75},
    "Defensive": {"base": 1.0, "low": 0.75, "high": 1.25},
    "Aggressively Defensive": {"base": 0.5, "low": 0.25, "high": 0.75},
}
DEFAULT_LEVERAGE = 1.5
DEFAULT_STANCE = "Neutral / Watchful"
REGIME_DIMENSION_ORDER = (
    ("market_behavior", "Market Behavior"),
    ("macro_momentum", "Macro Momentum"),
    ("liquidity", "Liquidity"),
    ("positioning", "Positioning"),
    ("risk_sentiment", "Risk Sentiment"),
    ("cycle_position", "Cycle Position"),
)
REGIME_DIMENSION_KEYS = {dim_key for dim_key, _label in REGIME_DIMENSION_ORDER}
REGIME_DIMENSION_ALIASES = {
    "market_behavior": "market_behavior",
    "marketbehavior": "market_behavior",
    "macro_momentum": "macro_momentum",
    "macromomentum": "macro_momentum",
    "liquidity": "liquidity",
    "positioning": "positioning",
    "risk_sentiment": "risk_sentiment",
    "risksentiment": "risk_sentiment",
    "cycle_position": "cycle_position",
    "cycleposition": "cycle_position",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_currency(amount: float) -> str:
    return format(float(amount), ",.2f")


# ---------------------------------------------------------------------------
# Leverage validation
# ---------------------------------------------------------------------------


def validate_and_clamp_leverage(leverage: float, stance: str) -> float:
    """Clamp the LLM's chosen leverage to the valid range for the given stance."""
    bounds = STANCE_LEVERAGE_MAP.get(stance)
    if bounds is None:
        log.warning("Unknown stance %r — using DEFAULT_LEVERAGE", stance)
        return DEFAULT_LEVERAGE
    clamped = max(bounds["low"], min(bounds["high"], float(leverage)))
    if clamped != leverage:
        log.warning(
            "Leverage %.2f outside stance %r range [%.2f, %.2f] — clamped to %.2f",
            leverage,
            stance,
            bounds["low"],
            bounds["high"],
            clamped,
        )
    return clamped


# ---------------------------------------------------------------------------
# Previous-day context
# ---------------------------------------------------------------------------


def load_last_daily_summary(history_dir: Path) -> str | None:
    """Load the most recent daily summary.json from the history archive."""
    if not history_dir.exists():
        return None
    dirs = sorted(
        [d for d in history_dir.iterdir() if d.is_dir() and len(d.name) == 10],
        reverse=True,
    )
    if not dirs:
        return None
    summary_path = dirs[0] / "summary.json"
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        return json.dumps(data, indent=2)
    except Exception:
        return None


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
    """Load portfolio positions from the ontology runtime and return a DataFrame."""
    from ontology.runtime_read_service import OntologyRuntimeReadService

    df = OntologyRuntimeReadService().positions_df()
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df["direction"] = df["direction"].fillna("").str.strip().str.lower()
    df["conviction"] = df["conviction"].fillna(3).astype(int)
    return df


# ---------------------------------------------------------------------------
# Risk data collection (portfolio-specific)
# ---------------------------------------------------------------------------


def collect_risk_data(portfolio_df) -> dict:
    """Collect per-position risk metrics from all risk modules."""
    import pandas as pd

    results = {}
    tickers = list(portfolio_df["ticker"])
    asset_map = dict(zip(portfolio_df["ticker"], portfolio_df["asset"]))  # noqa: B905

    # 1. Technical analysis (per-ticker MA signals + ROC)
    try:
        from portfolio.technical_analysis.technical_analysis import get_data as get_ta_data

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

    # 2. Price momentum (batch)
    try:
        from portfolio.momentum.price_momentum.momentum import get_data as get_momentum_data

        t0 = time.perf_counter()
        momentum = get_momentum_data()
        log.info("price momentum fetched in %.2fs", time.perf_counter() - t0)
        results["price_momentum"] = momentum
    except Exception as e:
        log.warning("price momentum fetch failed: %s", e, exc_info=True)
        results["price_momentum"] = {"error": str(e)}

    # 3. Portfolio risk metrics (volatility, drawdown, beta)
    try:
        from portfolio.portfolio_optimizer.portfolio_analyzer import (
            MARKET_TICKER_LONG,
            MARKET_TICKER_SHORT,
            compute_beta_frame,
            compute_contrarian_long_metrics,
            compute_defense_volatility,
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
        equity_tickers = [t for t in tickers if asset_map.get(t, "").lower() == "equity"]
        severe_dd = compute_severe_drawdown_flags(usd_prices, equity_tickers)
        results["severe_drawdown_flags"] = severe_dd

        # Contrarian metrics
        contrarian_tickers = list(
            portfolio_df.loc[
                portfolio_df["contrarian"].astype(str).str.lower().isin(["true", "1", "yes"]),
                "ticker",
            ]
        )
        if contrarian_tickers:
            contrarian_metrics = compute_contrarian_long_metrics(prices_all, contrarian_tickers)
            results["contrarian_metrics"] = contrarian_metrics.to_dict(orient="index")
        else:
            results["contrarian_metrics"] = {}

        # Beta frame
        valid_tickers = [t for t in tickers if t in rets.columns]
        beta_frame, _, _ = compute_beta_frame(rets, valid_tickers)
        results["beta_frame"] = beta_frame.to_dict(orient="index")

        log.info("portfolio risk metrics fetched in %.2fs", time.perf_counter() - t0)
    except Exception as e:
        log.warning("portfolio risk metrics fetch failed: %s", e, exc_info=True)
        results["portfolio_risk"] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Portfolio sizer (deterministic, no AI)
# ---------------------------------------------------------------------------


def run_sizer(portfolio_df, book: float, target_leverage: float = DEFAULT_LEVERAGE) -> dict:
    """Run portfolio sizer and return full result dict."""
    try:
        import pandas as pd

        from portfolio.portfolio_optimizer.portfolio_sizer import size_portfolio

        positions = [
            {
                "ticker": row["ticker"],
                "conviction": int(row["conviction"]),
                "group_name": normalize_group_name(row.get("group_name")),
                "group_conviction": int(row["group_conviction"])
                if row.get("group_conviction") is not None and not pd.isna(row.get("group_conviction"))
                else None,
            }
            for _, row in portfolio_df.iterrows()
            if row["direction"] in ("long", "short")
        ]
        return size_portfolio(positions=positions, book=book, target_leverage=target_leverage)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def load_configured_book_size() -> float | None:
    env_value = (os.getenv("TALISMAN_BOOK_SIZE") or "").strip()
    if env_value:
        try:
            parsed = float(env_value)
            if parsed > 0:
                return parsed
        except ValueError:
            log.warning("Ignoring invalid TALISMAN_BOOK_SIZE=%r", env_value)

    try:
        from api.portfolio_settings import get_configured_portfolio_book_size

        return get_configured_portfolio_book_size()
    except Exception:
        log.debug("No configured portfolio book size available", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Share adjustment computation (deterministic)
# ---------------------------------------------------------------------------


def compute_adjustments(sizer_result: dict, portfolio_df) -> pd.DataFrame:
    """Compare sizer target shares to current shares from the portfolio db.

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

    # Build current holdings map from the portfolio db
    current: dict[str, int] = {}
    for _, row in portfolio_df.iterrows():
        symbol = str(row.get("ticker", "")).strip().upper()
        if not symbol:
            continue
        shares = row.get("shares")
        if shares is None or pd.isna(shares):
            continue
        qty = int(shares)
        direction = str(row.get("direction", "")).strip().lower()
        if direction == "short":
            qty = -abs(qty)
        current[symbol] = qty

    # The report loads regular positions into portfolio_df, while the sizer
    # attaches current hedge holdings to hedges_df from the portfolio db.
    if hedges_df is not None and "current_shares" in hedges_df.columns:
        for _, row in hedges_df.iterrows():
            symbol = str(row.get("ticker", "")).strip().upper()
            if not symbol:
                continue
            shares = row.get("current_shares")
            if shares is None or pd.isna(shares):
                continue
            current[symbol] = int(shares)

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
                f"| {name} | {info.get('limit', 'N/A')} | {float(info.get('current', 0)):.4f} | {util_pct:.1f}% |"
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
        group_rows: dict[str, dict] = {}
        for _, row in weights_df.iterrows():
            group_name = normalize_group_name(row.get("group_name"))
            if not group_name:
                continue
            group = group_rows.setdefault(
                group_name.casefold(),
                {
                    "group_name": group_name,
                    "group_conviction": row.get("group_conviction"),
                    "tickers": [],
                    "weight": 0.0,
                },
            )
            group["tickers"].append(row.get("ticker"))
            group["weight"] += float(row.get("weight", 0) or 0)
        if group_rows:
            sections.append("## Group Summary\n")
            sections.append("| Group | Conv | Members | Weight |")
            sections.append("|-------|-----:|---------|-------:|")
            for group in group_rows.values():
                sections.append(
                    f"| {group['group_name']} | {group.get('group_conviction') or ''} | "
                    f"{', '.join(str(t) for t in group['tickers'])} | {group['weight'] * 100:.2f}% |"
                )
            sections.append("")

        sections.append("## Position Details\n")
        sections.append("| Ticker | Group | Dir | Conv | Weight | Beta SPY | Beta IWM | Vol | Shares | $ Weight |")
        sections.append("|--------|-------|-----|-----:|-------:|---------:|---------:|----:|-------:|---------:|")
        for _, row in weights_df.iterrows():
            sections.append(
                f"| {row['ticker']} | "
                f"{normalize_group_name(row.get('group_name')) or '-'} | "
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
# Pass 1: Market Analysis + Stance Determination
# ---------------------------------------------------------------------------


def _fallback_pass1_result() -> dict:
    return {
        "stance": DEFAULT_STANCE,
        "target_leverage": DEFAULT_LEVERAGE,
        "leverage_rationale": "Defaulting to neutral stance due to analysis failure.",
        "confidence": "low",
        "six_dimensions": {},
        "drivers": [],
        "watchlist_triggers": [],
        "parse_error": True,
    }


def _build_pass1_system_message(last_daily_json: str | None) -> str:
    """Build Pass 1 prompt from shared core + weekly overlay."""
    system_md = load_prompt_file(PROMPTS_DIR / "system.md", "prompts/system.md")
    weekly_system_md = load_prompt_file(
        PROMPTS_DIR / "weekly_system.md",
        "prompts/weekly_system.md",
    )
    parts = [system_md, weekly_system_md]
    if last_daily_json:
        parts.append(f"## Previous Session's Summary\n\n```json\n{last_daily_json}\n```")
    return "\n\n---\n\n".join(parts)


def _build_pass1_user_message(market_bundle: dict, perf_md: str) -> str:
    """Build user message for Pass 1: market analysis and stance determination."""
    prompt_bundle = _prepare_prompt_bundle(market_bundle)
    bundle_json = json.dumps(prompt_bundle, indent=2, default=str)

    stance_options = " | ".join(STANCE_LEVERAGE_MAP.keys())
    leverage_table_lines = []
    for stance_name, bounds in STANCE_LEVERAGE_MAP.items():
        leverage_table_lines.append(
            f"  - {stance_name}: base {bounds['base']}, range [{bounds['low']}, {bounds['high']}]"
        )
    leverage_table = "\n".join(leverage_table_lines)

    return f"""Here is today's market data bundle:

```json
{bundle_json}
```

{perf_md}

{RULES_TEXT}

Before writing the analysis, use the web search tool to find the key market-moving
news from the past 24 hours (Fed speakers, economic releases, geopolitical events,
major premarket moves, overnight developments) that are relevant to today's session.
Weave this context into your analysis and cite sources for news-driven claims.
Treat the news_digests block as user-curated high-signal leads from uploaded digest
files; use web search to verify and cite claims rather than citing the uploaded digest itself.

Analyze the market environment through the lens of the investment philosophy.

Before the separator, output only the existing-style `# Stance Rationale` section as
Markdown prose. Do not include Market Regime Assessment, Macro & Market Conditions,
or Watchlist in the Markdown before the separator; those sections are rendered from
the structured JSON.

In the JSON block, provide these structured sections:

1. `market_regime_assessment` — a headline plus exactly two paragraphs:
   - `headline`: one sentence naming the dominant regime.
   - `dominant_character`: 2-4 sentences on price action, breadth/leadership, and how the market is reacting to news.
   - `main_tension`: 2-4 sentences on the main tension beneath the surface.
2. `regime_evidence` — exactly six entries in this order. Each rating must be Supportive / Neutral / Cautionary / Adverse:
   - Market Behavior (breadth, sector internals, price action)
   - Macro Momentum (economic_growth, labor_market, housing, country_dashboard, earnings)
   - Liquidity (Fed balance sheet, central_banks, yield_curve, bond_dashboard, credit conditions, funding markets)
   - Positioning (COT data, sentiment, consensus)
   - Risk Sentiment (VIX term structure, credit spreads, safe haven flows)
   - Cycle Position (where in boom-bust, credit cycle phase)
3. `six_dimensions` — rating map for the Market Stance table. It must match the ratings in `regime_evidence`.
4. `watchlist` — split into `risks_to_upside` and `risks_to_downside`. Each side needs 1-4 trigger/implication entries.
5. `watchlist_triggers` — flattened strings derived from the watchlist entries.

Stance options: {stance_options}

Leverage ranges per stance:
{leverage_table}

Pick the stance that best reflects the current environment, then choose a specific
target_leverage within that stance's range. The leverage should reflect your conviction
level within the stance — e.g., a borderline Offensive environment might warrant 1.75
(low end of Offensive) rather than 2.25 (high end).

Constraints:
- Cite specific metrics from the data and news search.
- Use news_digests explicitly when it changes the assessment of market character, macro risk, or portfolio risk.
- Use labor_market, housing, central_banks, yield_curve, bond_dashboard, and country_dashboard explicitly when they inform Macro Momentum, Liquidity/Rates, Risk Sentiment, or Cycle Position.
- Assess the context for a long/short equity portfolio — not generic market commentary.
- Be direct and concise. No filler.
- Max 1200 words across the Markdown rationale and structured text fields.
- In `regime_evidence`, write `evidence` as 1-2 complete grammatical sentences with clear subjects and verbs, and include the specific metrics that support the rating. Do not omit numbers or make the sentence generic. Do not use compressed market-note fragments, comma-spliced ticker tape, or sentence fragments.
- Example evidence style: "Market behavior is neutral because the S&P 500 and Nasdaq 100 are at new highs, but breadth is not confirming the move: 57.3% of stocks are above their 200-DMA, 45.1% are above their 20-DMA, and only 22 stocks are making 52-week highs."
- `stance_implication` must be one complete sentence explaining the effect on stance or leverage.
- In `watchlist`, each `implication` must state what changes in stance, leverage, hedging, or exposure if relevant.

After the `# Stance Rationale` section, output the separator `{PASS1_SUMMARY_SEPARATOR}` on its own line, then a JSON block:
```json
{{
  "stance": "<{stance_options}>",
  "target_leverage": "<float between 0.25 and 2.75>",
  "leverage_rationale": "<one sentence explaining leverage choice within the stance range>",
  "confidence": "<high|medium|low>",
  "market_regime_assessment": {{
    "headline": "<one-sentence regime label>",
    "dominant_character": "<2-4 sentences on dominant market character>",
    "main_tension": "<2-4 sentences on the main tension beneath the surface>"
  }},
  "regime_evidence": [
    {{
      "dimension": "Market Behavior",
      "rating": "<Supportive|Neutral|Cautionary|Adverse>",
      "evidence": "<1-2 complete sentences that include specific supporting metrics>",
      "stance_implication": "<one sentence explaining impact on stance/leverage>"
    }},
    {{
      "dimension": "Macro Momentum",
      "rating": "<Supportive|Neutral|Cautionary|Adverse>",
      "evidence": "<1-2 complete sentences that include specific supporting metrics>",
      "stance_implication": "<one sentence explaining impact on stance/leverage>"
    }},
    {{
      "dimension": "Liquidity",
      "rating": "<Supportive|Neutral|Cautionary|Adverse>",
      "evidence": "<1-2 complete sentences that include specific supporting metrics>",
      "stance_implication": "<one sentence explaining impact on stance/leverage>"
    }},
    {{
      "dimension": "Positioning",
      "rating": "<Supportive|Neutral|Cautionary|Adverse>",
      "evidence": "<1-2 complete sentences that include specific supporting metrics>",
      "stance_implication": "<one sentence explaining impact on stance/leverage>"
    }},
    {{
      "dimension": "Risk Sentiment",
      "rating": "<Supportive|Neutral|Cautionary|Adverse>",
      "evidence": "<1-2 complete sentences that include specific supporting metrics>",
      "stance_implication": "<one sentence explaining impact on stance/leverage>"
    }},
    {{
      "dimension": "Cycle Position",
      "rating": "<Supportive|Neutral|Cautionary|Adverse>",
      "evidence": "<1-2 complete sentences that include specific supporting metrics>",
      "stance_implication": "<one sentence explaining impact on stance/leverage>"
    }}
  ],
  "six_dimensions": {{
    "market_behavior": "<Supportive|Neutral|Cautionary|Adverse>",
    "macro_momentum": "<Supportive|Neutral|Cautionary|Adverse>",
    "liquidity": "<Supportive|Neutral|Cautionary|Adverse>",
    "positioning": "<Supportive|Neutral|Cautionary|Adverse>",
    "risk_sentiment": "<Supportive|Neutral|Cautionary|Adverse>",
    "cycle_position": "<Supportive|Neutral|Cautionary|Adverse>"
  }},
  "drivers": ["<top 3-5 drivers>"],
  "watchlist": {{
    "risks_to_upside": [
      {{
        "trigger": "<specific observable trigger>",
        "implication": "<what changes if it happens>"
      }}
    ],
    "risks_to_downside": [
      {{
        "trigger": "<specific observable trigger>",
        "implication": "<what changes if it happens>"
      }}
    ]
  }},
  "watchlist_triggers": ["<flattened watchlist strings>"]
}}
```

End immediately after the JSON. No assistant meta text."""


def _clean_report_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalized_dimension_key(value) -> str:
    raw = _clean_report_text(value).lower()
    compact = "".join(ch if ch.isalnum() else "_" for ch in raw)
    while "__" in compact:
        compact = compact.replace("__", "_")
    compact = compact.strip("_")
    return REGIME_DIMENSION_ALIASES.get(compact, compact)


def _regime_evidence_by_key(regime_evidence) -> dict[str, dict]:
    if isinstance(regime_evidence, dict):
        items = []
        for dim, entry in regime_evidence.items():
            if isinstance(entry, dict):
                merged = {"dimension": dim, **entry}
            else:
                merged = {"dimension": dim, "rating": entry}
            items.append(merged)
    elif isinstance(regime_evidence, list):
        items = [entry for entry in regime_evidence if isinstance(entry, dict)]
    else:
        return {}

    by_key: dict[str, dict] = {}
    for entry in items:
        key = _normalized_dimension_key(entry.get("dimension") or entry.get("key") or entry.get("name"))
        if key in REGIME_DIMENSION_KEYS:
            by_key[key] = entry
    return by_key


def _derive_six_dimensions(stance_dict: dict) -> None:
    existing = stance_dict.get("six_dimensions")
    if isinstance(existing, dict) and existing:
        return

    evidence_by_key = _regime_evidence_by_key(stance_dict.get("regime_evidence"))
    derived = {}
    for dim_key, _label in REGIME_DIMENSION_ORDER:
        rating = _clean_report_text(evidence_by_key.get(dim_key, {}).get("rating"))
        if rating:
            derived[dim_key] = rating
    if derived:
        stance_dict["six_dimensions"] = derived


def _watchlist_entries(entries) -> list[str]:
    if not isinstance(entries, list):
        return []
    rendered = []
    for entry in entries:
        if isinstance(entry, dict):
            trigger = _clean_report_text(entry.get("trigger"))
            implication = _clean_report_text(entry.get("implication"))
        else:
            trigger = _clean_report_text(entry)
            implication = ""
        if not trigger:
            continue
        rendered.append(f"{trigger}: {implication}" if implication else trigger)
    return rendered


def _derive_watchlist_triggers(stance_dict: dict) -> None:
    existing = stance_dict.get("watchlist_triggers")
    if isinstance(existing, list) and existing:
        return

    watchlist = stance_dict.get("watchlist")
    if not isinstance(watchlist, dict):
        return

    flattened = _watchlist_entries(watchlist.get("risks_to_upside"))
    flattened.extend(_watchlist_entries(watchlist.get("risks_to_downside")))
    if flattened:
        stance_dict["watchlist_triggers"] = flattened


def _normalize_pass1_structured_fields(stance_dict: dict) -> None:
    _derive_six_dimensions(stance_dict)
    _derive_watchlist_triggers(stance_dict)


def parse_pass1_response(text: str) -> tuple[str, dict]:
    """Parse Pass 1 response into (market_analysis_md, stance_dict)."""
    if PASS1_SUMMARY_SEPARATOR in text:
        parts = text.split(PASS1_SUMMARY_SEPARATOR, 1)
        analysis_md = parts[0].strip()
        json_part = parts[1].strip()
        if json_part.startswith("```"):
            json_part = json_part.split("\n", 1)[1] if "\n" in json_part else json_part[3:]
        if json_part.endswith("```"):
            json_part = json_part[:-3]
        json_part = json_part.strip()
        try:
            stance_dict = json.loads(json_part)
        except json.JSONDecodeError:
            log.warning("Failed to parse Pass 1 stance JSON — using fallback")
            stance_dict = _fallback_pass1_result()
    else:
        log.warning("No Pass 1 separator found — using fallback")
        analysis_md = text.strip()
        stance_dict = _fallback_pass1_result()

    # Validate and clamp leverage
    raw_leverage = stance_dict.get("target_leverage", DEFAULT_LEVERAGE)
    try:
        raw_leverage = float(raw_leverage)
    except (TypeError, ValueError):
        log.warning(
            "Non-numeric target_leverage %r — defaulting to %.1f",
            raw_leverage,
            DEFAULT_LEVERAGE,
        )
        raw_leverage = DEFAULT_LEVERAGE

    stance = stance_dict.get("stance", DEFAULT_STANCE)
    if stance == "Neutral":
        stance = DEFAULT_STANCE
        stance_dict["stance"] = stance
    if stance not in STANCE_LEVERAGE_MAP:
        log.warning("Unknown stance %r — defaulting to %r", stance, DEFAULT_STANCE)
        stance = DEFAULT_STANCE
        stance_dict["stance"] = stance

    stance_dict["target_leverage"] = validate_and_clamp_leverage(raw_leverage, stance)
    _normalize_pass1_structured_fields(stance_dict)

    return analysis_md, stance_dict


def _render_market_regime_assessment(market_regime_assessment) -> str | None:
    if not isinstance(market_regime_assessment, dict):
        return None

    headline = _clean_report_text(market_regime_assessment.get("headline"))
    dominant_character = _clean_report_text(market_regime_assessment.get("dominant_character"))
    main_tension = _clean_report_text(market_regime_assessment.get("main_tension"))
    if not headline or not dominant_character or not main_tension:
        return None

    return "\n\n".join(
        [
            "# Market Regime Assessment",
            f"**{headline}**",
            dominant_character,
            main_tension,
        ]
    )


def _render_regime_evidence_dashboard(regime_evidence) -> str | None:
    evidence_by_key = _regime_evidence_by_key(regime_evidence)
    if any(dim_key not in evidence_by_key for dim_key, _label in REGIME_DIMENSION_ORDER):
        return None

    sections = ["# Macro & Market Conditions"]
    for dim_key, label in REGIME_DIMENSION_ORDER:
        entry = evidence_by_key[dim_key]
        rating = _clean_report_text(entry.get("rating"))
        evidence = _clean_report_text(entry.get("evidence"))
        stance_implication = _clean_report_text(entry.get("stance_implication") or entry.get("implication"))
        if not rating or not evidence or not stance_implication:
            return None
        sections.append(
            "\n\n".join(
                [
                    f"### {label} — {rating}",
                    f"**Evidence:** {evidence}",
                    f"**Stance implication:** {stance_implication}",
                ]
            )
        )
    return "\n\n".join(sections)


def _render_watchlist_entries(entries) -> list[str] | None:
    if not isinstance(entries, list) or not entries:
        return None

    lines = []
    for entry in entries:
        if not isinstance(entry, dict):
            return None
        trigger = _clean_report_text(entry.get("trigger"))
        implication = _clean_report_text(entry.get("implication"))
        if not trigger or not implication:
            return None
        lines.append(f"- **{trigger}:** {implication}")
    return lines


def _render_watchlist(watchlist) -> str | None:
    if not isinstance(watchlist, dict):
        return None

    upside_lines = _render_watchlist_entries(watchlist.get("risks_to_upside"))
    downside_lines = _render_watchlist_entries(watchlist.get("risks_to_downside"))
    if upside_lines is None or downside_lines is None:
        return None

    return "\n\n".join(
        [
            "# Watchlist",
            "## Risks to upside",
            "\n".join(upside_lines),
            "## Risks to downside",
            "\n".join(downside_lines),
        ]
    )


def _render_pass1_market_analysis(raw_analysis_md: str, stance_dict: dict) -> str:
    """Render structured Pass 1 sections, falling back to raw analysis when incomplete."""
    raw_analysis_md = (raw_analysis_md or "").strip()
    regime_md = _render_market_regime_assessment(stance_dict.get("market_regime_assessment"))
    evidence_md = _render_regime_evidence_dashboard(stance_dict.get("regime_evidence"))
    watchlist_md = _render_watchlist(stance_dict.get("watchlist"))
    if not raw_analysis_md or not regime_md or not evidence_md or not watchlist_md:
        return raw_analysis_md

    return "\n\n".join([regime_md, evidence_md, raw_analysis_md, watchlist_md])


# ---------------------------------------------------------------------------
# Pass 2: Portfolio Risk Analysis
# ---------------------------------------------------------------------------


def _build_pass2_system_message(stance_dict: dict) -> str:
    """Build system prompt for Pass 2, injecting stance context into daily_system.md."""
    daily_system_md = load_prompt_file(PROMPTS_DIR / "daily_system.md", "prompts/daily_system.md")
    stance = stance_dict.get("stance", DEFAULT_STANCE)
    leverage = stance_dict.get("target_leverage", DEFAULT_LEVERAGE)
    rationale = stance_dict.get("leverage_rationale", "")
    confidence = stance_dict.get("confidence", "medium")
    drivers = stance_dict.get("drivers", [])

    drivers_md = "\n".join(f"- {d}" for d in drivers) if drivers else "- N/A"

    stance_context = f"""## Today's Market Stance (from Pass 1 Analysis)

**Stance**: {stance}
**Target Leverage**: {leverage:.2f}x
**Confidence**: {confidence}
**Leverage Rationale**: {rationale}
**Key Drivers**:
{drivers_md}

Frame your portfolio risk analysis in the context of this stance:
- Under "{stance}" stance at {leverage:.2f}x leverage, flag risks that would jeopardize this positioning.
- Note whether current portfolio exposures are aligned with or diverge from the target leverage.
- If the sizer output at {leverage:.2f}x creates unusual concentration or constraint binding, highlight it.
"""
    return f"{daily_system_md}\n\n---\n\n{stance_context}"


def _build_pass2_user_message(
    bundle: dict,
    risk_summary_md: str,
    adjustments_md: str,
) -> str:
    """Build the user message for Pass 2 (portfolio risk analysis)."""
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
2. **Position-Level Flags** -- for each position with an actionable signal (deteriorating technicals, momentum divergence, severe drawdown, contrarian gating changes, high beta exposure), describe the concern and severity (low/medium/high)
3. **Portfolio-Level Risks** -- beta neutrality status, gross leverage vs limits, concentration, correlation risks
4. **Actionable Alerts** -- positions where the share adjustment is large or where risk metrics warrant immediate attention
5. **Stance Alignment** -- briefly assess whether the portfolio as sized at the target leverage aligns with the market stance from your system context. Flag any tension between the stance and current risk exposures. Note if the leverage level creates any binding constraints or unusual risk concentrations.

Constraints:
- Cite specific metrics from the data (vol, beta, drawdown %, ROC, MA signals).
- Be direct and concise. No filler.
- Focus on what changed or what is abnormal.
- The share adjustments table is deterministic input, not a final recommendation. Flag unusual adjustments and any source-quality, liquidity, or portfolio-risk reason the recommendations pass should reject or block them.

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
# Response parsing (Pass 2)
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
    """Parse LLM response into (report_md, summary_dict)."""
    if DAILY_SUMMARY_SEPARATOR in text:
        parts = text.split(DAILY_SUMMARY_SEPARATOR, 1)
        report_md = parts[0].strip()
        json_part = parts[1].strip()
        if json_part.startswith("```"):
            json_part = json_part.split("\n", 1)[1] if "\n" in json_part else json_part[3:]
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
# Recommendations pass
# ---------------------------------------------------------------------------


def _build_recommendations_system_message() -> str:
    core_md = load_prompt_file(PROMPTS_DIR / "system.md", "prompts/system.md")
    rec_md = load_prompt_file(PROMPTS_DIR / "recommendations_system.md", "prompts/recommendations_system.md")
    return f"{core_md}\n\n---\n\n{rec_md}"


def _build_recommendations_extra_context(risk_summary_md: str, adjustments_md: str) -> str:
    return f"""## Deterministic Risk Tables

{risk_summary_md}

{adjustments_md}

Use the sizer and adjustment table as evidence, not as authority. Reject or block actions that fail source-quality, thesis, liquidity, or portfolio-risk gates."""


def _generate_daily_recommendations(
    *,
    today_str: str,
    stance_dict: dict,
    data_quality: dict,
    evidence_bundle: dict,
    commentary_md: str,
    risk_summary_md: str,
    adjustments_md: str,
) -> tuple[str, dict]:
    stance = stance_dict.get("stance", DEFAULT_STANCE)
    if data_quality.get("recommendations_blocked"):
        payload = fallback_recommendations_payload(
            "daily",
            today_str,
            stance,
            data_quality,
            status="blocked",
            reason="Critical data quality blocks actionable daily recommendations.",
        )
        return format_recommendations_markdown(payload), payload

    system_msg = _build_recommendations_system_message()
    user_msg = build_recommendations_user_message(
        report_type="daily",
        as_of=today_str,
        stance=stance,
        data_quality=data_quality,
        evidence_bundle=evidence_bundle,
        commentary_md=commentary_md,
        extra_context_md=_build_recommendations_extra_context(risk_summary_md, adjustments_md),
    )
    try:
        raw_text, _ = call_report_llm(system_msg=system_msg, user_msg=user_msg, web_search=False, max_tokens=8192)
        try:
            memo_md, payload = parse_recommendations_response(
                raw_text,
                report_type="daily",
                as_of=today_str,
                stance=stance,
                data_quality=data_quality,
            )
        except Exception as exc:
            log.warning("Daily recommendations validation failed; attempting repair: %s", exc)
            memo_md, payload = repair_recommendations_response(
                raw_text,
                str(exc),
                report_type="daily",
                as_of=today_str,
                stance=stance,
                data_quality=data_quality,
            )
        memo_md = strip_llm_meta(memo_md)
        formatted = format_recommendations_markdown(payload)
        return "\n\n".join(part for part in [memo_md, formatted] if part).strip(), payload
    except Exception as exc:
        log.error("Daily recommendations generation failed: %s", exc, exc_info=True)
        payload = fallback_recommendations_payload(
            "daily",
            today_str,
            stance,
            data_quality,
            status="error",
            reason=f"Recommendation generation failed: {exc}",
        )
        return format_recommendations_markdown(payload), payload


# ---------------------------------------------------------------------------
# Summary merging + stance header
# ---------------------------------------------------------------------------


def _merge_summary(
    stance_dict: dict,
    risk_summary: dict,
    recommendation_payload: dict | None = None,
    data_quality: dict | None = None,
    persisted_recommendations: list[dict] | None = None,
) -> dict:
    """Merge Pass 1 stance dict with Pass 2 risk summary into final summary.json."""
    _normalize_pass1_structured_fields(stance_dict)
    summary = {
        # Stance fields (from Pass 1)
        "stance": stance_dict.get("stance", DEFAULT_STANCE),
        "target_leverage": stance_dict.get("target_leverage", DEFAULT_LEVERAGE),
        "stance_confidence": stance_dict.get("confidence", "low"),
        "leverage_rationale": stance_dict.get("leverage_rationale", ""),
        "six_dimensions": stance_dict.get("six_dimensions", {}),
        "stance_drivers": stance_dict.get("drivers", []),
        "watchlist_triggers": stance_dict.get("watchlist_triggers", []),
        "pass1_parse_error": stance_dict.get("parse_error", False),
        # Risk fields (from Pass 2)
        "risk_level": risk_summary.get("risk_level", "unknown"),
        "top_risks": risk_summary.get("top_risks", []),
        "positions_flagged": risk_summary.get("positions_flagged", []),
        "largest_adjustments": risk_summary.get("largest_adjustments", []),
        "pass2_parse_error": risk_summary.get("parse_error", False),
    }
    if recommendation_payload:
        summary["recommendations"] = {
            "status": recommendation_payload.get("recommendation_status"),
            "critical_data_quality": recommendation_payload.get("critical_data_quality"),
            "blocked_reasons": recommendation_payload.get("blocked_reasons", []),
            "action_count": len(recommendation_payload.get("recommended_actions", [])),
            "actionable_count": sum(
                1 for action in recommendation_payload.get("recommended_actions", []) if action.get("approval_required")
            ),
            "persisted_ids": [r.get("id") for r in persisted_recommendations or []],
            "separator": RECOMMENDATIONS_SEPARATOR,
        }
    if data_quality:
        summary["data_quality"] = data_quality
    return summary


def _build_stance_header_markdown(stance_dict: dict) -> str:
    """Build the stance header section for the final report."""
    _normalize_pass1_structured_fields(stance_dict)
    stance = stance_dict.get("stance", DEFAULT_STANCE)
    leverage = stance_dict.get("target_leverage", DEFAULT_LEVERAGE)
    confidence = stance_dict.get("confidence", "medium")
    rationale = stance_dict.get("leverage_rationale", "")
    parse_error = stance_dict.get("parse_error", False)
    six_dim = stance_dict.get("six_dimensions", {})

    header = f"### Stance: {stance} | Leverage: {leverage:.2f}x | Confidence: {confidence}\n"
    if parse_error:
        header += "\n> **Warning**: Pass 1 analysis failed. Stance and leverage are defaults.\n"
    if rationale:
        header += f"\n{rationale}\n"

    if six_dim:
        header += "\n| Dimension | Assessment |\n|---|---|\n"
        for dim_key, dim_val in six_dim.items():
            label = dim_key.replace("_", " ").title()
            header += f"| {label} | {dim_val} |\n"

    return header


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def write_daily_outputs(
    report_md: str,
    commentary_md: str,
    recommendations_md: str,
    recommendations_payload: dict,
    summary: dict,
    bundle: dict,
    adjustments_df,
    output_dir: Path,
    today: str,
):
    """Write all daily report outputs and archive."""
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "report.md").write_text(report_md, encoding="utf-8")
    (output_dir / "commentary.md").write_text(commentary_md, encoding="utf-8")
    (output_dir / "recommendations.md").write_text(recommendations_md, encoding="utf-8")
    (output_dir / "recommendations.json").write_text(
        json.dumps(recommendations_payload, indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_bundle(bundle, output_dir / "daily_bundle.json")
    if not adjustments_df.empty:
        adjustments_df.to_csv(output_dir / "adjustments.csv", index=False)
    log.info("Wrote daily outputs to %s", output_dir)

    # Index report for semantic search (best-effort)
    try:
        from api.retrieval import index_document

        index_document(
            doc_type="daily_report",
            content=report_md,
            source_path=str(output_dir / "report.md"),
            doc_id=f"daily-{today}",
        )
    except Exception:
        log.debug("Failed to index daily report for retrieval", exc_info=True)

    # Archive
    archive_dir = output_dir / "history" / today
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "report.md").write_text(report_md, encoding="utf-8")
    (archive_dir / "commentary.md").write_text(commentary_md, encoding="utf-8")
    (archive_dir / "recommendations.md").write_text(recommendations_md, encoding="utf-8")
    (archive_dir / "recommendations.json").write_text(
        json.dumps(recommendations_payload, indent=2, default=str),
        encoding="utf-8",
    )
    (archive_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_bundle(bundle, archive_dir / "daily_bundle.json")
    if not adjustments_df.empty:
        adjustments_df.to_csv(archive_dir / "adjustments.csv", index=False)
    log.info("Archived daily to %s", archive_dir)


# ---------------------------------------------------------------------------
# Main — two-pass pipeline
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Automated daily portfolio risk report (two-pass)")
    parser.add_argument("--force", action="store_true", help="Bypass weekday-morning gate")
    parser.add_argument(
        "--book",
        type=float,
        default=None,
        help="Book size in USD (default: portfolio setting, then sum of abs(shares * cost_basis) from the portfolio db)",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Disable web search in Pass 1",
    )
    args = parser.parse_args()

    if not args.force and not os.environ.get("FORCE_RUN") and not _is_weekday_morning_et():
        log.info("Not weekday 09:xx ET — exiting (use --force to override)")
        sys.exit(0)

    today_str = datetime.now(ET).strftime("%Y-%m-%d")
    log.info("=== Daily risk report run starting (%s) ===", today_str)

    # ---------------------------------------------------------------
    # STEP 1: Load portfolio from db
    # ---------------------------------------------------------------
    portfolio_df = load_portfolio()
    log.info("Loaded %d portfolio positions from db", len(portfolio_df))

    # ---------------------------------------------------------------
    # STEP 2: Determine book size
    # ---------------------------------------------------------------
    if args.book:
        book = args.book
    else:
        configured_book = load_configured_book_size()
        if configured_book is not None:
            book = configured_book
            log.info("Book size from portfolio settings: $%s", _format_currency(book))
        else:
            import pandas as pd

            shares = pd.to_numeric(portfolio_df.get("shares"), errors="coerce")
            cost_basis = pd.to_numeric(portfolio_df.get("cost_basis"), errors="coerce")
            position_values = (shares.abs() * cost_basis).dropna()
            if not position_values.empty and float(position_values.sum()) > 0:
                book = float(position_values.sum())
                log.info("Book size from portfolio db (shares * cost_basis): $%s", _format_currency(book))
            else:
                book = 100_000.0
                log.info("Using default book size: $%s", _format_currency(book))

    # ---------------------------------------------------------------
    # STEP 3: Load previous-day summary (feeds Pass 1 context)
    # ---------------------------------------------------------------
    last_daily_json = load_last_daily_summary(HISTORY_DIR)
    if last_daily_json:
        log.info("Loaded previous-day summary from history")
    else:
        log.info("No prior daily summary found in history")

    # ---------------------------------------------------------------
    # STEP 4: Collect market data (12 sources — same as weekly)
    # ---------------------------------------------------------------
    log.info("Collecting market data (12 sources)...")
    t_collect = time.perf_counter()
    market_data = collect_market_data(news_digest_days=DAILY_NEWS_DIGEST_DAYS)
    log.info("Market data collected in %.2fs", time.perf_counter() - t_collect)

    # ---------------------------------------------------------------
    # STEP 5: Build performance tables
    # ---------------------------------------------------------------
    perf_md = build_performance_markdown(market_data)
    market_bundle = serialize_bundle(market_data)

    # ---------------------------------------------------------------
    # STEP 6: PASS 1 — Market Analysis + Stance Determination
    # ---------------------------------------------------------------
    log.info("=== Pass 1: Market Analysis + Stance ===")
    pass1_system = _build_pass1_system_message(last_daily_json)
    pass1_user = _build_pass1_user_message(market_bundle, perf_md)

    pass1_web_search = not args.no_search
    if pass1_web_search:
        log.info("Pass 1 web search enabled with unrestricted domains")
    else:
        log.info("Pass 1 web search disabled")
    stance_dict = None
    market_analysis_md = None
    pass1_citations = []

    try:
        pass1_text, pass1_citations = call_report_llm(
            system_msg=pass1_system,
            user_msg=pass1_user,
            web_search=pass1_web_search,
        )
        market_analysis_md, stance_dict = parse_pass1_response(pass1_text)
        market_analysis_md = strip_llm_meta(market_analysis_md)
        log.info(
            "Pass 1 complete — stance: %s, leverage: %.2f",
            stance_dict["stance"],
            stance_dict["target_leverage"],
        )
    except Exception as e:
        log.error("Pass 1 (market analysis) failed: %s", e, exc_info=True)
        stance_dict = _fallback_pass1_result()
        stance_dict["error"] = str(e)
        market_analysis_md = f"**Pass 1 Error**: Market analysis failed.\n\n```\n{e}\n```"

    target_leverage = stance_dict["target_leverage"]
    log.info("Using target_leverage=%.2f for sizer", target_leverage)

    # ---------------------------------------------------------------
    # STEP 7: Collect portfolio risk data
    # ---------------------------------------------------------------
    log.info("Collecting risk data for %d positions...", len(portfolio_df))
    t_risk = time.perf_counter()
    risk_data = collect_risk_data(portfolio_df)
    log.info("Risk data collected in %.2fs", time.perf_counter() - t_risk)

    # ---------------------------------------------------------------
    # STEP 8: Run sizer at Pass 1's target leverage
    # ---------------------------------------------------------------
    log.info(
        "Running portfolio sizer (book=$%s, leverage=%.2f)...",
        _format_currency(book),
        target_leverage,
    )
    t_sizer = time.perf_counter()
    sizer_result = run_sizer(portfolio_df, book, target_leverage=target_leverage)
    log.info("Sizer completed in %.2fs", time.perf_counter() - t_sizer)

    if sizer_result.get("error"):
        log.error("Sizer failed: %s", sizer_result["error"])

    # ---------------------------------------------------------------
    # STEP 9: Compute share adjustments (deterministic)
    # ---------------------------------------------------------------
    adjustments_df = compute_adjustments(sizer_result, portfolio_df)

    # ---------------------------------------------------------------
    # STEP 10: Build deterministic Markdown sections
    # ---------------------------------------------------------------
    risk_summary_md = build_risk_summary_markdown(risk_data, sizer_result)
    adjustments_md = format_adjustments_markdown(adjustments_df)

    # ---------------------------------------------------------------
    # STEP 11: Serialize risk bundle for Pass 2
    # ---------------------------------------------------------------
    risk_bundle = serialize_bundle(
        {
            "risk_data": risk_data,
            "sizer_summary": {
                k: v for k, v in sizer_result.items() if k not in ("weights_df", "hedges_df", "max_scaled")
            },
        }
    )

    # ---------------------------------------------------------------
    # STEP 12: PASS 2 — Portfolio Risk Analysis
    # ---------------------------------------------------------------
    log.info("=== Pass 2: Portfolio Risk Analysis ===")
    pass2_system = _build_pass2_system_message(stance_dict)
    pass2_user = _build_pass2_user_message(risk_bundle, risk_summary_md, adjustments_md)

    risk_analysis_md = None
    risk_summary_dict = None
    pass2_citations = []

    try:
        pass2_text, pass2_citations = call_report_llm(
            system_msg=pass2_system,
            user_msg=pass2_user,
            web_search=False,
        )
        risk_analysis_md, risk_summary_dict = parse_daily_response(pass2_text)
        risk_analysis_md = strip_llm_meta(risk_analysis_md)
        log.info(
            "Pass 2 complete — risk_level: %s",
            risk_summary_dict.get("risk_level", "unknown"),
        )
    except Exception as e:
        log.error("Pass 2 (portfolio risk) failed: %s", e, exc_info=True)
        risk_analysis_md = f"**Pass 2 Error**: Risk analysis failed.\n\n```\n{e}\n```"
        risk_summary_dict = _fallback_daily_summary()
        risk_summary_dict["error"] = str(e)

    # ---------------------------------------------------------------
    # STEP 13: Compose commentary report
    # ---------------------------------------------------------------
    stance_header_md = _build_stance_header_markdown(stance_dict)
    market_analysis_report_md = _render_pass1_market_analysis(market_analysis_md, stance_dict)

    # Collect all citations
    all_citations = pass1_citations + pass2_citations

    commentary_md = "\n\n".join(
        [
            f"# Daily Commentary Report — {today_str}",
            "---",
            "## Market Stance",
            stance_header_md,
            "---",
            "## Market Analysis",
            market_analysis_report_md,
            "---",
            perf_md,
            "---",
            risk_summary_md,
            adjustments_md,
            "---",
            "## AI Risk Analysis",
            risk_analysis_md,
        ]
    )

    if all_citations:
        sources_lines = ["\n\n---\n\n## Sources\n"]
        seen = set()
        for title, url in all_citations:
            if url not in seen:
                seen.add(url)
                sources_lines.append(f"- [{title}]({url})")
        commentary_md += "\n".join(sources_lines)

    # ---------------------------------------------------------------
    # STEP 14: Data quality + recommendations
    # ---------------------------------------------------------------
    raw_recommendation_bundle = {
        "market_data": market_data,
        "portfolio_positions": portfolio_df.to_dict(orient="records"),
        "risk_data": risk_data,
        "sizer_summary": {k: v for k, v in sizer_result.items() if k not in ("weights_df", "hedges_df", "max_scaled")},
        "risk_summary": risk_summary_dict,
        "stance": stance_dict,
    }
    data_quality = assess_report_data_quality(raw_recommendation_bundle, "daily")
    recommendation_evidence_bundle = serialize_bundle(
        {
            **raw_recommendation_bundle,
            "data_quality": data_quality,
        }
    )

    recommendations_md, recommendations_payload = _generate_daily_recommendations(
        today_str=today_str,
        stance_dict=stance_dict,
        data_quality=data_quality,
        evidence_bundle=recommendation_evidence_bundle,
        commentary_md=commentary_md,
        risk_summary_md=risk_summary_md,
        adjustments_md=adjustments_md,
    )

    persisted_recommendations = persist_recommendations(
        recommendations_payload,
        source_report_path=str(OUTPUT_DIR / "recommendations.md"),
        source_json_path=str(OUTPUT_DIR / "recommendations.json"),
        prompt_metadata={
            "model": MODEL_HIGH,
            "prompt_hash": stable_hash(
                {
                    "system": _build_recommendations_system_message(),
                    "separator": RECOMMENDATIONS_SEPARATOR,
                }
            ),
            "input_hash": stable_hash(recommendation_evidence_bundle),
            "validation_status": "ok" if recommendations_payload.get("recommendation_status") != "error" else "error",
            "source_quality_summary": data_quality,
        },
    )

    report_md = "\n\n".join(
        [
            f"# Daily Portfolio Report — {today_str}",
            "---",
            commentary_md,
            "---",
            f"# Daily Recommendations — {today_str}",
            recommendations_md,
        ]
    )

    merged_summary = _merge_summary(
        stance_dict,
        risk_summary_dict,
        recommendations_payload,
        data_quality,
        persisted_recommendations,
    )

    # Full bundle for archive includes both market + risk data
    full_bundle = serialize_bundle(
        {
            **raw_recommendation_bundle,
            "data_quality": data_quality,
            "recommendations": recommendations_payload,
        }
    )

    write_daily_outputs(
        report_md,
        commentary_md,
        recommendations_md,
        recommendations_payload,
        merged_summary,
        full_bundle,
        adjustments_df,
        OUTPUT_DIR,
        today_str,
    )

    issue_title = f"Daily Report — {today_str} | {stance_dict['stance']}"
    issue_url = None
    try:
        issue_url = create_github_issue(issue_title, report_md)
    except Exception as e:
        log.error("GitHub Issue creation failed: %s", e, exc_info=True)
    if issue_url:
        metadata = {"issue_url": issue_url, "report_type": "daily", "as_of": today_str}
        (OUTPUT_DIR / "report_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        archive_dir = OUTPUT_DIR / "history" / today_str
        archive_dir.mkdir(parents=True, exist_ok=True)
        (archive_dir / "report_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    try:
        outcome_summary = evaluate_due_recommendations()
        log.info("Recommendation outcome evaluation: %s", outcome_summary)
    except Exception:
        log.warning("Recommendation outcome evaluation failed", exc_info=True)

    log.info("=== Daily risk report run complete ===")


if __name__ == "__main__":
    main()
