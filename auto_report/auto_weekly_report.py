#!/usr/bin/env python3
"""
Automated weekly market report.

Orchestrates data collection from existing modules, calls Claude to generate
a Markdown report, writes outputs, archives to history, and creates a GitHub Issue.

Run:
    python auto_report/auto_weekly_report.py --force   # bypass Friday-afternoon gate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Centralised path setup — replaces inline sys.path block
sys.path.insert(0, str(PROJECT_ROOT))
from paths import setup_paths

setup_paths()

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from auto_report.shared import (  # noqa: E402
    call_claude,
    create_github_issue,
    load_prompt_file,
    serialize_bundle,
    slim_error,
    strip_llm_meta,
)
from auto_report.shared import (
    write_bundle as _write_bundle_to_path,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("auto_weekly_report")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ET = ZoneInfo("America/New_York")
OUTPUT_DIR = SCRIPT_DIR / "outputs"
HISTORY_DIR = OUTPUT_DIR / "history"
PROMPTS_DIR = SCRIPT_DIR / "prompts"
SUMMARY_SEPARATOR = "<!-- SUMMARY_JSON -->"
THESIS_SEPARATOR = "<!-- THESIS_SUMMARY_JSON -->"
THESES_DIR = PROJECT_ROOT / "investment_theses"

DEFAULT_NEWS_SOURCES = [
    "bloomberg.com",
    "cnbc.com",
    "federalreserve.gov",
    "reuters.com",
    "wsj.com",
]

RULES_TEXT = """
STRICT FORMATTING RULES (Apply these to the data provided below):

MARKET BREADTH THRESHOLDS:
- 200-day MA: Flag if > 80% or < 15%
- 20-day MA: Flag if > 80% or < 20%
- 20-day Highs: Flag if > 50%
- 20-day Lows: Flag if > 50% (Capitulation signal)
- 52-week Highs: Flag if > 15%
- 52-week Lows: Flag if > 15%
- 24-week Highs: Flag if > 20%
- 24-week Lows: Flag if > 20%

TOP 50 S&P 500 BREADTH:
- Simply state the % below 50-DMA, % with >=3 distribution days (last 20), and % that broke prior 20-day low in last 5 days.

VIX TERM STRUCTURE:
- Signal is 'Complacency' if 3M/1M Ratio >= 1.25
- Signal is 'Fear' if Ratio < 1.0
- Otherwise 'Neutral'
"""

# ---------------------------------------------------------------------------
# Pure helpers (copied from api/routers/weekly_report.py — no FastAPI deps)
# ---------------------------------------------------------------------------


def _format_level(value: float, decimals_if_lt_100: int = 4) -> str:
    try:
        v = float(value)
    except Exception:
        return "N/A"
    if abs(v) >= 100:
        return f"{v:,.2f}"
    return f"{v:.{decimals_if_lt_100}f}"


def _pct_change(start: float, latest: float) -> float | None:
    try:
        s = float(start)
        l = float(latest)  # noqa: E741
    except Exception:
        return None
    if s == 0:
        return None
    return ((l - s) / s) * 100.0


def _build_perf_table(
    title: str,
    rows: list[tuple[str, float, float]],
    decimals_if_lt_100: int = 4,
) -> str:
    if not rows:
        return f"### {title}\n\n_No data available._\n"
    header = f"### {title}\n\n| Asset | Start | Latest | Change |\n|---|---:|---:|---:|\n"
    body_lines = []
    for name, start, latest in rows:
        pct = _pct_change(start, latest)
        pct_str = "N/A" if pct is None else f"{pct:+.2f}%"
        body_lines.append(
            f"| {name} | {_format_level(start, decimals_if_lt_100)} | {_format_level(latest, decimals_if_lt_100)} | {pct_str} |"
        )
    return header + "\n".join(body_lines) + "\n"


def _build_key_ratios_table(rows: list[tuple[str, dict]]) -> str:
    header = "## Key Ratios (Past Week)\n\n| Ratio | Start | Latest | Change | Date Range |\n|---|---:|---:|---:|---|\n"
    body_lines: list[str] = []
    for name, r in rows:
        if not isinstance(r, dict):
            body_lines.append(f"| {name} | N/A | N/A | N/A | N/A |")
            continue
        if "error" in r:
            err = str(r.get("error") or "Unknown error").strip()
            body_lines.append(f"| {name} | N/A | N/A | ERROR | {err} |")
            continue
        stats = r.get("stats") if isinstance(r.get("stats"), dict) else {}
        start_ratio = stats.get("start_ratio")
        end_ratio = stats.get("end_ratio")
        change = stats.get("change_pct")
        try:
            start_s = _format_level(float(start_ratio), decimals_if_lt_100=6)
            end_s = _format_level(float(end_ratio), decimals_if_lt_100=6)
        except Exception:
            start_s = "N/A"
            end_s = "N/A"
        try:
            change_pct = float(change) * 100.0
            change_s = f"{change_pct:+.2f}%"
        except Exception:
            change_s = "N/A"
        date_range = f"{stats.get('start_date', 'N/A')} → {stats.get('end_date', 'N/A')}"
        body_lines.append(f"| {name} | {start_s} | {end_s} | {change_s} | {date_range} |")
    return header + "\n".join(body_lines) + "\n"


def _insert_weekly_performance(report_md: str, perf_md: str) -> str:
    perf_md = (perf_md or "").strip()
    report_md = (report_md or "").strip()
    if not perf_md:
        return report_md
    lines = report_md.splitlines()
    if lines and lines[0].startswith("# "):
        first = lines[0]
        rest = "\n".join(lines[1:]).lstrip("\n")
        return f"{first}\n\n{perf_md}\n\n{rest}".strip()
    return f"{perf_md}\n\n{report_md}".strip()


_slim_error = slim_error


def _slim_ratio_result(value):
    if not isinstance(value, dict):
        return value
    if "error" in value:
        return _slim_error(value)
    stats = value.get("stats") if isinstance(value.get("stats"), dict) else None
    return {
        "ratio_label": value.get("ratio_label"),
        "name_a": value.get("name_a"),
        "name_b": value.get("name_b"),
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Schedule gate
# ---------------------------------------------------------------------------


def _is_friday_afternoon_et() -> bool:
    now_et = datetime.now(ET)
    return now_et.weekday() == 4 and now_et.hour == 16


def load_last_week_summary(history_dir: Path) -> str | None:
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
# Thesis monitoring helpers
# ---------------------------------------------------------------------------


def load_theses() -> dict[str, str | None]:
    """Load investment thesis markdown files for all portfolio tickers."""
    import csv

    portfolio_csv = PROJECT_ROOT / "portfolio" / "portfolio.csv"
    tickers: list[str] = []
    with open(portfolio_csv, newline="") as f:
        for row in csv.DictReader(f):
            t = row.get("ticker", "").strip()
            if t:
                tickers.append(t)

    theses: dict[str, str | None] = {}
    for ticker in tickers:
        thesis_path = THESES_DIR / f"{ticker}.md"
        if thesis_path.exists():
            try:
                content = thesis_path.read_text(encoding="utf-8").strip()
                theses[ticker] = content if content else None
            except Exception as e:
                log.warning("Failed to read thesis for %s: %s", ticker, e)
                theses[ticker] = None
        else:
            log.debug("No thesis file for %s", ticker)
            theses[ticker] = None
    return theses


def filter_news_7day(news_data: dict) -> dict[str, list[dict]]:
    """Filter portfolio news to last 7 calendar days, grouped by ticker."""
    import email.utils
    from datetime import UTC

    now_utc = datetime.now(UTC)
    cutoff = now_utc - timedelta(days=7)

    by_ticker = news_data.get("by_ticker", {})
    filtered: dict[str, list[dict]] = {}
    for ticker, articles in by_ticker.items():
        recent: list[dict] = []
        for article in articles:
            seendate = article.get("seendate", "")
            if not seendate:
                continue
            parsed = None
            try:
                parsed = datetime.fromisoformat(seendate.replace("Z", "+00:00"))
            except Exception:
                try:
                    parsed = email.utils.parsedate_to_datetime(seendate)
                except Exception:
                    continue
            if parsed is None:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            if parsed >= cutoff:
                recent.append(article)
        filtered[ticker] = recent
    return filtered


def collect_thesis_data() -> dict:
    """Collect all data needed for thesis monitoring."""
    import csv

    results: dict = {}

    # 1. Load theses
    results["theses"] = load_theses()

    # 2. Load portfolio positions
    portfolio_csv = PROJECT_ROOT / "portfolio" / "portfolio.csv"
    with open(portfolio_csv, newline="") as f:
        results["portfolio"] = [r for r in csv.DictReader(f) if r.get("ticker")]

    tickers = [p["ticker"] for p in results["portfolio"]]

    # 3. Portfolio news (7-day filtered)
    try:
        from portfolio_news import get_data as get_news_data

        t0 = time.perf_counter()
        news_data = get_news_data(refresh=False)
        results["news_7day"] = filter_news_7day(news_data)
        log.info("thesis news fetched and filtered in %.2fs", time.perf_counter() - t0)
    except Exception as e:
        log.warning("thesis news fetch failed: %s", e, exc_info=True)
        results["news_7day"] = {}

    # 4. Technical analysis (per-ticker summaries)
    try:
        from technical_analysis import get_data as get_ta_data

        t0 = time.perf_counter()
        ta_results: dict = {}
        for ticker in tickers:
            try:
                ta = get_ta_data(ticker, lookback="2Y")
                ta_results[ticker] = ta.get("summary", ta)
            except Exception as exc:
                ta_results[ticker] = {"error": str(exc)}
        log.info("thesis TA fetched in %.2fs", time.perf_counter() - t0)
        results["technical_analysis"] = ta_results
    except Exception as e:
        log.warning("thesis TA fetch failed: %s", e, exc_info=True)
        results["technical_analysis"] = {}

    # 5. Price momentum (batch)
    try:
        from momentum import get_data as get_momentum_data

        t0 = time.perf_counter()
        momentum = get_momentum_data()
        log.info("thesis momentum fetched in %.2fs", time.perf_counter() - t0)
        results["momentum"] = momentum
    except Exception as e:
        log.warning("thesis momentum fetch failed: %s", e, exc_info=True)
        results["momentum"] = {}

    return results


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_data() -> dict:
    results = {}
    week_start = (datetime.now() - timedelta(days=7)).date().isoformat()

    # 1. Indices
    try:
        from index_dashboard import INDEX_ORDER
        from index_dashboard import get_data as get_index_data

        t0 = time.perf_counter()
        indices = get_index_data("This Week")
        log.info("indices fetched in %.2fs", time.perf_counter() - t0)
        results["indices"] = {"data": indices, "order": list(INDEX_ORDER)}
    except Exception as e:
        log.warning("indices fetch failed: %s", e, exc_info=True)
        results["indices"] = {"error": str(e)}

    # 2. FX
    try:
        from fx_dashboard import PAIR_ORDER
        from fx_dashboard import get_data as get_fx_data

        t0 = time.perf_counter()
        fx = get_fx_data("This Week")
        log.info("fx fetched in %.2fs", time.perf_counter() - t0)
        results["fx"] = {"data": fx, "order": list(PAIR_ORDER)}
    except Exception as e:
        log.warning("fx fetch failed: %s", e, exc_info=True)
        results["fx"] = {"error": str(e)}

    # 3. Commodities
    try:
        from commodities_dashboard import (
            COMMODITY_ORDER,
        )
        from commodities_dashboard import (
            get_data as get_commodity_data,
        )

        t0 = time.perf_counter()
        commodities = get_commodity_data("This Week")
        log.info("commodities fetched in %.2fs", time.perf_counter() - t0)
        results["commodities"] = {
            "data": commodities,
            "order": list(COMMODITY_ORDER),
        }
    except Exception as e:
        log.warning("commodities fetch failed: %s", e, exc_info=True)
        results["commodities"] = {"error": str(e)}

    # 4. Market Breadth
    try:
        from market_breadth import get_data as get_breadth_data

        t0 = time.perf_counter()
        breadth = get_breadth_data(period="1y")
        # Drop raw ticker list to keep bundle lean
        breadth.pop("tickers", None)
        log.info("breadth fetched in %.2fs", time.perf_counter() - t0)
        results["breadth"] = breadth
    except Exception as e:
        log.warning("breadth fetch failed: %s", e, exc_info=True)
        results["breadth"] = {"error": str(e)}

    # 5. Top 50 Breadth
    try:
        from top50_breadth import get_data as get_top50_data

        t0 = time.perf_counter()
        top50 = get_top50_data()
        log.info("top50 breadth fetched in %.2fs", time.perf_counter() - t0)
        results["top50"] = top50
    except Exception as e:
        log.warning("top50 breadth fetch failed: %s", e, exc_info=True)
        results["top50"] = {"error": str(e)}

    # 6. VIX Term Structure
    try:
        from vix_term_structure import get_data as get_vix_data

        t0 = time.perf_counter()
        vix = get_vix_data()
        log.info("vix term structure fetched in %.2fs", time.perf_counter() - t0)
        results["vix"] = vix
    except Exception as e:
        log.warning("vix term structure fetch failed: %s", e, exc_info=True)
        results["vix"] = {"error": str(e)}

    # 7. Sector Metrics — pre-process weights_df
    try:
        from sector_metrics import get_data as get_sector_data

        t0 = time.perf_counter()
        sector = get_sector_data()
        weights_df = sector.get("weights_df")
        if weights_df is not None:
            import pandas as pd

            if isinstance(weights_df, pd.DataFrame):
                df = weights_df.reset_index()
                if "index" in df.columns and "Sector" not in df.columns:
                    df = df.rename(columns={"index": "Sector"})
                df = df.round(2)
                sector["weights_summary"] = df.to_dict(orient="records")
                del sector["weights_df"]
        log.info("sector metrics fetched in %.2fs", time.perf_counter() - t0)
        results["sector"] = sector
    except Exception as e:
        log.warning("sector metrics fetch failed: %s", e, exc_info=True)
        results["sector"] = {"error": str(e)}

    # 8. Positioning
    try:
        from positioning import DATASETS, DEFAULT_DOMAIN, fetch_multiple_instruments

        t0 = time.perf_counter()
        pos = fetch_multiple_instruments(
            domain=DEFAULT_DOMAIN,
            dataset_id=DATASETS.get("tff_futures_only", "tff_futures_only"),
            app_token=os.environ.get("SODA_APP_TOKEN"),
            instruments=["SP500", "NASDAQ", "US10Y", "EUR", "GOLD", "OIL"],
            start="2015-01-01",
            end=None,
        )
        log.info("positioning fetched in %.2fs", time.perf_counter() - t0)
        results["positioning"] = pos
    except Exception as e:
        log.warning("positioning fetch failed: %s", e, exc_info=True)
        results["positioning"] = {"error": str(e)}

    # 9. Ratios
    try:
        from technical_analysis import get_ratio_data

        t0 = time.perf_counter()
        silver_gold = _slim_ratio_result(get_ratio_data("SI=F", "GC=F", start_date=week_start))
        sp_eq = _slim_ratio_result(get_ratio_data("^GSPC", "RSP", start_date=week_start))
        log.info("ratios fetched in %.2fs", time.perf_counter() - t0)
        results["ratios"] = {"silver_gold": silver_gold, "sp500_rsp": sp_eq}
    except Exception as e:
        log.warning("ratios fetch failed: %s", e, exc_info=True)
        results["ratios"] = {"error": str(e)}

    # 10. Economic Growth
    try:
        from economic_growth import get_data as get_econ_growth_data

        t0 = time.perf_counter()
        econ_growth = get_econ_growth_data()
        log.info("economic growth fetched in %.2fs", time.perf_counter() - t0)
        results["economic_growth"] = econ_growth
    except Exception as e:
        log.warning("economic growth fetch failed: %s", e, exc_info=True)
        results["economic_growth"] = {"error": str(e)}

    # 11. Liquidity
    try:
        from liquidity import get_snapshot as get_liquidity_snapshot

        t0 = time.perf_counter()
        liquidity = get_liquidity_snapshot()
        log.info("liquidity fetched in %.2fs", time.perf_counter() - t0)
        results["liquidity"] = liquidity
    except Exception as e:
        log.warning("liquidity fetch failed: %s", e, exc_info=True)
        results["liquidity"] = {"error": str(e)}

    # 12. Industry Monitor
    try:
        from industry_monitor import get_data as get_industry_data

        t0 = time.perf_counter()
        industry = get_industry_data(refresh=False)
        log.info("industry monitor fetched in %.2fs", time.perf_counter() - t0)
        results["industry"] = industry
    except Exception as e:
        log.warning("industry monitor fetch failed: %s", e, exc_info=True)
        results["industry"] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Serialization (thin wrappers around shared utilities)
# ---------------------------------------------------------------------------


def write_bundle(bundle: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return _write_bundle_to_path(bundle, output_dir / "weekly_bundle.json")


# ---------------------------------------------------------------------------
# Performance tables (from raw, pre-serialized data)
# ---------------------------------------------------------------------------


def _series_map_to_rows(series_map: dict, order: list[str] | None) -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    if not isinstance(series_map, dict) or not series_map:
        return rows
    try:
        import pandas as pd
    except Exception:
        pd = None  # type: ignore[assignment]
    names = order or list(series_map.keys())
    for name in names:
        series = series_map.get(name)
        if series is None:
            continue
        try:
            if pd is not None and isinstance(series, pd.Series):
                s = series.dropna()
                if s.empty:
                    continue
                start = float(s.iloc[0])
                latest = float(s.iloc[-1])
            else:
                start = float(series[0])
                latest = float(series[-1])
            rows.append((str(name), start, latest))
        except Exception:
            continue
    return rows


def build_performance_markdown(raw_data: dict) -> str:
    # Extract data + orders
    idx = raw_data.get("indices", {})
    idx_data = idx.get("data", {}) if isinstance(idx, dict) else {}
    idx_order = idx.get("order") if isinstance(idx, dict) else None

    fx = raw_data.get("fx", {})
    fx_data = fx.get("data", {}) if isinstance(fx, dict) else {}
    fx_order = fx.get("order") if isinstance(fx, dict) else None

    com = raw_data.get("commodities", {})
    com_data = com.get("data", {}) if isinstance(com, dict) else {}
    com_order = com.get("order") if isinstance(com, dict) else None

    indices_rows = _series_map_to_rows(
        idx_data.get("indices", {}) if isinstance(idx_data, dict) else {},
        idx_order,
    )
    fx_rows = _series_map_to_rows(
        fx_data.get("pairs", {}) if isinstance(fx_data, dict) else {},
        fx_order,
    )
    commodities_rows = _series_map_to_rows(
        com_data.get("commodities", {}) if isinstance(com_data, dict) else {},
        com_order,
    )

    perf_md = "\n\n".join(
        [
            "## Weekly Performance",
            _build_perf_table("Indices", indices_rows, decimals_if_lt_100=2).strip(),
            _build_perf_table("FX", fx_rows, decimals_if_lt_100=4).strip(),
            _build_perf_table("Commodities", commodities_rows, decimals_if_lt_100=4).strip(),
        ]
    ).strip()

    # Key ratios
    ratios = raw_data.get("ratios", {})
    if isinstance(ratios, dict) and "error" not in ratios:
        silver_gold = ratios.get("silver_gold", {})
        sp_eq = ratios.get("sp500_rsp", {})
    else:
        silver_gold = ratios if isinstance(ratios, dict) else {}
        sp_eq = {}

    ratios_md = _build_key_ratios_table([("Silver/Gold", silver_gold), ("S&P 500 / RSP", sp_eq)]).strip()
    perf_md = f"{perf_md}\n\n{ratios_md}".strip()

    log.info(
        "performance tables built (indices=%d fx=%d commodities=%d)",
        len(indices_rows),
        len(fx_rows),
        len(commodities_rows),
    )
    return perf_md


# ---------------------------------------------------------------------------
# Prepare bundle for Claude prompt (strip bulky fields)
# ---------------------------------------------------------------------------


def _prepare_prompt_bundle(bundle: dict) -> dict:
    """Return a slimmed copy of the bundle for the Claude prompt.

    Strips bulky intraday series (indices/fx/commodities) down to
    start+latest values, and removes heavy fields from top50.
    """
    import copy

    prompt_bundle = copy.deepcopy(bundle)

    # --- Strip intraday series from dashboards ---
    # The serialized bundle contains full 15-min intraday Series
    # (~130 points per instrument) which bloat the prompt.  The
    # performance markdown table already captures start/latest, so
    # Claude only needs summary values per instrument.
    _SERIES_KEYS = {
        "indices": "indices",
        "fx": "pairs",
        "commodities": "commodities",
    }
    for block_key, series_key in _SERIES_KEYS.items():
        block = prompt_bundle.get(block_key)
        if not isinstance(block, dict) or "error" in block:
            continue
        data = block.get("data")
        if not isinstance(data, dict):
            continue
        series_map = data.get(series_key)
        if not isinstance(series_map, dict):
            continue
        for name, series_list in list(series_map.items()):
            if isinstance(series_list, list) and len(series_list) > 2:
                first = series_list[0]
                last = series_list[-1]
                series_map[name] = {"start": first, "latest": last}
        # Drop the order list — not needed by Claude
        block.pop("order", None)

    # --- Strip heavy fields from top50 ---
    top50 = prompt_bundle.get("top50")
    if isinstance(top50, dict):
        top50.pop("raw_df", None)
        top50.pop("tickers_below_50dma", None)
        top50.pop("tickers_3plus_dist", None)
        top50.pop("tickers_broke_20low", None)

    # --- Strip heavy fields from liquidity ---
    liq = prompt_bundle.get("liquidity")
    if isinstance(liq, dict) and "error" not in liq:
        liq.pop("df_weekly", None)
        liq.pop("composite_series", None)

    # --- Strip per-company detail from industry ---
    ind = prompt_bundle.get("industry")
    if isinstance(ind, dict) and "error" not in ind:
        by_sector = ind.get("by_sector")
        if isinstance(by_sector, dict):
            for sector_data in by_sector.values():
                if isinstance(sector_data, dict):
                    sector_data.pop("companies", None)
        ind.pop("counts", None)

    return prompt_bundle


# call_claude is imported from auto_report.shared


def _dedupe_citations(citations: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen_urls: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for title, url in citations:
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append((title, url))
    return deduped


def _append_sources_section(report_md: str, citations: list[tuple[str, str]]) -> str:
    unique_citations = _dedupe_citations(citations)
    if not unique_citations:
        return report_md
    sources_lines = ["\n\n---\n\n## Sources\n"]
    for title, url in unique_citations:
        sources_lines.append(f"- [{title}]({url})")
    return report_md + "\n".join(sources_lines)


def _build_user_message(bundle: dict, perf_md: str, web_search: bool = True) -> str:
    prompt_bundle = _prepare_prompt_bundle(bundle)
    bundle_json = json.dumps(prompt_bundle, indent=2, default=str)

    search_instruction = ""
    if web_search:
        search_instruction = """
Before writing the report, use the web search tool to find the key market-moving
news from the past week (Fed decisions, economic data releases, geopolitical events,
major earnings, trade policy) that explain the moves in the data below. Weave this
context into each relevant section and cite your sources for news-driven claims.
"""

    return f"""Here is this week's market data bundle:

```json
{bundle_json}
```

{perf_md}

{RULES_TEXT}
{search_instruction}
Write the weekly report with these exact sections:
1. **Executive Summary** — max 5 bullets
2. **Market Moves & Regime Shifts**
3. **Macro Data Highlights**
4. **Positioning**
5. **Key Risks & Signposts** — include specific thresholds/triggers
6. **Recommended Actions** — exactly one stance
7. **What Would Change the Stance** — if-then pivots

Constraints:
- Cite metrics from the data for major claims.
- If evidence is mixed, say so and define what would disambiguate.
- Keep it concise. No filler.

After the report, output the separator `{SUMMARY_SEPARATOR}` on its own line, then a JSON block:
```json
{{
  "stance": "<bullish|bearish|neutral|cautious>",
  "confidence": "<high|medium|low>",
  "drivers": ["<top 3-5 drivers>"],
  "watchlist_triggers": ["<3-5 specific triggers that would change stance>"]
}}
```

End immediately after the JSON. No assistant meta text."""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _fallback_summary() -> dict:
    return {
        "stance": "unknown",
        "confidence": "low",
        "drivers": [],
        "watchlist_triggers": [],
        "parse_error": True,
    }


def parse_response(text: str) -> tuple[str, dict]:
    if SUMMARY_SEPARATOR in text:
        parts = text.split(SUMMARY_SEPARATOR, 1)
        report_md = parts[0].strip()
        json_part = parts[1].strip()
        # Strip markdown code fences
        if json_part.startswith("```"):
            json_part = json_part.split("\n", 1)[1] if "\n" in json_part else json_part[3:]
        if json_part.endswith("```"):
            json_part = json_part[:-3]
        json_part = json_part.strip()
        try:
            summary = json.loads(json_part)
        except json.JSONDecodeError:
            log.warning("Failed to parse summary JSON from Claude response")
            summary = _fallback_summary()
    else:
        log.warning("No summary separator found in Claude response")
        report_md = text.strip()
        summary = _fallback_summary()
    return report_md, summary


# ---------------------------------------------------------------------------
# Thesis monitoring prompt, parsing, and merge
# ---------------------------------------------------------------------------


def _build_thesis_prompt(thesis_data: dict, web_search: bool = False) -> tuple[str, str]:
    """Build (system_msg, user_msg) for the thesis monitoring Claude call."""
    system_msg = (
        "You are an investment analyst evaluating portfolio positions against their "
        "original investment theses. For each position, determine whether recent data "
        "(news, technicals, momentum) supports, challenges, or is neutral to the thesis. "
        "Be specific about which data points matter and why. Focus on material developments "
        "that could change the thesis, not noise. Pay special attention to any earnings-related "
        "developments in the news flow."
    )

    theses = thesis_data.get("theses", {})
    news_7day = thesis_data.get("news_7day", {})
    ta = thesis_data.get("technical_analysis", {})
    momentum_data = thesis_data.get("momentum", {})
    portfolio = thesis_data.get("portfolio", [])

    # Build momentum lookup: ticker -> metrics
    momentum_by_ticker: dict = {}
    if isinstance(momentum_data, dict) and "results" in momentum_data:
        for r in momentum_data["results"]:
            momentum_by_ticker[r["ticker"]] = {
                k: v
                for k, v in r.items()
                if k
                in (
                    "avg20_roc63",
                    "avg20_vol_roc63",
                    "rel_roc42",
                    "avg10_rel_roc",
                    "benchmark",
                    "close",
                    "direction",
                )
            }

    # Build per-ticker sections
    ticker_sections: list[str] = []
    for pos in portfolio:
        ticker = pos["ticker"]
        direction = pos.get("direction", "long")
        conviction = pos.get("conviction", "")
        asset = pos.get("asset", "equity")
        distressed = pos.get("distressed", "false")

        section_parts = [
            f"### {ticker} (direction: {direction}, conviction: {conviction}, asset: {asset}, distressed: {distressed})"
        ]

        # Thesis
        thesis_text = theses.get(ticker)
        if thesis_text:
            section_parts.append(f"\n**Investment Thesis:**\n{thesis_text}")
        else:
            section_parts.append("\n**Investment Thesis:** _No thesis file provided._")

        # News (7-day)
        ticker_news = news_7day.get(ticker, []) if isinstance(news_7day, dict) else []
        if ticker_news:
            news_lines = [f"\n**Recent News (7-day, {len(ticker_news)} articles):**"]
            for article in ticker_news[:10]:  # Cap at 10 per ticker for token control
                date_str = article.get("seendate", "")[:10]
                source = article.get("source", "Unknown")
                title = article.get("title", "No title")
                url = article.get("url", "")
                line = f"- [{date_str}] [{source}] {title}"
                if url:
                    line += f" ({url})"
                news_lines.append(line)
            section_parts.append("\n".join(news_lines))
        else:
            section_parts.append("\n**Recent News (7-day):** _No articles found._")

        # Technical Analysis
        ticker_ta = ta.get(ticker) if isinstance(ta, dict) else None
        if isinstance(ticker_ta, list):
            ta_lines = ["\n**Technical Signals:**"]
            for signal in ticker_ta:
                ta_lines.append(
                    f"- {signal.get('Indicator', '?')}: {signal.get('Value', '?')} "
                    f"({signal.get('Signal', '?')}, {signal.get('Bias', '?')})"
                )
            section_parts.append("\n".join(ta_lines))
        elif isinstance(ticker_ta, dict) and "error" in ticker_ta:
            section_parts.append(f"\n**Technical Signals:** _Error: {ticker_ta['error']}_")
        else:
            section_parts.append("\n**Technical Signals:** _Not available._")

        # Momentum
        ticker_mom = momentum_by_ticker.get(ticker)
        if ticker_mom:
            avg_roc = ticker_mom.get("avg20_roc63")
            rel_roc = ticker_mom.get("rel_roc42")
            avg_rel = ticker_mom.get("avg10_rel_roc")
            bench = ticker_mom.get("benchmark", "SPY")
            parts = ["\n**Momentum:**"]
            if avg_roc is not None:
                parts.append(f"avg20_roc63={avg_roc:.2f}%")
            if rel_roc is not None:
                parts.append(f"rel_roc42={rel_roc:.2f}%")
            if avg_rel is not None:
                parts.append(f"avg10_rel_roc={avg_rel:.2f}%")
            parts.append(f"(vs {bench})")
            section_parts.append(" ".join(parts))
        else:
            section_parts.append("\n**Momentum:** _Not available._")

        ticker_sections.append("\n".join(section_parts))

    tickers_with_theses = [t for t, v in theses.items() if v is not None]
    tickers_without = [t for t, v in theses.items() if v is None]
    search_instruction = ""
    if web_search:
        search_instruction = """
Supplement the ticker-level RSS/IBKR headlines above with web search when needed to find
material developments from the past week, especially earnings, guidance changes, financings,
regulatory actions, M&A, or other thesis-relevant company news. Cite sources for any claim
that depends on web search.
"""

    user_msg = f"""Evaluate each portfolio position below against its investment thesis.

**Tickers with thesis files:** {", ".join(tickers_with_theses) or "None"}
**Tickers without thesis files:** {", ".join(tickers_without) or "None"}

---

{"---\n".join(ticker_sections)}

---

{search_instruction}

For each ticker, provide:
1. **Thesis Status**: strengthen | neutral | weaken | insufficient-data
2. **Technical Read**: improving | mixed | deteriorating
3. **Fundamental Read**: supportive | mixed | contradictory | insufficient-data
4. **Key Developments**: 1-3 specific data points from the news, technicals, or momentum
5. **Earnings Note**: Flag any earnings-related development if found, otherwise omit
6. **Action Signal**: hold | monitor | reassess | urgent review (with brief rationale)

Write a concise "## Portfolio Thesis Monitoring" section with a sub-section for each ticker (3-5 sentences max per ticker).

After all ticker evaluations, output the separator `{THESIS_SEPARATOR}` on its own line, then a JSON block:
```json
{{
  "thesis_evaluations": [
    {{
      "ticker": "<TICKER>",
      "thesis_status": "<strengthen|neutral|weaken|insufficient-data>",
      "technical_read": "<improving|mixed|deteriorating>",
      "fundamental_read": "<supportive|mixed|contradictory|insufficient-data>",
      "action": "<hold|monitor|reassess|urgent review>",
      "confidence": "<high|medium|low>",
      "key_developments": ["<1-3 evidence points>"],
      "earnings_note": "<note or null>",
      "risk_flag": "<emerging risk or null>"
    }}
  ],
  "positions_reviewed": ["<all tickers>"],
  "thesis_strengthened": ["<tickers where thesis_status is strengthen>"],
  "thesis_weakened": ["<tickers where thesis_status is weaken>"],
  "positions_needing_reassessment": ["<tickers with action reassess or urgent review>"],
  "missing_theses": ["<tickers without thesis files>"],
  "material_developments": [
    {{
      "ticker": "<TICKER>",
      "type": "<supports_thesis|contradicts_thesis|new_risk|earnings>",
      "summary": "<one-line summary>"
    }}
  ]
}}
```

End immediately after the JSON. No assistant meta text."""

    return system_msg, user_msg


def _fallback_thesis_summary() -> dict:
    return {
        "thesis_evaluations": [],
        "positions_reviewed": [],
        "thesis_strengthened": [],
        "thesis_weakened": [],
        "positions_needing_reassessment": [],
        "missing_theses": [],
        "material_developments": [],
        "parse_error": True,
    }


def parse_thesis_response(text: str) -> tuple[str, dict]:
    """Parse thesis monitoring Claude response into (markdown, summary_dict)."""
    if THESIS_SEPARATOR in text:
        parts = text.split(THESIS_SEPARATOR, 1)
        thesis_md = parts[0].strip()
        json_part = parts[1].strip()
        if json_part.startswith("```"):
            json_part = json_part.split("\n", 1)[1] if "\n" in json_part else json_part[3:]
        if json_part.endswith("```"):
            json_part = json_part[:-3]
        json_part = json_part.strip()
        try:
            summary = json.loads(json_part)
        except json.JSONDecodeError:
            log.warning("Failed to parse thesis summary JSON")
            summary = _fallback_thesis_summary()
    else:
        log.warning("No thesis separator found in response")
        thesis_md = text.strip()
        summary = _fallback_thesis_summary()
    return thesis_md, summary


def _merge_thesis_into_summary(base_summary: dict, thesis_summary: dict) -> dict:
    """Merge thesis monitoring results into the weekly summary.json."""
    merged = dict(base_summary)
    merged["thesis_monitoring"] = thesis_summary
    return merged


# ---------------------------------------------------------------------------
# Output writing and archival
# ---------------------------------------------------------------------------


def write_outputs(report_md: str, summary: dict, bundle: dict, output_dir: Path, today: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "report.md").write_text(report_md, encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote report.md and summary.json to %s", output_dir)

    # Archive
    archive_dir = output_dir / "history" / today
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "weekly_bundle.json").write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    (archive_dir / "report.md").write_text(report_md, encoding="utf-8")
    (archive_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Archived to %s", archive_dir)


# create_github_issue is imported from auto_report.shared


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Automated weekly market report")
    parser.add_argument("--force", action="store_true", help="Bypass Friday-afternoon schedule gate")
    parser.add_argument("--no-search", action="store_true", help="Disable web search for news context")
    parser.add_argument(
        "--news-sources",
        type=str,
        default=None,
        help="Comma-separated list of domains to restrict news search (overrides defaults)",
    )
    args = parser.parse_args()

    if not args.force and not os.environ.get("FORCE_RUN") and not _is_friday_afternoon_et():
        log.info("Not Friday 16:xx ET — exiting (use --force to override)")
        sys.exit(0)

    today_str = datetime.now(ET).strftime("%Y-%m-%d")
    log.info("=== Weekly report run starting (%s) ===", today_str)

    # 1. Load prompts
    system_md = load_prompt_file(PROMPTS_DIR / "system.md", "prompts/system.md")
    playbook_md = load_prompt_file(PROMPTS_DIR / "playbook.md", "prompts/playbook.md")

    # 2. Load last-week summary
    last_week = load_last_week_summary(HISTORY_DIR)
    system_parts = [system_md, playbook_md]
    if last_week:
        system_parts.append(f"## Last Week's Summary\n\n```json\n{last_week}\n```")
        log.info("Loaded last-week summary from history")
    else:
        log.info("No prior summary found in history")
    system_msg = "\n\n---\n\n".join(system_parts)

    # 3. Collect data
    log.info("Collecting data from all sources...")
    t_collect = time.perf_counter()
    raw_data = collect_data()
    log.info("Data collection completed in %.2fs", time.perf_counter() - t_collect)

    # 4. Serialize and write bundle
    bundle = serialize_bundle(raw_data)
    write_bundle(bundle, OUTPUT_DIR)

    # 5. Build deterministic performance tables (from raw data, before serialization flattened Series)
    perf_md = build_performance_markdown(raw_data)

    # 6. Resolve web search settings
    use_search = not args.no_search
    if use_search:
        if args.news_sources:
            allowed_domains = [d.strip() for d in args.news_sources.split(",") if d.strip()]
        else:
            allowed_domains = list(DEFAULT_NEWS_SOURCES)
        log.info("Web search enabled — domains: %s", allowed_domains)
    else:
        allowed_domains = None
        log.info("Web search disabled")

    # 7. Call Claude
    user_msg = _build_user_message(bundle, perf_md, web_search=use_search)
    report_md = None
    summary = None
    error_msg = None
    citations: list[tuple[str, str]] = []

    try:
        response_text, citations = call_claude(system_msg, user_msg, allowed_domains=allowed_domains)
        report_md, summary = parse_response(response_text)
        report_md = _insert_weekly_performance(report_md, perf_md)
        report_md = strip_llm_meta(report_md)
    except Exception as e:
        log.error("Claude call failed: %s", e, exc_info=True)
        error_msg = str(e)
        report_md = f"# Weekly Report — {today_str}\n\n**Error**: Claude generation failed.\n\n```\n{error_msg}\n```"
        summary = _fallback_summary()
        summary["error"] = error_msg

    # 7b. Thesis Monitoring Pass
    thesis_data: dict = {}
    if THESES_DIR.exists():
        log.info("=== Thesis Monitoring Pass ===")
        try:
            t_thesis = time.perf_counter()
            thesis_data = collect_thesis_data()
            thesis_system, thesis_user = _build_thesis_prompt(thesis_data, web_search=use_search)

            thesis_text, thesis_citations = call_claude(
                system_msg=thesis_system,
                user_msg=thesis_user,
                allowed_domains=allowed_domains,
                max_tokens=8192,
            )
            thesis_md, thesis_summary = parse_thesis_response(thesis_text)
            thesis_md = strip_llm_meta(thesis_md)

            if thesis_citations:
                citations.extend(thesis_citations)

            # Append thesis section to report
            if thesis_md:
                report_md += "\n\n---\n\n" + thesis_md

            # Merge thesis into summary
            summary = _merge_thesis_into_summary(summary, thesis_summary)

            log.info(
                "Thesis pass completed in %.2fs (%d evaluations)",
                time.perf_counter() - t_thesis,
                len(thesis_summary.get("thesis_evaluations", [])),
            )
        except Exception as e:
            log.error("Thesis monitoring pass failed: %s", e, exc_info=True)
            fallback = _fallback_thesis_summary()
            fallback["error"] = str(e)
            summary = _merge_thesis_into_summary(summary, fallback)
            report_md += f"\n\n---\n\n## Portfolio Thesis Monitoring\n\n**Error**: Thesis monitoring failed — {e}"
    else:
        log.info("No investment_theses/ directory found — skipping thesis monitoring")

    report_md = _append_sources_section(report_md, citations)
    if citations:
        log.info("Appended %d unique citation sources to report", len(_dedupe_citations(citations)))

    # 8. Write outputs + archive
    write_outputs(report_md, summary, bundle, OUTPUT_DIR, today_str)

    # 8b. Archive thesis bundle
    if thesis_data:
        archive_dir = OUTPUT_DIR / "history" / today_str
        archive_dir.mkdir(parents=True, exist_ok=True)
        try:
            thesis_bundle_serialized = serialize_bundle(thesis_data)
            (archive_dir / "thesis_bundle.json").write_text(
                json.dumps(thesis_bundle_serialized, indent=2, default=str),
                encoding="utf-8",
            )
            log.info("Archived thesis_bundle.json to %s", archive_dir)
        except Exception as e:
            log.warning("Failed to archive thesis bundle: %s", e)

    # 9. Create GitHub Issue
    issue_title = f"Weekly Market Report — {today_str}"
    try:
        create_github_issue(issue_title, report_md)
    except Exception as e:
        log.error("GitHub Issue creation failed: %s", e, exc_info=True)

    log.info("=== Weekly report run complete ===")


if __name__ == "__main__":
    main()
