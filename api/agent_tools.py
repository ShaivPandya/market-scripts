"""
Tool registry for the AI agent.

Each tool wraps an existing get_data() / get_snapshot() function from the
analysis modules. Tool definitions use a JSON-schema format that can be adapted
for different LLM tool-calling APIs.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from api.cache import get_cached, long_cache, set_cached, short_cache
from api.serializers import serialize_value

logger = logging.getLogger("api.agent")

_SEARCH_WEB_ALLOWED_DOMAINS_DEFAULT = [
    "bloomberg.com",
    "cnbc.com",
    "federalreserve.gov",
    "axios.com",
]
_INACCESSIBLE_DOMAINS_RX = re.compile(
    r"domains are not accessible to our user agent:\s*(\[[^\]]*\])",
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Tool definitions (generic function-calling schema)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "name": "get_liquidity",
        "description": (
            "Fetch the global liquidity dashboard. Returns a composite liquidity score, "
            "regime (ample/normal/tight/stress), regional scores per major economy, "
            "individual component z-scores and contributions, and 1W/1M/3M changes. "
            "Use this to assess whether the global liquidity backdrop supports or hinders risk assets."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "get_market_breadth",
        "description": (
            "Fetch S&P 500 market breadth data. Returns the percentage and count of stocks "
            "above their 200-day and 20-day moving averages, at 20-day / 52-week / 24-week "
            "highs and lows, and total analyzed. Use this to assess market participation "
            "and whether rallies or selloffs are broad-based or narrow."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "get_vix_term_structure",
        "description": (
            "Fetch VIX term structure data. Returns the latest VIX, VIX3M (3-month VIX), "
            "the 3M/1M ratio, and a signal (Fear when ratio < 1.0, Complacency when > 1.25, "
            "else Neutral). Also includes recent ratio history and signal hit dates. "
            "Use this to gauge near-term vs longer-term volatility expectations."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "get_positioning",
        "description": (
            "Fetch CFTC Commitments of Traders (COT) leveraged fund positioning data. "
            "Returns net % of open interest, positioning z-scores, deleveraging z-scores, "
            "and forced flow signals (long liquidation / short covering) for each instrument. "
            "Use this to assess crowded positions and squeeze risk."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "instruments": {
                    "type": "string",
                    "description": (
                        "Comma-separated instrument aliases. "
                        "Available: SP500, NASDAQ, RUSSELL, US10Y, EUR, JPY, GBP, AUD, CAD, GOLD, OIL. "
                        "Default: 'SP500,NASDAQ,RUSSELL,US10Y,EUR'"
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "get_signal_aggregator",
        "description": (
            "Fetch a unified cross-module market signal dashboard that combines VIX term structure, "
            "market breadth, liquidity, CFTC positioning, sector metrics, and momentum into a deterministic "
            "regime signal. Returns current regime label (risk-on/transitional/risk-off), factor scores, "
            "effective weights, failed modules, and historical weekly regime tracking with episodes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lookback_weeks": {
                    "type": "integer",
                    "description": "Weekly history length for regime tracking. Default: 156 (about 3 years).",
                },
                "positioning_instruments": {
                    "type": "string",
                    "description": (
                        "Comma-separated CFTC instrument aliases for positioning input. "
                        "Default: 'SP500,NASDAQ,RUSSELL,US10Y,EUR'."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "get_economic_growth",
        "description": (
            "Fetch cross-asset returns for growth regime assessment. Returns period returns "
            "(1D, 1W, 1M, 3M, 6M, YTD) for commodities (copper, oil, gold, CRB), "
            "equities (S&P 500, Russell 2000, transports, banks), and currency pairs. "
            "Use this to identify growth cycle signals from market prices."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "get_labor_market",
        "description": (
            "Fetch US labor market indicators. Returns time series and latest values for "
            "initial claims, continuing claims, unemployment rate, nonfarm payrolls, "
            "and wage growth. Use this to assess labor market tightness and recession risk."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "get_housing",
        "description": (
            "Fetch US housing market indicators. Returns time series and latest values for "
            "housing starts, building permits, NAHB housing market index, and existing home sales. "
            "Use this to assess the residential construction cycle, builder sentiment, and housing demand."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "get_sector_metrics",
        "description": (
            "Fetch S&P 500 sector metrics. Returns sector weights, weight changes over "
            "1M/3M/6M, relative performance vs SPY, and percentage of sector constituents "
            "above their 200-day moving average. Use this to identify sector rotation, "
            "concentration risk, and leadership changes."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "get_portfolio",
        "description": (
            "Fetch the user's portfolio dashboard. Returns current positions with their "
            "P&L, metadata (asset class, direction), and price data. "
            "Use this when the user asks about their portfolio, holdings, performance, "
            "or any specific position. Pair with get_thesis for investment reasoning context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timeframe": {
                    "type": "string",
                    "description": "Period for returns. Options: 'This Week', 'Daily', 'Weekly', 'Monthly'. Default: 'Daily'.",
                },
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "get_yield_curve",
        "description": (
            "Fetch government bond yield curve data for the US, Germany, UK, and Japan. "
            "Returns current yields across tenors (3M through 30Y) and comparison vs "
            "a lookback period. Use this to assess yield curve shape, inversions, and "
            "changes in rate expectations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lookback_days": {
                    "type": "integer",
                    "description": "Number of days to look back for comparison. Default: 90.",
                },
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "get_bond_dashboard",
        "description": (
            "Fetch government bond yield time-series for 2Y, 10Y, and 30Y tenors across "
            "US, UK, Germany, and Japan. Returns the past year of daily yields, latest values, "
            "and year-over-year changes in basis points per country and tenor. Use this to "
            "compare sovereign yield levels and trends across major economies."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tenor": {
                    "type": "string",
                    "description": "Filter to a single tenor: '2Y', '10Y', or '30Y'. Default: return all tenors.",
                },
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "get_sentiment",
        "description": (
            "Fetch market sentiment indicators. Returns put/call ratios (equity aggregate,"
            "SPY, QQQ, IWM), investor surveys (AAII bull/bear spread, NAAIM exposure index), "
            "and volatility indices (VIX, VXN, VVIX). Includes quality checks and latest-date "
            "validation metadata. If quality.ok is false, do not draw directional sentiment "
            "conclusions and treat sentiment as unavailable for this turn."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "get_central_banks",
        "description": (
            "Fetch central bank news and recent publications. Returns articles and documents "
            "from the Fed, ECB, BoE, BoJ, SNB, RBA, and other major central banks, grouped "
            "by source with counts. Use this to check for recent policy signals or speeches."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "get_industry_monitor",
        "description": (
            "Fetch what businesses and companies are actually saying from their earnings call "
            "transcripts. Covers leading (housing, trucking), coincident (banks, retail), and "
            "lagging (capital goods) industry sectors. Returns per-company sentiment (bullish/"
            "neutral/bearish), demand trends, pricing commentary, guidance outlook, macro quotes, "
            "and sector-level economic signals (expanding/stable/slowing/contracting). "
            "Use this when the user asks what businesses, companies, or management teams are "
            "saying about the economy, demand, or business conditions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "refresh": {
                    "type": "boolean",
                    "description": "If true, bypass cached data and recompute from source files. Default: false.",
                },
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "get_breakout",
        "description": (
            "Fetch macro breakout signals across asset classes. Returns recent breakouts "
            "with direction (up/down), date, market, and close price. Use this to identify "
            "which major markets are making significant technical moves."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "query_ontology",
        "description": (
            "Run a cross-module ontology query that joins portfolio positions with macro/market "
            "signals (VIX, breadth, sector stress, liquidity, and other read-only data modules). "
            "Returns per-position risk scores with evidence. Use this when users ask about "
            "portfolio risk exposure, positions in deteriorating conditions, or entity-level "
            "context. Pair with get_thesis for the investment reasoning behind positions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query, e.g. 'Which positions are in deteriorating macro conditions?'",
                },
                "intent": {
                    "type": "string",
                    "description": (
                        "Optional explicit intent. Allowed: "
                        "portfolio_risk_exposure, positions_in_deteriorating_macro, entity_context."
                    ),
                },
                "filters": {
                    "type": "object",
                    "description": "Optional filters: tickers, sectors, assets, max_results, min_risk_score.",
                },
                "timeframe": {
                    "type": "string",
                    "description": "Timeframe for portfolio-linked data. Options: This Week, Daily, Weekly, Monthly.",
                },
                "include_graph": {
                    "type": "boolean",
                    "description": "If true, include ontology nodes and edges in output.",
                },
                "run_id": {
                    "type": "string",
                    "description": "Optional ontology snapshot run_id for historical replay.",
                },
                "refresh_snapshot": {
                    "type": "boolean",
                    "description": "If true, bypass latest snapshot reuse and force a fresh ontology snapshot build.",
                },
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "get_thesis",
        "description": (
            "Fetch the investment thesis for a specific ticker. Returns the thesis markdown "
            "content (thesis statement, key catalysts, risk factors) and metadata (status, "
            "creation date, last update). Use this when the user asks about a position's "
            "investment reasoning, thesis, catalysts, kill conditions, or why they own "
            "something. Also useful for thesis pressure-tests and reviews."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker symbol (e.g. 'CRWD', 'AAPL'). Case-insensitive.",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "type": "function",
        "name": "get_thesis_evaluations",
        "description": (
            "Fetch the monitoring evaluation history for a specific ticker's thesis. Returns "
            "weekly evaluations (thesis status, technical read, fundamental read, recommended "
            "action, confidence, key developments, earnings notes, risk flags) and status "
            "change history. Use this to understand how a thesis has evolved over time, "
            "whether conviction has increased or decreased, and what developments have "
            "occurred since the thesis was written."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker symbol (e.g. 'CRWD', 'AAPL'). Case-insensitive.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of evaluations to return (most recent first). Default: 10.",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "type": "function",
        "name": "search_knowledge_base",
        "description": (
            "Search across all indexed research documents — investment theses, weekly reports, "
            "daily reports, and past conversation summaries — using semantic similarity. "
            "Use this when the user asks what they wrote about a topic, references past research, "
            "wants to find previous analysis on a ticker or theme, or asks 'what did I say about X'. "
            "Returns ranked snippets with source attribution."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query (e.g. 'cloud security thesis for CRWD', 'liquidity tightening analysis').",
                },
                "doc_types": {
                    "type": "string",
                    "description": (
                        "Comma-separated document types to search. Options: thesis, weekly_report, "
                        "daily_report, conversation_summary. Leave empty to search all."
                    ),
                },
                "tickers": {
                    "type": "string",
                    "description": "Comma-separated ticker filter (e.g. 'CRWD,AAPL'). Leave empty for all.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return. Default: 5.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "get_ontology_diff",
        "description": (
            "Compare two ontology snapshots to show what changed in the portfolio's risk profile. "
            "Returns new/removed positions, risk score deltas, signal transitions (stable→deteriorating), "
            "and component score changes. Use this when the user asks 'what changed', 'how has my risk "
            "changed', 'what's different since last week', or any temporal comparison of portfolio risk."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "run_id_before": {
                    "type": "string",
                    "description": "The older snapshot run_id to compare from. Leave empty to auto-select the most recent prior snapshot.",
                },
                "run_id_after": {
                    "type": "string",
                    "description": "The newer snapshot run_id to compare to. Leave empty to use the latest/current snapshot.",
                },
            },
            "required": [],
        },
    },
    # -----------------------------------------------------------------------
    # Web search
    # -----------------------------------------------------------------------
    {
        "type": "function",
        "name": "search_web",
        "description": (
            "Search the web for recent news, events, or developments related to a ticker, "
            "company, sector, or macro topic. Uses trusted financial news sources (Bloomberg, "
            "CNBC, Reuters, WSJ, FT, etc.). Returns a summary of findings with source citations. "
            "Use this to verify catalyst status, check for breaking news, confirm regulatory "
            "actions, or validate thesis assumptions against real-world events."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query. Be specific — include ticker, company name, and what you're looking for."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    # -----------------------------------------------------------------------
    # Investing OS tools — read
    # -----------------------------------------------------------------------
    {
        "type": "function",
        "name": "get_catalysts",
        "description": (
            "Fetch tracked catalysts for a given ticker. Returns a list of catalysts with "
            "their status (pending/played_out/failed/superseded), category, target date, and evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol (e.g. 'AAPL')"},
            },
            "required": ["ticker"],
        },
    },
    {
        "type": "function",
        "name": "get_kill_conditions",
        "description": (
            "Fetch kill conditions for a given ticker. Returns conditions that would invalidate "
            "the thesis, with status (active/triggered/retired), metric, and threshold."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol (e.g. 'AAPL')"},
            },
            "required": ["ticker"],
        },
    },
    {
        "type": "function",
        "name": "get_action_items",
        "description": (
            "Fetch open action items, optionally filtered by ticker. Returns tasks with "
            "urgency (low/normal/high/urgent), action type, and status."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Optional ticker filter"},
                "status": {"type": "string", "description": "Filter by status. Default: 'open'"},
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "get_watch_triggers",
        "description": (
            "Fetch active watch triggers, optionally filtered by ticker. Returns conditions "
            "the system is monitoring with trigger type and status."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Optional ticker filter"},
                "status": {"type": "string", "description": "Filter by status. Default: 'active'"},
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "get_pending_approvals",
        "description": (
            "Fetch pending approval items. These are proposed changes from workflows or agent "
            "that require user approval before being applied."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Optional ticker filter"},
                "status": {"type": "string", "description": "Filter by status. Default: 'pending'"},
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "get_dossier",
        "description": (
            "Fetch the complete position dossier for a ticker. Returns thesis, catalysts, "
            "kill conditions, evaluations, ontology risk, workflow runs, action items, "
            "triggers, research notes, and pending approvals — all in one call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol (e.g. 'MU')"},
            },
            "required": ["ticker"],
        },
    },
    {
        "type": "function",
        "name": "get_workflow_history",
        "description": ("Fetch recent workflow run history, optionally filtered by workflow name or ticker."),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Optional ticker filter"},
                "workflow_name": {"type": "string", "description": "Optional workflow name filter"},
                "limit": {"type": "integer", "description": "Max results (default 10)"},
            },
            "required": [],
        },
    },
    # -----------------------------------------------------------------------
    # Investing OS tools — propose (approval-gated writes)
    # -----------------------------------------------------------------------
    {
        "type": "function",
        "name": "propose_thesis_status_change",
        "description": (
            "Propose a thesis status change for a ticker. This creates a pending approval "
            "that the user must approve before the status is actually changed. "
            "Use this instead of directly modifying thesis status."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol"},
                "new_status": {
                    "type": "string",
                    "description": "Proposed new status: active|under_review|suspended|closed",
                },
                "reason": {"type": "string", "description": "Explanation for the proposed change"},
            },
            "required": ["ticker", "new_status", "reason"],
        },
    },
    {
        "type": "function",
        "name": "propose_action_item",
        "description": (
            "Propose a new action item. This creates a pending approval that the user must "
            "approve before the action item is created. Use this for recommending trades, "
            "research tasks, or position adjustments."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol (optional for non-ticker-specific actions)"},
                "description": {"type": "string", "description": "What needs to be done"},
                "action_type": {"type": "string", "description": "Type: review|resize|research|exit|enter|hedge|other"},
                "urgency": {"type": "string", "description": "Urgency: low|normal|high|urgent"},
                "reason": {"type": "string", "description": "Why this action is recommended"},
            },
            "required": ["description", "action_type", "reason"],
        },
    },
    {
        "type": "function",
        "name": "propose_catalyst_status_change",
        "description": (
            "Propose a catalyst status change. This creates a pending approval that the user must "
            "approve before the catalyst status is actually updated. Use this when evidence suggests "
            "a catalyst has played out, failed, or been superseded."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol"},
                "catalyst_id": {"type": "integer", "description": "ID of the catalyst to update"},
                "new_status": {
                    "type": "string",
                    "description": "Proposed new status: pending|played_out|failed|superseded",
                },
                "evidence": {"type": "string", "description": "Evidence supporting the status change"},
                "reason": {"type": "string", "description": "Explanation for the proposed change"},
            },
            "required": ["ticker", "catalyst_id", "new_status", "reason"],
        },
    },
    {
        "type": "function",
        "name": "propose_kill_condition_status_change",
        "description": (
            "Propose a kill condition status change. This creates a pending approval that the user must "
            "approve before the kill condition status is actually updated. Use this when a kill condition "
            "has been triggered or should be retired."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol"},
                "kill_condition_id": {"type": "integer", "description": "ID of the kill condition to update"},
                "new_status": {
                    "type": "string",
                    "description": "Proposed new status: active|triggered|retired",
                },
                "reason": {"type": "string", "description": "Explanation for the proposed change"},
            },
            "required": ["ticker", "kill_condition_id", "new_status", "reason"],
        },
    },
    {
        "type": "function",
        "name": "propose_watch_trigger",
        "description": (
            "Propose a new watch trigger. This creates a pending approval that the user must "
            "approve before the trigger is activated. Use this to set up monitoring conditions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol (optional)"},
                "condition": {
                    "type": "string",
                    "description": "The condition to watch for (e.g. 'AAPL breaks below $180')",
                },
                "trigger_type": {
                    "type": "string",
                    "description": "Type: price_level|technical|fundamental|event|macro|custom",
                },
                "reason": {"type": "string", "description": "Why this trigger matters"},
            },
            "required": ["condition", "trigger_type", "reason"],
        },
    },
]

# Tool name → index lookup
_TOOL_INDEX = {t["name"]: i for i, t in enumerate(TOOL_DEFINITIONS)}


# ---------------------------------------------------------------------------
# Compact payload helpers
# ---------------------------------------------------------------------------


_MAX_TOOL_RESPONSE_CHARS = 30_000
_MISSING = object()
_singleflight_lock = threading.Lock()


@dataclass(slots=True)
class _SingleFlightState:
    event: threading.Event
    value: object = _MISSING
    error: Exception | None = None


_singleflight_by_key: dict[str, _SingleFlightState] = {}


def _stable_json_dumps(data: object) -> str:
    try:
        return json.dumps(data, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return json.dumps(str(data), separators=(",", ":"))


def _safe_iso_date(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized).date().isoformat()
    except ValueError:
        return None


def _date_key(value: Any) -> tuple[int, str]:
    parsed = _safe_iso_date(value)
    return (1, parsed) if parsed else (0, "")


def _latest_by_date(rows: list[dict]) -> dict | None:
    dated = [r for r in rows if isinstance(r, dict) and _safe_iso_date(r.get("date"))]
    if not dated:
        return None
    return max(dated, key=lambda r: _date_key(r.get("date")))


def _sort_by_date(rows: list[dict]) -> list[dict]:
    out = [r for r in rows if isinstance(r, dict)]
    return sorted(out, key=lambda r: _date_key(r.get("date")))


def _is_stale(value: Any, max_days: int) -> bool:
    parsed = _safe_iso_date(value)
    if not parsed:
        return True
    dt = datetime.fromisoformat(parsed).date()
    return (datetime.now(UTC).date() - dt).days > max_days


def _compact_generic(value: Any, *, max_depth: int, list_limit: int, dict_limit: int, depth: int = 0) -> Any:
    if depth >= max_depth:
        if isinstance(value, list):
            return {"_summary": "list", "count": len(value)}
        if isinstance(value, dict):
            return {"_summary": "dict", "count": len(value)}
        return value

    if isinstance(value, list):
        keep = min(len(value), list_limit)
        out = [
            _compact_generic(v, max_depth=max_depth, list_limit=list_limit, dict_limit=dict_limit, depth=depth + 1)
            for v in value[:keep]
        ]
        if len(value) > keep:
            out.append({"_truncated": len(value) - keep})
        return out

    if isinstance(value, dict):
        sorted_items = sorted(value.items(), key=lambda kv: str(kv[0]))
        limited_items = sorted_items[:dict_limit]
        out_dict: dict[str, Any] = {}
        for key, raw_val in limited_items:
            key_str = str(key)
            out_dict[key_str] = _compact_generic(
                raw_val,
                max_depth=max_depth,
                list_limit=list_limit,
                dict_limit=dict_limit,
                depth=depth + 1,
            )
        if len(sorted_items) > len(limited_items):
            out_dict["_truncated_keys"] = len(sorted_items) - len(limited_items)
        return out_dict

    return value


def _compact_ontology_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return _compact_generic(payload, max_depth=4, list_limit=25, dict_limit=60)

    out: dict[str, Any] = {
        "run_id": payload.get("run_id"),
        "intent": payload.get("intent"),
        "interpreted_query": _compact_generic(
            payload.get("interpreted_query"), max_depth=3, list_limit=20, dict_limit=20
        ),
        "as_of": payload.get("as_of"),
        "source_status": _compact_generic(payload.get("source_status"), max_depth=3, list_limit=20, dict_limit=20),
        "aggregate": _compact_generic(payload.get("aggregate"), max_depth=3, list_limit=20, dict_limit=20),
    }

    raw_results = payload.get("results")
    results = raw_results if isinstance(raw_results, list) else []
    trimmed: list[dict] = []
    for row in results[:25]:
        if not isinstance(row, dict):
            continue
        raw_evidence = row.get("evidence")
        evidence = raw_evidence if isinstance(raw_evidence, list) else []
        trimmed.append(
            {
                "ticker": row.get("ticker"),
                "asset": row.get("asset"),
                "direction": row.get("direction"),
                "sector": row.get("sector"),
                "risk_score": row.get("risk_score"),
                "risk_level": row.get("risk_level"),
                "evidence": _compact_generic(evidence[:3], max_depth=3, list_limit=6, dict_limit=15),
            }
        )
    out["results"] = trimmed

    graph = payload.get("graph")
    if isinstance(graph, dict):
        raw_nodes = graph.get("nodes")
        nodes = raw_nodes if isinstance(raw_nodes, list) else []
        raw_edges = graph.get("edges")
        edges = raw_edges if isinstance(raw_edges, list) else []
        out["graph"] = {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "sample_nodes": _compact_generic(nodes[:5], max_depth=2, list_limit=5, dict_limit=10),
            "sample_edges": _compact_generic(edges[:5], max_depth=2, list_limit=5, dict_limit=10),
        }
    return out


def _compact_portfolio_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return _compact_generic(payload, max_depth=4, list_limit=25, dict_limit=60)

    if isinstance(payload.get("error"), str):
        return payload

    raw_positions = payload.get("positions")
    positions = raw_positions if isinstance(raw_positions, dict) else {}
    raw_metadata = payload.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}

    tickers = list(metadata.keys())
    for ticker in positions.keys():
        if ticker not in metadata:
            tickers.append(ticker)

    def _first_valid_point(series_rows: list[dict]) -> dict | None:
        for row in series_rows:
            if not isinstance(row, dict):
                continue
            value = _to_float(row.get("value"))
            if value is None:
                continue
            return {"date": row.get("date"), "value": value}
        return None

    def _last_valid_point(series_rows: list[dict]) -> dict | None:
        for row in reversed(series_rows):
            if not isinstance(row, dict):
                continue
            value = _to_float(row.get("value"))
            if value is None:
                continue
            return {"date": row.get("date"), "value": value}
        return None

    compact_rows: list[dict[str, Any]] = []
    for ticker in tickers:
        raw_meta = metadata.get(ticker)
        meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        series = positions.get(ticker)
        series_rows = series if isinstance(series, list) else []
        first = _first_valid_point(series_rows)
        last = _last_valid_point(series_rows)
        first_val = _to_float(first.get("value")) if isinstance(first, dict) else None
        last_val = _to_float(last.get("value")) if isinstance(last, dict) else None

        price_change = None
        price_return_pct = None
        if first_val is not None and last_val is not None and first_val != 0:
            price_change = round(last_val - first_val, 6)
            price_return_pct = round(((last_val - first_val) / first_val) * 100.0, 4)

        direction = str(meta.get("direction") or "").strip().lower()
        directional_return_pct = None
        if price_return_pct is not None:
            directional_return_pct = -price_return_pct if direction == "short" else price_return_pct

        compact_rows.append(
            {
                "ticker": ticker,
                "asset": meta.get("asset"),
                "direction": meta.get("direction"),
                "first_date": first.get("date") if isinstance(first, dict) else None,
                "first_price": first_val,
                "last_date": last.get("date") if isinstance(last, dict) else None,
                "last_price": last_val,
                "price_change": price_change,
                "price_return_pct": price_return_pct,
                "directional_return_pct": directional_return_pct,
                "data_points": len(series_rows),
            }
        )

    long_count = sum(1 for row in compact_rows if str(row.get("direction") or "").lower() == "long")
    short_count = sum(1 for row in compact_rows if str(row.get("direction") or "").lower() == "short")
    directional_returns = [
        r for r in (row.get("directional_return_pct") for row in compact_rows) if isinstance(r, (int, float))
    ]
    avg_directional_return_pct = (
        round(sum(float(r) for r in directional_returns) / len(directional_returns), 4) if directional_returns else None
    )

    compact_rows.sort(key=lambda row: str(row.get("ticker") or ""))
    extras = {k: v for k, v in payload.items() if k not in {"positions", "metadata", "timeframe", "timestamp"}}
    out: dict[str, Any] = {
        "timeframe": payload.get("timeframe"),
        "timestamp": payload.get("timestamp"),
        "summary": {
            "position_count": len(compact_rows),
            "long_count": long_count,
            "short_count": short_count,
            "average_directional_return_pct": avg_directional_return_pct,
        },
        "positions": compact_rows,
    }
    if metadata:
        out["metadata"] = _compact_generic(metadata, max_depth=3, list_limit=20, dict_limit=40)
    if extras:
        out["extra"] = _compact_generic(extras, max_depth=3, list_limit=20, dict_limit=20)
    return out


def _summarize_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"type": "dict", "keys": list(value.keys())[:40], "key_count": len(value)}
    if isinstance(value, list):
        return {"type": "list", "count": len(value), "sample": value[:5]}
    return {"type": type(value).__name__, "value": value}


def _attach_meta(payload: Any, meta: dict[str, Any]) -> dict[str, Any]:
    base = payload if isinstance(payload, dict) else {"data": payload}
    raw_meta = base.get("_meta")
    existing = raw_meta if isinstance(raw_meta, dict) else {}
    merged = dict(existing)
    merged.update(meta)
    base["_meta"] = merged
    return base


def _compact_tool_output(name: str, payload: Any, max_chars: int = _MAX_TOOL_RESPONSE_CHARS) -> tuple[dict, dict]:
    raw_chars = len(_stable_json_dumps(payload))
    truncated = False

    if name == "query_ontology":
        compacted = _compact_ontology_payload(payload)
    elif name == "get_portfolio":
        compacted = _compact_portfolio_payload(payload)
    else:
        compacted = payload

    compacted_chars = len(_stable_json_dumps(compacted))
    if compacted_chars > max_chars:
        truncated = True
        compacted = _compact_generic(compacted, max_depth=4, list_limit=40, dict_limit=60)
        compacted_chars = len(_stable_json_dumps(compacted))

    if compacted_chars > max_chars:
        truncated = True
        compacted = _compact_generic(compacted, max_depth=3, list_limit=20, dict_limit=30)
        compacted_chars = len(_stable_json_dumps(compacted))

    if compacted_chars > max_chars:
        truncated = True
        compacted = {"summary": _summarize_payload(compacted)}
        compacted_chars = len(_stable_json_dumps(compacted))

    meta = {
        "truncated": truncated,
        "raw_chars": raw_chars,
        "output_chars": compacted_chars,
        "max_chars": max_chars,
    }
    return _attach_meta(compacted, meta), meta


def _cached_singleflight(cache, key: str, loader: Callable[[], Any]) -> tuple[Any, str]:
    cached = get_cached(cache, key)
    if cached is not None:
        return cached, "hit"

    flight_key = f"{id(cache)}::{key}"
    with _singleflight_lock:
        state = _singleflight_by_key.get(flight_key)
        if state is None:
            state = _SingleFlightState(event=threading.Event())
            _singleflight_by_key[flight_key] = state
            owner = True
        else:
            owner = False

    if owner:
        try:
            value = loader()
            set_cached(cache, key, value)
            state.value = value
            return value, "miss_fetch"
        except Exception as exc:
            state.error = exc
            raise
        finally:
            state.event.set()
            with _singleflight_lock:
                _singleflight_by_key.pop(flight_key, None)

    state.event.wait(timeout=120)
    if state.error is not None:
        raise state.error
    if state.value is not _MISSING:
        return state.value, "miss_wait"

    # Safety fallback if owner did not publish a value.
    cached_after = get_cached(cache, key)
    if cached_after is not None:
        return cached_after, "miss_wait"
    value = loader()
    set_cached(cache, key, value)
    return value, "miss_refetch"


def _fetch_with_cache(cache, key: str, loader: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    value, cache_status = _cached_singleflight(cache, key, loader)
    return value, {"cache": cache_status}


def _extract_inaccessible_domains(exc: Exception) -> set[str]:
    text = str(exc)
    match = _INACCESSIBLE_DOMAINS_RX.search(text)
    if not match:
        return set()

    raw = match.group(1).strip()
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return set()
    if not isinstance(parsed, list):
        return set()

    blocked: set[str] = set()
    for item in parsed:
        domain = str(item).strip().lower()
        if domain:
            blocked.add(domain)
    return blocked


def _run_search_web(query: str) -> dict[str, Any]:
    from llm_utils import MODEL_HAIKU, call_claude_text

    allowed_domains = list(_SEARCH_WEB_ALLOWED_DOMAINS_DEFAULT)
    attempts = 0
    while attempts < 3:
        attempts += 1
        try:
            text, citations, _response = call_claude_text(
                prompt=f"Find the latest news and developments about: {query}",
                model=MODEL_HAIKU,
                api_key=None,
                max_tokens=2048,
                system=(
                    "You are a financial research assistant. Search for the most recent, "
                    "relevant information about the query. Return a concise summary of key "
                    "findings organized by topic. Include dates when available. "
                    "Focus on facts, not opinions."
                ),
                allowed_domains=allowed_domains,
                max_web_search_uses=3,
            )
            return {
                "query": query,
                "summary": text,
                "citations": [{"title": t, "url": u} for t, u in citations],
                "citation_count": len(citations),
            }
        except Exception as exc:  # noqa: BLE001 - tool should recover if possible
            blocked = _extract_inaccessible_domains(exc)
            if not blocked:
                raise

            remaining = [d for d in allowed_domains if d.lower() not in blocked]
            if len(remaining) == len(allowed_domains):
                raise
            if not remaining:
                raise RuntimeError("All configured search domains were rejected by Anthropic.") from exc

            logger.warning(
                "search_web pruned inaccessible domains blocked=%s remaining=%s",
                sorted(blocked),
                remaining,
            )
            allowed_domains = remaining

    raise RuntimeError("search_web failed after exhausting domain-pruning retries")


# ---------------------------------------------------------------------------
# Thesis helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_THESES_DIR = _PROJECT_ROOT / "investment_theses"
_MAX_THESIS_CONTENT_CHARS = 8_000


def _find_thesis_file(ticker: str) -> Path | None:
    """Case-insensitive glob for a thesis markdown file."""
    ticker_upper = ticker.strip().upper()
    if not ticker_upper:
        return None
    for path in _THESES_DIR.glob("*.md"):
        if path.stem.upper() == ticker_upper:
            return path
    return None


def _fetch_thesis(ticker: str) -> dict[str, Any]:
    from portfolio.thesis_db import get_thesis_meta

    meta = get_thesis_meta(ticker)
    thesis_path = _find_thesis_file(ticker)
    content: str | None = None
    truncated = False

    if thesis_path and thesis_path.is_file():
        raw = thesis_path.read_text(encoding="utf-8").strip()
        if len(raw) > _MAX_THESIS_CONTENT_CHARS:
            content = raw[:_MAX_THESIS_CONTENT_CHARS]
            truncated = True
        else:
            content = raw

    if meta is None and content is None:
        return {"error": f"No thesis found for ticker '{ticker}'", "ticker": ticker}

    return {
        "ticker": ticker,
        "meta": meta,
        "content": content,
        "content_truncated": truncated,
        "source_file": str(thesis_path) if thesis_path else None,
    }


def _fetch_thesis_evaluations(ticker: str, limit: int) -> dict[str, Any]:
    from portfolio.thesis_db import get_evaluations, get_status_history, get_thesis_meta

    meta = get_thesis_meta(ticker)
    evaluations = get_evaluations(ticker, limit=limit)
    status_history = get_status_history(ticker)

    if meta is None and not evaluations:
        return {
            "error": f"No thesis or evaluations found for ticker '{ticker}'",
            "ticker": ticker,
        }

    return {
        "ticker": ticker,
        "current_status": meta.get("status") if meta else None,
        "meta": meta,
        "evaluations": evaluations,
        "evaluation_count": len(evaluations),
        "status_history": status_history,
    }


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------


def execute_tool(name: str, arguments: dict) -> str:
    """Run the tool identified by *name* and return a JSON string for the model.

    Errors are caught and returned as ``{"error": "..."}`` so the model can
    inform the user instead of crashing the stream.
    """
    started = time.perf_counter()
    try:
        safe_args = arguments if isinstance(arguments, dict) else {}
        result, dispatch_meta = _dispatch(name, safe_args)
        payload, _compact_meta = _compact_tool_output(name, result)
        meta = dict(dispatch_meta)
        meta.update(
            {
                "tool": name,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                "status": "ok",
            }
        )
        quality = payload.get("quality") if isinstance(payload, dict) else None
        if isinstance(quality, dict):
            meta["quality_ok"] = bool(quality.get("ok"))
        payload = _attach_meta(payload, meta)
        return _stable_json_dumps(payload)
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        payload = _attach_meta(
            {"error": f"Failed to fetch {name}: {exc}"},
            {
                "tool": name,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                "status": "error",
            },
        )
        return _stable_json_dumps(payload)


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_agent_sentiment_snapshot(put_call: dict, surveys: dict, volatility: list) -> dict:
    put_call = put_call if isinstance(put_call, dict) else {}
    surveys = surveys if isinstance(surveys, dict) else {}
    volatility_rows = [r for r in volatility if isinstance(r, dict)] if isinstance(volatility, list) else []

    aaii_rows = _sort_by_date([r for r in surveys.get("aaii", []) if isinstance(r, dict)])
    naaim_rows = _sort_by_date([r for r in surveys.get("naaim", []) if isinstance(r, dict)])
    latest_aaii = _latest_by_date(aaii_rows)
    latest_naaim = _latest_by_date(naaim_rows)
    latest_vol = _latest_by_date(volatility_rows)

    raw_source_errors = surveys.get("errors")
    source_errors = raw_source_errors if isinstance(raw_source_errors, dict) else {}
    source_errors = {str(k): str(v) for k, v in source_errors.items() if isinstance(v, str) and v.strip()}
    issues: list[str] = []
    feed_status: dict[str, dict[str, Any]] = {}

    pc_as_of = (
        _safe_iso_date((put_call.get("equity") or {}).get("as_of"))
        if isinstance(put_call.get("equity"), dict)
        else None
    )
    put_call_stale = _is_stale(pc_as_of, max_days=3)
    feed_status["put_call"] = {"latest_date": pc_as_of, "stale": put_call_stale, "available": bool(put_call)}
    if not put_call:
        issues.append("Put/Call feed unavailable")
    elif put_call_stale:
        issues.append("Put/Call feed stale")

    aaii_date = _safe_iso_date((latest_aaii or {}).get("date")) if latest_aaii else None
    naaim_date = _safe_iso_date((latest_naaim or {}).get("date")) if latest_naaim else None
    vol_date = _safe_iso_date((latest_vol or {}).get("date")) if latest_vol else None
    aaii_stale = _is_stale(aaii_date, max_days=21)
    naaim_stale = _is_stale(naaim_date, max_days=21)
    vol_stale = _is_stale(vol_date, max_days=5)

    feed_status["aaii"] = {"latest_date": aaii_date, "stale": aaii_stale, "available": latest_aaii is not None}
    feed_status["naaim"] = {"latest_date": naaim_date, "stale": naaim_stale, "available": latest_naaim is not None}
    feed_status["volatility"] = {"latest_date": vol_date, "stale": vol_stale, "available": latest_vol is not None}

    if latest_aaii is None:
        issues.append("AAII feed unavailable")
    elif aaii_stale:
        issues.append("AAII feed stale")

    if latest_naaim is None:
        issues.append("NAAIM feed unavailable")
    elif naaim_stale:
        issues.append("NAAIM feed stale")

    if latest_vol is None:
        issues.append("Volatility feed unavailable")
    elif vol_stale:
        issues.append("Volatility feed stale")

    if latest_aaii:
        bull = _to_float(latest_aaii.get("bull"))
        bear = _to_float(latest_aaii.get("bear"))
        neutral = _to_float(latest_aaii.get("neutral"))
        if bull is not None and bear is not None and neutral is not None:
            total = bull + bear + neutral
            if total < 98.0 or total > 102.0:
                issues.append(f"AAII components inconsistent (sum={round(total, 2)})")

    for src, err in source_errors.items():
        issues.append(f"{src} source error: {err}")

    latest_dates = [d for d in (pc_as_of, aaii_date, naaim_date, vol_date) if d]
    as_of = max(latest_dates) if latest_dates else datetime.now(UTC).date().isoformat()
    quality_ok = len(issues) == 0

    def _pc_summary(key: str) -> dict:
        row = put_call.get(key)
        if not isinstance(row, dict):
            return {}
        return {
            "ratio": row.get("ratio"),
            "calls": row.get("calls"),
            "puts": row.get("puts"),
            "as_of": _safe_iso_date(row.get("as_of")) or row.get("as_of"),
        }

    snapshot = {
        "as_of": as_of,
        "latest": {
            "put_call": {
                "equity": _pc_summary("equity"),
                "spy": _pc_summary("spy"),
                "qqq": _pc_summary("qqq"),
                "iwm": _pc_summary("iwm"),
            },
            "surveys": {
                "aaii": latest_aaii,
                "naaim": latest_naaim,
            },
            "volatility": latest_vol,
        },
        "recent_trends": {
            "aaii_spread": [
                {"date": row.get("date"), "spread": row.get("spread"), "bull": row.get("bull"), "bear": row.get("bear")}
                for row in aaii_rows[-8:]
            ],
            "naaim_exposure": [{"date": row.get("date"), "exposure": row.get("exposure")} for row in naaim_rows[-8:]],
            "volatility": [
                {"date": row.get("date"), "vix": row.get("vix"), "vxn": row.get("vxn"), "vvix": row.get("vvix")}
                for row in _sort_by_date(volatility_rows)[-8:]
            ],
        },
        "quality": {
            "ok": quality_ok,
            "mode": "fail_closed",
            "allow_sentiment_conclusion": quality_ok,
            "issues": issues,
            "source_errors": source_errors,
            "feed_status": feed_status,
        },
    }
    return snapshot


def _dispatch(name: str, args: dict) -> tuple[object, dict[str, Any]]:
    """Route a tool call to the corresponding data function."""

    if name == "get_liquidity":
        key = "agent_liquidity"

        def _load():
            from macro.liquidity.liquidity import get_snapshot

            data = get_snapshot()
            filtered = {k: v for k, v in data.items() if k not in ("df_weekly", "composite_series")}
            return serialize_value(filtered)

        data, meta = _fetch_with_cache(long_cache, key, _load)
        return data, meta

    if name == "get_market_breadth":
        key = "agent_market_breadth"

        def _load():
            from equities.market_technicals.market_breadth import get_data

            return serialize_value(get_data())

        data, meta = _fetch_with_cache(long_cache, key, _load)
        return data, meta

    if name == "get_vix_term_structure":
        key = "agent_vix_term_structure:default"

        def _load():
            from equities.market_technicals.vix_term_structure import get_data

            return serialize_value(get_data())

        data, meta = _fetch_with_cache(short_cache, key, _load)
        return data, meta

    if name == "get_positioning":
        instruments = args.get("instruments", "SP500,NASDAQ,RUSSELL,US10Y,EUR")
        app_token = os.environ.get("SODA_APP_TOKEN") or None
        key = f"positioning_summary:{instruments}:2015-01-01:None:None:0:2.0"

        def _load():
            from macro.positioning.positioning import DATASETS, DEFAULT_DOMAIN, fetch_multiple_instruments

            instrument_list = [i.strip() for i in instruments.split(",") if i.strip()]
            data = fetch_multiple_instruments(
                domain=DEFAULT_DOMAIN,
                dataset_id=DATASETS.get("tff_futures_only", "tff_futures_only"),
                app_token=app_token,
                instruments=instrument_list,
                start="2015-01-01",
                end=None,
            )
            return serialize_value(data)

        data, meta = _fetch_with_cache(long_cache, key, _load)
        return data, meta

    if name == "get_signal_aggregator":
        from api.signal_aggregator import (
            DEFAULT_LOOKBACK_WEEKS,
            DEFAULT_POSITIONING_INSTRUMENTS,
            build_signal_aggregator,
        )

        lookback_weeks = int(args.get("lookback_weeks", DEFAULT_LOOKBACK_WEEKS))
        lookback_weeks = max(26, min(lookback_weeks, 520))
        positioning_instruments = str(args.get("positioning_instruments", DEFAULT_POSITIONING_INSTRUMENTS))
        key = f"signal_aggregator:{lookback_weeks}:{positioning_instruments}:False"

        def _load():
            data = build_signal_aggregator(
                lookback_weeks=lookback_weeks,
                positioning_instruments=positioning_instruments,
                include_raw_modules=False,
            )
            return serialize_value(data)

        data, meta = _fetch_with_cache(short_cache, key, _load)
        meta["high_cost"] = True
        return data, meta

    if name == "get_economic_growth":
        key = "economic_growth"

        def _load():
            from macro.economic_growth.economic_growth import get_data

            return serialize_value(get_data())

        data, meta = _fetch_with_cache(short_cache, key, _load)
        return data, meta

    if name == "get_labor_market":
        key = "labor_market"

        def _load():
            from macro.labor_market.labor_market import get_data

            return serialize_value(get_data())

        data, meta = _fetch_with_cache(short_cache, key, _load)
        return data, meta

    if name == "get_housing":
        key = "housing"

        def _load():
            from macro.housing.housing import get_data

            return serialize_value(get_data())

        data, meta = _fetch_with_cache(short_cache, key, _load)
        return data, meta

    if name == "get_sector_metrics":
        key = "sector_metrics"

        def _load():
            from equities.sector_metrics.sector_metrics import get_data

            return serialize_value(get_data())

        data, meta = _fetch_with_cache(long_cache, key, _load)
        meta["high_cost"] = True
        return data, meta

    if name == "get_portfolio":
        timeframe = args.get("timeframe", "Daily")
        key = f"portfolio:{timeframe}"

        def _load():
            from portfolio.portfolio_dashboard import get_data
            from portfolio.portfolio_db import get_positions

            raw = get_data(timeframe=timeframe)
            holdings = {
                p["ticker"]: {
                    "cost_basis": p.get("cost_basis"),
                    "shares": p.get("shares"),
                    "direction": p.get("direction"),
                    "asset": p.get("asset"),
                }
                for p in get_positions()
            }
            return serialize_value(
                {
                    "holdings": holdings,
                    "analytics": raw.get("analytics"),
                    "timeframe": raw.get("timeframe"),
                    "timestamp": raw.get("timestamp"),
                }
            )

        data, meta = _fetch_with_cache(short_cache, key, _load)
        return data, meta

    if name == "get_yield_curve":
        lookback_days = int(args.get("lookback_days", 90))
        key = f"yield_curve:{lookback_days}"

        def _load():
            from government_bonds.yield_curve import get_data

            return serialize_value(get_data(lookback_days=lookback_days))

        data, meta = _fetch_with_cache(short_cache, key, _load)
        return data, meta

    if name == "get_bond_dashboard":
        tenor = args.get("tenor")
        key = f"bond_dashboard:{tenor or 'all'}"

        def _load():
            from government_bonds.bond_dashboard import get_data

            data = get_data()
            if tenor and tenor in ("2Y", "10Y", "30Y"):
                for country in data.get("countries", {}).values():
                    tenors = country.get("tenors", {})
                    country["tenors"] = {t: v for t, v in tenors.items() if t == tenor}
            return serialize_value(data)

        data, meta = _fetch_with_cache(short_cache, key, _load)
        return data, meta

    if name == "get_sentiment":
        key = "agent_sentiment_snapshot:v2"

        def _load():
            from api.routers.sentiment import get_put_call, get_surveys, get_volatility

            source_errors: dict[str, str] = {}

            def _safe_fetch(label: str, fn: Callable[[], Any], fallback: Any):
                try:
                    return fn()
                except Exception as exc:  # noqa: BLE001 - surfaced in quality issues
                    source_errors[label] = str(exc)
                    return fallback

            with ThreadPoolExecutor(max_workers=3) as pool:
                f_pc = pool.submit(_safe_fetch, "put_call", lambda: get_put_call(lookback_days=180), {})
                f_sv = pool.submit(_safe_fetch, "surveys", get_surveys, {"aaii": [], "naaim": [], "errors": {}})
                f_vl = pool.submit(_safe_fetch, "volatility", lambda: get_volatility(lookback_days=365), [])
                put_call = f_pc.result()
                surveys = f_sv.result()
                volatility = f_vl.result()

            surveys_dict = surveys if isinstance(surveys, dict) else {"aaii": [], "naaim": [], "errors": {}}
            base_errors = surveys_dict.get("errors") if isinstance(surveys_dict.get("errors"), dict) else {}
            merged_errors = {str(k): str(v) for k, v in base_errors.items() if isinstance(v, str) and v.strip()}
            merged_errors.update(source_errors)
            surveys_dict["errors"] = merged_errors
            return _build_agent_sentiment_snapshot(
                put_call=put_call if isinstance(put_call, dict) else {},
                surveys=surveys_dict,
                volatility=volatility if isinstance(volatility, list) else [],
            )

        data, meta = _fetch_with_cache(short_cache, key, _load)
        quality = data.get("quality") if isinstance(data, dict) else {}
        if isinstance(quality, dict):
            meta["quality_ok"] = bool(quality.get("ok"))
        return data, meta

    if name == "get_central_banks":
        key = "central_banks"

        def _load():
            from macro.central_banks.central_bank import get_data

            return serialize_value(get_data())

        data, meta = _fetch_with_cache(long_cache, key, _load)
        return data, meta

    if name == "get_industry_monitor":
        refresh = bool(args.get("refresh", False))
        key = f"industry_monitor:{refresh}"
        from macro.industry.industry_monitor import get_data

        if refresh:
            return serialize_value(get_data(refresh=True)), {"cache": "bypass"}

        def _load():
            return serialize_value(get_data(refresh=False))

        data, meta = _fetch_with_cache(long_cache, key, _load)
        return data, meta

    if name == "get_breakout":
        key = "breakout"

        def _load():
            from macro.breakout.breakout import get_data

            return serialize_value(get_data())

        data, meta = _fetch_with_cache(short_cache, key, _load)
        return data, meta

    if name == "query_ontology":
        from ontology.service import OntologyQueryService

        raw_filters = args.get("filters")
        if isinstance(raw_filters, dict):
            filters = raw_filters
        elif isinstance(raw_filters, str):
            try:
                parsed = json.loads(raw_filters)
                filters = parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, TypeError):
                filters = {}
        else:
            filters = {}
        query = args.get("query")
        intent = args.get("intent")
        timeframe = args.get("timeframe", "Daily")
        include_graph = bool(args.get("include_graph", False))
        run_id = args.get("run_id")
        refresh_snapshot = bool(args.get("refresh_snapshot", False))

        cache_token = json.dumps(
            {
                "query": query,
                "intent": intent,
                "filters": filters,
                "timeframe": timeframe,
                "include_graph": include_graph,
                "run_id": run_id,
                "refresh_snapshot": refresh_snapshot,
            },
            sort_keys=True,
            default=str,
        )
        key = f"ontology_query:{cache_token}"

        def _load():
            service = OntologyQueryService()
            result = service.query(
                query=str(query) if isinstance(query, str) else None,
                intent=str(intent) if isinstance(intent, str) else None,
                filters=filters,
                timeframe=str(timeframe) if isinstance(timeframe, str) else "Daily",
                include_graph=include_graph,
                run_id=str(run_id) if isinstance(run_id, str) and run_id.strip() else None,
                refresh_snapshot=refresh_snapshot,
            )
            return serialize_value(result)

        data, meta = _fetch_with_cache(short_cache, key, _load)
        meta["high_cost"] = True
        return data, meta

    if name == "get_thesis":
        ticker_raw = str(args.get("ticker") or "").strip().upper()
        if not ticker_raw:
            return {"error": "Missing required parameter: ticker"}, {"cache": "n/a"}
        key = f"thesis:{ticker_raw}"

        def _load():
            return _fetch_thesis(ticker_raw)

        data, meta = _fetch_with_cache(long_cache, key, _load)
        return data, meta

    if name == "get_thesis_evaluations":
        ticker_raw = str(args.get("ticker") or "").strip().upper()
        if not ticker_raw:
            return {"error": "Missing required parameter: ticker"}, {"cache": "n/a"}
        limit = int(args.get("limit", 10))
        limit = max(1, min(limit, 50))
        key = f"thesis_evaluations:{ticker_raw}:{limit}"

        def _load():
            return _fetch_thesis_evaluations(ticker_raw, limit)

        data, meta = _fetch_with_cache(long_cache, key, _load)
        return data, meta

    if name == "search_knowledge_base":
        query = str(args.get("query") or "").strip()
        if not query:
            return {"error": "Missing required parameter: query"}, {"cache": "n/a"}
        doc_types_raw = str(args.get("doc_types") or "").strip()
        tickers_raw = str(args.get("tickers") or "").strip()
        top_k = int(args.get("top_k", 5))
        top_k = max(1, min(top_k, 20))

        doc_types = [t.strip() for t in doc_types_raw.split(",") if t.strip()] or None
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()] or None

        try:
            from api.retrieval import search

            results = search(query=query, doc_types=doc_types, tickers=tickers, top_k=top_k)
            return {"results": results, "query": query, "count": len(results)}, {"cache": "n/a"}
        except ImportError:
            return {
                "error": "Knowledge base search unavailable (sentence-transformers not installed)",
            }, {"cache": "n/a"}
        except Exception as exc:
            return {"error": f"Search failed: {exc}"}, {"cache": "n/a"}

    if name == "get_ontology_diff":
        run_id_before = str(args.get("run_id_before") or "").strip() or None
        run_id_after = str(args.get("run_id_after") or "").strip() or None

        try:
            from ontology.service import OntologyQueryService

            svc = OntologyQueryService()

            if run_id_before and run_id_after:
                diff = svc.compare_snapshots(run_id_before, run_id_after)
            else:
                # Auto-select: get latest two runs
                runs = svc.list_runs(limit=5)
                if len(runs) < 2:
                    return {"error": f"Need at least 2 ontology snapshots to compare. Only found {len(runs)}."}, {
                        "cache": "n/a"
                    }
                rid_after = run_id_after or str(runs[0].get("run_id", ""))
                rid_before = run_id_before or str(runs[1].get("run_id", ""))
                diff = svc.compare_snapshots(rid_before, rid_after)

            return serialize_value(diff), {"cache": "n/a"}
        except Exception as exc:
            return {"error": f"Ontology diff failed: {exc}"}, {"cache": "n/a"}

    # -------------------------------------------------------------------
    # Web search
    # -------------------------------------------------------------------
    if name == "search_web":
        query = str(args.get("query") or "").strip()
        if not query:
            return {"error": "Missing required parameter: query"}, {"cache": "n/a"}

        key = f"web_search:{query[:200].lower()}"

        def _load():
            return _run_search_web(query)

        data, meta = _fetch_with_cache(short_cache, key, _load)
        meta["high_cost"] = True
        return data, meta

    # -------------------------------------------------------------------
    # Investing OS — read tools
    # -------------------------------------------------------------------
    if name == "get_catalysts":
        from portfolio.core_db import get_catalysts

        ticker = args.get("ticker", "").strip().upper()
        return get_catalysts(ticker), {"cache": "n/a"}

    if name == "get_kill_conditions":
        from portfolio.core_db import get_kill_conditions

        ticker = args.get("ticker", "").strip().upper()
        return get_kill_conditions(ticker), {"cache": "n/a"}

    if name == "get_action_items":
        from portfolio.core_db import get_action_items

        return get_action_items(
            ticker=args.get("ticker"),
            status=args.get("status", "open"),
        ), {"cache": "n/a"}

    if name == "get_watch_triggers":
        from portfolio.core_db import get_watch_triggers

        return get_watch_triggers(
            ticker=args.get("ticker"),
            status=args.get("status", "active"),
        ), {"cache": "n/a"}

    if name == "get_pending_approvals":
        from portfolio.core_db import get_pending_approvals

        return get_pending_approvals(
            ticker=args.get("ticker"),
            status=args.get("status", "pending"),
        ), {"cache": "n/a"}

    if name == "get_dossier":
        from api.routers.dossier import get_dossier as _get_dossier

        ticker = args.get("ticker", "").strip().upper()
        return _get_dossier(ticker), {"cache": "n/a"}

    if name == "get_workflow_history":
        from portfolio.core_db import get_workflow_runs

        return get_workflow_runs(
            ticker=args.get("ticker"),
            workflow_name=args.get("workflow_name"),
            limit=int(args.get("limit", 10)),
        ), {"cache": "n/a"}

    # -------------------------------------------------------------------
    # Investing OS — propose tools (approval-gated writes)
    # -------------------------------------------------------------------
    if name == "propose_thesis_status_change":
        from portfolio.core_db import create_pending_approval

        ticker = args["ticker"].strip().upper()
        approval = create_pending_approval(
            entity_type="thesis_status",
            ticker=ticker,
            proposed_change={"new_status": args["new_status"], "reason": args["reason"]},
            reason=args["reason"],
            source_type="agent",
        )
        return {
            "status": "pending_approval_created",
            "approval_id": approval["id"],
            "message": f"Proposed thesis status change for {ticker} to '{args['new_status']}'. User must approve in Workspace.",
        }, {"cache": "n/a"}

    if name == "propose_action_item":
        from portfolio.core_db import create_pending_approval

        ticker = (args.get("ticker") or "").strip().upper() or None
        approval = create_pending_approval(
            entity_type="action_item",
            ticker=ticker,
            proposed_change={
                "description": args["description"],
                "action_type": args["action_type"],
                "urgency": args.get("urgency", "normal"),
            },
            reason=args["reason"],
            source_type="agent",
        )
        return {
            "status": "pending_approval_created",
            "approval_id": approval["id"],
            "message": f"Proposed action item{f' for {ticker}' if ticker else ''}. User must approve in Workspace.",
        }, {"cache": "n/a"}

    if name == "propose_catalyst_status_change":
        from portfolio.core_db import create_pending_approval

        ticker = args["ticker"].strip().upper()
        approval = create_pending_approval(
            entity_type="catalyst_status",
            ticker=ticker,
            entity_id=args["catalyst_id"],
            proposed_change={
                "catalyst_id": args["catalyst_id"],
                "status": args["new_status"],
                "evidence": args.get("evidence"),
            },
            reason=args["reason"],
            source_type="agent",
        )
        return {
            "status": "pending_approval_created",
            "approval_id": approval["id"],
            "message": f"Proposed catalyst status change for {ticker}. User must approve in Workspace.",
        }, {"cache": "n/a"}

    if name == "propose_kill_condition_status_change":
        from portfolio.core_db import create_pending_approval

        ticker = args["ticker"].strip().upper()
        approval = create_pending_approval(
            entity_type="kill_condition_status",
            ticker=ticker,
            entity_id=args["kill_condition_id"],
            proposed_change={
                "kill_condition_id": args["kill_condition_id"],
                "status": args["new_status"],
            },
            reason=args["reason"],
            source_type="agent",
        )
        return {
            "status": "pending_approval_created",
            "approval_id": approval["id"],
            "message": f"Proposed kill condition status change for {ticker}. User must approve in Workspace.",
        }, {"cache": "n/a"}

    if name == "propose_watch_trigger":
        from portfolio.core_db import create_pending_approval

        ticker = (args.get("ticker") or "").strip().upper() or None
        approval = create_pending_approval(
            entity_type="watch_trigger",
            ticker=ticker,
            proposed_change={
                "condition": args["condition"],
                "trigger_type": args["trigger_type"],
            },
            reason=args["reason"],
            source_type="agent",
        )
        return {
            "status": "pending_approval_created",
            "approval_id": approval["id"],
            "message": f"Proposed watch trigger{f' for {ticker}' if ticker else ''}. User must approve in Workspace.",
        }, {"cache": "n/a"}

    raise ValueError(f"Unknown tool: {name}")
