"""
Tool registry for the AI agent.

Each tool wraps an existing get_data() / get_snapshot() function from the
analysis modules. Tool definitions use a JSON-schema format that can be adapted
for different LLM tool-calling APIs.
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from api.agent_governance import (
    AgentGovernanceError,
    ToolTimeoutError,
    blocked_tool_payload,
    evaluate_tool_call,
    run_with_timeout,
    should_retry_tool_error,
    tool_governance_meta,
    validate_tool_output,
)
from api.cache import get_cached, long_cache, set_cached, short_cache
from api.serializers import serialize_value
from ontology.action_registry import (
    ActionContext as RegistryActionContext,
)
from ontology.action_registry import (
    ActionValidationError,
    get_tool_exposure,
    is_agent_tool_exposed,
    iter_tool_exposures,
    propose_action_from_tool,
    validate_tool_input,
)
from ontology.policy import Actor, PolicyDenied, actor_cache_key, admin_actor

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

_BASE_TOOL_DEFINITIONS: list[dict] = [
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
                "include_history": {
                    "type": "boolean",
                    "description": "Include weekly historical regime series. Default false for faster chat responses.",
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
            "direction, cost basis, share quantity, conviction, P&L, contribution, and "
            "price data. Portfolio performance fields are direction-adjusted: price declines "
            "are favorable for short positions. Never judge a position from raw price moves "
            "alone; combine direction, quantity, cost basis, conviction, and P&L/return fields. "
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
                "include_hedges": {
                    "type": "boolean",
                    "description": "Include hedge rows. Default false; use only for hedge, beta, or risk-exposure questions.",
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
                    "description": "Optional filters: tickers, sectors, assets, min_risk_score.",
                },
                "page": {
                    "type": "integer",
                    "description": "Optional 1-based results page. Defaults to 1.",
                },
                "page_size": {
                    "type": "integer",
                    "description": "Optional page size from 1 to 100. Defaults to 25.",
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
                "as_of": {
                    "type": "string",
                    "description": "Optional valid-time timestamp for temporal ontology reads.",
                },
                "tx_as_of": {
                    "type": "string",
                    "description": "Optional transaction-time timestamp for what Talisman knew then.",
                },
                "include_history": {
                    "type": "boolean",
                    "description": "If true, include historical temporal versions where supported.",
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
            "Search across all indexed research documents — investment theses, uploaded news digests, "
            "weekly reports, daily reports, and past conversation summaries — using semantic similarity. "
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
                        "Comma-separated document types to search. Options: thesis, news_digest, "
                        "weekly_report, daily_report, conversation_summary. Leave empty to search all."
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
                    "description": "Proposed new status: active|under_review|invalidated",
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
                    "description": (
                        "Type: price_level|technical|fundamental|fundamental_news|event|news_event|macro|custom"
                    ),
                },
                "definition": {
                    "type": "object",
                    "description": "Optional machine-readable executable trigger definition.",
                },
                "reason": {"type": "string", "description": "Why this trigger matters"},
            },
            "required": ["condition", "trigger_type", "reason"],
        },
    },
]


@dataclass(frozen=True, slots=True)
class AgentCapability:
    """Provider-neutral description of a capability Stan may call."""

    name: str
    description: str
    parameters: dict[str, Any]
    category: str
    access_mode: str
    aliases: tuple[str, ...] = ()
    selectable: bool = True
    required_scopes: tuple[str, ...] = ()
    account_scope: str | None = "default-account"
    portfolio_scope: str | None = "default-portfolio"
    data_sensitivity: str = "public_market"
    provider_egress: str = "external_allowed"
    timeout_s: float = 15.0
    retry_policy: dict[str, Any] | None = None
    token_budget: int | None = None
    cost_budget_usd: float | None = None
    rate_limit: dict[str, Any] | None = None
    audit_level: str = "standard"
    failure_mode: str = "partial_allowed"

    @property
    def schema_safe(self) -> bool:
        return self.parameters.get("type") == "object" and isinstance(self.parameters.get("properties"), dict)

    def to_tool_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def _schema(properties: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties or {}, "required": required or []}


def _cap(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
    *,
    category: str,
    access_mode: str = "read",
    aliases: tuple[str, ...] = (),
    selectable: bool = True,
) -> AgentCapability:
    return AgentCapability(
        name=name,
        description=description,
        parameters=parameters or _schema(),
        category=category,
        access_mode=access_mode,
        aliases=aliases,
        selectable=selectable,
    )


_BASE_CAPABILITY_META: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "get_liquidity": ("macro", "read", ("liquidity", "global liquidity", "credit")),
    "get_market_breadth": ("technical", "read", ("breadth", "market breadth", "participation")),
    "get_vix_term_structure": ("technical", "read", ("vix", "volatility", "term structure")),
    "get_positioning": ("macro", "read", ("positioning", "cftc", "cot", "crowded")),
    "get_signal_aggregator": ("macro", "read", ("signal aggregator", "regime", "risk on", "risk off")),
    "get_economic_growth": ("macro", "read", ("growth", "economic growth", "cross asset")),
    "get_labor_market": ("macro", "read", ("labor", "jobs", "claims", "payrolls")),
    "get_housing": ("macro", "read", ("housing", "starts", "permits", "nahb")),
    "get_sector_metrics": ("equities", "read", ("sector", "rotation", "sector metrics")),
    "get_portfolio": ("portfolio", "read", ("portfolio", "holdings", "positions", "pnl", "p&l")),
    "get_yield_curve": ("fixed_income", "read", ("yield curve", "rates", "bonds")),
    "get_bond_dashboard": ("fixed_income", "read", ("bond dashboard", "sovereign yields")),
    "get_sentiment": ("macro", "read", ("sentiment", "put call", "aaii", "naaim")),
    "get_central_banks": ("macro", "read", ("central banks", "fed", "ecb", "boe", "boj")),
    "get_industry_monitor": ("macro", "read", ("industry monitor", "transcripts", "management commentary")),
    "get_breakout": ("technical", "read", ("breakout", "technical breakouts")),
    "query_ontology": ("ontology", "read", ("ontology", "risk exposure", "portfolio risk")),
    "get_thesis": ("thesis", "read", ("thesis", "investment thesis")),
    "get_thesis_evaluations": ("thesis", "read", ("thesis evaluations", "monitoring history")),
    "search_knowledge_base": ("research", "read", ("knowledge base", "past research", "notes", "news digests")),
    "get_ontology_diff": ("ontology", "read", ("ontology diff", "risk changes")),
    "search_web": ("research", "read", ("web", "news", "latest", "recent")),
    "get_catalysts": ("process", "read", ("catalysts",)),
    "get_kill_conditions": ("process", "read", ("kill conditions", "invalidation")),
    "get_action_items": ("process", "read", ("action items", "tasks")),
    "get_watch_triggers": ("process", "read", ("watch triggers", "monitoring")),
    "get_pending_approvals": ("approvals", "read", ("approvals", "pending approvals")),
    "get_dossier": ("portfolio", "read", ("dossier", "position dossier")),
    "get_workflow_history": ("workflows", "read", ("workflow history", "workflow runs")),
    "propose_thesis_status_change": ("thesis", "proposal", ("propose thesis status", "thesis status")),
    "propose_action_item": ("process", "proposal", ("propose action", "action item")),
    "propose_catalyst_status_change": ("process", "proposal", ("propose catalyst status",)),
    "propose_kill_condition_status_change": ("process", "proposal", ("propose kill condition status",)),
    "propose_watch_trigger": ("process", "proposal", ("propose watch trigger", "set trigger")),
}


def _base_capability(tool: dict[str, Any]) -> AgentCapability:
    name = str(tool["name"])
    category, access_mode, aliases = _BASE_CAPABILITY_META.get(
        name,
        ("misc", "read", tuple(name.replace("_", " ").split())),
    )
    return _cap(
        name,
        str(tool.get("description", "")),
        tool.get("parameters") or _schema(),
        category=category,
        access_mode=access_mode,
        aliases=aliases + (name.replace("_", " "),),
    )


_STRING = {"type": "string"}
_NUMBER = {"type": "number"}
_INTEGER = {"type": "integer"}
_BOOLEAN = {"type": "boolean"}
_OBJECT = {"type": "object"}
_ARRAY_OBJECTS = {"type": "array", "items": {"type": "object"}}


_EXTRA_CAPABILITIES: list[AgentCapability] = [
    _cap(
        "search_agent_capabilities",
        "Search Stan's available app capabilities by natural-language query. Use when you need a tool that was not in the initially visible set.",
        _schema(
            {
                "query": {"type": "string", "description": "Capability or app feature to find."},
                "top_k": {"type": "integer", "description": "Maximum matches to return. Default 8."},
            },
            ["query"],
        ),
        category="agent",
        aliases=("capability search", "available tools", "what can you access"),
    ),
    _cap(
        "get_workspace",
        "Fetch the Workspace landing page aggregate: regime, portfolio summary, thesis pressure, approvals, action items, triggers, and workflow runs.",
        category="workspace",
        aliases=("workspace", "dashboard home"),
    ),
    _cap(
        "get_portfolio_positions",
        "Fetch editable portfolio positions, optionally including hedges.",
        _schema({"include_hedges": _BOOLEAN}),
        category="portfolio",
        aliases=("portfolio positions", "editable holdings"),
    ),
    _cap(
        "get_hedge_positions",
        "Fetch hedge positions from the portfolio editor.",
        category="portfolio",
        aliases=("hedge positions", "hedges"),
    ),
    _cap(
        "get_portfolio_news",
        "List uploaded news digests, or fetch one digest when digest_id is provided.",
        _schema({"digest_id": {"type": "string", "description": "Optional digest id for detail."}}),
        category="research",
        aliases=("news digests", "portfolio news", "uploaded news"),
    ),
    _cap(
        "get_research_notes",
        "Fetch research notes, optionally filtered by ticker.",
        _schema({"ticker": _STRING, "limit": _INTEGER}),
        category="research",
        aliases=("research notes", "notes"),
    ),
    _cap(
        "get_workflow_run",
        "Fetch one persisted workflow run by run_id.",
        _schema({"run_id": _STRING}, ["run_id"]),
        category="workflows",
        aliases=("workflow run detail", "run detail"),
    ),
    _cap(
        "get_weekly_report", "Fetch the weekly report payload.", category="reports", aliases=("weekly report", "weekly")
    ),
    _cap(
        "get_commodities",
        "Fetch the commodity dashboard across major commodities for a timeframe.",
        _schema(
            {"timeframe": {"type": "string", "description": "This Week, Daily, Weekly, or Monthly. Default Daily."}}
        ),
        category="commodities",
        aliases=("commodities dashboard", "commodity prices"),
    ),
    _cap(
        "get_commodities_curve",
        "Fetch futures curve data for CL, BZ, NG, or TTF.",
        _schema({"commodity": _STRING, "lookback_days": _INTEGER}),
        category="commodities",
        aliases=("commodities curve", "oil curve", "gas curve", "futures curve"),
    ),
    _cap(
        "get_commodity_research",
        "Fetch the commodity proxy research screener.",
        category="commodities",
        aliases=("commodity research", "commodity proxy", "aluminum research"),
    ),
    _cap(
        "get_country_dashboard",
        "Fetch the country dashboard.",
        category="macro",
        aliases=("country dashboard", "countries"),
    ),
    _cap(
        "get_index_dashboard", "Fetch the index dashboard.", category="equities", aliases=("index dashboard", "indices")
    ),
    _cap("get_fx_dashboard", "Fetch the FX dashboard.", category="fx", aliases=("fx dashboard", "currencies")),
    _cap(
        "get_momentum",
        "Fetch price momentum dashboard data.",
        category="portfolio",
        aliases=("momentum", "price momentum"),
    ),
    _cap(
        "get_top50_breadth",
        "Fetch S&P 500 top-50 breadth data.",
        category="technical",
        aliases=("top50 breadth", "top 50 breadth"),
    ),
    _cap(
        "get_price_volume_signals",
        "Fetch price-volume technical signals.",
        category="technical",
        aliases=("price volume", "volume signals"),
    ),
    _cap(
        "get_financials",
        "Fetch single-company financial history and metrics.",
        _schema({"ticker": _STRING}, ["ticker"]),
        category="equities",
        aliases=("financials", "company financials", "revenue", "eps"),
    ),
    _cap(
        "get_dcf_historical",
        "Fetch historical financials and multiples for DCF work.",
        _schema({"ticker": _STRING}, ["ticker"]),
        category="equities",
        aliases=("dcf historical", "valuation historical"),
    ),
    _cap(
        "run_dcf_valuation",
        "Run a DCF valuation from explicit assumptions.",
        _schema(
            {
                "ticker": _STRING,
                "revenue_growth_rates": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Five annual revenue growth rates as decimals.",
                },
                "ebitda_margin": _NUMBER,
                "tax_rate": _NUMBER,
                "da_pct_revenue": _NUMBER,
                "nwc_pct_revenue": _NUMBER,
                "capex_pct_revenue": _NUMBER,
                "wacc": _NUMBER,
                "terminal_growth_rates": _OBJECT,
                "exit_ev_ebitda": _OBJECT,
                "exit_ev_revenue": _OBJECT,
            },
            [
                "ticker",
                "revenue_growth_rates",
                "ebitda_margin",
                "da_pct_revenue",
                "nwc_pct_revenue",
                "capex_pct_revenue",
                "wacc",
                "exit_ev_ebitda",
                "exit_ev_revenue",
            ],
        ),
        category="equities",
        access_mode="compute",
        aliases=("run dcf", "dcf valuation", "valuation"),
    ),
    _cap(
        "run_chart",
        "Run technical analysis for a ticker.",
        _schema({"ticker": _STRING, "lookback": _STRING}, ["ticker"]),
        category="technical",
        access_mode="compute",
        aliases=("chart", "technical analysis"),
    ),
    _cap(
        "run_ratio_chart",
        "Run a ratio chart between two symbols.",
        _schema(
            {"symbol_a": _STRING, "symbol_b": _STRING, "start_date": _STRING, "end_date": _STRING, "method": _STRING},
            ["symbol_a", "symbol_b"],
        ),
        category="technical",
        access_mode="compute",
        aliases=("ratio chart", "pair ratio"),
    ),
    _cap(
        "get_fx_model_pairs",
        "List supported FX model pairs.",
        category="fx",
        aliases=("fx model pairs", "currency pairs"),
    ),
    _cap(
        "run_fx_model",
        "Run the FX valuation/forecast model for a supported pair.",
        _schema({"pair": _STRING, "bootstrap": _INTEGER, "skip_bis": _BOOLEAN, "horizons": _STRING}, ["pair"]),
        category="fx",
        access_mode="compute",
        aliases=("fx model", "currency model"),
    ),
    _cap(
        "run_quality_screen",
        "Run the quality equity screen.",
        _schema({"universe": _STRING, "tickers": _STRING, "benchmark": _STRING, "input_mode": _STRING}),
        category="screeners",
        access_mode="compute",
        aliases=("quality screen", "quality screener"),
    ),
    _cap(
        "run_short_screen",
        "Start or reuse a short screen job.",
        _schema(
            {
                "input_mode": _STRING,
                "universe": _STRING,
                "tickers": _STRING,
                "pb_threshold": _NUMBER,
                "loss_type": _STRING,
                "check_issuance": _BOOLEAN,
                "check_revenue": _BOOLEAN,
                "max_revenue_growth": _NUMBER,
                "check_eps": _BOOLEAN,
                "max_eps_growth": _NUMBER,
                "check_52w_positive": _BOOLEAN,
                "check_min_drawdown": _BOOLEAN,
                "min_drawdown_pct": _NUMBER,
                "check_max_drawdown": _BOOLEAN,
                "max_drawdown_pct": _NUMBER,
                "check_3m_neg_momentum": _BOOLEAN,
                "check_2m_neg_rel_momentum": _BOOLEAN,
                "rel_momentum_benchmark": _STRING,
            }
        ),
        category="screeners",
        access_mode="compute",
        aliases=("short screen", "short screener"),
    ),
    _cap(
        "run_long_screen",
        "Start or reuse a long screen job.",
        _schema(
            {
                "input_mode": _STRING,
                "universe": _STRING,
                "tickers": _STRING,
                "pb_threshold": _NUMBER,
                "profit_type": _STRING,
                "check_issuance": _BOOLEAN,
                "check_revenue": _BOOLEAN,
                "min_revenue_growth": _NUMBER,
                "check_eps": _BOOLEAN,
                "min_eps_growth": _NUMBER,
                "check_ebit_multiple": _BOOLEAN,
                "max_ebit_multiple": _NUMBER,
                "check_52w_positive": _BOOLEAN,
                "check_min_drawdown": _BOOLEAN,
                "min_drawdown_pct": _NUMBER,
                "check_max_drawdown": _BOOLEAN,
                "max_drawdown_pct": _NUMBER,
                "check_3m_pos_momentum": _BOOLEAN,
                "check_2m_pos_rel_momentum": _BOOLEAN,
                "rel_momentum_benchmark": _STRING,
            }
        ),
        category="screeners",
        access_mode="compute",
        aliases=("long screen", "long screener"),
    ),
    _cap(
        "run_fundamental_momentum",
        "Start or reuse an EPS/revenue fundamental momentum screen.",
        _schema(
            {
                "screen_type": _STRING,
                "universe": _STRING,
                "tickers": _STRING,
                "benchmark": _STRING,
                "input_mode": _STRING,
            }
        ),
        category="screeners",
        access_mode="compute",
        aliases=("fundamental momentum", "eps momentum", "revenue momentum"),
    ),
    _cap(
        "run_portfolio_analyzer",
        "Start or reuse the portfolio analyzer.",
        _schema({"book": _NUMBER, "target_leverage": _NUMBER, "beta_neutral": _BOOLEAN}),
        category="portfolio",
        access_mode="compute",
        aliases=("portfolio analyzer", "portfolio optimizer"),
    ),
    _cap(
        "run_portfolio_sizer",
        "Start or reuse the portfolio sizer.",
        _schema({"book": _NUMBER, "target_leverage": _NUMBER, "positions": _ARRAY_OBJECTS}),
        category="portfolio",
        access_mode="compute",
        aliases=("portfolio sizer", "sizing"),
    ),
    _cap(
        "get_portfolio_sizer_prefill",
        "Fetch portfolio sizer prefill positions.",
        category="portfolio",
        aliases=("sizer prefill",),
    ),
    _cap(
        "run_hedging_tool",
        "Start or reuse the hedging tool.",
        _schema({"book": _NUMBER, "positions": _ARRAY_OBJECTS}),
        category="portfolio",
        access_mode="compute",
        aliases=("hedging tool", "hedge analysis"),
    ),
    _cap(
        "get_hedging_tool_prefill",
        "Fetch hedging tool prefill positions.",
        category="portfolio",
        aliases=("hedging prefill",),
    ),
    _cap(
        "get_hedging_portfolio_weights",
        "Derive hedging weights from the portfolio database.",
        _schema({"book": _NUMBER}),
        category="portfolio",
        aliases=("hedging weights", "portfolio weights"),
    ),
    _cap(
        "run_hedging_recommendation",
        "Generate LLM hedging recommendations from hedging analysis tables.",
        _schema(
            {
                "net_beta_spy": _NUMBER,
                "net_beta_iwm": _NUMBER,
                "post_hedge_beta_spy": _NUMBER,
                "post_hedge_beta_iwm": _NUMBER,
                "gross_input": _NUMBER,
                "net_input": _NUMBER,
                "gross_after_hedging": _NUMBER,
                "volatility_after_hedging": _NUMBER,
                "hedge_spy_weight": _NUMBER,
                "hedge_iwm_weight": _NUMBER,
                "positions_df": _ARRAY_OBJECTS,
                "hedges_df": _ARRAY_OBJECTS,
                "book_size": _NUMBER,
            }
        ),
        category="portfolio",
        access_mode="compute",
        aliases=("hedging recommendation", "hedge recommendation"),
    ),
    _cap(
        "propose_portfolio_positions_update",
        "Propose replacing editable portfolio positions. Creates a pending approval.",
        _schema({"positions": _ARRAY_OBJECTS, "reason": _STRING}, ["positions", "reason"]),
        category="portfolio",
        access_mode="proposal",
        aliases=("propose portfolio edit", "update portfolio positions"),
    ),
    _cap(
        "propose_hedge_positions_update",
        "Propose replacing hedge positions. Creates a pending approval.",
        _schema({"positions": _ARRAY_OBJECTS, "reason": _STRING}, ["positions", "reason"]),
        category="portfolio",
        access_mode="proposal",
        aliases=("propose hedge edit", "update hedge positions"),
    ),
    _cap(
        "propose_thesis_content_update",
        "Propose replacing a ticker's thesis markdown. Creates a pending approval.",
        _schema({"ticker": _STRING, "content": _STRING, "reason": _STRING}, ["ticker", "content", "reason"]),
        category="thesis",
        access_mode="proposal",
        aliases=("propose thesis edit", "update thesis content"),
    ),
    _cap(
        "propose_catalyst",
        "Propose creating a catalyst. Creates a pending approval.",
        _schema(
            {"ticker": _STRING, "description": _STRING, "category": _STRING, "target_date": _STRING, "reason": _STRING},
            ["ticker", "description", "reason"],
        ),
        category="process",
        access_mode="proposal",
        aliases=("propose catalyst", "create catalyst"),
    ),
    _cap(
        "propose_kill_condition",
        "Propose creating a kill condition. Creates a pending approval.",
        _schema(
            {"ticker": _STRING, "condition": _STRING, "metric": _STRING, "threshold": _STRING, "reason": _STRING},
            ["ticker", "condition", "reason"],
        ),
        category="process",
        access_mode="proposal",
        aliases=("propose kill condition", "create kill condition"),
    ),
    _cap(
        "propose_research_note",
        "Propose creating a research note. Creates a pending approval.",
        _schema(
            {"title": _STRING, "content": _STRING, "ticker": _STRING, "note_type": _STRING, "reason": _STRING},
            ["title", "content", "reason"],
        ),
        category="research",
        access_mode="proposal",
        aliases=("propose research note", "create research note"),
    ),
    _cap(
        "propose_news_digest_delete",
        "Propose deleting an uploaded news digest. Creates a pending approval.",
        _schema({"digest_id": _STRING, "reason": _STRING}, ["digest_id", "reason"]),
        category="research",
        access_mode="proposal",
        aliases=("delete news digest", "remove digest"),
    ),
]


def _capability_from_exposure(tool) -> AgentCapability:
    return AgentCapability(
        name=tool.tool_name,
        description=tool.description,
        parameters=tool.parameters,
        category=tool.category,
        access_mode=tool.access_mode,
        aliases=tool.aliases,
        selectable=tool.selectable,
        required_scopes=tuple(tool.required_scopes),
        account_scope=tool.account_scope,
        portfolio_scope=tool.portfolio_scope,
        data_sensitivity=tool.data_sensitivity,
        provider_egress=tool.provider_egress,
        timeout_s=tool.timeout_s,
        retry_policy=dict(tool.retry_policy),
        token_budget=tool.token_budget,
        cost_budget_usd=tool.cost_budget_usd,
        rate_limit=dict(tool.rate_limit),
        audit_level=tool.audit_level,
        failure_mode=tool.failure_mode,
    )


def _build_capability_registry() -> list[AgentCapability]:
    by_name: dict[str, AgentCapability] = {}
    for tool in iter_tool_exposures(agent_exposed_only=True):
        cap = _capability_from_exposure(tool)
        if cap.name in by_name:
            raise RuntimeError(f"Duplicate agent capability: {cap.name}")
        by_name[cap.name] = cap
    return list(by_name.values())


AGENT_CAPABILITIES: list[AgentCapability] = _build_capability_registry()
AGENT_CAPABILITY_BY_NAME: dict[str, AgentCapability] = {cap.name: cap for cap in AGENT_CAPABILITIES}
TOOL_DEFINITIONS: list[dict[str, Any]] = [cap.to_tool_definition() for cap in AGENT_CAPABILITIES]

# Tool name -> index lookup
_TOOL_INDEX = {t["name"]: i for i, t in enumerate(TOOL_DEFINITIONS)}


def list_agent_capabilities() -> list[dict[str, Any]]:
    """Return non-secret metadata for Stan's callable app capabilities."""
    return [
        {
            "name": cap.name,
            "category": cap.category,
            "access_mode": cap.access_mode,
            "description": cap.description,
            "aliases": list(cap.aliases),
            "schema_safe": cap.schema_safe,
            "selectable": cap.selectable,
            "governance": {
                "required_scopes": list(cap.required_scopes),
                "account_scope": cap.account_scope,
                "portfolio_scope": cap.portfolio_scope,
                "data_sensitivity": cap.data_sensitivity,
                "provider_egress": cap.provider_egress,
                "timeout_s": cap.timeout_s,
                "retry_policy": cap.retry_policy or {},
                "token_budget": cap.token_budget,
                "cost_budget_usd": cap.cost_budget_usd,
                "rate_limit": cap.rate_limit or {},
                "audit_level": cap.audit_level,
                "failure_mode": cap.failure_mode,
            },
        }
        for cap in AGENT_CAPABILITIES
    ]


def _capability_search_text(cap: AgentCapability) -> str:
    return " ".join(
        [
            cap.name.replace("_", " "),
            cap.category.replace("_", " "),
            cap.access_mode,
            cap.description,
            " ".join(cap.aliases),
        ]
    ).lower()


def search_agent_capabilities(query: str, top_k: int = 8) -> dict[str, Any]:
    """Small lexical search used as a fallback tool discovery path."""
    q = (query or "").strip().lower()
    if not q:
        return {"query": query, "matches": [], "count": 0}
    tokens = [t for t in re.split(r"[^a-z0-9.]+", q) if t]
    matches: list[tuple[int, AgentCapability]] = []
    for cap in AGENT_CAPABILITIES:
        if not cap.selectable:
            continue
        haystack = _capability_search_text(cap)
        score = 0
        if q in haystack:
            score += 5
        for token in tokens:
            if token == cap.name.lower():
                score += 6
            elif token in haystack:
                score += 2
        if score:
            matches.append((score, cap))
    matches.sort(key=lambda item: (-item[0], item[1].name))
    safe_top_k = max(1, min(int(top_k or 8), 20))
    rows = [
        {
            "name": cap.name,
            "category": cap.category,
            "access_mode": cap.access_mode,
            "description": cap.description,
            "aliases": list(cap.aliases),
            "score": score,
        }
        for score, cap in matches[:safe_top_k]
    ]
    return {"query": query, "matches": rows, "count": len(rows)}


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
    if isinstance(payload.get("_meta"), dict):
        out["_meta"] = _compact_generic(payload.get("_meta"), max_depth=3, list_limit=20, dict_limit=20)

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

    # Agent-native portfolio payloads are already compact and include full
    # exposure context. Keep the shape stable for the model.
    if isinstance(payload.get("positions"), list):
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


def _series_edge_point(series: Any, *, first: bool) -> dict[str, Any] | None:
    if series is None:
        return None
    try:
        clean = series.dropna() if hasattr(series, "dropna") else series
        if hasattr(clean, "empty") and clean.empty:
            return None
        if isinstance(clean, list):
            rows = [row for row in clean if isinstance(row, dict)]
            if not rows:
                return None
            row = rows[0] if first else rows[-1]
            value = _to_float(row.get("value"))
            if value is None:
                return None
            return {"date": row.get("date"), "value": value}

        idx = 0 if first else -1
        value = _to_float(clean.iloc[idx])
        if value is None:
            return None
        date_value = clean.index[idx]
        if hasattr(date_value, "date"):
            date_out = date_value.date().isoformat()
        else:
            date_out = str(date_value)
        return {"date": date_out, "value": value}
    except Exception:
        return None


def _build_agent_portfolio_payload(
    raw: dict[str, Any], holdings: list[dict[str, Any]], *, include_hedges: bool
) -> dict[str, Any]:
    analytics_raw = raw.get("analytics")
    analytics: dict[str, Any] = analytics_raw if isinstance(analytics_raw, dict) else {}
    per_position_raw = analytics.get("per_position")
    per_position: dict[str, Any] = per_position_raw if isinstance(per_position_raw, dict) else {}
    portfolio_summary_raw = analytics.get("portfolio")
    portfolio_summary: dict[str, Any] = portfolio_summary_raw if isinstance(portfolio_summary_raw, dict) else {}
    raw_positions_raw = raw.get("positions")
    raw_positions: dict[str, Any] = raw_positions_raw if isinstance(raw_positions_raw, dict) else {}

    rows: list[dict[str, Any]] = []
    for holding in holdings:
        ticker = str(holding.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        perf = per_position.get(ticker)
        perf = perf if isinstance(perf, dict) else {}
        first = _series_edge_point(raw_positions.get(ticker), first=True)
        last = _series_edge_point(raw_positions.get(ticker), first=False)
        first_price = _to_float(first.get("value")) if isinstance(first, dict) else None
        current_price = _to_float(perf.get("current_price"))
        last_price = (
            current_price
            if current_price is not None
            else (_to_float(last.get("value")) if isinstance(last, dict) else None)
        )

        raw_price_return_pct = None
        if first_price is not None and last_price is not None and first_price != 0:
            raw_price_return_pct = round(((last_price - first_price) / first_price) * 100.0, 4)

        shares = holding.get("shares")
        row = {
            "ticker": ticker,
            "asset": holding.get("asset"),
            "direction": holding.get("direction"),
            "cost_basis": holding.get("cost_basis"),
            "shares": shares,
            "quantity": shares,
            "conviction": holding.get("conviction"),
            "contrarian": bool(holding.get("contrarian")),
            "role": holding.get("role") or "position",
            "current_price": last_price,
            "first_date": first.get("date") if isinstance(first, dict) else None,
            "first_price": first_price,
            "last_date": last.get("date") if isinstance(last, dict) else None,
            "raw_price_return_pct": raw_price_return_pct,
            "unrealized_pnl_pct": perf.get("unrealized_pnl_pct"),
            "unrealized_pnl_dollar": perf.get("unrealized_pnl_dollar"),
            "weekly_return_pct": perf.get("weekly_return_pct"),
            "monthly_return_pct": perf.get("monthly_return_pct"),
            "weekly_contribution_pct": perf.get("weekly_contribution_pct"),
            "monthly_contribution_pct": perf.get("monthly_contribution_pct"),
            "weight": perf.get("weight"),
            "drawdown_from_52w_pct": perf.get("drawdown_from_52w_pct"),
            "performance_basis": "direction_adjusted",
        }
        rows.append(row)

    long_count = sum(1 for row in rows if str(row.get("direction") or "").lower() == "long")
    short_count = sum(1 for row in rows if str(row.get("direction") or "").lower() == "short")
    payload: dict[str, Any] = serialize_value(
        {
            "timeframe": raw.get("timeframe"),
            "timestamp": raw.get("timestamp"),
            "context_scope": "positions_and_hedges" if include_hedges else "positions_only",
            "semantics": {
                "performance_fields": "direction_adjusted",
                "raw_price_return_pct": "not direction-adjusted; do not use alone for P&L judgment",
                "short_price_declines_are_favorable": True,
                "quantity_field": "shares",
            },
            "summary": {
                "position_count": len(rows),
                "long_count": long_count,
                "short_count": short_count,
                **portfolio_summary,
            },
            "positions": rows,
        }
    )
    return payload


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


def _cached_singleflight(
    cache,
    key: str,
    loader: Callable[[], Any],
    *,
    force_refresh: bool = False,
) -> tuple[Any, str]:
    if force_refresh:
        value = loader()
        set_cached(cache, key, value)
        return value, "refresh"

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


def _cache_freshness_meta(cache, value: Any, cache_status: str) -> dict[str, Any]:
    meta: dict[str, Any] = {"cache": cache_status}
    source_meta = value.get("_meta") if isinstance(value, dict) else None
    if isinstance(source_meta, dict):
        for key in ("fetched_at", "cache_ttl", "data_age_seconds", "stale"):
            if key in source_meta:
                meta[key] = source_meta[key]

    ttl = getattr(cache, "ttl", None)
    if "cache_ttl" not in meta and ttl is not None:
        meta["cache_ttl"] = ttl
    if "fetched_at" not in meta:
        meta["fetched_at"] = datetime.now().isoformat()
    if "fetched_at" in meta and "data_age_seconds" not in meta:
        try:
            fetched_at = datetime.fromisoformat(str(meta["fetched_at"]))
            now = datetime.now(fetched_at.tzinfo) if fetched_at.tzinfo else datetime.now()
            meta["data_age_seconds"] = max(0, round((now - fetched_at).total_seconds()))
        except Exception:
            pass
    if "stale" not in meta:
        age = _to_float(meta.get("data_age_seconds"))
        meta["stale"] = bool(age is not None and ttl is not None and age > float(ttl))
    if "stale" in meta:
        meta["fresh"] = not bool(meta["stale"])
    elif cache_status in {"hit", "miss_fetch", "miss_wait", "miss_refetch", "refresh"}:
        meta["fresh"] = True
    if cache_status == "refresh":
        meta["refreshed"] = True
    return meta


def _fetch_with_cache(
    cache,
    key: str,
    loader: Callable[[], Any],
    *,
    force_refresh: bool = False,
) -> tuple[Any, dict[str, Any]]:
    value, cache_status = _cached_singleflight(cache, key, loader, force_refresh=force_refresh)
    return value, _cache_freshness_meta(cache, value, cache_status)


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
    from llm_utils import MODEL_LOW, call_llm_text

    allowed_domains = list(_SEARCH_WEB_ALLOWED_DOMAINS_DEFAULT)
    attempts = 0
    while attempts < 3:
        attempts += 1
        try:
            text, citations, _response = call_llm_text(
                prompt=f"Find the latest news and developments about: {query}",
                model=MODEL_LOW,
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
                raise RuntimeError("All configured search domains were rejected by the LLM provider.") from exc

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


def _queue_tool_call_lineage_retry(
    *,
    name: str,
    safe_args: dict[str, Any],
    actor: Actor,
    provenance_event_id: str,
    provenance_context: dict[str, Any],
    status: str,
    payload: Any | None = None,
    error: str | None = None,
) -> None:
    from api import governance
    from portfolio import core_db

    lineage_root_id = governance.lineage_root(governance.REF_TOOL_CALL, provenance_event_id)
    bundle = governance.event_bundle(
        lineage_root_id=lineage_root_id,
        idempotency_key=f"tool_call:{provenance_event_id}:{status}:retry",
        provenance_events=[
            governance.provenance_event(
                event_id=provenance_event_id,
                event_type="tool_call",
                event_name=name,
                status=status,
                actor_type=getattr(actor, "actor_type", None),
                actor_id=getattr(actor, "actor_id", None),
                parent_event_id=provenance_context.get("parent_event_id"),
                workflow_run_id=provenance_context.get("workflow_run_id"),
                agent_session_id=provenance_context.get("agent_session_id"),
                input_value=safe_args,
                output_value=payload,
                summary={
                    "tool": name,
                    "status": status,
                    "arg_keys": sorted(str(key) for key in safe_args.keys()),
                    "call_id": provenance_context.get("call_id"),
                },
                metadata={"source": provenance_context.get("source") or "agent_tools.execute_tool"},
                error=error,
                lineage_root_id=lineage_root_id,
            )
        ],
        audit_events=[
            governance.audit_event(
                action_name=governance.EVENT_TOOL_CALL_COMPLETED,
                status=status,
                lineage_root_id=lineage_root_id,
                actor_type=getattr(actor, "actor_type", None) or "agent",
                actor_id=getattr(actor, "actor_id", None),
                object_refs=[{"type": governance.REF_TOOL_CALL, "id": provenance_event_id}],
                after_summary={"tool": name, "status": status},
                metadata={"arg_keys": sorted(str(key) for key in safe_args.keys())},
                error=error,
            )
        ],
    )
    core_db.enqueue_governance_outbox(
        bundle,
        idempotency_key=f"tool_call:{provenance_event_id}:{status}:retry",
        lineage_root_id=lineage_root_id,
    )


def _finish_tool_provenance(
    *,
    provenance_event_id: str | None,
    name: str,
    safe_args: dict[str, Any],
    actor: Actor,
    provenance_context: dict[str, Any],
    critical: bool,
    status: str,
    output_value: Any | None = None,
    summary: Any | None = None,
    metadata: Any | None = None,
    error: str | None = None,
) -> None:
    if not provenance_event_id:
        return
    try:
        from api import provenance

        provenance.finish_event(
            provenance_event_id,
            status=status,
            output_value=output_value,
            summary=summary,
            metadata=metadata,
            error=error,
            fail_closed=critical,
        )
    except Exception as exc:
        if critical:
            _queue_tool_call_lineage_retry(
                name=name,
                safe_args=safe_args,
                actor=actor,
                provenance_event_id=provenance_event_id,
                provenance_context=provenance_context,
                status=status,
                payload=output_value,
                error=error or str(exc) or exc.__class__.__name__,
            )
        else:
            logger.debug("Failed to finish tool provenance for %s", name, exc_info=True)


def _validated_tool_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    internal_args = {str(k): v for k, v in args.items() if str(k).startswith("_")}
    public_args = {str(k): v for k, v in args.items() if not str(k).startswith("_")}
    if name == "query_ontology" and isinstance(public_args.get("filters"), str):
        try:
            parsed_filters = json.loads(str(public_args["filters"]))
        except json.JSONDecodeError as exc:
            raise ActionValidationError("query_ontology.filters must be a valid JSON object") from exc
        if not isinstance(parsed_filters, dict):
            raise ActionValidationError("query_ontology.filters must decode to a JSON object")
        public_args["filters"] = parsed_filters
    typed = validate_tool_input(name, public_args)
    if hasattr(typed, "model_dump"):
        normalized = typed.model_dump(exclude_none=True)
    else:
        normalized = typed.dict(exclude_none=True)
    normalized.update(internal_args)
    return normalized


def _call_dispatch_with_governance(
    name: str,
    safe_args: dict[str, Any],
    *,
    actor: Actor,
    provenance_event_id: str | None,
    timeout_s: float,
    retry_policy: Mapping[str, Any],
) -> tuple[object, dict[str, Any], int]:
    max_attempts = max(1, int(retry_policy.get("max_attempts") or 1))
    backoff_s = max(0.0, float(retry_policy.get("backoff_s") or 0.0))
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result, dispatch_meta = run_with_timeout(
                lambda: _call_dispatch(name, safe_args, actor=actor, provenance_event_id=provenance_event_id),
                timeout_s=timeout_s,
            )
            return result, dispatch_meta, attempt
        except Exception as exc:  # noqa: BLE001 - tool failures are returned to the model
            last_exc = exc
            if attempt >= max_attempts or not should_retry_tool_error(exc):
                raise
            delay = backoff_s * (2 ** (attempt - 1))
            logger.warning(
                "Retryable tool error name=%s attempt=%d/%d delay=%.2fs error=%s",
                name,
                attempt,
                max_attempts,
                delay,
                exc,
            )
            if delay > 0:
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def execute_tool(
    name: str,
    arguments: dict,
    actor: Actor | None = None,
    provenance_context: dict[str, Any] | None = None,
) -> str:
    """Run the tool identified by *name* and return a JSON string for the model.

    Errors are caught and returned as ``{"error": "..."}`` so the model can
    inform the user instead of crashing the stream.
    """
    started = time.perf_counter()
    actor = actor or admin_actor(source="agent_tools")
    safe_args = dict(arguments) if isinstance(arguments, dict) else {}
    critical_tool_call = _is_proposal_tool(name)
    provenance_event_id: str | None = None
    pv_context = provenance_context or {}
    exposure = None
    policy_meta: dict[str, Any] = {}
    try:
        from api import provenance

        provenance_event_id = str(
            pv_context.get("event_id")
            or provenance.deterministic_id(
                "pv:tool_call",
                pv_context.get("agent_session_id") or pv_context.get("workflow_run_id") or "standalone",
                pv_context.get("parent_event_id"),
                name,
                provenance.stable_hash(safe_args),
                int(started * 1_000_000),
            )
        )
        provenance.start_event(
            event_id=provenance_event_id,
            event_type="tool_call",
            event_name=name,
            actor=actor,
            parent_event_id=pv_context.get("parent_event_id"),
            workflow_run_id=pv_context.get("workflow_run_id"),
            agent_session_id=pv_context.get("agent_session_id"),
            input_value=safe_args,
            summary={
                "tool": name,
                "arg_keys": sorted(str(key) for key in safe_args.keys()),
                "call_id": pv_context.get("call_id"),
            },
            metadata={
                "args_hash": provenance.stable_hash(safe_args),
                "source": pv_context.get("source") or "agent_tools.execute_tool",
            },
            criticality="financial_critical" if critical_tool_call else "operational",
            lineage_root_id=f"tool_call:{provenance_event_id}" if critical_tool_call else None,
            idempotency_key=f"tool_call:{provenance_event_id}:started" if critical_tool_call else None,
            retention_class="financial_lineage_7y" if critical_tool_call else provenance.DEFAULT_RETENTION_CLASS,
            fail_closed=critical_tool_call,
        )
    except Exception as exc:
        provenance_event_id = None
        if critical_tool_call:
            payload = _attach_meta(
                {"error": f"Failed to record mandatory tool lineage for {name}: {exc}", "type": "GovernanceWriteError"},
                {
                    "tool": name,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                    "status": "failed_closed",
                },
            )
            return _stable_json_dumps(payload)
    try:
        if not is_agent_tool_exposed(name):
            raise ValueError(f"Tool '{name}' is not exposed to the agent")
        exposure = get_tool_exposure(name)
        safe_args = _validated_tool_args(name, safe_args)
        decision = evaluate_tool_call(exposure, actor=actor, raw_args=safe_args)
        policy_meta = tool_governance_meta(exposure, decision)
        result, dispatch_meta, attempts = _call_dispatch_with_governance(
            name,
            safe_args,
            actor=actor,
            provenance_event_id=provenance_event_id,
            timeout_s=exposure.timeout_s,
            retry_policy=exposure.retry_policy,
        )
        validate_tool_output(exposure, result)
        payload, _compact_meta = _compact_tool_output(name, result)
        meta = dict(dispatch_meta)
        meta.update(
            {
                "tool": name,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                "attempts": attempts,
                "status": "ok",
                **policy_meta,
            }
        )
        if provenance_event_id:
            meta["provenance_event_id"] = provenance_event_id
        quality = payload.get("quality") if isinstance(payload, dict) else None
        if isinstance(quality, dict):
            meta["quality_ok"] = bool(quality.get("ok"))
        payload = _attach_meta(payload, meta)
        _finish_tool_provenance(
            provenance_event_id=provenance_event_id,
            name=name,
            safe_args=safe_args,
            actor=actor,
            provenance_context=pv_context,
            critical=critical_tool_call,
            status="succeeded",
            output_value=payload,
            summary={
                "tool": name,
                "duration_ms": meta["duration_ms"],
                "status": "ok",
                "quality_ok": meta.get("quality_ok"),
            },
            metadata={k: v for k, v in meta.items() if k != "tool"},
        )
        return _stable_json_dumps(payload)
    except (ActionValidationError, PydanticValidationError, AgentGovernanceError) as exc:
        logger.warning("Tool %s blocked by governance: %s", name, exc)
        status = "timeout" if isinstance(exc, ToolTimeoutError) else "blocked"
        if exposure is not None and not policy_meta:
            policy_meta = tool_governance_meta(exposure)
        payload = blocked_tool_payload(
            name,
            exc,
            status=status,
            meta={
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                **({"provenance_event_id": provenance_event_id} if provenance_event_id else {}),
                **policy_meta,
            },
        )
        _finish_tool_provenance(
            provenance_event_id=provenance_event_id,
            name=name,
            safe_args=safe_args,
            actor=actor,
            provenance_context=pv_context,
            critical=critical_tool_call,
            status="denied" if status == "blocked" else "timed_out",
            output_value=payload,
            summary={"tool": name, "status": status},
            metadata=policy_meta,
            error=str(exc) or exc.__class__.__name__,
        )
        return _stable_json_dumps(payload)
    except PolicyDenied as exc:
        payload = _attach_meta(
            {"error": "Access denied", "type": "PolicyDenied", "detail": exc.reason},
            {
                "tool": name,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                "status": "denied",
                **({"provenance_event_id": provenance_event_id} if provenance_event_id else {}),
                **policy_meta,
            },
        )
        _finish_tool_provenance(
            provenance_event_id=provenance_event_id,
            name=name,
            safe_args=safe_args,
            actor=actor,
            provenance_context=pv_context,
            critical=critical_tool_call,
            status="denied",
            output_value=payload,
            summary={"tool": name, "status": "denied"},
            error=exc.reason,
        )
        return _stable_json_dumps(payload)
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        payload = _attach_meta(
            {"error": f"Failed to fetch {name}: {exc}"},
            {
                "tool": name,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                "status": "error",
                **({"provenance_event_id": provenance_event_id} if provenance_event_id else {}),
                **policy_meta,
            },
        )
        _finish_tool_provenance(
            provenance_event_id=provenance_event_id,
            name=name,
            safe_args=safe_args,
            actor=actor,
            provenance_context=pv_context,
            critical=critical_tool_call,
            status="failed",
            output_value=payload,
            summary={"tool": name, "status": "error"},
            error=str(exc) or exc.__class__.__name__,
        )
        return _stable_json_dumps(payload)


def _call_dispatch(
    name: str,
    args: dict,
    *,
    actor: Actor,
    provenance_event_id: str | None = None,
) -> tuple[object, dict[str, Any]]:
    params = inspect.signature(_dispatch).parameters.values()
    param_names = {p.name for p in params}
    supports_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    kwargs: dict[str, Any] = {}
    if supports_kwargs or "actor" in param_names:
        kwargs["actor"] = actor
    if provenance_event_id and (supports_kwargs or "provenance_event_id" in param_names):
        kwargs["provenance_event_id"] = provenance_event_id
    if kwargs:
        return _dispatch(name, args, **kwargs)
    return _dispatch(name, args)


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


_AGENT_COMPUTE_WAIT_SECONDS = float(os.environ.get("AGENT_COMPUTE_WAIT_SECONDS", "8"))


def _model_validate(model_cls, payload: dict[str, Any]):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls(**payload)


def _call_with_optional_actor(func, *, actor: Actor, **kwargs):
    params = inspect.signature(func).parameters
    supports_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    supports_actor = supports_var_kwargs or "actor" in params
    call_kwargs = (
        dict(kwargs) if supports_var_kwargs else {key: value for key, value in kwargs.items() if key in params}
    )
    if supports_actor:
        call_kwargs["actor"] = actor
    return func(**call_kwargs)


def _run_registered_job_for_agent(
    job_type: str,
    payload: dict[str, Any],
    *,
    cache_key: str | None,
    poll_path: str,
) -> dict[str, Any]:
    from api.async_job_runner import enqueue_registered_job, poll_registered_job

    row, disposition = enqueue_registered_job(job_type, payload, cache_key=cache_key)
    job_id = str(row.get("job_id") or "")
    deadline = time.monotonic() + max(0.0, _AGENT_COMPUTE_WAIT_SECONDS)
    result = poll_registered_job(job_id)
    while result.get("status") in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(0.25)
        result = poll_registered_job(job_id)
    if result.get("status") in {"queued", "running"}:
        result["poll_url"] = poll_path.format(job_id=job_id)
        result["message"] = "Job is still running. Poll the returned URL or ask again with the job_id."
    result["disposition"] = disposition
    return result


def _is_proposal_tool(name: str) -> bool:
    try:
        return get_tool_exposure(name).access_mode == "proposal"
    except ActionValidationError:
        return False


def _registry_agent_context(actor: Actor, provenance_event_id: str | None = None) -> RegistryActionContext:
    source_id = actor.parent_actor_id or actor.actor_id
    return RegistryActionContext(
        actor_type="agent",
        actor_id=actor.actor_id,
        source_type="agent",
        source_id=source_id,
        provenance_event_id=provenance_event_id,
    )


def _dispatch(
    name: str,
    args: dict,
    actor: Actor | None = None,
    provenance_event_id: str | None = None,
) -> tuple[object, dict[str, Any]]:
    """Route a tool call to the corresponding data function."""
    actor = actor or admin_actor(source="agent_tools")
    force_refresh = bool(args.get("_force_refresh"))
    args = {k: v for k, v in args.items() if not str(k).startswith("_")}

    def fetch(cache, key: str, loader: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
        return _fetch_with_cache(cache, key, loader, force_refresh=force_refresh)

    if _is_proposal_tool(name):
        approval = propose_action_from_tool(name, args, _registry_agent_context(actor, provenance_event_id))
        return {
            "status": "pending_approval_created",
            "approval_id": approval["id"],
            "entity_type": approval["entity_type"],
            "ticker": approval.get("ticker"),
            "message": "Created pending approval. The user must approve it in Workspace before it is applied.",
        }, {"cache": "n/a"}

    if name == "search_agent_capabilities":
        query = str(args.get("query") or "").strip()
        top_k = int(args.get("top_k", 8))
        return search_agent_capabilities(query, top_k=top_k), {"cache": "n/a"}

    if name == "get_liquidity":
        key = "agent_liquidity"

        def _load():
            from macro.liquidity.liquidity import get_snapshot

            data = get_snapshot()
            filtered = {k: v for k, v in data.items() if k not in ("df_weekly", "composite_series")}
            return serialize_value(filtered)

        data, meta = fetch(long_cache, key, _load)
        return data, meta

    if name == "get_market_breadth":
        key = "agent_market_breadth"

        def _load():
            from api.exceptions import SnapshotUnavailableError
            from api.snapshot_keys import SNAPSHOT_MARKET_BREADTH
            from api.snapshot_store import get_snapshot_response, snapshots_required

            snapshot = get_snapshot_response(SNAPSHOT_MARKET_BREADTH)
            if snapshot is not None:
                return serialize_value(snapshot)
            if snapshots_required():
                raise SnapshotUnavailableError(SNAPSHOT_MARKET_BREADTH)

            from equities.market_technicals.market_breadth import get_data

            return serialize_value(get_data())

        data, meta = fetch(long_cache, key, _load)
        return data, meta

    if name == "get_vix_term_structure":
        key = "agent_vix_term_structure:default"

        def _load():
            from equities.market_technicals.vix_term_structure import get_data

            return serialize_value(get_data())

        data, meta = fetch(short_cache, key, _load)
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

        data, meta = fetch(long_cache, key, _load)
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
        include_history = bool(args.get("include_history", False))
        key = f"signal_aggregator:{lookback_weeks}:False:history={include_history}"

        def _load():
            from api.exceptions import SnapshotUnavailableError
            from api.signal_snapshot import get_signal_aggregator_snapshot_or_module_response
            from api.snapshot_keys import SNAPSHOT_SIGNAL_AGGREGATOR
            from api.snapshot_store import snapshots_required

            snapshot = get_signal_aggregator_snapshot_or_module_response(
                lookback_weeks=lookback_weeks,
                include_raw_modules=False,
            )
            if snapshot is not None:
                if not include_history:
                    snapshot = dict(snapshot)
                    snapshot["history"] = {
                        "frequency": "weekly",
                        "lookback_weeks": lookback_weeks,
                        "coverage": {
                            "included_factors": [],
                            "missing_factors": [],
                            "module_status": {"history": "skipped"},
                        },
                        "series": [],
                        "episodes": [],
                    }
                return serialize_value(snapshot)
            if snapshots_required():
                raise SnapshotUnavailableError(SNAPSHOT_SIGNAL_AGGREGATOR)

            data = build_signal_aggregator(
                lookback_weeks=lookback_weeks,
                positioning_instruments=positioning_instruments,
                include_raw_modules=False,
                include_history=include_history,
            )
            return serialize_value(data)

        data, meta = fetch(short_cache, key, _load)
        meta["high_cost"] = True
        return data, meta

    if name == "get_economic_growth":
        key = "economic_growth"

        def _load():
            from macro.economic_growth.economic_growth import get_data

            return serialize_value(get_data())

        data, meta = fetch(short_cache, key, _load)
        return data, meta

    if name == "get_labor_market":
        key = "labor_market"

        def _load():
            from macro.labor_market.labor_market import get_data

            return serialize_value(get_data())

        data, meta = fetch(short_cache, key, _load)
        return data, meta

    if name == "get_housing":
        key = "housing"

        def _load():
            from macro.housing.housing import get_data

            return serialize_value(get_data())

        data, meta = fetch(short_cache, key, _load)
        return data, meta

    if name == "get_sector_metrics":
        key = "sector_metrics"

        def _load():
            from api.exceptions import SnapshotUnavailableError
            from api.snapshot_keys import SNAPSHOT_SECTOR_METRICS
            from api.snapshot_store import get_snapshot_response, snapshots_required

            snapshot = get_snapshot_response(SNAPSHOT_SECTOR_METRICS)
            if snapshot is not None:
                return serialize_value(snapshot)
            if snapshots_required():
                raise SnapshotUnavailableError(SNAPSHOT_SECTOR_METRICS)

            from equities.sector_metrics.sector_metrics import get_data

            return serialize_value(get_data())

        data, meta = fetch(long_cache, key, _load)
        meta["high_cost"] = True
        return data, meta

    if name == "get_portfolio":
        timeframe = args.get("timeframe", "Daily")
        include_hedges = bool(args.get("include_hedges", False))
        key = f"portfolio:{timeframe}:hedges={include_hedges}"

        def _load():
            from portfolio.portfolio_dashboard import get_data
            from portfolio.portfolio_db import get_positions

            raw = get_data(timeframe=timeframe)
            holdings = get_positions(include_hedges=include_hedges)
            return _build_agent_portfolio_payload(raw, holdings, include_hedges=include_hedges)

        data, meta = fetch(short_cache, key, _load)
        return data, meta

    if name == "get_yield_curve":
        lookback_days = int(args.get("lookback_days", 90))
        key = f"yield_curve:{lookback_days}"

        def _load():
            from government_bonds.yield_curve import get_data

            return serialize_value(get_data(lookback_days=lookback_days))

        data, meta = fetch(short_cache, key, _load)
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

        data, meta = fetch(short_cache, key, _load)
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

        data, meta = fetch(short_cache, key, _load)
        quality = data.get("quality") if isinstance(data, dict) else {}
        if isinstance(quality, dict):
            meta["quality_ok"] = bool(quality.get("ok"))
        return data, meta

    if name == "get_central_banks":
        key = "central_banks"

        def _load():
            from macro.central_banks.central_bank import get_data

            return serialize_value(get_data())

        data, meta = fetch(long_cache, key, _load)
        return data, meta

    if name == "get_industry_monitor":
        refresh = bool(args.get("refresh", False))
        key = f"industry_monitor:{refresh}"
        from macro.industry.industry_monitor import get_data

        if refresh:
            return serialize_value(get_data(refresh=True)), {"cache": "bypass"}

        def _load():
            return serialize_value(get_data(refresh=False))

        data, meta = fetch(long_cache, key, _load)
        return data, meta

    if name == "get_breakout":
        key = "breakout"

        def _load():
            from macro.breakout.breakout import get_data

            return serialize_value(get_data())

        data, meta = fetch(short_cache, key, _load)
        return data, meta

    if name == "query_ontology":
        from ontology.service import OntologyQueryService

        raw_filters = args.get("filters")
        if isinstance(raw_filters, dict):
            filters = raw_filters
        elif isinstance(raw_filters, str):
            try:
                parsed = json.loads(raw_filters)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError("query_ontology.filters must be a valid JSON object") from exc
            if not isinstance(parsed, dict):
                raise ValueError("query_ontology.filters must decode to a JSON object")
            filters = parsed
        else:
            filters = {}
        ontology_query = str(args.get("query") or "").strip()
        intent = args.get("intent")
        timeframe = args.get("timeframe", "Daily")
        include_graph = bool(args.get("include_graph", False))
        run_id = args.get("run_id")
        as_of = args.get("as_of")
        tx_as_of = args.get("tx_as_of")
        include_history = bool(args.get("include_history", False))
        refresh_snapshot = bool(args.get("refresh_snapshot", False))
        page = max(1, int(args.get("page", 1) or 1))
        page_size = max(1, min(int(args.get("page_size", 25) or 25), 100))

        cache_token = json.dumps(
            {
                "query": ontology_query,
                "intent": intent,
                "filters": filters,
                "timeframe": timeframe,
                "include_graph": include_graph,
                "run_id": run_id,
                "as_of": as_of,
                "tx_as_of": tx_as_of,
                "include_history": include_history,
                "refresh_snapshot": refresh_snapshot,
                "page": page,
                "page_size": page_size,
                "actor": actor_cache_key(actor),
            },
            sort_keys=True,
            default=str,
        )
        key = f"ontology_query:{cache_token}"

        def _load():
            service = OntologyQueryService()
            result = _call_with_optional_actor(
                service.query,
                actor=actor,
                query=ontology_query or None,
                intent=str(intent) if isinstance(intent, str) else None,
                filters=filters,
                timeframe=str(timeframe) if isinstance(timeframe, str) else "Daily",
                include_graph=include_graph,
                run_id=str(run_id) if isinstance(run_id, str) and run_id.strip() else None,
                as_of=str(as_of) if isinstance(as_of, str) and as_of.strip() else None,
                tx_as_of=str(tx_as_of) if isinstance(tx_as_of, str) and tx_as_of.strip() else None,
                include_history=include_history,
                refresh_snapshot=refresh_snapshot,
                page=page,
                page_size=page_size,
            )
            return serialize_value(result)

        data, meta = fetch(short_cache, key, _load)
        meta["high_cost"] = True
        return data, meta

    if name == "get_thesis":
        ticker_raw = str(args.get("ticker") or "").strip().upper()
        if not ticker_raw:
            return {"error": "Missing required parameter: ticker"}, {"cache": "n/a"}
        key = f"thesis:{ticker_raw}"

        def _load():
            return _fetch_thesis(ticker_raw)

        data, meta = fetch(long_cache, key, _load)
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

        data, meta = fetch(long_cache, key, _load)
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
                diff = _call_with_optional_actor(
                    svc.compare_snapshots,
                    actor=actor,
                    run_id_a=run_id_before,
                    run_id_b=run_id_after,
                )
            else:
                # Auto-select: get latest two runs
                runs = _call_with_optional_actor(svc.list_runs, actor=actor, limit=5)
                if len(runs) < 2:
                    return {"error": f"Need at least 2 ontology snapshots to compare. Only found {len(runs)}."}, {
                        "cache": "n/a"
                    }
                rid_after = run_id_after or str(runs[0].get("run_id", ""))
                rid_before = run_id_before or str(runs[1].get("run_id", ""))
                diff = _call_with_optional_actor(
                    svc.compare_snapshots,
                    actor=actor,
                    run_id_a=rid_before,
                    run_id_b=rid_after,
                )

            return serialize_value(diff), {"cache": "n/a"}
        except PolicyDenied:
            raise
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

        data, meta = fetch(short_cache, key, _load)
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
    # Full app capability registry additions
    # -------------------------------------------------------------------
    if name == "get_workspace":
        from api.routers.workspace import get_workspace

        return serialize_value(get_workspace()), {"cache": "n/a"}

    if name == "get_portfolio_positions":
        from api.routers.portfolio_edit import get_portfolio_positions

        return get_portfolio_positions(include_hedges=bool(args.get("include_hedges", False))), {"cache": "n/a"}

    if name == "get_hedge_positions":
        from api.routers.portfolio_edit import get_hedge_positions_endpoint

        return get_hedge_positions_endpoint(), {"cache": "n/a"}

    if name == "get_portfolio_news":
        digest_id = str(args.get("digest_id") or "").strip()
        if digest_id:
            from api.routers.portfolio_news import get_portfolio_news_digest

            return get_portfolio_news_digest(digest_id), {"cache": "n/a"}
        from api.routers.portfolio_news import list_portfolio_news

        return list_portfolio_news(refresh=False), {"cache": "n/a"}

    if name == "get_research_notes":
        from api.routers.research_notes import list_research_notes

        limit = max(1, min(int(args.get("limit", 20)), 100))
        ticker = str(args.get("ticker") or "").strip().upper() or None
        return list_research_notes(ticker=ticker, limit=limit), {"cache": "n/a"}

    if name == "get_workflow_run":
        from api.routers.workflow_runs import get_workflow_run_detail

        return get_workflow_run_detail(str(args.get("run_id") or "")), {"cache": "n/a"}

    if name == "get_weekly_report":
        from api.routers.weekly_report import get_weekly_report

        return serialize_value(
            get_weekly_report(
                refresh=bool(args.get("refresh", False)),
                cached_only=bool(args.get("cached_only", False)),
            )
        ), {"cache": "n/a"}

    if name == "get_commodities":
        from api.routers.commodities import get_commodities

        return get_commodities(timeframe=str(args.get("timeframe") or "Daily")), {"cache": "n/a"}

    if name == "get_commodities_curve":
        from api.routers.commodities_curve import get_commodities_curve

        commodity = str(args.get("commodity") or "CL").strip().upper()
        lookback_days = int(args.get("lookback_days", 30))
        return get_commodities_curve(commodity=commodity, lookback_days=lookback_days), {"cache": "n/a"}

    if name == "get_commodity_research":
        from api.routers.commodity_research import get_commodity_research

        return get_commodity_research(), {"cache": "n/a"}

    if name == "get_country_dashboard":
        from api.routers.country_dashboard import get_country_dashboard

        return get_country_dashboard(metric=str(args.get("metric") or "Inflation")), {"cache": "n/a"}

    if name == "get_index_dashboard":
        from api.routers.index_dashboard import get_index_dashboard

        return get_index_dashboard(timeframe=str(args.get("timeframe") or "Daily")), {"cache": "n/a"}

    if name == "get_fx_dashboard":
        from api.routers.fx_dashboard import get_fx_dashboard

        return get_fx_dashboard(timeframe=str(args.get("timeframe") or "Daily")), {"cache": "n/a"}

    if name == "get_momentum":
        from api.routers.momentum import get_momentum

        return get_momentum(), {"cache": "n/a"}

    if name == "get_top50_breadth":
        from api.routers.market_technicals import get_top50_breadth

        return get_top50_breadth(), {"cache": "n/a"}

    if name == "get_price_volume_signals":
        from api.routers.market_technicals import get_price_volume_signals

        return get_price_volume_signals(), {"cache": "n/a"}

    if name == "get_financials":
        from api.routers.financials import FinancialsRequest, run_financials

        req = _model_validate(FinancialsRequest, args)
        return run_financials(req), {"cache": "n/a"}

    if name == "get_dcf_historical":
        from api.routers.dcf import get_dcf_historical

        return get_dcf_historical(str(args.get("ticker") or "")), {"cache": "n/a"}

    if name == "run_dcf_valuation":
        from api.routers.dcf import DCFValuationRequest, run_dcf_valuation

        req = _model_validate(DCFValuationRequest, args)
        return run_dcf_valuation(req), {"cache": "n/a"}

    if name == "run_chart":
        from api.routers.chart import ChartRequest, run_chart

        req = _model_validate(ChartRequest, args)
        return run_chart(req), {"cache": "n/a"}

    if name == "run_ratio_chart":
        from api.routers.chart import RatioChartRequest, run_chart_ratio

        req = _model_validate(RatioChartRequest, args)
        return run_chart_ratio(req), {"cache": "n/a"}

    if name == "get_fx_model_pairs":
        from api.routers.fx_model import list_pairs

        return list_pairs(), {"cache": "n/a"}

    if name == "run_fx_model":
        from api.routers.fx_model import FXModelRequest, run_fx_model

        req = _model_validate(FXModelRequest, args)
        return run_fx_model(req), {"cache": "n/a"}

    if name == "run_quality_screen":
        from api.routers.quality import QualityRequest, run_quality_screen

        req = _model_validate(QualityRequest, args)
        return run_quality_screen(req), {"cache": "n/a"}

    if name == "run_short_screen":
        from api.routers.short_screen import ShortScreenRequest
        from api.routers.short_screen import _cache_key as short_screen_cache_key

        req = _model_validate(ShortScreenRequest, args)
        return _run_registered_job_for_agent(
            "short_screen",
            req.model_dump(),
            cache_key=short_screen_cache_key(req),
            poll_path="/api/v1/short-screen/async/{job_id}",
        ), {"cache": "n/a"}

    if name == "run_long_screen":
        from api.routers.long_screen import LongScreenRequest
        from api.routers.long_screen import _cache_key as long_screen_cache_key

        req = _model_validate(LongScreenRequest, args)
        return _run_registered_job_for_agent(
            "long_screen",
            req.model_dump(),
            cache_key=long_screen_cache_key(req),
            poll_path="/api/v1/long-screen/async/{job_id}",
        ), {"cache": "n/a"}

    if name == "run_fundamental_momentum":
        from api.routers.fundamental_momentum import FMRequest
        from api.routers.fundamental_momentum import _cache_key as fundamental_momentum_cache_key

        req = _model_validate(FMRequest, args)
        return _run_registered_job_for_agent(
            "fundamental_momentum",
            req.model_dump(),
            cache_key=fundamental_momentum_cache_key(req),
            poll_path="/api/v1/fundamental-momentum/async/{job_id}",
        ), {"cache": "n/a"}

    if name == "run_portfolio_analyzer":
        from api.routers.analyzer import AnalyzerRequest
        from api.routers.analyzer import _cache_key as analyzer_cache_key

        req = _model_validate(AnalyzerRequest, args)
        return _run_registered_job_for_agent(
            "analyzer",
            req.model_dump(),
            cache_key=analyzer_cache_key(req),
            poll_path="/api/v1/portfolio-analyzer/async/{job_id}",
        ), {"cache": "n/a"}

    if name == "run_portfolio_sizer":
        from api.routers.sizer import SizerRequest
        from api.routers.sizer import _cache_key as sizer_cache_key

        req = _model_validate(SizerRequest, args)
        return _run_registered_job_for_agent(
            "sizer",
            req.model_dump(),
            cache_key=sizer_cache_key(req),
            poll_path="/api/v1/portfolio-sizer/async/{job_id}",
        ), {"cache": "n/a"}

    if name == "get_portfolio_sizer_prefill":
        from api.routers.sizer import get_sizer_prefill

        return get_sizer_prefill(), {"cache": "n/a"}

    if name == "run_hedging_tool":
        from api.routers.hedging import HedgingRequest
        from api.routers.hedging import _cache_key as hedging_cache_key

        req = _model_validate(HedgingRequest, args)
        return _run_registered_job_for_agent(
            "hedging",
            req.model_dump(),
            cache_key=hedging_cache_key(req),
            poll_path="/api/v1/hedging-tool/async/{job_id}",
        ), {"cache": "n/a"}

    if name == "get_hedging_tool_prefill":
        from api.routers.hedging import get_hedging_tool_prefill

        return get_hedging_tool_prefill(), {"cache": "n/a"}

    if name == "get_hedging_portfolio_weights":
        from api.routers.hedging import get_portfolio_weights

        return get_portfolio_weights(book=float(args.get("book", 100_000))), {"cache": "n/a"}

    if name == "run_hedging_recommendation":
        from api.routers.hedging import HedgingRecommendRequest, recommend_hedging_adjustments

        req = _model_validate(HedgingRecommendRequest, args)
        return recommend_hedging_adjustments(req), {"cache": "n/a"}

    raise ValueError(f"Unknown tool: {name}")
