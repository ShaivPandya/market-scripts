"""
Tool registry for the AI agent.

Each tool wraps an existing get_data() / get_snapshot() function from the
analysis modules.  Tool definitions follow the OpenAI function-calling schema
so they can be passed directly to the Responses API ``tools`` parameter.
"""

from __future__ import annotations

import json
import logging
import os

from api.cache import get_cached, long_cache, set_cached, short_cache
from api.serializers import serialize_value

logger = logging.getLogger("api.agent")

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling schema)
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
            "Use this when the user asks about their portfolio, holdings, or performance."
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
        "name": "get_sentiment",
        "description": (
            "Fetch market sentiment indicators. Returns put/call ratios (equity aggregate, "
            "SPY, QQQ, IWM), investor surveys (AAII bull/bear spread, NAAIM exposure index), "
            "and volatility indices (VIX, VXN, VVIX). Use this to assess whether the market "
            "is fearful, complacent, or neutral — and identify potential contrarian signals."
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
            "Use this when users ask portfolio risk exposure questions that require linked context."
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
            },
            "required": [],
        },
    },
]

# Tool name → index lookup
_TOOL_INDEX = {t["name"]: i for i, t in enumerate(TOOL_DEFINITIONS)}


# ---------------------------------------------------------------------------
# Truncation helper
# ---------------------------------------------------------------------------


def _truncate_for_context(data: object, max_chars: int = 30_000) -> str:
    """Convert *data* to a compact JSON string, truncating if necessary."""
    try:
        raw = json.dumps(data, default=str)
    except (TypeError, ValueError):
        raw = str(data)
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + "... [truncated]"


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------


def execute_tool(name: str, arguments: dict) -> str:
    """Run the tool identified by *name* and return a JSON string for the model.

    Errors are caught and returned as ``{"error": "..."}`` so the model can
    inform the user instead of crashing the stream.
    """
    try:
        result = _dispatch(name, arguments)
        return _truncate_for_context(result)
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return json.dumps({"error": f"Failed to fetch {name}: {exc}"})


def _dispatch(name: str, args: dict) -> object:
    """Route a tool call to the corresponding data function."""

    if name == "get_liquidity":
        key = "liquidity"
        cached = get_cached(short_cache, key)
        if cached is not None:
            return cached
        from liquidity import get_snapshot

        data = get_snapshot()
        filtered = {k: v for k, v in data.items() if k not in ("df_weekly", "composite_series")}
        result = serialize_value(filtered)
        set_cached(short_cache, key, result)
        return result

    if name == "get_market_breadth":
        key = "market_breadth"
        cached = get_cached(short_cache, key)
        if cached is not None:
            return cached
        from market_breadth import get_data

        data = get_data()
        result = serialize_value(data)
        set_cached(short_cache, key, result)
        return result

    if name == "get_vix_term_structure":
        key = "vix_term_structure"
        cached = get_cached(short_cache, key)
        if cached is not None:
            return cached
        from vix_term_structure import get_data

        data = get_data()
        result = serialize_value(data)
        set_cached(short_cache, key, result)
        return result

    if name == "get_positioning":
        instruments = args.get("instruments", "SP500,NASDAQ,RUSSELL,US10Y,EUR")
        app_token = os.environ.get("SODA_APP_TOKEN") or None
        key = f"positioning_summary:{instruments}:2015-01-01:None:None:0:2.0"
        cached = get_cached(long_cache, key)
        if cached is not None:
            return cached
        from positioning import DATASETS, DEFAULT_DOMAIN, fetch_multiple_instruments

        instrument_list = [i.strip() for i in instruments.split(",") if i.strip()]
        data = fetch_multiple_instruments(
            domain=DEFAULT_DOMAIN,
            dataset_id=DATASETS.get("tff_futures_only", "tff_futures_only"),
            app_token=app_token,
            instruments=instrument_list,
            start="2015-01-01",
        )
        result = serialize_value(data)
        set_cached(long_cache, key, result)
        return result

    if name == "get_economic_growth":
        key = "economic_growth"
        cached = get_cached(short_cache, key)
        if cached is not None:
            return cached
        from economic_growth import get_data

        data = get_data()
        result = serialize_value(data)
        set_cached(short_cache, key, result)
        return result

    if name == "get_labor_market":
        key = "labor_market"
        cached = get_cached(short_cache, key)
        if cached is not None:
            return cached
        from labor_market import get_data

        data = get_data()
        result = serialize_value(data)
        set_cached(short_cache, key, result)
        return result

    if name == "get_sector_metrics":
        key = "sector_metrics"
        cached = get_cached(long_cache, key)
        if cached is not None:
            return cached
        from sector_metrics import get_data

        data = get_data()
        result = serialize_value(data)
        set_cached(long_cache, key, result)
        return result

    if name == "get_portfolio":
        timeframe = args.get("timeframe", "Daily")
        key = f"portfolio:{timeframe}"
        cached = get_cached(short_cache, key)
        if cached is not None:
            return cached
        from portfolio_dashboard import get_data

        data = get_data(timeframe=timeframe)
        result = serialize_value(data)
        set_cached(short_cache, key, result)
        return result

    if name == "get_yield_curve":
        lookback_days = int(args.get("lookback_days", 90))
        key = f"yield_curve:{lookback_days}"
        cached = get_cached(short_cache, key)
        if cached is not None:
            return cached
        from yield_curve import get_data

        data = get_data(lookback_days=lookback_days)
        result = serialize_value(data)
        set_cached(short_cache, key, result)
        return result

    if name == "get_sentiment":
        key = "agent_sentiment"
        cached = get_cached(short_cache, key)
        if cached is not None:
            return cached
        from sentiment import get_put_call, get_surveys, get_volatility

        combined = {
            "put_call": get_put_call(lookback_days=180),
            "surveys": get_surveys(),
            "volatility": get_volatility(lookback_days=365),
        }
        result = serialize_value(combined)
        set_cached(short_cache, key, result)
        return result

    if name == "get_central_banks":
        key = "central_banks"
        cached = get_cached(long_cache, key)
        if cached is not None:
            return cached
        from central_bank import get_data

        data = get_data()
        result = serialize_value(data)
        set_cached(long_cache, key, result)
        return result

    if name == "get_breakout":
        key = "breakout"
        cached = get_cached(short_cache, key)
        if cached is not None:
            return cached
        from breakout import get_data

        data = get_data()
        result = serialize_value(data)
        set_cached(short_cache, key, result)
        return result

    if name == "query_ontology":
        from ontology.service import OntologyQueryService

        filters = args.get("filters") if isinstance(args.get("filters"), dict) else {}
        query = args.get("query")
        intent = args.get("intent")
        timeframe = args.get("timeframe", "Daily")
        include_graph = bool(args.get("include_graph", False))

        cache_token = json.dumps(
            {
                "query": query,
                "intent": intent,
                "filters": filters,
                "timeframe": timeframe,
                "include_graph": include_graph,
            },
            sort_keys=True,
            default=str,
        )
        key = f"ontology_query:{cache_token}"
        cached = get_cached(short_cache, key)
        if cached is not None:
            return cached

        service = OntologyQueryService()
        result = service.query(
            query=str(query) if isinstance(query, str) else None,
            intent=str(intent) if isinstance(intent, str) else None,
            filters=filters,
            timeframe=str(timeframe) if isinstance(timeframe, str) else "Daily",
            include_graph=include_graph,
        )
        serialized = serialize_value(result)
        set_cached(short_cache, key, serialized)
        return serialized

    return json.dumps({"error": f"Unknown tool: {name}"})
