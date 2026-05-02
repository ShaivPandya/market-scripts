"""
AI Agent chat endpoint with streaming (SSE) and function calling.

Uses the configured LLM provider and the tool definitions from :mod:`api.agent_tools`
to answer cross-cutting investment questions by fetching live data from the
platform's analysis modules.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.agent_tools import AGENT_CAPABILITY_BY_NAME, TOOL_DEFINITIONS, execute_tool, list_agent_capabilities
from api.exceptions import ConfigurationError
from api.workflows import AVAILABLE_WORKFLOWS, execute_workflow
from llm_utils import (
    MODEL_MID,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    api_key_env,
    get_llm_client,
    resolve_model,
    selected_provider,
)

router = APIRouter()
logger = logging.getLogger("api.agent")

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "auto_report" / "prompts"


_prompt_cache: dict[str, tuple[float, str]] = {}


def _load_required_prompt_file(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise ConfigurationError(f"Missing required prompt file: {path}")
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    cached = _prompt_cache.get(filename)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ConfigurationError(f"Prompt file is empty: {path}")
    _prompt_cache[filename] = (mtime, content)
    return content


def _build_screen_context_section(ctx: ScreenContextModel | None) -> str:
    """Format screen context from the frontend into a system prompt section."""
    if ctx is None:
        return ""
    lines = [
        "## Current Screen Context",
        "",
        f"The user is currently viewing: **{ctx.page_name}** (`{ctx.route}`)",
    ]
    if ctx.ticker:
        lines.append(f"Active ticker: **{ctx.ticker}**")
    if ctx.filters:
        filters_str = ", ".join(f"{k}={v}" for k, v in ctx.filters.items())
        lines.append(f"Active filters: {filters_str}")
    if ctx.metrics:
        lines.append("")
        lines.append("### Key metrics currently on screen:")
        for key, value in ctx.metrics.items():
            lines.append(f"- **{key}**: {value}")
    if ctx.summary:
        lines.append("")
        lines.append(f"Screen summary: {ctx.summary}")
    if ctx.corresponding_tools:
        tools_str = ", ".join(f"`{t}`" for t in ctx.corresponding_tools)
        lines.append("")
        lines.append(
            f"**Data overlap notice**: The data on screen was produced by the same source as tools: {tools_str}. "
            "Prefer using the screen context metrics above rather than re-calling these tools, "
            "unless the user explicitly asks for a fresh fetch or you need additional detail "
            "beyond what is summarized here."
        )
    return "\n".join(lines)


def _build_agent_instructions(screen_context: ScreenContextModel | None = None) -> str:
    core_md = _load_required_prompt_file("system.md")
    try:
        agent_md = _load_required_prompt_file("agent_system.md")
    except (ConfigurationError, OSError):
        logger.warning("Failed to load agent_system.md, using core prompt only")
        agent_md = None
    parts = [core_md] + ([agent_md] if agent_md else [])
    base = "\n\n---\n\n".join(parts)

    memory_section = _build_memory_context()
    if memory_section:
        base += "\n\n---\n\n" + memory_section

    screen_section = _build_screen_context_section(screen_context)
    if screen_section:
        base += "\n\n---\n\n" + screen_section

    return base


_memory_cache: tuple[float, str] | None = None
_MEMORY_CACHE_TTL = 60.0


def _build_memory_context() -> str:
    """Build a Recent Research Context section from past session summaries."""
    global _memory_cache
    now = time.monotonic()
    if _memory_cache is not None and now - _memory_cache[0] < _MEMORY_CACHE_TTL:
        return _memory_cache[1]

    try:
        from api.memory_db import get_recent_summaries

        summaries = get_recent_summaries(limit=5)
        if not summaries:
            _memory_cache = (now, "")
            return ""

        lines = [
            "## Recent Research Context\n",
            "**WARNING: The summaries below are historical records of past conversations. "
            "ALL prices, percentages, scores, and numerical data in these summaries are "
            "from the date shown and are STALE. NEVER cite any numerical value from these "
            "summaries as current. Always use tools to fetch current data before making "
            "any market claims. These summaries exist only to remind you what topics and "
            "tickers were previously discussed, NOT to provide current data.**\n",
        ]
        for s in summaries:
            ended = str(s.get("ended_at") or s.get("started_at") or "")[:10]
            summary = str(s.get("summary") or "").strip()
            tickers = s.get("key_tickers")
            if not summary:
                continue
            header = f"[{ended}]"
            if isinstance(tickers, list) and tickers:
                header += f" ({', '.join(tickers[:5])})"
            lines.append(f"{header} {summary}")

        result = "\n".join(lines) if len(lines) > 1 else ""
        _memory_cache = (now, result)
        return result
    except Exception:
        logger.debug("Failed to load memory context", exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ScreenContextModel(BaseModel):
    page_name: str
    route: str
    ticker: str | None = None
    metrics: dict[str, str] | None = None
    filters: dict[str, str] | None = None
    summary: str | None = None
    corresponding_tools: list[str] | None = None


class AgentChatRequest(BaseModel):
    messages: list[ChatMessage]
    screen_context: ScreenContextModel | None = None


class AgentChatRequestV2(BaseModel):
    """V2 request: frontend sends only the new message + session ID."""

    session_id: str | None = None
    message: str
    screen_context: ScreenContextModel | None = None


@router.get("/agent/workflows")
def list_workflows():
    """List available deterministic workflows."""
    return [{"name": name, **info} for name, info in AVAILABLE_WORKFLOWS.items()]


@router.get("/agent/capabilities")
def list_capabilities():
    """List Stan's provider-neutral app capabilities."""
    return {"capabilities": list_agent_capabilities(), "count": len(TOOL_DEFINITIONS)}


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _sse_ping() -> str:
    return _sse("ping", {"ts": round(time.time(), 3)})


def _sse_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        # Prevent GZipMiddleware and upstream proxies from buffering small SSE frames.
        "Content-Encoding": "identity",
    }


MAX_TOOL_CONTINUATION_ROUNDS = 8
MAX_API_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds
SSE_KEEPALIVE_INTERVAL_S = 15.0
LLM_MAX_TOKENS = 8_192
LLM_CHAT_MAX_TOKENS = 2_048
ANTHROPIC_TOOL_DEFINITIONS: list[dict] = [
    {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get("parameters", {"type": "object", "properties": {}, "required": []}),
    }
    for tool in TOOL_DEFINITIONS
    if isinstance(tool.get("name"), str)
]
OPENAI_TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool.get("parameters", {"type": "object", "properties": {}, "required": []}),
    }
    for tool in TOOL_DEFINITIONS
    if isinstance(tool.get("name"), str)
]
_TOOL_DEFINITIONS_BY_PROVIDER = {
    PROVIDER_ANTHROPIC: ANTHROPIC_TOOL_DEFINITIONS,
    PROVIDER_OPENAI: OPENAI_TOOL_DEFINITIONS,
}
_TOOL_DEFINITION_BY_NAME_BY_PROVIDER = {
    provider: {tool["name"]: tool for tool in tools} for provider, tools in _TOOL_DEFINITIONS_BY_PROVIDER.items()
}
_TOOL_NAMES = {tool["name"] for tool in TOOL_DEFINITIONS if isinstance(tool.get("name"), str)}
_HIGH_COST_TOOLS = {"get_sector_metrics", "query_ontology", "get_signal_aggregator"}
_CASUAL_RX = re.compile(
    r"^\s*(hi|hello|hey|yo|thanks|thank you|cool|ok|okay|who are you|what can you do)[\s!.?]*$",
    flags=re.IGNORECASE,
)
_FRESH_RX = re.compile(
    r"\b(refresh|fresh|latest|current|right now|as of now|up[- ]?to[- ]?date|today)\b",
    flags=re.IGNORECASE,
)
_RETRIEVAL_INTENT_RX = re.compile(
    r"\b(past|previous|earlier|history|conversation|research note|notes|thesis|what did i|what have i|wrote|written)\b",
    flags=re.IGNORECASE,
)
_HEDGE_CONTEXT_RX = re.compile(r"\b(hedge|hedges|hedging|beta|net exposure|gross exposure)\b", flags=re.IGNORECASE)
_DATA_SEEKING_RX = re.compile(
    r"\b("
    r"portfolio|holding|position|performance|p&l|pnl|risks?|market|macro|liquidity|breadth|vix|volatility|"
    r"positioning|sentiment|sector|yield|curve|bond|labor|housing|growth|central bank|industry|"
    r"thesis|catalyst|kill condition|dossier|workflow|approval|action item|trigger|search|news|"
    r"commodity|commodities|country|index|indices|fx|currency|financials|dcf|valuation|chart|"
    r"screener|screen|analyzer|sizer|hedging|workspace|research note|weekly report"
    r")\b",
    flags=re.IGNORECASE,
)


def _read_llm_api_key() -> tuple[str, str]:
    try:
        provider = selected_provider()
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    key_env = api_key_env(provider)
    api_key = (os.environ.get(key_env) or "").strip().strip("\"'")
    if not api_key:
        raise ConfigurationError(key_env)

    # A common misconfiguration is placing an OpenAI key into ANTHROPIC_API_KEY.
    if provider == PROVIDER_ANTHROPIC and (
        api_key.startswith("sk-proj-") or (api_key.startswith("sk-") and not api_key.startswith("sk-ant-"))
    ):
        raise ConfigurationError("ANTHROPIC_API_KEY (must be an Anthropic key beginning with sk-ant-)")

    return provider, api_key


def _get_provider_client(provider: str, api_key: str):
    return get_llm_client(provider, api_key=api_key)


def _format_stream_error(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    raw = str(exc)
    lowered = raw.lower()

    if status_code == 401 or "invalid x-api-key" in lowered or "authentication_error" in lowered:
        try:
            provider = selected_provider()
            key_env = api_key_env(provider)
        except Exception:
            provider = "configured provider"
            key_env = "the selected provider API key"
        return f"Agent authentication failed. Set a valid {provider} API key in {key_env} and restart the backend."

    if status_code == 529 or "overloaded" in lowered:
        return "The AI model is temporarily overloaded. Please try again in a few seconds."

    if status_code == 429 or "rate_limit" in lowered:
        return "Rate limit reached. Please wait a moment before sending another message."

    return raw


def _extract_last_user_text(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user" and msg.content.strip():
            return msg.content.strip()
    return ""


def _is_casual(user_text: str) -> bool:
    text = (user_text or "").strip()
    return not text or bool(_CASUAL_RX.match(text))


def _casual_response(user_text: str) -> str:
    text = (user_text or "").strip().lower()
    if "thank" in text or text in {"thanks", "cool", "ok", "okay"}:
        return "Anytime."
    if "who are you" in text or "what can you do" in text:
        return "I'm Stan. I can help with portfolio, market, macro, thesis, and risk questions."
    return "Hey. What are you looking at?"


def _wants_fresh_data(user_text: str) -> bool:
    return bool(_FRESH_RX.search(user_text or ""))


def _should_use_retrieval(user_text: str) -> bool:
    return bool(_RETRIEVAL_INTENT_RX.search(user_text or ""))


def _wants_hedge_context(user_text: str) -> bool:
    return bool(_HEDGE_CONTEXT_RX.search(user_text or ""))


def _is_data_seeking(user_text: str) -> bool:
    return bool(_DATA_SEEKING_RX.search(user_text or ""))


def _select_tool_names(user_text: str) -> list[str]:
    text = (user_text or "").lower()
    selected: list[str] = []

    def add(*names: str) -> None:
        for name in names:
            if name in _TOOL_NAMES and name not in selected:
                selected.append(name)

    if re.search(r"\b(portfolio|holding|holdings|position|positions|p&l|pnl|performance|exposure|risks?)\b", text):
        add("get_portfolio", "query_ontology")
    if re.search(r"\b(update|edit|replace|change)\b.*\b(portfolio|holding|holdings|position|positions)\b", text):
        add("get_portfolio_positions", "propose_portfolio_positions_update")
    if re.search(r"\b(update|edit|replace|change)\b.*\b(hedge|hedges|hedging)\b", text):
        add("get_hedge_positions", "propose_hedge_positions_update")
    if re.search(r"\b(hedge|hedges|hedging|beta|net exposure|gross exposure)\b", text):
        add("get_portfolio", "query_ontology", "get_ontology_diff")
    if re.search(r"\b(market|risk environment|risks?|regime|risk-on|risk-off|macro|cross[- ]?asset)\b", text):
        add(
            "get_signal_aggregator",
            "get_liquidity",
            "get_market_breadth",
            "get_vix_term_structure",
            "get_positioning",
            "get_sentiment",
        )
    if "liquidity" in text:
        add("get_liquidity")
    if "breadth" in text:
        add("get_market_breadth")
    if re.search(r"\b(vix|volatility)\b", text):
        add("get_vix_term_structure", "get_sentiment")
    if "positioning" in text or "crowded" in text:
        add("get_positioning")
    if "sentiment" in text:
        add("get_sentiment")
    if "sector" in text or "rotation" in text:
        add("get_sector_metrics")
    if "yield" in text or "curve" in text or "bond" in text or "rates" in text:
        add("get_yield_curve", "get_bond_dashboard")
    if "labor" in text or "jobs" in text or "claims" in text:
        add("get_labor_market")
    if "housing" in text:
        add("get_housing")
    if "growth" in text:
        add("get_economic_growth")
    if "central bank" in text or "fed" in text or "ecb" in text:
        add("get_central_banks")
    if "industry" in text or "companies saying" in text or "management" in text:
        add("get_industry_monitor")
    if "breakout" in text:
        add("get_breakout")
    if re.search(r"\b(thesis|catalyst|kill condition|dossier|conviction)\b", text):
        add("get_portfolio", "get_dossier", "get_thesis", "get_thesis_evaluations")
    if re.search(r"\b(action item|approval|trigger|workflow)\b", text):
        add("get_action_items", "get_pending_approvals", "get_watch_triggers", "get_workflow_history")
    if re.search(r"\b(search|news|latest|catalyst status|regulatory)\b", text):
        add("search_web", "search_knowledge_base")
    if _should_use_retrieval(user_text):
        add("search_knowledge_base")
    if "commodity" in text or "commodities" in text or "oil curve" in text or "gas curve" in text:
        add("get_commodities", "get_commodities_curve", "get_commodity_research")
    if "country" in text or "countries" in text:
        add("get_country_dashboard")
    if "index" in text or "indices" in text:
        add("get_index_dashboard")
    if re.search(r"\b(fx|currency|currencies|eurusd|usdjpy|gbpusd)\b", text):
        add("get_fx_dashboard", "get_fx_model_pairs", "run_fx_model")
    if "financials" in text or "income statement" in text or "balance sheet" in text:
        add("get_financials")
    if "dcf" in text or "valuation" in text:
        add("get_dcf_historical", "run_dcf_valuation")
    if "chart" in text or "technical analysis" in text:
        add("run_chart", "run_ratio_chart", "get_price_volume_signals")
    if "screener" in text or "screen" in text:
        add("run_quality_screen", "run_short_screen", "run_long_screen", "run_fundamental_momentum")
    if "portfolio analyzer" in text or "portfolio optimizer" in text:
        add("run_portfolio_analyzer")
    if "portfolio sizer" in text or "sizing" in text:
        add("run_portfolio_sizer", "get_portfolio_sizer_prefill")
    if "news digest" in text or "uploaded news" in text or "portfolio news" in text:
        add("get_portfolio_news")
    if "research note" in text:
        add("get_research_notes", "propose_research_note")
    if "workspace" in text:
        add("get_workspace")

    stop = {"AND", "THE", "FOR", "MY", "ALL", "HOW", "CAN", "ARE", "HAS", "DO", "WHAT", "THIS", "THAT"}
    ticker_candidates = [m for m in _TICKER_RX.findall(user_text or "") if m not in stop and len(m) >= 2]
    if ticker_candidates:
        add(
            "get_portfolio",
            "get_dossier",
            "get_thesis",
            "get_financials",
            "get_dcf_historical",
            "run_chart",
            "search_web",
        )

    # Registry lexical pass. This catches newly registered app capabilities
    # without adding a new regex branch for every route.
    registry_matches: list[tuple[int, str]] = []
    for cap in AGENT_CAPABILITY_BY_NAME.values():
        if not cap.selectable or cap.name == "search_agent_capabilities":
            continue
        terms = [cap.name.replace("_", " "), *cap.aliases]
        score = 0
        for term in terms:
            term_l = term.lower().strip()
            if not term_l:
                continue
            if term_l in text:
                score = max(score, 10 if len(term_l) > 4 else 4)
        if score:
            registry_matches.append((score, cap.name))
    for _score, name in sorted(registry_matches, key=lambda row: (-row[0], row[1]))[:12]:
        add(name)

    if not selected and _is_data_seeking(user_text):
        add("get_signal_aggregator", "get_liquidity", "get_market_breadth", "get_vix_term_structure", "get_portfolio")

    add("search_agent_capabilities")
    return selected


def _tool_definitions_from_names(provider: str, names: list[str]) -> list[dict]:
    definitions = _TOOL_DEFINITION_BY_NAME_BY_PROVIDER[provider]
    return [definitions[name] for name in names if name in definitions]


def _select_tool_definitions(user_text: str, provider: str) -> list[dict]:
    return _tool_definitions_from_names(provider, _select_tool_names(user_text))


def _execution_args(name: str, args: dict, *, force_refresh: bool, user_text: str) -> dict:
    out = dict(args) if isinstance(args, dict) else {}
    if force_refresh:
        out["_force_refresh"] = True
    if name == "get_portfolio" and _wants_hedge_context(user_text):
        out["include_hedges"] = True
    return out


# ---------------------------------------------------------------------------
# Workflow detection
# ---------------------------------------------------------------------------

_WORKFLOW_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bmorning\s+brief\b", re.IGNORECASE), "morning_brief"),
    (re.compile(r"\bdaily\s+brief\b", re.IGNORECASE), "morning_brief"),
    (re.compile(r"\breview\s+(?:my\s+|the\s+)?(?:thesis|investment\s+thesis)\b", re.IGNORECASE), "thesis_review"),
    (re.compile(r"\bthesis\s+review\b", re.IGNORECASE), "thesis_review"),
    (re.compile(r"\bpre[- ]?earnings?\s+(?:prep|brief|analysis)\b", re.IGNORECASE), "pre_earnings"),
    (re.compile(r"\bearnings?\s+prep\b", re.IGNORECASE), "pre_earnings"),
    (re.compile(r"\bpost[- ]?earnings?\s+(?:review|debrief)\b", re.IGNORECASE), "post_earnings_review"),
    (re.compile(r"\bearnings?\s+(?:review|debrief)\b", re.IGNORECASE), "post_earnings_review"),
    (re.compile(r"\bweekly\s+(?:portfolio\s+)?review\b", re.IGNORECASE), "weekly_portfolio_review"),
    (re.compile(r"\bportfolio\s+review\b", re.IGNORECASE), "weekly_portfolio_review"),
    (re.compile(r"\b(?:thesis\s+)?invalidation\s+check\b", re.IGNORECASE), "thesis_invalidation_check"),
    (re.compile(r"\bkill\s+condition\s+check\b", re.IGNORECASE), "thesis_invalidation_check"),
]

_TICKER_RX = re.compile(r"\b([A-Z]{1,5})\b")


def _detect_workflow(user_text: str) -> tuple[str | None, str | None]:
    """Detect if a user message triggers a workflow.

    Returns (workflow_name, ticker) or (None, None).
    """
    text = (user_text or "").strip()
    if not text:
        return None, None

    # Check for explicit workflow trigger (from frontend buttons)
    if text.startswith("/workflow:"):
        parts = text.split(":", 2)
        wf_name = parts[1].strip() if len(parts) > 1 else ""
        ticker = parts[2].strip().upper() if len(parts) > 2 and parts[2].strip() else None
        if wf_name in AVAILABLE_WORKFLOWS:
            return wf_name, ticker
        return None, None

    # Check natural language patterns
    for pattern, wf_name in _WORKFLOW_PATTERNS:
        if pattern.search(text):
            ticker = None
            wf_def = AVAILABLE_WORKFLOWS[wf_name]
            if wf_def["requires_ticker"]:
                # Try to extract a ticker from the message
                # Filter out common English words that match ticker pattern
                stop = {"AND", "THE", "FOR", "MY", "ALL", "HOW", "CAN", "ARE", "HAS", "DO"}
                candidates = [m for m in _TICKER_RX.findall(text) if m not in stop and len(m) >= 2]
                ticker = candidates[0] if candidates else None
            return wf_name, ticker
    return None, None


def _tool_call_signature(name: str, args: dict) -> str:
    try:
        args_key = json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))
    except Exception:
        args_key = "{}"
    return f"{name}::{args_key}"


def _dedupe_tool_calls(calls: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for c in calls:
        try:
            args_key = json.dumps(c.get("args", {}), sort_keys=True, default=str, separators=(",", ":"))
        except Exception:
            args_key = "{}"
        key = f"{c['name']}::{args_key}"
        entry = grouped.get(key)
        if entry is None:
            grouped[key] = {
                "name": c["name"],
                "args": c.get("args", {}) if isinstance(c.get("args"), dict) else {},
                "call_ids": [c["call_id"]],
            }
            continue
        entry["call_ids"].append(c["call_id"])
    return list(grouped.values())


def _execute_tools_parallel(
    calls: list[dict],
) -> list[tuple[dict, str, float]]:
    """Execute deduplicated tool calls in parallel and measure runtime."""
    if len(calls) == 1:
        c = calls[0]
        started = time.perf_counter()
        result = execute_tool(c["name"], c["args"])
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return [(c, result, elapsed_ms)]

    with ThreadPoolExecutor(max_workers=min(len(calls), 8)) as pool:
        futures = []
        for c in calls:
            started = time.perf_counter()
            fut = pool.submit(execute_tool, c["name"], c["args"])
            futures.append((c, fut, started))
        out: list[tuple[dict, str, float]] = []
        for c, fut, started in futures:
            result = fut.result()
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            out.append((c, result, elapsed_ms))
        return out


def _execute_tools_parallel_keepalive(
    calls: list[dict],
):
    """Execute tool calls while yielding None periodically as an SSE keepalive signal."""
    if not calls:
        return

    with ThreadPoolExecutor(max_workers=min(len(calls), 8)) as pool:
        future_meta = {}
        for c in calls:
            started = time.perf_counter()
            fut = pool.submit(execute_tool, c["name"], c["args"])
            future_meta[fut] = (c, started)

        pending = set(future_meta)
        while pending:
            done, pending = wait(
                pending,
                timeout=SSE_KEEPALIVE_INTERVAL_S,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                yield None
                continue

            for fut in done:
                c, started = future_meta[fut]
                result = fut.result()
                elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                yield (c, result, elapsed_ms)


def _tool_error_message(result_str: str) -> str | None:
    try:
        payload = json.loads(result_str)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    err = payload.get("error")
    if isinstance(err, str) and err.strip():
        return err.strip()
    return None


def _tool_meta(result_str: str) -> dict:
    try:
        payload = json.loads(result_str)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    meta = payload.get("_meta")
    return meta if isinstance(meta, dict) else {}


def _capability_names_from_search_result(result_str: str) -> list[str]:
    try:
        payload = json.loads(result_str)
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    matches = payload.get("matches")
    if not isinstance(matches, list):
        return []
    names: list[str] = []
    for row in matches:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if isinstance(name, str) and name in _TOOL_NAMES and name not in names:
            names.append(name)
    return names


def _is_retryable_error(exc: Exception) -> bool:
    """Check if an API error is transient and worth retrying."""
    status_code = getattr(exc, "status_code", None)
    if status_code in (429, 529):
        return True
    lowered = str(exc).lower()
    return "overloaded" in lowered or "rate_limit" in lowered


def _serialize_content_blocks(blocks: list[object]) -> list[dict]:
    serialized: list[dict] = []
    for block in blocks:
        if isinstance(block, dict):
            serialized.append(block)
            continue

        model_dump = getattr(block, "model_dump", None)
        if callable(model_dump):
            serialized.append(model_dump(exclude_none=True))
            continue

        to_dict = getattr(block, "to_dict", None)
        if callable(to_dict):
            serialized.append(to_dict())
    return serialized


def _serialize_output_items(response: object) -> list[dict]:
    return _serialize_content_blocks(list(_obj_list(response, "output")))


def _extract_tool_calls(content_blocks: list[dict]) -> list[dict]:
    calls: list[dict] = []
    for block in content_blocks:
        if block.get("type") != "tool_use":
            continue
        name = block.get("name")
        call_id = block.get("id")
        args = block.get("input", {})
        if not isinstance(name, str) or not isinstance(call_id, str):
            continue
        if not isinstance(args, dict):
            args = {}
        calls.append({"name": name, "call_id": call_id, "args": args})
    return calls


def _extract_openai_tool_calls(output_items: list[dict]) -> list[dict]:
    calls: list[dict] = []
    for item in output_items:
        if item.get("type") != "function_call":
            continue
        name = item.get("name")
        call_id = item.get("call_id") or item.get("id")
        raw_args = item.get("arguments", {})
        args: dict
        if isinstance(raw_args, str) and raw_args.strip():
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed_args = {}
            args = parsed_args if isinstance(parsed_args, dict) else {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        if isinstance(name, str) and isinstance(call_id, str):
            calls.append({"name": name, "call_id": call_id, "args": args})
    return calls


def _obj_list(value: object, key: str) -> list[object]:
    if isinstance(value, dict):
        out = value.get(key, [])
    else:
        out = getattr(value, key, [])
    return list(out or [])


def _obj_value(value: object, key: str, default: object = None) -> object:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _tool_definitions_for_provider(provider: str) -> list[dict]:
    return _TOOL_DEFINITIONS_BY_PROVIDER[provider]


def _model_stream_kwargs(
    *,
    provider: str,
    instructions: str,
    conversation: list[dict],
    max_tokens: int,
    tool_defs: list[dict] | None = None,
    force_tool_use: bool = False,
) -> dict[str, object]:
    if provider == PROVIDER_ANTHROPIC:
        kwargs: dict[str, object] = {
            "model": resolve_model(MODEL_MID, PROVIDER_ANTHROPIC),
            "max_tokens": max_tokens,
            "system": instructions,
            "messages": conversation,
        }
        if tool_defs:
            kwargs["tools"] = tool_defs
        if force_tool_use and tool_defs:
            kwargs["tool_choice"] = {"type": "any"}
        return kwargs

    kwargs = {
        "model": resolve_model(MODEL_MID, PROVIDER_OPENAI),
        "max_output_tokens": max_tokens,
        "instructions": instructions,
        "input": conversation,
    }
    if tool_defs:
        kwargs["tools"] = tool_defs
    if force_tool_use and tool_defs:
        kwargs["tool_choice"] = "required"
    return kwargs


def _openai_text_type(role: object) -> str:
    return "output_text" if role == "assistant" else "input_text"


def _initial_conversation(provider: str, messages: list[ChatMessage]) -> list[dict]:
    if provider == PROVIDER_ANTHROPIC:
        return [{"role": m.role, "content": m.content} for m in messages]
    return [{"role": m.role, "content": [{"type": _openai_text_type(m.role), "text": m.content}]} for m in messages]


def _openai_conversation_from_context(conversation: list[dict[str, object]]) -> list[dict]:
    out: list[dict] = []
    for msg in conversation:
        role = msg.get("role")
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append({"role": role, "content": [{"type": _openai_text_type(role), "text": content}]})
        else:
            out.append({"role": role, "content": content})
    return out


def _openai_user_prompt(prompt: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]


def _stream_llm_response(
    client: Any, provider: str, stream_kwargs: dict[str, object], text_parts: list[str] | None = None
):
    if provider == PROVIDER_ANTHROPIC:
        with client.messages.stream(**stream_kwargs) as stream:
            for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    if text_parts is not None:
                        text_parts.append(event.delta.text)
                    yield _sse("delta", {"text": event.delta.text})
                elif event.type == "content_block_start" and event.content_block.type == "tool_use":
                    yield _sse(
                        "tool_call",
                        {
                            "name": event.content_block.name,
                            "id": event.content_block.id,
                        },
                    )
            return stream.get_final_message()

    emitted_call_ids: set[str] = set()
    with client.responses.stream(**stream_kwargs) as stream:
        for event in stream:
            event_type = _obj_value(event, "type")
            if event_type == "response.output_text.delta":
                delta = _obj_value(event, "delta", "")
                if isinstance(delta, str) and delta:
                    if text_parts is not None:
                        text_parts.append(delta)
                    yield _sse("delta", {"text": delta})
            elif event_type == "response.output_item.added":
                item = _obj_value(event, "item", {})
                if _obj_value(item, "type") == "function_call":
                    call_id = _obj_value(item, "call_id") or _obj_value(item, "id")
                    name = _obj_value(item, "name")
                    if isinstance(call_id, str) and isinstance(name, str) and call_id not in emitted_call_ids:
                        emitted_call_ids.add(call_id)
                        yield _sse("tool_call", {"name": name, "id": call_id})
        get_final = getattr(stream, "get_final_response", None)
        if callable(get_final):
            return get_final()
        get_final = getattr(stream, "get_final_message", None)
        if callable(get_final):
            return get_final()
        return None


def _usage_dict(message: object) -> dict:
    usage = _obj_value(message, "usage")
    if not usage:
        return {}
    input_tokens = _obj_value(usage, "input_tokens", _obj_value(usage, "prompt_tokens", None))
    output_tokens = _obj_value(usage, "output_tokens", _obj_value(usage, "completion_tokens", None))
    out: dict[str, int] = {}
    if isinstance(input_tokens, int):
        out["input_tokens"] = input_tokens
    if isinstance(output_tokens, int):
        out["output_tokens"] = output_tokens
    return out


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/agent/chat")
def agent_chat(req: AgentChatRequest):
    provider, api_key = _read_llm_api_key()
    instructions = _build_agent_instructions(screen_context=req.screen_context)
    latest_user_text = _extract_last_user_text(req.messages)
    casual = _is_casual(latest_user_text)
    workflow_name, workflow_ticker = _detect_workflow(latest_user_text)
    tool_defs = _tool_definitions_for_provider(provider)
    logger.info(
        "agent_tool_policy provider=%s casual=%s workflow=%s ticker=%s tools=%d",
        provider,
        casual,
        workflow_name,
        workflow_ticker,
        len(tool_defs),
    )

    def generate():  # noqa: C901 — complex but linear control flow
        yield _sse_ping()
        client = _get_provider_client(provider, api_key)

        # --- Workflow path: deterministic tool execution → single synthesis ---
        if workflow_name:
            try:
                wf_def = AVAILABLE_WORKFLOWS.get(workflow_name, {})
                if wf_def.get("requires_ticker") and not workflow_ticker:
                    yield _sse(
                        "delta",
                        {
                            "text": f"I need a ticker to run the **{wf_def.get('label', workflow_name)}** workflow. Which position would you like me to review?"
                        },
                    )
                    yield _sse("done", {"usage": {}})
                    return

                # Emit tool calls as they execute
                run_id, synthesis_prompt, sections = execute_workflow(workflow_name, workflow_ticker)
                for section in sections:
                    yield _sse("tool_call", {"name": section["tool"], "id": section["tool"]})
                    yield _sse("tool_result", {"name": section["tool"], "id": section["tool"], "status": "ok"})

                # Single synthesis call — no tools, just model reasoning over the data
                synthesis_chunks: list[str] = []
                for attempt in range(MAX_API_RETRIES):
                    try:
                        synthesis_conversation = (
                            [{"role": "user", "content": synthesis_prompt}]
                            if provider == PROVIDER_ANTHROPIC
                            else _openai_user_prompt(synthesis_prompt)
                        )
                        stream_kwargs = _model_stream_kwargs(
                            provider=provider,
                            instructions=instructions,
                            conversation=synthesis_conversation,
                            max_tokens=LLM_MAX_TOKENS,
                        )
                        final_message = yield from _stream_llm_response(
                            client,
                            provider,
                            stream_kwargs,
                            synthesis_chunks,
                        )
                        break
                    except Exception as retry_exc:
                        if attempt < MAX_API_RETRIES - 1 and _is_retryable_error(retry_exc):
                            delay = RETRY_BASE_DELAY * (2**attempt)
                            logger.warning(
                                "Retryable API error (attempt %d/%d), retrying in %.1fs: %s",
                                attempt + 1,
                                MAX_API_RETRIES,
                                delay,
                                retry_exc,
                            )
                            time.sleep(delay)
                            continue
                        raise

                # Persist the completed workflow run
                synthesis_text = "".join(synthesis_chunks)
                try:
                    from api.workflow_artifacts import extract_artifacts, persist_artifacts
                    from portfolio.core_db import complete_workflow_run

                    artifacts = extract_artifacts(synthesis_text, workflow_name)
                    complete_workflow_run(run_id, synthesis_text, artifacts, sections)
                    if artifacts:
                        persist_artifacts(run_id, workflow_ticker, artifacts)
                except Exception:
                    logger.debug("Failed to persist workflow run %s", run_id, exc_info=True)

                usage = _usage_dict(final_message)
                yield _sse("done", {"usage": usage, "workflow_run_id": run_id})
                return

            except Exception as exc:
                logger.exception("Workflow %s failed", workflow_name)
                try:
                    from portfolio.core_db import fail_workflow_run

                    fail_workflow_run(run_id, str(exc))
                except Exception:
                    pass
                yield _sse("error", {"message": f"Workflow failed: {exc}"})
                yield _sse("done", {"usage": {}})
                return

        # --- Normal tool-calling path ---
        conversation = _initial_conversation(provider, req.messages)
        continuation_round = 0
        # Force tool use on the first round for non-casual queries so
        # answers are always grounded in live data.  When rich screen
        # context is present (metrics or summary from the frontend),
        # the agent already has data to reason from — let it decide
        # whether additional tool calls are needed.
        has_rich_screen_data = req.screen_context is not None and (
            req.screen_context.metrics or req.screen_context.summary
        )
        force_tool_use = not casual and not has_rich_screen_data
        tool_result_cache: dict[str, str] = {}

        try:
            while True:
                if continuation_round >= MAX_TOOL_CONTINUATION_ROUNDS:
                    yield _sse(
                        "error",
                        {"message": (f"Tool-call loop limit reached ({MAX_TOOL_CONTINUATION_ROUNDS} rounds).")},
                    )
                    yield _sse("done", {"usage": {}})
                    return

                stream_kwargs = _model_stream_kwargs(
                    provider=provider,
                    instructions=instructions,
                    conversation=conversation,
                    max_tokens=LLM_MAX_TOKENS,
                    tool_defs=tool_defs,
                    force_tool_use=force_tool_use,
                )

                for attempt in range(MAX_API_RETRIES):
                    try:
                        final_message = yield from _stream_llm_response(client, provider, stream_kwargs)
                        break
                    except Exception as retry_exc:
                        if attempt < MAX_API_RETRIES - 1 and _is_retryable_error(retry_exc):
                            delay = RETRY_BASE_DELAY * (2**attempt)
                            logger.warning(
                                "Retryable API error (attempt %d/%d), retrying in %.1fs: %s",
                                attempt + 1,
                                MAX_API_RETRIES,
                                delay,
                                retry_exc,
                            )
                            time.sleep(delay)
                            continue
                        raise

                if provider == PROVIDER_ANTHROPIC:
                    assistant_content = _serialize_content_blocks(list(final_message.content))
                    deferred_calls = _extract_tool_calls(assistant_content)
                else:
                    assistant_content = _serialize_output_items(final_message)
                    deferred_calls = _extract_openai_tool_calls(assistant_content)

                if deferred_calls:
                    tool_counts = Counter(c["name"] for c in deferred_calls)
                    repeated_high_cost = [
                        f"{name}x{count}"
                        for name, count in tool_counts.items()
                        if name in _HIGH_COST_TOOLS and count > 1
                    ]
                    if repeated_high_cost:
                        logger.warning("agent tool round has repeated high-cost calls: %s", repeated_high_cost)

                    unique_calls = _dedupe_tool_calls(deferred_calls)
                    logger.info(
                        "agent_tool_round requested=%s unique=%s",
                        [c["name"] for c in deferred_calls],
                        [c["name"] for c in unique_calls],
                    )

                    tool_results: list[dict] = []
                    turn_cache_hits: set[str] = set()
                    pending_calls: list[dict] = []
                    executed_by_signature: dict[str, tuple[str, float]] = {}

                    for call_info in unique_calls:
                        signature = _tool_call_signature(call_info["name"], call_info["args"])
                        if signature in tool_result_cache:
                            turn_cache_hits.add(signature)
                            continue
                        pending_calls.append(call_info)

                    if pending_calls:
                        for tool_item in _execute_tools_parallel_keepalive(pending_calls):
                            if tool_item is None:
                                yield _sse_ping()
                                continue
                            call_info, result_str, elapsed_ms = tool_item
                            signature = _tool_call_signature(call_info["name"], call_info["args"])
                            tool_result_cache[signature] = result_str
                            executed_by_signature[signature] = (result_str, elapsed_ms)

                    for call_info in unique_calls:
                        signature = _tool_call_signature(call_info["name"], call_info["args"])
                        if signature in turn_cache_hits:
                            result_str = tool_result_cache[signature]
                            elapsed_ms = 0.0
                        else:
                            result_str, elapsed_ms = executed_by_signature[signature]

                        err_msg = _tool_error_message(result_str)
                        meta = _tool_meta(result_str)
                        cache_status = "turn_hit" if signature in turn_cache_hits else str(meta.get("cache", "unknown"))
                        logger.info(
                            "agent_tool_exec name=%s duration_ms=%.1f cache=%s status=%s quality_ok=%s",
                            call_info["name"],
                            elapsed_ms,
                            cache_status,
                            "error" if err_msg else "ok",
                            str(meta.get("quality_ok", "n/a")),
                        )

                        for call_id in call_info.get("call_ids", []):
                            payload = {
                                "name": call_info["name"],
                                "id": call_id,
                                "status": "error" if err_msg else "ok",
                            }
                            if err_msg:
                                payload["message"] = err_msg
                            yield _sse("tool_result", payload)

                            if provider == PROVIDER_ANTHROPIC:
                                result_block: dict[str, object] = {
                                    "type": "tool_result",
                                    "tool_use_id": call_id,
                                    "content": result_str,
                                }
                                if err_msg:
                                    result_block["is_error"] = True
                            else:
                                result_block = {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": result_str,
                                }
                            tool_results.append(result_block)

                    if provider == PROVIDER_ANTHROPIC:
                        conversation.append({"role": "assistant", "content": assistant_content})
                        conversation.append({"role": "user", "content": tool_results})
                    else:
                        conversation.extend(assistant_content)
                        conversation.extend(tool_results)
                    # After the first tool round, let the model decide whether it
                    # needs more data (tool_choice: auto).
                    force_tool_use = False
                    continuation_round += 1
                    continue

                if provider == PROVIDER_ANTHROPIC and final_message.stop_reason == "pause_turn":
                    conversation.append({"role": "assistant", "content": assistant_content})
                    conversation.append({"role": "user", "content": [{"type": "text", "text": "Continue."}]})
                    force_tool_use = False
                    continuation_round += 1
                    continue

                usage = _usage_dict(final_message)
                yield _sse("done", {"usage": usage})
                return

        except Exception as exc:
            logger.exception("Agent stream error")
            yield _sse("error", {"message": _format_stream_error(exc)})
            yield _sse("done", {"usage": {}})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


# ---------------------------------------------------------------------------
# V2 Endpoint — server-managed rolling memory
# ---------------------------------------------------------------------------


@router.post("/agent/chat/v2")
def agent_chat_v2(req: AgentChatRequestV2):
    """Chat endpoint with server-managed conversation memory.

    The frontend sends only the new message + session_id.  The server
    assembles optimal context from a rolling summary, verbatim window,
    and retrieval hits.
    """
    casual = _is_casual(req.message)
    try:
        provider = selected_provider()
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    workflow_name, workflow_ticker = _detect_workflow(req.message)
    active_tool_names = [] if casual else _select_tool_names(req.message)
    tool_defs = _tool_definitions_from_names(provider, active_tool_names)
    force_refresh = _wants_fresh_data(req.message)
    enable_retrieval = _should_use_retrieval(req.message)
    logger.info(
        "agent_v2 provider=%s casual=%s workflow=%s ticker=%s tools=%d refresh=%s retrieval=%s session=%s",
        provider,
        casual,
        workflow_name,
        workflow_ticker,
        len(tool_defs),
        force_refresh,
        enable_retrieval,
        req.session_id,
    )

    def generate():  # noqa: C901
        nonlocal tool_defs
        yield _sse_ping()
        from api.memory_manager import build_conversation_context, finalize_turn_async

        if casual and not workflow_name:
            from api import memory_db

            session = memory_db.get_or_create_session(req.session_id)
            session_id = str(session["session_id"])
            text = _casual_response(req.message)
            yield _sse("delta", {"text": text})
            user_msg = {"role": "user", "content": req.message, "timestamp": time.time()}
            assistant_msg = {"role": "assistant", "content": text, "timestamp": time.time()}
            finalize_turn_async(session_id, user_msg, assistant_msg)
            yield _sse("done", {"usage": {}, "session_id": session_id})
            return

        provider, api_key = _read_llm_api_key()
        instructions = _build_agent_instructions(screen_context=req.screen_context)
        client = _get_provider_client(provider, api_key)
        raw_conversation, session_id = build_conversation_context(
            req.session_id,
            req.message,
            enable_retrieval=enable_retrieval,
        )
        conversation = (
            raw_conversation if provider == PROVIDER_ANTHROPIC else _openai_conversation_from_context(raw_conversation)
        )

        # --- Workflow path ---
        if workflow_name:
            try:
                wf_def = AVAILABLE_WORKFLOWS.get(workflow_name, {})
                if wf_def.get("requires_ticker") and not workflow_ticker:
                    yield _sse(
                        "delta",
                        {
                            "text": f"I need a ticker to run the **{wf_def.get('label', workflow_name)}** workflow. Which position would you like me to review?"
                        },
                    )
                    yield _sse("done", {"usage": {}, "session_id": session_id})
                    return

                run_id, synthesis_prompt, sections = execute_workflow(workflow_name, workflow_ticker)
                for section in sections:
                    yield _sse("tool_call", {"name": section["tool"], "id": section["tool"]})
                    yield _sse("tool_result", {"name": section["tool"], "id": section["tool"], "status": "ok"})

                synthesis_chunks: list[str] = []
                for attempt in range(MAX_API_RETRIES):
                    try:
                        synthesis_conversation = (
                            [{"role": "user", "content": synthesis_prompt}]
                            if provider == PROVIDER_ANTHROPIC
                            else _openai_user_prompt(synthesis_prompt)
                        )
                        stream_kwargs = _model_stream_kwargs(
                            provider=provider,
                            instructions=instructions,
                            conversation=synthesis_conversation,
                            max_tokens=LLM_MAX_TOKENS,
                        )
                        final_message = yield from _stream_llm_response(
                            client,
                            provider,
                            stream_kwargs,
                            synthesis_chunks,
                        )
                        break
                    except Exception as retry_exc:
                        if attempt < MAX_API_RETRIES - 1 and _is_retryable_error(retry_exc):
                            time.sleep(RETRY_BASE_DELAY * (2**attempt))
                            continue
                        raise

                synthesis_text = "".join(synthesis_chunks)
                try:
                    from api.workflow_artifacts import extract_artifacts, persist_artifacts
                    from portfolio.core_db import complete_workflow_run

                    artifacts = extract_artifacts(synthesis_text, workflow_name)
                    complete_workflow_run(run_id, synthesis_text, artifacts, sections)
                    if artifacts:
                        persist_artifacts(run_id, workflow_ticker, artifacts)
                except Exception:
                    logger.debug("Failed to persist workflow run %s", run_id, exc_info=True)

                usage = _usage_dict(final_message)
                # Finalize turn before last yield
                user_msg = {"role": "user", "content": req.message, "timestamp": time.time()}
                assistant_msg = {"role": "assistant", "content": synthesis_text, "timestamp": time.time()}
                finalize_turn_async(session_id, user_msg, assistant_msg)
                yield _sse("done", {"usage": usage, "session_id": session_id, "workflow_run_id": run_id})
                return

            except Exception as exc:
                logger.exception("Workflow %s failed (v2)", workflow_name)
                try:
                    from portfolio.core_db import fail_workflow_run

                    fail_workflow_run(run_id, str(exc))
                except Exception:
                    pass
                yield _sse("error", {"message": f"Workflow failed: {exc}"})
                yield _sse("done", {"usage": {}, "session_id": session_id})
                return

        # --- Normal tool-calling path ---
        has_rich_screen_data = req.screen_context is not None and (
            req.screen_context.metrics or req.screen_context.summary
        )
        force_tool_use = bool(tool_defs) and _is_data_seeking(req.message) and not has_rich_screen_data
        tool_result_cache: dict[str, str] = {}
        continuation_round = 0
        text_parts: list[str] = []

        try:
            while True:
                if continuation_round >= MAX_TOOL_CONTINUATION_ROUNDS:
                    yield _sse(
                        "error",
                        {"message": f"Tool-call loop limit reached ({MAX_TOOL_CONTINUATION_ROUNDS} rounds)."},
                    )
                    yield _sse("done", {"usage": {}, "session_id": session_id})
                    return

                stream_kwargs = _model_stream_kwargs(
                    provider=provider,
                    instructions=instructions,
                    conversation=conversation,
                    max_tokens=LLM_CHAT_MAX_TOKENS,
                    tool_defs=tool_defs,
                    force_tool_use=force_tool_use,
                )

                for attempt in range(MAX_API_RETRIES):
                    try:
                        final_message = yield from _stream_llm_response(
                            client,
                            provider,
                            stream_kwargs,
                            text_parts,
                        )
                        break
                    except Exception as retry_exc:
                        if attempt < MAX_API_RETRIES - 1 and _is_retryable_error(retry_exc):
                            time.sleep(RETRY_BASE_DELAY * (2**attempt))
                            continue
                        raise

                if provider == PROVIDER_ANTHROPIC:
                    assistant_content = _serialize_content_blocks(list(final_message.content))
                    deferred_calls = _extract_tool_calls(assistant_content)
                else:
                    assistant_content = _serialize_output_items(final_message)
                    deferred_calls = _extract_openai_tool_calls(assistant_content)

                if deferred_calls:
                    tool_counts = Counter(c["name"] for c in deferred_calls)
                    repeated_high_cost = [
                        f"{name}x{count}"
                        for name, count in tool_counts.items()
                        if name in _HIGH_COST_TOOLS and count > 1
                    ]
                    if repeated_high_cost:
                        logger.warning("agent tool round has repeated high-cost calls: %s", repeated_high_cost)

                    unique_calls = _dedupe_tool_calls(deferred_calls)
                    for call_info in unique_calls:
                        call_info["args"] = _execution_args(
                            call_info["name"],
                            call_info.get("args", {}),
                            force_refresh=force_refresh,
                            user_text=req.message,
                        )
                    logger.info(
                        "agent_v2_tool_round requested=%s unique=%s",
                        [c["name"] for c in deferred_calls],
                        [c["name"] for c in unique_calls],
                    )

                    tool_results: list[dict] = []
                    turn_cache_hits: set[str] = set()
                    pending_calls: list[dict] = []
                    executed_by_signature: dict[str, tuple[str, float]] = {}

                    for call_info in unique_calls:
                        signature = _tool_call_signature(call_info["name"], call_info["args"])
                        if signature in tool_result_cache:
                            turn_cache_hits.add(signature)
                            continue
                        pending_calls.append(call_info)

                    if pending_calls:
                        for tool_item in _execute_tools_parallel_keepalive(pending_calls):
                            if tool_item is None:
                                yield _sse_ping()
                                continue
                            call_info, result_str, elapsed_ms = tool_item
                            signature = _tool_call_signature(call_info["name"], call_info["args"])
                            tool_result_cache[signature] = result_str
                            executed_by_signature[signature] = (result_str, elapsed_ms)

                    for call_info in unique_calls:
                        signature = _tool_call_signature(call_info["name"], call_info["args"])
                        if signature in turn_cache_hits:
                            result_str = tool_result_cache[signature]
                            elapsed_ms = 0.0
                        else:
                            result_str, elapsed_ms = executed_by_signature[signature]

                        err_msg = _tool_error_message(result_str)
                        meta = _tool_meta(result_str)
                        cache_status = "turn_hit" if signature in turn_cache_hits else str(meta.get("cache", "unknown"))
                        logger.info(
                            "agent_v2_tool_exec name=%s duration_ms=%.1f cache=%s status=%s",
                            call_info["name"],
                            elapsed_ms,
                            cache_status,
                            "error" if err_msg else "ok",
                        )
                        if call_info["name"] == "search_agent_capabilities" and not err_msg:
                            discovered = [
                                tool_name
                                for tool_name in _capability_names_from_search_result(result_str)
                                if tool_name not in active_tool_names
                            ]
                            if discovered:
                                active_tool_names.extend(discovered)
                                tool_defs = _tool_definitions_from_names(provider, active_tool_names)
                                logger.info("agent_v2_tool_expansion added=%s total=%d", discovered, len(tool_defs))

                        for call_id in call_info.get("call_ids", []):
                            payload = {
                                "name": call_info["name"],
                                "id": call_id,
                                "status": "error" if err_msg else "ok",
                            }
                            if err_msg:
                                payload["message"] = err_msg
                            yield _sse("tool_result", payload)

                            if provider == PROVIDER_ANTHROPIC:
                                result_block: dict[str, object] = {
                                    "type": "tool_result",
                                    "tool_use_id": call_id,
                                    "content": result_str,
                                }
                                if err_msg:
                                    result_block["is_error"] = True
                            else:
                                result_block = {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": result_str,
                                }
                            tool_results.append(result_block)

                    if provider == PROVIDER_ANTHROPIC:
                        conversation.append({"role": "assistant", "content": assistant_content})
                        conversation.append({"role": "user", "content": tool_results})
                    else:
                        conversation.extend(assistant_content)
                        conversation.extend(tool_results)
                    force_tool_use = False
                    continuation_round += 1
                    continue

                if provider == PROVIDER_ANTHROPIC and final_message.stop_reason == "pause_turn":
                    conversation.append({"role": "assistant", "content": assistant_content})
                    conversation.append({"role": "user", "content": [{"type": "text", "text": "Continue."}]})
                    force_tool_use = False
                    continuation_round += 1
                    continue

                usage = _usage_dict(final_message)
                # Finalize turn before last yield
                full_text = "".join(text_parts)
                user_msg = {"role": "user", "content": req.message, "timestamp": time.time()}
                assistant_msg = {"role": "assistant", "content": full_text, "timestamp": time.time()}
                finalize_turn_async(session_id, user_msg, assistant_msg)
                yield _sse("done", {"usage": usage, "session_id": session_id})
                return

        except Exception as exc:
            logger.exception("Agent v2 stream error")
            yield _sse("error", {"message": _format_stream_error(exc)})
            yield _sse("done", {"usage": {}, "session_id": session_id})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )
