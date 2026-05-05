"""
AI Agent chat endpoint with streaming (SSE) and function calling.

Uses the configured LLM provider and the tool definitions from :mod:`api.agent_tools`
to answer cross-cutting investment questions by fetching live data from the
platform's analysis modules.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import re
import time
from collections import Counter
from collections.abc import Sized
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import cache, lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from api.agent_governance import (
    AgentBudgetExceeded,
    AgentBudgetState,
    blocked_tool_payload,
    prepare_model_egress,
)
from api.agent_models import (
    AgentChatJobRequest,
    AgentChatRequest,
    AgentChatRequestV2,
    AgentResponsePreferences,
    ChatMessage,
    ScreenContextModel,
)
from api.exceptions import ConfigurationError
from api.job_events import append_job_event, list_job_events
from api.job_queue import cancel_job, get_job
from api.routers.auth import require_actor
from api.workflows import AVAILABLE_WORKFLOWS, execute_workflow
from llm_utils import (
    MODEL_MID,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    api_key_env,
    apply_reasoning_config,
    get_llm_client,
    reasoning_effort_for_tier,
    resolve_model,
    selected_provider,
)
from ontology.action_registry import get_tool_exposure
from ontology.policy import Actor, actor_to_dict, agent_actor

router = APIRouter()
logger = logging.getLogger("api.agent")
ActorDep = Annotated[Actor, Depends(require_actor)]


@lru_cache(maxsize=1)
def _agent_capability_by_name() -> dict[str, Any]:
    from api.agent_tools import AGENT_CAPABILITY_BY_NAME

    return AGENT_CAPABILITY_BY_NAME


@lru_cache(maxsize=1)
def _tool_definitions() -> tuple[dict[str, Any], ...]:
    from api.agent_tools import TOOL_DEFINITIONS

    return tuple(TOOL_DEFINITIONS)


@lru_cache(maxsize=1)
def _tool_names() -> frozenset[str]:
    return frozenset(tool["name"] for tool in _tool_definitions() if isinstance(tool.get("name"), str))


def _list_agent_capabilities() -> list[dict[str, Any]]:
    from api.agent_tools import list_agent_capabilities

    return list_agent_capabilities()


def execute_tool(name: str, arguments: dict, **kwargs: Any) -> str:
    """Lazy wrapper kept patchable for tests and local tool execution hooks."""
    from api.agent_tools import execute_tool as _execute_tool

    return _execute_tool(name, arguments, **kwargs)


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


def _with_response_preferences(base: str, prefs: AgentResponsePreferences | None) -> str:
    preference_section = _build_response_preferences_section(prefs)
    if not preference_section:
        return base
    return base + "\n\n---\n\n" + preference_section


def _build_response_preferences_section(prefs: AgentResponsePreferences | None) -> str:
    if prefs is None:
        return ""

    lines = [
        "## User Response Preferences",
        "",
        "Apply these preferences to response style only. They do not override tool-use, data-quality, approval-gating, or investment-process instructions.",
        f"- Personality: {prefs.personality}",
        f"- Warmth: {prefs.warmth}",
        f"- Enthusiasm: {prefs.enthusiasm}",
        f"- Headers and lists: {prefs.headers_lists}",
        f"- Emoji: {prefs.emoji}",
        f"- Fast answers: {'yes' if prefs.fast_answers else 'no'}",
    ]
    if prefs.fast_answers:
        lines.append("- For simple questions, answer directly and keep supporting detail minimal.")
    if prefs.custom_instructions:
        lines.extend(
            [
                "",
                "Custom response instructions:",
                prefs.custom_instructions.strip(),
            ]
        )
    return "\n".join(lines)


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


@router.get("/agent/workflows")
def list_workflows():
    """List available deterministic workflows."""
    return [{"name": name, **info} for name, info in AVAILABLE_WORKFLOWS.items()]


@router.get("/agent/capabilities")
def list_capabilities():
    """List Stan's provider-neutral app capabilities."""
    return {"capabilities": _list_agent_capabilities(), "count": len(_tool_definitions())}


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


def _agent_chat_job_cache_key(req: AgentChatJobRequest) -> str:
    session_id = str(req.session_id or "new")
    if req.client_turn_id:
        return f"agent_chat_turn:{session_id}:{req.client_turn_id}"
    payload = req.model_dump(
        exclude={"actor", "finalize_synchronously", "allow_workflow_handoff"},
        exclude_none=True,
    )
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]
    return f"agent_chat_turn:{session_id}:active:{digest}"


def _agent_job_session_id(row: dict[str, Any] | None) -> str | None:
    payload = row.get("payload_json") if isinstance(row, dict) else None
    if isinstance(payload, dict):
        session_id = payload.get("session_id")
        if isinstance(session_id, str) and session_id:
            return session_id
    return None


def _agent_async_payload(row: dict[str, Any], *, events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    from api.async_job_runner import job_response

    payload = job_response(row)
    session_id = _agent_job_session_id(row)
    if session_id:
        payload["session_id"] = session_id
    if events is not None:
        payload["events"] = events
        payload["next_seq"] = max([int(event["seq"]) for event in events], default=0)
    return payload


def _enqueue_agent_chat_turn(
    req: AgentChatRequestV2,
    actor: Actor,
    *,
    emit_starting_event: bool = True,
) -> tuple[dict[str, Any], str]:
    from api import memory_db
    from api.async_job_runner import enqueue_registered_job

    session = memory_db.get_or_create_session(req.session_id)
    session_id = str(session["session_id"])
    job_req = AgentChatJobRequest.model_validate(
        {
            **req.model_dump(exclude={"session_id"}),
            "session_id": session_id,
            "actor": actor_to_dict(actor),
            "message_count": int(session.get("message_count") or 0),
            "finalize_synchronously": True,
            "allow_workflow_handoff": False,
        }
    )
    cache_key = _agent_chat_job_cache_key(job_req)
    reuse_completed = bool(job_req.client_turn_id)
    row, disposition = enqueue_registered_job(
        "agent_chat_turn",
        job_req.model_dump(exclude_none=True),
        cache_key=cache_key,
        reuse_completed=reuse_completed,
    )
    job_id = str(row.get("job_id") or "")
    current_row = get_job(job_id) or row
    events = list_job_events(job_id, after_seq=0)
    if (
        emit_starting_event
        and disposition == "created"
        and not events
        and str(current_row.get("status") or "") == "queued"
    ):
        append_job_event(job_id, "status", {"status": "starting", "session_id": session_id})
        events = list_job_events(job_id, after_seq=0)
    payload = _agent_async_payload(get_job(job_id) or current_row, events=events)
    payload["disposition"] = disposition
    return payload, disposition


MAX_TOOL_CONTINUATION_ROUNDS = 8
MAX_API_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds
SSE_KEEPALIVE_INTERVAL_S = 15.0
LLM_MAX_TOKENS = 8_192
LLM_CHAT_MAX_TOKENS = 2_048


class _LazyProviderToolDefinitions:
    def __init__(self, provider: str):
        self.provider = provider

    def _items(self) -> list[dict]:
        return list(_tool_definition_by_name_for_provider(self.provider).values())

    def __iter__(self):
        return iter(self._items())

    def __len__(self) -> int:
        return len(self._items())

    def __getitem__(self, index: int) -> dict:
        return self._items()[index]


ANTHROPIC_TOOL_DEFINITIONS = _LazyProviderToolDefinitions(PROVIDER_ANTHROPIC)
OPENAI_TOOL_DEFINITIONS = _LazyProviderToolDefinitions(PROVIDER_OPENAI)
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
    r"thesis|catalysts?|kill conditions?|dossier|workflow|approvals?|action items?|triggers?|search|news|"
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


def _prefers_no_followups(prefs: AgentResponsePreferences | None) -> bool:
    custom = (prefs.custom_instructions or "").lower() if prefs else ""
    return "no follow-up" in custom or "do not ask follow" in custom or "don't ask follow" in custom


def _chat_reasoning_effort(provider: str, prefs: AgentResponsePreferences | None) -> str | None:
    return reasoning_effort_for_tier(MODEL_MID, provider) if prefs and prefs.thinking_enabled else None


def _casual_response(user_text: str, prefs: AgentResponsePreferences | None = None) -> str:
    text = (user_text or "").strip().lower()
    if "thank" in text or text in {"thanks", "cool", "ok", "okay"}:
        return "Anytime."
    if "who are you" in text or "what can you do" in text:
        return "I'm Stan. I can help with portfolio, market, macro, thesis, and risk questions."
    if _prefers_no_followups(prefs):
        return "Hey."
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
            if name in _tool_names() and name not in selected:
                selected.append(name)

    if re.search(r"\b(portfolio|holding|holdings|position|positions|p&l|pnl|performance|exposure|risks?)\b", text):
        add("get_portfolio", "get_portfolio_risk", "query_ontology")
    if re.search(r"\brecommendation\b.*\brisk\b|\brisk\b.*\brecommendation\b", text):
        add("get_recommendation_risk", "get_portfolio_risk")
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
    if "catalyst" in text and re.search(
        r"\b(played out|played-out|play out|materiali[sz]ed|happened|occurred|announced|launched|approved|failed|status)\b",
        text,
    ):
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
    for cap in _agent_capability_by_name().values():
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
    definitions = _tool_definition_by_name_for_provider(provider)
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
    actor: Actor | None = None,
) -> list[tuple[dict, str, float]]:
    """Execute deduplicated tool calls in parallel and measure runtime."""
    if len(calls) == 1:
        c = calls[0]
        started = time.perf_counter()
        result = _execute_tool_for_actor(c["name"], c["args"], actor)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return [(c, result, elapsed_ms)]

    with ThreadPoolExecutor(max_workers=min(len(calls), 8)) as pool:
        futures = []
        for c in calls:
            started = time.perf_counter()
            fut = pool.submit(
                _execute_tool_for_actor,
                c["name"],
                c["args"],
                actor,
                c.get("_provenance_context"),
            )
            futures.append((c, fut, started))
        out: list[tuple[dict, str, float]] = []
        for c, fut, started in futures:
            result = fut.result()
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            out.append((c, result, elapsed_ms))
        return out


def _execute_tools_parallel_keepalive(
    calls: list[dict],
    actor: Actor | None = None,
):
    """Execute tool calls while yielding None periodically as an SSE keepalive signal."""
    if not calls:
        return

    with ThreadPoolExecutor(max_workers=min(len(calls), 8)) as pool:
        future_meta = {}
        for c in calls:
            started = time.perf_counter()
            fut = pool.submit(
                _execute_tool_for_actor,
                c["name"],
                c["args"],
                actor,
                c.get("_provenance_context"),
            )
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


def _execute_workflow_keepalive(
    workflow_name: str,
    workflow_ticker: str | None,
    *,
    actor: Actor | None,
):
    """Run a deterministic workflow while emitting SSE keepalive frames."""
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(execute_workflow, workflow_name, workflow_ticker, actor)
    try:
        pending = {future}
        while pending:
            done, pending = wait(
                pending,
                timeout=SSE_KEEPALIVE_INTERVAL_S,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                yield _sse_ping()
                continue
            return future.result()
    finally:
        if not future.done():
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)


def _execute_tool_for_actor(
    name: str,
    args: dict,
    actor: Actor | None,
    provenance_context: dict[str, Any] | None = None,
) -> str:
    params = inspect.signature(execute_tool).parameters.values()
    param_names = {p.name for p in params}
    supports_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    kwargs: dict[str, Any] = {}
    if supports_kwargs or "actor" in param_names:
        kwargs["actor"] = actor
    if provenance_context and (supports_kwargs or "provenance_context" in param_names):
        kwargs["provenance_context"] = provenance_context
    if kwargs:
        return execute_tool(name, args, **kwargs)
    return execute_tool(name, args)


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


def _tool_result_status(result_str: str) -> str:
    meta = _tool_meta(result_str)
    raw = meta.get("status")
    if raw in {"blocked", "timeout", "cancelled", "partial", "retrying", "denied", "failed_closed"}:
        if raw in {"denied", "failed_closed"}:
            return "blocked"
        return str(raw)
    return "error" if _tool_error_message(result_str) else "ok"


def _blocked_tool_result(name: str, exc: Exception, *, status: str = "blocked") -> str:
    payload = blocked_tool_payload(name, exc, status=status)
    return json.dumps(payload, sort_keys=True, default=str)


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
        if isinstance(name, str) and name in _tool_names() and name not in names:
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


@cache
def _tool_definition_by_name_for_provider(provider: str) -> dict[str, dict]:
    tools: list[dict] = []
    for tool in _tool_definitions():
        name = tool.get("name")
        if not isinstance(name, str):
            continue
        if provider == PROVIDER_ANTHROPIC:
            tools.append(
                {
                    "name": name,
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("parameters", {"type": "object", "properties": {}, "required": []}),
                }
            )
        elif provider == PROVIDER_OPENAI:
            tools.append(
                {
                    "type": "function",
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}, "required": []}),
                }
            )
        else:
            raise KeyError(provider)
    return {tool["name"]: tool for tool in tools}


def _tool_definitions_for_provider(provider: str) -> list[dict]:
    return list(_tool_definition_by_name_for_provider(provider).values())


def _model_stream_kwargs(
    *,
    provider: str,
    instructions: str,
    conversation: list[dict],
    max_tokens: int,
    tool_defs: list[dict] | None = None,
    force_tool_use: bool = False,
    reasoning_effort: str | None = None,
) -> dict[str, object]:
    if provider == PROVIDER_ANTHROPIC:
        resolved_model = resolve_model(MODEL_MID, PROVIDER_ANTHROPIC)
        kwargs: dict[str, object] = {
            "model": resolved_model,
            "max_tokens": max_tokens,
            "system": instructions,
            "messages": conversation,
        }
        apply_reasoning_config(
            kwargs,
            provider=PROVIDER_ANTHROPIC,
            model=resolved_model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        if tool_defs:
            kwargs["tools"] = tool_defs
        if force_tool_use and tool_defs and reasoning_effort is None:
            kwargs["tool_choice"] = {"type": "any"}
        return kwargs

    resolved_model = resolve_model(MODEL_MID, PROVIDER_OPENAI)
    kwargs = {
        "model": resolved_model,
        "max_output_tokens": max_tokens,
        "instructions": instructions,
        "input": conversation,
    }
    apply_reasoning_config(
        kwargs,
        provider=PROVIDER_OPENAI,
        model=resolved_model,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
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


def _start_agent_turn_provenance(
    *,
    session_id: str | None,
    message: object,
    provider: str,
    actor: Actor | None,
    workflow_name: str | None = None,
    workflow_ticker: str | None = None,
) -> str | None:
    try:
        from api import provenance

        event_id = provenance.deterministic_id(
            "pv:agent_turn",
            session_id or "legacy",
            provenance.stable_hash(message),
            int(time.time() * 1_000_000),
        )
        provenance.start_event(
            event_id=event_id,
            event_type="agent_turn",
            event_name="agent_chat",
            actor=actor,
            agent_session_id=session_id,
            input_value=message,
            summary={
                "provider": provider,
                "session_id": session_id,
                "workflow_name": workflow_name,
                "workflow_ticker": workflow_ticker,
                "message_hash": provenance.stable_hash(message),
            },
            metadata={"message_type": type(message).__name__},
        )
        return event_id
    except Exception:
        logger.debug("Failed to start agent turn provenance session=%s", session_id, exc_info=True)
        return None


def _finish_agent_turn_provenance(
    event_id: str | None,
    *,
    status: str,
    output_value: object | None = None,
    usage: dict | None = None,
    error: str | None = None,
) -> None:
    try:
        from api import provenance

        provenance.finish_event(
            event_id,
            status=status,
            output_value=output_value,
            summary={"status": status, "usage": usage or {}},
            metadata={"usage": usage or {}},
            error=error,
        )
    except Exception:
        logger.debug("Failed to finish agent turn provenance event=%s", event_id, exc_info=True)


def _start_model_call_provenance(
    *,
    parent_event_id: str | None,
    session_id: str | None,
    workflow_run_id: str | None,
    provider: str,
    purpose: str,
    stream_kwargs: dict[str, object],
    actor: Actor | None,
    attempt: int,
    round_index: int | None = None,
) -> str | None:
    try:
        from api import provenance

        model = stream_kwargs.get("model")
        conversation = stream_kwargs.get("messages") or stream_kwargs.get("input")
        tools = stream_kwargs.get("tools")
        event_id = provenance.deterministic_id(
            "pv:model_call",
            session_id or workflow_run_id or "legacy",
            purpose,
            round_index,
            attempt,
            int(time.time() * 1_000_000),
        )
        provenance.start_event(
            event_id=event_id,
            event_type="model_call",
            event_name=str(model or provider),
            actor=actor,
            parent_event_id=parent_event_id,
            workflow_run_id=workflow_run_id,
            agent_session_id=session_id,
            input_value={
                "instructions": stream_kwargs.get("instructions") or stream_kwargs.get("system"),
                "conversation": conversation,
            },
            summary={
                "provider": provider,
                "model": model,
                "purpose": purpose,
                "attempt": attempt,
                "round_index": round_index,
                "tool_count": len(tools) if isinstance(tools, Sized) else 0,
            },
            metadata={
                "max_tokens": stream_kwargs.get("max_tokens"),
                "tool_choice": stream_kwargs.get("tool_choice"),
                "reasoning": stream_kwargs.get("reasoning"),
            },
        )
        return event_id
    except Exception:
        logger.debug("Failed to start model call provenance purpose=%s", purpose, exc_info=True)
        return None


def _finish_model_call_provenance(
    event_id: str | None,
    *,
    status: str,
    final_message: object | None = None,
    output_text: str | None = None,
    error: str | None = None,
) -> None:
    usage = _usage_dict(final_message) if final_message is not None else {}
    try:
        from api import provenance

        provenance.finish_event(
            event_id,
            status=status,
            output_value=output_text if output_text is not None else final_message,
            summary={"status": status, "usage": usage},
            metadata={"usage": usage},
            error=error,
        )
    except Exception:
        logger.debug("Failed to finish model call provenance event=%s", event_id, exc_info=True)


def _attach_tool_provenance_context(
    calls: list[dict],
    *,
    parent_event_id: str | None,
    session_id: str | None,
    workflow_run_id: str | None,
    source: str,
) -> None:
    if not (parent_event_id or session_id or workflow_run_id):
        return
    for call in calls:
        call_id = next((str(cid) for cid in call.get("call_ids", []) if cid), None)
        call["_provenance_context"] = {
            "parent_event_id": parent_event_id,
            "agent_session_id": session_id,
            "workflow_run_id": workflow_run_id,
            "call_id": call_id,
            "source": source,
        }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/agent/chat")
def agent_chat(req: AgentChatRequest, actor: ActorDep):
    tool_actor = agent_actor(actor)
    provider, api_key = _read_llm_api_key()
    reasoning_effort = _chat_reasoning_effort(provider, req.response_preferences)
    instructions = _with_response_preferences(
        _build_agent_instructions(screen_context=req.screen_context),
        req.response_preferences,
    )
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
        budget = AgentBudgetState()
        agent_turn_event_id = _start_agent_turn_provenance(
            session_id=None,
            message=[m.model_dump() for m in req.messages],
            provider=provider,
            actor=tool_actor,
            workflow_name=workflow_name,
            workflow_ticker=workflow_ticker,
        )

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
                    _finish_agent_turn_provenance(agent_turn_event_id, status="succeeded", usage={})
                    yield _sse("done", {"usage": {}})
                    return

                # Emit tool calls as they execute
                run_id, synthesis_prompt, sections = yield from _execute_workflow_keepalive(
                    workflow_name,
                    workflow_ticker,
                    actor=tool_actor,
                )
                workflow_tool_calls = [
                    {"name": str(section["tool"]), "id": str(section["tool"]), "status": "ok"} for section in sections
                ]
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
                            reasoning_effort=reasoning_effort,
                        )
                        stream_kwargs, egress_meta = prepare_model_egress(
                            provider=provider,
                            purpose="workflow_synthesis",
                            stream_kwargs=stream_kwargs,
                            actor=tool_actor,
                            budget=budget,
                            parent_event_id=agent_turn_event_id,
                            session_id=None,
                            workflow_run_id=run_id,
                        )
                        yield _sse("egress_recorded", egress_meta)
                        yield _sse("budget_update", budget.to_meta())
                        model_event_id = _start_model_call_provenance(
                            parent_event_id=agent_turn_event_id,
                            session_id=None,
                            workflow_run_id=run_id,
                            provider=provider,
                            purpose="workflow_synthesis",
                            stream_kwargs=stream_kwargs,
                            actor=tool_actor,
                            attempt=attempt,
                        )
                        final_message = yield from _stream_llm_response(
                            client,
                            provider,
                            stream_kwargs,
                            synthesis_chunks,
                        )
                        _finish_model_call_provenance(
                            model_event_id,
                            status="succeeded",
                            final_message=final_message,
                            output_text="".join(synthesis_chunks),
                        )
                        budget.record_model_usage(_usage_dict(final_message))
                        yield _sse("budget_update", budget.to_meta())
                        break
                    except Exception as retry_exc:
                        _finish_model_call_provenance(
                            locals().get("model_event_id"),
                            status="failed",
                            error=str(retry_exc) or retry_exc.__class__.__name__,
                        )
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
                _finish_agent_turn_provenance(
                    agent_turn_event_id,
                    status="succeeded",
                    output_value=synthesis_text,
                    usage=usage,
                )
                yield _sse(
                    "done",
                    {
                        "usage": usage,
                        "workflow_run_id": run_id,
                        "tool_calls": workflow_tool_calls,
                        "tools_used": [call["name"] for call in workflow_tool_calls],
                    },
                )
                return

            except Exception as exc:
                logger.exception("Workflow %s failed", workflow_name)
                try:
                    from portfolio.core_db import fail_workflow_run

                    fail_workflow_run(run_id, str(exc))
                except Exception:
                    pass
                _finish_agent_turn_provenance(
                    agent_turn_event_id,
                    status="failed",
                    usage={},
                    error=str(exc) or exc.__class__.__name__,
                )
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
                    _finish_agent_turn_provenance(
                        agent_turn_event_id,
                        status="failed",
                        usage={},
                        error=f"Tool-call loop limit reached ({MAX_TOOL_CONTINUATION_ROUNDS} rounds).",
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
                    reasoning_effort=reasoning_effort,
                )

                model_event_id: str | None = None
                for attempt in range(MAX_API_RETRIES):
                    try:
                        stream_kwargs, egress_meta = prepare_model_egress(
                            provider=provider,
                            purpose="agent_chat",
                            stream_kwargs=stream_kwargs,
                            actor=tool_actor,
                            budget=budget,
                            parent_event_id=agent_turn_event_id,
                            session_id=None,
                            workflow_run_id=None,
                        )
                        yield _sse("egress_recorded", egress_meta)
                        yield _sse("budget_update", budget.to_meta())
                        model_event_id = _start_model_call_provenance(
                            parent_event_id=agent_turn_event_id,
                            session_id=None,
                            workflow_run_id=None,
                            provider=provider,
                            purpose="agent_chat",
                            stream_kwargs=stream_kwargs,
                            actor=tool_actor,
                            attempt=attempt,
                            round_index=continuation_round,
                        )
                        final_message = yield from _stream_llm_response(client, provider, stream_kwargs)
                        _finish_model_call_provenance(
                            model_event_id,
                            status="succeeded",
                            final_message=final_message,
                        )
                        budget.record_model_usage(_usage_dict(final_message))
                        yield _sse("budget_update", budget.to_meta())
                        break
                    except Exception as retry_exc:
                        _finish_model_call_provenance(
                            model_event_id,
                            status="failed",
                            error=str(retry_exc) or retry_exc.__class__.__name__,
                        )
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
                        budgeted_pending: list[dict] = []
                        for call_info in pending_calls:
                            signature = _tool_call_signature(call_info["name"], call_info["args"])
                            try:
                                budget.check_tool_call(get_tool_exposure(call_info["name"]))
                                budgeted_pending.append(call_info)
                                for call_id in call_info.get("call_ids", []):
                                    yield _sse(
                                        "tool_progress",
                                        {"name": call_info["name"], "id": call_id, "status": "running"},
                                    )
                            except AgentBudgetExceeded as exc:
                                result_str = _blocked_tool_result(call_info["name"], exc)
                                executed_by_signature[signature] = (result_str, 0.0)
                                for call_id in call_info.get("call_ids", []):
                                    payload = {
                                        "name": call_info["name"],
                                        "id": call_id,
                                        "status": "blocked",
                                        "message": str(exc),
                                    }
                                    yield _sse("policy_failure", payload)
                                    yield _sse("blocked", payload)
                        pending_calls = budgeted_pending
                        yield _sse("budget_update", budget.to_meta())

                    if pending_calls:
                        _attach_tool_provenance_context(
                            pending_calls,
                            parent_event_id=model_event_id,
                            session_id=None,
                            workflow_run_id=None,
                            source="agent.chat",
                        )
                        for tool_item in _execute_tools_parallel_keepalive(pending_calls, actor=tool_actor):
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
                        result_status = _tool_result_status(result_str)
                        cache_status = "turn_hit" if signature in turn_cache_hits else str(meta.get("cache", "unknown"))
                        logger.info(
                            "agent_tool_exec name=%s duration_ms=%.1f cache=%s status=%s quality_ok=%s",
                            call_info["name"],
                            elapsed_ms,
                            cache_status,
                            result_status,
                            str(meta.get("quality_ok", "n/a")),
                        )

                        for call_id in call_info.get("call_ids", []):
                            payload = {
                                "name": call_info["name"],
                                "id": call_id,
                                "status": result_status,
                            }
                            if meta.get("policy_decision_id"):
                                payload["policy_decision_id"] = meta.get("policy_decision_id")
                            if meta.get("duration_ms") is not None:
                                payload["elapsed_ms"] = meta.get("duration_ms")
                            if err_msg:
                                payload["message"] = err_msg
                            yield _sse("tool_result", payload)
                            if result_status == "blocked":
                                yield _sse("policy_failure", payload)
                                yield _sse("blocked", payload)
                            elif result_status == "timeout":
                                yield _sse("timeout", payload)

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
                _finish_agent_turn_provenance(
                    agent_turn_event_id,
                    status="succeeded",
                    output_value=final_message,
                    usage=usage,
                )
                yield _sse("done", {"usage": usage})
                return

        except Exception as exc:
            logger.exception("Agent stream error")
            _finish_agent_turn_provenance(
                agent_turn_event_id,
                status="failed",
                usage={},
                error=str(exc) or exc.__class__.__name__,
            )
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


@router.post("/agent/chat/async")
def start_agent_chat_async(req: AgentChatRequestV2, actor: ActorDep):
    payload, _disposition = _enqueue_agent_chat_turn(req, actor)
    status_code = 200 if payload.get("status") in {"done", "error", "cancelled"} else 202
    return JSONResponse(payload, status_code=status_code)


@router.get("/agent/chat/async/{job_id}/events")
def get_agent_chat_async_events(
    job_id: str,
    after_seq: int = Query(0, ge=0),
    wait_ms: int = Query(0, ge=0, le=14_000),
):
    from api.async_job_runner import poll_registered_job

    deadline = time.monotonic() + min(wait_ms, 14_000) / 1000.0
    events: list[dict[str, Any]] = []
    status_payload: dict[str, Any]
    row = get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    while True:
        try:
            status_payload = poll_registered_job(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown job_id") from None
        row = get_job(job_id) or row
        events = list_job_events(job_id, after_seq=after_seq)
        if events or status_payload.get("status") not in {"queued", "running"} or time.monotonic() >= deadline:
            break
        time.sleep(0.25)

    next_seq = max([int(event["seq"]) for event in events], default=after_seq)
    payload: dict[str, Any] = {
        **status_payload,
        "session_id": _agent_job_session_id(row),
        "events": events,
        "next_seq": next_seq,
    }
    return payload


@router.post("/agent/chat/async/{job_id}/cancel")
def cancel_agent_chat_async(job_id: str):
    from api.async_job_runner import poll_registered_job
    from api.job_registry import get_job_spec

    row = get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    ttl = get_job_spec("agent_chat_turn").failed_ttl_s
    cancel_job(job_id, "Job cancelled by user", result_ttl_seconds=ttl)
    session_id = _agent_job_session_id(row)
    append_job_event(job_id, "status", {"status": "cancelled", "session_id": session_id})
    append_job_event(job_id, "cancelled", {"status": "cancelled", "session_id": session_id})
    append_job_event(job_id, "error", {"message": "Cancelled.", "session_id": session_id})
    try:
        payload = poll_registered_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job_id") from None
    payload["session_id"] = session_id
    payload["events"] = list_job_events(job_id, after_seq=0)
    payload["next_seq"] = max([int(event["seq"]) for event in payload["events"]], default=0)
    return payload


@router.post("/agent/chat/v2")
def agent_chat_v2(req: AgentChatRequestV2, actor: ActorDep):
    tool_actor = agent_actor(actor)
    """Chat endpoint with server-managed conversation memory.

    The frontend sends only the new message + session_id.  The server
    assembles optimal context from a rolling summary, verbatim window,
    and retrieval hits.
    """
    casual = _is_casual(req.message)
    workflow_name, workflow_ticker = _detect_workflow(req.message)
    if workflow_name and req.allow_workflow_handoff:
        provider_label = "deferred"
        active_tool_names: list[str] = []
        tool_defs: list[dict] = []
    else:
        try:
            provider = selected_provider()
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        provider_label = provider
        active_tool_names = [] if casual else _select_tool_names(req.message)
        tool_defs = _tool_definitions_from_names(provider, active_tool_names)
    force_refresh = _wants_fresh_data(req.message)
    enable_retrieval = _should_use_retrieval(req.message)
    logger.info(
        "agent_v2 provider=%s casual=%s workflow=%s ticker=%s tools=%d refresh=%s retrieval=%s session=%s",
        provider_label,
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
        if workflow_name and req.allow_workflow_handoff:
            payload, _disposition = _enqueue_agent_chat_turn(req, actor)
            yield _sse("handoff", payload)
            return

        from api.memory_manager import build_conversation_context, finalize_turn, finalize_turn_async

        finalize_turn_fn = finalize_turn if req.finalize_synchronously else finalize_turn_async

        agent_turn_event_id: str | None = None
        if casual and not workflow_name:
            from api import memory_db

            session = memory_db.get_or_create_session(req.session_id)
            session_id = str(session["session_id"])
            agent_turn_event_id = _start_agent_turn_provenance(
                session_id=session_id,
                message=req.message,
                provider=selected_provider(),
                actor=tool_actor,
            )
            text = _casual_response(req.message, req.response_preferences)
            yield _sse("delta", {"text": text})
            turn_meta = {"client_turn_id": req.client_turn_id} if req.client_turn_id else {}
            user_msg = {"role": "user", "content": req.message, "timestamp": time.time(), **turn_meta}
            assistant_msg = {"role": "assistant", "content": text, "timestamp": time.time(), **turn_meta}
            finalize_turn_fn(session_id, user_msg, assistant_msg)
            _finish_agent_turn_provenance(
                agent_turn_event_id,
                status="succeeded",
                output_value=text,
                usage={},
            )
            yield _sse("done", {"usage": {}, "session_id": session_id})
            return

        provider, api_key = _read_llm_api_key()
        reasoning_effort = _chat_reasoning_effort(provider, req.response_preferences)
        instructions = _with_response_preferences(
            _build_agent_instructions(screen_context=req.screen_context),
            req.response_preferences,
        )
        client = _get_provider_client(provider, api_key)
        budget = AgentBudgetState()
        raw_conversation, session_id = build_conversation_context(
            req.session_id,
            req.message,
            enable_retrieval=enable_retrieval,
        )
        agent_turn_event_id = _start_agent_turn_provenance(
            session_id=session_id,
            message=req.message,
            provider=provider,
            actor=tool_actor,
            workflow_name=workflow_name,
            workflow_ticker=workflow_ticker,
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
                    _finish_agent_turn_provenance(agent_turn_event_id, status="succeeded", usage={})
                    yield _sse("done", {"usage": {}, "session_id": session_id})
                    return

                run_id, synthesis_prompt, sections = yield from _execute_workflow_keepalive(
                    workflow_name,
                    workflow_ticker,
                    actor=tool_actor,
                )
                workflow_tool_calls = [
                    {"name": str(section["tool"]), "id": str(section["tool"]), "status": "ok"} for section in sections
                ]
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
                            reasoning_effort=reasoning_effort,
                        )
                        stream_kwargs, egress_meta = prepare_model_egress(
                            provider=provider,
                            purpose="workflow_synthesis",
                            stream_kwargs=stream_kwargs,
                            actor=tool_actor,
                            budget=budget,
                            parent_event_id=agent_turn_event_id,
                            session_id=session_id,
                            workflow_run_id=run_id,
                        )
                        yield _sse("egress_recorded", egress_meta)
                        yield _sse("budget_update", budget.to_meta())
                        model_event_id = _start_model_call_provenance(
                            parent_event_id=agent_turn_event_id,
                            session_id=session_id,
                            workflow_run_id=run_id,
                            provider=provider,
                            purpose="workflow_synthesis",
                            stream_kwargs=stream_kwargs,
                            actor=tool_actor,
                            attempt=attempt,
                        )
                        final_message = yield from _stream_llm_response(
                            client,
                            provider,
                            stream_kwargs,
                            synthesis_chunks,
                        )
                        _finish_model_call_provenance(
                            model_event_id,
                            status="succeeded",
                            final_message=final_message,
                            output_text="".join(synthesis_chunks),
                        )
                        budget.record_model_usage(_usage_dict(final_message))
                        yield _sse("budget_update", budget.to_meta())
                        break
                    except Exception as retry_exc:
                        _finish_model_call_provenance(
                            locals().get("model_event_id"),
                            status="failed",
                            error=str(retry_exc) or retry_exc.__class__.__name__,
                        )
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
                turn_meta = {"client_turn_id": req.client_turn_id} if req.client_turn_id else {}
                user_msg = {"role": "user", "content": req.message, "timestamp": time.time(), **turn_meta}
                assistant_msg = {
                    "role": "assistant",
                    "content": synthesis_text,
                    "timestamp": time.time(),
                    "toolCalls": workflow_tool_calls,
                    **turn_meta,
                }
                finalize_turn_fn(session_id, user_msg, assistant_msg)
                _finish_agent_turn_provenance(
                    agent_turn_event_id,
                    status="succeeded",
                    output_value=synthesis_text,
                    usage=usage,
                )
                yield _sse(
                    "done",
                    {
                        "usage": usage,
                        "session_id": session_id,
                        "workflow_run_id": run_id,
                        "tool_calls": workflow_tool_calls,
                        "tools_used": [call["name"] for call in workflow_tool_calls],
                    },
                )
                return

            except Exception as exc:
                logger.exception("Workflow %s failed (v2)", workflow_name)
                try:
                    from portfolio.core_db import fail_workflow_run

                    fail_workflow_run(run_id, str(exc))
                except Exception:
                    pass
                _finish_agent_turn_provenance(
                    agent_turn_event_id,
                    status="failed",
                    usage={},
                    error=str(exc) or exc.__class__.__name__,
                )
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
                    _finish_agent_turn_provenance(
                        agent_turn_event_id,
                        status="failed",
                        usage={},
                        error=f"Tool-call loop limit reached ({MAX_TOOL_CONTINUATION_ROUNDS} rounds).",
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
                    reasoning_effort=reasoning_effort,
                )

                model_event_id: str | None = None
                for attempt in range(MAX_API_RETRIES):
                    try:
                        stream_kwargs, egress_meta = prepare_model_egress(
                            provider=provider,
                            purpose="agent_chat",
                            stream_kwargs=stream_kwargs,
                            actor=tool_actor,
                            budget=budget,
                            parent_event_id=agent_turn_event_id,
                            session_id=session_id,
                            workflow_run_id=None,
                        )
                        yield _sse("egress_recorded", egress_meta)
                        yield _sse("budget_update", budget.to_meta())
                        model_event_id = _start_model_call_provenance(
                            parent_event_id=agent_turn_event_id,
                            session_id=session_id,
                            workflow_run_id=None,
                            provider=provider,
                            purpose="agent_chat",
                            stream_kwargs=stream_kwargs,
                            actor=tool_actor,
                            attempt=attempt,
                            round_index=continuation_round,
                        )
                        final_message = yield from _stream_llm_response(
                            client,
                            provider,
                            stream_kwargs,
                            text_parts,
                        )
                        _finish_model_call_provenance(
                            model_event_id,
                            status="succeeded",
                            final_message=final_message,
                            output_text="".join(text_parts),
                        )
                        budget.record_model_usage(_usage_dict(final_message))
                        yield _sse("budget_update", budget.to_meta())
                        break
                    except Exception as retry_exc:
                        _finish_model_call_provenance(
                            model_event_id,
                            status="failed",
                            error=str(retry_exc) or retry_exc.__class__.__name__,
                        )
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
                        budgeted_pending: list[dict] = []
                        for call_info in pending_calls:
                            signature = _tool_call_signature(call_info["name"], call_info["args"])
                            try:
                                budget.check_tool_call(get_tool_exposure(call_info["name"]))
                                budgeted_pending.append(call_info)
                                for call_id in call_info.get("call_ids", []):
                                    yield _sse(
                                        "tool_progress",
                                        {"name": call_info["name"], "id": call_id, "status": "running"},
                                    )
                            except AgentBudgetExceeded as exc:
                                result_str = _blocked_tool_result(call_info["name"], exc)
                                executed_by_signature[signature] = (result_str, 0.0)
                                for call_id in call_info.get("call_ids", []):
                                    payload = {
                                        "name": call_info["name"],
                                        "id": call_id,
                                        "status": "blocked",
                                        "message": str(exc),
                                    }
                                    yield _sse("policy_failure", payload)
                                    yield _sse("blocked", payload)
                        pending_calls = budgeted_pending
                        yield _sse("budget_update", budget.to_meta())

                    if pending_calls:
                        _attach_tool_provenance_context(
                            pending_calls,
                            parent_event_id=model_event_id,
                            session_id=session_id,
                            workflow_run_id=None,
                            source="agent.chat.v2",
                        )
                        for tool_item in _execute_tools_parallel_keepalive(pending_calls, actor=tool_actor):
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
                        result_status = _tool_result_status(result_str)
                        cache_status = "turn_hit" if signature in turn_cache_hits else str(meta.get("cache", "unknown"))
                        logger.info(
                            "agent_v2_tool_exec name=%s duration_ms=%.1f cache=%s status=%s",
                            call_info["name"],
                            elapsed_ms,
                            cache_status,
                            result_status,
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
                                "status": result_status,
                            }
                            if meta.get("policy_decision_id"):
                                payload["policy_decision_id"] = meta.get("policy_decision_id")
                            if meta.get("duration_ms") is not None:
                                payload["elapsed_ms"] = meta.get("duration_ms")
                            if err_msg:
                                payload["message"] = err_msg
                            yield _sse("tool_result", payload)
                            if result_status == "blocked":
                                yield _sse("policy_failure", payload)
                                yield _sse("blocked", payload)
                            elif result_status == "timeout":
                                yield _sse("timeout", payload)

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
                turn_meta = {"client_turn_id": req.client_turn_id} if req.client_turn_id else {}
                user_msg = {"role": "user", "content": req.message, "timestamp": time.time(), **turn_meta}
                assistant_msg = {"role": "assistant", "content": full_text, "timestamp": time.time(), **turn_meta}
                finalize_turn_fn(session_id, user_msg, assistant_msg)
                _finish_agent_turn_provenance(
                    agent_turn_event_id,
                    status="succeeded",
                    output_value=full_text,
                    usage=usage,
                )
                yield _sse("done", {"usage": usage, "session_id": session_id})
                return

        except Exception as exc:
            logger.exception("Agent v2 stream error")
            _finish_agent_turn_provenance(
                agent_turn_event_id,
                status="failed",
                usage={},
                error=str(exc) or exc.__class__.__name__,
            )
            yield _sse("error", {"message": _format_stream_error(exc)})
            yield _sse("done", {"usage": {}, "session_id": session_id})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )
