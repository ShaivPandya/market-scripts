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

from api.agent_domain_policy import (
    DOMAIN_BLOCK_RESPONSE,
    DOMAIN_CLARIFY_RESPONSE,
    MIXED_DOMAIN_INSTRUCTION,
    AgentDomainClassification,
    analyze_agent_domain,
)
from api.agent_governance import (
    AgentBudgetExceeded,
    AgentBudgetState,
    ModelGatewayDenied,
    blocked_tool_payload,
    prepare_model_egress,
)
from api.agent_models import (
    AgentChatJobRequest,
    AgentChatRequest,
    AgentResponsePreferences,
    ChatMessage,
    ScreenContextModel,
)
from api.exceptions import ConfigurationError
from api.job_events import append_job_event, list_job_events
from api.job_queue import cancel_job, get_job
from api.routers.auth import require_actor
from api.tool_data_quality import aggregate_tool_data_quality
from api.workflows import AVAILABLE_WORKFLOWS, execute_workflow
from decision_quality.candidate_gates import apply_opportunity_candidate_gates
from decision_quality.context_packs import (
    build_context_pack_metadata,
    build_context_pack_tool_calls,
    resolve_context_pack,
)
from decision_quality.gates import apply_decision_quality_gates
from decision_quality.intent_router import (
    RouteDecision,
    build_regex_route_decision,
    build_route_context,
    intent_router_training_capture_mismatch_only,
    resolve_agent_route,
    should_capture_training_row,
    training_row_from_telemetry,
)
from decision_quality.models import (
    DecisionQuality,
    DecisionQualityGate,
    decision_quality_schema,
    parse_decision_quality,
)
from decision_quality.opportunity_candidate import (
    OpportunityCandidate,
    OpportunityCandidateGate,
    opportunity_candidate_schema,
    parse_opportunity_candidate,
)
from llm_utils import (
    MODEL_MID,
    PROVIDER_ANTHROPIC,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    api_key_env,
    apply_reasoning_config,
    call_llm_json,
    extract_text,
    get_llm_client,
    reasoning_effort_for_tier,
    require_api_key,
    resolve_model,
    selected_provider,
    selected_provider_for_tier,
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


def _with_domain_guardrail_instruction(base: str, classification: AgentDomainClassification) -> str:
    if not classification.contains_unsupported_request:
        return base
    return base + "\n\n---\n\n## Domain Guardrail\n\n" + MIXED_DOMAIN_INSTRUCTION


def _domain_guardrail_text(classification: AgentDomainClassification) -> str:
    if classification.decision == "clarify":
        return DOMAIN_CLARIFY_RESPONSE
    return DOMAIN_BLOCK_RESPONSE


def _domain_done_meta(classification: AgentDomainClassification) -> dict[str, Any]:
    return {
        "domain_decision": classification.decision,
        "domain_reason": classification.reason,
        "domain_contains_unsupported_request": classification.contains_unsupported_request,
    }


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


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


_PHASE_LABELS = {
    "model_thinking": "Thinking...",
    "tool_running": "Running tools...",
    "model_writing": "Writing answer...",
    "finalizing": "Finalizing...",
}


def _phase_payload(phase: str, turn_started: float, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "phase": phase,
        "label": str(extra.pop("label", _PHASE_LABELS.get(phase, phase))),
        "elapsed_ms": _elapsed_ms(turn_started),
    }
    payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


def _phase_sse(phase: str, turn_started: float, **extra: Any) -> str:
    return _sse("phase", _phase_payload(phase, turn_started, **extra))


def _new_agent_timings() -> dict[str, Any]:
    return {"models": [], "tools": [], "first_token_ms": None}


def _done_payload(base: dict[str, Any], timings: dict[str, Any], turn_started: float) -> dict[str, Any]:
    out = dict(base)
    compact = {
        "total_ms": _elapsed_ms(turn_started),
        "first_token_ms": timings.get("first_token_ms"),
        "models": timings.get("models") or [],
        "tools": timings.get("tools") or [],
    }
    out["timings"] = compact
    return out


def _domain_guardrail_stream(classification: AgentDomainClassification):
    yield _sse_ping()
    yield _sse("delta", {"text": _domain_guardrail_text(classification)})
    yield _sse("done", {"usage": {}, **_domain_done_meta(classification)})


def _domain_guardrail_stream_v2(req: AgentChatRequest, classification: AgentDomainClassification):
    from api import memory_db
    from api.memory_manager import finalize_turn, finalize_turn_async

    turn_started = time.perf_counter()
    timings = _new_agent_timings()
    text = _domain_guardrail_text(classification)
    session = memory_db.get_or_create_session(req.session_id)
    session_id = str(session["session_id"])
    turn_meta = {"client_turn_id": req.client_turn_id} if req.client_turn_id else {}
    user_msg = {"role": "user", "content": req.message, "timestamp": time.time(), **turn_meta}
    assistant_msg = {"role": "assistant", "content": text, "timestamp": time.time(), **turn_meta}
    finalize_turn_fn = finalize_turn if req.finalize_synchronously else finalize_turn_async

    yield _sse_ping()
    timings["first_token_ms"] = _elapsed_ms(turn_started)
    yield _sse("delta", {"text": text})
    finalize_turn_fn(session_id, user_msg, assistant_msg)
    yield _sse(
        "done",
        _done_payload(
            {"usage": {}, "session_id": session_id, **_domain_done_meta(classification)},
            timings,
            turn_started,
        ),
    )


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
    req: AgentChatRequest,
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
LLM_CHAT_MAX_TOKENS = 8_192
MAX_OUTPUT_CONTINUATION_ROUNDS = 3
DECISION_QUALITY_CHAT_CONTEXT_CHARS = 28_000
DECISION_QUALITY_CHAT_STRUCTURED_MAX_TOKENS = 5_000
DECISION_QUALITY_CHAT_SYNTHESIS_MAX_TOKENS = 3_500


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
PORTFOLIO_SUMMARY_MAX_TOKENS = 2_048
_CASUAL_RX = re.compile(
    r"^\s*(hi|hello|hey|yo|thanks|thank you|cool|ok|okay|who are you|what can you do)[\s!.?]*$",
    flags=re.IGNORECASE,
)
_FRESH_RX = re.compile(
    r"\b(refresh|fresh|latest|current|right now|as of now|up[- ]?to[- ]?date|today)\b",
    flags=re.IGNORECASE,
)
_RETRIEVAL_INTENT_RX = re.compile(
    r"\b(past|previous|earlier|history|conversation|thesis|what did i|what have i|wrote|written)\b",
    flags=re.IGNORECASE,
)
_HEDGE_CONTEXT_RX = re.compile(r"\b(hedge|hedges|hedging|beta|net exposure|gross exposure)\b", flags=re.IGNORECASE)
_SIMPLE_PORTFOLIO_SUMMARY_RX = re.compile(
    r"\b("
    r"summar(?:y|ize|ise)|performance|doing|how\s+is|how\s+are|p&l|pnl|snapshot|update"
    r")\b",
    flags=re.IGNORECASE,
)
_PORTFOLIO_SUMMARY_EXCLUSION_RX = re.compile(
    r"\b("
    r"risk|risks|macro|liquidity|hedge|hedges|hedging|beta|thesis|recommend|recommendation|"
    r"edit|update|replace|change|analyzer|optimizer|sizer|size|sizing|workflow|approval|"
    r"trigger|action item|news|latest|search|sector|exposure|dossier|valuation|dcf|chart|"
    r"technical|compare|versus|vs\.?|why|should|buy|sell|trim|add|exit"
    r")\b",
    flags=re.IGNORECASE,
)
_DATA_SEEKING_RX = re.compile(
    r"\b("
    r"portfolio|holding|position|performance|p&l|pnl|risks?|market|macro|liquidity|breadth|vix|volatility|"
    r"positioning|sentiment|sector|yield|curve|bond|labor|housing|growth|central bank|industry|"
    r"thesis|catalysts?|kill conditions?|dossier|workflow|approvals?|action items?|triggers?|search|news|"
    r"commodity|commodities|country|index|indices|fx|currency|financials|dcf|valuation|chart|"
    r"screener|screen|analyzer|sizer|hedging|workspace"
    r")\b",
    flags=re.IGNORECASE,
)


def _read_llm_api_key() -> tuple[str, str]:
    try:
        provider = selected_provider_for_tier(MODEL_MID)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    try:
        return provider, require_api_key(provider)
    except RuntimeError as exc:
        raise ConfigurationError(str(exc)) from exc


def _get_provider_client(provider: str, api_key: str):
    return get_llm_client(provider, api_key=api_key)


def _format_stream_error(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    raw = str(exc)
    lowered = raw.lower()

    if (
        status_code == 401
        or "invalid x-api-key" in lowered
        or "authentication_error" in lowered
        or "api_key_invalid" in lowered
    ):
        try:
            provider = selected_provider_for_tier(MODEL_MID)
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


def _is_simple_portfolio_summary(user_text: str) -> bool:
    text = (user_text or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if "portfolio" not in lowered and "holdings" not in lowered:
        return False
    if _PORTFOLIO_SUMMARY_EXCLUSION_RX.search(text):
        return False
    return bool(_SIMPLE_PORTFOLIO_SUMMARY_RX.search(text))


def _build_portfolio_summary_prompt(user_text: str, portfolio_result: str) -> str:
    return (
        "The user asked for a simple portfolio performance summary.\n\n"
        "Use only the portfolio tool JSON below. Do not infer facts that are not present. "
        "Keep the answer concise and analytical. Mention the position count, long/short mix, "
        "overall performance fields that are available, what is working, what is not working, "
        "and one or two key observations. Do not recommend trades or portfolio changes.\n\n"
        f"User request:\n{user_text.strip()}\n\n"
        f"Portfolio tool JSON:\n{portfolio_result}"
    )


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _format_pct(value: Any) -> str | None:
    number = _as_float(value)
    if number is None:
        return None
    return f"{number:+.1f}%"


def _portfolio_summary_fallback(portfolio_result: str) -> str:
    """Deterministic fallback so a successful portfolio read never renders blank."""

    try:
        payload = json.loads(portfolio_result)
    except json.JSONDecodeError:
        return "I pulled the portfolio data, but could not synthesize a readable summary from the tool output."

    if not isinstance(payload, dict):
        return "I pulled the portfolio data, but could not synthesize a readable summary from the tool output."
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        return f"I tried to read the portfolio, but the portfolio tool returned an error: {error.strip()}"

    summary_value = payload.get("summary")
    positions_value = payload.get("positions")
    summary: dict[Any, Any] = summary_value if isinstance(summary_value, dict) else {}
    positions: list[Any] = positions_value if isinstance(positions_value, list) else []
    position_count = summary.get("position_count", len(positions))
    long_count = summary.get("long_count")
    short_count = summary.get("short_count")

    book_bits = [f"{position_count} positions"]
    if long_count is not None or short_count is not None:
        book_bits.append(f"{long_count or 0} long / {short_count or 0} short")

    perf_bits: list[str] = []
    for label, key in (
        ("weekly", "weekly_portfolio_return_pct"),
        ("monthly", "monthly_portfolio_return_pct"),
        ("average directional", "average_directional_return_pct"),
    ):
        formatted = _format_pct(summary.get(key))
        if formatted:
            perf_bits.append(f"{label} {formatted}")

    ranked: list[tuple[float, str, str]] = []
    for item in positions:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        for key in ("monthly_contribution_pct", "weekly_contribution_pct", "unrealized_pnl_pct"):
            value = _as_float(item.get(key))
            if value is not None:
                ranked.append((value, ticker, key))
                break

    lines = [f"Portfolio read complete: {', '.join(book_bits)}."]
    if perf_bits:
        lines.append(f"Available aggregate performance: {', '.join(perf_bits)}.")
    if ranked:
        ranked.sort(key=lambda row: row[0], reverse=True)
        best = ", ".join(f"{ticker} {_format_pct(value)}" for value, ticker, _key in ranked[:3])
        weakest_rows = list(reversed(ranked[-3:]))
        weakest = ", ".join(f"{ticker} {_format_pct(value)}" for value, ticker, _key in weakest_rows)
        lines.append(f"Top available contributors: {best}.")
        lines.append(f"Weakest available contributors: {weakest}.")
    return " ".join(lines)


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
    if "management quality" in text or "management team" in text or "owner mindset" in text:
        add("get_portfolio", "get_dossier", "search_knowledge_base")
    if re.search(r"\b(thesis|catalyst|kill condition|dossier|conviction)\b", text):
        add(
            "get_portfolio",
            "get_dossier",
            "get_thesis",
            "get_thesis_evaluations",
            "get_position_valuation",
            "run_chart",
            "get_price_volume_signals",
            "search_knowledge_base",
        )
    if re.search(r"\bcatalysts?\b", text):
        add("get_catalysts")
    if re.search(r"\bcatalysts?\b", text) and re.search(
        r"\b(create|add|stage|propose|generate|build|track|persist|save)\b",
        text,
    ):
        add("propose_catalyst")
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
    if "valuation" in text or "multiple" in text or "multiples" in text:
        add("get_position_valuation")
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
    if "workspace" in text:
        add("get_workspace")

    stop = {"AND", "THE", "FOR", "MY", "ALL", "HOW", "CAN", "ARE", "HAS", "DO", "WHAT", "THIS", "THAT"}
    ticker_candidates = [m for m in _TICKER_RX.findall(user_text or "") if m not in stop and len(m) >= 2]
    if not ticker_candidates:
        for alias, ticker in sorted(_COMPANY_TICKER_ALIASES.items(), key=lambda item: -len(item[0])):
            if re.search(rf"\b{re.escape(alias)}\b", text):
                ticker_candidates = [ticker]
                break
    if ticker_candidates:
        add(
            "get_portfolio",
            "get_dossier",
            "get_thesis",
            "get_financials",
            "get_position_valuation",
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

_DOSSIER_PRESSURE_TEST_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bdossier\s+pressure[- ]?test\b", re.IGNORECASE),
    re.compile(r"\brun\s+(?:a\s+)?pressure[- ]?test\s+(?:on|for)\b", re.IGNORECASE),
    re.compile(r"\bpressure[- ]?test\s+(?:this\s+)?(?:position|thesis|dossier)\b", re.IGNORECASE),
    re.compile(r"\bpoke\s+holes\s+in\s+(?:this\s+)?(?:position|thesis)\b", re.IGNORECASE),
]

_TICKER_RX = re.compile(r"\b([A-Z]{1,5})\b")
_TICKER_STOP_WORDS = {"AND", "THE", "FOR", "MY", "ALL", "HOW", "CAN", "ARE", "HAS", "DO", "WHAT", "THIS", "THAT"}
_COMPANY_TICKER_ALIASES = {
    "meta": "META",
    "facebook": "META",
    "uber": "UBER",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "apple": "AAPL",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "microsoft": "MSFT",
    "netflix": "NFLX",
}
_DECISION_QUALITY_CHAT_INTENT_RX = re.compile(
    r"\b("
    r"thesis|investment thesis|idea|pitch|pressure[- ]?test|what do you think|"
    r"poke holes|devil'?s advocate|conviction|mispricing|invalidation|kill condition|"
    r"should i (?:buy|add|short|sell)|long|short"
    r")\b",
    flags=re.IGNORECASE,
)
_OPPORTUNITY_DISCOVERY_INTENT_RX = re.compile(
    r"\b("
    r"scan|scout|find|discover|look\s+for|interesting|opportunities?|opportunity|"
    r"what\s+(?:should|could)\s+i|names?\s+to|ideas?\s+(?:in|for|on)|"
    r"anything\s+(?:look|worth)|rank|triage|screen|watchlist"
    r")\b",
    flags=re.IGNORECASE,
)
_PASTED_THESIS_TERMS_RX = re.compile(
    r"\b(revenue|margin|valuation|multiple|target price|upside|catalyst|risk|moat|capex|cash flow|fcf)\b",
    flags=re.IGNORECASE,
)


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _extract_candidate_ticker(user_text: str, screen_context: ScreenContextModel | None = None) -> str | None:
    screen_ticker = (screen_context.ticker if screen_context else None) or ""
    if screen_ticker.strip():
        return screen_ticker.strip().upper()
    candidates = [m for m in _TICKER_RX.findall(user_text or "") if m not in _TICKER_STOP_WORDS and len(m) >= 2]
    if candidates:
        return str(candidates[0]).upper()
    lowered = (user_text or "").lower()
    for alias, ticker in sorted(_COMPANY_TICKER_ALIASES.items(), key=lambda item: -len(item[0])):
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return ticker
    return None


def _looks_like_pasted_thesis(user_text: str) -> bool:
    text = (user_text or "").strip()
    return len(text) >= 240 and bool(_PASTED_THESIS_TERMS_RX.search(text))


def _should_run_decision_quality_chat(
    user_text: str,
    screen_context: ScreenContextModel | None = None,
) -> bool:
    if not _env_flag("AGENT_DECISION_QUALITY_CHAT_ENABLED", default=True):
        return False
    text = (user_text or "").strip()
    if not text or not _DECISION_QUALITY_CHAT_INTENT_RX.search(text):
        return False
    return bool(
        _extract_candidate_ticker(text, screen_context)
        or _looks_like_pasted_thesis(text)
        or _should_use_retrieval(text)
    )


def _should_run_opportunity_candidate_preflight(
    user_text: str,
    screen_context: ScreenContextModel | None = None,
) -> bool:
    if not _env_flag("AGENT_OPPORTUNITY_CANDIDATE_PREFLIGHT_ENABLED", default=True):
        return False
    text = (user_text or "").strip()
    if not text:
        return False
    has_ticker = bool(_extract_candidate_ticker(text, screen_context))
    if _OPPORTUNITY_DISCOVERY_INTENT_RX.search(text):
        return True
    if _looks_like_pasted_thesis(text):
        return True
    if _DECISION_QUALITY_CHAT_INTENT_RX.search(text) and (
        has_ticker or _looks_like_pasted_thesis(text) or _should_use_retrieval(text)
    ):
        return True
    if has_ticker and re.search(r"\b(idea|thesis|pitch|stock|name|company)\b", text, re.IGNORECASE):
        return True
    return False


def _decision_quality_chat_tool_calls(
    user_text: str,
    screen_context: ScreenContextModel | None = None,
    *,
    route_decision: RouteDecision | None = None,
    opportunity_candidate_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    screen_payload = screen_context.model_dump(mode="json") if screen_context else None
    pack = resolve_context_pack(
        user_text=user_text,
        intent_class=route_decision.intent_class if route_decision else None,
        tool_pack=route_decision.tool_pack if route_decision else None,
        screen_context=screen_payload,
        opportunity_candidate_metadata=opportunity_candidate_metadata,
    )
    return build_context_pack_tool_calls(
        pack=pack,
        user_text=user_text,
        screen_context=screen_payload,
        allowed_tool_names=_tool_names(),
    )


def _truncate_for_prompt(value: str, *, limit: int) -> str:
    text = value if len(value) <= limit else value[:limit] + "\n...[truncated]"
    return text


def _json_for_prompt(value: Any, *, limit: int = DECISION_QUALITY_CHAT_CONTEXT_CHARS) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, default=str)
    except TypeError:
        text = str(value)
    return _truncate_for_prompt(text, limit=limit)


def _parse_tool_result_for_prompt(result_str: str) -> Any:
    try:
        return json.loads(result_str)
    except Exception:
        return result_str


def _build_decision_quality_chat_context(
    *,
    user_text: str,
    screen_context: ScreenContextModel | None,
    raw_conversation: list[dict[str, object]],
    tool_results: list[dict[str, Any]],
    route_decision: RouteDecision | None = None,
    opportunity_candidate_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    history = raw_conversation[-8:]
    tool_data_quality = aggregate_tool_data_quality(tool_results)
    screen_payload = screen_context.model_dump(mode="json") if screen_context else None
    data_quality = {
        key: tool_data_quality.get(key)
        for key in (
            "critical_data_quality",
            "source_quality",
            "quality",
            "overall_status",
            "tool_errors",
            "blocker_count",
            "warning_count",
            "blocking_reason_codes",
            "price_confirmation_status",
            "source_health_status",
        )
    }
    context_pack = build_context_pack_metadata(
        user_text=user_text,
        intent_class=route_decision.intent_class if route_decision else None,
        tool_pack=route_decision.tool_pack if route_decision else None,
        screen_context=screen_payload,
        opportunity_candidate_metadata=opportunity_candidate_metadata,
        tool_results=tool_results,
        data_quality=data_quality,
    )
    return {
        "user_message": user_text,
        "screen_context": screen_payload,
        "recent_conversation": history,
        "tool_results": tool_results,
        "tool_data_quality": tool_data_quality,
        "data_quality": data_quality,
        "context_pack": context_pack,
    }


def _build_decision_quality_structured_prompt(context_bundle: dict[str, Any]) -> str:
    return (
        "Pressure-test this live Stan chat investment idea using only the supplied context. "
        "Return exactly one DecisionQuality JSON object, not a wrapper and not markdown. "
        "If current price action, portfolio sizing, catalyst verification, or source data is missing, "
        "surface that in actionability.missing_inputs and price_action_read.data_needed instead of inventing it. "
        "Treat tool_data_quality and data_quality as binding: stale, blocked, missing, or unconfirmed price "
        "sources must keep recommended_action at watch or research and must not imply an actionable trade. "
        "Use watch or research when the evidence is not actionable yet.\n\n"
        f"Live chat context:\n{_json_for_prompt(context_bundle)}"
    )


def _run_decision_quality_structured_pass(
    *,
    context_bundle: dict[str, Any],
    provider: str,
    api_key: str,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    parsed, citations, response, diagnostics = call_llm_json(
        prompt=_build_decision_quality_structured_prompt(context_bundle),
        model=MODEL_MID,
        api_key=api_key,
        max_tokens=DECISION_QUALITY_CHAT_STRUCTURED_MAX_TOKENS,
        system=_load_required_prompt_file("decision_quality.md"),
        provider=provider,
        enable_web_search=False,
        reasoning_effort=reasoning_effort,
        json_schema=decision_quality_schema(),
        json_schema_name="decision_quality_chat",
    )
    raw_decision_quality = (
        parsed.get("decision_quality") if isinstance(parsed, dict) and "decision_quality" in parsed else parsed
    )
    decision_quality, parse_errors = parse_decision_quality(raw_decision_quality)
    data_quality = context_bundle.get("data_quality") if isinstance(context_bundle.get("data_quality"), dict) else None
    gate = apply_decision_quality_gates(
        decision_quality,
        current_action=decision_quality.recommended_action if decision_quality else "watch",
        recommendation_status="clear",
        data_quality=data_quality,
        parse_errors=parse_errors,
    )
    return {
        "decision_quality": decision_quality,
        "parse_errors": parse_errors,
        "gate": gate,
        "data_quality": data_quality,
        "tool_data_quality": context_bundle.get("tool_data_quality"),
        "raw": parsed,
        "citations": [{"title": title, "url": url} for title, url in citations],
        "usage": _usage_dict(response),
        "diagnostics": diagnostics,
    }


def _build_decision_quality_chat_synthesis_prompt(
    *,
    user_text: str,
    context_bundle: dict[str, Any],
    dq_result: dict[str, Any],
) -> str:
    decision_quality = dq_result.get("decision_quality")
    gate = dq_result.get("gate")
    dq_payload = decision_quality.model_dump(mode="json") if isinstance(decision_quality, DecisionQuality) else None
    gate_payload = gate.model_dump(mode="json") if isinstance(gate, DecisionQualityGate) else None
    return (
        "The user asked Stan to pressure-test an investment idea.\n\n"
        "The DecisionQuality object and gate result below are private working state. "
        "The gate result is binding for the final stance: if final_action is watch, research, avoid, or do_nothing, "
        "the answer must not sound like a confident buy/add/short. "
        "If tool_data_quality or data_quality shows stale, blocked, missing, or unconfirmed price sources, "
        "surface those as missing inputs or blockers instead of burying them in prose.\n\n"
        f"User request:\n{user_text.strip()}\n\n"
        f"Context bundle:\n{_json_for_prompt(context_bundle)}\n\n"
        f"DecisionQuality:\n{_json_for_prompt(dq_payload)}\n\n"
        f"Gate:\n{_json_for_prompt(gate_payload)}\n\n"
        f"Parse errors, if any:\n{_json_for_prompt(dq_result.get('parse_errors') or [])}"
    )


def _decision_quality_chat_done_meta(dq_result: dict[str, Any] | None) -> dict[str, Any]:
    if not dq_result:
        return {"ran": False}
    decision_quality = dq_result.get("decision_quality")
    gate = dq_result.get("gate")
    missing_inputs_count = 0
    confidence = None
    if isinstance(decision_quality, DecisionQuality):
        missing_inputs_count = len(decision_quality.actionability.missing_inputs)
        missing_inputs_count += len(decision_quality.price_action_read.data_needed)
        confidence = decision_quality.confidence
    data_quality = dq_result.get("data_quality") if isinstance(dq_result.get("data_quality"), dict) else {}
    meta: dict[str, Any] = {
        "ran": True,
        "gate_status": gate.status if isinstance(gate, DecisionQualityGate) else "invalid",
        "final_action": gate.final_action if isinstance(gate, DecisionQualityGate) else "watch",
        "confidence": confidence,
        "missing_inputs_count": missing_inputs_count,
    }
    if data_quality:
        meta["tool_quality"] = {
            "blocker_count": data_quality.get("blocker_count", 0),
            "warning_count": data_quality.get("warning_count", 0),
            "blocking_reason_codes": list(data_quality.get("blocking_reason_codes") or []),
            "price_confirmation_status": data_quality.get("price_confirmation_status"),
            "source_health_status": data_quality.get("source_health_status"),
            "critical_data_quality": data_quality.get("critical_data_quality"),
        }
    return meta


def _build_opportunity_candidate_structured_prompt(context_bundle: dict[str, Any]) -> str:
    return (
        "Triage this live Stan chat opportunity using only the supplied context. "
        "Return exactly one OpportunityCandidate JSON object, not a wrapper and not markdown. "
        "This is a pre-decision triage pass only: do not recommend buy, add, short, sell, trim, reduce, "
        "exit, hedge, or rebalance. Use graduate_to_decision_quality only when a full pressure-test is warranted. "
        "Treat context_pack, tool_data_quality, and data_quality as binding: if the selected context pack is "
        "incomplete or required tools are missing/stale, keep next_action at research/watch and list the gaps "
        "in missing_inputs instead of inventing them.\n\n"
        f"Live chat context:\n{_json_for_prompt(context_bundle)}"
    )


def _run_opportunity_candidate_structured_pass(
    *,
    context_bundle: dict[str, Any],
    provider: str,
    api_key: str,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    parsed, citations, response, diagnostics = call_llm_json(
        prompt=_build_opportunity_candidate_structured_prompt(context_bundle),
        model=MODEL_MID,
        api_key=api_key,
        max_tokens=DECISION_QUALITY_CHAT_STRUCTURED_MAX_TOKENS,
        system=_load_required_prompt_file("opportunity_candidate.md"),
        provider=provider,
        enable_web_search=False,
        reasoning_effort=reasoning_effort,
        json_schema=opportunity_candidate_schema(),
        json_schema_name="opportunity_candidate_chat",
    )
    raw_candidate = (
        parsed.get("opportunity_candidate")
        if isinstance(parsed, dict) and "opportunity_candidate" in parsed
        else parsed
    )
    opportunity_candidate, parse_errors = parse_opportunity_candidate(raw_candidate)
    context_pack = context_bundle.get("context_pack") if isinstance(context_bundle.get("context_pack"), dict) else None
    data_quality = context_bundle.get("data_quality") if isinstance(context_bundle.get("data_quality"), dict) else None
    gate = apply_opportunity_candidate_gates(
        opportunity_candidate,
        parse_errors=parse_errors,
        context_pack=context_pack,
        data_quality=data_quality,
    )
    return {
        "opportunity_candidate": opportunity_candidate,
        "parse_errors": parse_errors,
        "gate": gate,
        "context_pack": context_pack,
        "raw": parsed,
        "citations": [{"title": title, "url": url} for title, url in citations],
        "usage": _usage_dict(response),
        "diagnostics": diagnostics,
    }


def _build_opportunity_candidate_synthesis_prompt(
    *,
    user_text: str,
    context_bundle: dict[str, Any],
    oc_result: dict[str, Any],
) -> str:
    opportunity_candidate = oc_result.get("opportunity_candidate")
    gate = oc_result.get("gate")
    candidate_payload = (
        opportunity_candidate.model_dump(mode="json")
        if isinstance(opportunity_candidate, OpportunityCandidate)
        else None
    )
    gate_payload = gate.model_dump(mode="json") if isinstance(gate, OpportunityCandidateGate) else None
    return (
        "The user asked Stan to triage a possible investment opportunity.\n\n"
        "The OpportunityCandidate object and gate result below are private working state. "
        "The gate result is binding for the final triage stance: if final_action is watch, research, avoid, "
        "or do_nothing, the answer must not sound like a confident buy/add/short recommendation.\n\n"
        f"User request:\n{user_text.strip()}\n\n"
        f"Context bundle:\n{_json_for_prompt(context_bundle)}\n\n"
        f"OpportunityCandidate:\n{_json_for_prompt(candidate_payload)}\n\n"
        f"Gate:\n{_json_for_prompt(gate_payload)}\n\n"
        f"Parse errors, if any:\n{_json_for_prompt(oc_result.get('parse_errors') or [])}"
    )


def _opportunity_candidate_done_meta(oc_result: dict[str, Any] | None) -> dict[str, Any]:
    if not oc_result:
        return {"ran": False}
    opportunity_candidate = oc_result.get("opportunity_candidate")
    gate = oc_result.get("gate")
    missing_inputs_count = 0
    if isinstance(opportunity_candidate, OpportunityCandidate):
        missing_inputs_count = len(opportunity_candidate.missing_inputs)
    return {
        "ran": True,
        "gate_status": gate.status if isinstance(gate, OpportunityCandidateGate) else "invalid",
        "final_action": gate.final_action if isinstance(gate, OpportunityCandidateGate) else "research",
        "should_graduate": gate.should_graduate if isinstance(gate, OpportunityCandidateGate) else False,
        "missing_inputs_count": missing_inputs_count,
        "context_pack": oc_result.get("context_pack") if isinstance(oc_result.get("context_pack"), dict) else None,
    }


def _opportunity_candidate_fallback(oc_result: dict[str, Any]) -> str:
    opportunity_candidate = oc_result.get("opportunity_candidate")
    gate = oc_result.get("gate")
    if not isinstance(opportunity_candidate, OpportunityCandidate):
        return (
            "I cannot triage this cleanly yet. The opportunity pass did not produce a valid candidate object, "
            "so the right next step is research: get the trigger, current thesis source, price action, and "
            "missing inputs before treating it as actionable."
        )
    final_action = (
        gate.final_action if isinstance(gate, OpportunityCandidateGate) else opportunity_candidate.next_action
    )
    missing_text = (
        "; ".join(opportunity_candidate.missing_inputs[:4]) if opportunity_candidate.missing_inputs else "none flagged"
    )
    summary = opportunity_candidate.summary or opportunity_candidate.variant_view or opportunity_candidate.trigger
    return (
        f"Bottom line: I would treat this as {final_action}. "
        f"The trigger is {opportunity_candidate.trigger}. "
        f"Why now: {opportunity_candidate.why_now or 'not established yet'}. "
        f"Consensus: {opportunity_candidate.consensus or 'not established yet'}. "
        f"Variant view: {opportunity_candidate.variant_view or 'not established yet'}. "
        f"Price confirmation: {opportunity_candidate.price_confirmation or 'needs more work'}. "
        f"Missing inputs: {missing_text}. "
        f"Summary: {summary}."
    )


def _decision_quality_chat_fallback(dq_result: dict[str, Any]) -> str:
    decision_quality = dq_result.get("decision_quality")
    gate = dq_result.get("gate")
    if not isinstance(decision_quality, DecisionQuality):
        return (
            "I cannot pressure-test this cleanly yet. The thesis pass did not produce a valid decision-quality "
            "object, so the right next step is research: get the current thesis source, price action, catalyst "
            "status, invalidation threshold, and portfolio sizing context before making it actionable."
        )
    final_action = gate.final_action if isinstance(gate, DecisionQualityGate) else decision_quality.recommended_action
    missing = decision_quality.actionability.missing_inputs + decision_quality.price_action_read.data_needed
    missing_text = "; ".join(missing[:4]) if missing else "none flagged"
    return (
        f"Bottom line: I would treat this as {final_action}, not a lazy buy call. "
        f"The thesis is: {decision_quality.simple_thesis} "
        f"The biggest issue is {decision_quality.evidence_against[0].claim if decision_quality.evidence_against else decision_quality.confidence_reason}. "
        f"Reason-now: {decision_quality.catalyst_or_reason_now.event_or_condition} over "
        f"{decision_quality.catalyst_or_reason_now.expected_timeframe}. "
        f"Price action: {decision_quality.price_action_read.observed_behavior or 'needs a current chart read'}. "
        f"Invalidation: {decision_quality.invalidation.metric_or_event} at {decision_quality.invalidation.threshold} "
        f"within {decision_quality.invalidation.timeframe}. Missing inputs: {missing_text}. "
        f"Size only within the stated risk budget: {decision_quality.sizing_context.add_conditions}. "
        f"If right: {decision_quality.trade_after_trade.if_right} If wrong: {decision_quality.trade_after_trade.if_wrong} "
        f"Review on: {decision_quality.trade_after_trade.next_review_trigger}."
    )


def _detect_workflow(
    user_text: str,
    screen_context: ScreenContextModel | None = None,
) -> tuple[str | None, str | None]:
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

    screen_ticker = (screen_context.ticker if screen_context else None) or ""
    on_position_dossier = bool(screen_context and screen_context.page_name == "Position Dossier")
    if on_position_dossier:
        for pattern in _DOSSIER_PRESSURE_TEST_PATTERNS:
            if pattern.search(text):
                ticker = screen_ticker.strip().upper() or _extract_candidate_ticker(text, screen_context)
                if ticker:
                    return "position_dossier_pressure_test", ticker

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
                ticker = candidates[0] if candidates else screen_ticker.strip().upper() or None
            return wf_name, ticker
    return None, None


def _resolve_chat_route(
    *,
    user_text: str,
    screen_context: ScreenContextModel | None,
    recent_conversation: list[dict[str, Any]] | None = None,
    opportunity_candidate_metadata: dict[str, Any] | None = None,
    llm_credentials: tuple[str, str] | None = None,
    reasoning_effort: str | None = None,
) -> tuple[RouteDecision, dict[str, Any]]:
    regex_baseline = build_regex_route_decision(
        user_text=user_text,
        select_tool_names=_select_tool_names,
        detect_workflow=_detect_workflow,
        should_run_hidden_dq=_should_run_decision_quality_chat,
        should_run_opportunity_preflight=_should_run_opportunity_candidate_preflight,
        screen_context=screen_context,
    )
    context = build_route_context(
        user_text=user_text,
        screen_context=screen_context,
        recent_conversation=recent_conversation,
        opportunity_candidate_metadata=opportunity_candidate_metadata,
        allowed_tool_names=list(_tool_names()),
        workflow_hints=list(AVAILABLE_WORKFLOWS.keys()),
    )
    system_prompt: str | None = None
    try:
        system_prompt = _load_required_prompt_file("intent_router.md")
    except ConfigurationError:
        system_prompt = None
    provider, api_key = llm_credentials if llm_credentials else (None, None)
    return resolve_agent_route(
        context=context,
        regex_baseline=regex_baseline,
        provider=provider,
        api_key=api_key,
        system_prompt=system_prompt,
        reasoning_effort=reasoning_effort,
    )


def _load_recent_conversation_for_routing(session_id: str | None) -> list[dict[str, Any]]:
    from api import memory_db

    session = memory_db.get_or_create_session(session_id)
    server_messages = session.get("server_messages") or []
    if not isinstance(server_messages, list):
        return []
    return [item for item in server_messages[-6:] if isinstance(item, dict)]


def _capture_intent_router_training_row(
    *,
    req: AgentChatRequest,
    route_decision: RouteDecision,
    route_meta: dict[str, Any],
    recent_conversation: list[dict[str, Any]] | None = None,
    opportunity_candidate_metadata: dict[str, Any] | None = None,
) -> None:
    should_capture, sampling_reason = should_capture_training_row(route_meta=route_meta)
    if not should_capture:
        return
    try:
        from api.intent_router_training_store import insert_training_row

        route_context = build_route_context(
            user_text=req.message,
            screen_context=req.screen_context,
            recent_conversation=recent_conversation,
            opportunity_candidate_metadata=opportunity_candidate_metadata,
        )
        screen_context = route_context.screen_context
        row = training_row_from_telemetry(
            user_text=req.message,
            route_meta=route_meta,
            session_id=req.session_id,
            client_turn_id=req.client_turn_id,
            screen_context=screen_context,
            recent_session_features=route_context.recent_session_features,
            applied_route=route_decision.to_meta(),
            opportunity_candidate_metadata=opportunity_candidate_metadata,
            capture_policy="mismatch_only" if intent_router_training_capture_mismatch_only() else "shadow_all",
            sampling_reason=sampling_reason,
        )
        insert_training_row(row)
    except Exception:
        logger.exception("intent_router_training_capture_failed session_id=%s", req.session_id)


def _opportunity_candidate_metadata_from_result(oc_result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not oc_result:
        return None
    opportunity_candidate = oc_result.get("opportunity_candidate")
    if isinstance(opportunity_candidate, OpportunityCandidate):
        return opportunity_candidate.model_dump(mode="json")
    if isinstance(opportunity_candidate, dict):
        return opportunity_candidate
    return None


def _update_intent_router_training_oc_metadata(
    *,
    session_id: str | None,
    client_turn_id: str | None,
    oc_result: dict[str, Any] | None,
) -> None:
    metadata = _opportunity_candidate_metadata_from_result(oc_result)
    if not metadata:
        return
    try:
        from api.intent_router_training_store import update_opportunity_candidate_metadata

        update_opportunity_candidate_metadata(
            session_id=session_id,
            client_turn_id=client_turn_id,
            opportunity_candidate_metadata=metadata,
        )
    except Exception:
        logger.exception(
            "intent_router_training_oc_metadata_failed session_id=%s client_turn_id=%s",
            session_id,
            client_turn_id,
        )


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
    domain_classification: AgentDomainClassification | None = None,
) -> list[tuple[dict, str, float]]:
    """Execute deduplicated tool calls in parallel and measure runtime."""
    if len(calls) == 1:
        c = calls[0]
        started = time.perf_counter()
        result = _execute_tool_for_actor(c["name"], c["args"], actor, domain_classification=domain_classification)
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
                domain_classification,
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
    domain_classification: AgentDomainClassification | None = None,
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
                domain_classification,
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
    domain_classification: AgentDomainClassification | None = None,
) -> str:
    if domain_classification is not None and domain_classification.decision != "allow":
        return _blocked_tool_result(
            name,
            RuntimeError(f"Agent domain guardrail blocked tool execution: {domain_classification.reason}"),
        )
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


def _blocked_model_egress_payload(exc: ModelGatewayDenied) -> dict[str, Any]:
    manifest = exc.manifest
    decision_id = str(manifest.get("policy_decision_id") or "model_egress")
    return {
        "name": "model_egress",
        "id": decision_id,
        "status": "blocked",
        "message": str(exc),
        "policy_decision_id": decision_id,
        "decision": manifest.get("decision"),
        "decision_reason": manifest.get("decision_reason"),
        "data_sensitivity": manifest.get("data_sensitivity"),
        "provider": manifest.get("provider"),
        "model": manifest.get("model"),
    }


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


def _serialize_gemini_response_parts(response: object) -> list[dict]:
    parts: list[dict] = []
    for candidate in _obj_list(response, "candidates"):
        content = _obj_value(candidate, "content", {})
        for part in _obj_list(content, "parts"):
            serialized = _serialize_gemini_part(part)
            if serialized:
                parts.append(serialized)
    return parts


def _serialize_gemini_part(part: object) -> dict:
    if isinstance(part, dict):
        return dict(part)
    to_json_dict = getattr(part, "to_json_dict", None)
    if callable(to_json_dict):
        try:
            value = to_json_dict()
            if isinstance(value, dict):
                return value
        except Exception:
            pass

    out: dict[str, object] = {}
    text = _obj_value(part, "text")
    if isinstance(text, str) and text:
        out["text"] = text
    thought = _obj_value(part, "thought")
    if isinstance(thought, bool):
        out["thought"] = thought
    function_call = _obj_value(part, "function_call", _obj_value(part, "functionCall"))
    if function_call:
        out["function_call"] = _serialize_gemini_function_call(function_call)
    function_response = _obj_value(part, "function_response", _obj_value(part, "functionResponse"))
    if function_response:
        out["function_response"] = function_response
    return out


def _serialize_gemini_function_call(function_call: object) -> dict:
    if isinstance(function_call, dict):
        return dict(function_call)
    to_json_dict = getattr(function_call, "to_json_dict", None)
    if callable(to_json_dict):
        try:
            value = to_json_dict()
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    return {
        key: value
        for key, value in {
            "id": _obj_value(function_call, "id"),
            "name": _obj_value(function_call, "name"),
            "args": _obj_value(function_call, "args"),
        }.items()
        if value is not None
    }


def _extract_gemini_tool_calls(parts: list[dict]) -> list[dict]:
    calls: list[dict] = []
    for index, part in enumerate(parts):
        function_call = part.get("function_call") or part.get("functionCall")
        if not isinstance(function_call, dict):
            continue
        name = function_call.get("name")
        raw_args = function_call.get("args") or function_call.get("arguments") or {}
        args = raw_args if isinstance(raw_args, dict) else {}
        call_id = function_call.get("id") or function_call.get("call_id") or f"gemini:{name}:{index}"
        if isinstance(name, str) and isinstance(call_id, str):
            calls.append({"name": name, "call_id": call_id, "args": args})
    return calls


def _gemini_aggregate_response(parts: list[dict], usage_source: object | None) -> dict:
    usage = (
        _obj_value(usage_source, "usage_metadata", _obj_value(usage_source, "usageMetadata")) if usage_source else None
    )
    return {
        "candidates": [{"content": {"parts": parts}}],
        "usage_metadata": usage,
    }


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


def _dict_value(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


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
        elif provider == PROVIDER_GEMINI:
            tools.append(
                {
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

    if provider == PROVIDER_GEMINI:
        resolved_model = resolve_model(MODEL_MID, PROVIDER_GEMINI)
        config: dict[str, object] = {
            "max_output_tokens": max_tokens,
            "system_instruction": instructions,
        }
        kwargs = {
            "model": resolved_model,
            "contents": conversation,
            "config": config,
        }
        apply_reasoning_config(
            kwargs,
            provider=PROVIDER_GEMINI,
            model=resolved_model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        config = _dict_value(kwargs.get("config"))
        if tool_defs:
            config["tools"] = [{"function_declarations": tool_defs}]
            config["tool_config"] = {
                "function_calling_config": {
                    "mode": "ANY" if force_tool_use else "AUTO",
                }
            }
        kwargs["config"] = config
        return kwargs

    resolved_model = resolve_model(MODEL_MID, provider)
    kwargs = {
        "model": resolved_model,
        "max_output_tokens": max_tokens,
        "instructions": instructions,
        "input": conversation,
    }
    apply_reasoning_config(
        kwargs,
        provider=provider,
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


def _gemini_role(role: object) -> str:
    return "model" if role == "assistant" else "user"


def _gemini_text_content(role: object, text: str) -> dict:
    return {"role": _gemini_role(role), "parts": [{"text": text}]}


def _initial_conversation(provider: str, messages: list[ChatMessage]) -> list[dict]:
    if provider == PROVIDER_ANTHROPIC:
        return [{"role": m.role, "content": m.content} for m in messages]
    if provider == PROVIDER_GEMINI:
        return [_gemini_text_content(m.role, m.content) for m in messages]
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


def _gemini_conversation_from_context(conversation: list[dict[str, object]]) -> list[dict]:
    out: list[dict] = []
    for msg in conversation:
        role = msg.get("role")
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append(_gemini_text_content(role, content))
        elif isinstance(content, list):
            out.append({"role": _gemini_role(role), "parts": content})
        else:
            out.append(_gemini_text_content(role, str(content)))
    return out


def _openai_user_prompt(prompt: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]


def _gemini_user_prompt(prompt: str) -> list[dict]:
    return [{"role": "user", "parts": [{"text": prompt}]}]


def _user_prompt_for_provider(provider: str, prompt: str) -> list[dict]:
    if provider == PROVIDER_ANTHROPIC:
        return [{"role": "user", "content": prompt}]
    if provider == PROVIDER_GEMINI:
        return _gemini_user_prompt(prompt)
    return _openai_user_prompt(prompt)


def _stream_llm_response(
    client: Any,
    provider: str,
    stream_kwargs: dict[str, object],
    text_parts: list[str] | None = None,
    *,
    model_timing: dict[str, Any] | None = None,
    turn_timings: dict[str, Any] | None = None,
    turn_started: float | None = None,
):
    model_started = time.perf_counter()

    def record_first_token() -> None:
        if model_timing is not None and model_timing.get("first_token_ms") is None:
            model_timing["first_token_ms"] = _elapsed_ms(model_started)
        if turn_timings is not None and turn_started is not None and turn_timings.get("first_token_ms") is None:
            turn_timings["first_token_ms"] = _elapsed_ms(turn_started)

    def emit_final_text_if_missing(final_message: object | None):
        if text_parts is None or any(part.strip() for part in text_parts):
            return
        final_text = extract_text(final_message).strip()
        if not final_text:
            return
        record_first_token()
        text_parts.append(final_text)
        yield _sse("delta", {"text": final_text})

    if provider == PROVIDER_ANTHROPIC:
        with client.messages.stream(**stream_kwargs) as stream:
            for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    record_first_token()
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
            final_message = stream.get_final_message()
            yield from emit_final_text_if_missing(final_message)
            return final_message

    if provider == PROVIDER_GEMINI:
        gemini_emitted_call_ids: set[str] = set()
        aggregate_parts: list[dict] = []
        last_chunk: object | None = None
        for chunk in client.models.generate_content_stream(**stream_kwargs):
            last_chunk = chunk
            for part in _serialize_gemini_response_parts(chunk):
                aggregate_parts.append(part)
                function_call = part.get("function_call") or part.get("functionCall")
                if isinstance(function_call, dict):
                    name = function_call.get("name")
                    call_id = (
                        function_call.get("id")
                        or function_call.get("call_id")
                        or f"gemini:{name}:{len(aggregate_parts)}"
                    )
                    if "function_call" in part:
                        part["function_call"] = {**function_call, "id": call_id}
                    elif "functionCall" in part:
                        part["functionCall"] = {**function_call, "id": call_id}
                    if isinstance(name, str) and isinstance(call_id, str) and call_id not in gemini_emitted_call_ids:
                        gemini_emitted_call_ids.add(call_id)
                        yield _sse("tool_call", {"name": name, "id": call_id})
                    continue
                if part.get("thought") is True:
                    continue
                delta = part.get("text")
                if isinstance(delta, str) and delta:
                    record_first_token()
                    if text_parts is not None:
                        text_parts.append(delta)
                    yield _sse("delta", {"text": delta})
        final_response = _gemini_aggregate_response(aggregate_parts, last_chunk)
        yield from emit_final_text_if_missing(final_response)
        return final_response

    emitted_call_ids: set[str] = set()
    with client.responses.stream(**stream_kwargs) as stream:
        for event in stream:
            event_type = _obj_value(event, "type")
            if event_type == "response.output_text.delta":
                delta = _obj_value(event, "delta", "")
                if isinstance(delta, str) and delta:
                    record_first_token()
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
            final_response = get_final()
            yield from emit_final_text_if_missing(final_response)
            return final_response
        get_final = getattr(stream, "get_final_message", None)
        if callable(get_final):
            final_message = get_final()
            yield from emit_final_text_if_missing(final_message)
            return final_message
        return None


def _usage_dict(message: object) -> dict:
    usage = _obj_value(message, "usage") or _obj_value(message, "usage_metadata", _obj_value(message, "usageMetadata"))
    if not usage:
        return {}
    input_tokens = _obj_value(
        usage,
        "input_tokens",
        _obj_value(usage, "prompt_tokens", _obj_value(usage, "prompt_token_count", None)),
    )
    output_tokens = _obj_value(
        usage,
        "output_tokens",
        _obj_value(usage, "completion_tokens", _obj_value(usage, "candidates_token_count", None)),
    )
    out: dict[str, int] = {}
    if isinstance(input_tokens, int):
        out["input_tokens"] = input_tokens
    if isinstance(output_tokens, int):
        out["output_tokens"] = output_tokens
    return out


def _response_stop_reason(provider: str, message: object | None) -> str:
    if message is None:
        return ""
    if provider == PROVIDER_ANTHROPIC:
        return str(_obj_value(message, "stop_reason") or "")
    if provider == PROVIDER_GEMINI:
        reasons = []
        for candidate in _obj_list(message, "candidates"):
            reason = _obj_value(candidate, "finish_reason", _obj_value(candidate, "finishReason"))
            if reason:
                reasons.append(str(reason))
        return ",".join(reasons)

    incomplete = _obj_value(message, "incomplete_details") or {}
    reason = _obj_value(incomplete, "reason")
    if reason:
        return str(reason)
    return str(_obj_value(message, "status") or "")


def _hit_output_token_limit(provider: str, message: object | None) -> bool:
    reason = _response_stop_reason(provider, message).strip().lower()
    return reason in {"max_tokens", "max_output_tokens"} or "max_token" in reason


def _append_output_continuation_request(
    provider: str,
    conversation: list[dict],
    assistant_content: list[dict],
) -> None:
    prompt = (
        "Continue exactly from where the previous assistant response stopped. "
        "Do not repeat earlier text, do not call tools, and finish the answer."
    )
    if provider == PROVIDER_ANTHROPIC:
        conversation.append({"role": "assistant", "content": assistant_content})
        conversation.append({"role": "user", "content": prompt})
    elif provider == PROVIDER_GEMINI:
        conversation.append({"role": "model", "parts": assistant_content})
        conversation.append(_gemini_text_content("user", prompt))
    else:
        conversation.extend(assistant_content)
        conversation.extend(_openai_user_prompt(prompt))


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
            session_id or "session",
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
        config = _dict_value(stream_kwargs.get("config"))
        conversation = stream_kwargs.get("messages") or stream_kwargs.get("input") or stream_kwargs.get("contents")
        tools = stream_kwargs.get("tools") or config.get("tools")
        event_id = provenance.deterministic_id(
            "pv:model_call",
            session_id or workflow_run_id or "session",
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
                "instructions": stream_kwargs.get("instructions")
                or stream_kwargs.get("system")
                or config.get("system_instruction"),
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
                "max_tokens": stream_kwargs.get("max_tokens")
                or stream_kwargs.get("max_output_tokens")
                or config.get("max_output_tokens"),
                "tool_choice": stream_kwargs.get("tool_choice") or config.get("tool_config"),
                "reasoning": stream_kwargs.get("reasoning") or config.get("thinking_config"),
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


def _record_model_timing(
    timings: dict[str, Any],
    model_timing: dict[str, Any],
    *,
    started: float,
    status: str,
    provider: str,
    model: object,
) -> None:
    model_timing["duration_ms"] = _elapsed_ms(started)
    model_timing["status"] = status
    models = timings.setdefault("models", [])
    if isinstance(models, list):
        models.append(dict(model_timing))
    logger.info(
        "agent_chat_model_call phase=%s purpose=%s attempt=%s round=%s provider=%s model=%s duration_ms=%.1f first_token_ms=%s status=%s",
        model_timing.get("phase"),
        model_timing.get("purpose"),
        model_timing.get("attempt"),
        model_timing.get("round_index"),
        provider,
        model,
        float(model_timing["duration_ms"]),
        model_timing.get("first_token_ms"),
        status,
    )


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
# Endpoint — server-managed rolling memory
# ---------------------------------------------------------------------------


@router.post("/agent/chat/async")
def start_agent_chat_async(req: AgentChatRequest, actor: ActorDep):
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


@router.post("/agent/chat")
def agent_chat(req: AgentChatRequest, actor: ActorDep):
    tool_actor = agent_actor(actor)
    """Chat endpoint with server-managed conversation memory.

    The frontend sends only the new message + session_id.  The server
    assembles optimal context from a rolling summary, verbatim window,
    and retrieval hits.
    """
    domain_classification = analyze_agent_domain(req.message, screen_context=req.screen_context)
    if domain_classification.decision != "allow":
        return StreamingResponse(
            _domain_guardrail_stream_v2(req, domain_classification),
            media_type="text/event-stream",
            headers=_sse_headers(),
        )

    casual = _is_casual(req.message)
    llm_credentials: tuple[str, str] | None = None
    route_decision = build_regex_route_decision(
        user_text=req.message,
        select_tool_names=_select_tool_names,
        detect_workflow=_detect_workflow,
        should_run_hidden_dq=_should_run_decision_quality_chat,
        should_run_opportunity_preflight=_should_run_opportunity_candidate_preflight,
        screen_context=req.screen_context,
    )
    route_meta: dict[str, Any] = {
        "enabled": False,
        "applied_source": "regex",
        "regex_baseline": route_decision.to_meta(),
    }
    workflow_name = route_decision.workflow_name
    workflow_ticker = route_decision.workflow_ticker
    if workflow_name and req.allow_workflow_handoff:
        provider_label = "deferred"
        active_tool_names: list[str] = []
        tool_defs: list[dict] = []
    else:
        try:
            provider = selected_provider_for_tier(MODEL_MID)
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        provider_label = provider
        active_tool_names = [] if casual else list(route_decision.tool_names)
        tool_defs = _tool_definitions_from_names(provider, active_tool_names)
        if not casual:
            llm_credentials = _read_llm_api_key()
            recent_conversation = _load_recent_conversation_for_routing(req.session_id)
            route_decision, route_meta = _resolve_chat_route(
                user_text=req.message,
                screen_context=req.screen_context,
                recent_conversation=recent_conversation,
                llm_credentials=llm_credentials,
                reasoning_effort=_chat_reasoning_effort(provider, req.response_preferences),
            )
            _capture_intent_router_training_row(
                req=req,
                route_decision=route_decision,
                route_meta=route_meta,
                recent_conversation=recent_conversation,
            )
            workflow_name = route_decision.workflow_name
            workflow_ticker = route_decision.workflow_ticker
            if not (workflow_name and req.allow_workflow_handoff):
                active_tool_names = list(route_decision.tool_names)
                tool_defs = _tool_definitions_from_names(provider, active_tool_names)
    force_refresh = _wants_fresh_data(req.message)
    enable_retrieval = _should_use_retrieval(req.message)
    logger.info(
        "agent_chat provider=%s casual=%s workflow=%s ticker=%s tools=%d refresh=%s retrieval=%s session=%s "
        "route_source=%s route_intent=%s route_confidence=%.2f",
        provider_label,
        casual,
        workflow_name,
        workflow_ticker,
        len(tool_defs),
        force_refresh,
        enable_retrieval,
        req.session_id,
        route_meta.get("applied_source", route_decision.source),
        route_decision.intent_class,
        route_decision.confidence,
    )

    def generate():  # noqa: C901
        nonlocal tool_defs
        turn_started = time.perf_counter()
        timings = _new_agent_timings()
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
                provider=selected_provider_for_tier(MODEL_MID),
                actor=tool_actor,
            )
            text = _casual_response(req.message, req.response_preferences)
            timings["first_token_ms"] = _elapsed_ms(turn_started)
            yield _sse("delta", {"text": text})
            turn_meta = {"client_turn_id": req.client_turn_id} if req.client_turn_id else {}
            user_msg = {"role": "user", "content": req.message, "timestamp": time.time(), **turn_meta}
            assistant_msg = {"role": "assistant", "content": text, "timestamp": time.time(), **turn_meta}
            yield _phase_sse("finalizing", turn_started)
            finalize_turn_fn(session_id, user_msg, assistant_msg)
            _finish_agent_turn_provenance(
                agent_turn_event_id,
                status="succeeded",
                output_value=text,
                usage={},
            )
            yield _sse("done", _done_payload({"usage": {}, "session_id": session_id}, timings, turn_started))
            return

        provider, api_key = llm_credentials or _read_llm_api_key()
        reasoning_effort = _chat_reasoning_effort(provider, req.response_preferences)
        instructions = _with_domain_guardrail_instruction(
            _with_response_preferences(
                _build_agent_instructions(screen_context=req.screen_context),
                req.response_preferences,
            ),
            domain_classification,
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
        if provider == PROVIDER_ANTHROPIC:
            conversation = raw_conversation
        elif provider == PROVIDER_GEMINI:
            conversation = _gemini_conversation_from_context(raw_conversation)
        else:
            conversation = _openai_conversation_from_context(raw_conversation)

        if not workflow_name and _is_simple_portfolio_summary(req.message):
            call_info = {
                "name": "get_portfolio",
                "args": _execution_args("get_portfolio", {}, force_refresh=force_refresh, user_text=req.message),
                "call_ids": ["portfolio-summary:get_portfolio"],
            }
            yield _sse("tool_call", {"name": call_info["name"], "id": call_info["call_ids"][0]})
            yield _phase_sse(
                "tool_running",
                turn_started,
                label="Reading portfolio...",
                tool_names=[call_info["name"]],
                round_index=0,
            )
            yield _sse(
                "tool_progress",
                {"name": call_info["name"], "id": call_info["call_ids"][0], "status": "running"},
            )
            _attach_tool_provenance_context(
                [call_info],
                parent_event_id=agent_turn_event_id,
                session_id=session_id,
                workflow_run_id=None,
                source="agent.chat.portfolio_summary",
            )
            portfolio_result = ""
            portfolio_elapsed_ms = 0.0
            for tool_item in _execute_tools_parallel_keepalive(
                [call_info],
                actor=tool_actor,
                domain_classification=domain_classification,
            ):
                if tool_item is None:
                    yield _sse_ping()
                    continue
                _tool_call, portfolio_result, portfolio_elapsed_ms = tool_item

            err_msg = _tool_error_message(portfolio_result)
            meta = _tool_meta(portfolio_result)
            result_status = _tool_result_status(portfolio_result)
            cache_status = str(meta.get("cache", "unknown"))
            timings["tools"].append(
                {
                    "name": call_info["name"],
                    "duration_ms": portfolio_elapsed_ms,
                    "cache": cache_status,
                    "status": result_status,
                }
            )
            logger.info(
                "agent_chat_tool_exec name=%s duration_ms=%.1f cache=%s status=%s",
                call_info["name"],
                portfolio_elapsed_ms,
                cache_status,
                result_status,
            )
            tool_payload = {"name": call_info["name"], "id": call_info["call_ids"][0], "status": result_status}
            if meta.get("policy_decision_id"):
                tool_payload["policy_decision_id"] = meta.get("policy_decision_id")
            if meta.get("duration_ms") is not None:
                tool_payload["elapsed_ms"] = meta.get("duration_ms")
            if err_msg:
                tool_payload["message"] = err_msg
            yield _sse("tool_result", tool_payload)
            if result_status == "blocked":
                yield _sse("policy_failure", tool_payload)
                yield _sse("blocked", tool_payload)
            elif result_status == "timeout":
                yield _sse("timeout", tool_payload)

            synthesis_chunks: list[str] = []
            final_message: object | None = None
            for attempt in range(MAX_API_RETRIES):
                model_timing: dict[str, Any] = {
                    "phase": "model_writing",
                    "purpose": "portfolio_summary_synthesis",
                    "attempt": attempt,
                    "round_index": 0,
                    "first_token_ms": None,
                }
                model_started = time.perf_counter()
                model_event_id: str | None = None
                try:
                    yield _phase_sse(
                        "model_writing",
                        turn_started,
                        model_purpose="portfolio_summary_synthesis",
                        attempt=attempt,
                    )
                    synthesis_conversation = _user_prompt_for_provider(
                        provider,
                        _build_portfolio_summary_prompt(req.message, portfolio_result),
                    )
                    stream_kwargs = _model_stream_kwargs(
                        provider=provider,
                        instructions=instructions,
                        conversation=synthesis_conversation,
                        max_tokens=PORTFOLIO_SUMMARY_MAX_TOKENS,
                        reasoning_effort=reasoning_effort,
                    )
                    stream_kwargs, egress_meta = prepare_model_egress(
                        provider=provider,
                        purpose="portfolio_summary_synthesis",
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
                        purpose="portfolio_summary_synthesis",
                        stream_kwargs=stream_kwargs,
                        actor=tool_actor,
                        attempt=attempt,
                        round_index=0,
                    )
                    final_message = yield from _stream_llm_response(
                        client,
                        provider,
                        stream_kwargs,
                        synthesis_chunks,
                        model_timing=model_timing,
                        turn_timings=timings,
                        turn_started=turn_started,
                    )
                    _record_model_timing(
                        timings,
                        model_timing,
                        started=model_started,
                        status="ok",
                        provider=provider,
                        model=stream_kwargs.get("model"),
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
                except ModelGatewayDenied as exc:
                    yield _sse("egress_recorded", exc.manifest)
                    yield _sse("blocked", _blocked_model_egress_payload(exc))
                    raise
                except Exception as retry_exc:
                    _record_model_timing(
                        timings,
                        model_timing,
                        started=model_started,
                        status="error",
                        provider=provider,
                        model=locals().get("stream_kwargs", {}).get("model")
                        if isinstance(locals().get("stream_kwargs"), dict)
                        else None,
                    )
                    _finish_model_call_provenance(
                        model_event_id,
                        status="failed",
                        error=str(retry_exc) or retry_exc.__class__.__name__,
                    )
                    if attempt < MAX_API_RETRIES - 1 and _is_retryable_error(retry_exc):
                        time.sleep(RETRY_BASE_DELAY * (2**attempt))
                        continue
                    raise

            synthesis_text = "".join(synthesis_chunks)
            if not synthesis_text.strip():
                logger.warning("Portfolio summary synthesis returned empty text; using deterministic fallback")
                synthesis_text = _portfolio_summary_fallback(portfolio_result)
                yield _sse("delta", {"text": synthesis_text})
            turn_meta = {"client_turn_id": req.client_turn_id} if req.client_turn_id else {}
            user_msg = {"role": "user", "content": req.message, "timestamp": time.time(), **turn_meta}
            assistant_msg = {"role": "assistant", "content": synthesis_text, "timestamp": time.time(), **turn_meta}
            yield _phase_sse("finalizing", turn_started)
            finalize_turn_fn(session_id, user_msg, assistant_msg)
            _finish_agent_turn_provenance(
                agent_turn_event_id,
                status="succeeded",
                output_value=synthesis_text,
                usage=_usage_dict(final_message),
            )
            yield _sse(
                "done",
                _done_payload(
                    {
                        "usage": _usage_dict(final_message),
                        "session_id": session_id,
                        "tool_calls": [tool_payload],
                        "tools_used": [call_info["name"]],
                    },
                    timings,
                    turn_started,
                ),
            )
            return

        run_opportunity_preflight = not workflow_name and route_decision.run_opportunity_preflight
        run_decision_quality_chat = not workflow_name and route_decision.run_hidden_dq
        if run_opportunity_preflight or run_decision_quality_chat:
            dq_tool_calls = _decision_quality_chat_tool_calls(
                req.message,
                req.screen_context,
                route_decision=route_decision,
            )
            tool_payloads: list[dict[str, Any]] = []
            dq_tool_results: list[dict[str, Any]] = []
            oc_result: dict[str, Any] | None = None
            dq_result: dict[str, Any] | None = None
            final_message: object | None = None
            try:
                if dq_tool_calls:
                    yield _phase_sse(
                        "tool_running",
                        turn_started,
                        label=(
                            "Triaging opportunity..."
                            if run_opportunity_preflight and not run_decision_quality_chat
                            else "Pressure-testing thesis..."
                        ),
                        tool_names=[str(call.get("name")) for call in dq_tool_calls],
                        round_index=0,
                    )
                    budgeted_calls: list[dict[str, Any]] = []
                    for call_info in dq_tool_calls:
                        call_id = str((call_info.get("call_ids") or [call_info["name"]])[0])
                        yield _sse("tool_call", {"name": call_info["name"], "id": call_id})
                        try:
                            budget.check_tool_call(get_tool_exposure(call_info["name"]))
                            budgeted_calls.append(call_info)
                            yield _sse(
                                "tool_progress",
                                {"name": call_info["name"], "id": call_id, "status": "running"},
                            )
                        except AgentBudgetExceeded as exc:
                            result_str = _blocked_tool_result(call_info["name"], exc)
                            status = _tool_result_status(result_str)
                            payload = {"name": call_info["name"], "id": call_id, "status": status, "message": str(exc)}
                            yield _sse("policy_failure", payload)
                            yield _sse("blocked", payload)
                            yield _sse("tool_result", payload)
                            tool_payloads.append(payload)
                            dq_tool_results.append(
                                {
                                    "name": call_info["name"],
                                    "args": call_info.get("args") or {},
                                    "status": status,
                                    "result": _parse_tool_result_for_prompt(result_str),
                                }
                            )
                    yield _sse("budget_update", budget.to_meta())
                    _attach_tool_provenance_context(
                        budgeted_calls,
                        parent_event_id=agent_turn_event_id,
                        session_id=session_id,
                        workflow_run_id=None,
                        source="agent.chat.decision_quality",
                    )
                    for tool_item in _execute_tools_parallel_keepalive(
                        budgeted_calls,
                        actor=tool_actor,
                        domain_classification=domain_classification,
                    ):
                        if tool_item is None:
                            yield _sse_ping()
                            continue
                        call_info, result_str, elapsed_ms = tool_item
                        call_id = str((call_info.get("call_ids") or [call_info["name"]])[0])
                        err_msg = _tool_error_message(result_str)
                        meta = _tool_meta(result_str)
                        result_status = _tool_result_status(result_str)
                        cache_status = str(meta.get("cache", "unknown"))
                        timings["tools"].append(
                            {
                                "name": call_info["name"],
                                "duration_ms": elapsed_ms,
                                "cache": cache_status,
                                "status": result_status,
                            }
                        )
                        logger.info(
                            "agent_chat_tool_exec name=%s duration_ms=%.1f cache=%s status=%s",
                            call_info["name"],
                            elapsed_ms,
                            cache_status,
                            result_status,
                        )
                        payload = {"name": call_info["name"], "id": call_id, "status": result_status}
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
                        tool_payloads.append(payload)
                        dq_tool_results.append(
                            {
                                "name": call_info["name"],
                                "args": call_info.get("args") or {},
                                "status": result_status,
                                "result": _parse_tool_result_for_prompt(result_str),
                            }
                        )

                context_bundle = _build_decision_quality_chat_context(
                    user_text=req.message,
                    screen_context=req.screen_context,
                    raw_conversation=raw_conversation,
                    tool_results=dq_tool_results,
                    route_decision=route_decision,
                )

                if run_opportunity_preflight:
                    for attempt in range(MAX_API_RETRIES):
                        model_timing = {
                            "phase": "model_thinking",
                            "purpose": "opportunity_candidate_chat_structured",
                            "attempt": attempt,
                            "round_index": 0,
                            "first_token_ms": None,
                        }
                        model_started = time.perf_counter()
                        try:
                            yield _phase_sse(
                                "model_thinking",
                                turn_started,
                                model_purpose="opportunity_candidate_chat_structured",
                                attempt=attempt,
                                round_index=0,
                            )
                            budget.check_model_call(
                                estimated_input_tokens=max(1, len(_json_for_prompt(context_bundle)) // 4),
                                estimated_cost_usd=0.0,
                            )
                            yield _sse("budget_update", budget.to_meta())
                            oc_result = _run_opportunity_candidate_structured_pass(
                                context_bundle=context_bundle,
                                provider=provider,
                                api_key=api_key,
                                reasoning_effort=reasoning_effort,
                            )
                            _record_model_timing(
                                timings,
                                model_timing,
                                started=model_started,
                                status="ok",
                                provider=provider,
                                model=resolve_model(MODEL_MID, provider),
                            )
                            budget.record_model_usage(oc_result.get("usage") or {})
                            yield _sse("budget_update", budget.to_meta())
                            _update_intent_router_training_oc_metadata(
                                session_id=req.session_id,
                                client_turn_id=req.client_turn_id,
                                oc_result=oc_result,
                            )
                            break
                        except Exception as retry_exc:
                            _record_model_timing(
                                timings,
                                model_timing,
                                started=model_started,
                                status="error",
                                provider=provider,
                                model=resolve_model(MODEL_MID, provider),
                            )
                            if attempt < MAX_API_RETRIES - 1 and _is_retryable_error(retry_exc):
                                time.sleep(RETRY_BASE_DELAY * (2**attempt))
                                continue
                            raise

                should_run_full_dq = run_decision_quality_chat and (
                    not run_opportunity_preflight
                    or (
                        isinstance(oc_result.get("gate"), OpportunityCandidateGate)
                        and oc_result["gate"].should_graduate
                    )
                )

                if should_run_full_dq:
                    for attempt in range(MAX_API_RETRIES):
                        model_timing = {
                            "phase": "model_thinking",
                            "purpose": "decision_quality_chat_structured",
                            "attempt": attempt,
                            "round_index": 0,
                            "first_token_ms": None,
                        }
                        model_started = time.perf_counter()
                        try:
                            yield _phase_sse(
                                "model_thinking",
                                turn_started,
                                model_purpose="decision_quality_chat_structured",
                                attempt=attempt,
                                round_index=0,
                            )
                            budget.check_model_call(
                                estimated_input_tokens=max(1, len(_json_for_prompt(context_bundle)) // 4),
                                estimated_cost_usd=0.0,
                            )
                            yield _sse("budget_update", budget.to_meta())
                            dq_result = _run_decision_quality_structured_pass(
                                context_bundle=context_bundle,
                                provider=provider,
                                api_key=api_key,
                                reasoning_effort=reasoning_effort,
                            )
                            _record_model_timing(
                                timings,
                                model_timing,
                                started=model_started,
                                status="ok",
                                provider=provider,
                                model=resolve_model(MODEL_MID, provider),
                            )
                            budget.record_model_usage(dq_result.get("usage") or {})
                            yield _sse("budget_update", budget.to_meta())
                            break
                        except Exception as retry_exc:
                            _record_model_timing(
                                timings,
                                model_timing,
                                started=model_started,
                                status="error",
                                provider=provider,
                                model=resolve_model(MODEL_MID, provider),
                            )
                            if attempt < MAX_API_RETRIES - 1 and _is_retryable_error(retry_exc):
                                time.sleep(RETRY_BASE_DELAY * (2**attempt))
                                continue
                            raise

                synthesis_chunks: list[str] = []
                if should_run_full_dq:
                    for attempt in range(MAX_API_RETRIES):
                        model_timing = {
                            "phase": "model_writing",
                            "purpose": "decision_quality_chat_synthesis",
                            "attempt": attempt,
                            "round_index": 0,
                            "first_token_ms": None,
                        }
                        model_started = time.perf_counter()
                        model_event_id: str | None = None
                        try:
                            yield _phase_sse(
                                "model_writing",
                                turn_started,
                                model_purpose="decision_quality_chat_synthesis",
                                attempt=attempt,
                                round_index=0,
                            )
                            synthesis_instructions = (
                                instructions
                                + "\n\n---\n\n"
                                + _load_required_prompt_file("decision_quality_chat_synthesis.md")
                            )
                            synthesis_conversation = _user_prompt_for_provider(
                                provider,
                                _build_decision_quality_chat_synthesis_prompt(
                                    user_text=req.message,
                                    context_bundle=context_bundle,
                                    dq_result=dq_result or {},
                                ),
                            )
                            stream_kwargs = _model_stream_kwargs(
                                provider=provider,
                                instructions=synthesis_instructions,
                                conversation=synthesis_conversation,
                                max_tokens=DECISION_QUALITY_CHAT_SYNTHESIS_MAX_TOKENS,
                                reasoning_effort=reasoning_effort,
                            )
                            stream_kwargs, egress_meta = prepare_model_egress(
                                provider=provider,
                                purpose="decision_quality_chat_synthesis",
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
                                purpose="decision_quality_chat_synthesis",
                                stream_kwargs=stream_kwargs,
                                actor=tool_actor,
                                attempt=attempt,
                                round_index=0,
                            )
                            final_message = yield from _stream_llm_response(
                                client,
                                provider,
                                stream_kwargs,
                                synthesis_chunks,
                                model_timing=model_timing,
                                turn_timings=timings,
                                turn_started=turn_started,
                            )
                            _record_model_timing(
                                timings,
                                model_timing,
                                started=model_started,
                                status="ok",
                                provider=provider,
                                model=stream_kwargs.get("model"),
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
                        except ModelGatewayDenied as exc:
                            yield _sse("egress_recorded", exc.manifest)
                            yield _sse("blocked", _blocked_model_egress_payload(exc))
                            raise
                        except Exception as retry_exc:
                            _record_model_timing(
                                timings,
                                model_timing,
                                started=model_started,
                                status="error",
                                provider=provider,
                                model=locals().get("stream_kwargs", {}).get("model")
                                if isinstance(locals().get("stream_kwargs"), dict)
                                else None,
                            )
                            _finish_model_call_provenance(
                                model_event_id,
                                status="failed",
                                error=str(retry_exc) or retry_exc.__class__.__name__,
                            )
                            if attempt < MAX_API_RETRIES - 1 and _is_retryable_error(retry_exc):
                                time.sleep(RETRY_BASE_DELAY * (2**attempt))
                                continue
                            raise
                else:
                    for attempt in range(MAX_API_RETRIES):
                        model_timing = {
                            "phase": "model_writing",
                            "purpose": "opportunity_candidate_chat_synthesis",
                            "attempt": attempt,
                            "round_index": 0,
                            "first_token_ms": None,
                        }
                        model_started = time.perf_counter()
                        model_event_id: str | None = None
                        try:
                            yield _phase_sse(
                                "model_writing",
                                turn_started,
                                model_purpose="opportunity_candidate_chat_synthesis",
                                attempt=attempt,
                                round_index=0,
                            )
                            synthesis_instructions = (
                                instructions
                                + "\n\n---\n\n"
                                + _load_required_prompt_file("opportunity_candidate_synthesis.md")
                            )
                            synthesis_conversation = _user_prompt_for_provider(
                                provider,
                                _build_opportunity_candidate_synthesis_prompt(
                                    user_text=req.message,
                                    context_bundle=context_bundle,
                                    oc_result=oc_result or {},
                                ),
                            )
                            stream_kwargs = _model_stream_kwargs(
                                provider=provider,
                                instructions=synthesis_instructions,
                                conversation=synthesis_conversation,
                                max_tokens=DECISION_QUALITY_CHAT_SYNTHESIS_MAX_TOKENS,
                                reasoning_effort=reasoning_effort,
                            )
                            stream_kwargs, egress_meta = prepare_model_egress(
                                provider=provider,
                                purpose="opportunity_candidate_chat_synthesis",
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
                                purpose="opportunity_candidate_chat_synthesis",
                                stream_kwargs=stream_kwargs,
                                actor=tool_actor,
                                attempt=attempt,
                                round_index=0,
                            )
                            final_message = yield from _stream_llm_response(
                                client,
                                provider,
                                stream_kwargs,
                                synthesis_chunks,
                                model_timing=model_timing,
                                turn_timings=timings,
                                turn_started=turn_started,
                            )
                            _record_model_timing(
                                timings,
                                model_timing,
                                started=model_started,
                                status="ok",
                                provider=provider,
                                model=stream_kwargs.get("model"),
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
                        except ModelGatewayDenied as exc:
                            yield _sse("egress_recorded", exc.manifest)
                            yield _sse("blocked", _blocked_model_egress_payload(exc))
                            raise
                        except Exception as retry_exc:
                            _record_model_timing(
                                timings,
                                model_timing,
                                started=model_started,
                                status="error",
                                provider=provider,
                                model=locals().get("stream_kwargs", {}).get("model")
                                if isinstance(locals().get("stream_kwargs"), dict)
                                else None,
                            )
                            _finish_model_call_provenance(
                                model_event_id,
                                status="failed",
                                error=str(retry_exc) or retry_exc.__class__.__name__,
                            )
                            if attempt < MAX_API_RETRIES - 1 and _is_retryable_error(retry_exc):
                                time.sleep(RETRY_BASE_DELAY * (2**attempt))
                                continue
                            raise

                synthesis_text = "".join(synthesis_chunks)
                if not synthesis_text.strip():
                    if should_run_full_dq:
                        logger.warning(
                            "Decision-quality chat synthesis returned empty text; using deterministic fallback"
                        )
                        synthesis_text = _decision_quality_chat_fallback(dq_result or {})
                    else:
                        logger.warning(
                            "Opportunity-candidate chat synthesis returned empty text; using deterministic fallback"
                        )
                        synthesis_text = _opportunity_candidate_fallback(oc_result or {})
                    yield _sse("delta", {"text": synthesis_text})

                usage = _usage_dict(final_message)
                done_payload_data: dict[str, Any] = {
                    "usage": usage,
                    "session_id": session_id,
                    "tool_calls": tool_payloads,
                    "tools_used": [call["name"] for call in tool_payloads],
                    "context_pack": context_bundle.get("context_pack"),
                    "opportunity_candidate_preflight": _opportunity_candidate_done_meta(oc_result),
                    "intent_router": {
                        "applied": route_decision.to_meta(),
                        "telemetry": route_meta,
                    },
                }
                if should_run_full_dq:
                    done_payload_data["decision_quality_chat"] = _decision_quality_chat_done_meta(dq_result)
                turn_meta = {"client_turn_id": req.client_turn_id} if req.client_turn_id else {}
                user_msg = {"role": "user", "content": req.message, "timestamp": time.time(), **turn_meta}
                assistant_msg = {
                    "role": "assistant",
                    "content": synthesis_text,
                    "timestamp": time.time(),
                    "toolCalls": tool_payloads,
                    **turn_meta,
                }
                yield _phase_sse("finalizing", turn_started)
                finalize_turn_fn(session_id, user_msg, assistant_msg)
                _finish_agent_turn_provenance(
                    agent_turn_event_id,
                    status="succeeded",
                    output_value=synthesis_text,
                    usage=usage,
                )
                yield _sse(
                    "done",
                    _done_payload(
                        done_payload_data,
                        timings,
                        turn_started,
                    ),
                )
                return
            except Exception as exc:
                logger.exception("Decision-quality chat path failed")
                _finish_agent_turn_provenance(
                    agent_turn_event_id,
                    status="failed",
                    usage={},
                    error=str(exc) or exc.__class__.__name__,
                )
                yield _sse("error", {"message": _format_stream_error(exc)})
                yield _sse(
                    "done",
                    _done_payload(
                        {
                            "usage": {},
                            "session_id": session_id,
                            "tool_calls": tool_payloads,
                            "tools_used": [call["name"] for call in tool_payloads],
                            "context_pack": locals().get("context_bundle", {}).get("context_pack")
                            if isinstance(locals().get("context_bundle"), dict)
                            else None,
                            "opportunity_candidate_preflight": _opportunity_candidate_done_meta(oc_result),
                            "decision_quality_chat": _decision_quality_chat_done_meta(dq_result),
                            "intent_router": {
                                "applied": route_decision.to_meta(),
                                "telemetry": route_meta,
                            },
                        },
                        timings,
                        turn_started,
                    ),
                )
                return

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
                    yield _sse("done", _done_payload({"usage": {}, "session_id": session_id}, timings, turn_started))
                    return

                yield _phase_sse(
                    "tool_running",
                    turn_started,
                    label="Running workflow...",
                    tool_names=[workflow_name],
                )
                run_id, synthesis_prompt, sections = yield from _execute_workflow_keepalive(
                    workflow_name,
                    workflow_ticker,
                    actor=tool_actor,
                )
                workflow_tool_calls = [
                    {"name": str(section["tool"]), "id": str(section["tool"]), "status": "ok"} for section in sections
                ]
                for section in sections:
                    timings["tools"].append(
                        {
                            "name": str(section["tool"]),
                            "duration_ms": section.get("duration_ms"),
                            "cache": "workflow",
                            "status": "ok",
                        }
                    )
                    yield _sse("tool_call", {"name": section["tool"], "id": section["tool"]})
                    yield _sse("tool_result", {"name": section["tool"], "id": section["tool"], "status": "ok"})

                synthesis_chunks: list[str] = []
                for attempt in range(MAX_API_RETRIES):
                    model_timing: dict[str, Any] = {
                        "phase": "model_writing",
                        "purpose": "workflow_synthesis",
                        "attempt": attempt,
                        "round_index": None,
                        "first_token_ms": None,
                    }
                    model_started = time.perf_counter()
                    model_event_id: str | None = None
                    try:
                        yield _phase_sse(
                            "model_writing",
                            turn_started,
                            model_purpose="workflow_synthesis",
                            attempt=attempt,
                        )
                        synthesis_conversation = _user_prompt_for_provider(provider, synthesis_prompt)
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
                            model_timing=model_timing,
                            turn_timings=timings,
                            turn_started=turn_started,
                        )
                        _record_model_timing(
                            timings,
                            model_timing,
                            started=model_started,
                            status="ok",
                            provider=provider,
                            model=stream_kwargs.get("model"),
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
                    except ModelGatewayDenied as exc:
                        yield _sse("egress_recorded", exc.manifest)
                        yield _sse("blocked", _blocked_model_egress_payload(exc))
                        raise
                    except Exception as retry_exc:
                        _record_model_timing(
                            timings,
                            model_timing,
                            started=model_started,
                            status="error",
                            provider=provider,
                            model=locals().get("stream_kwargs", {}).get("model")
                            if isinstance(locals().get("stream_kwargs"), dict)
                            else None,
                        )
                        _finish_model_call_provenance(
                            model_event_id,
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
                    from api.workflows import complete_workflow_run

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
                yield _phase_sse("finalizing", turn_started)
                finalize_turn_fn(session_id, user_msg, assistant_msg)
                _finish_agent_turn_provenance(
                    agent_turn_event_id,
                    status="succeeded",
                    output_value=synthesis_text,
                    usage=usage,
                )
                yield _sse(
                    "done",
                    _done_payload(
                        {
                            "usage": usage,
                            "session_id": session_id,
                            "workflow_run_id": run_id,
                            "tool_calls": workflow_tool_calls,
                            "tools_used": [call["name"] for call in workflow_tool_calls],
                        },
                        timings,
                        turn_started,
                    ),
                )
                return

            except Exception as exc:
                logger.exception("Workflow %s failed", workflow_name)
                try:
                    from api.workflows import fail_workflow_run

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
                yield _sse("done", _done_payload({"usage": {}, "session_id": session_id}, timings, turn_started))
                return

        # --- Normal tool-calling path ---
        has_rich_screen_data = req.screen_context is not None and (
            req.screen_context.metrics or req.screen_context.summary
        )
        force_tool_use = bool(tool_defs) and _is_data_seeking(req.message) and not has_rich_screen_data
        tool_result_cache: dict[str, str] = {}
        continuation_round = 0
        output_continuation_rounds = 0
        text_only_continuation = False
        text_parts: list[str] = []

        try:
            while True:
                final_synthesis_round = continuation_round >= MAX_TOOL_CONTINUATION_ROUNDS
                round_tool_defs = [] if final_synthesis_round or text_only_continuation else tool_defs
                round_force_tool_use = False if final_synthesis_round or text_only_continuation else force_tool_use
                text_only_continuation = False

                stream_kwargs = _model_stream_kwargs(
                    provider=provider,
                    instructions=instructions,
                    conversation=conversation,
                    max_tokens=LLM_CHAT_MAX_TOKENS,
                    tool_defs=round_tool_defs,
                    force_tool_use=round_force_tool_use,
                    reasoning_effort=reasoning_effort,
                )

                for attempt in range(MAX_API_RETRIES):
                    model_phase = "model_thinking" if continuation_round == 0 and tool_defs else "model_writing"
                    model_timing: dict[str, Any] = {
                        "phase": model_phase,
                        "purpose": "agent_chat",
                        "attempt": attempt,
                        "round_index": continuation_round,
                        "first_token_ms": None,
                    }
                    model_started = time.perf_counter()
                    model_event_id: str | None = None
                    try:
                        yield _phase_sse(
                            model_phase,
                            turn_started,
                            round_index=continuation_round,
                            model_purpose="agent_chat",
                            attempt=attempt,
                        )
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
                            model_timing=model_timing,
                            turn_timings=timings,
                            turn_started=turn_started,
                        )
                        _record_model_timing(
                            timings,
                            model_timing,
                            started=model_started,
                            status="ok",
                            provider=provider,
                            model=stream_kwargs.get("model"),
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
                    except ModelGatewayDenied as exc:
                        yield _sse("egress_recorded", exc.manifest)
                        yield _sse("blocked", _blocked_model_egress_payload(exc))
                        raise
                    except Exception as retry_exc:
                        _record_model_timing(
                            timings,
                            model_timing,
                            started=model_started,
                            status="error",
                            provider=provider,
                            model=locals().get("stream_kwargs", {}).get("model")
                            if isinstance(locals().get("stream_kwargs"), dict)
                            else None,
                        )
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
                elif provider == PROVIDER_GEMINI:
                    assistant_content = _serialize_gemini_response_parts(final_message)
                    deferred_calls = _extract_gemini_tool_calls(assistant_content)
                else:
                    assistant_content = _serialize_output_items(final_message)
                    deferred_calls = _extract_openai_tool_calls(assistant_content)
                if final_synthesis_round:
                    deferred_calls = []

                if not deferred_calls and _hit_output_token_limit(provider, final_message):
                    if output_continuation_rounds >= MAX_OUTPUT_CONTINUATION_ROUNDS:
                        raise RuntimeError(
                            "Agent response hit the model output limit before completion. "
                            "Try a narrower prompt or increase the chat output token budget."
                        )
                    logger.info(
                        "agent_chat_output_continuation provider=%s round=%d stop_reason=%s",
                        provider,
                        output_continuation_rounds + 1,
                        _response_stop_reason(provider, final_message),
                    )
                    _append_output_continuation_request(provider, conversation, assistant_content)
                    output_continuation_rounds += 1
                    text_only_continuation = True
                    force_tool_use = False
                    continuation_round += 1
                    continue

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
                        "agent_chat_tool_round requested=%s unique=%s",
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
                        yield _phase_sse(
                            "tool_running",
                            turn_started,
                            tool_names=[str(call.get("name")) for call in pending_calls],
                            round_index=continuation_round,
                        )
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
                            source="agent.chat",
                        )
                        for tool_item in _execute_tools_parallel_keepalive(
                            pending_calls,
                            actor=tool_actor,
                            domain_classification=domain_classification,
                        ):
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
                            "agent_chat_tool_exec name=%s duration_ms=%.1f cache=%s status=%s",
                            call_info["name"],
                            elapsed_ms,
                            cache_status,
                            result_status,
                        )
                        timings["tools"].append(
                            {
                                "name": call_info["name"],
                                "duration_ms": elapsed_ms,
                                "cache": cache_status,
                                "status": result_status,
                            }
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
                                logger.info("agent_chat_tool_expansion added=%s total=%d", discovered, len(tool_defs))

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
                            elif provider == PROVIDER_GEMINI:
                                result_block = {
                                    "function_response": {
                                        "name": call_info["name"],
                                        "response": {"result": result_str},
                                    }
                                }
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
                    elif provider == PROVIDER_GEMINI:
                        conversation.append({"role": "model", "parts": assistant_content})
                        conversation.append({"role": "tool", "parts": tool_results})
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
                yield _phase_sse("finalizing", turn_started)
                finalize_turn_fn(session_id, user_msg, assistant_msg)
                _finish_agent_turn_provenance(
                    agent_turn_event_id,
                    status="succeeded",
                    output_value=full_text,
                    usage=usage,
                )
                yield _sse(
                    "done",
                    _done_payload(
                        {
                            "usage": usage,
                            "session_id": session_id,
                            "intent_router": {
                                "applied": route_decision.to_meta(),
                                "telemetry": route_meta,
                            },
                        },
                        timings,
                        turn_started,
                    ),
                )
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
            yield _sse("done", _done_payload({"usage": {}, "session_id": session_id}, timings, turn_started))

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )
