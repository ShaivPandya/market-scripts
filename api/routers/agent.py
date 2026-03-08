"""
AI Agent chat endpoint with streaming (SSE) and function calling.

Uses Anthropic's Messages API with Claude Sonnet 4.6 and the tool definitions from
:mod:`api.agent_tools` to answer cross-cutting investment questions by
fetching live data from the platform's analysis modules.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.agent_tools import TOOL_DEFINITIONS, execute_tool
from api.exceptions import ConfigurationError

router = APIRouter()
logger = logging.getLogger("api.agent")

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "auto_report" / "prompts"


def _load_required_prompt_file(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise ConfigurationError(f"Missing required prompt file: {path}")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ConfigurationError(f"Prompt file is empty: {path}")
    return content


def _build_agent_instructions() -> str:
    core_md = _load_required_prompt_file("system.md")
    agent_md = _load_required_prompt_file("agent_system.md")
    return "\n\n---\n\n".join([core_md, agent_md])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AgentChatRequest(BaseModel):
    messages: list[ChatMessage]


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


MAX_TOOL_CONTINUATION_ROUNDS = 8
CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_MAX_TOKENS = 8_192
ANTHROPIC_TOOL_DEFINITIONS: list[dict] = [
    {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get("parameters", {"type": "object", "properties": {}, "required": []}),
    }
    for tool in TOOL_DEFINITIONS
    if isinstance(tool.get("name"), str)
]
_TOOL_DEF_BY_NAME: dict[str, dict] = {t["name"]: t for t in ANTHROPIC_TOOL_DEFINITIONS}

_INTENT_TOOL_BUNDLES: dict[str, tuple[str, ...]] = {
    "portfolio": (
        "get_portfolio",
        "query_ontology",
        "get_signal_aggregator",
        "get_sector_metrics",
        "get_sentiment",
        "get_market_breadth",
        "get_vix_term_structure",
        "get_liquidity",
    ),
    "sentiment": (
        "get_sentiment",
        "get_vix_term_structure",
        "get_market_breadth",
        "get_positioning",
        "get_signal_aggregator",
    ),
    "technical": (
        "get_market_breadth",
        "get_vix_term_structure",
        "get_sector_metrics",
        "get_breakout",
        "get_signal_aggregator",
    ),
    "macro": (
        "get_liquidity",
        "get_economic_growth",
        "get_labor_market",
        "get_yield_curve",
        "get_positioning",
        "get_signal_aggregator",
        "get_sentiment",
    ),
    "central_bank": (
        "get_central_banks",
        "get_liquidity",
        "get_yield_curve",
        "get_signal_aggregator",
    ),
    "general": (
        "get_signal_aggregator",
        "get_sentiment",
        "get_market_breadth",
        "get_vix_term_structure",
        "get_liquidity",
        "get_portfolio",
        "query_ontology",
    ),
}
_HIGH_COST_TOOLS = {"get_sector_metrics", "query_ontology", "get_signal_aggregator"}
_CASUAL_RX = re.compile(
    r"^\s*(hi|hello|hey|yo|thanks|thank you|cool|ok|okay|who are you|what can you do)[\s!.?]*$",
    flags=re.IGNORECASE,
)
_INTENT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "portfolio",
        ("portfolio", "holding", "position", "book", "exposure", "ticker", "risk exposure", "ontology"),
    ),
    ("sentiment", ("sentiment", "bull", "bear", "fear", "put/call", "put call", "naaim", "aaii", "vvix")),
    (
        "technical",
        ("breadth", "technicals", "technical", "vix term", "momentum", "breakout", "distribution day", "sector"),
    ),
    (
        "macro",
        ("macro", "growth", "inflation", "claims", "labor", "liquidity", "yield curve", "rates", "treasury"),
    ),
    ("central_bank", ("fed", "ecb", "boj", "boe", "central bank", "policy rate", "fomc")),
)
_TOOL_REQUIRED_KEYWORDS = (
    "market",
    "portfolio",
    "position",
    "ticker",
    "sentiment",
    "breadth",
    "liquidity",
    "yield",
    "macro",
    "vix",
    "risk",
    "rate",
    "central bank",
)


def _read_anthropic_api_key() -> str:
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip().strip("\"'")
    if not api_key:
        raise ConfigurationError("ANTHROPIC_API_KEY")

    # A common misconfiguration is placing an OpenAI key into ANTHROPIC_API_KEY.
    if api_key.startswith("sk-proj-") or (api_key.startswith("sk-") and not api_key.startswith("sk-ant-")):
        raise ConfigurationError("ANTHROPIC_API_KEY (must be an Anthropic key beginning with sk-ant-)")

    return api_key


def _format_stream_error(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    raw = str(exc)
    lowered = raw.lower()

    if status_code == 401 or "invalid x-api-key" in lowered or "authentication_error" in lowered:
        return (
            "Agent authentication failed. Set a valid Anthropic API key in ANTHROPIC_API_KEY and restart the backend."
        )

    return raw


def _extract_last_user_text(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user" and msg.content.strip():
            return msg.content.strip()
    return ""


def _classify_intent(user_text: str) -> str:
    text = (user_text or "").strip().lower()
    if not text:
        return "general"
    for intent, patterns in _INTENT_PATTERNS:
        if any(p in text for p in patterns):
            return intent
    return "general"


def _requires_tool_use(user_text: str) -> bool:
    text = (user_text or "").strip().lower()
    if not text or _CASUAL_RX.match(text):
        return False
    if any(k in text for k in _TOOL_REQUIRED_KEYWORDS):
        return True
    # Default to tool-backed answers unless the prompt is clearly casual/meta.
    return True


def _tool_defs_for_names(tool_names: list[str]) -> list[dict]:
    out = []
    for name in tool_names:
        td = _TOOL_DEF_BY_NAME.get(name)
        if td is not None:
            out.append(td)
    return out


def _select_initial_tool_defs(user_text: str) -> tuple[str, list[dict]]:
    intent = _classify_intent(user_text)
    bundle = list(_INTENT_TOOL_BUNDLES.get(intent, _INTENT_TOOL_BUNDLES["general"]))
    tool_defs = _tool_defs_for_names(bundle)
    if not tool_defs:
        return intent, ANTHROPIC_TOOL_DEFINITIONS
    return intent, tool_defs


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


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/agent/chat")
def agent_chat(req: AgentChatRequest):
    api_key = _read_anthropic_api_key()
    instructions = _build_agent_instructions()
    latest_user_text = _extract_last_user_text(req.messages)
    intent, initial_tool_defs = _select_initial_tool_defs(latest_user_text)
    should_force_tools = _requires_tool_use(latest_user_text)
    logger.info(
        "agent_tool_policy intent=%s force_tool_use=%s initial_tools=%s",
        intent,
        should_force_tools,
        [t["name"] for t in initial_tool_defs],
    )

    def generate():  # noqa: C901 — complex but linear control flow
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        conversation: list[dict[str, object]] = [{"role": m.role, "content": m.content} for m in req.messages]
        continuation_round = 0
        force_tool_use = should_force_tools
        current_tool_defs = initial_tool_defs
        expanded_toolset = len(current_tool_defs) == len(ANTHROPIC_TOOL_DEFINITIONS)

        try:
            while True:
                if continuation_round >= MAX_TOOL_CONTINUATION_ROUNDS:
                    yield _sse(
                        "error",
                        {"message": (f"Tool-call loop limit reached ({MAX_TOOL_CONTINUATION_ROUNDS} rounds).")},
                    )
                    yield _sse("done", {"usage": {}})
                    return

                stream_kwargs: dict[str, object] = dict(
                    model=CLAUDE_MODEL,
                    max_tokens=CLAUDE_MAX_TOKENS,
                    system=instructions,
                    messages=conversation,
                    tools=current_tool_defs,
                )
                if force_tool_use:
                    stream_kwargs["tool_choice"] = {"type": "any"}

                with client.messages.stream(**stream_kwargs) as stream:
                    for event in stream:
                        if event.type == "content_block_delta" and event.delta.type == "text_delta":
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

                assistant_content = _serialize_content_blocks(list(final_message.content))
                deferred_calls = _extract_tool_calls(assistant_content)

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
                    had_tool_error = False
                    for call_info, result_str, elapsed_ms in _execute_tools_parallel(unique_calls):
                        err_msg = _tool_error_message(result_str)
                        if err_msg:
                            had_tool_error = True
                        meta = _tool_meta(result_str)
                        logger.info(
                            "agent_tool_exec name=%s duration_ms=%.1f cache=%s status=%s quality_ok=%s",
                            call_info["name"],
                            elapsed_ms,
                            str(meta.get("cache", "unknown")),
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

                            result_block: dict[str, object] = {
                                "type": "tool_result",
                                "tool_use_id": call_id,
                                "content": result_str,
                            }
                            if err_msg:
                                result_block["is_error"] = True
                            tool_results.append(result_block)

                    conversation.append({"role": "assistant", "content": assistant_content})
                    conversation.append({"role": "user", "content": tool_results})
                    if had_tool_error and not expanded_toolset:
                        current_tool_defs = ANTHROPIC_TOOL_DEFINITIONS
                        expanded_toolset = True
                        force_tool_use = True
                        logger.warning("agent_tool_policy expanding toolset to full due to tool errors")
                    else:
                        force_tool_use = False
                    continuation_round += 1
                    continue

                # If a fact-backed answer was requested but no tool was called from a narrowed bundle,
                # retry once with full tool access before finalizing.
                if force_tool_use and not expanded_toolset:
                    current_tool_defs = ANTHROPIC_TOOL_DEFINITIONS
                    expanded_toolset = True
                    continuation_round += 1
                    logger.warning(
                        "agent_tool_policy no tools used under intent bundle; expanding to full toolset and retrying"
                    )
                    continue

                if final_message.stop_reason == "pause_turn":
                    conversation.append({"role": "assistant", "content": assistant_content})
                    conversation.append({"role": "user", "content": [{"type": "text", "text": "Continue."}]})
                    force_tool_use = False
                    continuation_round += 1
                    continue

                usage = {}
                if hasattr(final_message, "usage") and final_message.usage:
                    usage = {
                        "input_tokens": final_message.usage.input_tokens,
                        "output_tokens": final_message.usage.output_tokens,
                    }
                yield _sse("done", {"usage": usage})
                return

        except Exception as exc:
            logger.exception("Agent stream error")
            yield _sse("error", {"message": _format_stream_error(exc)})
            yield _sse("done", {"usage": {}})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
