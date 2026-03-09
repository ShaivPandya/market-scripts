"""
AI Agent chat endpoint with streaming (SSE) and function calling.

Uses Anthropic's Messages API with Claude Sonnet 4.6 and the tool definitions from
:mod:`api.agent_tools` to answer cross-cutting investment questions by
fetching live data from the platform's analysis modules.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import threading
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
from api.workflows import AVAILABLE_WORKFLOWS, execute_workflow

router = APIRouter()
logger = logging.getLogger("api.agent")

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "auto_report" / "prompts"


@functools.lru_cache(maxsize=4)
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
    base = "\n\n---\n\n".join([core_md, agent_md])

    memory_section = _build_memory_context()
    if memory_section:
        base += "\n\n---\n\n" + memory_section

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

        lines = ["## Recent Research Context\n"]
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


class AgentChatRequest(BaseModel):
    messages: list[ChatMessage]


@router.get("/agent/workflows")
def list_workflows():
    """List available deterministic workflows."""
    return [{"name": name, **info} for name, info in AVAILABLE_WORKFLOWS.items()]


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


MAX_TOOL_CONTINUATION_ROUNDS = 8
MAX_API_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds
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
_HIGH_COST_TOOLS = {"get_sector_metrics", "query_ontology", "get_signal_aggregator"}
_CASUAL_RX = re.compile(
    r"^\s*(hi|hello|hey|yo|thanks|thank you|cool|ok|okay|who are you|what can you do)[\s!.?]*$",
    flags=re.IGNORECASE,
)


def _read_anthropic_api_key() -> str:
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip().strip("\"'")
    if not api_key:
        raise ConfigurationError("ANTHROPIC_API_KEY")

    # A common misconfiguration is placing an OpenAI key into ANTHROPIC_API_KEY.
    if api_key.startswith("sk-proj-") or (api_key.startswith("sk-") and not api_key.startswith("sk-ant-")):
        raise ConfigurationError("ANTHROPIC_API_KEY (must be an Anthropic key beginning with sk-ant-)")

    return api_key


_anthropic_client = None
_client_lock = threading.Lock()


def _get_anthropic_client(api_key: str):
    """Return a cached Anthropic client, creating one on first call."""
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client
    with _client_lock:
        if _anthropic_client is not None:
            return _anthropic_client
        from anthropic import Anthropic

        _anthropic_client = Anthropic(api_key=api_key)
        return _anthropic_client


def _format_stream_error(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    raw = str(exc)
    lowered = raw.lower()

    if status_code == 401 or "invalid x-api-key" in lowered or "authentication_error" in lowered:
        return (
            "Agent authentication failed. Set a valid Anthropic API key in ANTHROPIC_API_KEY and restart the backend."
        )

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
    casual = _is_casual(latest_user_text)
    workflow_name, workflow_ticker = _detect_workflow(latest_user_text)
    tool_defs = ANTHROPIC_TOOL_DEFINITIONS
    logger.info(
        "agent_tool_policy casual=%s workflow=%s ticker=%s tools=%d",
        casual,
        workflow_name,
        workflow_ticker,
        len(tool_defs),
    )

    def generate():  # noqa: C901 — complex but linear control flow
        client = _get_anthropic_client(api_key)

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
                synthesis_prompt, sections = execute_workflow(workflow_name, workflow_ticker)
                for section in sections:
                    yield _sse("tool_call", {"name": section["tool"], "id": section["tool"]})
                    yield _sse("tool_result", {"name": section["tool"], "id": section["tool"], "status": "ok"})

                # Single synthesis call — no tools, just Claude reasoning over the data
                for attempt in range(MAX_API_RETRIES):
                    try:
                        with client.messages.stream(
                            model=CLAUDE_MODEL,
                            max_tokens=CLAUDE_MAX_TOKENS,
                            system=instructions,
                            messages=[{"role": "user", "content": synthesis_prompt}],
                        ) as stream:
                            for event in stream:
                                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                                    yield _sse("delta", {"text": event.delta.text})
                            final_message = stream.get_final_message()
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

                usage = {}
                if hasattr(final_message, "usage") and final_message.usage:
                    usage = {
                        "input_tokens": final_message.usage.input_tokens,
                        "output_tokens": final_message.usage.output_tokens,
                    }
                yield _sse("done", {"usage": usage})
                return

            except Exception as exc:
                logger.exception("Workflow %s failed", workflow_name)
                yield _sse("error", {"message": f"Workflow failed: {exc}"})
                yield _sse("done", {"usage": {}})
                return

        # --- Normal tool-calling path ---
        conversation: list[dict[str, object]] = [{"role": m.role, "content": m.content} for m in req.messages]
        continuation_round = 0
        # Force tool use on the first round for non-casual queries so
        # answers are always grounded in live data.
        force_tool_use = not casual
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

                stream_kwargs: dict[str, object] = dict(
                    model=CLAUDE_MODEL,
                    max_tokens=CLAUDE_MAX_TOKENS,
                    system=instructions,
                    messages=conversation,
                    tools=tool_defs,
                )
                if force_tool_use:
                    stream_kwargs["tool_choice"] = {"type": "any"}

                for attempt in range(MAX_API_RETRIES):
                    try:
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
                        for call_info, result_str, elapsed_ms in _execute_tools_parallel(pending_calls):
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
                    # After the first tool round, let Claude decide whether it
                    # needs more data (tool_choice: auto).
                    force_tool_use = False
                    continuation_round += 1
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
