"""
AI Agent chat endpoint with streaming (SSE) and function calling.

Uses the OpenAI Responses API with GPT-5.4 and the tool definitions from
:mod:`api.agent_tools` to answer cross-cutting investment questions by
fetching live data from the platform's analysis modules.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.agent_tools import TOOL_DEFINITIONS, execute_tool
from api.exceptions import ConfigurationError

router = APIRouter()
logger = logging.getLogger("api.agent")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert investment analyst assistant embedded in a market analysis platform. \
You have access to real-time market data through a set of tools that fetch data from \
various financial data sources (FRED, Yahoo Finance, CFTC, SEC, central bank feeds, etc.).

## Your Role
You help professional investors analyze markets, assess risks, and understand the \
current macro and micro environment. You provide data-driven analysis, not investment \
advice. You cite specific numbers from the data you retrieve.

## Available Data Tools
You have access to the following data-fetching tools. Call them when you need current \
data to answer a question:

- **get_liquidity**: Global liquidity dashboard (composite score, regime, regional scores, components, changes)
- **get_market_breadth**: S&P 500 market breadth (% above 200/20 DMA, new highs/lows)
- **get_vix_term_structure**: VIX term structure (VIX, VIX3M, ratio, signal)
- **get_positioning**: CFTC COT leveraged fund positioning (net %, z-scores, forced flows)
- **get_signal_aggregator**: Unified cross-module regime dashboard (factor scores, composite regime, history, failures)
- **get_economic_growth**: Cross-asset returns for growth regime assessment (commodities, equities, FX)
- **get_labor_market**: US labor market indicators (claims, wages, JOLTS, hours)
- **get_sector_metrics**: S&P 500 sector weights, changes, relative performance, trend quality
- **get_portfolio**: User's portfolio positions and P&L
- **get_yield_curve**: Government bond yield curves (US, DE, UK, JP)
- **get_sentiment**: Put/call ratios, investor surveys (AAII/NAAIM), volatility indices (VIX/VXN/VVIX)
- **get_central_banks**: Central bank news, speeches, and policy documents
- **get_industry_monitor**: Industry transcript-based trend and momentum monitor (banks, trucking, retail, housing)
- **get_breakout**: Macro breakout signals across asset classes
- **query_ontology**: Cross-module ontology query joining portfolio positions with macro and technical risk signals

## Behavioral Guidelines
1. When asked about a topic covered by your tools, ALWAYS fetch the data first rather than speculating. Do not make claims about current market conditions without data.
2. When answering cross-cutting questions (e.g., "What's the overall risk environment?"), call multiple relevant tools to build a comprehensive picture.
3. Present analysis in clear, flowing prose. Use numbers and specifics from the data. Avoid vague generalities.
4. If a tool call fails, tell the user what happened and work with the data you do have.
5. Never fabricate data points. If you don't have data, say so.
6. Keep responses focused and professional. Use bullet points sparingly — prefer flowing prose.
7. You may use markdown formatting (headers, bold, tables) to structure longer responses.
8. When the user asks about their portfolio alongside market data, fetch both the portfolio dashboard and relevant market tools to provide integrated analysis.
9. Prefer query_ontology for portfolio risk-exposure questions that require joining portfolio, sectors, VIX, breadth, and macro conditions.
"""

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


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/agent/chat")
def agent_chat(req: AgentChatRequest):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ConfigurationError("OPENAI_API_KEY")

    def generate():  # noqa: C901 — complex but linear control flow
        from openai import OpenAI

        client = OpenAI()

        # Build messages for the first call
        input_messages = [{"role": m.role, "content": m.content} for m in req.messages]

        try:
            # Initial streaming call
            stream = client.responses.create(
                model="gpt-5.4",
                instructions=SYSTEM_PROMPT,
                input=input_messages,
                tools=TOOL_DEFINITIONS,
                stream=True,
            )

            # Collect output items from the stream
            response_id: str | None = None
            pending_tool_calls: list[dict] = []
            current_fn_name: str | None = None
            current_fn_call_id: str | None = None
            current_fn_args: str = ""

            for event in stream:
                # Capture the response ID for continuations
                if event.type == "response.created":
                    response_id = event.response.id

                # Text deltas → stream to client
                elif event.type == "response.output_text.delta":
                    yield _sse("delta", {"text": event.delta})

                # Function call started
                elif event.type == "response.function_call_arguments.delta":
                    current_fn_args += event.delta

                elif event.type == "response.output_item.added":
                    if event.item.type == "function_call":
                        current_fn_name = event.item.name
                        current_fn_call_id = event.item.call_id
                        current_fn_args = ""
                        yield _sse(
                            "tool_call",
                            {
                                "name": current_fn_name,
                                "id": current_fn_call_id,
                            },
                        )

                elif event.type == "response.output_item.done":
                    if event.item.type == "function_call":
                        # Execute the tool
                        try:
                            args = json.loads(current_fn_args) if current_fn_args else {}
                        except json.JSONDecodeError:
                            args = {}
                        result_str = execute_tool(event.item.name, args)
                        pending_tool_calls.append(
                            {
                                "call_id": event.item.call_id,
                                "output": result_str,
                            }
                        )
                        yield _sse(
                            "tool_result",
                            {
                                "name": event.item.name,
                                "id": event.item.call_id,
                                "status": "ok",
                            },
                        )

                elif event.type == "response.completed":
                    # If there were tool calls, we need to continue
                    if not pending_tool_calls:
                        usage = {}
                        if hasattr(event.response, "usage") and event.response.usage:
                            usage = {
                                "input_tokens": event.response.usage.input_tokens,
                                "output_tokens": event.response.usage.output_tokens,
                            }
                        yield _sse("done", {"usage": usage})

            # Tool-call continuation loop
            while pending_tool_calls and response_id:
                tool_outputs = []
                for tc in pending_tool_calls:
                    tool_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": tc["call_id"],
                            "output": tc["output"],
                        }
                    )
                pending_tool_calls = []

                stream = client.responses.create(
                    model="gpt-5.4",
                    previous_response_id=response_id,
                    input=tool_outputs,
                    tools=TOOL_DEFINITIONS,
                    stream=True,
                )

                current_fn_name = None
                current_fn_call_id = None
                current_fn_args = ""

                for event in stream:
                    if event.type == "response.created":
                        response_id = event.response.id

                    elif event.type == "response.output_text.delta":
                        yield _sse("delta", {"text": event.delta})

                    elif event.type == "response.function_call_arguments.delta":
                        current_fn_args += event.delta

                    elif event.type == "response.output_item.added":
                        if event.item.type == "function_call":
                            current_fn_name = event.item.name
                            current_fn_call_id = event.item.call_id
                            current_fn_args = ""
                            yield _sse(
                                "tool_call",
                                {
                                    "name": current_fn_name,
                                    "id": current_fn_call_id,
                                },
                            )

                    elif event.type == "response.output_item.done":
                        if event.item.type == "function_call":
                            try:
                                args = json.loads(current_fn_args) if current_fn_args else {}
                            except json.JSONDecodeError:
                                args = {}
                            result_str = execute_tool(event.item.name, args)
                            pending_tool_calls.append(
                                {
                                    "call_id": event.item.call_id,
                                    "output": result_str,
                                }
                            )
                            yield _sse(
                                "tool_result",
                                {
                                    "name": event.item.name,
                                    "id": event.item.call_id,
                                    "status": "ok",
                                },
                            )

                    elif event.type == "response.completed":
                        if not pending_tool_calls:
                            usage = {}
                            if hasattr(event.response, "usage") and event.response.usage:
                                usage = {
                                    "input_tokens": event.response.usage.input_tokens,
                                    "output_tokens": event.response.usage.output_tokens,
                                }
                            yield _sse("done", {"usage": usage})

        except Exception as exc:
            logger.exception("Agent stream error")
            yield _sse("error", {"message": str(exc)})
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
