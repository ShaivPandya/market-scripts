"""
Deterministic workflow engine for the AI agent.

Complex multi-step research tasks (morning brief, thesis review, pre-earnings
prep) execute a fixed tool sequence, collect all data, then hand it to Claude
for synthesis only.  This avoids multi-round tool discovery and ensures
consistent, repeatable outputs.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from api.agent_tools import execute_tool

logger = logging.getLogger("api.workflows")

# ---------------------------------------------------------------------------
# Workflow definitions
# ---------------------------------------------------------------------------

AVAILABLE_WORKFLOWS = {
    "morning_brief": {
        "label": "Morning Brief",
        "description": "Quick macro + portfolio + signals overview to start the day.",
        "requires_ticker": False,
    },
    "thesis_review": {
        "label": "Thesis Review",
        "description": "Deep review of a position's investment thesis with risk context.",
        "requires_ticker": True,
    },
    "pre_earnings": {
        "label": "Pre-Earnings Prep",
        "description": "Earnings briefing with thesis, sector context, and past research.",
        "requires_ticker": True,
    },
}


# ---------------------------------------------------------------------------
# Tool execution helpers
# ---------------------------------------------------------------------------


def _exec_tool(name: str, args: dict | None = None) -> tuple[str, dict, float]:
    """Execute a single tool, return (result_json, parsed_dict, elapsed_ms)."""
    args = args or {}
    started = time.perf_counter()
    try:
        result_str = execute_tool(name, args)
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        try:
            parsed = json.loads(result_str)
        except Exception:
            parsed = {"raw": result_str}
        return result_str, parsed, elapsed
    except Exception as exc:
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        logger.warning("Workflow tool %s failed: %s", name, exc)
        return json.dumps({"error": str(exc)}), {"error": str(exc)}, elapsed


def _exec_parallel(calls: list[tuple[str, dict]]) -> list[tuple[str, dict, float]]:
    """Execute multiple tool calls in parallel."""
    if len(calls) == 1:
        return [_exec_tool(calls[0][0], calls[0][1])]
    with ThreadPoolExecutor(max_workers=min(len(calls), 6)) as pool:
        futures = [(name, pool.submit(_exec_tool, name, args)) for name, args in calls]
        return [(name, fut.result()[1], fut.result()[2]) for name, fut in futures]


# ---------------------------------------------------------------------------
# Workflow: Morning Brief
# ---------------------------------------------------------------------------

_MORNING_BRIEF_SYNTHESIS = """\
You are delivering a 2-minute morning brief to a portfolio manager.

Given the data below, provide a concise, actionable morning brief covering:

1. **Regime & Risk Posture** – Signal aggregator composite, regime classification, whether posture should be risk-on/off/neutral
2. **Key Signals** – The 2-3 most important factor scores or signal changes from the aggregator
3. **Portfolio Snapshot** – Total P&L, notable movers (biggest winners/losers), any position-level alerts
4. **Technical Alerts** – Any breakout or breakdown signals
5. **Central Bank / Macro** – Overnight CB actions or notable macro developments

End with 2-3 specific **action items** (e.g., "Review CRWD position — thesis under pressure from sector rotation").

Be direct, use numbers, skip hedging language. This is for a professional investor who wants signal, not noise.
"""


def run_morning_brief() -> tuple[str, list[dict[str, Any]]]:
    """Execute morning brief workflow.

    Returns (synthesis_prompt, tool_data_sections).
    """
    calls: list[tuple[str, dict[str, Any]]] = [
        ("get_signal_aggregator", {}),
        ("get_portfolio", {}),
        ("get_breakout", {}),
        ("get_central_banks", {}),
    ]

    results: list[tuple[str, dict, float]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(name, pool.submit(_exec_tool, name, args)) for name, args in calls]
        for name, fut in futures:
            _str, parsed, elapsed = fut.result()
            results.append((name, parsed, elapsed))
            logger.info("workflow=morning_brief tool=%s duration_ms=%.1f", name, elapsed)

    # Build data sections for the synthesis prompt
    sections: list[dict[str, Any]] = []
    data_text_parts: list[str] = []
    for name, parsed, elapsed in results:
        section = {"tool": name, "data": parsed, "duration_ms": elapsed}
        sections.append(section)
        data_text_parts.append(f"### {name}\n```json\n{json.dumps(parsed, indent=1, default=str)[:6000]}\n```")

    data_block = "\n\n".join(data_text_parts)
    synthesis_prompt = f"{_MORNING_BRIEF_SYNTHESIS}\n\n---\n\n{data_block}"

    return synthesis_prompt, sections


# ---------------------------------------------------------------------------
# Workflow: Thesis Review
# ---------------------------------------------------------------------------

_THESIS_REVIEW_SYNTHESIS = """\
You are conducting a thorough thesis review for {ticker}.

Given the data below, assess:

1. **Thesis Status** – Is the thesis intact, weakening, or invalidated? Cite specific evidence.
2. **Key Catalysts** – Which catalysts have played out, which are pending, which are at risk?
3. **Kill Conditions** – Are any kill switches being triggered or approaching?
4. **Risk Context** – What does the ontology risk score say? How does macro/sector environment affect this position?
5. **Evaluation Trend** – Has conviction been rising or falling based on evaluation history?
6. **Position Sizing** – Is current position size appropriate given thesis conviction and risk?

End with a clear **recommendation**: hold, add, trim, or exit — with reasoning.
"""


def run_thesis_review(ticker: str) -> tuple[str, list[dict[str, Any]]]:
    """Execute thesis review workflow for a specific ticker."""
    ticker = ticker.upper()

    # Phase 1: parallel fetch
    calls = [
        ("get_thesis", {"ticker": ticker}),
        ("get_thesis_evaluations", {"ticker": ticker}),
        ("get_portfolio", {}),
        ("query_ontology", {"filters": json.dumps({"tickers": [ticker]})}),
        ("get_industry_monitor", {}),
    ]

    results: list[tuple[str, dict, float]] = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [(name, pool.submit(_exec_tool, name, args)) for name, args in calls]
        for name, fut in futures:
            _str, parsed, elapsed = fut.result()
            results.append((name, parsed, elapsed))
            logger.info("workflow=thesis_review ticker=%s tool=%s duration_ms=%.1f", ticker, name, elapsed)

    sections: list[dict[str, Any]] = []
    data_text_parts: list[str] = []
    for name, parsed, elapsed in results:
        section = {"tool": name, "data": parsed, "duration_ms": elapsed}
        sections.append(section)
        data_text_parts.append(f"### {name}\n```json\n{json.dumps(parsed, indent=1, default=str)[:6000]}\n```")

    data_block = "\n\n".join(data_text_parts)
    synthesis_prompt = _THESIS_REVIEW_SYNTHESIS.format(ticker=ticker) + f"\n\n---\n\n{data_block}"

    return synthesis_prompt, sections


# ---------------------------------------------------------------------------
# Workflow: Pre-Earnings Prep
# ---------------------------------------------------------------------------

_PRE_EARNINGS_SYNTHESIS = """\
You are preparing a pre-earnings briefing for {ticker}.

Given the data below, deliver:

1. **Position Context** – Current size, P&L, thesis status, key catalysts for this earnings
2. **What to Watch** – The 3-5 most important metrics/KPIs for this earnings report
3. **Consensus vs Thesis** – How does your thesis differ from consensus? What would confirm/disconfirm it?
4. **Sector Context** – How is the sector performing? Any read-through from peers who already reported?
5. **Risk Scenarios**:
   - **Bull case**: What happens if earnings beat + guide up? Position sizing response.
   - **Bear case**: What happens if earnings miss or guide down? Downside protection plan.
   - **Base case**: Most likely outcome and appropriate reaction.
6. **Past Research** – Any relevant insights from prior analysis on this name

End with a specific **game plan**: pre-earnings positioning adjustments (if any) and post-earnings reaction framework.
"""


def run_pre_earnings(ticker: str) -> tuple[str, list[dict[str, Any]]]:
    """Execute pre-earnings prep workflow for a specific ticker."""
    ticker = ticker.upper()

    calls = [
        ("get_thesis", {"ticker": ticker}),
        ("get_portfolio", {}),
        ("get_sector_metrics", {}),
        ("get_industry_monitor", {}),
    ]

    results: list[tuple[str, dict, float]] = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [(name, pool.submit(_exec_tool, name, args)) for name, args in calls]
        for name, fut in futures:
            _str, parsed, elapsed = fut.result()
            results.append((name, parsed, elapsed))
            logger.info("workflow=pre_earnings ticker=%s tool=%s duration_ms=%.1f", ticker, name, elapsed)

    # Also try knowledge base search (may fail if no embeddings)
    try:
        kb_str, kb_parsed, kb_elapsed = _exec_tool(
            "search_knowledge_base",
            {"query": f"{ticker} earnings analysis", "tickers": ticker, "top_k": 3},
        )
        results.append(("search_knowledge_base", kb_parsed, kb_elapsed))
        logger.info("workflow=pre_earnings ticker=%s tool=search_knowledge_base duration_ms=%.1f", ticker, kb_elapsed)
    except Exception:
        pass

    sections: list[dict[str, Any]] = []
    data_text_parts: list[str] = []
    for name, parsed, elapsed in results:
        section = {"tool": name, "data": parsed, "duration_ms": elapsed}
        sections.append(section)
        data_text_parts.append(f"### {name}\n```json\n{json.dumps(parsed, indent=1, default=str)[:6000]}\n```")

    data_block = "\n\n".join(data_text_parts)
    synthesis_prompt = _PRE_EARNINGS_SYNTHESIS.format(ticker=ticker) + f"\n\n---\n\n{data_block}"

    return synthesis_prompt, sections


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def execute_workflow(
    workflow_name: str,
    ticker: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Run a named workflow. Returns (synthesis_prompt, tool_data_sections).

    Raises ValueError if workflow_name is unknown or ticker is required but missing.
    """
    wf = AVAILABLE_WORKFLOWS.get(workflow_name)
    if wf is None:
        raise ValueError(f"Unknown workflow: {workflow_name}. Available: {list(AVAILABLE_WORKFLOWS)}")

    if wf["requires_ticker"] and not ticker:
        raise ValueError(f"Workflow '{workflow_name}' requires a ticker parameter")

    if workflow_name == "morning_brief":
        return run_morning_brief()
    elif workflow_name == "thesis_review":
        return run_thesis_review(ticker)  # type: ignore[arg-type]
    elif workflow_name == "pre_earnings":
        return run_pre_earnings(ticker)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Workflow '{workflow_name}' is defined but has no implementation")
