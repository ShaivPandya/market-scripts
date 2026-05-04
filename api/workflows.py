"""
Deterministic workflow engine for the AI agent.

Complex multi-step research tasks execute a fixed tool sequence, collect all
data, then hand it to Claude for synthesis only.  This avoids multi-round tool
discovery and ensures consistent, repeatable outputs.

Each workflow run is persisted in core_db.workflow_runs for auditability.
"""

from __future__ import annotations

import inspect
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from api.agent_tools import execute_tool
from api.audit import emit_audit_event
from ontology.action_registry import get_tool_exposure
from ontology.policy import Actor, admin_actor

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
    "post_earnings_review": {
        "label": "Post-Earnings Review",
        "description": "Debrief after earnings: assess thesis, update catalysts, determine action.",
        "requires_ticker": True,
    },
    "weekly_portfolio_review": {
        "label": "Weekly Portfolio Review",
        "description": "Portfolio-level risk assessment, positions needing attention, macro context.",
        "requires_ticker": False,
    },
    "thesis_invalidation_check": {
        "label": "Thesis Invalidation Check",
        "description": "Check if any kill conditions are approaching or triggered for a position.",
        "requires_ticker": True,
    },
}


# ---------------------------------------------------------------------------
# Tool execution helpers
# ---------------------------------------------------------------------------


def _workflow_tool_context(
    *,
    workflow_run_id: str | None,
    workflow_name: str | None,
    tool_index: int,
    tool_name: str,
) -> dict[str, Any] | None:
    if not workflow_run_id:
        return None
    try:
        from api import provenance

        workflow_event_id = provenance.deterministic_id("pv:workflow_run", workflow_run_id)
        tool_event_id = provenance.deterministic_id("pv:workflow_tool_call", workflow_run_id, tool_index, tool_name)
        return {
            "event_id": tool_event_id,
            "workflow_run_id": workflow_run_id,
            "parent_event_id": workflow_event_id,
            "call_id": f"{workflow_name or 'workflow'}:{tool_index}:{tool_name}",
            "source": f"workflow.{workflow_name or 'unknown'}",
        }
    except Exception:
        logger.debug("Failed to build workflow provenance context run_id=%s", workflow_run_id, exc_info=True)
        return None


def _exec_tool(
    name: str,
    args: dict | None = None,
    actor: Actor | None = None,
    provenance_context: dict[str, Any] | None = None,
) -> tuple[str, dict, float]:
    """Execute a single tool, return (result_json, parsed_dict, elapsed_ms)."""
    args = args or {}
    actor = actor or admin_actor(source="workflow")
    exposure = get_tool_exposure(name)
    if exposure.access_mode not in {"read", "compute"}:
        raise ValueError(f"Workflow tool '{name}' is not allowed for workflow execution")
    started = time.perf_counter()
    emit_audit_event(
        "workflow.tool.started",
        "workflow",
        "started",
        actor=actor,
        object_refs=[{"type": "tool", "id": name}],
        metadata={"tool_name": name, "arg_keys": sorted(args.keys())},
    )
    try:
        result_str = _execute_tool_for_actor(name, args, actor, provenance_context=provenance_context)
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        try:
            parsed = json.loads(result_str)
        except Exception:
            parsed = {"raw": result_str}
        emit_audit_event(
            "workflow.tool.completed",
            "workflow",
            "succeeded",
            actor=actor,
            object_refs=[{"type": "tool", "id": name}],
            after_summary={
                "tool_name": name,
                "duration_ms": elapsed,
                "status": "error" if isinstance(parsed, dict) and parsed.get("error") else "ok",
                "result_keys": sorted(parsed.keys()) if isinstance(parsed, dict) else [],
            },
        )
        return result_str, parsed, elapsed
    except Exception as exc:
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        logger.warning("Workflow tool %s failed: %s", name, exc)
        emit_audit_event(
            "workflow.tool.failed",
            "workflow",
            "failed",
            actor=actor,
            object_refs=[{"type": "tool", "id": name}],
            after_summary={"tool_name": name, "duration_ms": elapsed},
            error=str(exc),
        )
        return json.dumps({"error": str(exc)}), {"error": str(exc)}, elapsed


def _execute_tool_for_actor(
    name: str,
    args: dict,
    actor: Actor,
    *,
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


def _exec_parallel(
    calls: list[tuple[str, dict]],
    actor: Actor | None = None,
    *,
    workflow_run_id: str | None = None,
    workflow_name: str | None = None,
    start_index: int = 0,
) -> list[tuple[str, dict, float]]:
    """Execute multiple tool calls in parallel."""
    actor = actor or admin_actor(source="workflow")
    if len(calls) == 1:
        name, args = calls[0]
        _raw, parsed, elapsed = _exec_tool(
            name,
            args,
            actor=actor,
            provenance_context=_workflow_tool_context(
                workflow_run_id=workflow_run_id,
                workflow_name=workflow_name,
                tool_index=start_index,
                tool_name=name,
            ),
        )
        return [(name, parsed, elapsed)]
    with ThreadPoolExecutor(max_workers=min(len(calls), 6)) as pool:
        futures = []
        for offset, (name, args) in enumerate(calls):
            futures.append(
                (
                    name,
                    pool.submit(
                        _exec_tool,
                        name,
                        args,
                        actor,
                        _workflow_tool_context(
                            workflow_run_id=workflow_run_id,
                            workflow_name=workflow_name,
                            tool_index=start_index + offset,
                            tool_name=name,
                        ),
                    ),
                )
            )
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


def run_morning_brief(
    actor: Actor | None = None,
    workflow_run_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Execute morning brief workflow.

    Returns (synthesis_prompt, tool_data_sections).
    """
    calls: list[tuple[str, dict[str, Any]]] = [
        ("get_signal_aggregator", {}),
        ("get_portfolio", {}),
        ("get_breakout", {}),
        ("get_central_banks", {}),
    ]

    actor = actor or admin_actor(source="workflow")
    results = _exec_parallel(
        calls,
        actor=actor,
        workflow_run_id=workflow_run_id,
        workflow_name="morning_brief",
    )
    for name, _parsed, elapsed in results:
        logger.info("workflow=morning_brief tool=%s duration_ms=%.1f", name, elapsed)

    sections, data_block = _build_sections(results)
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
7. **Recent News** – What recent headlines are relevant? Do they support or challenge the thesis?

End with a clear **recommendation**: hold, add, trim, or exit — with reasoning.

After your analysis, output a structured JSON block fenced with ```artifacts
{{
  "evaluation_draft": {{
    "ticker": "{ticker}",
    "thesis_status": "strengthen|neutral|weaken|insufficient-data",
    "technical_read": "...",
    "fundamental_read": "...",
    "action": "hold|add|trim|exit|reassess",
    "confidence": "high|medium|low",
    "key_developments": ["..."],
    "earnings_note": null,
    "risk_flag": null
  }}
}}
```
"""


def run_thesis_review(
    ticker: str,
    actor: Actor | None = None,
    workflow_run_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Execute thesis review workflow for a specific ticker."""
    ticker = ticker.upper()

    # Phase 1: parallel fetch
    calls: list[tuple[str, dict[str, Any]]] = [
        ("get_thesis", {"ticker": ticker}),
        ("get_thesis_evaluations", {"ticker": ticker}),
        ("get_portfolio", {}),
        ("query_ontology", {"filters": {"tickers": [ticker]}}),
        ("get_industry_monitor", {}),
        ("search_web", {"query": f"{ticker} recent news developments catalysts"}),
    ]

    actor = actor or admin_actor(source="workflow")
    results = _exec_parallel(
        calls,
        actor=actor,
        workflow_run_id=workflow_run_id,
        workflow_name="thesis_review",
    )
    for name, _parsed, elapsed in results:
        logger.info("workflow=thesis_review ticker=%s tool=%s duration_ms=%.1f", ticker, name, elapsed)

    sections, data_block = _build_sections(results)
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


def run_pre_earnings(
    ticker: str,
    actor: Actor | None = None,
    workflow_run_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Execute pre-earnings prep workflow for a specific ticker."""
    ticker = ticker.upper()

    calls: list[tuple[str, dict[str, Any]]] = [
        ("get_thesis", {"ticker": ticker}),
        ("get_portfolio", {}),
        ("get_sector_metrics", {}),
        ("get_industry_monitor", {}),
    ]

    actor = actor or admin_actor(source="workflow")
    results = _exec_parallel(
        calls,
        actor=actor,
        workflow_run_id=workflow_run_id,
        workflow_name="pre_earnings",
    )
    for name, _parsed, elapsed in results:
        logger.info("workflow=pre_earnings ticker=%s tool=%s duration_ms=%.1f", ticker, name, elapsed)

    # Also try knowledge base search (may fail if no embeddings)
    try:
        _kb_str, kb_parsed, kb_elapsed = _exec_tool(
            "search_knowledge_base",
            {"query": f"{ticker} earnings analysis", "tickers": ticker, "top_k": 3},
            actor=actor,
            provenance_context=_workflow_tool_context(
                workflow_run_id=workflow_run_id,
                workflow_name="pre_earnings",
                tool_index=len(results),
                tool_name="search_knowledge_base",
            ),
        )
        results.append(("search_knowledge_base", kb_parsed, kb_elapsed))
        logger.info("workflow=pre_earnings ticker=%s tool=search_knowledge_base duration_ms=%.1f", ticker, kb_elapsed)
    except Exception:
        pass

    sections, data_block = _build_sections(results)
    synthesis_prompt = _PRE_EARNINGS_SYNTHESIS.format(ticker=ticker) + f"\n\n---\n\n{data_block}"

    return synthesis_prompt, sections


# ---------------------------------------------------------------------------
# Workflow: Post-Earnings Review
# ---------------------------------------------------------------------------

_POST_EARNINGS_SYNTHESIS = """\
You are conducting a post-earnings review for {ticker}.

Given the data below, assess:

1. **Earnings vs Thesis** – Did the earnings results confirm or challenge the thesis? Cite specific numbers.
2. **Catalyst Update** – Which catalysts played out? Which were disconfirmed? Any new ones emerged?
3. **Kill Condition Check** – Are any kill conditions now closer to being triggered?
4. **Management Commentary** – What did management signal about forward outlook?
5. **Position Sizing** – Should the position be added to, trimmed, or held given the results?

End with a clear **recommendation** and any thesis status change if warranted.

After your analysis, output a structured JSON block fenced with ```artifacts
{{
  "evaluation_draft": {{
    "ticker": "{ticker}",
    "thesis_status": "active|under_review|invalidated",
    "technical_read": "...",
    "fundamental_read": "...",
    "action": "hold|add|trim|exit",
    "confidence": "high|medium|low",
    "key_developments": ["..."],
    "earnings_note": "..."
  }},
  "action_items": [
    {{"description": "...", "action_type": "review|resize|research|exit|enter|hedge|other", "urgency": "low|normal|high|urgent"}}
  ],
  "catalyst_updates": [
    {{"catalyst_id": null, "description": "...", "status": "pending|played_out|failed|superseded"}}
  ]
}}
```
"""


def run_post_earnings_review(
    ticker: str,
    actor: Actor | None = None,
    workflow_run_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Execute post-earnings review workflow."""
    ticker = ticker.upper()

    calls: list[tuple[str, dict[str, Any]]] = [
        ("get_thesis", {"ticker": ticker}),
        ("get_thesis_evaluations", {"ticker": ticker}),
        ("get_portfolio", {}),
        ("get_financials", {"ticker": ticker}),
    ]

    actor = actor or admin_actor(source="workflow")
    results = _exec_parallel(
        calls,
        actor=actor,
        workflow_run_id=workflow_run_id,
        workflow_name="post_earnings_review",
    )
    for name, _parsed, elapsed in results:
        logger.info("workflow=post_earnings_review ticker=%s tool=%s duration_ms=%.1f", ticker, name, elapsed)

    sections, data_block = _build_sections(results)
    synthesis_prompt = _POST_EARNINGS_SYNTHESIS.format(ticker=ticker) + f"\n\n---\n\n{data_block}"
    return synthesis_prompt, sections


# ---------------------------------------------------------------------------
# Workflow: Weekly Portfolio Review
# ---------------------------------------------------------------------------

_WEEKLY_PORTFOLIO_SYNTHESIS = """\
You are conducting a weekly portfolio review.

Given the data below, provide:

1. **Portfolio Health** – Overall P&L, position count, regime context
2. **Risk Assessment** – Which positions are under the most thesis pressure? Why?
3. **Macro Alignment** – Does the current regime support the portfolio's positioning?
4. **Positions Needing Attention** – Rank the top 3-5 positions that need review this week
5. **Suggested Actions** – Specific action items for the week ahead

End with a prioritized list of **action items** for the week.

After your analysis, output a structured JSON block fenced with ```artifacts
{{
  "action_items": [
    {{"ticker": "...", "description": "...", "action_type": "review|resize|research|exit|enter|hedge|other", "urgency": "low|normal|high|urgent"}}
  ],
  "watch_triggers": [
    {{"condition": "...", "trigger_type": "price_level|technical|fundamental|event|macro|custom", "ticker": null}}
  ]
}}
```
"""


def run_weekly_portfolio_review(
    actor: Actor | None = None,
    workflow_run_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Execute weekly portfolio review workflow."""
    calls: list[tuple[str, dict[str, Any]]] = [
        ("get_portfolio", {}),
        ("get_signal_aggregator", {}),
        ("query_ontology", {}),
    ]

    actor = actor or admin_actor(source="workflow")
    results = _exec_parallel(
        calls,
        actor=actor,
        workflow_run_id=workflow_run_id,
        workflow_name="weekly_portfolio_review",
    )
    for name, _parsed, elapsed in results:
        logger.info("workflow=weekly_portfolio_review tool=%s duration_ms=%.1f", name, elapsed)

    # Also fetch latest evaluations for all tickers
    try:
        _eval_str, eval_parsed, eval_elapsed = _exec_tool(
            "get_thesis_evaluations",
            {"ticker": "__all__"},
            actor=actor,
            provenance_context=_workflow_tool_context(
                workflow_run_id=workflow_run_id,
                workflow_name="weekly_portfolio_review",
                tool_index=len(results),
                tool_name="get_thesis_evaluations",
            ),
        )
        results.append(("get_thesis_evaluations", eval_parsed, eval_elapsed))
    except Exception:
        pass

    sections, data_block = _build_sections(results)
    synthesis_prompt = _WEEKLY_PORTFOLIO_SYNTHESIS + f"\n\n---\n\n{data_block}"
    return synthesis_prompt, sections


# ---------------------------------------------------------------------------
# Workflow: Thesis Invalidation Check
# ---------------------------------------------------------------------------

_THESIS_INVALIDATION_SYNTHESIS = """\
You are checking if the investment thesis for {ticker} is at risk of invalidation.

Given the data below — including the explicit kill conditions — assess:

1. **Kill Condition Status** – For each kill condition, is it approaching, triggered, or safely distant? Cite evidence.
2. **Thesis Integrity** – Is the core thesis still intact? What would change your mind?
3. **Risk Score Context** – What does the ontology risk assessment show?
4. **Evaluation Trend** – Has conviction been declining? Any pattern?
5. **Recent News** – Are there recent news events bearing on kill conditions or thesis integrity?
6. **Recommendation** – Should the thesis status be changed? If so, to what?

Be honest and direct. If the thesis is invalidated, say so clearly.

After your analysis, output a structured JSON block fenced with ```artifacts
{{
  "kill_condition_updates": [
    {{"kill_condition_id": null, "condition": "...", "status": "active|triggered|retired"}}
  ],
  "action_items": [
    {{"description": "...", "action_type": "review|resize|research|exit|enter|hedge|other", "urgency": "low|normal|high|urgent"}}
  ],
  "thesis_status_change": {{
    "new_status": "active|under_review|invalidated",
    "reason": "..."
  }}
}}
```
"""


def run_thesis_invalidation_check(
    ticker: str,
    actor: Actor | None = None,
    workflow_run_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Execute thesis invalidation check workflow."""
    ticker = ticker.upper()

    calls: list[tuple[str, dict[str, Any]]] = [
        ("get_thesis", {"ticker": ticker}),
        ("get_thesis_evaluations", {"ticker": ticker}),
        ("query_ontology", {"filters": {"tickers": [ticker]}}),
        ("search_web", {"query": f"{ticker} recent news risks regulatory"}),
    ]

    actor = actor or admin_actor(source="workflow")
    results = _exec_parallel(
        calls,
        actor=actor,
        workflow_run_id=workflow_run_id,
        workflow_name="thesis_invalidation_check",
    )
    for name, _parsed, elapsed in results:
        logger.info("workflow=thesis_invalidation_check ticker=%s tool=%s duration_ms=%.1f", ticker, name, elapsed)

    # Also fetch kill conditions from core_db
    try:
        from portfolio.core_db import get_kill_conditions

        kcs = get_kill_conditions(ticker)
        results.append(("kill_conditions", {"ticker": ticker, "conditions": kcs}, 0.0))
    except Exception:
        pass

    sections, data_block = _build_sections(results)
    synthesis_prompt = _THESIS_INVALIDATION_SYNTHESIS.format(ticker=ticker) + f"\n\n---\n\n{data_block}"
    return synthesis_prompt, sections


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_sections(results: list[tuple[str, dict, float]]) -> tuple[list[dict[str, Any]], str]:
    """Build tool data sections and a text block for the synthesis prompt."""
    sections: list[dict[str, Any]] = []
    data_text_parts: list[str] = []
    for name, parsed, elapsed in results:
        section = {"tool": name, "data": parsed, "duration_ms": elapsed}
        sections.append(section)
        data_text_parts.append(f"### {name}\n```json\n{json.dumps(parsed, indent=1, default=str)[:6000]}\n```")
    data_block = "\n\n".join(data_text_parts)
    return sections, data_block


def _record_workflow_tool_provenance(
    *,
    run_id: str,
    workflow_name: str,
    sections: list[dict[str, Any]],
    actor: Actor | None,
) -> None:
    try:
        from api import provenance

        workflow_event_id = provenance.deterministic_id("pv:workflow_run", run_id)
        for idx, section in enumerate(sections):
            tool_name = str(section.get("tool") or f"tool_{idx}")
            tool_event_id = provenance.deterministic_id("pv:workflow_tool_call", run_id, idx, tool_name)
            data = section.get("data")
            existing_event_id: str | None = None
            if isinstance(data, dict):
                meta = data.get("_meta")
                if isinstance(meta, dict) and meta.get("provenance_event_id"):
                    existing_event_id = str(meta["provenance_event_id"])
            if existing_event_id:
                provenance.link_refs(
                    event_id=existing_event_id,
                    source_ref_type=provenance.REF_WORKFLOW_RUN,
                    source_ref_id=run_id,
                    target_ref_type=provenance.REF_TOOL_CALL,
                    target_ref_id=existing_event_id,
                    link_type=provenance.LINK_EXECUTED,
                    metadata={"workflow_name": workflow_name, "tool_index": idx, "tool": tool_name},
                )
                continue
            if isinstance(data, dict):
                data_shape: dict[str, Any] = {
                    "result_type": "dict",
                    "result_keys": sorted(str(key) for key in data.keys())[:50],
                }
            elif isinstance(data, list):
                data_shape = {"result_type": "list", "result_count": len(data)}
            else:
                data_shape = {"result_type": type(data).__name__}

            provenance.start_event(
                event_id=tool_event_id,
                event_type=provenance.EVENT_TOOL_CALL,
                event_name=tool_name,
                actor=actor,
                parent_event_id=workflow_event_id,
                workflow_run_id=run_id,
                summary={
                    "workflow_name": workflow_name,
                    "tool": tool_name,
                    "status": "succeeded",
                    "duration_ms": section.get("duration_ms"),
                },
                metadata=data_shape,
            )
            provenance.finish_event(
                tool_event_id,
                status="succeeded",
                output_value=data,
                summary={
                    "workflow_name": workflow_name,
                    "tool": tool_name,
                    "status": "succeeded",
                    "duration_ms": section.get("duration_ms"),
                },
                metadata=data_shape,
            )
            provenance.link_refs(
                event_id=tool_event_id,
                source_ref_type=provenance.REF_WORKFLOW_RUN,
                source_ref_id=run_id,
                target_ref_type=provenance.REF_TOOL_CALL,
                target_ref_id=tool_event_id,
                link_type=provenance.LINK_EXECUTED,
                metadata={"workflow_name": workflow_name, "tool_index": idx},
            )
    except Exception:
        logger.debug("Failed to record workflow tool provenance run_id=%s", run_id, exc_info=True)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_WORKFLOW_RUNNERS = {
    "morning_brief": lambda _, actor, run_id: run_morning_brief(actor=actor, workflow_run_id=run_id),
    "thesis_review": lambda t, actor, run_id: run_thesis_review(t, actor=actor, workflow_run_id=run_id),
    "pre_earnings": lambda t, actor, run_id: run_pre_earnings(t, actor=actor, workflow_run_id=run_id),
    "post_earnings_review": lambda t, actor, run_id: run_post_earnings_review(
        t,
        actor=actor,
        workflow_run_id=run_id,
    ),
    "weekly_portfolio_review": lambda _, actor, run_id: run_weekly_portfolio_review(
        actor=actor,
        workflow_run_id=run_id,
    ),
    "thesis_invalidation_check": lambda t, actor, run_id: run_thesis_invalidation_check(
        t,
        actor=actor,
        workflow_run_id=run_id,
    ),
}


def execute_workflow(
    workflow_name: str,
    ticker: str | None = None,
    actor: Actor | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Run a named workflow. Returns (run_id, synthesis_prompt, tool_data_sections).

    Creates a persistent WorkflowRun record. Caller is responsible for calling
    complete_workflow_run() or fail_workflow_run() after synthesis.

    Raises ValueError if workflow_name is unknown or ticker is required but missing.
    """
    wf = AVAILABLE_WORKFLOWS.get(workflow_name)
    if wf is None:
        raise ValueError(f"Unknown workflow: {workflow_name}. Available: {list(AVAILABLE_WORKFLOWS)}")

    if wf["requires_ticker"] and not ticker:
        raise ValueError(f"Workflow '{workflow_name}' requires a ticker parameter")

    runner = _WORKFLOW_RUNNERS.get(workflow_name)
    if runner is None:
        raise ValueError(f"Workflow '{workflow_name}' is defined but has no implementation")

    # Governed workflows must have a durable run id before tool execution.
    from portfolio.core_db import create_workflow_run

    run = create_workflow_run(workflow_name, ticker)
    run_id = run["run_id"]
    emit_audit_event(
        "workflow.execution.started",
        "workflow",
        "started",
        actor=actor or admin_actor(source="workflow"),
        object_refs=[{"type": "workflow_run", "id": run_id}, {"type": "workflow", "id": workflow_name}],
        after_summary={"workflow_name": workflow_name, "ticker": ticker, "status": "running"},
        source_lineage={"run_id": run_id},
        fail_closed=True,
        criticality="financial_critical",
        lineage_root_id=f"workflow_run:{run_id}",
        idempotency_key=f"workflow.execution.started:{run_id}",
        retention_class="financial_lineage_7y",
    )

    effective_actor = actor or admin_actor(source="workflow")
    synthesis_prompt, sections = runner(ticker, effective_actor, run_id)
    _record_workflow_tool_provenance(
        run_id=run_id,
        workflow_name=workflow_name,
        sections=sections,
        actor=effective_actor,
    )
    return run_id, synthesis_prompt, sections
