"""LLM-assisted intent/tool router with deterministic regex fallback."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from llm_utils import MODEL_LOW, call_llm_json

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.70

INTENT_CLASSES = frozenset(
    {
        "thesis_review",
        "opportunity_discovery",
        "catalyst_status",
        "portfolio_query",
        "workflow_handoff",
        "general_research",
        "casual",
    }
)

_HIGH_RISK_ACTION_RX = re.compile(
    r"\b(should i (?:buy|add|short|sell)|buy now|add now|short now|sell now)\b",
    flags=re.IGNORECASE,
)


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def intent_router_enabled() -> bool:
    return _env_flag("AGENT_INTENT_ROUTER_ENABLED", default=False)


def intent_router_shadow_mode() -> bool:
    return _env_flag("AGENT_INTENT_ROUTER_SHADOW_MODE", default=True)


def intent_router_confidence_threshold() -> float:
    raw = os.environ.get("AGENT_INTENT_ROUTER_CONFIDENCE_THRESHOLD")
    if raw is None:
        return DEFAULT_CONFIDENCE_THRESHOLD
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE_THRESHOLD


@dataclass(frozen=True)
class RouteDecision:
    intent_class: str
    run_hidden_dq: bool
    run_opportunity_preflight: bool
    workflow_name: str | None
    workflow_ticker: str | None
    tool_names: list[str]
    confidence: float
    source: str
    fallback_reason: str | None = None
    tool_pack: str | None = None

    def to_meta(self) -> dict[str, Any]:
        return {
            "intent_class": self.intent_class,
            "run_hidden_dq": self.run_hidden_dq,
            "run_opportunity_preflight": self.run_opportunity_preflight,
            "workflow_name": self.workflow_name,
            "workflow_ticker": self.workflow_ticker,
            "tool_names": list(self.tool_names),
            "tool_pack": self.tool_pack,
            "confidence": self.confidence,
            "source": self.source,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class RouteContext:
    user_text: str
    screen_context: dict[str, Any] | None = None
    recent_session_features: list[dict[str, Any]] = field(default_factory=list)
    opportunity_candidate_metadata: dict[str, Any] | None = None
    allowed_tool_names: tuple[str, ...] = ()
    workflow_hints: tuple[str, ...] = ()


def intent_router_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "intent_class",
            "run_hidden_decision_quality",
            "run_opportunity_candidate_preflight",
            "workflow_name",
            "workflow_ticker",
            "required_tool_names",
            "confidence",
            "tool_pack",
        ],
        "properties": {
            "intent_class": {
                "type": "string",
                "enum": sorted(INTENT_CLASSES),
            },
            "run_hidden_decision_quality": {"type": "boolean"},
            "run_opportunity_candidate_preflight": {"type": "boolean"},
            "workflow_name": {"type": ["string", "null"]},
            "workflow_ticker": {"type": ["string", "null"]},
            "required_tool_names": {
                "type": "array",
                "items": {"type": "string"},
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "tool_pack": {"type": ["string", "null"]},
        },
    }


def _normalize_tool_names(
    names: list[str] | None,
    *,
    allowed: set[str],
    baseline: list[str],
) -> list[str]:
    selected: list[str] = []
    for name in names or []:
        if name in allowed and name not in selected:
            selected.append(name)
    if not selected:
        return list(baseline)
    if "search_agent_capabilities" in allowed and "search_agent_capabilities" not in selected:
        selected.append("search_agent_capabilities")
    return selected


def _infer_intent_from_flags(
    *,
    run_hidden_dq: bool,
    run_opportunity_preflight: bool,
    workflow_name: str | None,
    user_text: str,
) -> str:
    if workflow_name:
        return "workflow_handoff"
    lowered = (user_text or "").lower()
    if run_hidden_dq:
        return "thesis_review"
    if run_opportunity_preflight:
        if re.search(r"\b(catalyst|played out|materiali[sz]ed|status)\b", lowered):
            return "catalyst_status"
        if re.search(r"\b(scan|scout|discover|opportunit)\b", lowered):
            return "opportunity_discovery"
        return "opportunity_discovery"
    if re.search(r"\b(portfolio|holding|position|p&l|pnl|exposure)\b", lowered):
        return "portfolio_query"
    if re.search(r"\b(catalyst|played out|materiali[sz]ed|status)\b", lowered):
        return "catalyst_status"
    return "general_research"


def build_regex_route_decision(
    *,
    user_text: str,
    select_tool_names: Callable[[str], list[str]],
    detect_workflow: Callable[[str, Any], tuple[str | None, str | None]],
    should_run_hidden_dq: Callable[[str, Any], bool],
    should_run_opportunity_preflight: Callable[[str, Any], bool],
    screen_context: Any = None,
) -> RouteDecision:
    workflow_name, workflow_ticker = detect_workflow(user_text, screen_context)
    tool_names = select_tool_names(user_text)
    run_hidden_dq = should_run_hidden_dq(user_text, screen_context)
    run_opportunity_preflight = should_run_opportunity_preflight(user_text, screen_context)
    intent_class = _infer_intent_from_flags(
        run_hidden_dq=run_hidden_dq,
        run_opportunity_preflight=run_opportunity_preflight,
        workflow_name=workflow_name,
        user_text=user_text,
    )
    return RouteDecision(
        intent_class=intent_class,
        run_hidden_dq=run_hidden_dq,
        run_opportunity_preflight=run_opportunity_preflight,
        workflow_name=workflow_name,
        workflow_ticker=workflow_ticker,
        tool_names=tool_names,
        confidence=1.0,
        source="regex",
        fallback_reason=None,
        tool_pack="regex_baseline",
    )


def _enforce_safety_floor(
    decision: RouteDecision,
    *,
    regex_baseline: RouteDecision,
    user_text: str,
) -> RouteDecision:
    """Never let the LLM router disable safety paths the regex baseline would run."""
    run_hidden_dq = decision.run_hidden_dq or regex_baseline.run_hidden_dq
    run_opportunity_preflight = decision.run_opportunity_preflight or regex_baseline.run_opportunity_preflight
    workflow_name = regex_baseline.workflow_name or decision.workflow_name
    workflow_ticker = regex_baseline.workflow_ticker or decision.workflow_ticker
    if _HIGH_RISK_ACTION_RX.search(user_text or ""):
        run_hidden_dq = True
    tool_names = decision.tool_names
    if regex_baseline.run_hidden_dq or regex_baseline.run_opportunity_preflight:
        merged: list[str] = []
        for name in regex_baseline.tool_names + tool_names:
            if name not in merged:
                merged.append(name)
        tool_names = merged
    return RouteDecision(
        intent_class=decision.intent_class,
        run_hidden_dq=run_hidden_dq,
        run_opportunity_preflight=run_opportunity_preflight,
        workflow_name=workflow_name,
        workflow_ticker=workflow_ticker,
        tool_names=tool_names,
        confidence=decision.confidence,
        source=decision.source,
        fallback_reason=decision.fallback_reason,
        tool_pack=decision.tool_pack,
    )


def _parse_llm_route(
    parsed: Any,
    *,
    context: RouteContext,
    regex_baseline: RouteDecision,
) -> RouteDecision | None:
    if not isinstance(parsed, dict):
        return None
    try:
        confidence = float(parsed.get("confidence"))
    except (TypeError, ValueError):
        return None
    intent_class = str(parsed.get("intent_class") or "general_research")
    if intent_class not in INTENT_CLASSES:
        intent_class = "general_research"
    allowed = set(context.allowed_tool_names) or set(regex_baseline.tool_names)
    tool_names = _normalize_tool_names(
        [str(item) for item in parsed.get("required_tool_names") or [] if str(item).strip()],
        allowed=allowed,
        baseline=regex_baseline.tool_names,
    )
    workflow_name = parsed.get("workflow_name")
    workflow_ticker = parsed.get("workflow_ticker")
    wf_name = str(workflow_name).strip() if isinstance(workflow_name, str) and workflow_name.strip() else None
    wf_ticker = (
        str(workflow_ticker).strip().upper() if isinstance(workflow_ticker, str) and workflow_ticker.strip() else None
    )
    decision = RouteDecision(
        intent_class=intent_class,
        run_hidden_dq=bool(parsed.get("run_hidden_decision_quality")),
        run_opportunity_preflight=bool(parsed.get("run_opportunity_candidate_preflight")),
        workflow_name=wf_name,
        workflow_ticker=wf_ticker,
        tool_names=tool_names,
        confidence=max(0.0, min(1.0, confidence)),
        source="llm",
        fallback_reason=None,
        tool_pack=str(parsed.get("tool_pack") or intent_class),
    )
    return _enforce_safety_floor(decision, regex_baseline=regex_baseline, user_text=context.user_text)


def _build_router_prompt(context: RouteContext, regex_baseline: RouteDecision) -> str:
    payload = {
        "user_message": context.user_text,
        "screen_context": context.screen_context,
        "recent_session_features": context.recent_session_features[-6:],
        "opportunity_candidate_metadata": context.opportunity_candidate_metadata,
        "allowed_tools": list(context.allowed_tool_names) or list(regex_baseline.tool_names),
        "workflow_hints": list(context.workflow_hints),
        "regex_baseline": regex_baseline.to_meta(),
    }
    return (
        "Classify routing for this Stan chat turn. "
        "Return one JSON object only.\n\n"
        f"Routing payload:\n{json.dumps(payload, ensure_ascii=True, indent=2, default=str)}"
    )


def run_llm_route_decision(
    *,
    context: RouteContext,
    regex_baseline: RouteDecision,
    provider: str,
    api_key: str,
    system_prompt: str,
    reasoning_effort: str | None = None,
) -> RouteDecision | None:
    parsed, _citations, _response, diagnostics = call_llm_json(
        prompt=_build_router_prompt(context, regex_baseline),
        model=MODEL_LOW,
        api_key=api_key,
        max_tokens=900,
        system=system_prompt,
        provider=provider,
        enable_web_search=False,
        reasoning_effort=reasoning_effort,
        json_schema=intent_router_schema(),
        json_schema_name="agent_intent_router",
    )
    if diagnostics.get("error"):
        logger.warning("intent_router_llm_error error=%s", diagnostics.get("error"))
        return None
    return _parse_llm_route(parsed, context=context, regex_baseline=regex_baseline)


def compare_route_decisions(
    *,
    applied: RouteDecision,
    candidate: RouteDecision,
) -> dict[str, Any]:
    return {
        "intent_match": applied.intent_class == candidate.intent_class,
        "hidden_dq_match": applied.run_hidden_dq == candidate.run_hidden_dq,
        "opportunity_preflight_match": applied.run_opportunity_preflight == candidate.run_opportunity_preflight,
        "workflow_match": applied.workflow_name == candidate.workflow_name,
        "tool_overlap": sorted(set(applied.tool_names) & set(candidate.tool_names)),
        "tool_only_in_applied": sorted(set(applied.tool_names) - set(candidate.tool_names)),
        "tool_only_in_candidate": sorted(set(candidate.tool_names) - set(applied.tool_names)),
        "applied_source": applied.source,
        "candidate_source": candidate.source,
        "candidate_confidence": candidate.confidence,
    }


def resolve_agent_route(
    *,
    context: RouteContext,
    regex_baseline: RouteDecision,
    provider: str | None = None,
    api_key: str | None = None,
    system_prompt: str | None = None,
    reasoning_effort: str | None = None,
) -> tuple[RouteDecision, dict[str, Any]]:
    """Resolve effective routing and telemetry metadata."""
    threshold = intent_router_confidence_threshold()
    meta: dict[str, Any] = {
        "enabled": intent_router_enabled(),
        "shadow_mode": intent_router_shadow_mode(),
        "confidence_threshold": threshold,
        "regex_baseline": regex_baseline.to_meta(),
    }

    if not intent_router_enabled():
        meta["applied_source"] = "regex"
        meta["llm_skipped"] = True
        return regex_baseline, meta

    if not provider or not api_key or not system_prompt:
        meta["applied_source"] = "regex"
        meta["llm_skipped"] = True
        meta["fallback_reason"] = "missing_llm_credentials"
        return regex_baseline, meta

    llm_decision = run_llm_route_decision(
        context=context,
        regex_baseline=regex_baseline,
        provider=provider,
        api_key=api_key,
        system_prompt=system_prompt,
        reasoning_effort=reasoning_effort,
    )
    if llm_decision is None:
        meta["applied_source"] = "regex"
        meta["llm_skipped"] = False
        meta["fallback_reason"] = "llm_parse_or_call_failed"
        return (
            RouteDecision(
                intent_class=regex_baseline.intent_class,
                run_hidden_dq=regex_baseline.run_hidden_dq,
                run_opportunity_preflight=regex_baseline.run_opportunity_preflight,
                workflow_name=regex_baseline.workflow_name,
                workflow_ticker=regex_baseline.workflow_ticker,
                tool_names=regex_baseline.tool_names,
                confidence=regex_baseline.confidence,
                source="regex",
                fallback_reason="llm_parse_or_call_failed",
                tool_pack=regex_baseline.tool_pack,
            ),
            meta,
        )

    meta["llm_candidate"] = llm_decision.to_meta()
    meta["shadow_comparison"] = compare_route_decisions(applied=regex_baseline, candidate=llm_decision)

    if intent_router_shadow_mode():
        meta["applied_source"] = "regex_shadow"
        meta["shadow_mode"] = True
        return regex_baseline, meta

    if llm_decision.confidence < threshold:
        meta["applied_source"] = "regex"
        meta["fallback_reason"] = "confidence_below_threshold"
        return (
            RouteDecision(
                intent_class=regex_baseline.intent_class,
                run_hidden_dq=regex_baseline.run_hidden_dq,
                run_opportunity_preflight=regex_baseline.run_opportunity_preflight,
                workflow_name=regex_baseline.workflow_name,
                workflow_ticker=regex_baseline.workflow_ticker,
                tool_names=regex_baseline.tool_names,
                confidence=llm_decision.confidence,
                source="regex",
                fallback_reason="confidence_below_threshold",
                tool_pack=regex_baseline.tool_pack,
            ),
            meta,
        )

    meta["applied_source"] = "llm"
    return llm_decision, meta


def training_row_from_telemetry(
    *,
    user_text: str,
    route_meta: dict[str, Any],
    session_id: str | None = None,
    screen_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize one shadow-mode row for offline router training/eval."""
    applied = route_meta.get("regex_baseline")
    candidate = route_meta.get("llm_candidate")
    return {
        "session_id": session_id,
        "user_text": user_text,
        "screen_context": screen_context,
        "regex_baseline": applied,
        "llm_candidate": candidate,
        "shadow_comparison": route_meta.get("shadow_comparison"),
        "applied_source": route_meta.get("applied_source"),
        "fallback_reason": route_meta.get("fallback_reason"),
        "confidence_threshold": route_meta.get("confidence_threshold"),
    }


def build_route_context(
    *,
    user_text: str,
    screen_context: Any = None,
    recent_conversation: list[dict[str, Any]] | None = None,
    opportunity_candidate_metadata: dict[str, Any] | None = None,
    allowed_tool_names: list[str] | None = None,
    workflow_hints: list[str] | None = None,
) -> RouteContext:
    screen_payload: dict[str, Any] | None = None
    if screen_context is not None and hasattr(screen_context, "model_dump"):
        screen_payload = screen_context.model_dump(mode="json")
    elif isinstance(screen_context, dict):
        screen_payload = screen_context

    session_features: list[dict[str, Any]] = []
    for item in recent_conversation or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            session_features.append({"role": role, "content": content[:400]})

    return RouteContext(
        user_text=user_text,
        screen_context=screen_payload,
        recent_session_features=session_features,
        opportunity_candidate_metadata=opportunity_candidate_metadata,
        allowed_tool_names=tuple(allowed_tool_names or ()),
        workflow_hints=tuple(workflow_hints or ()),
    )
