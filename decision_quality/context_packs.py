"""Opportunity-type-specific context pack registry and resolver."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from api.tool_data_quality import PRICE_CONFIRMATION_TOOLS, normalize_tool_quality
from decision_quality.models import OpportunityType

ContextPackId = Literal[
    "catalyst",
    "quality_entry",
    "turnaround",
    "dislocation",
    "hedge",
    "event_driven",
    "short",
    "credit_liquidity",
    "macro_rates",
]

CONTEXT_PACK_IDS: frozenset[str] = frozenset(
    {
        "catalyst",
        "quality_entry",
        "turnaround",
        "dislocation",
        "hedge",
        "event_driven",
        "short",
        "credit_liquidity",
        "macro_rates",
    }
)

DEFAULT_CONTEXT_PACK: ContextPackId = "quality_entry"


@dataclass(frozen=True)
class ContextPackTemplate:
    pack_id: ContextPackId
    label: str
    opportunity_types: tuple[OpportunityType, ...]
    required_tools: tuple[str, ...]
    conditional_tools: tuple[str, ...] = ()
    required_source_classes: tuple[str, ...] = ()
    min_reliability_tier: str = "standard"
    required_dq_dimensions: tuple[str, ...] = ()
    missing_input_labels: dict[str, str] = field(default_factory=dict)

    def missing_label_for_tool(self, tool_name: str) -> str:
        return self.missing_input_labels.get(tool_name, f"{tool_name} context")


CONTEXT_PACKS: dict[str, ContextPackTemplate] = {
    "quality_entry": ContextPackTemplate(
        pack_id="quality_entry",
        label="Quality compounder entry",
        opportunity_types=("quality_compounder", "undervalued_asset"),
        required_tools=("get_portfolio", "get_dossier", "get_thesis", "get_position_valuation", "run_chart"),
        conditional_tools=("get_thesis_evaluations", "search_knowledge_base"),
        required_source_classes=("thesis", "valuation", "price_action"),
        min_reliability_tier="standard",
        required_dq_dimensions=(
            "simple_thesis",
            "mispricing",
            "catalyst_or_reason_now",
            "evidence_for",
            "evidence_against",
            "price_action",
            "invalidation",
            "missing_inputs",
            "confidence_sizing",
            "trade_after_trade",
        ),
        missing_input_labels={
            "get_thesis": "current thesis source",
            "get_position_valuation": "valuation and bear/base/bull framing",
            "run_chart": "price-action confirmation",
            "get_dossier": "position dossier context",
        },
    ),
    "catalyst": ContextPackTemplate(
        pack_id="catalyst",
        label="Catalyst status",
        opportunity_types=("policy_inflection", "regime_shift"),
        required_tools=("get_dossier", "run_chart", "search_web"),
        conditional_tools=("get_thesis", "search_knowledge_base"),
        required_source_classes=("catalyst", "price_action", "news"),
        min_reliability_tier="standard",
        required_dq_dimensions=("catalyst_or_reason_now", "price_action", "missing_inputs"),
        missing_input_labels={
            "search_web": "fresh catalyst verification",
            "run_chart": "price reaction to catalyst",
            "get_dossier": "issuer dossier context",
        },
    ),
    "turnaround": ContextPackTemplate(
        pack_id="turnaround",
        label="Turnaround setup",
        opportunity_types=("undervalued_asset", "cyclical_upturn"),
        required_tools=("get_dossier", "get_thesis", "get_position_valuation", "run_chart"),
        conditional_tools=("get_thesis_evaluations", "search_knowledge_base"),
        required_source_classes=("thesis", "valuation", "price_action"),
        min_reliability_tier="standard",
        required_dq_dimensions=("mispricing", "catalyst_or_reason_now", "price_action", "invalidation"),
        missing_input_labels={
            "get_thesis": "turnaround thesis source",
            "get_position_valuation": "normalized valuation context",
            "run_chart": "inflection price confirmation",
        },
    ),
    "dislocation": ContextPackTemplate(
        pack_id="dislocation",
        label="Factor or narrative dislocation",
        opportunity_types=("reflexive_process", "cyclical_upturn", "regime_shift"),
        required_tools=("get_dossier", "run_chart", "get_price_volume_signals"),
        conditional_tools=("get_position_valuation", "get_thesis", "search_knowledge_base"),
        required_source_classes=("price_action", "positioning"),
        min_reliability_tier="standard",
        required_dq_dimensions=("mispricing", "price_action", "missing_inputs"),
        missing_input_labels={
            "run_chart": "dislocation price confirmation",
            "get_price_volume_signals": "volume and positioning confirmation",
            "get_dossier": "issuer dossier context",
        },
    ),
    "hedge": ContextPackTemplate(
        pack_id="hedge",
        label="Portfolio hedge",
        opportunity_types=("regime_shift",),
        required_tools=("get_portfolio", "run_chart", "get_price_volume_signals"),
        conditional_tools=("get_portfolio_risk", "query_ontology"),
        required_source_classes=("portfolio", "price_action"),
        min_reliability_tier="standard",
        required_dq_dimensions=("price_action", "missing_inputs", "confidence_sizing"),
        missing_input_labels={
            "get_portfolio": "portfolio exposure context",
            "run_chart": "hedge instrument price confirmation",
            "get_price_volume_signals": "macro/hedge price confirmation",
        },
    ),
    "event_driven": ContextPackTemplate(
        pack_id="event_driven",
        label="Event-driven setup",
        opportunity_types=("policy_inflection",),
        required_tools=("get_dossier", "search_web", "run_chart"),
        conditional_tools=("get_thesis", "search_knowledge_base"),
        required_source_classes=("catalyst", "news", "price_action"),
        min_reliability_tier="standard",
        required_dq_dimensions=("catalyst_or_reason_now", "price_action", "invalidation"),
        missing_input_labels={
            "search_web": "event timeline verification",
            "run_chart": "event price reaction",
        },
    ),
    "short": ContextPackTemplate(
        pack_id="short",
        label="Short or avoid setup",
        opportunity_types=("unsustainable_process", "crowded_narrative_avoid"),
        required_tools=("get_dossier", "get_thesis", "run_chart", "get_position_valuation"),
        conditional_tools=("get_thesis_evaluations", "search_knowledge_base"),
        required_source_classes=("thesis", "valuation", "price_action", "crowding"),
        min_reliability_tier="standard",
        required_dq_dimensions=("mispricing", "evidence_against", "price_action", "invalidation"),
        missing_input_labels={
            "get_thesis": "short thesis source",
            "run_chart": "breakdown price confirmation",
            "get_position_valuation": "valuation stretch evidence",
        },
    ),
    "credit_liquidity": ContextPackTemplate(
        pack_id="credit_liquidity",
        label="Credit or liquidity stress",
        opportunity_types=("forced_liquidation",),
        required_tools=("get_dossier", "get_position_valuation", "search_web"),
        conditional_tools=("run_chart", "search_knowledge_base"),
        required_source_classes=("valuation", "liquidity", "news"),
        min_reliability_tier="standard",
        required_dq_dimensions=("mispricing", "catalyst_or_reason_now", "missing_inputs"),
        missing_input_labels={
            "search_web": "liquidity or credit event verification",
            "get_position_valuation": "distressed valuation context",
        },
    ),
    "macro_rates": ContextPackTemplate(
        pack_id="macro_rates",
        label="Macro and rates context",
        opportunity_types=("regime_shift", "policy_inflection"),
        required_tools=("get_yield_curve", "get_bond_dashboard", "get_price_volume_signals"),
        conditional_tools=("search_knowledge_base", "search_web"),
        required_source_classes=("macro", "rates"),
        min_reliability_tier="supplemental",
        required_dq_dimensions=("catalyst_or_reason_now", "missing_inputs", "confidence_sizing"),
        missing_input_labels={
            "get_yield_curve": "current yield curve shape",
            "get_bond_dashboard": "rates and spread context",
            "get_price_volume_signals": "cross-asset price confirmation",
        },
    ),
}

_OPPORTUNITY_TYPE_TO_PACK: dict[str, ContextPackId] = {}
for _pack in CONTEXT_PACKS.values():
    for _otype in _pack.opportunity_types:
        if _otype not in _OPPORTUNITY_TYPE_TO_PACK:
            _OPPORTUNITY_TYPE_TO_PACK[_otype] = _pack.pack_id

_TOOL_PACK_TO_CONTEXT_PACK: dict[str, ContextPackId] = {
    "thesis_review": "quality_entry",
    "catalyst_status": "catalyst",
    "opportunity_scan": "dislocation",
    "macro_stance": "macro_rates",
    "portfolio_query": "hedge",
}

_INTENT_CLASS_TO_CONTEXT_PACK: dict[str, ContextPackId] = {
    "thesis_review": "quality_entry",
    "catalyst_status": "catalyst",
    "opportunity_discovery": "dislocation",
    "portfolio_query": "hedge",
    "general_research": "macro_rates",
}

_KEYWORD_PACK_RULES: tuple[tuple[re.Pattern[str], ContextPackId], ...] = (
    (re.compile(r"\b(catalyst|played out|materiali[sz]ed|reason[- ]now)\b", re.I), "catalyst"),
    (re.compile(r"\b(compounder|quality|durable|franchise)\b", re.I), "quality_entry"),
    (re.compile(r"\b(turnaround|restructur|inflection)\b", re.I), "turnaround"),
    (re.compile(r"\b(dislocation|oversold|factor|mean reversion)\b", re.I), "dislocation"),
    (re.compile(r"\b(hedge|beta|net exposure|gross exposure)\b", re.I), "hedge"),
    (re.compile(r"\b(event[- ]driven|earnings|filing|approval)\b", re.I), "event_driven"),
    (re.compile(r"\b(short|overvalued|crowded|avoid)\b", re.I), "short"),
    (re.compile(r"\b(credit|liquidity|distressed|default)\b", re.I), "credit_liquidity"),
    (re.compile(r"\b(yield curve|rates|macro|fed|central bank)\b", re.I), "macro_rates"),
)


def get_context_pack(pack_id: str | None) -> ContextPackTemplate:
    normalized = str(pack_id or DEFAULT_CONTEXT_PACK).strip().lower()
    if normalized in CONTEXT_PACKS:
        return CONTEXT_PACKS[normalized]
    return CONTEXT_PACKS[DEFAULT_CONTEXT_PACK]


def _screen_ticker(screen_context: dict[str, Any] | None) -> str | None:
    if not isinstance(screen_context, dict):
        return None
    ticker = str(screen_context.get("ticker") or "").strip().upper()
    return ticker or None


def _extract_ticker(user_text: str, screen_context: dict[str, Any] | None) -> str | None:
    ticker = _screen_ticker(screen_context)
    if ticker:
        return ticker
    stop = {"AND", "THE", "FOR", "MY", "ALL", "HOW", "CAN", "ARE", "HAS", "DO"}
    matches = re.findall(r"\b([A-Z]{1,5})\b", user_text or "")
    for match in matches:
        if match not in stop and len(match) >= 2:
            return match
    return None


def resolve_context_pack(
    *,
    user_text: str,
    intent_class: str | None = None,
    tool_pack: str | None = None,
    screen_context: dict[str, Any] | None = None,
    opportunity_candidate_metadata: dict[str, Any] | None = None,
) -> ContextPackTemplate:
    """Resolve the best context pack for a hidden OC/DQ pass."""
    metadata = opportunity_candidate_metadata if isinstance(opportunity_candidate_metadata, dict) else {}
    opportunity_type = str(metadata.get("opportunity_type") or "").strip().lower()
    if opportunity_type and opportunity_type in _OPPORTUNITY_TYPE_TO_PACK:
        return get_context_pack(_OPPORTUNITY_TYPE_TO_PACK[opportunity_type])

    for pattern, pack_id in _KEYWORD_PACK_RULES:
        if pattern.search(user_text or ""):
            return get_context_pack(pack_id)

    normalized_tool_pack = str(tool_pack or "").strip().lower()
    if normalized_tool_pack in _TOOL_PACK_TO_CONTEXT_PACK:
        return get_context_pack(_TOOL_PACK_TO_CONTEXT_PACK[normalized_tool_pack])
    if normalized_tool_pack in CONTEXT_PACK_IDS:
        return get_context_pack(normalized_tool_pack)

    normalized_intent = str(intent_class or "").strip().lower()
    if normalized_intent in _INTENT_CLASS_TO_CONTEXT_PACK:
        pack = get_context_pack(_INTENT_CLASS_TO_CONTEXT_PACK[normalized_intent])
        if pack.pack_id == "macro_rates" and _extract_ticker(user_text, screen_context):
            return get_context_pack("quality_entry")
        return pack

    if not _extract_ticker(user_text, screen_context) and re.search(
        r"\b(yield|rates|macro|curve|bond)\b", user_text or "", flags=re.IGNORECASE
    ):
        return get_context_pack("macro_rates")

    return get_context_pack(DEFAULT_CONTEXT_PACK)


def _wants_fresh_data(user_text: str) -> bool:
    return bool(
        re.search(
            r"\b(latest|today|current|now|recent|news|catalyst|regulatory|approval|litigation)\b",
            user_text or "",
            flags=re.IGNORECASE,
        )
    )


def build_context_pack_tool_calls(
    *,
    pack: ContextPackTemplate,
    user_text: str,
    screen_context: dict[str, Any] | None = None,
    allowed_tool_names: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Build hidden chat tool calls for a resolved context pack."""
    allowed = set(allowed_tool_names or ())
    ticker = _extract_ticker(user_text, screen_context)
    calls: list[dict[str, Any]] = []

    def add(name: str, args: dict[str, Any]) -> None:
        if allowed and name not in allowed:
            return
        calls.append(
            {
                "name": name,
                "args": args,
                "call_ids": [f"decision-quality-chat:{name}:{len(calls)}"],
            }
        )

    required = list(pack.required_tools) + list(pack.conditional_tools)
    seen: set[str] = set()

    for tool_name in required:
        if tool_name in seen:
            continue
        seen.add(tool_name)

        if tool_name == "get_portfolio":
            add("get_portfolio", {})
        elif tool_name == "get_portfolio_risk":
            add("get_portfolio_risk", {})
        elif tool_name == "query_ontology":
            add("query_ontology", {"query": user_text[:200]})
        elif tool_name == "get_dossier" and ticker:
            add("get_dossier", {"ticker": ticker})
        elif tool_name == "get_thesis" and ticker:
            add("get_thesis", {"ticker": ticker})
        elif tool_name == "get_thesis_evaluations" and ticker:
            add("get_thesis_evaluations", {"ticker": ticker, "limit": 5})
        elif tool_name == "get_position_valuation" and ticker:
            add("get_position_valuation", {"ticker": ticker})
        elif tool_name == "run_chart" and ticker:
            add("run_chart", {"ticker": ticker, "lookback": "1y"})
        elif tool_name == "get_price_volume_signals":
            add("get_price_volume_signals", {"ticker": ticker} if ticker else {})
        elif tool_name == "search_knowledge_base" and ticker:
            add(
                "search_knowledge_base",
                {"query": f"{ticker} thesis catalysts risks invalidation", "tickers": ticker, "top_k": 5},
            )
        elif tool_name == "search_knowledge_base":
            add("search_knowledge_base", {"query": user_text[:200], "top_k": 5})
        elif tool_name == "get_yield_curve":
            add("get_yield_curve", {})
        elif tool_name == "get_bond_dashboard":
            add("get_bond_dashboard", {})
        elif tool_name == "search_web":
            if _wants_fresh_data(user_text) or tool_name in pack.required_tools:
                query = f"{ticker or ''} {user_text}".strip()
                add("search_web", {"query": query[:300]})

    if pack.pack_id == "quality_entry" and ticker and "search_knowledge_base" not in seen:
        add(
            "search_knowledge_base",
            {"query": f"{ticker} thesis catalysts risks invalidation", "tickers": ticker, "top_k": 5},
        )

    return calls


def _tool_is_satisfied(tool_result: dict[str, Any]) -> bool:
    summary = normalize_tool_quality(tool_result)
    if summary.get("blocks_actionable"):
        return False
    if summary.get("tool_status") in {"blocked", "error", "timeout", "failed", "failed_closed", "denied"}:
        return False
    if summary.get("price_confirmation") in {"missing", "stale", "blocked"}:
        return False
    missing_fields = summary.get("missing_fields") or []
    if missing_fields:
        return False
    return True


def assess_context_pack(
    *,
    pack: ContextPackTemplate,
    tool_results: list[dict[str, Any]],
    data_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate whether a context pack is complete for graduation."""
    results_by_name = {
        str(item.get("name")): item for item in tool_results if isinstance(item, dict) and item.get("name")
    }
    required_tools = list(pack.required_tools)
    satisfied_tools = [name for name in required_tools if name in results_by_name and _tool_is_satisfied(results_by_name[name])]
    missing_tools = [name for name in required_tools if name not in satisfied_tools]

    missing_inputs: list[str] = []
    for tool_name in missing_tools:
        missing_inputs.append(pack.missing_label_for_tool(tool_name))

    dq = data_quality if isinstance(data_quality, dict) else {}
    blocking_reason_codes = [str(item) for item in dq.get("blocking_reason_codes") or []]
    if "MISSING_PRICE_CONFIRMATION" in blocking_reason_codes:
        label = "price-action confirmation"
        if label not in missing_inputs:
            missing_inputs.append(label)
    if dq.get("critical_data_quality") in {"stale", "failed"}:
        label = "fresh required source data"
        if label not in missing_inputs:
            missing_inputs.append(label)

    price_tools = [name for name in required_tools if name in PRICE_CONFIRMATION_TOOLS]
    if price_tools and not any(name in satisfied_tools for name in price_tools):
        label = "price-action confirmation"
        if label not in missing_inputs:
            missing_inputs.append(label)

    is_complete = not missing_tools and not blocking_reason_codes
    return {
        "pack_id": pack.pack_id,
        "label": pack.label,
        "opportunity_types": list(pack.opportunity_types),
        "required_tools": required_tools,
        "satisfied_tools": satisfied_tools,
        "missing_tools": missing_tools,
        "missing_inputs": missing_inputs,
        "required_dq_dimensions": list(pack.required_dq_dimensions),
        "required_source_classes": list(pack.required_source_classes),
        "min_reliability_tier": pack.min_reliability_tier,
        "is_complete": is_complete,
        "blocking_reason_codes": blocking_reason_codes,
    }


def build_context_pack_metadata(
    *,
    user_text: str,
    intent_class: str | None = None,
    tool_pack: str | None = None,
    screen_context: dict[str, Any] | None = None,
    opportunity_candidate_metadata: dict[str, Any] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    data_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pack = resolve_context_pack(
        user_text=user_text,
        intent_class=intent_class,
        tool_pack=tool_pack,
        screen_context=screen_context,
        opportunity_candidate_metadata=opportunity_candidate_metadata,
    )
    assessment = assess_context_pack(
        pack=pack,
        tool_results=tool_results or [],
        data_quality=data_quality,
    )
    return assessment
