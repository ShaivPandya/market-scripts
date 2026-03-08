from __future__ import annotations

import json
import os
import re
from typing import Any

from ontology.models import InterpretedQuery

ALLOWED_INTENTS = {
    "portfolio_risk_exposure",
    "positions_in_deteriorating_macro",
    "entity_context",
}

_TICKER_RE = re.compile(r"\b[A-Z][A-Z0-9.\-=]{1,9}\b")
_STOP_TICKERS = {
    "AND",
    "THE",
    "WITH",
    "FROM",
    "THAT",
    "WHAT",
    "WHICH",
    "SHOW",
    "PORTFOLIO",
    "RISK",
    "EXPOSURE",
    "SIGNAL",
    "SIGNALS",
    "BREADTH",
    "VIX",
    "FX",
}


def parse_hybrid_query(
    query: str | None,
    intent: str | None,
    filters: dict[str, Any] | None,
    known_sectors: set[str] | None = None,
) -> InterpretedQuery:
    clean_filters = _coerce_filters(_as_dict(filters))

    if intent in ALLOWED_INTENTS:
        return InterpretedQuery(
            intent=intent,
            source="structured",
            filters=clean_filters,
            original_query=query,
        )

    if query:
        parsed_llm = _parse_with_llm(query)
        if parsed_llm:
            merged = _merge_filters(_as_dict(parsed_llm.get("filters")), clean_filters)
            parsed_intent = str(parsed_llm.get("intent") or "portfolio_risk_exposure")
            if parsed_intent not in ALLOWED_INTENTS:
                parsed_intent = "portfolio_risk_exposure"
            return InterpretedQuery(
                intent=parsed_intent,
                source="llm",
                filters=merged,
                entity=parsed_llm.get("entity"),
                original_query=query,
            )

        fallback = _deterministic_parse(query, known_sectors=known_sectors)
        merged = _merge_filters(_as_dict(fallback.get("filters")), clean_filters)
        return InterpretedQuery(
            intent=fallback["intent"],
            source="deterministic_fallback",
            filters=merged,
            entity=fallback.get("entity"),
            original_query=query,
        )

    return InterpretedQuery(
        intent="portfolio_risk_exposure",
        source="structured",
        filters=clean_filters,
        original_query=query,
    )


def _coerce_filters(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for k in ("tickers", "sectors", "assets"):
        v = raw.get(k)
        if isinstance(v, list):
            out[k] = [str(i).strip() for i in v if str(i).strip()]

    for k in ("max_results", "min_risk_score"):
        if k in raw:
            out[k] = raw.get(k)

    return out


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _merge_filters(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = _coerce_filters(base)
    for key, value in _coerce_filters(override).items():
        merged[key] = value
    return merged


def _deterministic_parse(query: str, known_sectors: set[str] | None = None) -> dict[str, Any]:
    text = query.strip()
    lower = text.lower()

    if any(term in lower for term in ("deteriorating", "macro conditions", "macro deterioration")):
        intent = "positions_in_deteriorating_macro"
    elif any(term in lower for term in ("risk exposure", "vix", "breadth", "fear signal")):
        intent = "portfolio_risk_exposure"
    elif any(term in lower for term in ("entity context", "context", "tell me about")):
        intent = "entity_context"
    else:
        intent = "portfolio_risk_exposure"

    tickers = _extract_tickers(text)
    sectors = _extract_sectors(lower, known_sectors or set())

    assets: list[str] = []
    if "equit" in lower:
        assets.append("equity")
    if "commodit" in lower:
        assets.append("commodity")
    if " fx" in f" {lower}" or "currency" in lower:
        assets.append("fx")
    if "bond" in lower or "yield" in lower:
        assets.append("bond")

    filters: dict[str, Any] = {}
    if tickers:
        filters["tickers"] = tickers
    if sectors:
        filters["sectors"] = sectors
    if assets:
        filters["assets"] = assets

    entity = tickers[0] if tickers else (sectors[0] if sectors else None)

    return {
        "intent": intent,
        "filters": filters,
        "entity": entity,
    }


def _extract_tickers(text: str) -> list[str]:
    out: list[str] = []
    for match in _TICKER_RE.findall(text.upper()):
        candidate = match.strip().upper()
        if candidate in _STOP_TICKERS:
            continue
        if candidate.startswith("HTTP"):
            continue
        out.append(candidate)
    # Preserve order, unique
    unique: list[str] = []
    seen: set[str] = set()
    for t in out:
        if t not in seen:
            unique.append(t)
            seen.add(t)
    return unique


def _extract_sectors(query_lower: str, known_sectors: set[str]) -> list[str]:
    out: list[str] = []
    for sector in sorted(known_sectors):
        if sector.lower() in query_lower:
            out.append(sector)
    return out


def _parse_with_llm(query: str) -> dict[str, Any] | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    prompt = (
        "Extract a portfolio-ontology query intent and optional filters as strict JSON. "
        "Allowed intents: portfolio_risk_exposure, positions_in_deteriorating_macro, entity_context. "
        "Return JSON object with keys: intent, filters, entity. "
        "filters may include tickers (array), sectors (array), assets (array), max_results (int), min_risk_score (float). "
        "No markdown.\n\n"
        f"User query: {query}"
    )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.responses.create(model="gpt-5-mini", input=prompt)
        text = (resp.output_text or "").strip()
        if not text:
            return None

        parsed = _parse_json_payload(text)
        if not isinstance(parsed, dict):
            return None

        intent = parsed.get("intent")
        if not isinstance(intent, str) or intent not in ALLOWED_INTENTS:
            return None

        filters = _as_dict(parsed.get("filters"))
        entity = parsed.get("entity")
        entity_str = str(entity) if isinstance(entity, (str, int, float)) else None

        return {
            "intent": intent,
            "filters": _coerce_filters(filters),
            "entity": entity_str,
        }
    except Exception:
        return None


def _parse_json_payload(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
