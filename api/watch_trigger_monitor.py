"""Executable watch-trigger monitor with gated news awareness."""

from __future__ import annotations

import hashlib
import json
import operator
import re
from datetime import UTC, datetime
from typing import Any

from api.generated_approval_filters import should_suppress_generated_review_approval

OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

NEWS_TRIGGER_TYPES = {"news_event", "fundamental_news"}
DETERMINISTIC_TRIGGER_TYPES = {"price_level", "technical", "macro"}
DEFAULT_NEWS_LOOKBACK_DAYS = 7
DEFAULT_NEWS_MATERIALITY = 0.6
MAX_STORY_MATCHES = 5
MAX_WEB_VERIFICATIONS = 2


def _monitor_trigger_source_id(trigger_id: Any) -> str:
    text = str(trigger_id or "").strip()
    if text.startswith("watch_trigger:"):
        return text
    return f"watch_trigger:{text}"


MACRO_FIELD_ALIASES = {
    "regime score": "regime.score",
    "liquidity score": "factors.liquidity.score",
    "breadth score": "factors.breadth.score",
    "vix score": "factors.vix.score",
    "vix": "factors.vix.highlights.vix",
    "sector score": "factors.sector.score",
    "momentum score": "factors.momentum.score",
}


def _compare(actual: Any, op: str, expected: Any) -> bool:
    if op not in OPS:
        raise ValueError(f"Unsupported trigger operator: {op}")
    if actual is None or expected is None:
        if op in {"==", "!="}:
            return bool(OPS[op](actual, expected))
        return False
    try:
        actual_f = float(actual)
        expected_f = float(expected)
        return bool(OPS[op](actual_f, expected_f))
    except (TypeError, ValueError):
        return bool(OPS[op](str(actual), str(expected)))


def _nested_get(value: Any, path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if isinstance(current, list):
            if part.isdigit():
                idx = int(part)
                current = current[idx] if 0 <= idx < len(current) else None
            else:
                current = next(
                    (
                        item
                        for item in current
                        if isinstance(item, dict) and str(item.get("key") or item.get("name") or "") == part
                    ),
                    None,
                )
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _canonical_hash(value: Any, length: int = 12) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _latest_price(ticker: str) -> dict[str, Any]:
    import yfinance as yf

    hist = yf.download(ticker, period="10d", interval="1d", progress=False, auto_adjust=True)
    if hist is None or hist.empty or "Close" not in hist:
        raise RuntimeError(f"No close price history for {ticker}")
    close = hist["Close"]
    if getattr(close, "ndim", 1) > 1:
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        raise RuntimeError(f"Empty close price history for {ticker}")
    latest_idx = close.index[-1]
    return {"value": float(close.iloc[-1]), "as_of": str(getattr(latest_idx, "date", lambda: latest_idx)())}


def _evaluate_price_level(definition: dict[str, Any], fallback_ticker: str | None) -> dict[str, Any]:
    ticker = str(definition.get("ticker") or fallback_ticker or "").upper()
    if not ticker:
        raise ValueError("price_level trigger requires ticker")
    price = _latest_price(ticker)
    op = str(definition.get("operator") or definition.get("op") or ">=")
    threshold = definition.get("threshold", definition.get("value"))
    fired = _compare(price["value"], op, threshold)
    return {
        "type": "price_level",
        "fired": fired,
        "actual": price["value"],
        "operator": op,
        "expected": threshold,
        "evidence": f"{ticker} close {price['value']:.2f} {op} {threshold}",
        "as_of": price["as_of"],
    }


def _evaluate_technical(definition: dict[str, Any], fallback_ticker: str | None) -> dict[str, Any]:
    ticker = str(definition.get("ticker") or fallback_ticker or "").upper()
    if not ticker:
        raise ValueError("technical trigger requires ticker")
    from portfolio.technical_analysis.technical_analysis import get_data

    data = get_data(ticker, lookback=str(definition.get("lookback") or "2Y"))
    summary = data.get("summary") if isinstance(data, dict) else []
    indicator_contains = str(definition.get("indicator_contains") or definition.get("indicator") or "").lower()
    field = str(definition.get("field") or "Signal")
    op = str(definition.get("operator") or definition.get("op") or "==")
    expected = definition.get("expected", definition.get("value"))
    matches = []
    for row in summary if isinstance(summary, list) else []:
        if not isinstance(row, dict):
            continue
        indicator = str(row.get("Indicator") or "")
        if indicator_contains and indicator_contains not in indicator.lower():
            continue
        actual = row.get(field)
        if _compare(actual, op, expected):
            matches.append({"indicator": indicator, "field": field, "actual": actual})
    fired = bool(matches)
    return {
        "type": "technical",
        "fired": fired,
        "actual": matches,
        "operator": op,
        "expected": expected,
        "evidence": f"{ticker} technical trigger matched {len(matches)} row(s)",
        "as_of": str(data.get("timestamp")) if isinstance(data, dict) else None,
    }


def _evaluate_macro(definition: dict[str, Any], _fallback_ticker: str | None) -> dict[str, Any]:
    from api.signal_snapshot import get_signal_aggregator_snapshot_or_module_response

    data = get_signal_aggregator_snapshot_or_module_response(lookback_weeks=156, include_raw_modules=False)
    if data is None:
        from api.signal_aggregator import build_signal_aggregator

        data = build_signal_aggregator(include_history=False)
    field_path = str(definition.get("field") or "regime.score")
    actual = _nested_get(data, field_path) if isinstance(data, dict) else None
    op = str(definition.get("operator") or definition.get("op") or ">=")
    expected = definition.get("threshold", definition.get("value"))
    fired = _compare(actual, op, expected)
    return {
        "type": "macro",
        "fired": fired,
        "actual": actual,
        "operator": op,
        "expected": expected,
        "evidence": f"macro {field_path}={actual} {op} {expected}",
        "as_of": str(_nested_get(data, "_meta.snapshot.as_of")) if isinstance(data, dict) else None,
    }


def _extract_sources(text: str) -> list[str]:
    sources: list[str] = []
    for match in re.finditer(r"\(([^()]{2,80})\)", text):
        candidate = _clean_text(match.group(1))
        if any(ch.isdigit() for ch in candidate):
            continue
        for part in re.split(r"\s*/\s*|\s*,\s*", candidate):
            source = _clean_text(part).strip("-")
            if source and source.lower() not in {"body content", "content"}:
                sources.append(source)
    deduped: list[str] = []
    seen: set[str] = set()
    for source in sources:
        key = source.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(source)
    return deduped


def _flatten_digest_context(context: dict[str, Any]) -> list[dict[str, Any]]:
    stories: list[dict[str, Any]] = []
    for digest in context.get("digests") or []:
        if not isinstance(digest, dict):
            continue
        for section in digest.get("sections") or []:
            if not isinstance(section, dict):
                continue
            for story in section.get("stories") or []:
                if not isinstance(story, dict):
                    continue
                headline = _clean_text(story.get("headline"))
                notes = [_clean_text(note) for note in story.get("notes") or [] if _clean_text(note)]
                text = " ".join([headline, *notes])
                stories.append(
                    {
                        "digest_id": digest.get("id"),
                        "digest_title": digest.get("title"),
                        "generated_date": digest.get("generated_date"),
                        "section": section.get("name"),
                        "headline": headline,
                        "notes": notes,
                        "text": text,
                        "sources": _extract_sources(text),
                    }
                )
    return stories


def _load_news_context(days: int) -> dict[str, Any]:
    from portfolio.news_digests import get_report_context

    return get_report_context(days=days, max_digests=5, max_stories=80, notes_per_story=3)


def _search_web(query: str) -> dict[str, Any]:
    from api.agent_tools import _run_search_web

    return _run_search_web(query)


def _verify_news_matches(matches: list[dict[str, Any]], definition: dict[str, Any]) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    max_searches = int(definition.get("max_web_verifications") or MAX_WEB_VERIFICATIONS)
    for match in matches[: max(0, max_searches)]:
        query = _clean_text(
            definition.get("web_query")
            or " ".join(
                str(part)
                for part in [
                    definition.get("ticker"),
                    match.get("headline"),
                    "latest primary source",
                ]
                if part
            )
        )
        if not query:
            continue
        try:
            result = _search_web(query)
        except Exception as exc:  # noqa: BLE001 - monitor should record search failures, not fail the trigger.
            verified.append({"query": query, "summary": "", "citations": [], "error": str(exc)})
            continue
        verified.append(
            {
                "query": query,
                "summary": _clean_text(result.get("summary")),
                "citations": [
                    {"title": _clean_text(item.get("title")), "url": _clean_text(item.get("url"))}
                    for item in result.get("citations") or []
                    if isinstance(item, dict) and item.get("url")
                ],
            }
        )
    return verified


def _linked_context(definition: dict[str, Any], fallback_ticker: str | None) -> dict[str, Any]:
    context: dict[str, Any] = {"claims": [], "catalysts": [], "kill_conditions": [], "terms": []}
    ticker = str(definition.get("ticker") or fallback_ticker or "").upper()
    try:
        from ontology.runtime_read_service import OntologyRuntimeReadService

        reads = OntologyRuntimeReadService()
        claim_ids = {str(item) for item in _as_list(definition.get("linked_claim_ids")) if str(item).strip()}
        claim_matches = [reads.get(claim_id) for claim_id in claim_ids]
        claims = [claim for claim in claim_matches if claim is not None]
        if not claims and ticker and definition.get("include_ticker_claims", True):
            claims = reads.thesis_claims(ticker=ticker, status="active", limit=20)
        context["claims"] = claims

        catalysts = reads.catalysts(ticker) if ticker else []
        catalyst_ids = {str(item) for item in _as_list(definition.get("linked_catalyst_ids")) if str(item).strip()}
        if catalyst_ids:
            catalysts = [item for item in catalysts if str(item.get("id") or item.get("object_uid")) in catalyst_ids]
        context["catalysts"] = catalysts

        kill_conditions = reads.kill_conditions(ticker) if ticker else []
        kill_ids = {str(item) for item in _as_list(definition.get("linked_kill_condition_ids")) if str(item).strip()}
        if kill_ids:
            kill_conditions = [
                item for item in kill_conditions if str(item.get("id") or item.get("object_uid")) in kill_ids
            ]
        context["kill_conditions"] = kill_conditions
    except Exception:
        return context

    terms: list[str] = []
    for claim in context["claims"]:
        if isinstance(claim, dict):
            terms.extend(
                [
                    str(claim.get("claim") or ""),
                    str(claim.get("expected_evidence") or ""),
                    str(claim.get("disconfirming_evidence") or ""),
                ]
            )
    for catalyst in context["catalysts"]:
        if isinstance(catalyst, dict):
            terms.append(str(catalyst.get("description") or ""))
    for condition in context["kill_conditions"]:
        if isinstance(condition, dict):
            terms.append(str(condition.get("condition") or ""))
            terms.append(str(condition.get("metric") or ""))
            terms.append(str(condition.get("threshold") or ""))
    context["terms"] = [_clean_text(term) for term in terms if _clean_text(term)]
    return context


def _hit_count(needles: list[str], haystack: str) -> int:
    haystack_lower = haystack.lower()
    return sum(1 for needle in needles if needle and needle.lower() in haystack_lower)


def _short_terms(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9+-]{2,}", text)
    stop = {
        "and",
        "the",
        "for",
        "with",
        "from",
        "that",
        "this",
        "expected",
        "evidence",
        "condition",
        "growth",
        "remains",
    }
    return [word for word in words if word.lower() not in stop][:8]


def _match_news_stories(
    definition: dict[str, Any],
    trigger: dict[str, Any],
    stories: list[dict[str, Any]],
    *,
    enrichment: bool = False,
) -> list[dict[str, Any]]:
    ticker = str(definition.get("ticker") or trigger.get("ticker") or "").upper()
    entities = [_clean_text(item) for item in _as_list(definition.get("entities")) if _clean_text(item)]
    if ticker and ticker not in entities:
        entities.append(ticker)
    topics = [_clean_text(item) for item in _as_list(definition.get("topics")) if _clean_text(item)]

    linked = _linked_context(definition, ticker)
    context_terms: list[str] = []
    for term in linked.get("terms") or []:
        context_terms.extend(_short_terms(str(term)))
    context_terms = list(dict.fromkeys(context_terms))

    threshold = float(definition.get("materiality_threshold") or (0.45 if enrichment else DEFAULT_NEWS_MATERIALITY))
    matches: list[dict[str, Any]] = []
    for story in stories:
        text = str(story.get("text") or "")
        entity_hits = _hit_count(entities, text)
        topic_hits = _hit_count(topics, text)
        context_hits = _hit_count(context_terms, text)
        if entities and not entity_hits:
            continue
        if topics and not topic_hits and not context_hits:
            continue
        score = 0.0
        score += 0.45 if (entity_hits or not entities) else 0.0
        score += min(0.35, 0.12 * topic_hits)
        score += min(0.2, 0.04 * context_hits)
        if not topics and not context_terms and entities:
            score += 0.15
        if score < threshold:
            continue
        polarity = "needs_review"
        story_lower = text.lower()
        if any(str(term).lower() in story_lower for term in _as_list(definition.get("disconfirming_evidence"))):
            polarity = "disconfirming"
        elif any(str(term).lower() in story_lower for term in _as_list(definition.get("expected_evidence"))):
            polarity = "supporting"
        matches.append(
            {
                "headline": story.get("headline"),
                "notes": story.get("notes") or [],
                "digest_id": story.get("digest_id"),
                "digest_title": story.get("digest_title"),
                "generated_date": story.get("generated_date"),
                "section": story.get("section"),
                "sources": story.get("sources") or [],
                "entity_hits": entity_hits,
                "topic_hits": topic_hits,
                "context_hits": context_hits,
                "materiality_score": round(score, 3),
                "polarity": polarity,
            }
        )

    matches.sort(key=lambda item: float(item.get("materiality_score") or 0), reverse=True)
    return matches[:MAX_STORY_MATCHES]


def _source_requirements_met(
    matches: list[dict[str, Any]],
    verifications: list[dict[str, Any]],
    definition: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    requirements = _as_dict(definition.get("source_requirements"))
    min_sources = int(requirements.get("min_sources") or definition.get("min_sources") or 1)
    primary_required = bool(requirements.get("primary_source_required") or definition.get("primary_source_required"))
    source_names = {
        str(source).strip().lower() for match in matches for source in match.get("sources") or [] if str(source).strip()
    }
    citation_urls = {
        str(citation.get("url")).strip()
        for verification in verifications
        for citation in verification.get("citations") or []
        if isinstance(citation, dict) and citation.get("url")
    }
    source_count = len(source_names) + len(citation_urls)
    met = source_count >= min_sources and (not primary_required or bool(citation_urls))
    return met, {
        "min_sources": min_sources,
        "primary_source_required": primary_required,
        "source_count": source_count,
        "sources": sorted(source_names),
        "citation_count": len(citation_urls),
    }


def _web_fallback_match(
    definition: dict[str, Any], trigger: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ticker = str(definition.get("ticker") or trigger.get("ticker") or "").upper()
    entities = " ".join(str(item) for item in _as_list(definition.get("entities")) if item)
    topics = " ".join(str(item) for item in _as_list(definition.get("topics")) if item)
    query = _clean_text(definition.get("web_query") or f"{ticker} {entities} {topics} latest material news")
    if not query:
        return [], []
    try:
        result = _search_web(query)
    except Exception as exc:  # noqa: BLE001
        return [], [{"query": query, "summary": "", "citations": [], "error": str(exc)}]
    citations = [
        {"title": _clean_text(item.get("title")), "url": _clean_text(item.get("url"))}
        for item in result.get("citations") or []
        if isinstance(item, dict) and item.get("url")
    ]
    if not citations:
        return [], [{"query": query, "summary": _clean_text(result.get("summary")), "citations": []}]
    return [
        {
            "headline": f"Web fallback match for {query}",
            "notes": [_clean_text(result.get("summary"))],
            "digest_id": None,
            "digest_title": None,
            "generated_date": None,
            "section": "web_fallback",
            "sources": [],
            "entity_hits": 1,
            "topic_hits": 1,
            "context_hits": 0,
            "materiality_score": 1.0,
            "polarity": "needs_review",
        }
    ], [{"query": query, "summary": _clean_text(result.get("summary")), "citations": citations}]


def _evaluate_news_trigger(definition: dict[str, Any], trigger: dict[str, Any]) -> dict[str, Any]:
    lookback_days = int(definition.get("lookback_days") or DEFAULT_NEWS_LOOKBACK_DAYS)
    context = _load_news_context(lookback_days)
    stories = _flatten_digest_context(context)
    matches = _match_news_stories(definition, trigger, stories)
    verifications = _verify_news_matches(matches, definition)
    fallback_used = False
    if (not matches and (not context.get("digests") or context.get("fallback_used"))) or (
        matches and not any(match.get("sources") for match in matches) and context.get("fallback_used")
    ):
        fallback_used = True
        matches, verifications = _web_fallback_match(definition, trigger)

    requirements_met, source_summary = _source_requirements_met(matches, verifications, definition)
    fired = bool(matches and requirements_met)
    if not matches:
        evidence = "No matching recent sourced news found."
    elif not requirements_met:
        evidence = "News match found but source requirements were not met."
    else:
        evidence = f"Needs review: {len(matches)} news match(es) met source/materiality thresholds."
    return {
        "type": str(definition.get("type") or trigger.get("trigger_type") or "news_event"),
        "fired": fired,
        "needs_review": fired,
        "portfolio_mutation_allowed": False,
        "thesis_mutation_allowed": False,
        "trade_recommendation_allowed": False,
        "evidence": evidence,
        "as_of": datetime.now(UTC).date().isoformat(),
        "news": {
            "lookback_days": lookback_days,
            "digest_context": {
                "fallback_used": bool(context.get("fallback_used")),
                "counts": context.get("counts") or {},
                "cutoff_date": context.get("cutoff_date"),
            },
            "web_fallback_used": fallback_used,
            "matches": matches,
            "verifications": verifications,
            "source_requirements": source_summary,
        },
    }


def _news_enrichment(definition: dict[str, Any], trigger: dict[str, Any]) -> dict[str, Any]:
    lookback_days = int(definition.get("news_lookback_days") or definition.get("lookback_days") or 3)
    context = _load_news_context(lookback_days)
    stories = _flatten_digest_context(context)
    matches = _match_news_stories(definition, trigger, stories, enrichment=True)
    verifications = _verify_news_matches(matches, definition)
    return {
        "lookback_days": lookback_days,
        "digest_context": {
            "fallback_used": bool(context.get("fallback_used")),
            "counts": context.get("counts") or {},
            "cutoff_date": context.get("cutoff_date"),
        },
        "matches": matches,
        "verifications": verifications,
    }


def _infer_definition(trigger: dict[str, Any]) -> dict[str, Any] | None:
    condition = _clean_text(trigger.get("condition"))
    lower = condition.lower()
    ticker = str(trigger.get("ticker") or "").upper()
    trigger_type = str(trigger.get("trigger_type") or "").lower()

    price_match = re.search(r"\b([A-Z]{1,6})\b.*?(>=|<=|>|<)\s*\$?(\d+(?:\.\d+)?)", condition)
    if price_match and trigger_type in {"price_level", "custom", ""}:
        return {
            "type": "price_level",
            "ticker": ticker or price_match.group(1).upper(),
            "operator": price_match.group(2),
            "threshold": float(price_match.group(3)),
        }

    directional_price = re.search(
        r"\b([A-Z]{1,6})\b.*?\b(breaks above|breaks over|above|over|breaks below|below|under)\b.*?\$?(\d+(?:\.\d+)?)",
        condition,
        flags=re.IGNORECASE,
    )
    if directional_price and trigger_type in {"price_level", "custom", ""}:
        direction = directional_price.group(2).lower()
        return {
            "type": "price_level",
            "ticker": ticker or directional_price.group(1).upper(),
            "operator": "<=" if any(word in direction for word in ("below", "under")) else ">=",
            "threshold": float(directional_price.group(3)),
        }

    technical_match = re.search(
        r"\b([A-Z]{1,6})\b.*?\b(above|below)\b.*?\b(20|50|100|200)\s*(d|day|dma|sma|ma)\b",
        condition,
        flags=re.IGNORECASE,
    )
    if technical_match and trigger_type in {"technical", "custom", ""}:
        return {
            "type": "technical",
            "ticker": ticker or technical_match.group(1).upper(),
            "indicator_contains": f"{technical_match.group(3)}D",
            "field": "Signal",
            "expected": "Above" if technical_match.group(2).lower() == "above" else "Below",
        }

    for phrase, field in MACRO_FIELD_ALIASES.items():
        if phrase not in lower:
            continue
        macro_match = re.search(r"(>=|<=|>|<|==|!=)\s*(\d+(?:\.\d+)?)", condition)
        if macro_match and trigger_type in {"macro", "custom", ""}:
            return {
                "type": "macro",
                "field": field,
                "operator": macro_match.group(1),
                "threshold": float(macro_match.group(2)),
            }
    return None


def evaluate_trigger(trigger: dict[str, Any]) -> dict[str, Any]:
    definition = trigger.get("definition_json") or trigger.get("definition")
    inferred = None
    if not isinstance(definition, dict) or not definition:
        inferred = _infer_definition(trigger)
        definition = inferred
    if not isinstance(definition, dict) or not definition:
        return {
            "fired": False,
            "skipped": True,
            "evidence": "Trigger has no machine-readable definition.",
        }

    trigger_type = str(definition.get("type") or trigger.get("trigger_type") or "").lower()
    fallback_ticker = trigger.get("ticker")
    if trigger_type == "price_level":
        result = _evaluate_price_level(definition, fallback_ticker)
    elif trigger_type == "technical":
        result = _evaluate_technical(definition, fallback_ticker)
    elif trigger_type == "macro":
        result = _evaluate_macro(definition, fallback_ticker)
    elif trigger_type in NEWS_TRIGGER_TYPES:
        result = _evaluate_news_trigger(definition, trigger)
    else:
        return {"fired": False, "skipped": True, "evidence": f"Unsupported trigger type: {trigger_type}"}

    if inferred:
        result["inferred_definition"] = inferred
    if result.get("fired") and trigger_type in DETERMINISTIC_TRIGGER_TYPES:
        enrichment = _news_enrichment(definition, trigger)
        result["news_enrichment"] = enrichment
        matches = enrichment.get("matches") or []
        if matches:
            headline = _clean_text(matches[0].get("headline"))
            result["evidence"] = f"{result.get('evidence')}. News context: {headline}"
    return result


def _action_description(trigger: dict[str, Any], result: dict[str, Any]) -> str:
    condition = _clean_text(trigger.get("condition"))
    trigger_type = str(result.get("type") or trigger.get("trigger_type") or "")
    if trigger_type in NEWS_TRIGGER_TYPES:
        return f"Needs review: news monitor matched watch trigger: {condition}"
    return f"Review fired watch trigger: {condition}"


def _result_fingerprint(result: dict[str, Any]) -> str:
    return _canonical_hash(
        {
            "type": result.get("type"),
            "actual": result.get("actual"),
            "expected": result.get("expected"),
            "news": result.get("news"),
            "news_enrichment": result.get("news_enrichment"),
            "as_of": result.get("as_of"),
        }
    )


def run_watch_trigger_monitor(_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    from ontology.command_service import OntologyCommandContext, OntologyCommandService
    from ontology.policy import system_actor
    from ontology.runtime_read_service import OntologyRuntimeReadService

    command_service = OntologyCommandService()
    reads = OntologyRuntimeReadService()

    def propose_action(action_id: str, payload: dict[str, Any], *, source_id: str, reason: str) -> dict[str, Any]:
        return command_service.propose_action(
            action_id,
            payload,
            OntologyCommandContext(
                actor=system_actor("watch_trigger_monitor"),
                source_type="workflow",
                source_id=source_id,
            ),
            reason=reason,
        )

    checked = 0
    fired = 0
    skipped = 0
    errors = 0
    for trigger in reads.watch_triggers(status="active"):
        checked += 1
        trigger_id = str(trigger.get("object_uid") or trigger.get("id") or "").strip()
        if not trigger_id:
            errors += 1
            continue
        trigger_source_id = _monitor_trigger_source_id(trigger_id)
        try:
            result = evaluate_trigger(trigger)
            if result.get("inferred_definition") and not trigger.get("definition_json"):
                propose_action(
                    "update_watch_trigger_definition",
                    {"trigger_id": trigger_id, "definition": result["inferred_definition"]},
                    source_id=trigger_source_id,
                    reason=f"Infer watch trigger definition for {trigger_id}",
                )
            evidence = str(result.get("evidence") or "")
            if result.get("skipped"):
                skipped += 1
                propose_action(
                    "update_watch_trigger_check",
                    {"trigger_id": trigger_id, "result": result, "evidence": evidence},
                    source_id=trigger_source_id,
                    reason=f"Record skipped watch trigger check for {trigger_id}",
                )
                continue
            if result.get("fired"):
                fired += 1
                fingerprint = _result_fingerprint(result)
                source_id = f"{trigger_source_id}:{fingerprint}"
                propose_action(
                    "fire_watch_trigger",
                    {"trigger_id": trigger_id, "result": result, "evidence": evidence},
                    source_id=source_id,
                    reason=f"Watch trigger {trigger_id} fired",
                )
                action_item_payload = {
                    "description": _action_description(trigger, result),
                    "action_type": "review",
                    "ticker": trigger.get("ticker"),
                    "urgency": "high",
                    "alert_context": {
                        "change_summary": _action_description(trigger, result),
                        "source": "monitor_hit",
                        "ticker": trigger.get("ticker"),
                    },
                }
                if not should_suppress_generated_review_approval(
                    "create_action_item",
                    action_item_payload,
                    source_type="workflow",
                ):
                    propose_action(
                        "create_action_item",
                        action_item_payload,
                        source_id=source_id,
                        reason=f"Create action item for fired watch trigger {trigger_id}",
                    )
            else:
                propose_action(
                    "update_watch_trigger_check",
                    {"trigger_id": trigger_id, "result": result, "evidence": evidence},
                    source_id=trigger_source_id,
                    reason=f"Record watch trigger check for {trigger_id}",
                )
        except Exception as exc:
            errors += 1
            propose_action(
                "update_watch_trigger_check",
                {"trigger_id": trigger_id, "result": {"error": str(exc), "fired": False}, "evidence": str(exc)},
                source_id=trigger_source_id,
                reason=f"Record watch trigger monitor error for {trigger_id}",
            )
    return {"checked": checked, "fired": fired, "skipped": skipped, "errors": errors}
