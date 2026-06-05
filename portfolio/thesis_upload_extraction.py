"""Thesis upload: enrich markdown and propose kill conditions from dossier context."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from llm_utils import MODEL_MID, call_llm_text
from portfolio.thesis_backfill import _extract_label_and_description, _parse_bullets
from portfolio.thesis_sync import _normalize_match_text, _replace_section

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM = """You are a buy-side investment analyst.
Given a position thesis and optional overview / management-quality context, extract:
1. Key catalysts — specific, monitorable drivers of the investment case.
2. Kill conditions — concrete invalidation triggers with measurable metric and threshold when possible.

Use only information supported by the provided documents. Do not invent facts.
Output JSON matching the schema exactly."""

_EXTRACTION_USER = """Ticker: {ticker}

<thesis_markdown>
{thesis}
</thesis_markdown>

<overview_markdown>
{overview}
</overview_markdown>

<management_quality_markdown>
{management}
</management_quality_markdown>

Extract catalysts and kill_conditions. Kill conditions should reflect risks from the thesis,
financial/industry sensitivities from overview when present, and management/setback risks
from management quality when present."""

_EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "catalysts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["label", "description"],
                "additionalProperties": False,
            },
        },
        "kill_conditions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string"},
                    "metric": {"type": ["string", "null"]},
                    "threshold": {"type": ["string", "null"]},
                    "rationale": {"type": ["string", "null"]},
                },
                "required": ["condition"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["catalysts", "kill_conditions"],
    "additionalProperties": False,
}


def _blank_bullet(bullet: str) -> bool:
    text = (bullet or "").strip().lower()
    return not text or text in {"tbd", "n/a", "none", "-"}


def catalyst_section_needs_fill(content: str) -> bool:
    bullets = _parse_bullets(content, "Key Catalysts")
    meaningful = [b for b in bullets if not _blank_bullet(b)]
    return len(meaningful) == 0


def _format_catalyst_bullet(label: str, description: str) -> str:
    label = label.strip()
    description = (description or label).strip()
    if not label:
        return f"- {description}"
    if description and description != label:
        return f"- **{label}:** {description}"
    return f"- **{label}**"


def merge_catalyst_bullets_into_thesis(content: str, catalysts: list[dict[str, Any]]) -> str:
    if not catalyst_section_needs_fill(content):
        return content
    lines = [
        _format_catalyst_bullet(str(c.get("label") or ""), str(c.get("description") or ""))
        for c in catalysts
        if str(c.get("label") or c.get("description") or "").strip()
    ]
    if not lines:
        return content
    return _replace_section(content, "Key Catalysts", lines)


def read_dossier_context_markdown(ticker: str) -> tuple[str | None, str | None]:
    overview_text: str | None = None
    management_text: str | None = None
    normalized = ticker.strip().upper()

    try:
        from portfolio.overview_content import overview_exists, read_overview

        if overview_exists(normalized):
            overview_text = read_overview(normalized)
    except Exception:
        logger.debug("overview context unavailable for %s", normalized, exc_info=True)

    try:
        from portfolio.management_quality_content import management_quality_exists, read_management_quality

        if management_quality_exists(normalized):
            management_text = read_management_quality(normalized)
    except Exception:
        logger.debug("management quality context unavailable for %s", normalized, exc_info=True)

    return overview_text, management_text


def _parse_llm_extraction_payload(raw: str) -> dict[str, Any] | None:
    cleaned = (raw or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _normalize_catalyst_records(records: list[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        description = str(item.get("description") or label).strip()
        if not label and not description:
            continue
        if not label:
            label = description[:80]
        normalized.append({"label": label, "description": description})
    return normalized


def _normalize_kill_condition_records(records: list[Any]) -> list[dict[str, str | None]]:
    normalized: list[dict[str, str | None]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        condition = str(item.get("condition") or "").strip()
        if not condition or _blank_bullet(condition):
            continue
        metric = str(item.get("metric") or "").strip() or None
        threshold = str(item.get("threshold") or "").strip() or None
        normalized.append(
            {
                "condition": condition,
                "metric": metric,
                "threshold": threshold,
                "rationale": str(item.get("rationale") or "").strip() or None,
            }
        )
    return normalized


def _fallback_extraction_from_thesis(thesis_content: str) -> dict[str, Any]:
    catalysts: list[dict[str, str]] = []
    for bullet in _parse_bullets(thesis_content, "Key Catalysts"):
        if _blank_bullet(bullet):
            continue
        label, desc = _extract_label_and_description(bullet)
        catalysts.append({"label": label, "description": desc})

    kill_conditions: list[dict[str, str | None]] = []
    for bullet in _parse_bullets(thesis_content, "Risk Factors"):
        if _blank_bullet(bullet):
            continue
        label, desc = _extract_label_and_description(bullet)
        condition = f"{label}: {desc}" if desc != label else label
        kill_conditions.append({"condition": condition, "metric": None, "threshold": None, "rationale": None})

    return {"catalysts": catalysts, "kill_conditions": kill_conditions}


def extract_entities_from_thesis_upload(ticker: str, thesis_content: str) -> dict[str, Any]:
    """LLM extraction with deterministic fallback from thesis markdown sections."""
    normalized_ticker = ticker.strip().upper()
    overview, management = read_dossier_context_markdown(normalized_ticker)

    try:
        generated, _citations, _response = call_llm_text(
            prompt=_EXTRACTION_USER.format(
                ticker=normalized_ticker,
                thesis=thesis_content.strip(),
                overview=(overview or "").strip() or "(not available)",
                management=(management or "").strip() or "(not available)",
            ),
            model=MODEL_MID,
            api_key=None,
            max_tokens=4096,
            system=_EXTRACTION_SYSTEM,
            json_schema=_EXTRACTION_JSON_SCHEMA,
            json_schema_name="thesis_upload_entity_extraction",
        )
        parsed = _parse_llm_extraction_payload(generated)
        if parsed:
            return {
                "catalysts": _normalize_catalyst_records(list(parsed.get("catalysts") or [])),
                "kill_conditions": _normalize_kill_condition_records(list(parsed.get("kill_conditions") or [])),
            }
    except Exception:
        logger.warning(
            "thesis upload entity extraction LLM failed for %s; using markdown fallback",
            normalized_ticker,
            exc_info=True,
        )

    return _fallback_extraction_from_thesis(thesis_content)


def count_meaningful_catalyst_bullets(content: str) -> int:
    return len([b for b in _parse_bullets(content, "Key Catalysts") if not _blank_bullet(b)])


def existing_kill_condition_keys(ticker: str) -> set[str]:
    keys: set[str] = set()
    try:
        from ontology.runtime_read_service import OntologyRuntimeReadService

        for row in OntologyRuntimeReadService().kill_conditions(ticker):
            condition = str(row.get("condition") or "").strip()
            if condition:
                keys.add(_normalize_match_text(condition))
    except Exception:
        logger.debug("could not load existing kill conditions for %s", ticker, exc_info=True)
    return keys


def dedupe_kill_condition_candidates(
    candidates: list[dict[str, str | None]],
    *,
    existing_keys: set[str],
    staged_keys: set[str] | None = None,
) -> tuple[list[dict[str, str | None]], int]:
    seen = set(existing_keys)
    if staged_keys:
        seen |= staged_keys
    unique: list[dict[str, str | None]] = []
    skipped = 0
    for candidate in candidates:
        key = _normalize_match_text(str(candidate.get("condition") or ""))
        if not key or key in seen:
            skipped += 1
            continue
        seen.add(key)
        unique.append(candidate)
    return unique, skipped
