"""Ontology-backed thesis markdown parsing and rendering helpers.

Direction 1 (markdown -> ontology):
  When thesis markdown is saved/generated, parse ## Key Catalysts and
  ## Risk Factors sections for ontology command projection.

Direction 2 (ontology -> markdown):
  Regenerate the corresponding markdown sections from current ontology state.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from api.state_storage import exists_text, read_text, write_text

logger = logging.getLogger(__name__)

_CLAIMS_SECTION = "Thesis Claims"
_MARKDOWN_SOURCE_TYPE = "workflow"


def _thesis_paths(ticker: str) -> tuple[Path, str]:
    from paths import PROJECT_ROOT

    return PROJECT_ROOT / "investment_theses" / f"{ticker}.md", f"live/theses/{ticker}.md"


def _markdown_source_id(ticker: str) -> str:
    return f"thesis_markdown:{ticker.upper()}"


# ---------------------------------------------------------------------------
# Helpers: markdown <-> DB formatting
# ---------------------------------------------------------------------------


def _format_entity_bullet(text: str) -> str:
    """Convert a DB description like 'Label: Rest' to '- **Label:** Rest'."""
    if ": " in text:
        label, rest = text.split(": ", 1)
        return f"- **{label}:** {rest}"
    return f"- **{text}**"


def _replace_section_lines(content: str, section_header: str, new_lines: list[str]) -> str:
    """Replace the body under a ## section header in markdown content."""
    lines = content.splitlines()
    result: list[str] = []
    in_section = False
    replaced = False
    pattern = rf"^## {re.escape(section_header)}\s*$"

    for line in lines:
        if re.match(pattern, line.strip()):
            in_section = True
            replaced = True
            result.append(line)
            result.extend(new_lines or ["- TBD"])
            continue

        if in_section:
            stripped = line.strip()
            if stripped.startswith("## ") or stripped.startswith("# "):
                in_section = False
                result.append("")
                result.append(line)
            continue

        result.append(line)

    if not replaced:
        if result and result[-1].strip():
            result.append("")
        result.append(f"## {section_header}")
        result.extend(new_lines or ["- TBD"])

    return "\n".join(result)


def _replace_section(content: str, section_header: str, new_bullets: list[str]) -> str:
    """Replace bullets under a ## section header in markdown content."""
    return _replace_section_lines(content, section_header, new_bullets)


def _section_lines(content: str, section_header: str) -> list[str] | None:
    """Return lines under a ## section, or None if the section is absent."""
    pattern = rf"^## {re.escape(section_header)}\s*$"
    lines = content.splitlines()
    in_section = False
    collected: list[str] = []
    found = False

    for line in lines:
        if re.match(pattern, line.strip()):
            in_section = True
            found = True
            continue
        if in_section and (line.strip().startswith("## ") or line.strip().startswith("# ")):
            break
        if in_section:
            collected.append(line)

    return collected if found else None


def _normalize_match_text(value: str) -> str:
    value = re.sub(r"<!--.*?-->", "", value)
    value = value.replace("**", "").replace("__", "")
    value = re.sub(r"\s+", " ", value.strip().lower())
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _entity_label(text: str) -> str:
    return text.split(": ", 1)[0].strip() if ": " in text else text.strip()


def _unique_label_map(entities: list[dict], field: str) -> dict[str, int]:
    labels: dict[str, int] = {}
    duplicates: set[str] = set()
    for entity in entities:
        label = _normalize_match_text(_entity_label(str(entity.get(field) or "")))
        if not label:
            continue
        if label in labels:
            duplicates.add(label)
        else:
            labels[label] = int(entity["id"])
    for duplicate in duplicates:
        labels.pop(duplicate, None)
    return labels


def _id_to_label_map(entities: list[dict], field: str) -> dict[int, str]:
    return {int(entity["id"]): _entity_label(str(entity.get(field) or "")) for entity in entities if entity.get("id")}


def _split_refs(value: str | None) -> list[str]:
    if not value:
        return []
    delimiter = ";" if ";" in value else ","
    return [part.strip() for part in value.split(delimiter) if part.strip()]


def _parse_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"false", "0", "no", "optional"}


def normalize_source_requirements(value: Any) -> list[dict[str, Any]]:
    """Normalize source requirements into typed requirement objects."""

    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            normalized.append(
                {
                    "type": "custom",
                    "description": text,
                    "required": True,
                    "freshness_days": None,
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        req_type = str(item.get("type") or "custom").strip() or "custom"
        description = str(item.get("description") or item.get("label") or req_type).strip()
        required_raw = item.get("required", True)
        required = _parse_bool(required_raw, default=True) if isinstance(required_raw, str) else bool(required_raw)
        freshness_raw = item.get("freshness_days")
        freshness_days = None
        if freshness_raw not in (None, ""):
            try:
                freshness_days = max(0, int(freshness_raw))
            except (TypeError, ValueError):
                freshness_days = None
        normalized.append(
            {
                "type": req_type,
                "description": description,
                "required": required,
                "freshness_days": freshness_days,
            }
        )
    return normalized


def _parse_source_requirement(text: str) -> dict[str, Any]:
    if "=" not in text:
        return normalize_source_requirements([text])[0]

    parts: dict[str, str] = {}
    for part in text.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parts[key.strip().lower()] = value.strip()
    return normalize_source_requirements(
        [
            {
                "type": parts.get("type") or "custom",
                "description": parts.get("description") or parts.get("label") or parts.get("type") or "custom",
                "required": _parse_bool(parts.get("required"), default=True),
                "freshness_days": parts.get("freshness_days"),
            }
        ]
    )[0]


def _parse_claim_heading(line: str) -> str:
    text = line.strip()[2:].strip()
    match = re.match(r"\*\*(.+?)[:\*]+\*?\*?\s*(.*)", text)
    if match:
        label = match.group(1).strip().rstrip(":")
        desc = match.group(2).strip()
        return f"{label}: {desc}" if desc and desc != label else label
    return text


def _parse_confidence(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        confidence = float(value.strip().rstrip("%"))
    except ValueError:
        return None
    if "%" in value:
        confidence /= 100
    return confidence if 0 <= confidence <= 1 else None


def _parse_status(value: str | None) -> str:
    status = (value or "active").strip().lower().replace(" ", "_")
    return status if status in {"active", "supported", "challenged", "disconfirmed", "retired"} else "active"


def _parse_claim_blocks(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("- "):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _parse_structured_claims(content: str) -> list[dict] | None:
    lines = _section_lines(content, _CLAIMS_SECTION)
    if lines is None:
        return None

    claims: list[dict] = []
    for block in _parse_claim_blocks(lines):
        first = block[0]
        if not first.strip() or first.strip() == "- TBD":
            continue
        record: dict[str, Any] = {
            "claim": _parse_claim_heading(first),
            "expected_evidence": None,
            "disconfirming_evidence": None,
            "source_requirements": [],
            "cadence": None,
            "confidence": None,
            "status": "active",
            "linked_catalyst_labels": [],
            "linked_kill_condition_labels": [],
            "parsed_from_text": False,
        }
        in_sources = False
        for raw_line in block[1:]:
            stripped = raw_line.strip()
            if not stripped:
                continue
            id_match = re.search(r"<!--\s*thesis-claim:id=(\d+)\s*-->", stripped)
            if id_match:
                record["id"] = int(id_match.group(1))
                continue
            if not stripped.startswith("- "):
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            field_text = stripped[2:].strip()
            lower = field_text.lower()
            if (
                in_sources
                and indent >= 4
                and not any(
                    lower.startswith(prefix)
                    for prefix in (
                        "status:",
                        "expected evidence:",
                        "disconfirming evidence:",
                        "source requirements:",
                        "cadence:",
                        "confidence:",
                        "catalysts:",
                        "kill conditions:",
                    )
                )
            ):
                record["source_requirements"].append(_parse_source_requirement(field_text))
                continue
            in_sources = False
            if ":" not in field_text:
                continue
            key, value = field_text.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key == "status":
                record["status"] = _parse_status(value)
            elif key == "expected evidence":
                record["expected_evidence"] = value or None
            elif key == "disconfirming evidence":
                record["disconfirming_evidence"] = value or None
            elif key == "source requirements":
                in_sources = True
                if value:
                    record["source_requirements"].append(_parse_source_requirement(value))
            elif key == "cadence":
                record["cadence"] = value or None
            elif key == "confidence":
                record["confidence"] = _parse_confidence(value)
            elif key == "catalysts":
                record["linked_catalyst_labels"] = _split_refs(value)
            elif key == "kill conditions":
                record["linked_kill_condition_labels"] = _split_refs(value)
        if record["claim"] and record["claim"].strip() != "TBD":
            claims.append(record)
    return claims


def _parse_text_claims(content: str) -> list[dict]:
    from portfolio.thesis_backfill import _extract_label_and_description, _parse_bullets

    claims: list[dict] = []
    for bullet in _parse_bullets(content, "Thesis"):
        if not bullet or bullet.strip() == "TBD":
            continue
        label, desc = _extract_label_and_description(bullet)
        claim = f"{label}: {desc}" if desc != label else desc
        claims.append(
            {
                "claim": claim,
                "expected_evidence": None,
                "disconfirming_evidence": None,
                "source_requirements": [],
                "cadence": None,
                "confidence": None,
                "status": "active",
                "linked_catalyst_labels": [],
                "linked_kill_condition_labels": [],
                "parsed_from_text": True,
            }
        )
    return claims


def _resolve_claim_links(claims: list[dict], catalysts: list[dict], kill_conditions: list[dict]) -> list[dict]:
    catalyst_labels = _unique_label_map(catalysts, "description")
    kill_condition_labels = _unique_label_map(kill_conditions, "condition")
    resolved: list[dict] = []
    for claim in claims:
        linked_catalyst_ids = [
            catalyst_labels[label]
            for label in (_normalize_match_text(v) for v in claim.pop("linked_catalyst_labels", []))
            if label in catalyst_labels
        ]
        linked_kill_condition_ids = [
            kill_condition_labels[label]
            for label in (_normalize_match_text(v) for v in claim.pop("linked_kill_condition_labels", []))
            if label in kill_condition_labels
        ]
        resolved.append(
            {
                **claim,
                "linked_catalyst_ids": linked_catalyst_ids,
                "linked_kill_condition_ids": linked_kill_condition_ids,
            }
        )
    return resolved


def _upsert_markdown_claims(ticker: str, records: list[dict]) -> int:
    return len(records)


def sync_claims_from_content(ticker: str, content: str) -> int:
    """Parse thesis claims from markdown content.

    The ontology command service owns mutation; this helper only reports the
    number of claim records parsed from markdown.
    """
    ticker = ticker.upper()
    parsed = _parse_structured_claims(content)
    claims = parsed if parsed is not None else _parse_text_claims(content)
    return len(claims or [])


def _format_confidence(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        return f"{float(value):.2f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return None


def _format_source_requirement(req: dict[str, Any]) -> str:
    freshness = req.get("freshness_days")
    freshness_text = "" if freshness in (None, "") else str(freshness)
    required = "true" if req.get("required", True) else "false"
    return (
        f"type={req.get('type') or 'custom'}; "
        f"description={req.get('description') or req.get('type') or 'custom'}; "
        f"required={required}; freshness_days={freshness_text}"
    )


def _format_claim_lines(
    claim: dict, catalyst_labels: dict[int, str], kill_condition_labels: dict[int, str]
) -> list[str]:
    claim_text = str(claim.get("claim") or "").strip()
    if ": " in claim_text:
        label, rest = claim_text.split(": ", 1)
        lines = [f"- **{label}:** {rest}"]
    else:
        lines = [f"- **{claim_text}**"]
    lines.append(f"  <!-- thesis-claim:id={claim['id']} -->")
    lines.append(f"  - Status: {claim.get('status') or 'active'}")
    if claim.get("expected_evidence"):
        lines.append(f"  - Expected evidence: {claim['expected_evidence']}")
    if claim.get("disconfirming_evidence"):
        lines.append(f"  - Disconfirming evidence: {claim['disconfirming_evidence']}")
    source_requirements = claim.get("source_requirements") or claim.get("source_requirements_json") or []
    if source_requirements:
        lines.append("  - Source requirements:")
        for req in source_requirements:
            if isinstance(req, dict):
                lines.append(f"    - {_format_source_requirement(req)}")
            else:
                lines.append(f"    - {req}")
    if claim.get("cadence"):
        lines.append(f"  - Cadence: {claim['cadence']}")
    confidence = _format_confidence(claim.get("confidence"))
    if confidence is not None:
        lines.append(f"  - Confidence: {confidence}")
    linked_catalyst_ids = claim.get("linked_catalyst_ids") or claim.get("linked_catalyst_ids_json") or []
    catalyst_refs = [catalyst_labels[int(cid)] for cid in linked_catalyst_ids if int(cid) in catalyst_labels]
    if catalyst_refs:
        lines.append(f"  - Catalysts: {'; '.join(catalyst_refs)}")
    linked_kill_condition_ids = (
        claim.get("linked_kill_condition_ids") or claim.get("linked_kill_condition_ids_json") or []
    )
    kc_refs = [
        kill_condition_labels[int(kid)] for kid in linked_kill_condition_ids if int(kid) in kill_condition_labels
    ]
    if kc_refs:
        lines.append(f"  - Kill conditions: {'; '.join(kc_refs)}")
    return lines


# ---------------------------------------------------------------------------
# Direction 1: Markdown -> DB
# ---------------------------------------------------------------------------


def sync_entities_from_markdown(ticker: str) -> dict[str, int]:
    """Parse thesis markdown and return entity counts.

    Mutation happens inside ``OntologyCommandService.save_thesis_content``.
    """
    from portfolio.thesis_backfill import (
        _parse_bullets,
    )

    ticker = ticker.upper()
    thesis_path, thesis_key = _thesis_paths(ticker)
    if not exists_text(thesis_path, thesis_key):
        return {"catalysts": 0, "kill_conditions": 0, "thesis_claims": 0}

    content = read_text(thesis_path, thesis_key, encoding="utf-8").strip()
    if not content:
        return {"catalysts": 0, "kill_conditions": 0, "thesis_claims": 0}

    catalyst_bullets = _parse_bullets(content, "Key Catalysts")
    risk_bullets = _parse_bullets(content, "Risk Factors")
    cat_count = len([item for item in catalyst_bullets if item and item.strip() != "TBD"])
    kc_count = len([item for item in risk_bullets if item and item.strip() != "TBD"])
    claim_count = sync_claims_from_content(ticker, content)

    logger.info(
        "thesis_sync: %s markdown parsed: %d catalysts, %d kill conditions, %d thesis claims",
        ticker,
        cat_count,
        kc_count,
        claim_count,
    )
    return {"catalysts": cat_count, "kill_conditions": kc_count, "thesis_claims": claim_count}


# ---------------------------------------------------------------------------
# Direction 2: DB -> Markdown
# ---------------------------------------------------------------------------


def sync_markdown_from_entities(ticker: str) -> bool:
    """Read ontology entities and update thesis markdown sections.

    Regenerates ## Key Catalysts, ## Risk Factors, and ## Thesis Claims from
    current ontology objects, keeping the rest of the thesis unchanged.

    Returns True if the file was updated, False if no thesis file exists.
    """
    from ontology.runtime_read_service import OntologyRuntimeReadService

    ticker = ticker.upper()
    thesis_path, thesis_key = _thesis_paths(ticker)
    if not exists_text(thesis_path, thesis_key):
        return False

    content = read_text(thesis_path, thesis_key, encoding="utf-8")

    runtime = OntologyRuntimeReadService()
    catalysts = runtime.catalysts(ticker, limit=500)
    kill_conditions = runtime.kill_conditions(ticker, limit=500)
    claims = runtime.thesis_claims(ticker, limit=500)

    cat_bullets = [_format_entity_bullet(c["description"]) for c in catalysts if c.get("status") == "pending"]
    kc_bullets = [_format_entity_bullet(k["condition"]) for k in kill_conditions if k.get("status") == "active"]
    catalyst_labels = _id_to_label_map(catalysts, "description")
    kill_condition_labels = _id_to_label_map(kill_conditions, "condition")
    claim_lines: list[str] = []
    for claim in claims:
        if claim.get("status") == "retired":
            continue
        claim_lines.extend(_format_claim_lines(claim, catalyst_labels, kill_condition_labels))
        claim_lines.append("")
    if claim_lines and not claim_lines[-1].strip():
        claim_lines.pop()

    updated = _replace_section(content, "Key Catalysts", cat_bullets)
    updated = _replace_section(updated, "Risk Factors", kc_bullets)
    updated = _replace_section_lines(updated, _CLAIMS_SECTION, claim_lines)

    new_content = updated.rstrip() + "\n"
    if new_content == content:
        return False  # No changes needed

    write_text(
        thesis_path,
        thesis_key,
        new_content,
        encoding="utf-8",
        content_type="text/markdown; charset=utf-8",
    )
    logger.info("thesis_sync: updated markdown for %s", ticker)
    return True
