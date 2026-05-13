"""Generate thesis markdown from ontology-backed research objects."""

from __future__ import annotations

import logging

from api.state_storage import exists_text, read_text
from ontology.runtime_read_service import OntologyRuntimeReadService
from paths import PROJECT_ROOT
from portfolio.thesis_sync import _format_entity_bullet

logger = logging.getLogger(__name__)


def generate_thesis_markdown(ticker: str) -> str:
    """Generate a thesis markdown document for a ticker."""

    ticker = ticker.strip().upper()
    runtime = OntologyRuntimeReadService()
    meta = runtime.thesis(ticker) or {}
    sections: list[str] = [f"# {ticker}"]

    meta_parts = []
    if meta.get("direction"):
        meta_parts.append(f"**Direction:** {meta['direction']}")
    if meta.get("timeframe"):
        meta_parts.append(f"**Timeframe:** {meta['timeframe']}")
    if meta.get("status"):
        meta_parts.append(f"**Status:** {meta['status']}")
    if meta.get("last_evaluated"):
        meta_parts.append(f"**Last Evaluated:** {meta['last_evaluated']}")
    if meta_parts:
        sections.append(" | ".join(meta_parts))

    thesis_content = _existing_core_thesis_section(ticker)
    if thesis_content:
        sections.append(f"\n{thesis_content}")

    catalysts = runtime.catalysts(ticker, limit=500)
    active_catalysts = [row for row in catalysts if str(row.get("status") or "pending").lower() == "pending"]
    if active_catalysts:
        lines = ["## Key Catalysts", ""]
        for catalyst in active_catalysts:
            description = str(catalyst.get("description") or catalyst.get("name") or "").strip()
            if not description:
                continue
            status = str(catalyst.get("status") or "pending")
            status_tag = f" [{status}]" if status != "pending" else ""
            target = f" (target: {catalyst['target_date']})" if catalyst.get("target_date") else ""
            lines.append(f"{_format_entity_bullet(description)}{target}{status_tag}")
            if catalyst.get("evidence"):
                lines.append(f"  - Evidence: {catalyst['evidence']}")
        sections.append("\n".join(lines))

    kill_conditions = runtime.kill_conditions(ticker, limit=500)
    active_conditions = [row for row in kill_conditions if str(row.get("status") or "active").lower() == "active"]
    if active_conditions:
        lines = ["## Risk Factors / Kill Conditions", ""]
        for condition in active_conditions:
            text = str(condition.get("condition") or condition.get("description") or "").strip()
            if not text:
                continue
            metric = (
                f" (metric: {condition['metric']}, threshold: {condition['threshold']})"
                if condition.get("metric")
                else ""
            )
            lines.append(f"{_format_entity_bullet(text)}{metric}")
        sections.append("\n".join(lines))

    claims = [row for row in runtime.thesis_claims(ticker, limit=500) if row.get("status") != "retired"]
    if claims:
        lines = ["## Thesis Claims", ""]
        for claim in claims:
            lines.extend(_format_claim(claim))
        sections.append("\n".join(lines))

    evaluations = runtime.evaluations(ticker, limit=1)
    if evaluations:
        ev = evaluations[0]
        lines = ["## Latest Evaluation", ""]
        lines.append(
            f"**Date:** {ev.get('evaluated_at', 'N/A')} | "
            f"**Action:** {ev.get('action', 'N/A')} | "
            f"**Confidence:** {ev.get('confidence', 'N/A')}"
        )
        if ev.get("risk_flag"):
            lines.append(f"\n**Risk Flag:** {ev['risk_flag']}")
        if ev.get("technical_read"):
            lines.append(f"\n**Technical:** {ev['technical_read']}")
        if ev.get("fundamental_read"):
            lines.append(f"\n**Fundamental:** {ev['fundamental_read']}")
        developments = ev.get("key_developments")
        if isinstance(developments, list) and developments:
            lines.append("\n**Key Developments:**")
            for item in developments:
                lines.append(f"- {item}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections) + "\n"


def _existing_core_thesis_section(ticker: str) -> str | None:
    thesis_path = PROJECT_ROOT / "investment_theses" / f"{ticker}.md"
    thesis_key = f"live/theses/{ticker}.md"
    try:
        if not exists_text(thesis_path, thesis_key):
            return None
        raw = read_text(thesis_path, thesis_key, encoding="utf-8").strip()
    except Exception:
        logger.debug("Unable to read existing thesis markdown for %s", ticker, exc_info=True)
        return None

    lines = raw.splitlines()
    core_lines: list[str] = []
    in_core = False
    for line in lines:
        if line.strip().startswith("# ") and not in_core:
            in_core = True
            continue
        if in_core and line.strip().startswith("## "):
            break
        if in_core:
            core_lines.append(line)
    content = "\n".join(core_lines).strip()
    return content or None


def _format_claim(claim: dict) -> list[str]:
    claim_text = str(claim.get("claim") or "").strip()
    if not claim_text:
        return []
    if ": " in claim_text:
        label, rest = claim_text.split(": ", 1)
        lines = [f"- **{label}:** {rest}"]
    else:
        lines = [f"- **{claim_text}**"]
    lines.append(f"  - Status: {claim.get('status') or 'active'}")
    if claim.get("expected_evidence"):
        lines.append(f"  - Expected evidence: {claim['expected_evidence']}")
    if claim.get("disconfirming_evidence"):
        lines.append(f"  - Disconfirming evidence: {claim['disconfirming_evidence']}")
    if claim.get("cadence"):
        lines.append(f"  - Cadence: {claim['cadence']}")
    if claim.get("confidence") is not None:
        try:
            confidence = f"{float(claim['confidence']):.2f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            confidence = str(claim["confidence"])
        lines.append(f"  - Confidence: {confidence}")
    return lines
