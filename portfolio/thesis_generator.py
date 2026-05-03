"""
thesis_generator.py -- Generate thesis markdown from structured entities.

Reverse of thesis_backfill.py: reads catalysts, kill conditions, evaluations,
and thesis metadata from the databases and produces a formatted markdown document.
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_thesis_markdown(ticker: str) -> str:
    """Generate a full thesis markdown document for a given ticker.

    Pulls from thesis_db (meta, evaluations) and core_db (catalysts, kill conditions).
    Returns a markdown string.
    """
    ticker = ticker.strip().upper()
    sections: list[str] = []

    # Thesis metadata
    meta = None
    try:
        from portfolio.thesis_db import get_thesis_meta

        meta = get_thesis_meta(ticker)
    except Exception:
        pass

    # Header
    sections.append(f"# {ticker}")
    if meta:
        parts = []
        if meta.get("direction"):
            parts.append(f"**Direction:** {meta['direction']}")
        if meta.get("timeframe"):
            parts.append(f"**Timeframe:** {meta['timeframe']}")
        if meta.get("status"):
            parts.append(f"**Status:** {meta['status']}")
        if meta.get("last_evaluated"):
            parts.append(f"**Last Evaluated:** {meta['last_evaluated']}")
        if parts:
            sections.append(" | ".join(parts))

    # Thesis content (original markdown if available)
    thesis_content = None
    try:
        from api.state_storage import exists_text, read_text
        from paths import PROJECT_ROOT

        thesis_path = PROJECT_ROOT / "investment_theses" / f"{ticker}.md"
        thesis_key = f"live/theses/{ticker}.md"
        if exists_text(thesis_path, thesis_key):
            raw = read_text(thesis_path, thesis_key, encoding="utf-8").strip()
            # Extract core thesis section (between first # and ## Key Catalysts)
            lines = raw.splitlines()
            core_lines: list[str] = []
            in_core = False
            for line in lines:
                if line.strip().startswith("# ") and not in_core:
                    in_core = True
                    continue  # skip the header, we already have it
                if in_core and line.strip().startswith("## "):
                    break
                if in_core:
                    core_lines.append(line)
            thesis_content = "\n".join(core_lines).strip()
    except Exception:
        pass

    if thesis_content:
        sections.append(f"\n{thesis_content}")

    # Catalysts from core_db
    try:
        from portfolio.core_db import get_catalysts

        catalysts = get_catalysts(ticker)
        if catalysts:
            lines = ["## Key Catalysts", ""]
            for c in catalysts:
                status_tag = f" [{c['status']}]" if c["status"] != "pending" else ""
                target = f" (target: {c['target_date']})" if c.get("target_date") else ""
                lines.append(f"- **{c['category']}:** {c['description']}{target}{status_tag}")
                if c.get("evidence"):
                    lines.append(f"  - Evidence: {c['evidence']}")
            sections.append("\n".join(lines))
    except Exception:
        pass

    # Kill conditions from core_db
    try:
        from portfolio.core_db import get_kill_conditions

        conditions = get_kill_conditions(ticker)
        if conditions:
            lines = ["## Risk Factors / Kill Conditions", ""]
            for k in conditions:
                status_tag = f" **[{k['status'].upper()}]**" if k["status"] != "active" else ""
                metric = f" (metric: {k['metric']}, threshold: {k['threshold']})" if k.get("metric") else ""
                lines.append(f"- {k['condition']}{metric}{status_tag}")
            sections.append("\n".join(lines))
    except Exception:
        pass

    # Thesis claims from core_db
    try:
        from portfolio.core_db import get_catalysts, get_kill_conditions, get_thesis_claims

        catalysts = get_catalysts(ticker)
        conditions = get_kill_conditions(ticker)
        catalyst_labels = {
            int(c["id"]): str(c.get("description") or "").split(": ", 1)[0].strip() for c in catalysts if c.get("id")
        }
        condition_labels = {
            int(k["id"]): str(k.get("condition") or "").split(": ", 1)[0].strip() for k in conditions if k.get("id")
        }
        claims = [c for c in get_thesis_claims(ticker=ticker, limit=500) if c.get("status") != "retired"]
        if claims:
            lines = ["## Thesis Claims", ""]
            for claim in claims:
                claim_text = str(claim.get("claim") or "").strip()
                if ": " in claim_text:
                    label, rest = claim_text.split(": ", 1)
                    lines.append(f"- **{label}:** {rest}")
                else:
                    lines.append(f"- **{claim_text}**")
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
                            freshness = req.get("freshness_days")
                            freshness_text = "" if freshness in (None, "") else str(freshness)
                            required = "true" if req.get("required", True) else "false"
                            lines.append(
                                "    - "
                                f"type={req.get('type') or 'custom'}; "
                                f"description={req.get('description') or req.get('type') or 'custom'}; "
                                f"required={required}; freshness_days={freshness_text}"
                            )
                        else:
                            lines.append(f"    - {req}")
                if claim.get("cadence"):
                    lines.append(f"  - Cadence: {claim['cadence']}")
                if claim.get("confidence") is not None:
                    lines.append(f"  - Confidence: {float(claim['confidence']):.2f}".rstrip("0").rstrip("."))
                linked_catalysts = [
                    catalyst_labels[int(cid)]
                    for cid in claim.get("linked_catalyst_ids", claim.get("linked_catalyst_ids_json", []))
                    if int(cid) in catalyst_labels
                ]
                if linked_catalysts:
                    lines.append(f"  - Catalysts: {'; '.join(linked_catalysts)}")
                linked_conditions = [
                    condition_labels[int(kid)]
                    for kid in claim.get("linked_kill_condition_ids", claim.get("linked_kill_condition_ids_json", []))
                    if int(kid) in condition_labels
                ]
                if linked_conditions:
                    lines.append(f"  - Kill conditions: {'; '.join(linked_conditions)}")
            sections.append("\n".join(lines))
    except Exception:
        pass

    # Latest evaluation
    try:
        from portfolio.thesis_db import get_evaluations

        evals = get_evaluations(ticker, limit=1)
        if evals:
            ev = evals[0]
            lines = ["## Latest Evaluation", ""]
            lines.append(
                f"**Date:** {ev.get('evaluated_at', 'N/A')} | **Action:** {ev.get('action', 'N/A')} | **Confidence:** {ev.get('confidence', 'N/A')}"
            )
            if ev.get("risk_flag"):
                lines.append(f"\n**Risk Flag:** {ev['risk_flag']}")
            if ev.get("technical_read"):
                lines.append(f"\n**Technical:** {ev['technical_read']}")
            if ev.get("fundamental_read"):
                lines.append(f"\n**Fundamental:** {ev['fundamental_read']}")
            devs = ev.get("key_developments")
            if devs and isinstance(devs, list):
                lines.append("\n**Key Developments:**")
                for d in devs:
                    lines.append(f"- {d}")
            sections.append("\n".join(lines))
    except Exception:
        pass

    return "\n\n".join(sections) + "\n"
