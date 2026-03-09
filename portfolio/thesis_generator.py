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
        from paths import PROJECT_ROOT

        thesis_path = PROJECT_ROOT / "investment_theses" / f"{ticker}.md"
        if thesis_path.exists():
            raw = thesis_path.read_text(encoding="utf-8").strip()
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
