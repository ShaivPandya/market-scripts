"""thesis_sync.py -- Bidirectional sync between thesis markdown and DB entities.

Direction 1 (markdown -> DB):
  When thesis markdown is saved/generated, parse ## Key Catalysts and
  ## Risk Factors sections and replace backfill-created DB entries.

Direction 2 (DB -> markdown):
  When catalysts/kill conditions are created or updated via API,
  regenerate the corresponding markdown sections from DB state.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers: markdown <-> DB formatting
# ---------------------------------------------------------------------------


def _format_entity_bullet(text: str) -> str:
    """Convert a DB description like 'Label: Rest' to '- **Label:** Rest'."""
    if ": " in text:
        label, rest = text.split(": ", 1)
        return f"- **{label}:** {rest}"
    return f"- **{text}**"


def _replace_section(content: str, section_header: str, new_bullets: list[str]) -> str:
    """Replace bullets under a ## section header in markdown content."""
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
            # Insert new bullets
            for bullet in new_bullets:
                result.append(bullet)
            if not new_bullets:
                result.append("- TBD")
            continue

        if in_section:
            stripped = line.strip()
            # Next section or heading ends the current section
            if stripped.startswith("## ") or stripped.startswith("# "):
                in_section = False
                result.append("")  # blank line before next section
                result.append(line)
            # Skip old content (bullets, blank lines, etc.)
            continue

        result.append(line)

    # If section didn't exist in the file, append it
    if not replaced:
        result.append("")
        result.append(f"## {section_header}")
        for bullet in new_bullets:
            result.append(bullet)
        if not new_bullets:
            result.append("- TBD")

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Direction 1: Markdown -> DB
# ---------------------------------------------------------------------------


def sync_entities_from_markdown(ticker: str) -> dict[str, int]:
    """Parse thesis markdown and sync catalysts/kill conditions to DB.

    Replaces all 'backfill'-created entities for this ticker with
    freshly parsed entries from the thesis markdown.
    User/agent/workflow-created entities are left untouched.

    Returns: {"catalysts": N, "kill_conditions": N}
    """
    from paths import PROJECT_ROOT
    from portfolio.core_db import (
        create_catalyst,
        create_kill_condition,
        delete_catalysts_by_ticker,
        delete_kill_conditions_by_ticker,
    )
    from portfolio.thesis_backfill import (
        _categorize_catalyst,
        _extract_label_and_description,
        _parse_bullets,
    )

    ticker = ticker.upper()
    thesis_path = PROJECT_ROOT / "investment_theses" / f"{ticker}.md"
    if not thesis_path.exists():
        return {"catalysts": 0, "kill_conditions": 0}

    content = thesis_path.read_text(encoding="utf-8").strip()
    if not content:
        return {"catalysts": 0, "kill_conditions": 0}

    # Remove old backfill entries
    delete_catalysts_by_ticker(ticker, created_by="backfill")
    delete_kill_conditions_by_ticker(ticker, created_by="backfill")

    # Parse and recreate
    catalyst_bullets = _parse_bullets(content, "Key Catalysts")
    risk_bullets = _parse_bullets(content, "Risk Factors")

    cat_count = 0
    for bullet in catalyst_bullets:
        if not bullet or bullet.strip() == "TBD":
            continue
        label, desc = _extract_label_and_description(bullet)
        category = _categorize_catalyst(label, desc)
        create_catalyst(
            ticker=ticker,
            description=f"{label}: {desc}" if desc != label else label,
            category=category,
            created_by="backfill",
        )
        cat_count += 1

    kc_count = 0
    for bullet in risk_bullets:
        if not bullet or bullet.strip() == "TBD":
            continue
        label, desc = _extract_label_and_description(bullet)
        create_kill_condition(
            ticker=ticker,
            condition=f"{label}: {desc}" if desc != label else label,
            created_by="backfill",
        )
        kc_count += 1

    logger.info("thesis_sync: %s markdown->DB: %d catalysts, %d kill conditions", ticker, cat_count, kc_count)
    return {"catalysts": cat_count, "kill_conditions": kc_count}


# ---------------------------------------------------------------------------
# Direction 2: DB -> Markdown
# ---------------------------------------------------------------------------


def sync_markdown_from_entities(ticker: str) -> bool:
    """Read DB entities and update thesis markdown sections.

    Regenerates ## Key Catalysts and ## Risk Factors from all DB entries
    (regardless of created_by), keeping the rest of the thesis unchanged.

    Returns True if the file was updated, False if no thesis file exists.
    """
    from paths import PROJECT_ROOT
    from portfolio.core_db import get_catalysts, get_kill_conditions

    ticker = ticker.upper()
    thesis_path = PROJECT_ROOT / "investment_theses" / f"{ticker}.md"
    if not thesis_path.exists():
        return False

    content = thesis_path.read_text(encoding="utf-8")

    # Only include active entities in the markdown
    catalysts = get_catalysts(ticker)
    kill_conditions = get_kill_conditions(ticker)

    cat_bullets = [
        _format_entity_bullet(c["description"])
        for c in catalysts
        if c.get("status") == "pending"
    ]
    kc_bullets = [
        _format_entity_bullet(k["condition"])
        for k in kill_conditions
        if k.get("status") == "active"
    ]

    updated = _replace_section(content, "Key Catalysts", cat_bullets)
    updated = _replace_section(updated, "Risk Factors", kc_bullets)

    new_content = updated.rstrip() + "\n"
    if new_content == content:
        return False  # No changes needed

    thesis_path.write_text(new_content, encoding="utf-8")
    logger.info("thesis_sync: updated markdown for %s", ticker)
    return True
