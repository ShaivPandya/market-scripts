"""
thesis_backfill.py -- Parse thesis markdown files into structured catalysts and kill conditions.

Reads each .md file in investment_theses/, extracts bullets from
## Key Catalysts -> catalysts table, ## Risk Factors -> kill_conditions table.

Can be run as a CLI script or called from _init_db on first run.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_bullets(text: str, section_header: str) -> list[str]:
    """Extract bullet items under a specific ## section header."""
    pattern = rf"^## {re.escape(section_header)}\s*$"
    lines = text.splitlines()
    in_section = False
    bullets: list[str] = []

    for line in lines:
        if re.match(pattern, line.strip()):
            in_section = True
            continue
        if in_section:
            # New section starts
            if line.strip().startswith("## ") or line.strip().startswith("# "):
                break
            stripped = line.strip()
            if stripped.startswith("- "):
                bullets.append(stripped[2:].strip())

    return bullets


def _extract_label_and_description(bullet: str) -> tuple[str, str]:
    """Extract a bold label and remaining description from a bullet.

    E.g. '**HBM ramp:** HBM3 fully sold out...' -> ('HBM ramp', 'HBM3 fully sold out...')
    """
    match = re.match(r"\*\*(.+?)[:\*]+\*?\*?\s*(.*)", bullet)
    if match:
        label = match.group(1).strip().rstrip(":")
        desc = match.group(2).strip()
        return label, desc if desc else label
    return bullet[:80], bullet


def _categorize_catalyst(label: str, description: str) -> str:
    """Infer catalyst category from content."""
    text = (label + " " + description).lower()
    if any(w in text for w in ["earnings", "revenue", "margin", "guidance", "financial", "dividend", "buyback"]):
        return "fundamental"
    if any(w in text for w in ["breakout", "technical", "chart", "momentum", "support", "resistance"]):
        return "technical"
    if any(w in text for w in ["macro", "rate", "gdp", "inflation", "liquidity", "fed", "ecb", "boj"]):
        return "macro"
    if any(
        w in text for w in ["regulation", "regulatory", "chips act", "subsid", "tariff", "sanction", "geopolitical"]
    ):
        return "regulatory"
    if any(w in text for w in ["event", "launch", "ipo", "merger", "acquisition", "fda"]):
        return "event"
    return "fundamental"


def backfill_from_markdown(theses_dir: Path | None = None) -> dict[str, dict[str, int]]:
    """Parse all thesis markdown files and insert catalysts/kill conditions.

    Returns a summary dict: {ticker: {catalysts: N, kill_conditions: N}}.
    """
    from portfolio.core_db import create_catalyst, create_kill_condition, get_catalysts, get_kill_conditions

    if theses_dir is None:
        from paths import PROJECT_ROOT

        theses_dir = PROJECT_ROOT / "investment_theses"

    if not theses_dir.exists():
        logger.info("No theses directory found at %s", theses_dir)
        return {}

    summary: dict[str, dict[str, int]] = {}

    for md_file in sorted(theses_dir.glob("*.md")):
        ticker = md_file.stem.upper()
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        # Skip if this ticker already has backfilled data
        existing_catalysts = get_catalysts(ticker)
        if any(c.get("created_by") == "backfill" for c in existing_catalysts):
            continue

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

        summary[ticker] = {"catalysts": cat_count, "kill_conditions": kc_count}
        if cat_count or kc_count:
            logger.info(
                "thesis_backfill: %s -> %d catalysts, %d kill conditions",
                ticker,
                cat_count,
                kc_count,
            )

    return summary


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Allow custom theses dir as argument
    theses_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    result = backfill_from_markdown(theses_dir)
    for ticker, counts in result.items():
        print(f"{ticker}: {counts['catalysts']} catalysts, {counts['kill_conditions']} kill conditions")
    print(f"\nTotal: {len(result)} tickers backfilled")
