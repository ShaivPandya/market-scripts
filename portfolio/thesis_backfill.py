"""Parse thesis markdown and seed ontology-backed thesis objects."""

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
    """Parse thesis markdown files and apply them through the ontology command layer.

    Returns a summary dict: {ticker: {catalysts: N, kill_conditions: N}}.
    """
    from ontology.command_service import OntologyCommandContext, OntologyCommandService
    from ontology.policy import system_actor
    from portfolio.thesis_sync import _parse_structured_claims, _parse_text_claims

    if theses_dir is None:
        from paths import PROJECT_ROOT

        theses_dir = PROJECT_ROOT / "investment_theses"

    if not theses_dir.exists():
        logger.info("No theses directory found at %s", theses_dir)
        return {}

    summary: dict[str, dict[str, int]] = {}
    service = OntologyCommandService()
    context = OntologyCommandContext(
        actor=system_actor("thesis_backfill"),
        source_type="thesis_markdown",
        source_id=str(theses_dir),
    )

    for md_file in sorted(theses_dir.glob("*.md")):
        ticker = md_file.stem.upper()
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        catalyst_bullets = _parse_bullets(content, "Key Catalysts")
        risk_bullets = _parse_bullets(content, "Risk Factors")
        claims = _parse_structured_claims(content)
        claim_count = len(claims if claims is not None else _parse_text_claims(content))
        approval = service.propose_action(
            "save_thesis_content",
            {"ticker": ticker, "content": content, "preserve_exact_content": True},
            context,
            reason="Seed ontology thesis content from markdown",
        )
        service.resolve_approval(
            str(approval["id"]),
            "approved",
            "Seed ontology thesis content from markdown",
            context,
        )
        cat_count = len([item for item in catalyst_bullets if item and item.strip() != "TBD"])
        kc_count = len([item for item in risk_bullets if item and item.strip() != "TBD"])
        summary[ticker] = {"catalysts": cat_count, "kill_conditions": kc_count, "thesis_claims": claim_count}
        logger.info(
            "thesis_backfill: %s -> %d catalysts, %d kill conditions, %d thesis claims",
            ticker,
            cat_count,
            kc_count,
            claim_count,
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
