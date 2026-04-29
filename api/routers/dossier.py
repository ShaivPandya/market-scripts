"""Position dossier aggregate API endpoint -- unified per-ticker view."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from api.exceptions import NotFoundError

router = APIRouter()
logger = logging.getLogger("api.dossier")


@router.get("/dossier/{ticker}")
def get_dossier(ticker: str):
    """Return unified dossier for a single position."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise NotFoundError("Position", "")

    from portfolio.core_db import (
        get_action_items,
        get_catalysts,
        get_kill_conditions,
        get_pending_approvals,
        get_research_notes,
        get_watch_triggers,
        get_workflow_runs,
    )

    # Position from portfolio_db
    position = None
    try:
        from portfolio.portfolio_db import get_positions

        for pos in get_positions():
            if str(pos.get("ticker", "")).upper() == ticker:
                position = pos
                break
    except Exception:
        pass

    # Thesis meta + content
    thesis_meta = None
    thesis_content = None
    try:
        from portfolio.thesis_db import get_thesis_meta

        thesis_meta = get_thesis_meta(ticker)
    except Exception:
        pass

    try:
        from paths import PROJECT_ROOT

        thesis_path = PROJECT_ROOT / "investment_theses" / f"{ticker}.md"
        if thesis_path.exists():
            thesis_content = thesis_path.read_text(encoding="utf-8")
    except Exception:
        pass

    # Overview content (equity overview markdown)
    overview_content = None
    try:
        from paths import PROJECT_ROOT as _PR

        overview_path = _PR / "investment_overviews" / f"{ticker}.md"
        if overview_path.exists():
            overview_content = overview_path.read_text(encoding="utf-8")
    except Exception:
        pass

    overview_parsed = None
    if overview_content:
        try:
            from api.routers.overview import parse_overview_markdown

            overview_parsed = parse_overview_markdown(overview_content)
        except Exception:
            pass

    # Evaluations from thesis_db
    evaluations = []
    try:
        from portfolio.thesis_db import get_evaluations

        evaluations = get_evaluations(ticker, limit=52)
    except Exception:
        pass

    # Status history
    status_history = []
    try:
        from portfolio.thesis_db import get_status_history

        status_history = get_status_history(ticker)
    except Exception:
        pass

    # Core entities from core_db
    catalysts = get_catalysts(ticker)
    kill_conditions = get_kill_conditions(ticker)
    workflow_runs = get_workflow_runs(ticker=ticker, limit=10)
    action_items = get_action_items(ticker=ticker, status="open")
    watch_triggers = get_watch_triggers(ticker=ticker)
    research_notes = get_research_notes(ticker=ticker, limit=20)
    pending_approvals = get_pending_approvals(ticker=ticker)

    # Ontology risk is loaded lazily by the frontend Risk tab. Keep the field
    # in the aggregate payload for backwards compatibility without triggering
    # expensive ontology/macro ingestion during dossier navigation.
    ontology_risk = None

    return {
        "ticker": ticker,
        "position": position,
        "overview_content": overview_content,
        "overview_parsed": overview_parsed,
        "thesis": {
            "meta": thesis_meta,
            "content": thesis_content,
            "status_history": status_history,
        },
        "evaluations": evaluations,
        "catalysts": catalysts,
        "kill_conditions": kill_conditions,
        "ontology_risk": ontology_risk,
        "workflow_runs": workflow_runs,
        "action_items": action_items,
        "watch_triggers": watch_triggers,
        "research_notes": research_notes,
        "pending_approvals": pending_approvals,
    }
