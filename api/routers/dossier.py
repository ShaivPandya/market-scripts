"""Position dossier aggregate API endpoint -- unified per-ticker view."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from api.decision_state import normalize_action_item, normalize_approval, normalize_decision_outcome
from api.exceptions import NotFoundError
from ontology.change_summary import ChangeSummaryInputError, build_dossier_change_summary
from ontology.runtime_read_service import OntologyRuntimeReadService

router = APIRouter()
logger = logging.getLogger("api.dossier")


@router.get("/dossier/{ticker}")
def get_dossier(ticker: str, since: str | None = None):
    """Return unified dossier for a single position."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise NotFoundError("Position", "")

    reads = OntologyRuntimeReadService()
    ontology_bundle = reads.dossier_bundle(ticker)
    position = ontology_bundle.get("position")
    related_portfolio_legs: list[dict[str, Any]] = []
    if position is None:
        related_portfolio_legs = [
            row
            for row in reads.positions(include_hedges=False)
            if str(row.get("ticker") or "").strip().upper() == ticker
            or str(row.get("underlying_ticker") or "").strip().upper() == ticker
        ]
        if related_portfolio_legs:
            position = related_portfolio_legs[0]
    thesis_meta = ontology_bundle.get("thesis_meta")
    thesis_content = None

    try:
        from api.state_storage import exists_text, read_text
        from paths import PROJECT_ROOT

        thesis_path = PROJECT_ROOT / "investment_theses" / f"{ticker}.md"
        thesis_key = f"live/theses/{ticker}.md"
        if exists_text(thesis_path, thesis_key):
            thesis_content = read_text(thesis_path, thesis_key, encoding="utf-8")
    except Exception:
        pass

    # Overview content (equity overview markdown)
    overview_content = None
    try:
        from api.state_storage import exists_text, read_text
        from paths import PROJECT_ROOT as _PR

        overview_path = _PR / "investment_overviews" / f"{ticker}.md"
        overview_key = f"live/overviews/{ticker}.md"
        if exists_text(overview_path, overview_key):
            overview_content = read_text(overview_path, overview_key, encoding="utf-8")
    except Exception:
        pass

    overview_parsed = None
    if overview_content:
        try:
            from api.routers.overview import parse_overview_markdown

            overview_parsed = parse_overview_markdown(overview_content)
        except Exception:
            pass

    # Management quality content
    management_quality_content = None
    management_quality_assessment = ontology_bundle.get("management_quality_assessment")
    try:
        from api.routers.management_quality import _render_management_quality_markdown

        if management_quality_assessment:
            management_quality_content = _render_management_quality_markdown(ticker, management_quality_assessment)
    except Exception:
        management_quality_assessment = None

    management_quality_parsed = None
    if management_quality_content:
        try:
            from api.routers.management_quality import parse_management_quality_markdown

            management_quality_parsed = parse_management_quality_markdown(management_quality_content)
        except Exception:
            pass

    evaluations = ontology_bundle.get("evaluations", [])
    status_history = reads.thesis_status_history(ticker, limit=20)
    catalysts = ontology_bundle.get("catalysts", [])
    kill_conditions = ontology_bundle.get("kill_conditions", [])
    thesis_claims = ontology_bundle.get("thesis_claims", [])
    workflow_runs = ontology_bundle.get("workflow_runs", [])
    action_items = [normalize_action_item(a) for a in ontology_bundle.get("action_items", [])]
    watch_triggers = ontology_bundle.get("watch_triggers", [])
    monitor_hits = ontology_bundle.get("monitor_hits", [])
    pending_approvals = [normalize_approval(a) for a in ontology_bundle.get("pending_approvals", [])]
    decision_outcomes = [
        normalize_decision_outcome(item)
        for item in ontology_bundle.get("decision_outcomes", [])
        if isinstance(item, dict)
    ]
    try:
        what_changed = build_dossier_change_summary(ontology_bundle, ticker, since=since)
    except ChangeSummaryInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    evidence_ledger = reads.evidence_ledger(ticker)

    # Ontology risk is loaded lazily by the frontend Risk tab to avoid
    # expensive macro ingestion during dossier navigation.
    ontology_risk = None

    return {
        "ticker": ticker,
        "position": position,
        "related_portfolio_legs": related_portfolio_legs,
        "overview_content": overview_content,
        "overview_parsed": overview_parsed,
        "management_quality": {
            "content": management_quality_content,
            "parsed": management_quality_assessment.get("parsed")
            if management_quality_assessment
            else management_quality_parsed,
            "assessment": management_quality_assessment,
        },
        "what_changed": what_changed,
        "evidence_ledger": evidence_ledger,
        "thesis": {
            "meta": thesis_meta,
            "content": thesis_content,
            "status_history": status_history,
        },
        "evaluations": evaluations,
        "catalysts": catalysts,
        "kill_conditions": kill_conditions,
        "thesis_claims": thesis_claims,
        "ontology_risk": ontology_risk,
        "workflow_runs": workflow_runs,
        "action_items": action_items,
        "watch_triggers": watch_triggers,
        "monitor_hits": monitor_hits,
        "pending_approvals": pending_approvals,
        "decision_outcomes": decision_outcomes,
    }
