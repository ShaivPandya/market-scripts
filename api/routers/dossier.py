"""Position dossier aggregate API endpoint -- unified per-ticker view."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from api.decision_state import normalize_action_item, normalize_approval
from api.exceptions import NotFoundError
from ontology.domain_write_service import ontology_primary_writes_enabled
from ontology.runtime_read_service import OntologyRuntimeReadService

router = APIRouter()
logger = logging.getLogger("api.dossier")


@router.get("/dossier/{ticker}")
def get_dossier(ticker: str):
    """Return unified dossier for a single position."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise NotFoundError("Position", "")

    reads = OntologyRuntimeReadService()
    ontology_bundle = reads.dossier_bundle(ticker)
    position = ontology_bundle.get("position")
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
    if ontology_primary_writes_enabled():
        try:
            from api.routers.management_quality import _render_management_quality_markdown

            if management_quality_assessment:
                management_quality_content = _render_management_quality_markdown(ticker, management_quality_assessment)
        except Exception:
            management_quality_assessment = None
    try:
        from portfolio.management_quality_content import (
            management_quality_exists,
            read_management_quality,
        )

        if management_quality_exists(ticker):
            management_quality_content = read_management_quality(ticker)
    except Exception:
        pass

    management_quality_parsed = None
    if management_quality_content:
        try:
            from api.routers.management_quality import parse_management_quality_markdown

            management_quality_parsed = parse_management_quality_markdown(management_quality_content)
        except Exception:
            pass

    evaluations = ontology_bundle.get("evaluations", [])
    status_history: list[dict[str, Any]] = []
    catalysts = ontology_bundle.get("catalysts", [])
    kill_conditions = ontology_bundle.get("kill_conditions", [])
    thesis_claims = ontology_bundle.get("thesis_claims", [])
    workflow_runs = ontology_bundle.get("workflow_runs", [])
    action_items = [normalize_action_item(a) for a in ontology_bundle.get("action_items", [])]
    watch_triggers = ontology_bundle.get("watch_triggers", [])
    pending_approvals = [normalize_approval(a) for a in ontology_bundle.get("pending_approvals", [])]

    # Ontology risk is loaded lazily by the frontend Risk tab. Keep the field
    # in the aggregate payload for backwards compatibility without triggering
    # expensive ontology/macro ingestion during dossier navigation.
    ontology_risk = None

    return {
        "ticker": ticker,
        "position": position,
        "overview_content": overview_content,
        "overview_parsed": overview_parsed,
        "management_quality": {
            "content": management_quality_content,
            "parsed": management_quality_assessment.get("parsed")
            if management_quality_assessment
            else management_quality_parsed,
            "assessment": management_quality_assessment,
        },
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
        "pending_approvals": pending_approvals,
    }
