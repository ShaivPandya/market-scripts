from __future__ import annotations

from typing import Any

import pytest

from ontology.command_service import OntologyCommandContext, OntologyCommandService
from ontology.evidence_ledger import build_ticker_evidence_ledger, parse_evidence_items, write_claim_evidence_graph
from ontology.object_service import OntologyObjectService
from ontology.policy import admin_actor
from ontology.runtime_read_service import OntologyRuntimeReadService
from tests.test_ontology_command_service import NormalizingTemporalRepo


def test_parse_evidence_items_accepts_text_and_json():
    assert parse_evidence_items("Quarterly revenue beat") == ["Quarterly revenue beat"]
    assert parse_evidence_items('[{"summary": "Revenue beat", "url": "https://example.com"}]') == [
        {"summary": "Revenue beat", "url": "https://example.com"}
    ]
    assert parse_evidence_items(None) == []


def test_write_claim_evidence_graph_creates_objects_and_relations():
    repo = NormalizingTemporalRepo()
    service = OntologyObjectService(repository=repo)  # type: ignore[arg-type]
    claim_uid = "thesis_claim:MU:ai-demand"
    rows = write_claim_evidence_graph(
        service,
        claim_uid=claim_uid,
        claim_key="MU:AI demand remains durable",
        expected_evidence="Revenue growth accelerated in HBM",
        disconfirming_evidence={"summary": "Inventory build reported", "url": "https://example.com/report"},
        valid_from="2026-05-31T00:00:00Z",
        actor={"actor_type": "test", "actor_id": "unit"},
        provenance_id="pv:test:claim-evidence",
    )

    assert rows
    object_types = {row["object_type"] for row in repo.objects.values()}
    assert "Evidence" in object_types
    assert "Citation" in object_types
    relation_types = {row["relation_type"] for row in repo.relations}
    assert "claim_supported_by_evidence" in relation_types
    assert "claim_disconfirmed_by_evidence" in relation_types
    assert "evidence_has_citation" in relation_types


def test_save_thesis_content_writes_claim_evidence_relations(monkeypatch, tmp_path):
    import portfolio.thesis_content as thesis_content

    indexed: list[dict[str, Any]] = []
    thesis_dir = tmp_path / "investment_theses"
    thesis_dir.mkdir()
    monkeypatch.setattr(thesis_content, "THESES_DIR", thesis_dir)
    monkeypatch.setattr("api.retrieval.index_document", lambda **kwargs: indexed.append(kwargs))

    repo = NormalizingTemporalRepo()
    service = OntologyCommandService(OntologyObjectService(repository=repo))  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")
    content = """# MU

## Thesis Claims
- **AI demand remains durable:** HBM attach rates keep expanding.
  - Expected evidence: Revenue growth accelerated in HBM
  - Disconfirming evidence: Inventory build reported
"""

    approval = service.propose_action(
        "save_thesis_content",
        {"ticker": "mu", "content": content},
        context,
        reason="unit",
    )
    applied = service.resolve_approval(approval["id"], "approved", "apply", context)

    assert applied["application_status"] == "applied"
    relation_types = {row["relation_type"] for row in repo.relations}
    assert "claim_supported_by_evidence" in relation_types
    assert "claim_disconfirmed_by_evidence" in relation_types


def test_build_ticker_evidence_ledger_returns_claim_bundles():
    repo = NormalizingTemporalRepo()
    object_service = OntologyObjectService(repository=repo)  # type: ignore[arg-type]
    claim_row = object_service.write_object(
        "ThesisClaim",
        "MU:AI demand remains durable",
        {
            "ticker": "MU",
            "claim": "AI demand remains durable",
            "expected_evidence": "Revenue growth accelerated in HBM",
            "status": "active",
            "ontology_run_id": "operational",
        },
        "2026-05-31T00:00:00Z",
        actor={"actor_type": "test", "actor_id": "unit"},
        provenance="pv:test:thesis-claim",
    )
    claim_uid = str(claim_row.get("object_uid") or "")
    write_claim_evidence_graph(
        object_service,
        claim_uid=claim_uid,
        claim_key="MU:AI demand remains durable",
        expected_evidence="Revenue growth accelerated in HBM",
        disconfirming_evidence=None,
        valid_from="2026-05-31T00:00:00Z",
        actor={"actor_type": "test", "actor_id": "unit"},
        provenance_id="pv:test:ledger-read",
    )

    reads = OntologyRuntimeReadService(object_service=object_service)
    ledger = build_ticker_evidence_ledger(reads, "MU")

    assert ledger["ticker"] == "MU"
    assert ledger["counts"]["evidence_items"] >= 1
    assert ledger["claims"][0]["supporting_evidence"]


def test_dossier_router_includes_evidence_ledger(monkeypatch):
    import api.routers.dossier as dossier_router
    import api.state_storage as state_storage
    import portfolio.management_quality_content as management_quality_content

    class _Reads:
        def dossier_bundle(self, ticker: str):
            return {
                "position": {"ticker": ticker},
                "thesis_meta": {"ticker": ticker, "status": "active"},
                "management_quality_assessment": None,
                "evaluations": [],
                "catalysts": [],
                "kill_conditions": [],
                "thesis_claims": [],
                "workflow_runs": [],
                "action_items": [],
                "watch_triggers": [],
                "pending_approvals": [],
            }

        def evidence_ledger(self, ticker: str):
            assert ticker == "MU"
            return {
                "ticker": "MU",
                "generated_at": "2026-05-31T00:00:00Z",
                "claims": [],
                "recommendations": [],
                "counts": {"claims": 0, "recommendations": 0, "evidence_items": 0},
            }

    monkeypatch.setattr(dossier_router, "OntologyRuntimeReadService", _Reads)
    monkeypatch.setattr(state_storage, "exists_text", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(management_quality_content, "management_quality_exists", lambda *_args, **_kwargs: False)

    payload = dossier_router.get_dossier("mu")

    assert payload["evidence_ledger"]["ticker"] == "MU"
    assert payload["evidence_ledger"]["counts"]["evidence_items"] == 0
