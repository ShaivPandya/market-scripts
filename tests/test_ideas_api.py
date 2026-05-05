from __future__ import annotations

import time

import pytest

OVERVIEW_MARKDOWN = """# AAPL Overview

## Financials
- **3-Year Avg. YoY Revenue Growth**: +8.0% supported by services growth.
- **3-Year Avg. YoY EPS Growth**: +10.0% supported by buybacks.
- **Debt**: Balanced maturity schedule.
| Tranche | Rate | Maturity |
|---------|------|----------|
| Notes | 3.0% | 2030 |
- **Reinvestment Costs**: R&D remains elevated.

## Sensitivity to Extrinsic Factors
| Factor | Sensitivity | Capacity to Deal |
|--------|-------------|------------------|
| Currency/FX | Medium | Global pricing helps offset pressure. |

## Industry

### Porter's Five Forces
- **Threat of New Entrants - Low**: Ecosystem and scale are durable.

### Supply Outlook
- **Components**: Supply remains moderate.

### Demand Outlook
- **Installed Base**: Strong demand from services adoption.
"""

MANAGEMENT_QUALITY_MARKDOWN = """# AAPL Management Quality

## Executive Summary
- **Overall Rating**: Strong
- **Bottom Line**: Managers have generally acted like owners.
- **Owner Mindset**: Strong - Capital allocation has been shareholder-oriented.
- **Business Value Understanding**: Strong - Management understands services and ecosystem value.
- **Follow-through / Character**: Mixed - Most targets were met, with some product delays.

## Management Scorecard
| Question | Rating | Evidence |
|----------|--------|----------|
| Do managers think and act like owners? | Strong | Buybacks and capital returns were disciplined. |

## Most Impressive Accomplishments
- **Services growth (2025)**: Expanded recurring revenue.

## Biggest Setbacks and Responses
- **Product delay (2024)**: Launch slipped. **Response**: Mixed - Reset timing.
"""


@pytest.fixture(autouse=True)
def _isolate_ideas_runtime(tmp_path, monkeypatch):
    from api import job_queue
    from api.routers import ideas as ideas_router
    from portfolio import core_db

    if core_db._conn:
        core_db._conn.close()
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "ideas_core.db")
    monkeypatch.setattr(core_db, "_conn", None)
    job_queue._memory_jobs.clear()

    monkeypatch.setattr(
        ideas_router,
        "_read_state_text",
        lambda folder, ticker: (
            OVERVIEW_MARKDOWN
            if folder == "investment_overviews"
            else MANAGEMENT_QUALITY_MARKDOWN
            if folder == "investment_management_quality"
            else None,
            None,
        ),
    )

    def _context(idea: dict):
        return {
            "idea": idea,
            "ticker": idea["ticker"],
            "overview_content": OVERVIEW_MARKDOWN,
            "thesis_content": None,
            "management_quality_content": MANAGEMENT_QUALITY_MARKDOWN,
            "portfolio": {"ok": True, "data": {"positions": []}},
            "signal_aggregator": {"ok": True, "data": {"regime": "risk-on"}},
            "industry_monitor": {"ok": True, "data": {}},
            "dossier": {"ok": True, "data": {}},
            "tool_errors": [],
            "evaluated_at": "2026-05-05T12:00:00+00:00",
        }

    def _evaluation(context: dict):
        result = {
            "idea_id": context["idea"]["id"],
            "ticker": context["ticker"],
            "evaluated_at": context["evaluated_at"],
            "action": "buy",
            "recommendation_status": "clear",
            "score": 82,
            "confidence": 0.78,
            "thesis_statement": "High-quality idea with supportive setup.",
            "rationale": "Macro, quality, and portfolio context are acceptable for review.",
            "factor_scores": {
                "macro_support": {"score": 80, "status": "supportive", "rationale": "Risk-on regime."},
                "industry_attractiveness": {
                    "score": 78,
                    "status": "good",
                    "rationale": "Industry context is acceptable.",
                },
                "business_quality": {"score": 86, "status": "strong", "rationale": "Overview supports quality."},
                "management_quality": {
                    "score": 74,
                    "status": "good",
                    "rationale": "Evidence is sufficient for review.",
                },
                "valuation_asymmetry": {"score": 80, "status": "good", "rationale": "Upside/downside is acceptable."},
                "portfolio_fit": {"score": 90, "status": "good", "rationale": "No current concentration conflict."},
            },
            "missing_information": [],
            "data_quality": {"critical_data_quality": "ok", "source_quality": "ok", "quality": "ok"},
            "evidence": [{"source": "overview", "summary": "Business evidence is present."}],
            "disconfirming_evidence": [{"source": "risk", "summary": "Monitor valuation sensitivity."}],
            "catalyst": "Earnings and valuation review.",
            "invalidation": "Invalidate if valuation downside overwhelms upside.",
            "portfolio_fit": {"status": "acceptable"},
        }
        result["recommendation_record"] = ideas_router._recommendation_record_from_result(context["idea"], result)
        return result

    monkeypatch.setattr(ideas_router, "_build_context", _context)
    monkeypatch.setattr(ideas_router, "_call_llm_evaluator", _evaluation)
    yield
    if core_db._conn:
        core_db._conn.close()
    monkeypatch.setattr(core_db, "_conn", None)
    job_queue._memory_jobs.clear()


def _poll_until_done(auth_client, job_id: str) -> dict:
    for _ in range(40):
        resp = auth_client.get(f"/api/v1/ideas/evaluate/async/{job_id}")
        assert resp.status_code == 200
        payload = resp.json()
        if payload["status"] in {"done", "error", "cancelled"}:
            return payload
        time.sleep(0.05)
    pytest.fail("idea evaluation job did not finish")


def test_ideas_crud_evaluate_and_accept(auth_client):
    created = auth_client.post(
        "/api/v1/ideas",
        json={
            "ticker": "AAPL",
            "company_name": "Apple",
            "user_notes": "Review for quality growth.",
            "tags": ["quality"],
        },
    )
    assert created.status_code == 200
    created_payload = created.json()
    idea = created_payload["idea"]
    assert idea["ticker"] == "AAPL"
    assert created_payload["documents"]["overview_parsed"]["financials"]["revenue_growth"]["value"] == "+8.0%"
    assert created_payload["documents"]["overview_parsed"]["porters_five_forces"][0]["rating"] == "Low"
    assert created_payload["documents"]["management_quality_parsed"]["summary"]["overall_rating"] == "Strong"

    listed = auth_client.get("/api/v1/ideas")
    assert listed.status_code == 200
    assert listed.json()["count"] == 1

    started = auth_client.post(f"/api/v1/ideas/{idea['id']}/evaluate/async", json={})
    assert started.status_code in {200, 202}
    job = _poll_until_done(auth_client, started.json()["job_id"])
    assert job["status"] == "done"
    evaluation = job["result"]["evaluation"]
    assert evaluation["action"] == "buy"
    assert evaluation["missing_information"] == []

    accepted = auth_client.post(f"/api/v1/ideas/{idea['id']}/evaluations/{evaluation['id']}/accept", json={})
    assert accepted.status_code == 200
    accepted_payload = accepted.json()
    assert accepted_payload["status"] == "accepted"
    assert accepted_payload["idea"]["status"] == "accepted"
    assert accepted_payload["recommendation"]["action"] == "buy"
    assert accepted_payload["recommendation"]["policy_gate_disclosures_json"]
    assert accepted_payload["action_proposal"]["approval_id"] is not None


def test_critical_missing_information_forces_watch():
    from api.routers import ideas as ideas_router

    context = {
        "idea": {"id": 1, "ticker": "BCS"},
        "ticker": "BCS",
        "tool_errors": [],
        "evaluated_at": "2026-05-05T12:00:00+00:00",
    }
    result = ideas_router._normalize_llm_result(
        context,
        {
            "action": "buy",
            "recommendation_status": "clear",
            "missing_information": [{"field": "management", "severity": "critical", "reason": "No evidence."}],
            "factor_scores": {"management_quality": {"score": 0}},
            "rationale": "Incomplete.",
        },
    )

    assert result["action"] == "watch"
    assert result["missing_information"][0]["field"] == "management"
