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


def test_delete_idea_removes_from_list_detail_and_archived_view(auth_client):
    created = auth_client.post(
        "/api/v1/ideas",
        json={
            "ticker": "AAPL",
            "company_name": "Apple",
            "user_notes": "Delete test.",
            "tags": ["delete"],
        },
    )
    assert created.status_code == 200
    idea = created.json()["idea"]

    deleted = auth_client.delete(f"/api/v1/ideas/{idea['id']}")

    assert deleted.status_code == 200
    deleted_payload = deleted.json()
    assert deleted_payload["status"] == "deleted"
    assert deleted_payload["deleted"] is True
    assert str(deleted_payload["idea_id"]) == str(idea["id"])

    listed = auth_client.get("/api/v1/ideas")
    assert listed.status_code == 200
    assert all(str(row["id"]) != str(idea["id"]) for row in listed.json()["ideas"])

    listed_with_archived = auth_client.get("/api/v1/ideas", params={"include_archived": True})
    assert listed_with_archived.status_code == 200
    assert all(str(row["id"]) != str(idea["id"]) for row in listed_with_archived.json()["ideas"])

    detail = auth_client.get(f"/api/v1/ideas/{idea['id']}")
    assert detail.status_code == 404


def test_evaluate_all_persists_comparative_run_and_excludes_inactive(auth_client, monkeypatch):
    from api.routers import ideas as ideas_router

    monkeypatch.setattr(
        ideas_router,
        "_call_llm_comparison_ranker",
        lambda evaluations: ideas_router._deterministic_comparison_result(evaluations),
    )

    for ticker, status in [
        ("MSFT", "watching"),
        ("AAPL", "ready_for_review"),
        ("TSLA", "accepted"),
        ("META", "rejected"),
        ("IBM", "watching"),
    ]:
        resp = auth_client.post(
            "/api/v1/ideas",
            json={
                "ticker": ticker,
                "company_name": ticker,
                "user_notes": f"Review {ticker}.",
                "tags": ["test"],
                "status": status,
            },
        )
        assert resp.status_code == 200
        if ticker == "IBM":
            archived = auth_client.delete(f"/api/v1/ideas/{resp.json()['idea']['id']}")
            assert archived.status_code == 200

    started = auth_client.post("/api/v1/ideas/evaluate-all/async", json={})
    assert started.status_code in {200, 202}
    job = _poll_until_done(auth_client, started.json()["job_id"])

    assert job["status"] == "done"
    run = job["result"]["run"]
    assert run["ranking_count"] == 2
    assert run["scope_statuses"] == ["watching", "researching", "ready_for_review"]
    assert [row["ticker"] for row in run["rankings"]] == ["AAPL", "MSFT"]
    assert {row["confidence_level"] for row in run["rankings"]} == {"high"}
    assert len(job["result"]["evaluations"]) == 2

    listed = auth_client.get("/api/v1/ideas/comparison-runs", params={"limit": 1})
    assert listed.status_code == 200
    listed_payload = listed.json()
    assert listed_payload["count"] == 1
    assert listed_payload["runs"][0]["run_id"] == run["run_id"]
    assert [row["ticker"] for row in listed_payload["runs"][0]["rankings"]] == ["AAPL", "MSFT"]


def test_create_update_defaults_analyzer_direction(auth_client):
    created = auth_client.post(
        "/api/v1/ideas",
        json={"ticker": "NFLX", "company_name": "Netflix", "user_notes": "Direction defaults.", "tags": []},
    )
    assert created.status_code == 200
    idea = created.json()["idea"]
    assert idea["metadata"]["analyzer_direction"] == "inactive"

    updated = auth_client.put(f"/api/v1/ideas/{idea['id']}", json={"analyzer_direction": "long"})
    assert updated.status_code == 200
    assert updated.json()["idea"]["metadata"]["analyzer_direction"] == "long"


def test_analyzer_context_overrides_canonical_scores_and_preserves_six_factor_average():
    from api.routers import ideas as ideas_router

    context = {
        "idea": {"id": "idea:1", "ticker": "AAPL"},
        "ticker": "AAPL",
        "tool_errors": [],
        "evaluated_at": "2026-05-05T12:00:00+00:00",
        "analyzer_context": {
            "status": "available",
            "row": {
                "industry_quality_score": 80,
                "business_quality_qual_score": 70,
                "management_quality_score": 90,
                "valuation_signal": 1.5,
                "fundamental_momentum_signal": 3,
                "price_mom_signal": -3,
            },
            "warnings": [],
            "qualitative_evidence": {},
        },
    }
    result = ideas_router._normalize_llm_result(
        context,
        {
            "action": "watch",
            "recommendation_status": "clear",
            "score": 10,
            "confidence": 0.8,
            "rationale": "Test result.",
            "factor_scores": {
                "macro_support": {"score": 60},
                "industry_attractiveness": {"score": 10},
                "business_quality": {"score": 10},
                "management_quality": {"score": 10},
                "valuation_asymmetry": {"score": 10},
                "portfolio_fit": {"score": 40},
                "fundamental_momentum": {"score": 100},
            },
        },
    )

    assert result["evaluation_schema_version"] == ideas_router.IDEA_EVALUATION_SCHEMA_VERSION
    assert result["factor_scores"]["industry_attractiveness"]["score"] == 80
    assert result["factor_scores"]["business_quality"]["score"] == 70
    assert result["factor_scores"]["management_quality"]["score"] == 90
    assert result["factor_scores"]["valuation_asymmetry"]["score"] == 75
    assert result["score"] == 69.2
    assert "fundamental_momentum" not in result["factor_scores"]
    assert "price_momentum" not in result["factor_scores"]


def test_analyzer_context_forwards_structured_short_squeeze_risk():
    from api.routers import ideas as ideas_router

    contexts = ideas_router._analyzer_contexts_from_result(
        {
            "status": "ok",
            "raw_result": {
                "timestamp": "2026-05-05T12:00:00+00:00",
                "weights_df": [
                    {
                        "ticker": "SQUEEZE",
                        "source_type": "idea",
                        "scenario_score": -1.2,
                        "short_cover_risk": 0.0,
                        "long_risk_penalty": 0.0,
                        "short_squeeze_risk": False,
                        "short_squeeze_metrics_available": True,
                        "drawdown_metrics_available": False,
                    }
                ],
                "course_of_action": {
                    "summary": {"as_of": "2026-05-05T12:01:00+00:00"},
                    "action_queue": [
                        {
                            "ticker": "SQUEEZE",
                            "source_type": "idea",
                            "action": "Watch",
                            "scenario_score": -0.4,
                            "score_delta": 0.6,
                            "short_cover_risk": 0.8,
                            "long_risk_penalty": 0.0,
                            "risk_flags": {
                                "short_squeeze_risk": True,
                                "risk_data_missing": False,
                            },
                            "risk_parts": {
                                "short_squeeze_cover_risk": 0.8,
                            },
                            "warnings": ["Short squeeze risk elevated"],
                        }
                    ],
                },
            },
        }
    )

    context = contexts["SQUEEZE"]
    assert context["action_label"] == "Watch"
    assert context["scenario_score"] == -0.4
    assert context["short_cover_risk"] == 0.8
    assert context["risk_flags"]["short_squeeze_risk"] is True
    assert context["risk_parts"]["short_squeeze_cover_risk"] == 0.8
    assert context["short_squeeze_metrics_available"] is True


def test_analyzer_context_forwards_missing_squeeze_metrics_without_false_risk():
    from api.routers import ideas as ideas_router

    contexts = ideas_router._analyzer_contexts_from_result(
        {
            "status": "ok",
            "raw_result": {
                "weights_df": [
                    {
                        "ticker": "MISSRISK",
                        "source_type": "idea",
                        "short_squeeze_risk": True,
                        "short_squeeze_data_missing": False,
                        "risk_data_missing": False,
                        "short_squeeze_metrics_available": False,
                    }
                ],
                "course_of_action": {
                    "action_queue": [
                        {
                            "ticker": "MISSRISK",
                            "action": "Watch",
                            "short_cover_risk": 0.0,
                            "risk_flags": {
                                "short_squeeze_risk": False,
                                "short_squeeze_data_missing": True,
                                "risk_data_missing": True,
                            },
                            "risk_parts": {},
                            "warnings": ["Short squeeze metrics unavailable", "Risk metrics unavailable"],
                        }
                    ],
                },
            },
        }
    )

    context = contexts["MISSRISK"]
    assert context["short_cover_risk"] == 0.0
    assert context["risk_flags"]["short_squeeze_risk"] is False
    assert context["risk_flags"]["short_squeeze_data_missing"] is True
    assert context["risk_flags"]["risk_data_missing"] is True
    assert context["short_squeeze_metrics_available"] is False


def test_analyzer_risk_is_display_only_for_idea_action():
    from api.routers import ideas as ideas_router

    context = {
        "idea": {"id": "idea:1", "ticker": "RISKY"},
        "ticker": "RISKY",
        "tool_errors": [],
        "evaluated_at": "2026-05-05T12:00:00+00:00",
        "analyzer_context": {
            "status": "available",
            "action_label": "Watch",
            "short_cover_risk": 0.8,
            "risk_flags": {"short_squeeze_risk": True, "risk_data_missing": False},
            "risk_parts": {"short_squeeze_cover_risk": 0.8},
            "row": {
                "industry_quality_score": 80,
                "business_quality_qual_score": 80,
                "management_quality_score": 80,
                "valuation_signal": 0,
            },
            "warnings": ["Short squeeze risk elevated"],
            "qualitative_evidence": {},
        },
    }

    result = ideas_router._normalize_llm_result(
        context,
        {
            "action": "buy",
            "recommendation_status": "clear",
            "score": 85,
            "confidence": 0.8,
            "rationale": "The thesis is attractive, but analyzer risk should be displayed separately.",
            "factor_scores": {
                "macro_support": {"score": 80},
                "industry_attractiveness": {"score": 80},
                "business_quality": {"score": 80},
                "management_quality": {"score": 80},
                "valuation_asymmetry": {"score": 80},
                "portfolio_fit": {"score": 80},
            },
            "missing_information": [],
        },
    )

    assert result["action"] == "buy"
    assert result["analyzer_context"]["risk_flags"]["short_squeeze_risk"] is True
    assert any(item["source"] == "analyzer_risk" for item in result["evidence"])


def test_evaluate_all_computes_one_analyzer_result_for_enabled_ideas(auth_client, monkeypatch):
    from api.routers import ideas as ideas_router

    calls = {"analyzer": 0}

    def fake_analyzer_result():
        calls["analyzer"] += 1
        return {"status": "ok", "raw_result": {"weights_df": []}}

    monkeypatch.setattr(ideas_router, "_compute_portfolio_plus_ideas_analyzer_result", fake_analyzer_result)
    monkeypatch.setattr(
        ideas_router,
        "_analyzer_contexts_from_result",
        lambda _result: {
            "MSFT": {
                "status": "available",
                "ticker": "MSFT",
                "row": {
                    "industry_quality_score": 80,
                    "business_quality_qual_score": 80,
                    "management_quality_score": 80,
                    "valuation_signal": 0,
                },
            },
            "AAPL": {
                "status": "available",
                "ticker": "AAPL",
                "row": {
                    "industry_quality_score": 70,
                    "business_quality_qual_score": 70,
                    "management_quality_score": 70,
                    "valuation_signal": 0,
                },
            },
        },
    )
    monkeypatch.setattr(
        ideas_router,
        "_call_llm_comparison_ranker",
        lambda evaluations: ideas_router._deterministic_comparison_result(evaluations),
    )

    for ticker, direction in [("MSFT", "long"), ("AAPL", "short")]:
        created = auth_client.post(
            "/api/v1/ideas",
            json={
                "ticker": ticker,
                "company_name": ticker,
                "user_notes": f"Review {ticker}.",
                "tags": ["test"],
                "status": "watching",
                "analyzer_direction": direction,
            },
        )
        assert created.status_code == 200

    started = auth_client.post("/api/v1/ideas/evaluate-all/async", json={})
    assert started.status_code in {200, 202}
    job = _poll_until_done(auth_client, started.json()["job_id"])

    assert job["status"] == "done"
    assert calls["analyzer"] == 1
    assert {row["analyzer_context"]["status"] for row in job["result"]["evaluations"]} == {"available"}


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


def test_comparison_fallback_caps_confidence_when_evidence_is_missing():
    from api.routers import ideas as ideas_router

    result = ideas_router._deterministic_comparison_result(
        [
            {
                "id": 11,
                "idea_id": 101,
                "ticker": "FULL",
                "action": "watch",
                "score": 70,
                "confidence": 0.8,
                "missing_information": [],
                "rationale": "Complete enough for ranking.",
            },
            {
                "id": 12,
                "idea_id": 102,
                "ticker": "MISS",
                "action": "watch",
                "score": 72,
                "confidence": 0.9,
                "missing_information": [{"field": "overview", "severity": "critical", "reason": "No overview."}],
                "rationale": "Missing critical evidence.",
            },
        ]
    )

    missing_row = next(row for row in result["rankings"] if row["ticker"] == "MISS")
    assert missing_row["confidence"] == 0.35
    assert missing_row["confidence_level"] == "low"
