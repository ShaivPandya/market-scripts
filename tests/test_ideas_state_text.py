from __future__ import annotations


def test_read_state_text_success_does_not_return_error(monkeypatch, tmp_path):
    import paths
    from api import state_storage
    from api.routers import ideas as ideas_router

    seen = {}

    def fake_exists(local_path, gcs_key):
        seen["exists"] = (local_path, gcs_key)
        return True

    def fake_read(local_path, gcs_key, *, encoding="utf-8"):
        seen["read"] = (local_path, gcs_key, encoding)
        return "# MU Management Quality\n"

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(state_storage, "exists_text", fake_exists)
    monkeypatch.setattr(state_storage, "read_text", fake_read)

    content, error = ideas_router._read_state_text("investment_management_quality", "MU")

    assert content == "# MU Management Quality\n"
    assert error is None
    assert seen["exists"] == (tmp_path / "investment_management_quality" / "MU.md", "live/management_quality/MU.md")
    assert seen["read"] == (
        tmp_path / "investment_management_quality" / "MU.md",
        "live/management_quality/MU.md",
        "utf-8",
    )


def test_normalize_factor_scores_gives_numeric_rows_specific_rationale():
    from api.routers import ideas as ideas_router

    rows = ideas_router._normalize_factor_scores({"valuation": 35})

    assert rows["valuation_asymmetry"]["score"] == 35
    assert rows["valuation_asymmetry"]["rationale"] != "Evaluator returned a numeric factor score."
    assert "valuation asymmetry" in rows["valuation_asymmetry"]["rationale"]
    assert rows["valuation_asymmetry"]["source"] == "evaluator"


def test_normalize_missing_rows_replaces_duplicate_reason():
    from api.routers import ideas as ideas_router

    rows = ideas_router._normalize_missing_rows(
        [
            {
                "field": "Current ANET share price, market cap, and forward P/E vs. 5-year range",
                "severity": "medium",
                "reason": "Current ANET share price, market cap, and forward P/E vs. 5-year range",
            }
        ]
    )

    assert rows[0]["reason"] != rows[0]["field"]
    assert "valuation asymmetry" in rows[0]["reason"]


def _idea_evaluator_context():
    return {
        "idea": {
            "id": "investment_idea:ACMR",
            "ticker": "ACMR",
            "asset": "equity",
            "instrument_type": "security",
            "price_symbol": "ACMR",
            "contract_multiplier": 1.0,
            "user_notes": "Fast growing, low PEG ratio",
        },
        "ticker": "ACMR",
        "instrument": {
            "ticker": "ACMR",
            "asset": "equity",
            "instrument_type": "security",
            "price_symbol": "ACMR",
            "contract_multiplier": 1.0,
            "currency": "USD",
        },
        "asset": "equity",
        "instrument_type": "security",
        "overview_content": "Business overview.",
        "thesis_content": "Thesis.",
        "management_quality_content": "Management quality.",
        "tool_errors": [],
        "analyzer_context": {"status": "inactive", "ticker": "ACMR"},
        "use_portfolio_context": True,
        "evaluated_at": "2026-05-16T10:44:00+00:00",
    }


def _idea_evaluator_payload(ideas_router):
    factor = {
        "score": 55,
        "status": "reviewable",
        "rationale": "Stored evidence supports review.",
        "source": "test",
        "missing": [],
    }
    return {
        "thesis_statement": "Model thesis",
        "action": "watch",
        "recommendation_status": "review_required",
        "score": 55,
        "confidence": 0.5,
        "rationale": "Model rationale",
        "factor_scores": {name: dict(factor) for name in ideas_router.CANONICAL_IDEA_FACTORS},
        "missing_information": [],
        "data_quality": {"critical_data_quality": "ok", "source_quality": "ok", "quality": "ok"},
        "evidence": [],
        "disconfirming_evidence": [],
        "catalyst": "Review next filing.",
        "invalidation": "Reject if growth slows.",
        "portfolio_fit": {"status": "needs_review", "note": "No position change staged."},
        "decision_quality": None,
    }


def test_call_llm_evaluator_uses_json_helper_with_web_search(monkeypatch):
    import llm_utils
    from api.routers import ideas as ideas_router

    seen = {}

    def fake_call_llm_json(**kwargs):
        seen.update(kwargs)
        return (
            _idea_evaluator_payload(ideas_router),
            [],
            None,
            {
                "status": "ok",
                "provider": "anthropic",
                "model": "claude-opus-4-7",
                "attempts": 1,
                "web_search_status": "enabled",
            },
        )

    monkeypatch.setattr(llm_utils, "has_llm_api_key", lambda: True)
    monkeypatch.setattr(llm_utils, "call_llm_json", fake_call_llm_json)

    result = ideas_router._call_llm_evaluator(_idea_evaluator_context())

    assert seen["enable_web_search"] is True
    assert seen["max_web_search_uses"] == 4
    assert result["thesis_statement"] == "Model thesis"
    assert result["rationale"] == "Model rationale"
    assert result["data_quality"]["evaluator"]["status"] == "ok"


def test_call_llm_evaluator_fallback_includes_evaluator_diagnostics(monkeypatch):
    import llm_utils
    from api.routers import ideas as ideas_router

    def fake_call_llm_json(**kwargs):
        return (
            None,
            [],
            None,
            {
                "status": "fallback",
                "provider": "anthropic",
                "model": "claude-opus-4-7",
                "attempts": 2,
                "web_search_status": "enabled",
                "failure_reason": "model response was not valid JSON",
            },
        )

    monkeypatch.setattr(llm_utils, "has_llm_api_key", lambda: True)
    monkeypatch.setattr(llm_utils, "call_llm_json", fake_call_llm_json)

    result = ideas_router._call_llm_evaluator(_idea_evaluator_context())

    assert "Evaluator fallback reason: model response was not valid JSON" in result["rationale"]
    assert result["data_quality"]["evaluator"]["status"] == "fallback"
    assert result["data_quality"]["evaluator"]["failure_reason"] == "model response was not valid JSON"
