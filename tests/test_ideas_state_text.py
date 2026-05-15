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
