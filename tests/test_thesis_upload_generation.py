from __future__ import annotations

from unittest.mock import MagicMock

from api.routers import thesis as thesis_router

SAMPLE_THESIS = """# MU

## Thesis
- Memory cycle recovery supports earnings.

## Key Catalysts
- TBD

## Risk Factors
- **Margin compression:** Pricing weakens.
"""


def test_generate_thesis_from_upload_stages_kill_conditions_not_risk_projection(monkeypatch):
    calls: list[dict] = []

    def fake_stage(action_id, payload, **kwargs):
        calls.append({"action_id": action_id, "payload": dict(payload), "apply": kwargs.get("apply")})
        return {
            "status": "applied" if kwargs.get("apply") else "pending_approval_created",
            "approval_id": f"approval:{action_id}:{len(calls)}",
            "application_status": "applied" if kwargs.get("apply") else "pending",
            "action_id": action_id,
            "entity_type": "thesis",
            "ticker": payload.get("ticker"),
            "proposed_change": payload,
        }

    monkeypatch.setattr(thesis_router, "stage_api_action", fake_stage)
    monkeypatch.setattr(
        "portfolio.thesis_upload_extraction.extract_entities_from_thesis_upload",
        lambda _ticker, _content: {
            "catalysts": [{"label": "HBM ramp", "description": "Supply tightens."}],
            "kill_conditions": [
                {
                    "condition": "Revenue growth falls below 5% YoY",
                    "metric": "revenue_growth",
                    "threshold": "< 5%",
                    "rationale": "Overview sensitivity",
                },
                {
                    "condition": "Revenue growth falls below 5% YoY",
                    "metric": "revenue_growth",
                    "threshold": "< 5%",
                    "rationale": "duplicate",
                },
            ],
        },
    )
    monkeypatch.setattr("portfolio.thesis_upload_extraction.existing_kill_condition_keys", lambda _ticker: set())

    result = thesis_router.generate_thesis_from_upload_bytes(
        "mu",
        SAMPLE_THESIS.encode("utf-8"),
        content_type="text/markdown",
        filename="mu.md",
    )

    save_calls = [c for c in calls if c["action_id"] == "save_thesis_content"]
    assert len(save_calls) == 1
    save_payload = save_calls[0]["payload"]
    assert save_payload["project_risk_factors_to_kill_conditions"] is False
    assert "**HBM ramp:**" in save_payload["content"]
    assert save_calls[0]["apply"] is True

    kill_calls = [c for c in calls if c["action_id"] == "create_kill_condition"]
    assert len(kill_calls) == 1
    assert kill_calls[0]["apply"] is False
    assert kill_calls[0]["payload"]["metric"] == "revenue_growth"

    assert result["extraction_summary"]["kill_condition_proposal_count"] == 1
    assert result["extraction_summary"]["skipped_duplicate_count"] == 1
    assert len(result["staged_proposals"]) == 1


def test_generate_thesis_from_upload_skips_existing_kill_conditions(monkeypatch):
    calls: list[dict] = []

    def fake_stage(action_id, payload, **kwargs):
        calls.append({"action_id": action_id, "payload": dict(payload), "apply": kwargs.get("apply")})
        return {
            "status": "applied",
            "approval_id": "approval:save",
            "application_status": "applied",
            "action_id": action_id,
            "entity_type": "thesis",
            "ticker": payload.get("ticker"),
            "proposed_change": payload,
        }

    monkeypatch.setattr(thesis_router, "stage_api_action", fake_stage)
    monkeypatch.setattr(
        "portfolio.thesis_upload_extraction.extract_entities_from_thesis_upload",
        lambda _ticker, _content: {
            "catalysts": [],
            "kill_conditions": [{"condition": "Margin compression accelerates", "metric": None, "threshold": None}],
        },
    )
    from portfolio.thesis_sync import _normalize_match_text

    monkeypatch.setattr(
        "portfolio.thesis_upload_extraction.existing_kill_condition_keys",
        lambda _ticker: {_normalize_match_text("Margin compression accelerates")},
    )

    result = thesis_router.generate_thesis_from_upload_bytes(
        "mu",
        SAMPLE_THESIS.encode("utf-8"),
        content_type="text/markdown",
        filename="mu.md",
    )

    assert [c for c in calls if c["action_id"] == "create_kill_condition"] == []
    assert result["extraction_summary"]["kill_condition_proposal_count"] == 0
    assert result["extraction_summary"]["skipped_duplicate_count"] == 1


def test_extract_entities_fallback_from_markdown(monkeypatch):
    from portfolio import thesis_upload_extraction as extraction

    monkeypatch.setattr(extraction, "call_llm_text", MagicMock(side_effect=RuntimeError("no llm")))

    result = extraction.extract_entities_from_thesis_upload("MU", SAMPLE_THESIS)

    assert any(c["label"] == "Margin compression" for c in result["catalysts"]) is False
    assert len(result["kill_conditions"]) == 1
    assert "Margin compression" in result["kill_conditions"][0]["condition"]


def test_merge_catalysts_only_when_section_empty():
    from portfolio.thesis_upload_extraction import catalyst_section_needs_fill, merge_catalyst_bullets_into_thesis

    assert catalyst_section_needs_fill(SAMPLE_THESIS) is True
    merged = merge_catalyst_bullets_into_thesis(
        SAMPLE_THESIS,
        [{"label": "HBM ramp", "description": "Tight supply."}],
    )
    assert "**HBM ramp:**" in merged

    populated = SAMPLE_THESIS.replace("- TBD", "- **Existing:** Already set.")
    assert catalyst_section_needs_fill(populated) is False
    unchanged = merge_catalyst_bullets_into_thesis(
        populated,
        [{"label": "Ignored", "description": "Should not apply."}],
    )
    assert unchanged == populated


def test_read_dossier_context_markdown(monkeypatch, tmp_path):
    from portfolio import overview_content
    from portfolio import thesis_upload_extraction as extraction

    overview_dir = tmp_path / "overviews"
    overview_dir.mkdir()
    monkeypatch.setattr(overview_content, "OVERVIEWS_DIR", overview_dir)
    monkeypatch.setattr(overview_content, "OVERVIEWS_GCS_PREFIX", "test/overviews")
    (overview_dir / "MU.md").write_text("# MU Overview\n\n## Financials\n- growth", encoding="utf-8")

    overview, management = extraction.read_dossier_context_markdown("MU")
    assert overview is not None
    assert "Financials" in overview
    assert management is None
