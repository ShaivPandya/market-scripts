from __future__ import annotations

from api.tool_data_quality import aggregate_tool_data_quality, normalize_tool_quality


def test_normalize_tool_quality_marks_blocked_critical_chart_as_blocking():
    envelope = normalize_tool_quality(
        {
            "name": "run_chart",
            "status": "blocked",
            "result": {
                "error": "Chart access blocked by policy.",
                "_meta": {"status": "blocked", "reliability_tier": "critical"},
            },
        }
    )

    assert envelope["blocks_actionable"] is True
    assert envelope["price_confirmation"] == "blocked"
    assert envelope["gate_action"] == "block"


def test_normalize_tool_quality_marks_missing_price_confirmation():
    envelope = normalize_tool_quality(
        {
            "name": "run_chart",
            "status": "ok",
            "result": {
                "ticker": "META",
                "technical_read": "",
                "data_needed": ["current META chart"],
            },
        }
    )

    assert envelope["price_confirmation"] == "missing"
    assert envelope["blocks_actionable"] is True
    assert envelope["missing_fields"] == ["current META chart"]


def test_aggregate_tool_data_quality_sets_gate_payload_for_stale_critical_sources():
    aggregate = aggregate_tool_data_quality(
        [
            {
                "name": "run_chart",
                "status": "ok",
                "result": {
                    "ticker": "NVDA",
                    "technical_read": "Stale chart read.",
                    "_meta": {"source_status": "stale", "stale": True, "reliability_tier": "critical"},
                },
            },
            {
                "name": "get_dossier",
                "status": "ok",
                "result": {"ticker": "NVDA", "summary": "ok dossier"},
            },
        ]
    )

    assert aggregate["critical_data_quality"] == "stale"
    assert aggregate["blocker_count"] >= 1
    assert "CRITICAL_DATA_QUALITY" in aggregate["blocking_reason_codes"]
    assert aggregate["price_confirmation_status"] == "stale"


def test_aggregate_tool_data_quality_keeps_standard_blocked_tool_as_warning():
    aggregate = aggregate_tool_data_quality(
        [
            {
                "name": "get_dossier",
                "status": "blocked",
                "result": {
                    "error": "Permission denied.",
                    "_meta": {"status": "blocked", "reliability_tier": "standard"},
                },
            },
            {
                "name": "run_chart",
                "status": "ok",
                "result": {"ticker": "NVDA", "technical_read": "Neutral trend."},
            },
        ]
    )

    assert aggregate["blocker_count"] == 0
    assert aggregate["warning_count"] >= 1
    assert aggregate["source_health_status"] == "warning"
    assert aggregate["critical_data_quality"] == "ok"
