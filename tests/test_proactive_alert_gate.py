from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_quality.proactive_alert_gate import (
    apply_proactive_alert_gate,
    apply_recommendation_scout_skeptic_sizer_gate,
    build_chat_scout_skeptic_sizer_gate,
    evaluate_proactive_alert_gate,
    is_high_stakes_action_item,
    proactive_alert_gate_enabled,
    proactive_alert_llm_passes_enabled,
    should_apply_proactive_alert_gate,
    should_apply_recommendation_gate,
)

OC_CASES = Path("docs/opportunity_candidate_evals/cases")
DQ_CASES = Path("docs/decision_quality_evals/cases")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _base_resize_payload(**overrides) -> dict:
    payload = {
        "ticker": "MU",
        "action_type": "resize",
        "description": "Continuous optimizer: Trim MU after material change.",
        "urgency": "high",
    }
    payload.update(overrides)
    return payload


def test_gate_feature_flags():
    assert proactive_alert_gate_enabled() is True
    assert should_apply_proactive_alert_gate("create_action_item", "workflow") is True
    assert should_apply_proactive_alert_gate("create_action_item", "user") is False
    assert should_apply_proactive_alert_gate("create_recommendation", "workflow") is False
    assert is_high_stakes_action_item({"action_type": "resize"}) is True
    assert is_high_stakes_action_item({"action_type": "review"}) is False


def test_high_stakes_alert_without_dq_downgrades_to_research(monkeypatch):
    monkeypatch.setenv("PROACTIVE_ALERT_DQ_GATE_ENABLED", "true")
    payload = _base_resize_payload()
    updated, gate = apply_proactive_alert_gate(
        "create_action_item",
        payload,
        source_type="workflow",
        alert_context={"change_summary": "Trim MU after material change.", "source": "workflow"},
    )

    assert gate.applied is True
    assert gate.action_allowed is False
    assert gate.scout.status == "pass"
    assert gate.skeptic.status == "fail"
    assert gate.sizer.status == "fail"
    assert updated["action_type"] == "research"
    assert "scout_skeptic_sizer_gate" in updated
    assert updated["scout_skeptic_sizer_gate"]["gate_status"] in {"downgraded", "blocked"}


def test_high_stakes_alert_with_full_artifacts_can_pass(monkeypatch):
    monkeypatch.setenv("PROACTIVE_ALERT_DQ_GATE_ENABLED", "true")
    oc_case = _load_json(OC_CASES / "opportunity_candidate_graduate_nvda_2026.json")
    dq_case = _load_json(DQ_CASES / "mu_ai_memory_cycle_2025.json")
    payload = _base_resize_payload(
        ticker="NVDA",
        opportunity_candidate=oc_case["gold_output"],
        decision_quality=dq_case["gold_output"],
    )
    gate = evaluate_proactive_alert_gate(
        payload,
        alert_context={"change_summary": "Trim NVDA after optimizer signal.", "source": "workflow"},
    )

    assert gate.scout.status == "pass"
    assert gate.skeptic.status == "pass"
    assert gate.sizer.status == "pass"
    assert gate.action_allowed is True


def test_sizer_limits_watch_dq_to_monitor_only(monkeypatch):
    monkeypatch.setenv("PROACTIVE_ALERT_DQ_GATE_ENABLED", "true")
    oc_case = _load_json(OC_CASES / "opportunity_candidate_graduate_nvda_2026.json")
    dq_case = _load_json(DQ_CASES / "cost_quality_asset_bad_entry_watch_2026.json")
    payload = _base_resize_payload(
        ticker="COST",
        opportunity_candidate=oc_case["gold_output"],
        decision_quality=dq_case["gold_output"],
    )
    updated, gate = apply_proactive_alert_gate(
        "create_action_item",
        payload,
        source_type="workflow",
        alert_context={"change_summary": "Costco quality name flagged by optimizer.", "source": "workflow"},
    )

    assert gate.scout.status == "pass"
    assert gate.skeptic.status == "pass"
    assert gate.sizer.status == "pass"
    assert gate.sizer.final_action == "watch"
    assert gate.action_allowed is False
    assert updated["action_type"] == "research"


def test_gate_disabled_leaves_payload_unchanged(monkeypatch):
    monkeypatch.setenv("PROACTIVE_ALERT_DQ_GATE_ENABLED", "false")
    payload = _base_resize_payload()
    updated, gate = apply_proactive_alert_gate(
        "create_action_item",
        payload,
        source_type="workflow",
        alert_context={"change_summary": "Trim MU"},
    )

    assert gate.enabled is False
    assert updated["action_type"] == "resize"
    assert "scout_skeptic_sizer_gate" not in updated


@pytest.mark.parametrize(
    ("action_type", "expected_applied"),
    [
        ("review", False),
        ("research", False),
        ("resize", True),
        ("exit", True),
    ],
)
def test_gate_applies_only_to_high_stakes_action_items(action_type, expected_applied):
    payload = {"action_type": action_type, "description": "Alert", "ticker": "MU"}
    _, gate = apply_proactive_alert_gate("create_action_item", payload, source_type="workflow")
    assert gate.applied is expected_applied


def test_recommendation_gate_downgrades_sparse_actionable_buy(monkeypatch):
    monkeypatch.setenv("SCOUT_SKEPTIC_SIZER_GATE_ENABLED", "true")
    record = {
        "action": "buy",
        "ticker": "MU",
        "rationale": "Buy MU ahead of catalyst.",
        "critical_data_quality": "ok",
        "source_quality": "ok",
    }
    gate = apply_recommendation_scout_skeptic_sizer_gate(record)

    assert gate is not None
    assert gate.action_allowed is False
    assert record["action"] == "watch"
    assert record["scout_skeptic_sizer_gate"]["skeptic"]["status"] == "fail"


def test_chat_gate_emits_skeptic_block_trace():
    oc_case = _load_json(OC_CASES / "opportunity_candidate_scout_pass_skeptic_block_nvda_2026.json")
    candidate, _errors = __import__(
        "decision_quality.opportunity_candidate", fromlist=["parse_opportunity_candidate"]
    ).parse_opportunity_candidate(oc_case["gold_output"])
    oc_gate = __import__(
        "decision_quality.candidate_gates", fromlist=["apply_opportunity_candidate_gates"]
    ).apply_opportunity_candidate_gates(candidate)
    trace = build_chat_scout_skeptic_sizer_gate(
        oc_result={
            "opportunity_candidate": candidate,
            "gate": oc_gate,
            "parse_errors": [],
        },
        dq_result=None,
        context_bundle={"data_quality": {"overall_status": "ok"}},
        should_run_full_dq=False,
    )

    assert trace["scout"]["status"] == "pass"
    assert trace["skeptic"]["status"] == "fail"
    assert trace["final_action_type"] == "research"


def test_chat_gate_emits_sizer_watch_limit_trace():
    oc_case = _load_json(OC_CASES / "opportunity_candidate_graduate_nvda_2026.json")
    dq_case = _load_json(DQ_CASES / "cost_quality_asset_bad_entry_watch_2026.json")
    candidate, _errors = __import__(
        "decision_quality.opportunity_candidate", fromlist=["parse_opportunity_candidate"]
    ).parse_opportunity_candidate(oc_case["gold_output"])
    decision_quality, _dq_errors = __import__(
        "decision_quality.models", fromlist=["parse_decision_quality"]
    ).parse_decision_quality(dq_case["gold_output"])
    oc_gate = __import__(
        "decision_quality.candidate_gates", fromlist=["apply_opportunity_candidate_gates"]
    ).apply_opportunity_candidate_gates(candidate)
    dq_gate = __import__(
        "decision_quality.gates", fromlist=["apply_decision_quality_gates"]
    ).apply_decision_quality_gates(
        decision_quality,
        current_action="buy",
        recommendation_status="clear",
    )
    trace = build_chat_scout_skeptic_sizer_gate(
        oc_result={
            "opportunity_candidate": candidate,
            "gate": oc_gate,
            "parse_errors": [],
        },
        dq_result={
            "decision_quality": decision_quality,
            "gate": dq_gate,
            "parse_errors": [],
        },
        context_bundle={"data_quality": {"overall_status": "ok"}},
        should_run_full_dq=True,
    )

    assert trace["scout"]["status"] == "pass"
    assert trace["skeptic"]["status"] == "pass"
    assert trace["sizer"]["status"] == "pass"
    assert trace["final_action_type"] == "watch"
    assert (trace["sizer"].get("max_sizing_delta_bps") or 0) <= 50


def test_llm_proactive_passes_default_disabled():
    assert proactive_alert_llm_passes_enabled() is False


def test_recommendation_gate_flag():
    assert should_apply_recommendation_gate("create_recommendation") is True
    assert should_apply_recommendation_gate("create_action_item") is False
