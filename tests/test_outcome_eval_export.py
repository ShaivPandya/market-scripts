from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_quality.eval_corpus import infer_process_attribution_tags
from decision_quality.eval_runner import EvalCase, build_solver_payload, run_case
from decision_quality.outcome_eval_export import (
    build_as_of_input_snapshot,
    build_case_from_outcome,
    build_outcome_context,
    build_outcome_linkage,
)


def _sample_outcome_row(*, final_label_status: str = "confirmed") -> dict:
    return {
        "decision_outcome_id": "rec:test-nvda-1",
        "source_kind": "recommendation",
        "recommendation_id": "test-nvda-1",
        "ticker": "NVDA",
        "as_of": "2026-04-10",
        "horizon": "1 month",
        "outcome_status": "evaluated",
        "final_label_status": final_label_status,
        "process_label": "good_process_bad_outcome",
        "final_postmortem": "Process was sound but catalyst failed before the review horizon ended.",
        "lessons_learned": "Catalyst failed before the review horizon ended; timing was late.",
        "metrics": {
            "forward_return_pct": -8.4,
            "benchmark_relative_return_pct": -9.6,
            "directionally_right": False,
            "timing_vs_expected_onset": "late",
            "process_label": "good_process_bad_outcome",
        },
        "decision_quality_snapshot": {
            "simple_thesis": "Buy NVDA as the dominant AI accelerator platform.",
            "opportunity_type": "quality_compounder",
            "recommended_action": "buy",
            "confidence": 0.74,
            "actionability": {"status": "actionable", "reason": "Clear thesis and catalyst.", "missing_inputs": []},
            "mispricing": {
                "consensus_view": "Consensus",
                "variant_view": "Variant",
                "pricing_evidence": "Evidence",
                "why_consensus_is_wrong": "Why wrong",
            },
            "catalyst_or_reason_now": {
                "event_or_condition": "Earnings",
                "expected_timeframe": "1 month",
                "why_now": "Now",
                "source_evidence": [],
            },
            "invalidation": {
                "observable": "Breakdown",
                "metric_or_event": "Price",
                "threshold": "Below MA cluster",
                "timeframe": "2 weeks",
                "implication": "Thesis weaker",
            },
            "evidence_for": [],
            "evidence_against": [],
            "price_action_read": {
                "observed_behavior": "Above MAs",
                "interpretation": "Supportive",
                "confirms_thesis": True,
                "data_needed": [],
            },
            "expression": {
                "primary": "Buy NVDA",
                "instrument_type": "single_name_equity",
                "directness": "direct",
                "alternatives": [],
                "follow_on": "",
            },
            "conviction": {
                "level": 4,
                "max_level": 5,
                "raw_target_weight": 0.16,
                "upgrade_condition": "",
            },
            "confidence_reason": "Moderately high",
            "sizing_context": {
                "starting_size": "16%",
                "add_conditions": "",
                "liquidity_constraints": "",
                "portfolio_constraints": "",
                "sizing_delta": {
                    "direction": "increase",
                    "amount": 0.16,
                    "unit": "portfolio_weight",
                    "basis": "target_weight",
                    "condition": "",
                },
            },
            "trade_after_trade": {
                "if_right": "Press",
                "if_wrong": "Reduce",
                "next_review_trigger": "Earnings",
            },
            "embedded_macro_exposure": "Long AI capex",
        },
    }


def _sample_parent_row() -> dict:
    return {
        "recommendation_id": "test-nvda-1",
        "action": "buy",
        "confidence": 0.74,
        "source_quality": "ok",
        "approval_status": "approved",
        "decision_type": "new_idea",
    }


def test_build_as_of_input_snapshot_omits_realized_outcome_fields():
    snapshot = build_as_of_input_snapshot(_sample_outcome_row(), _sample_parent_row())
    serialized = json.dumps(snapshot)

    assert snapshot["ticker"] == "NVDA"
    assert snapshot["decision_context"]["recommended_action"] == "buy"
    assert "forward_return_pct" not in serialized
    assert "final_postmortem" not in serialized
    assert "lessons_learned" not in serialized


def test_build_outcome_linkage_includes_calibration_dimensions():
    linkage = build_outcome_linkage(
        _sample_outcome_row(),
        _sample_parent_row(),
        lesson_tags=["timing_wrong", "catalyst_failed"],
    )

    assert linkage["process_label"] == "good_process_bad_outcome"
    assert linkage["confidence_bin"] == "medium"
    assert linkage["actionability_stance"] == "actionable"
    assert linkage["data_quality_tier"] == "adequate"
    assert "timing_wrong" in linkage["process_attribution_tags"]
    assert "catalyst_failed" in linkage["process_attribution_tags"]


def test_infer_process_attribution_tags_from_lessons_and_metrics():
    tags = infer_process_attribution_tags(
        _sample_outcome_row(),
        _sample_parent_row(),
        lesson_tags=["catalyst_failed"],
    )

    assert "process_good" in tags
    assert "outcome_bad" in tags
    assert "timing_wrong" in tags
    assert "catalyst_failed" in tags


def test_build_case_from_outcome_requires_finalized_status():
    with pytest.raises(ValueError, match="final_label_status"):
        build_case_from_outcome(_sample_outcome_row(final_label_status="draft"), _sample_parent_row())


def test_build_case_from_outcome_includes_outcome_metadata():
    case = build_case_from_outcome(
        _sample_outcome_row(),
        _sample_parent_row(),
        input_snapshot_path="docs/decision_quality_evals/inputs/example.json",
        input_snapshot_sha256="abc123",
        lesson_tags=["timing_wrong"],
        status="review",
    )

    assert case["status"] == "review"
    assert "outcome_calibration" in case["corpus_tags"]
    assert case["outcome_linkage"]["decision_outcome_id"] == "rec:test-nvda-1"
    assert case["outcome_context"]["metrics"]["forward_return_pct"] == -8.4
    assert case["gold_output"]["recommended_action"] == "buy"


def test_build_outcome_context_marks_future_fields_as_grading_only():
    context = build_outcome_context(_sample_outcome_row(), _sample_parent_row())

    assert context["available_as_of_date"] is False
    assert context["process_label"] == "good_process_bad_outcome"
    assert "grading and calibration only" in context["notes"]


def test_fixture_case_solver_payload_excludes_outcome_authoring_fields(monkeypatch):
    def fail_call(*_args, **_kwargs):
        raise AssertionError("dry-run should not call the LLM")

    monkeypatch.setattr("decision_quality.eval_runner.call_llm_text", fail_call)
    path = Path("docs/decision_quality_evals/cases/nvda_outcome_calibration_review_2026.json")
    case = EvalCase(path=path, data=json.loads(path.read_text(encoding="utf-8")))
    payload = build_solver_payload(case)
    result = run_case(case, dry_run=True, judge=False)
    payload_blob = json.dumps(payload, sort_keys=True)
    prompt_blob = str(result.get("solver_prompt") or "")

    assert "outcome_linkage" not in payload_blob
    assert "outcome_context" not in payload_blob
    assert "forward_return_pct" not in payload_blob
    assert "good_process_bad_outcome" not in payload_blob
    assert "catalyst failed" not in payload_blob.lower()
    assert "forward_return_pct" not in prompt_blob
    assert "lessons_learned" not in prompt_blob


def test_load_outcome_row_raises_for_missing_outcome(monkeypatch):
    from decision_quality import outcome_eval_export

    class FakeReads:
        def get(self, _uid):
            return None

    monkeypatch.setattr(
        "ontology.runtime_read_service.OntologyRuntimeReadService",
        lambda: FakeReads(),
    )

    with pytest.raises(ValueError, match="DecisionOutcome not found"):
        outcome_eval_export.load_outcome_row("rec:missing")
