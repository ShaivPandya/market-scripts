from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_quality.eval_runner import load_cases as load_structured_cases
from decision_quality.supervised_labels import (
    assign_split,
    check_split_leakage,
    labels_from_structured_dq_gold,
    normalize_missing_input_tags,
    normalize_synthesis_stance,
    normalize_triage_action,
    row_is_training_eligible,
    split_group_for_case,
)
from decision_quality.synthesis_supervised import (
    apply_supervised_triage_overlay,
    build_context_features,
    featurize_context_row,
)
from decision_quality.synthesis_supervised_training import (
    chat_case_to_row,
    check_rollout_gates,
    export_training_dataset,
    structured_case_to_row,
    train_baseline_classifier,
)


def test_normalize_triage_and_stance_labels():
    assert normalize_triage_action("watch") == "watch"
    assert normalize_triage_action("buy") == "graduate_to_decision_quality"
    assert normalize_synthesis_stance("watch_only") == "watch_only"
    assert normalize_synthesis_stance("watch") == "watch_only"
    assert normalize_missing_input_tags(["Current chart read", "premium valuation"]) == ["chart", "valuation"]


def test_labels_from_structured_dq_gold_watch_case():
    gold = {
        "recommended_action": "watch",
        "actionability": {
            "status": "watch_only",
            "missing_inputs": ["Exact forward valuation multiple"],
        },
        "price_action_read": {"data_needed": ["Relative performance versus XLP"]},
    }
    labels = labels_from_structured_dq_gold(gold)
    assert labels["label_next_action"] == "watch"
    assert labels["label_synthesis_stance"] == "watch_only"
    assert "valuation" in labels["label_missing_input_tags"]


def test_split_group_and_leakage_check():
    case_data = {"screen_context": {"ticker": "COST"}, "as_of_date": "2024-12-13"}
    group = split_group_for_case(case_id="cost_quality_asset_bad_entry_watch_2026", case_data=case_data)
    assert group == "COST:2024-12-13"
    split_a = assign_split(group)
    split_b = assign_split(group)
    assert split_a == split_b
    rows = [
        {"split_group": group, "split": "train", "label_next_action": "watch", "eval_status": "approved"},
        {"split_group": group, "split": "holdout", "label_next_action": "watch", "eval_status": "approved"},
    ]
    assert check_split_leakage(rows) == [group]


def test_structured_case_to_row_from_fixture():
    cases = load_structured_cases(case_selectors=["cost_quality_asset_bad_entry_watch_2026"], statuses={"approved"})
    assert cases
    row = structured_case_to_row(cases[0])
    assert row["source"] == "structured_dq_eval"
    assert row["label_next_action"] == "watch"
    assert row["split"] in {"train", "validation", "holdout"}
    assert row_is_training_eligible(row)


def test_export_and_train_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "decision_quality.synthesis_supervised_training.DEFAULT_OUTPUT_DIR",
        tmp_path / "outputs",
    )
    monkeypatch.setattr(
        "decision_quality.synthesis_supervised_training.DEFAULT_MODEL_DIR",
        tmp_path / "models",
    )
    monkeypatch.setattr(
        "decision_quality.synthesis_supervised_training.DEFAULT_REGISTRY_PATH",
        tmp_path / "models" / "registry.json",
    )

    manifest = export_training_dataset(
        output_dir=tmp_path / "outputs",
        statuses=frozenset({"review", "approved"}),
    )
    assert manifest["row_count"] >= 2
    assert manifest["leakage_check_passed"] is True

    result = train_baseline_classifier(output_dir=tmp_path / "models")
    assert Path(result["artifact_path"]).exists()
    registry = json.loads((tmp_path / "models" / "registry.json").read_text(encoding="utf-8"))
    assert registry["active_model_path"] == result["artifact_path"]
    assert "triage_accuracy" in registry["metrics"]


def test_rollout_gate_passes_with_good_metrics():
    report = check_rollout_gates(
        metrics={"triage_accuracy": 0.8, "stance_accuracy": 0.7, "graduate_recall": 0.75},
        baseline_metrics={"graduate_recall": 0.7},
    )
    assert report["passed"] is True


def test_apply_supervised_overlay_shadow_mode(monkeypatch):
    from decision_quality.opportunity_candidate import OpportunityCandidate

    monkeypatch.setenv("AGENT_SYNTHESIS_SUPERVISED_ENABLED", "false")
    monkeypatch.setenv("AGENT_SYNTHESIS_SUPERVISED_SHADOW_MODE", "true")
    candidate = OpportunityCandidate(
        trigger="User asked about NVDA",
        consensus="AI leader",
        variant_view="Pullback entry",
        why_now="Recent drawdown",
        price_confirmation="Needs chart check",
        next_action="graduate_to_decision_quality",
    )
    context_bundle = {
        "user_message": "Should I buy NVDA here?",
        "screen_context": {"ticker": "NVDA"},
        "context_pack": {"pack_id": "quality_entry", "is_complete": False, "missing_inputs": ["valuation"]},
        "data_quality": {"critical_data_quality": "ok"},
    }
    updated, meta = apply_supervised_triage_overlay(
        opportunity_candidate=candidate,
        context_bundle=context_bundle,
        user_text="Should I buy NVDA here?",
        model_path=None,
    )
    assert updated.next_action == "graduate_to_decision_quality"
    assert meta["skipped"] is True


def test_featurize_context_row_includes_pack_metadata():
    row = {
        "user_text": "Costco looks great, should I buy?",
        "screen_context": {"ticker": "COST"},
        "context_features": build_context_features(
            {
                "context_pack": {
                    "pack_id": "quality_entry",
                    "is_complete": False,
                    "missing_inputs": ["valuation reset"],
                    "opportunity_types": ["quality_compounder"],
                },
                "data_quality": {"critical_data_quality": "ok"},
            }
        ),
    }
    text = featurize_context_row(row)
    assert "Costco" in text
    assert "quality_entry" in text
    assert "quality_compounder" in text


def test_chat_case_to_row_when_preflight_expected():
    from decision_quality.chat_eval_runner import load_cases as load_chat_cases

    cases = load_chat_cases(case_selectors=["asset_trade_cost_quality_bad_entry_2026_chat"], statuses={"approved"})
    assert cases
    row = chat_case_to_row(cases[0])
    assert row["label_next_action"] == "watch"
    assert row["source"] == "chat_eval"
