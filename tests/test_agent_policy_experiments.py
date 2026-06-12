from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from decision_quality.agent_policy_experiments import (
    ActionCandidate,
    ExperimentManifest,
    LoggedDecisionExample,
    PropensityMetadata,
    RewardComponent,
    build_experiment_report,
    logged_example_from_intent_router_row,
    main,
)


def _route(
    *,
    source: str,
    intent_class: str = "thesis_review",
    confidence: float = 0.8,
    tools: list[str] | None = None,
    **extra,
) -> dict:
    return {
        "source": source,
        "intent_class": intent_class,
        "run_hidden_dq": True,
        "run_opportunity_preflight": False,
        "workflow_name": None,
        "workflow_ticker": None,
        "tool_names": tools or ["get_thesis"],
        "confidence": confidence,
        **extra,
    }


def _reward(value: float = 1.0) -> RewardComponent:
    return RewardComponent(
        component_id="routing_reward",
        category="routing",
        source="process_reward",
        value=value,
        weight=1.0,
    )


def _example(
    *,
    candidate: ActionCandidate | None = None,
    logged: ActionCandidate | None = None,
    propensity: PropensityMetadata | None = None,
    reward_components: list[RewardComponent] | None = None,
) -> LoggedDecisionExample:
    logged = logged or ActionCandidate(
        action_id="regex:thesis",
        source="regex",
        intent_class="thesis_review",
        tool_names=["get_thesis"],
        confidence=0.9,
    )
    candidate = candidate or logged
    return LoggedDecisionExample(
        example_id="row-1",
        context={"user_text": "Review NVDA"},
        action_candidates=[logged, candidate],
        logged_action_id=logged.action_id,
        propensity=propensity,
        reward_components=reward_components or [_reward()],
        split_group="sess-1",
    )


def test_reward_component_rejects_future_leakage():
    with pytest.raises(ValidationError):
        RewardComponent(
            component_id="bad",
            category="routing",
            source="outcome_label",
            value=1.0,
            metadata={"forward_return_pct": 12.5},
        )


def test_intent_router_row_converts_to_logged_policy_example():
    row = {
        "row_id": "router-row-1",
        "session_id": "sess-1",
        "client_turn_id": "turn-1",
        "user_text": "Review NVDA",
        "regex_baseline": _route(source="regex", confidence=0.7),
        "llm_candidate": _route(source="llm", intent_class="general_research", confidence=0.4, tools=[]),
        "applied_route": _route(source="regex", confidence=0.7),
        "applied_source": "regex_shadow",
        "shadow_comparison": {"intent_match": False},
        "propensity": {"logging_policy": "regex_shadow", "action_probability": 1.0},
    }
    example = logged_example_from_intent_router_row(row)
    assert example.example_id == "router-row-1"
    assert example.logged_action_id.startswith("regex:")
    assert {candidate.source for candidate in example.action_candidates} == {"regex", "llm"}
    assert example.propensity is not None
    assert example.reward_components[0].category == "routing"


def test_missing_propensity_is_reported_as_exclusion():
    manifest = ExperimentManifest(
        experiment_id="missing-propensity",
        baseline_policy="logged",
        candidate_policy="highest_confidence",
        require_propensity=True,
    )
    report = build_experiment_report([_example()], manifest=manifest, generated_at="2026-06-12T00:00:00+00:00")
    assert report["evaluated_count"] == 0
    assert report["exclusion_counts"]["missing_propensity"] == 1
    assert report["comparison"]["summary"]["regression_detected"] is True


def test_valid_propensity_builds_ips_confidence_interval():
    logged = ActionCandidate(
        action_id="regex:thesis",
        source="regex",
        intent_class="thesis_review",
        confidence=0.9,
    )
    example = _example(
        logged=logged,
        candidate=logged,
        propensity=PropensityMetadata(
            logging_policy="regex_shadow",
            logged_action_id=logged.action_id,
            action_probability=0.5,
        ),
    )
    manifest = ExperimentManifest(
        experiment_id="valid-propensity",
        baseline_policy="logged",
        candidate_policy="highest_confidence",
    )
    report = build_experiment_report([example], manifest=manifest, generated_at="2026-06-12T00:00:00+00:00")
    assert report["evaluated_count"] == 1
    assert report["candidate_ips_reward"]["mean"] == 2.0
    assert report["comparison"]["summary"]["regression_detected"] is False


def test_gate_boundary_violation_blocks_candidate_policy():
    logged = ActionCandidate(action_id="regex:thesis", source="regex", confidence=0.4)
    candidate = ActionCandidate(
        action_id="llm:override",
        source="llm",
        confidence=0.9,
        gate_overrides=["policy_override"],
    )
    manifest = ExperimentManifest(
        experiment_id="gate-boundary",
        baseline_policy="logged",
        candidate_policy="highest_confidence",
        require_propensity=False,
    )
    report = build_experiment_report(
        [_example(logged=logged, candidate=candidate)],
        manifest=manifest,
        generated_at="2026-06-12T00:00:00+00:00",
    )
    assert report["gate_boundary"]["violations"]
    assert report["exclusion_counts"]["gate_boundary_violation"] == 1


def test_report_is_reproducible_with_fixed_generated_at():
    logged = ActionCandidate(action_id="regex:thesis", source="regex", confidence=0.9)
    example = _example(
        logged=logged,
        candidate=logged,
        propensity=PropensityMetadata(
            logging_policy="regex_shadow",
            logged_action_id=logged.action_id,
            action_probability=1.0,
        ),
    )
    manifest = ExperimentManifest(experiment_id="reproducible", baseline_policy="logged")
    first = build_experiment_report([example], manifest=manifest, generated_at="2026-06-12T00:00:00+00:00")
    second = build_experiment_report([example], manifest=manifest, generated_at="2026-06-12T00:00:00+00:00")
    assert first == second


def test_cli_dry_run_reads_jsonl(tmp_path, capsys):
    row = {
        "row_id": "router-row-cli",
        "session_id": "sess-cli",
        "client_turn_id": "turn-cli",
        "user_text": "Review NVDA",
        "regex_baseline": _route(source="regex", confidence=0.9),
        "applied_route": _route(source="regex", confidence=0.9),
        "applied_source": "regex_shadow",
        "propensity": {"logging_policy": "regex_shadow", "action_probability": 1.0},
    }
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    exit_code = main(["report", "--input-jsonl", str(input_path), "--dry-run"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["row_count"] == 1
    assert payload["manifest"]["problem_type"] == "contextual_bandit"


def test_cli_writes_report_when_not_dry_run(tmp_path):
    row = {
        "row_id": "router-row-write",
        "session_id": "sess-write",
        "client_turn_id": "turn-write",
        "user_text": "Review NVDA",
        "regex_baseline": _route(source="regex", confidence=0.9),
        "applied_route": _route(source="regex", confidence=0.9),
        "applied_source": "regex_shadow",
        "propensity": {"logging_policy": "regex_shadow", "action_probability": 1.0},
    }
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert main(["report", "--input-jsonl", str(input_path), "--output-dir", str(tmp_path / "out")]) == 0
    reports = list((tmp_path / "out").glob("*/experiment_report.json"))
    assert len(reports) == 1
    assert json.loads(Path(reports[0]).read_text(encoding="utf-8"))["row_count"] == 1
