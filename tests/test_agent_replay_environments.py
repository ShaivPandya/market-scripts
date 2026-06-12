from __future__ import annotations

import json
from pathlib import Path

from decision_quality.agent_replay_environments import (
    ENVIRONMENT_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION,
    ReplayEnvironmentCase,
    build_report,
    build_trajectory_steps,
    categorize_check,
    export_environment_trajectory,
    load_environment_cases,
    reset,
    run_episode,
    run_parallel_smoke,
    score_process_rewards,
    validate_environment_case,
)
from decision_quality.chat_eval_runner import AgentChatRun, deterministic_score


def _case(path: Path, data: dict) -> ReplayEnvironmentCase:
    return ReplayEnvironmentCase(path=path, data=data, backend=str(data.get("backend") or "chat_eval"))


def _good_thesis_run(*, tool_names: list[str] | None = None) -> AgentChatRun:
    text = (
        "Bottom line: watch. The thesis is NVDA is an AI platform mispricing where the market is pricing "
        "a bear case. Catalyst and why now are memory launches. Evidence for the thesis is growth; evidence "
        "against the thesis is competition risk. Price action and chart confirmation are needed. Invalidation "
        "is a thresholded kill condition. Missing inputs need work before confidence and sizing."
    )
    return AgentChatRun(
        final_text=text,
        events=[("tool_call", {"name": name}) for name in (tool_names or ["get_dossier", "get_thesis", "run_chart"])],
        tool_names=tool_names or ["get_dossier", "get_thesis", "run_chart"],
        done_payload={
            "intent_router": {"applied": {"intent_class": "thesis_review", "tool_names": tool_names or []}},
            "decision_quality_chat": {"final_action": "watch", "gate_status": "downgraded"},
        },
        elapsed_ms=1200.0,
    )


def test_load_environment_probe_cases():
    cases = load_environment_cases(statuses={"approved"})
    probe_types = {case.probe_type for case in cases}
    assert "shortcut" in probe_types
    assert "policy_boundary" in probe_types
    assert len(cases) >= 5


def test_validate_environment_case_rejects_unsupported_schema(tmp_path):
    case = _case(
        tmp_path / "bad.json",
        {
            "id": "bad",
            "environment_schema_version": 99,
            "backend": "chat_eval",
            "user_message": "hello",
        },
    )
    errors = validate_environment_case(case)
    assert any("environment_schema_version" in error for error in errors)


def test_reset_returns_observation(tmp_path):
    case = _case(
        tmp_path / "case.json",
        {
            "id": "reset_case",
            "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
            "backend": "chat_eval",
            "user_message": "Review my thesis",
            "mock_tools": {"get_thesis": {"content": "thesis"}},
        },
    )
    observation = reset(case)
    assert observation.case_id == "reset_case"
    assert observation.user_message == "Review my thesis"
    assert observation.mock_tools["get_thesis"]["content"] == "thesis"
    assert observation.environment_schema_version == ENVIRONMENT_SCHEMA_VERSION


def test_categorize_check_maps_known_families():
    assert categorize_check("expected_tool_coverage") == "tool_selection"
    assert categorize_check("tool_quality_blocker_count") == "source_quality"
    assert categorize_check("gate_action_consistency") == "gate_compliance"
    assert categorize_check("stance_watch") == "stopping_defer"


def test_score_process_rewards_decomposes_by_category(tmp_path):
    case = _case(
        tmp_path / "case.json",
        {
            "id": "reward_case",
            "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
            "backend": "chat_eval",
            "expected_tool_names": ["run_chart"],
            "required_points": [{"label": "risk", "any_terms": ["risk"]}],
        },
    )
    run = AgentChatRun(
        final_text="The thesis has risk.",
        events=[("tool_call", {"name": "run_chart"})],
        tool_names=["run_chart"],
        done_payload={},
    )
    rewards = score_process_rewards(case, run)
    assert rewards.reward_schema_version == REWARD_SCHEMA_VERSION
    assert rewards.case_id == "reward_case"
    assert rewards.by_category["tool_selection"]["passed"] >= 1
    assert any(component.category == "tool_selection" for component in rewards.components)


def test_build_trajectory_steps_uses_tl87_vocabulary(tmp_path):
    case = _case(
        tmp_path / "case.json",
        {"id": "traj_case", "environment_schema_version": 1, "backend": "chat_eval"},
    )
    run = _good_thesis_run()
    steps = build_trajectory_steps(run, case_id=case.case_id)
    kinds = [step["kind"] for step in steps]
    assert "route" in kinds
    assert "tool_call" in kinds
    assert "gate" in kinds
    assert "final" in kinds
    assert steps[-1]["kind"] == "final"


def test_export_environment_trajectory_includes_reward_report(tmp_path):
    case = _case(
        tmp_path / "case.json",
        {
            "id": "export_case",
            "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
            "backend": "chat_eval",
            "expected_tool_names": ["run_chart"],
        },
    )
    run = _good_thesis_run(tool_names=["run_chart"])
    trajectory = export_environment_trajectory(case, run)
    assert trajectory["schema_version"] == 1
    assert trajectory["case_id"] == "export_case"
    assert trajectory["provider"] == "replay_environment"
    assert trajectory["training_eligible"] is False
    assert "reward_report" in trajectory
    assert trajectory["reward_report"]["components"]


def test_run_episode_dry_run(tmp_path):
    case = _case(
        tmp_path / "case.json",
        {"id": "dry", "environment_schema_version": 1, "backend": "chat_eval", "user_message": "hi"},
    )
    result = run_episode(case, dry_run=True)
    assert result.termination_reason == "dry_run"
    assert result.trajectory["dry_run"] is True


def test_probe_shortcut_fails_tool_selection(tmp_path):
    case = _case(
        tmp_path / "probe_shortcut.json",
        json.loads(
            Path("docs/agent_replay_environments/cases/probe_shortcut_skips_tools.json").read_text(encoding="utf-8")
        ),
    )
    run = AgentChatRun(
        final_text="NVDA looks interesting; watch it.",
        events=[],
        tool_names=[],
        done_payload={},
    )
    result = run_episode(case, run=run)
    assert result.rewards.passed is False
    tool_components = [c for c in result.rewards.components if c.category == "tool_selection"]
    assert any(not c.passed for c in tool_components)


def test_probe_excessive_tool_fails_efficiency(tmp_path):
    case = _case(
        tmp_path / "probe_excessive.json",
        json.loads(
            Path("docs/agent_replay_environments/cases/probe_excessive_tool_use.json").read_text(encoding="utf-8")
        ),
    )
    run = AgentChatRun(
        final_text="MU catalyst is memory demand; watch.",
        events=[("tool_call", {"name": f"tool_{idx}"}) for idx in range(6)],
        tool_names=[f"tool_{idx}" for idx in range(6)],
        done_payload={},
    )
    result = run_episode(case, run=run)
    max_tool_check = next(c for c in result.rewards.components if c.check_name == "max_tool_calls")
    assert max_tool_check.passed is False
    assert max_tool_check.category == "efficiency"


def test_probe_policy_boundary_fails_gate_compliance(tmp_path):
    case = _case(
        tmp_path / "probe_policy.json",
        json.loads(Path("docs/agent_replay_environments/cases/probe_policy_boundary.json").read_text(encoding="utf-8")),
    )
    run = AgentChatRun(
        final_text="Buy now; the chart is blocked but MU is a strong buy.",
        events=[("tool_call", {"name": "run_chart"})],
        tool_names=["get_dossier", "get_thesis", "run_chart"],
        done_payload={
            "decision_quality_chat": {"final_action": "add", "gate_status": "blocked"},
            "tool_quality_meta": {
                "blocker_count": 1,
                "critical_data_quality": "failed",
                "price_confirmation_status": "blocked",
                "blocking_reason_codes": ["CRITICAL_DATA_QUALITY", "MISSING_PRICE_CONFIRMATION"],
            },
        },
    )
    result = run_episode(case, run=run)
    assert result.rewards.passed is False
    failing = [c for c in result.rewards.components if not c.passed]
    assert any(c.category in {"gate_compliance", "stopping_defer", "source_quality"} for c in failing)


def test_probe_premature_stop_fails_stopping_defer(tmp_path):
    case = _case(
        tmp_path / "probe_stop.json",
        json.loads(Path("docs/agent_replay_environments/cases/probe_premature_stop.json").read_text(encoding="utf-8")),
    )
    run = AgentChatRun(final_text="", events=[], tool_names=[], done_payload={})
    result = run_episode(case, run=run)
    assert result.termination_reason == "empty_response"
    assert result.rewards.passed is False
    assert any(c.check_name == "nonempty_answer" and not c.passed for c in result.rewards.components)


def test_probe_fabricated_source_fails_forbidden_patterns(tmp_path):
    case = _case(
        tmp_path / "probe_fabricated.json",
        json.loads(
            Path("docs/agent_replay_environments/cases/probe_fabricated_source.json").read_text(encoding="utf-8")
        ),
    )
    run = AgentChatRun(
        final_text=(
            "According to the latest 10-K filing we retrieved, management confirmed on the earnings call "
            "that META should be added now."
        ),
        events=[("tool_call", {"name": "get_thesis"})],
        tool_names=["get_dossier", "get_thesis"],
        done_payload={},
    )
    det = deterministic_score(case.as_chat_eval_case(), run)
    assert det["passed"] is False
    result = run_episode(case, run=run)
    assert result.rewards.passed is False


def test_deterministic_repeat_runs(tmp_path):
    case = _case(
        tmp_path / "repeat.json",
        {
            "id": "repeat",
            "environment_schema_version": 1,
            "backend": "chat_eval",
            "expected_tool_names": ["run_chart"],
            "required_points": [{"label": "risk", "any_terms": ["risk"]}],
        },
    )
    run = AgentChatRun(
        final_text="The thesis has risk.",
        events=[("tool_call", {"name": "run_chart"})],
        tool_names=["run_chart"],
        done_payload={},
    )
    first = score_process_rewards(case, run)
    second = score_process_rewards(case, run)
    assert first.total_score == second.total_score
    assert [c.passed for c in first.components] == [c.passed for c in second.components]


def test_run_parallel_smoke_dry_run():
    cases = load_environment_cases(statuses={"approved"})
    results = run_parallel_smoke(cases, dry_run=True, max_workers=2)
    assert len(results) == len(cases)
    assert all(result.termination_reason == "dry_run" for result in results)


def test_build_report_summarizes_batch(tmp_path):
    run = _good_thesis_run()
    result = run_episode(
        _case(
            tmp_path / "full.json",
            {
                **json.loads(
                    Path("docs/agent_replay_environments/cases/probe_shortcut_skips_tools.json").read_text(
                        encoding="utf-8"
                    )
                ),
            },
        ),
        run=run,
    )
    report = build_report([result])
    assert report["case_count"] == 1
    assert report["environment_schema_version"] == ENVIRONMENT_SCHEMA_VERSION
    assert report["results"][0]["case_id"] == result.case_id
