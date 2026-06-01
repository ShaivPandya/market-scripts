from __future__ import annotations

import hashlib
import json
from pathlib import Path

from decision_quality.chat_eval_runner import (
    AgentChatRun,
    ChatEvalCase,
    build_report,
    deterministic_score,
    load_cases,
    mocked_tool_executor,
    parse_sse_events,
    run_case,
    run_from_sse_text,
    validate_case_input_refs,
)


def _case(path: Path, data: dict) -> ChatEvalCase:
    return ChatEvalCase(path=path, data=data)


def test_load_cases_and_validate_hashed_input_refs(tmp_path):
    input_path = tmp_path / "inputs" / "deck.json"
    input_path.parent.mkdir()
    input_path.write_text(json.dumps({"ticker": "UBER"}), encoding="utf-8")
    sha = hashlib.sha256(input_path.read_bytes()).hexdigest()

    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    case_path = cases_dir / "uber.json"
    case_path.write_text(
        json.dumps(
            {
                "id": "uber_chat",
                "status": "review",
                "user_message": "Here is my Uber thesis?",
                "input_refs": [{"path": "inputs/deck.json", "sha256": sha}],
            }
        ),
        encoding="utf-8",
    )

    cases = load_cases(cases_dir=cases_dir)

    assert [case.case_id for case in cases] == ["uber_chat"]
    assert validate_case_input_refs(cases[0], root=tmp_path) == []

    cases[0].data["input_refs"][0]["sha256"] = "bad"
    errors = validate_case_input_refs(cases[0], root=tmp_path)
    assert "sha256 mismatch" in errors[0]


def test_load_cases_filters_by_corpus_tag(tmp_path):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "routing.json").write_text(
        json.dumps({"id": "routing", "status": "approved", "corpus_tags": ["routing_tool_use"]}),
        encoding="utf-8",
    )
    (cases_dir / "chat.json").write_text(
        json.dumps({"id": "chat", "status": "approved", "corpus_tags": ["chat_behavior"]}),
        encoding="utf-8",
    )

    cases = load_cases(statuses={"approved"}, corpus_tags={"routing_tool_use"}, cases_dir=cases_dir)

    assert [case.case_id for case in cases] == ["routing"]


def test_sse_parsing_and_run_summary():
    raw = "\n\n".join(
        [
            'event: tool_call\ndata: {"name":"run_chart","id":"1"}',
            'event: delta\ndata: {"text":"Bottom line: "}',
            'event: delta\ndata: {"text":"watch it."}',
            'event: done\ndata: {"tools_used":["run_chart"],"decision_quality_chat":{"ran":true}}',
        ]
    )

    events = parse_sse_events(raw)
    run = run_from_sse_text(raw)

    assert events[0] == ("tool_call", {"name": "run_chart", "id": "1"})
    assert run.final_text == "Bottom line: watch it."
    assert run.tool_names == ["run_chart"]
    assert run.done_payload["decision_quality_chat"]["ran"] is True


def test_mocked_tool_executor_returns_sequences_and_blocks_missing():
    execute = mocked_tool_executor({"get_thesis": [{"a": 1}, {"a": 2}]})

    assert json.loads(execute("get_thesis", {})) == {"a": 1}
    assert json.loads(execute("get_thesis", {})) == {"a": 2}
    missing = json.loads(execute("run_chart", {}))
    assert missing["_meta"]["status"] == "failed_closed"


def test_deterministic_score_checks_required_points_dimensions_and_forbidden_patterns(tmp_path):
    case = _case(
        tmp_path / "case.json",
        {
            "expected_tool_names": ["run_chart"],
            "required_points": [
                {"label": "av risk", "all_terms": ["AV"], "any_terms": ["Waymo", "Tesla"]},
            ],
            "required_decision_quality_dimensions": [
                "simple_thesis",
                "mispricing",
                "catalyst_or_reason_now",
                "evidence_for",
                "evidence_against",
                "price_action",
                "invalidation",
                "missing_inputs",
                "confidence_sizing",
                "trade_after_trade",
            ],
            "forbidden_patterns": ["could be a good buy"],
            "expected_stance": {"label": "watch", "any_terms": ["watch"], "forbidden_terms": ["buy now"]},
        },
    )
    text = (
        "Bottom line: watch. The thesis is Uber is an AV platform mispricing where the market is pricing "
        "a bear case. Catalyst and why now are AV launches. Evidence for the thesis is growth; evidence "
        "against the thesis is Waymo AV risk. Price action and chart confirmation are needed. Invalidation "
        "is a thresholded kill condition. Missing inputs need work before confidence and sizing. If right "
        "add later; if wrong cut it; review next quarter."
    )
    run = AgentChatRun(
        final_text=text,
        events=[],
        tool_names=["run_chart"],
        done_payload={"decision_quality_chat": {"final_action": "watch", "gate_status": "downgraded"}},
    )

    score = deterministic_score(case, run)

    assert score["passed"] is True
    assert score["score"] == 100.0


def test_deterministic_score_checks_workflow_expectations(tmp_path):
    case = _case(
        tmp_path / "case.json",
        {
            "expected_tool_names": ["get_thesis", "query_ontology"],
            "required_points": [],
            "required_decision_quality_dimensions": [],
            "forbidden_patterns": [r"\bwrote the action item directly\b"],
            "workflow_expectations": {
                "requires_workflow_run_id": True,
                "expected_artifact_keys": ["evaluation_draft", "action_items"],
                "requires_pending_approval_language": True,
            },
        },
    )
    text = (
        "I proposed the research item as a pending approval for Workspace review.\n"
        "```artifacts\n"
        '{"evaluation_draft": {"ticker": "NVDA", "thesis_status": "watch"}, '
        '"action_items": [{"description": "Research NVDA memory cycle", "action_type": "research"}]}\n'
        "```"
    )
    run = AgentChatRun(
        final_text=text,
        events=[],
        tool_names=["get_thesis", "query_ontology"],
        done_payload={
            "workflow_run_id": "workflow:thesis_review:unit",
            "tool_calls": [
                {"name": "get_thesis", "status": "ok"},
                {"name": "query_ontology", "status": "ok"},
            ],
        },
    )

    score = deterministic_score(case, run)

    assert score["passed"] is True
    assert {check["name"] for check in score["checks"]} >= {
        "workflow_run_id_present",
        "workflow_tool_metadata",
        "workflow_artifacts_parseable",
        "workflow_artifact_keys",
        "workflow_pending_approval_boundary",
    }


def test_deterministic_score_fails_workflow_direct_write_and_bad_artifacts(tmp_path):
    case = _case(
        tmp_path / "case.json",
        {
            "expected_tool_names": ["get_thesis", "query_ontology"],
            "required_points": [],
            "required_decision_quality_dimensions": [],
            "forbidden_patterns": [r"\bwrote the action item directly\b"],
            "workflow_expectations": {
                "requires_workflow_run_id": True,
                "expected_artifact_keys": ["evaluation_draft", "action_items"],
                "requires_pending_approval_language": True,
            },
        },
    )
    run = AgentChatRun(
        final_text="I wrote the action item directly.\n```artifacts\n{bad json\n```",
        events=[],
        tool_names=["get_thesis"],
        done_payload={
            "tool_calls": [
                {"name": "get_thesis", "status": "ok"},
                {"name": "query_ontology", "status": "failed"},
            ],
        },
    )

    score = deterministic_score(case, run)
    failed = {check["name"] for check in score["checks"] if not check["passed"]}

    assert score["passed"] is False
    assert "expected_tool_coverage" in failed
    assert "workflow_run_id_present" in failed
    assert "workflow_tool_metadata" in failed
    assert "workflow_artifacts_parseable" in failed
    assert "workflow_pending_approval_boundary" in failed
    assert any(name.startswith("forbidden_") for name in failed)


def test_deterministic_score_checks_tool_quality_expectations(tmp_path):
    case = _case(
        tmp_path / "case.json",
        {
            "expected_tool_names": ["run_chart"],
            "required_points": [],
            "required_decision_quality_dimensions": [],
            "forbidden_patterns": [],
            "tool_quality_expectations": {
                "requires_tool_quality_meta": True,
                "min_blocker_count": 1,
                "expected_price_confirmation_status": "blocked",
                "expected_critical_data_quality": "failed",
                "required_blocking_reason_codes": ["CRITICAL_DATA_QUALITY", "MISSING_PRICE_CONFIRMATION"],
                "forbid_actionable_language_when_blocked": True,
                "required_missing_input_terms": ["chart", "blocked"],
            },
        },
    )
    run = AgentChatRun(
        final_text="Bottom line: watch it until the blocked chart input is resolved.",
        events=[],
        tool_names=["run_chart"],
        done_payload={
            "decision_quality_chat": {
                "ran": True,
                "final_action": "watch",
                "gate_status": "downgraded",
                "tool_quality": {
                    "blocker_count": 1,
                    "warning_count": 0,
                    "blocking_reason_codes": ["CRITICAL_DATA_QUALITY", "MISSING_PRICE_CONFIRMATION"],
                    "price_confirmation_status": "blocked",
                    "source_health_status": "blocked",
                    "critical_data_quality": "failed",
                },
            }
        },
    )

    score = deterministic_score(case, run)

    assert score["passed"] is True
    assert {check["name"] for check in score["checks"]} >= {
        "tool_quality_meta_present",
        "tool_quality_min_blocker_count",
        "tool_quality_price_confirmation_status",
        "tool_quality_critical_data_quality",
        "tool_quality_blocking_reason_codes",
        "tool_quality_no_actionable_language",
        "tool_quality_missing_input_terms",
    }


def test_deterministic_score_checks_context_pack_expectations(tmp_path):
    case = _case(
        tmp_path / "case.json",
        {
            "expected_tool_names": ["run_chart", "search_web"],
            "required_points": [],
            "required_decision_quality_dimensions": [],
            "forbidden_patterns": [],
            "context_pack_expectations": {
                "requires_context_pack_meta": True,
                "expected_context_pack": "catalyst",
                "expected_opportunity_type": "policy_inflection",
                "required_tool_names": ["search_web", "run_chart"],
                "expect_complete": False,
                "required_missing_input_terms": ["catalyst"],
                "forbid_actionable_when_incomplete": True,
            },
        },
    )
    run = AgentChatRun(
        final_text="Bottom line: research until catalyst and price confirmation are clearer.",
        events=[],
        tool_names=["run_chart", "search_web", "get_dossier"],
        done_payload={
            "context_pack": {
                "pack_id": "catalyst",
                "opportunity_types": ["policy_inflection", "regime_shift"],
                "is_complete": False,
                "missing_inputs": ["price reaction to catalyst"],
            }
        },
    )

    score = deterministic_score(case, run)

    assert score["passed"] is True
    assert {check["name"] for check in score["checks"]} >= {
        "context_pack_meta_present",
        "context_pack_id",
        "context_pack_opportunity_type",
        "context_pack_required_tool_names",
        "context_pack_incomplete",
        "context_pack_missing_input_terms",
        "context_pack_no_actionable_language",
    }


def test_run_case_with_fake_agent_and_report(tmp_path):
    case = _case(
        tmp_path / "case.json",
        {
            "id": "meta_chat",
            "status": "review",
            "user_message": "Here is my Meta thesis?",
            "expected_tool_names": [],
            "required_points": [],
            "required_decision_quality_dimensions": [],
            "forbidden_patterns": [],
        },
    )

    result = run_case(
        case,
        agent_runner=lambda _case: AgentChatRun(
            final_text="Bottom line: research it.",
            events=[],
            tool_names=[],
            done_payload={},
        ),
    )
    report = build_report([result], fail_under_deterministic=100.0)

    assert result["deterministic"]["passed"] is True
    assert report["summary"]["case_count"] == 1
    assert report["summary"]["deterministic_failures"] == []
