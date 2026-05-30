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
