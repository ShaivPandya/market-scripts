from __future__ import annotations

from api import workflows
from api.agent_models import ScreenContextModel
from api.routers import agent as agent_router


def test_position_dossier_pressure_test_prompt_includes_guardrail_and_tools(monkeypatch):
    captured_calls: list[tuple[str, dict]] = []

    def fake_parallel(calls, **_kwargs):
        captured_calls.extend(calls)
        return [(name, {"ok": True}, 1.0) for name, _args in calls]

    monkeypatch.setattr(workflows, "_exec_parallel", fake_parallel)

    prompt, sections = workflows.run_position_dossier_pressure_test("MU")

    tool_names = [name for name, _args in captured_calls]
    assert "get_dossier" in tool_names
    assert "get_thesis" in tool_names
    assert "get_position_valuation" in tool_names
    assert "run_chart" in tool_names
    assert "get_catalysts" in tool_names
    assert "get_kill_conditions" in tool_names
    assert "list_source_artifacts" in tool_names
    assert "cost basis is average/book cost" in prompt
    assert "Thesis Under Test" in prompt
    assert "Missing Inputs" in prompt
    assert sections[0]["tool"] == tool_names[0]


def test_detect_workflow_routes_dossier_pressure_test_from_position_dossier():
    screen = ScreenContextModel(
        page_name="Position Dossier",
        route="/dossier/MU",
        ticker="MU",
    )
    workflow_name, ticker = agent_router._detect_workflow("Pressure-test this position", screen)
    assert workflow_name == "position_dossier_pressure_test"
    assert ticker == "MU"


def test_detect_workflow_explicit_command():
    workflow_name, ticker = agent_router._detect_workflow("/workflow:position_dossier_pressure_test:NVDA")
    assert workflow_name == "position_dossier_pressure_test"
    assert ticker == "NVDA"


def test_detect_workflow_does_not_route_generic_pressure_test_off_dossier():
    screen = ScreenContextModel(page_name="Research", route="/research/meta", ticker="META")
    workflow_name, ticker = agent_router._detect_workflow("Pressure-test this thesis", screen)
    assert workflow_name is None
    assert ticker is None
