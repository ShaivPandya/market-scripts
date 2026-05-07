from __future__ import annotations

from pathlib import Path

from api import workflows
from api.routers import agent as agent_router
from auto_report import auto_daily_report, auto_weekly_report


def test_weekly_system_message_uses_weekly_overlay(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_load(path: Path, name: str) -> str:
        calls.append((Path(path).name, name))
        return f"<{name}>"

    monkeypatch.setattr(auto_weekly_report, "load_prompt_file", fake_load)

    msg = auto_weekly_report._build_system_message(None)

    assert calls[:2] == [
        ("system.md", "prompts/system.md"),
        ("weekly_system.md", "prompts/weekly_system.md"),
    ]
    assert "<prompts/system.md>" in msg
    assert "<prompts/weekly_system.md>" in msg


def test_daily_pass1_system_message_uses_weekly_overlay(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_load(path: Path, name: str) -> str:
        calls.append((Path(path).name, name))
        return f"<{name}>"

    monkeypatch.setattr(auto_daily_report, "load_prompt_file", fake_load)

    msg = auto_daily_report._build_pass1_system_message(None)

    assert calls[:2] == [
        ("system.md", "prompts/system.md"),
        ("weekly_system.md", "prompts/weekly_system.md"),
    ]
    assert "<prompts/system.md>" in msg
    assert "<prompts/weekly_system.md>" in msg


def test_daily_pass1_user_message_requests_structured_market_sections():
    msg = auto_daily_report._build_pass1_user_message({}, "## Weekly Performance")

    assert "output only the existing-style `# Stance Rationale` section" in msg
    assert "`market_regime_assessment`" in msg
    assert "`regime_evidence`" in msg
    assert "`watchlist`" in msg
    assert "Regime Evidence Dashboard" in msg


def test_agent_instructions_compose_core_and_overlay(tmp_path, monkeypatch):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "system.md").write_text("core philosophy", encoding="utf-8")
    (prompts_dir / "agent_system.md").write_text("agent overlay", encoding="utf-8")

    monkeypatch.setattr(agent_router, "PROMPTS_DIR", prompts_dir)
    monkeypatch.setattr(agent_router, "_build_memory_context", lambda: "")

    instructions = agent_router._build_agent_instructions()

    assert instructions == "core philosophy\n\n---\n\nagent overlay"


def test_agent_instructions_missing_overlay_degrades(tmp_path, monkeypatch):
    """When agent_system.md is missing, _build_agent_instructions degrades gracefully."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "system.md").write_text("core philosophy", encoding="utf-8")

    monkeypatch.setattr(agent_router, "PROMPTS_DIR", prompts_dir)
    monkeypatch.setattr(agent_router, "_build_memory_context", lambda: "")

    instructions = agent_router._build_agent_instructions()
    assert instructions == "core philosophy"


def test_agent_instructions_empty_overlay_degrades(tmp_path, monkeypatch):
    """When agent_system.md is empty, _build_agent_instructions degrades gracefully."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "system.md").write_text("core philosophy", encoding="utf-8")
    (prompts_dir / "agent_system.md").write_text(" \n", encoding="utf-8")

    monkeypatch.setattr(agent_router, "PROMPTS_DIR", prompts_dir)
    monkeypatch.setattr(agent_router, "_build_memory_context", lambda: "")

    instructions = agent_router._build_agent_instructions()
    assert instructions == "core philosophy"


def test_portfolio_workflow_prompts_include_entry_history_guardrail(monkeypatch):
    monkeypatch.setattr(
        workflows, "_exec_parallel", lambda *_args, **_kwargs: [("get_portfolio", {"positions": []}, 1.0)]
    )

    def fail_optional_tool(*_args, **_kwargs):
        raise RuntimeError("optional tool unavailable")

    monkeypatch.setattr(workflows, "_exec_tool", fail_optional_tool)

    prompt_builders = [
        lambda: workflows.run_morning_brief(),
        lambda: workflows.run_thesis_review("MU"),
        lambda: workflows.run_pre_earnings("MU"),
        lambda: workflows.run_post_earnings_review("MU"),
        lambda: workflows.run_weekly_portfolio_review(),
    ]

    for build_prompt in prompt_builders:
        prompt, _sections = build_prompt()
        assert "cost basis is average/book cost" in prompt
        assert "price history is market-window context only" in prompt
        assert "first purchase price/date" in prompt
        assert " ".join(("tax", "lots")) not in prompt.lower()
