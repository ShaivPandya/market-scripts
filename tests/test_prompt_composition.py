from __future__ import annotations

from pathlib import Path

import pytest

from api.exceptions import ConfigurationError
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


def test_agent_instructions_compose_core_and_overlay(tmp_path, monkeypatch):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "system.md").write_text("core philosophy", encoding="utf-8")
    (prompts_dir / "agent_system.md").write_text("agent overlay", encoding="utf-8")

    monkeypatch.setattr(agent_router, "PROMPTS_DIR", prompts_dir)

    instructions = agent_router._build_agent_instructions()

    assert instructions == "core philosophy\n\n---\n\nagent overlay"


def test_agent_instructions_missing_overlay_raises(tmp_path, monkeypatch):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "system.md").write_text("core philosophy", encoding="utf-8")

    monkeypatch.setattr(agent_router, "PROMPTS_DIR", prompts_dir)

    with pytest.raises(ConfigurationError, match="agent_system.md"):
        agent_router._build_agent_instructions()


def test_agent_instructions_empty_overlay_raises(tmp_path, monkeypatch):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "system.md").write_text("core philosophy", encoding="utf-8")
    (prompts_dir / "agent_system.md").write_text(" \n", encoding="utf-8")

    monkeypatch.setattr(agent_router, "PROMPTS_DIR", prompts_dir)

    with pytest.raises(ConfigurationError, match="agent_system.md"):
        agent_router._build_agent_instructions()
