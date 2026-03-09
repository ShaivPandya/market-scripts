"""Tests for prompt loading cache invalidation and degraded fallback."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


def test_prompt_mtime_cache_reloads_on_change(tmp_path):
    """Prompt cache reloads when file modification time changes."""
    from api.routers.agent import _load_required_prompt_file, _prompt_cache

    # Clear the cache
    _prompt_cache.clear()

    prompt_file = tmp_path / "test_prompt.md"
    prompt_file.write_text("version 1")

    with patch("api.routers.agent.PROMPTS_DIR", tmp_path):
        result1 = _load_required_prompt_file("test_prompt.md")
        assert result1 == "version 1"

        # Update the file (need to ensure mtime changes)
        prompt_file.write_text("version 2")
        # Force a different mtime
        os.utime(prompt_file, (prompt_file.stat().st_mtime + 1, prompt_file.stat().st_mtime + 1))

        result2 = _load_required_prompt_file("test_prompt.md")
        assert result2 == "version 2"

    _prompt_cache.clear()


def test_prompt_mtime_cache_returns_cached_when_unchanged(tmp_path):
    """Prompt cache returns cached content when file hasn't changed."""
    from api.routers.agent import _load_required_prompt_file, _prompt_cache

    _prompt_cache.clear()

    prompt_file = tmp_path / "test_prompt2.md"
    prompt_file.write_text("stable content")

    with patch("api.routers.agent.PROMPTS_DIR", tmp_path):
        result1 = _load_required_prompt_file("test_prompt2.md")
        result2 = _load_required_prompt_file("test_prompt2.md")
        assert result1 == result2 == "stable content"

    _prompt_cache.clear()


def test_build_agent_instructions_degrades_without_agent_prompt(tmp_path):
    """Agent instructions degrade gracefully when agent_system.md is missing."""
    from api.routers.agent import _build_agent_instructions, _prompt_cache

    _prompt_cache.clear()

    core_file = tmp_path / "system.md"
    core_file.write_text("Core system prompt content")
    # agent_system.md does NOT exist

    with patch("api.routers.agent.PROMPTS_DIR", tmp_path):
        with patch("api.routers.agent._build_memory_context", return_value=""):
            result = _build_agent_instructions()
            assert "Core system prompt content" in result
            # Should not crash, just use core prompt alone

    _prompt_cache.clear()


def test_safe_import_router_returns_healthy_for_valid_module():
    """safe_import_router returns (router, True) for valid modules."""
    from api.safe_import import safe_import_router

    router, healthy = safe_import_router("api.routers.auth")
    assert healthy is True
    assert router is not None


def test_safe_import_router_returns_stub_for_invalid_module():
    """safe_import_router returns (stub, False) for invalid modules."""
    from api.safe_import import safe_import_router

    router, healthy = safe_import_router("api.routers.nonexistent_module_xyz")
    assert healthy is False
    assert router is not None  # stub router returned


def test_degraded_modules_tracked():
    """Degraded modules are tracked and queryable."""
    from api.safe_import import _degraded_modules, get_degraded_modules, safe_import_router

    safe_import_router("api.routers.totally_fake_module_abc")
    degraded = get_degraded_modules()
    assert "api.routers.totally_fake_module_abc" in degraded


def test_exception_hierarchy():
    """New exception types have correct status codes."""
    from api.exceptions import NotFoundError, ValidationError

    nf = NotFoundError("Thesis", "MU")
    assert nf.status_code == 404
    assert "MU" in nf.message

    ve = ValidationError("Bad ticker format")
    assert ve.status_code == 422
    assert "Bad ticker" in ve.message
