from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from decision_quality.capture_chat_eval import _observed_tool_names, build_case
from decision_quality.eval_corpus import normalize_failure_tags


def test_normalize_failure_tags_maps_aliases():
    tags, failure_type, errors = normalize_failure_tags(["generic", "wrong_routing"])
    assert errors == []
    assert tags == ["generic_answer", "wrong_routing"]
    assert failure_type == "generic_answer"


def test_normalize_failure_tags_rejects_unknown():
    tags, failure_type, errors = normalize_failure_tags(["not_a_real_tag"])
    assert tags == []
    assert failure_type == "other"
    assert errors == ["unknown failure tag: not_a_real_tag"]


def test_observed_tool_names_dedupes():
    names = _observed_tool_names(
        [
            {"name": "get_thesis"},
            {"name": "get_thesis"},
            {"tool_name": "run_chart"},
        ]
    )
    assert names == ["get_thesis", "run_chart"]


def test_build_case_seeds_routing_metadata():
    session = {
        "transcript": [
            {"role": "user", "content": "What about NVDA?"},
            {
                "role": "assistant",
                "content": "Buy now.",
                "toolCalls": [{"name": "search_web"}],
            },
        ],
        "screen_context": {"ticker": "NVDA"},
    }
    with patch("api.memory_db.get_session", return_value=session):
        case = build_case(
            session_id="sess-1",
            turn_index=0,
            failure_tags=["generic", "wrong_routing"],
        )

    assert case["status"] == "draft"
    assert case["failure_tags"] == ["generic_answer", "wrong_routing"]
    assert case["failure_type"] == "generic_answer"
    assert "routing_tool_use" in case["corpus_tags"]
    assert case["expected_tool_names"] == ["search_web"]
    assert case["routing_expectations"]["required_tool_names"] == ["search_web"]
    assert case["source_session_id"] == "sess-1"
    assert case["bad_answer"] == "Buy now."


def test_build_case_redacts_email():
    session = {
        "transcript": [
            {"role": "user", "content": "Contact me at user@example.com"},
            {"role": "assistant", "content": "Noted."},
        ]
    }
    with patch("api.memory_db.get_session", return_value=session):
        case = build_case(session_id="sess-2", turn_index=0, failure_tags=[])

    assert "[REDACTED]" in case["user_message"]
    assert "user@example.com" not in case["user_message"]
