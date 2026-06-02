from __future__ import annotations

import pytest

from auto_report import auto_daily_report


def test_parse_pass1_missing_separator_is_contract_failure():
    analysis_md, stance = auto_daily_report.parse_pass1_response(
        "I'll search for key market-moving news from the past 24 hours before completing the analysis."
    )

    assert analysis_md.startswith("I'll search")
    assert stance["parse_error"] is True
    assert "missing separator" in stance["parse_error_reason"]

    with pytest.raises(ValueError, match="missing separator"):
        auto_daily_report._parse_pass1_response_or_raise(analysis_md)


def test_pass1_no_search_prompt_does_not_request_search_tool():
    prompt = auto_daily_report._build_pass1_user_message({}, "## Weekly Performance\n\nNo data.", web_search=False)

    assert "Web search is disabled for this pass" in prompt
    assert "Do not say you will search or browse" in prompt
    assert "use the web search tool" not in prompt
    assert "use web search to verify" not in prompt
