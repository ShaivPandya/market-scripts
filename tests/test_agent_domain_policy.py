from __future__ import annotations

from api.agent_domain_policy import analyze_agent_domain, classify_agent_domain


def test_agent_domain_policy_allows_finance_business_questions():
    allowed = [
        "how is my portfolio doing?",
        "what moved the S&P 500?",
        "is Apple attractive here?",
        "compare EURUSD and rates",
        "fix the portfolio analyzer bug",
    ]

    for text in allowed:
        assert classify_agent_domain(text) == "allow"


def test_agent_domain_policy_blocks_unrelated_questions():
    blocked = [
        "give me a chicken recipe",
        "plan a trip to Tokyo",
        "who won the Lakers game?",
        "should I take Advil?",
        "write quicksort in Python",
    ]

    for text in blocked:
        assert classify_agent_domain(text) == "block"


def test_agent_domain_policy_clarifies_ambiguous_questions_without_context():
    clarify = [
        "what do you think?",
        "Mercury",
        "is it good?",
    ]

    for text in clarify:
        assert classify_agent_domain(text) == "clarify"


def test_agent_domain_policy_flags_mixed_supported_and_unsupported_request():
    result = analyze_agent_domain("summarize my portfolio and give me a dinner recipe")

    assert result.decision == "allow"
    assert result.contains_unsupported_request is True
