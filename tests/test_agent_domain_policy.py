from __future__ import annotations

from api.agent_domain_policy import analyze_agent_domain, classify_agent_domain


def test_agent_domain_policy_allows_finance_business_questions():
    allowed = [
        "how is my portfolio doing?",
        "what moved the S&P 500?",
        "is Apple attractive here?",
        "compare EURUSD and rates",
        "fix the portfolio analyzer bug",
        "can you make the proposals to update the status?",
    ]

    for text in allowed:
        assert classify_agent_domain(text) == "allow"


def test_agent_domain_policy_allows_unrelated_questions_with_soft_flag():
    unrelated = [
        "give me a chicken recipe",
        "plan a trip to Tokyo",
        "who won the Lakers game?",
        "should I take Advil?",
        "write quicksort in Python",
    ]

    for text in unrelated:
        result = analyze_agent_domain(text)
        assert result.decision == "allow"
        assert result.contains_unsupported_request is True


def test_agent_domain_policy_allows_ambiguous_followups_without_clear_unrelated_intent():
    allowed = [
        "what do you think?",
        "Mercury",
        "is it good?",
    ]

    for text in allowed:
        assert classify_agent_domain(text) == "allow"


def test_agent_domain_policy_clarifies_empty_messages():
    assert classify_agent_domain(" ") == "clarify"


def test_agent_domain_policy_flags_mixed_supported_and_unsupported_request():
    result = analyze_agent_domain("summarize my portfolio and give me a dinner recipe")

    assert result.decision == "allow"
    assert result.contains_unsupported_request is True
