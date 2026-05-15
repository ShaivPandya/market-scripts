from __future__ import annotations

import pytest

from api.financial_policy_settings import set_financial_policy_matrix_setting
from portfolio.policy_gate import evaluate_policy_gate
from portfolio.policy_matrix import (
    FinancialPolicyFacts,
    default_financial_policy_matrix,
    evaluate_financial_policy_matrix,
    normalize_financial_policy_matrix,
)


def test_policy_matrix_matching_precedence_and_limit_overrides():
    matrix = normalize_financial_policy_matrix(
        {
            "schema_version": 1,
            "policy_id": "test-policy",
            "rules": [
                {
                    "id": "low-priority-limit",
                    "priority": 10,
                    "match": {"action_ids": ["update_portfolio_positions"]},
                    "limits": {"max_position_weight_pct": 0.12},
                    "outcome": "use_checks",
                },
                {
                    "id": "high-risk-block",
                    "priority": 100,
                    "match": {"risk_levels": ["high"]},
                    "limits": {"max_position_weight_pct": 0.05},
                    "outcome": "blocked",
                    "reason": "High-risk financial actions are blocked.",
                    "remediation": "Restage with a lower risk profile.",
                },
            ],
        }
    )

    decision = evaluate_financial_policy_matrix(
        matrix,
        FinancialPolicyFacts(
            action_id="update_portfolio_positions",
            action_kind="portfolio_positions",
            risk_level="high",
        ),
    )

    assert decision["rule_id"] == "high-risk-block"
    assert decision["outcome"] == "blocked"
    assert decision["limit_overrides"]["max_position_weight_pct"] == {
        "value": 0.05,
        "rule_id": "high-risk-block",
    }
    assert [rule["id"] for rule in decision["matched_rules"]] == ["high-risk-block", "low-priority-limit"]


def test_default_policy_matrix_preserves_existing_clean_gate_behavior():
    set_financial_policy_matrix_setting(default_financial_policy_matrix())

    gate = evaluate_policy_gate(
        "create_recommendation",
        {
            "record": {
                "report_type": "idea",
                "as_of": "2026-05-15",
                "ticker": "MU",
                "action": "buy",
                "disconfirming_evidence": ["Memory remains cyclical."],
                "invalidation": "Review if HBM demand weakens.",
            }
        },
    )

    assert gate["decision"] == "pass"
    assert gate["rule_id"] == "default.current_checks"
    assert gate["approval_required"] is True
    assert gate["approval_mode"] == "approval_required"
    assert gate["failure_reasons"] == []
    assert gate["warnings"] == []


def test_policy_gate_applies_role_specific_limit_override():
    set_financial_policy_matrix_setting(
        {
            "schema_version": 1,
            "policy_id": "role-specific",
            "rules": [
                {
                    "id": "admin-low-position-limit",
                    "priority": 50,
                    "match": {"actor_roles": ["admin"], "action_ids": ["update_portfolio_positions"]},
                    "limits": {"max_position_weight_pct": 0.05},
                    "outcome": "use_checks",
                }
            ],
        }
    )

    gate = evaluate_policy_gate(
        "update_portfolio_positions",
        {
            "book_size": 1000,
            "positions": [
                {
                    "ticker": "MU",
                    "asset": "equity",
                    "direction": "long",
                    "notional_base": 100,
                }
            ],
        },
        context={"actor_roles": ["admin"]},
    )

    assert gate["decision"] == "review_required"
    assert gate["limit_overrides"]["max_position_weight_pct"]["value"] == 0.05
    assert gate["failure_reasons"][0]["code"] == "concentration_limit"


def test_policy_matrix_emits_dual_control_approval_requirements():
    set_financial_policy_matrix_setting(
        {
            "schema_version": 1,
            "policy_id": "dual-control",
            "rules": [
                {
                    "id": "high-risk-two-approvers",
                    "priority": 100,
                    "match": {"action_ids": ["update_portfolio_positions"]},
                    "outcome": "use_checks",
                    "approval_requirements": [
                        {
                            "id": "research_lead",
                            "label": "Research lead",
                            "actor_roles": ["admin"],
                            "scope_type": "ticker",
                            "scope_id": "MU",
                            "allow_requester": False,
                        },
                        {
                            "id": "portfolio_manager",
                            "label": "Portfolio manager",
                            "actor_roles": ["admin"],
                            "scope_type": "portfolio",
                            "scope_id": "default",
                            "allow_requester": False,
                        },
                    ],
                }
            ],
        }
    )

    gate = evaluate_policy_gate(
        "update_portfolio_positions",
        {"positions": [{"ticker": "MU", "asset": "equity", "direction": "long", "notional_base": 10}]},
        context={"actor_roles": ["admin"]},
    )

    assert gate["rule_id"] == "high-risk-two-approvers"
    assert [requirement["id"] for requirement in gate["approval_requirements"]] == [
        "research_lead",
        "portfolio_manager",
    ]
    assert gate["approval_requirements"][0]["allow_requester"] is False


def test_policy_gate_blocks_self_apply_by_matrix_rule():
    set_financial_policy_matrix_setting(
        {
            "schema_version": 1,
            "policy_id": "self-apply-guard",
            "rules": [
                {
                    "id": "block-self-apply",
                    "priority": 100,
                    "match": {"request_modes": ["self_apply"]},
                    "outcome": "blocked",
                    "reason": "Self-apply is disabled for this policy.",
                    "remediation": "Create a proposal for manual approval.",
                }
            ],
        }
    )

    gate = evaluate_policy_gate(
        "update_portfolio_positions",
        {"positions": [{"ticker": "MU", "asset": "equity", "direction": "long", "notional_base": 10}]},
        context={"request_mode": "self_apply"},
    )

    assert gate["decision"] == "blocked"
    assert gate["rule_id"] == "block-self-apply"
    assert gate["approval_mode"] == "self_apply"
    assert gate["failure_reasons"][0]["message"] == "Self-apply is disabled for this policy."


def test_policy_matrix_rejects_invalid_limit_key():
    with pytest.raises(ValueError, match="unsupported limit key"):
        normalize_financial_policy_matrix(
            {
                "schema_version": 1,
                "policy_id": "bad-policy",
                "rules": [{"id": "bad", "limits": {"max_magic": 1}, "outcome": "use_checks"}],
            }
        )
