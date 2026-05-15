"""Configurable financial approval policy matrix.

The matrix is intentionally small and deterministic: rules match a normalized
financial-action context, may override numeric limits, and may set the final
gate outcome. Empty match lists are wildcards.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

POLICY_MATRIX_VERSION: Literal[1] = 1
POLICY_MATRIX_OUTCOMES = ("use_checks", "pass", "warn", "review_required", "blocked")
POLICY_MATRIX_APPROVAL_MODES = ("approval_required", "self_apply", "break_glass", "none")
POLICY_MATRIX_REQUEST_MODES = ("proposal", "self_apply", "break_glass")
POLICY_MATRIX_RISK_LEVELS = ("low", "medium", "high", "unknown")
POLICY_MATRIX_DATA_FRESHNESS = ("ok", "degraded", "stale", "failed", "missing")
POLICY_MATRIX_ACTION_IDS = (
    "create_recommendation",
    "create_action_item",
    "update_portfolio_positions",
    "update_hedge_positions",
)
POLICY_MATRIX_ACTION_KINDS = (
    "buy",
    "sell",
    "reduce",
    "exit",
    "rebalance",
    "hedge",
    "enter",
    "resize",
    "portfolio_positions",
    "hedge_positions",
    "recommendation",
    "action_item",
)
POLICY_MATRIX_LIMIT_KEYS = (
    "max_position_weight_pct",
    "max_issuer_weight_pct",
    "max_asset_class_weight_pct",
    "max_gross_leverage",
    "max_net_leverage",
    "max_daily_volatility_pct",
    "max_drawdown_pct",
    "max_stress_loss_pct",
    "max_exit_days",
)

Outcome = Literal["use_checks", "pass", "warn", "review_required", "blocked"]
ApprovalMode = Literal["approval_required", "self_apply", "break_glass", "none"]

_RULE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        values = [str(part).strip() for part in value]
    else:
        values = [str(value).strip()]
    return [item for item in dict.fromkeys(values) if item]


def _lower_list(value: Any) -> list[str]:
    return [item.lower() for item in _list(value)]


class FinancialPolicyRuleMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_ids: list[str] = Field(default_factory=list)
    action_kinds: list[str] = Field(default_factory=list)
    request_modes: list[str] = Field(default_factory=list)
    actor_roles: list[str] = Field(default_factory=list)
    actor_ids: list[str] = Field(default_factory=list)
    account_ids: list[str] = Field(default_factory=list)
    portfolio_ids: list[str] = Field(default_factory=list)
    risk_levels: list[str] = Field(default_factory=list)
    data_freshness: list[str] = Field(default_factory=list)

    @field_validator(
        "action_ids",
        "action_kinds",
        "request_modes",
        "actor_roles",
        "actor_ids",
        "account_ids",
        "portfolio_ids",
        "risk_levels",
        "data_freshness",
        mode="before",
    )
    @classmethod
    def _coerce_list(cls, value: Any) -> list[str]:
        return _list(value)

    @model_validator(mode="after")
    def _normalize(self) -> FinancialPolicyRuleMatch:
        self.action_ids = _lower_list(self.action_ids)
        self.action_kinds = _lower_list(self.action_kinds)
        self.request_modes = _lower_list(self.request_modes)
        self.actor_roles = _lower_list(self.actor_roles)
        self.actor_ids = _lower_list(self.actor_ids)
        self.account_ids = _lower_list(self.account_ids)
        self.portfolio_ids = _lower_list(self.portfolio_ids)
        self.risk_levels = _lower_list(self.risk_levels)
        self.data_freshness = _lower_list(self.data_freshness)
        _ensure_known(self.request_modes, POLICY_MATRIX_REQUEST_MODES, "request_modes")
        _ensure_known(self.risk_levels, POLICY_MATRIX_RISK_LEVELS, "risk_levels")
        _ensure_known(self.data_freshness, POLICY_MATRIX_DATA_FRESHNESS, "data_freshness")
        return self


class FinancialPolicyApprovalRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str = ""
    min_count: int = Field(default=1, ge=1, le=20)
    actor_roles: list[str] = Field(default_factory=list)
    actor_ids: list[str] = Field(default_factory=list)
    scope_type: str | None = None
    scope_id: str | None = None
    allow_requester: bool = False
    allow_actor_reuse: bool = False

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not _RULE_ID_RE.match(normalized):
            raise ValueError("approval requirement id must start with a letter or number")
        return normalized

    @field_validator("label")
    @classmethod
    def _normalize_label(cls, value: str) -> str:
        return str(value or "").strip()[:200]

    @field_validator("actor_roles", "actor_ids", mode="before")
    @classmethod
    def _coerce_list(cls, value: Any) -> list[str]:
        return _list(value)

    @field_validator("scope_type", "scope_id", mode="before")
    @classmethod
    def _optional_text(cls, value: Any) -> str | None:
        text = str(value or "").strip()
        return text or None

    @model_validator(mode="after")
    def _normalize(self) -> FinancialPolicyApprovalRequirement:
        self.actor_roles = _lower_list(self.actor_roles)
        self.actor_ids = _lower_list(self.actor_ids)
        if not self.label:
            self.label = self.id.replace("_", " ").replace("-", " ").title()
        return self


class FinancialPolicyRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    enabled: bool = True
    priority: int = Field(default=0, ge=0, le=100_000)
    match: FinancialPolicyRuleMatch = Field(default_factory=FinancialPolicyRuleMatch)
    limits: dict[str, float] = Field(default_factory=dict)
    outcome: Outcome = "use_checks"
    approval_mode: ApprovalMode | None = None
    approval_requirements: list[FinancialPolicyApprovalRequirement] = Field(default_factory=list)
    reason: str = ""
    remediation: str = ""

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not _RULE_ID_RE.match(normalized):
            raise ValueError(
                "rule id must start with a letter or number and contain only letters, numbers, _, ., :, or -"
            )
        return normalized

    @field_validator("limits")
    @classmethod
    def _validate_limits(cls, value: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, raw in dict(value or {}).items():
            normalized_key = str(key or "").strip()
            if normalized_key not in POLICY_MATRIX_LIMIT_KEYS:
                raise ValueError(f"unsupported limit key: {normalized_key}")
            try:
                parsed = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{normalized_key} must be numeric") from exc
            if not math.isfinite(parsed) or parsed < 0:
                raise ValueError(f"{normalized_key} must be a finite non-negative number")
            out[normalized_key] = parsed
        return out

    @field_validator("reason", "remediation")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return str(value or "").strip()[:1000]


class FinancialPolicyMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = POLICY_MATRIX_VERSION
    policy_id: str = "default-financial-approval-policy"
    description: str = "Financial approval policy matrix"
    rules: list[FinancialPolicyRule] = Field(default_factory=list)

    @field_validator("policy_id")
    @classmethod
    def _validate_policy_id(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not _RULE_ID_RE.match(normalized):
            raise ValueError(
                "policy_id must start with a letter or number and contain only letters, numbers, _, ., :, or -"
            )
        return normalized

    @field_validator("description")
    @classmethod
    def _normalize_description(cls, value: str) -> str:
        return str(value or "").strip()[:1000]

    @model_validator(mode="after")
    def _validate_unique_rules(self) -> FinancialPolicyMatrix:
        seen: set[str] = set()
        for rule in self.rules:
            key = rule.id.lower()
            if key in seen:
                raise ValueError(f"duplicate policy rule id: {rule.id}")
            seen.add(key)
        return self


@dataclass(frozen=True)
class FinancialPolicyFacts:
    action_id: str
    action_kind: str
    request_mode: str = "proposal"
    actor_id: str = ""
    actor_roles: tuple[str, ...] = ()
    account_id: str = "default-account"
    portfolio_id: str = "default-portfolio"
    risk_level: str = "unknown"
    data_freshness: str = "missing"

    def normalized(self) -> FinancialPolicyFacts:
        return FinancialPolicyFacts(
            action_id=self.action_id.lower(),
            action_kind=self.action_kind.lower(),
            request_mode=self.request_mode.lower(),
            actor_id=self.actor_id.lower(),
            actor_roles=tuple(role.lower() for role in self.actor_roles),
            account_id=self.account_id.lower(),
            portfolio_id=self.portfolio_id.lower(),
            risk_level=self.risk_level.lower(),
            data_freshness=self.data_freshness.lower(),
        )


def default_financial_policy_matrix() -> dict[str, Any]:
    return normalize_financial_policy_matrix(
        {
            "schema_version": POLICY_MATRIX_VERSION,
            "policy_id": "default-financial-approval-policy",
            "description": "Default financial approval policy preserving current deterministic gate behavior.",
            "rules": [
                {
                    "id": "default.current_checks",
                    "enabled": True,
                    "priority": 0,
                    "match": {},
                    "limits": {},
                    "outcome": "use_checks",
                    "approval_mode": None,
                    "reason": "Use deterministic financial policy checks.",
                    "remediation": "Review policy gate warnings, failures, and approval notes before applying.",
                }
            ],
        }
    )


def normalize_financial_policy_matrix(value: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(value or default_financial_policy_matrix())
    if not raw:
        raw = default_financial_policy_matrix()
    matrix = FinancialPolicyMatrix.model_validate(raw)
    return matrix.model_dump(mode="json")


def policy_matrix_metadata() -> dict[str, Any]:
    return {
        "outcomes": list(POLICY_MATRIX_OUTCOMES),
        "approval_modes": list(POLICY_MATRIX_APPROVAL_MODES),
        "request_modes": list(POLICY_MATRIX_REQUEST_MODES),
        "risk_levels": list(POLICY_MATRIX_RISK_LEVELS),
        "data_freshness": list(POLICY_MATRIX_DATA_FRESHNESS),
        "action_ids": list(POLICY_MATRIX_ACTION_IDS),
        "action_kinds": list(POLICY_MATRIX_ACTION_KINDS),
        "limit_keys": list(POLICY_MATRIX_LIMIT_KEYS),
    }


def evaluate_financial_policy_matrix(
    matrix: Mapping[str, Any],
    facts: FinancialPolicyFacts,
) -> dict[str, Any]:
    normalized_matrix = FinancialPolicyMatrix.model_validate(dict(matrix or default_financial_policy_matrix()))
    normalized_facts = facts.normalized()
    matching_rules = [
        rule
        for rule in sorted(normalized_matrix.rules, key=lambda item: (-item.priority, item.id.lower()))
        if rule.enabled and rule_matches(rule, normalized_facts)
    ]

    outcome_rule = matching_rules[0] if matching_rules else None
    limit_overrides: dict[str, dict[str, Any]] = {}
    for rule in matching_rules:
        for limit_key, value in rule.limits.items():
            if limit_key not in limit_overrides:
                limit_overrides[limit_key] = {"value": value, "rule_id": rule.id}

    default_mode = (
        normalized_facts.request_mode
        if normalized_facts.request_mode in {"self_apply", "break_glass"}
        else "approval_required"
    )
    approval_mode = outcome_rule.approval_mode if outcome_rule and outcome_rule.approval_mode else default_mode
    return {
        "policy_id": normalized_matrix.policy_id,
        "schema_version": normalized_matrix.schema_version,
        "rule_id": outcome_rule.id if outcome_rule else None,
        "outcome": outcome_rule.outcome if outcome_rule else "use_checks",
        "approval_mode": approval_mode,
        "approval_required": approval_mode != "none",
        "reason": outcome_rule.reason if outcome_rule else "",
        "remediation": outcome_rule.remediation if outcome_rule else "",
        "limit_overrides": limit_overrides,
        "approval_requirements": [
            requirement.model_dump(mode="json")
            for requirement in (outcome_rule.approval_requirements if outcome_rule else [])
        ],
        "matched_rules": [
            {
                "id": rule.id,
                "priority": rule.priority,
                "outcome": rule.outcome,
                "approval_mode": rule.approval_mode,
                "approval_requirement_count": len(rule.approval_requirements),
                "limit_keys": sorted(rule.limits),
            }
            for rule in matching_rules
        ],
    }


def rule_matches(rule: FinancialPolicyRule, facts: FinancialPolicyFacts) -> bool:
    match = rule.match
    return (
        _matches(match.action_ids, facts.action_id)
        and _matches(match.action_kinds, facts.action_kind)
        and _matches(match.request_modes, facts.request_mode)
        and _matches_any(match.actor_roles, facts.actor_roles)
        and _matches(match.actor_ids, facts.actor_id)
        and _matches(match.account_ids, facts.account_id)
        and _matches(match.portfolio_ids, facts.portfolio_id)
        and _matches(match.risk_levels, facts.risk_level)
        and _matches(match.data_freshness, facts.data_freshness)
    )


def _matches(allowed: list[str], value: str) -> bool:
    return not allowed or "*" in allowed or value in allowed


def _matches_any(allowed: list[str], values: tuple[str, ...]) -> bool:
    return not allowed or "*" in allowed or bool(set(allowed) & set(values))


def _ensure_known(values: list[str], allowed: tuple[str, ...], field_name: str) -> None:
    unknown = sorted(value for value in values if value != "*" and value not in allowed)
    if unknown:
        raise ValueError(f"{field_name} contains unsupported value(s): {', '.join(unknown)}")
