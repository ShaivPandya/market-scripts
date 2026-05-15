"""Persistent financial policy matrix settings."""

from __future__ import annotations

import json
from typing import Any

from api.llm_settings import get_setting, set_setting
from portfolio.policy_matrix import default_financial_policy_matrix, normalize_financial_policy_matrix

FINANCIAL_POLICY_MATRIX_KEY = "financial.policy_matrix"


def get_financial_policy_matrix_setting() -> dict[str, Any]:
    row = get_setting(FINANCIAL_POLICY_MATRIX_KEY)
    if not row:
        return default_financial_policy_matrix()
    try:
        raw = json.loads(str(row.get("value") or "{}"))
        if not isinstance(raw, dict):
            return default_financial_policy_matrix()
        return normalize_financial_policy_matrix(raw)
    except (TypeError, json.JSONDecodeError, ValueError):
        return default_financial_policy_matrix()


def set_financial_policy_matrix_setting(policy: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_financial_policy_matrix(policy)
    return set_setting(FINANCIAL_POLICY_MATRIX_KEY, json.dumps(normalized, sort_keys=True, separators=(",", ":")))
