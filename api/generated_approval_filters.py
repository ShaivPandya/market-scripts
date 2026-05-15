"""Shared filters for generated approval proposals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

AUTOMATED_APPROVAL_SOURCE_TYPES = {"workflow", "system"}


def should_suppress_generated_review_approval(
    action_id: str,
    proposed_change: Mapping[str, Any],
    *,
    source_type: str,
) -> bool:
    """Return true for automated review-only action-item approvals."""

    if str(source_type or "").strip().lower() not in AUTOMATED_APPROVAL_SOURCE_TYPES:
        return False
    if str(action_id or "").strip() != "create_action_item":
        return False
    action_type = str(proposed_change.get("action_type") or "").strip().lower()
    return action_type == "review"
