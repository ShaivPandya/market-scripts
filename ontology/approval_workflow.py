"""Shared helpers for multi-actor approval workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_REQUIREMENT_ID = "primary"


def default_approval_requirement() -> dict[str, Any]:
    return {
        "id": DEFAULT_REQUIREMENT_ID,
        "label": "Approval",
        "min_count": 1,
        "actor_roles": [],
        "actor_ids": [],
        "scope_type": None,
        "scope_id": None,
        "allow_requester": True,
        "allow_actor_reuse": False,
    }


def normalize_approval_requirements(value: Any) -> list[dict[str, Any]]:
    """Normalize persisted/policy approval requirement slots.

    Missing legacy values are treated as one permissive human approval to
    preserve the previous single-approver behavior.
    """

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return [default_approval_requirement()]
    raw_items = [item for item in value if isinstance(item, Mapping)]
    if not raw_items:
        return [default_approval_requirement()]

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_items, start=1):
        raw_id = str(item.get("id") or "").strip()
        requirement_id = raw_id or f"requirement-{index}"
        if requirement_id in seen:
            requirement_id = f"{requirement_id}-{index}"
        seen.add(requirement_id)

        label = str(item.get("label") or "").strip() or requirement_id.replace("_", " ").replace("-", " ").title()
        out.append(
            {
                "id": requirement_id,
                "label": label,
                "min_count": max(1, _int(item.get("min_count"), default=1)),
                "actor_roles": _normalized_list(item.get("actor_roles"), lower=True),
                "actor_ids": _normalized_list(item.get("actor_ids"), lower=True),
                "scope_type": _optional_text(item.get("scope_type")),
                "scope_id": _optional_text(item.get("scope_id")),
                "allow_requester": bool(item.get("allow_requester", False)),
                "allow_actor_reuse": bool(item.get("allow_actor_reuse", False)),
            }
        )
    return out


def normalize_approval_decisions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        requirement_id = str(item.get("requirement_id") or "").strip()
        actor_id = str(item.get("actor_id") or "").strip()
        decision = str(item.get("decision") or "").strip().lower()
        if not requirement_id or not actor_id or decision not in {"approved", "rejected"}:
            continue
        out.append(
            {
                "requirement_id": requirement_id,
                "actor_id": actor_id,
                "actor_type": str(item.get("actor_type") or "user").strip() or "user",
                "actor_roles": _normalized_list(item.get("actor_roles"), lower=True),
                "decision": decision,
                "note": _optional_text(item.get("note")),
                "decided_at": _optional_text(item.get("decided_at")),
            }
        )
    return out


def approval_requirement_progress(
    requirements: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    requirement_rows: list[dict[str, Any]] = []
    total_required = 0
    recorded_count = 0
    for requirement in requirements:
        min_count = max(1, _int(requirement.get("min_count"), default=1))
        approved_count = _approved_count(requirement, decisions)
        clipped_approved = min(approved_count, min_count)
        remaining_count = max(0, min_count - clipped_approved)
        total_required += min_count
        recorded_count += clipped_approved
        requirement_rows.append(
            {
                **requirement,
                "approved_count": clipped_approved,
                "remaining_count": remaining_count,
                "satisfied": remaining_count == 0,
            }
        )
    remaining = [row for row in requirement_rows if not row["satisfied"]]
    return {
        "total_required": total_required,
        "recorded_count": recorded_count,
        "remaining_count": sum(row["remaining_count"] for row in remaining),
        "completed": not remaining,
        "requirements": requirement_rows,
        "remaining_requirements": remaining,
    }


def approval_requirement_denial_reason(
    requirement: Mapping[str, Any],
    decisions: list[dict[str, Any]],
    *,
    actor_id: str,
    actor_roles: Sequence[str],
    requested_by_actor_id: str | None,
) -> str | None:
    normalized_actor_id = actor_id.strip().lower()
    normalized_roles = {str(role).strip().lower() for role in actor_roles if str(role).strip()}
    actor_ids = set(_normalized_list(requirement.get("actor_ids"), lower=True))
    allowed_actor_roles = set(_normalized_list(requirement.get("actor_roles"), lower=True))

    if actor_ids and normalized_actor_id not in actor_ids:
        return "Actor is not listed as an allowed approver for this requirement."
    if allowed_actor_roles and not normalized_roles.intersection(allowed_actor_roles):
        return "Actor does not have an allowed role for this approval requirement."
    if (
        requested_by_actor_id
        and not bool(requirement.get("allow_requester"))
        and normalized_actor_id == requested_by_actor_id.strip().lower()
    ):
        return "The requesting actor cannot approve this requirement."

    requirement_id = str(requirement.get("id") or "").strip()
    if any(
        str(decision.get("requirement_id") or "") == requirement_id
        and str(decision.get("actor_id") or "").strip().lower() == normalized_actor_id
        and str(decision.get("decision") or "").lower() == "approved"
        for decision in decisions
    ):
        return "Actor has already approved this requirement."

    if not bool(requirement.get("allow_actor_reuse")):
        for decision in decisions:
            if str(decision.get("decision") or "").lower() != "approved":
                continue
            if str(decision.get("actor_id") or "").strip().lower() != normalized_actor_id:
                continue
            if str(decision.get("requirement_id") or "") != requirement_id:
                return "Actor has already satisfied another approval requirement for this action."

    if _approved_count(requirement, decisions) >= max(1, _int(requirement.get("min_count"), default=1)):
        return "Approval requirement is already satisfied."
    return None


def select_approval_requirement(
    requirements: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    *,
    actor_id: str,
    actor_roles: Sequence[str],
    requested_by_actor_id: str | None,
    requirement_id: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    if requirement_id:
        for requirement in requirements:
            if str(requirement.get("id") or "") == requirement_id:
                reason = approval_requirement_denial_reason(
                    requirement,
                    decisions,
                    actor_id=actor_id,
                    actor_roles=actor_roles,
                    requested_by_actor_id=requested_by_actor_id,
                )
                return (None, reason) if reason else (requirement, None)
        return None, f"Unknown approval requirement: {requirement_id}"

    denial_reasons: list[str] = []
    satisfied_reasons: list[str] = []
    for requirement in requirements:
        reason = approval_requirement_denial_reason(
            requirement,
            decisions,
            actor_id=actor_id,
            actor_roles=actor_roles,
            requested_by_actor_id=requested_by_actor_id,
        )
        if reason is None:
            return requirement, None
        lower_reason = reason.lower()
        if (
            "already approved this requirement" in lower_reason
            or "approval requirement is already satisfied" in lower_reason
        ):
            satisfied_reasons.append(reason)
        else:
            denial_reasons.append(reason)
    if denial_reasons:
        return None, denial_reasons[0]
    if satisfied_reasons:
        return None, satisfied_reasons[0]
    return None, "No remaining approval requirements."


def _approved_count(requirement: Mapping[str, Any], decisions: list[dict[str, Any]]) -> int:
    requirement_id = str(requirement.get("id") or "").strip()
    actor_ids: set[str] = set()
    for decision in decisions:
        if str(decision.get("decision") or "").lower() != "approved":
            continue
        if str(decision.get("requirement_id") or "") != requirement_id:
            continue
        actor_id = str(decision.get("actor_id") or "").strip().lower()
        if actor_id:
            actor_ids.add(actor_id)
    return len(actor_ids)


def _normalized_list(value: Any, *, lower: bool = False) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    elif isinstance(value, Sequence):
        values = [str(part).strip() for part in value]
    else:
        values = [str(value).strip()]
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not item:
            continue
        normalized = item.lower() if lower else item
        if normalized in seen:
            continue
        out.append(normalized)
        seen.add(normalized)
    return out


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
