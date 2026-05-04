from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from api.decision_state import normalize_staged_response
from api.exceptions import ConflictError, DataFetchError, NotFoundError, ValidationError
from portfolio.action_registry import (
    ActionAuthorizationError,
    ActionConflictError,
    ActionContext,
    ActionNotFoundError,
    ActionValidationError,
    execute_action,
    propose_action,
)


def execute_api_action(
    action_id: str,
    payload: dict[str, Any],
    *,
    source_id: str,
    validation_status_code: int = 422,
    data_fetch_source: str | None = None,
) -> dict[str, Any]:
    try:
        return execute_action(
            action_id,
            payload,
            ActionContext(actor_type="user", source_type="api", source_id=source_id),
        ).output
    except ActionValidationError as exc:
        if validation_status_code == 400:
            raise HTTPException(status_code=400, detail=exc.message) from exc
        raise ValidationError(exc.message) from exc
    except ActionNotFoundError as exc:
        raise NotFoundError(exc.resource, exc.identifier) from exc
    except ActionConflictError as exc:
        raise ConflictError(exc.message) from exc
    except ActionAuthorizationError as exc:
        raise HTTPException(status_code=403, detail=exc.message) from exc
    except Exception as exc:
        if data_fetch_source:
            raise DataFetchError(source=data_fetch_source, detail=str(exc)) from exc
        raise


def stage_api_action(
    action_id: str,
    payload: dict[str, Any],
    *,
    source_id: str,
    reason: str | None = None,
    apply: bool = False,
    approval_note: str | None = None,
    entity_id: int | None = None,
    validation_status_code: int = 422,
    data_fetch_source: str | None = None,
) -> dict[str, Any]:
    """Create an approval for a financial mutation, optionally applying it in the same audited request."""

    proposal_reason = str(reason or "").strip() or f"Requested via {source_id}"
    try:
        approval = propose_action(
            action_id,
            payload,
            ActionContext(actor_type="user", source_type="user", source_id=source_id),
            reason=proposal_reason,
            entity_id=entity_id,
        )
        application_status = str(approval.get("application_status") or "pending")
        status_value = "pending_approval_created"
        if apply:
            from portfolio import core_db

            note = str(approval_note or "").strip()
            if not note:
                raise ActionValidationError("approval_note is required when apply=true")
            approval = core_db.resolve_approval(
                int(approval["id"]),
                "approved",
                note,
                actor_type="user",
                parent_action_run_id=None,
            )
            application_status = str(approval.get("application_status") or "applied")
            status_value = "applied" if application_status == "applied" else application_status
        return normalize_staged_response(
            _staged_response(
                status=status_value,
                approval=approval,
                action_id=action_id,
                proposed_change=payload,
                reason=proposal_reason,
                application_status=application_status,
            )
        )
    except ActionValidationError as exc:
        if validation_status_code == 400:
            raise HTTPException(status_code=400, detail=exc.message) from exc
        raise ValidationError(exc.message) from exc
    except ActionNotFoundError as exc:
        raise NotFoundError(exc.resource, exc.identifier) from exc
    except ActionConflictError as exc:
        raise ConflictError(exc.message) from exc
    except ActionAuthorizationError as exc:
        raise HTTPException(status_code=403, detail=exc.message) from exc
    except Exception as exc:
        if data_fetch_source:
            raise DataFetchError(source=data_fetch_source, detail=str(exc)) from exc
        raise


def _staged_response(
    *,
    status: str,
    approval: dict[str, Any],
    action_id: str,
    proposed_change: dict[str, Any],
    reason: str | None,
    application_status: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "approval_id": approval.get("id"),
        "application_status": application_status,
        "action_id": action_id,
        "entity_type": approval.get("entity_type"),
        "ticker": approval.get("ticker"),
        "proposed_change": approval.get("proposed_change") or proposed_change,
        "summary": {
            "reason": reason,
            "risk_class": approval.get("risk_class"),
            "approval_mode": approval.get("approval_mode"),
            "approval_note_required": bool(approval.get("approval_note_required")),
        },
    }
