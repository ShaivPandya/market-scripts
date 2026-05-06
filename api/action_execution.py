from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from api.decision_state import normalize_staged_response
from api.exceptions import ConflictError, DataFetchError, NotFoundError, ValidationError
from ontology.command_service import (
    OntologyCommandConflict,
    OntologyCommandContext,
    OntologyCommandNotFound,
    OntologyCommandService,
    OntologyCommandValidationError,
)
from ontology.policy import admin_actor


def execute_api_action(
    action_id: str,
    payload: dict[str, Any],
    *,
    source_id: str,
    validation_status_code: int = 422,
    data_fetch_source: str | None = None,
) -> dict[str, Any]:
    service = OntologyCommandService()
    context = OntologyCommandContext(
        actor=admin_actor(source="api"),
        source_type="api",
        source_id=source_id,
    )
    try:
        if action_id == "resolve_approval":
            return service.resolve_approval(
                str(payload.get("approval_id") or ""),
                str(payload.get("status") or ""),
                str(payload.get("note") or "") or None,
                context,
            )
        approval = service.propose_action(action_id, payload, context, reason=f"Requested via {source_id}")
        return service.resolve_approval(str(approval["id"]), "approved", "Approved via API action.", context)
    except OntologyCommandValidationError as exc:
        if validation_status_code == 400:
            raise HTTPException(status_code=400, detail=exc.message) from exc
        raise ValidationError(exc.message) from exc
    except OntologyCommandNotFound as exc:
        raise NotFoundError(exc.resource, exc.identifier) from exc
    except OntologyCommandConflict as exc:
        raise ConflictError(exc.message) from exc
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
    entity_id: str | int | None = None,
    validation_status_code: int = 422,
    data_fetch_source: str | None = None,
) -> dict[str, Any]:
    """Create an approval for a financial mutation, optionally applying it in the same audited request."""

    proposal_reason = str(reason or "").strip() or f"Requested via {source_id}"
    service = OntologyCommandService()
    context = OntologyCommandContext(
        actor=admin_actor(source="api"),
        source_type="user",
        source_id=source_id,
    )
    try:
        approval = service.propose_action(
            action_id,
            payload,
            context,
            reason=proposal_reason,
            entity_id=str(entity_id) if entity_id is not None else None,
        )
        application_status = str(approval.get("application_status") or "pending")
        status_value = "pending_approval_created"
        if apply:
            note = str(approval_note or "").strip()
            if not note:
                raise OntologyCommandValidationError("approval_note is required when apply=true")
            approval = service.resolve_approval(str(approval["id"]), "approved", note, context)
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
    except OntologyCommandValidationError as exc:
        if validation_status_code == 400:
            raise HTTPException(status_code=400, detail=exc.message) from exc
        raise ValidationError(exc.message) from exc
    except OntologyCommandNotFound as exc:
        raise NotFoundError(exc.resource, exc.identifier) from exc
    except OntologyCommandConflict as exc:
        raise ConflictError(exc.message) from exc
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
