from __future__ import annotations

from fastapi import HTTPException

from api.exceptions import ConflictError, DataFetchError, NotFoundError, ValidationError
from portfolio.action_registry import (
    ActionAuthorizationError,
    ActionConflictError,
    ActionContext,
    ActionNotFoundError,
    ActionValidationError,
    execute_action,
)


def execute_api_action(
    action_id: str,
    payload: dict,
    *,
    source_id: str,
    validation_status_code: int = 422,
    data_fetch_source: str | None = None,
) -> dict:
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
