"""First-class domain action registry for portfolio mutations."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)
PydanticValidationError = ValidationError

ActionId = str
ActionActor = Literal["user", "admin", "agent", "workflow", "approval_apply", "system"]

_TICKER_RE = re.compile(r"^[A-Z0-9.]{1,20}$")
_EXECUTE_ACTORS = {"user", "admin", "approval_apply", "system"}
_PROPOSE_ACTORS = {"user", "admin", "agent", "workflow", "system"}


class ActionError(RuntimeError):
    """Base action-layer error with a stable code for routers/tests."""

    def __init__(self, message: str, *, code: str = "action_error"):
        super().__init__(message)
        self.message = message
        self.code = code


class ActionValidationError(ActionError):
    def __init__(self, message: str):
        super().__init__(message, code="validation_error")


class ActionAuthorizationError(ActionError):
    def __init__(self, message: str):
        super().__init__(message, code="authorization_error")


class ActionConflictError(ActionError):
    def __init__(self, message: str):
        super().__init__(message, code="conflict")


class ActionNotFoundError(ActionError):
    def __init__(self, resource: str, identifier: str):
        super().__init__(f"{resource} '{identifier}' not found", code="not_found")
        self.resource = resource
        self.identifier = identifier


@dataclass(frozen=True)
class ActionContext:
    actor_type: ActionActor = "user"
    actor_id: str | None = None
    source_type: str | None = None
    source_id: str | None = None
    approval_id: int | None = None
    parent_action_run_id: int | None = None
    action_run_id: int | None = None


@dataclass(frozen=True)
class ActionCallback:
    name: str
    fn: Callable[[], None]


@dataclass(frozen=True)
class ActionResult:
    output: dict[str, Any]
    post_commit_callbacks: tuple[ActionCallback, ...] = ()


ActionHandler = Callable[[BaseModel, ActionContext], ActionResult]
ApprovalPayloadBuilder = Callable[[BaseModel], dict[str, Any]]
TickerExtractor = Callable[[BaseModel], str | None]


@dataclass(frozen=True)
class DomainAction:
    action_id: ActionId
    input_model: type[BaseModel]
    handler: ActionHandler
    schema_version: int = 1
    execute_actor_types: frozenset[str] = frozenset(_EXECUTE_ACTORS)
    propose_actor_types: frozenset[str] = frozenset(_PROPOSE_ACTORS)
    approval_entity_type: str | None = None
    approval_payload: ApprovalPayloadBuilder | None = None
    approval_ticker: TickerExtractor | None = None


class PortfolioPositionInput(BaseModel):
    ticker: str
    asset: Literal["equity", "commodity", "fx", "bond"]
    direction: Literal["long", "short"]
    contrarian: bool = False
    conviction: int = Field(default=3, ge=1, le=5)
    cost_basis: float | None = None
    shares: float | None = None

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        ticker = str(value or "").strip().upper()
        if not ticker:
            raise ValueError("Ticker cannot be empty.")
        if not _TICKER_RE.match(ticker):
            raise ValueError(f"Invalid ticker format: '{ticker}'. Only letters, digits, and dots are allowed.")
        return ticker


class UpdatePortfolioPositionsInput(BaseModel):
    positions: list[PortfolioPositionInput]


class HedgePositionInput(BaseModel):
    ticker: str
    direction: Literal["long", "short"]
    cost_basis: float | None = None
    shares: float | None = None

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        ticker = str(value or "").strip().upper()
        if not ticker:
            raise ValueError("Ticker cannot be empty.")
        if not _TICKER_RE.match(ticker):
            raise ValueError(f"Invalid ticker format: '{ticker}'.")
        return ticker


class UpdateHedgePositionsInput(BaseModel):
    positions: list[HedgePositionInput]


class ChangeThesisStatusInput(BaseModel):
    ticker: str
    status: str
    reason: str = ""

    @model_validator(mode="before")
    @classmethod
    def _support_legacy_approval_shape(cls, value: Any) -> Any:
        if isinstance(value, dict) and "status" not in value and "new_status" in value:
            value = {**value, "status": value.get("new_status")}
        return value

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        ticker = str(value or "").strip().upper()
        if not ticker:
            raise ValueError("Ticker cannot be empty.")
        if not _TICKER_RE.match(ticker):
            raise ValueError(f"Invalid ticker format: '{ticker}'. Only letters, digits, and dots are allowed.")
        return ticker

    @field_validator("status")
    @classmethod
    def _normalize_status(cls, value: str) -> str:
        status = str(value or "").strip().lower()
        if status not in {"active", "under_review", "invalidated"}:
            raise ValueError(f"Invalid status: '{status}'. Must be active, under_review, or invalidated.")
        return status

    @field_validator("reason")
    @classmethod
    def _strip_reason(cls, value: str) -> str:
        return str(value or "").strip()


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _model_payload(model: BaseModel) -> dict[str, Any]:
    return model.model_dump()


def _validation_message(exc: PydanticValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return str(exc)
    first = errors[0]
    loc = ".".join(str(part) for part in first.get("loc", ()) if part != "__root__")
    msg = str(first.get("msg") or "Invalid action input")
    return f"{loc}: {msg}" if loc else msg


def _audit_start(action: DomainAction, raw_input: dict[str, Any], context: ActionContext) -> tuple[int, str]:
    from portfolio import core_db

    input_hash = _stable_hash(raw_input)
    run = core_db.create_action_run(
        action_id=action.action_id,
        action_schema_version=action.schema_version,
        actor_type=context.actor_type,
        actor_id=context.actor_id,
        source_type=context.source_type,
        source_id=context.source_id,
        approval_id=context.approval_id,
        parent_action_run_id=context.parent_action_run_id,
        input_hash=input_hash,
        input_payload=raw_input,
    )
    run_id = int(run["id"])
    core_db.record_action_event(run_id, "start", payload={"action_id": action.action_id})
    return run_id, input_hash


def _audit_fail(run_id: int, message: str, *, rolled_back: bool = False) -> None:
    from portfolio import core_db

    core_db.record_action_event(run_id, "error", message=message)
    core_db.complete_action_run(run_id, status="rolled_back" if rolled_back else "failed", error=message)


def execute_action(
    action_id: ActionId, raw_input: dict[str, Any], context: ActionContext | None = None
) -> ActionResult:
    action = get_action(action_id)
    context = context or ActionContext()
    run_id, _input_hash = _audit_start(action, raw_input, context)
    context = replace(context, action_run_id=run_id)

    from portfolio import core_db

    try:
        try:
            typed_input = action.input_model.model_validate(raw_input)
        except PydanticValidationError as exc:
            message = _validation_message(exc)
            core_db.record_action_event(run_id, "validation_failed", message=message, payload=exc.errors())
            raise ActionValidationError(message) from exc

        normalized = _model_payload(typed_input)
        core_db.record_action_event(run_id, "validated", payload=normalized)

        if context.actor_type not in action.execute_actor_types:
            message = f"Actor '{context.actor_type}' is not authorized to execute {action.action_id}"
            core_db.record_action_event(run_id, "authorization_denied", message=message)
            raise ActionAuthorizationError(message)
        core_db.record_action_event(run_id, "authorized", payload={"actor_type": context.actor_type})

        core_db.record_action_event(run_id, "mutation_started")
        result = action.handler(typed_input, context)
        core_db.record_action_event(run_id, "mutation_completed", payload=result.output)

        for callback in result.post_commit_callbacks:
            try:
                callback.fn()
                core_db.record_action_event(run_id, "callback_completed", payload={"name": callback.name})
            except Exception as exc:
                logger.warning("Action callback failed: %s", callback.name, exc_info=True)
                core_db.record_action_event(
                    run_id,
                    "callback_failed",
                    message=str(exc) or exc.__class__.__name__,
                    payload={"name": callback.name},
                )

        core_db.record_action_event(run_id, "complete", payload=result.output)
        core_db.complete_action_run(run_id, status="succeeded", output_payload=result.output)
        return result
    except ActionError as exc:
        if core_db.get_action_run(run_id).get("status") == "running":
            _audit_fail(run_id, exc.message)
        raise
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        if core_db.get_action_run(run_id).get("status") == "running":
            _audit_fail(run_id, message)
        raise


def propose_action(
    action_id: ActionId,
    raw_input: dict[str, Any],
    context: ActionContext,
    *,
    reason: str | None = None,
    entity_id: int | None = None,
    once: bool = False,
) -> dict[str, Any]:
    action = get_action(action_id)
    proposal_action = DomainAction(
        action_id=f"{action.action_id}:propose",
        input_model=action.input_model,
        handler=lambda _input, _context: ActionResult({"status": "pending_approval_created"}),
        schema_version=action.schema_version,
        execute_actor_types=action.propose_actor_types,
    )
    run_id, input_hash = _audit_start(proposal_action, raw_input, context)

    from portfolio import core_db

    try:
        try:
            typed_input = action.input_model.model_validate(raw_input)
        except PydanticValidationError as exc:
            message = _validation_message(exc)
            core_db.record_action_event(run_id, "validation_failed", message=message, payload=exc.errors())
            raise ActionValidationError(message) from exc
        if context.actor_type not in action.propose_actor_types:
            message = f"Actor '{context.actor_type}' is not authorized to propose {action.action_id}"
            core_db.record_action_event(run_id, "authorization_denied", message=message)
            raise ActionAuthorizationError(message)
        if not action.approval_entity_type:
            raise ActionValidationError(f"Action {action.action_id} cannot be proposed for approval")

        approval_payload = (
            action.approval_payload(typed_input) if action.approval_payload else _model_payload(typed_input)
        )
        ticker = action.approval_ticker(typed_input) if action.approval_ticker else None
        create = core_db.create_pending_approval_once if once else core_db.create_pending_approval
        approval = create(
            entity_type=action.approval_entity_type,
            proposed_change=approval_payload,
            entity_id=entity_id,
            ticker=ticker,
            reason=reason,
            source_type=context.source_type or context.actor_type,
            source_id=context.source_id,
            action_id=action.action_id,
            action_schema_version=action.schema_version,
            action_input_hash=input_hash,
        )
        output = {
            "status": "pending_approval_created",
            "approval_id": approval["id"],
            "entity_type": approval["entity_type"],
            "ticker": approval.get("ticker"),
        }
        core_db.record_action_event(run_id, "approval_created", payload=output)
        core_db.complete_action_run(run_id, status="succeeded", output_payload=output)
        return approval
    except ActionError as exc:
        _audit_fail(run_id, exc.message)
        raise
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        _audit_fail(run_id, message)
        raise


def _ensure_unique_tickers(rows: list[BaseModel]) -> None:
    seen: set[str] = set()
    for row in rows:
        ticker = str(row.ticker)  # type: ignore[attr-defined]
        if ticker in seen:
            raise ActionValidationError(f"Duplicate ticker: '{ticker}'.")
        seen.add(ticker)


def _position_rows(input_model: UpdatePortfolioPositionsInput) -> list[dict[str, Any]]:
    return [
        {
            "ticker": pos.ticker,
            "asset": pos.asset,
            "direction": pos.direction,
            "contrarian": pos.contrarian,
            "conviction": pos.conviction,
            "cost_basis": pos.cost_basis,
            "shares": pos.shares,
        }
        for pos in input_model.positions
    ]


def _hedge_rows(input_model: UpdateHedgePositionsInput) -> list[dict[str, Any]]:
    return [
        {
            "ticker": pos.ticker,
            "asset": "equity",
            "direction": pos.direction,
            "contrarian": False,
            "conviction": 3,
            "cost_basis": pos.cost_basis,
            "shares": pos.shares,
        }
        for pos in input_model.positions
    ]


def _portfolio_callbacks() -> tuple[ActionCallback, ...]:
    def _reload() -> None:
        from portfolio.portfolio_dashboard import reload_portfolio

        reload_portfolio()

    def _invalidate() -> None:
        from api.cache import invalidate_all

        invalidate_all()

    return (ActionCallback("reload_portfolio", _reload), ActionCallback("invalidate_all", _invalidate))


def _restore_positions(previous_rows: list[dict[str, Any]], *, role: str, context: ActionContext, reason: str) -> None:
    from portfolio import core_db
    from portfolio.portfolio_db import save_positions

    try:
        save_positions(previous_rows, role=role)
        if context.action_run_id is not None:
            core_db.record_action_event(
                context.action_run_id, "rollback_completed", message=reason, payload={"role": role}
            )
    except Exception as rollback_exc:
        if context.action_run_id is not None:
            core_db.record_action_event(
                context.action_run_id,
                "rollback_failed",
                message=str(rollback_exc) or rollback_exc.__class__.__name__,
                payload={"role": role},
            )


def _update_portfolio_positions(input_model: BaseModel, context: ActionContext) -> ActionResult:
    typed = cast(UpdatePortfolioPositionsInput, input_model)
    if not typed.positions:
        raise ActionValidationError("At least one position is required.")
    _ensure_unique_tickers(typed.positions)

    from portfolio.portfolio_db import get_positions, save_positions

    previous = get_positions(include_hedges=False)
    rows = _position_rows(typed)
    try:
        save_positions(rows, role="position")
        updated = get_positions(include_hedges=False)
        if len(updated) != len(rows):
            raise RuntimeError("Portfolio position postcondition failed: saved row count mismatch")
    except Exception as exc:
        _restore_positions(previous, role="position", context=context, reason=str(exc) or exc.__class__.__name__)
        raise
    return ActionResult({"status": "ok", "count": len(rows)}, _portfolio_callbacks())


def _update_hedge_positions(input_model: BaseModel, context: ActionContext) -> ActionResult:
    typed = cast(UpdateHedgePositionsInput, input_model)
    _ensure_unique_tickers(typed.positions)

    from portfolio.portfolio_db import get_positions, save_positions

    rows = _hedge_rows(typed)
    tickers = {row["ticker"] for row in rows}
    existing_position_tickers = {p["ticker"] for p in get_positions(include_hedges=False)}
    collisions = tickers & existing_position_tickers
    if collisions:
        raise ActionConflictError(
            f"Ticker(s) already exist as portfolio positions: {sorted(collisions)}. "
            "A ticker cannot be both a position and a hedge."
        )

    previous = get_positions(include_hedges=True)
    previous_hedges = [row for row in previous if row.get("role") == "hedge"]
    try:
        save_positions(rows, role="hedge")
        updated = get_positions(include_hedges=True)
        hedge_count = len([row for row in updated if row.get("role") == "hedge"])
        if hedge_count != len(rows):
            raise RuntimeError("Hedge position postcondition failed: saved row count mismatch")
    except Exception as exc:
        _restore_positions(previous_hedges, role="hedge", context=context, reason=str(exc) or exc.__class__.__name__)
        raise
    return ActionResult({"status": "ok", "count": len(rows)}, _portfolio_callbacks())


def _change_thesis_status(input_model: BaseModel, _context: ActionContext) -> ActionResult:
    typed = cast(ChangeThesisStatusInput, input_model)
    from portfolio.thesis_db import get_thesis_meta, update_thesis_status

    current = get_thesis_meta(typed.ticker)
    if not current:
        raise ActionNotFoundError("Thesis", typed.ticker)
    old_status = str(current.get("status") or "")
    if old_status == typed.status:
        return ActionResult(
            {
                "ticker": typed.ticker,
                "old_status": old_status,
                "new_status": typed.status,
                "updated_at": current.get("updated_at"),
                "changed": False,
            }
        )
    updated = update_thesis_status(typed.ticker, typed.status, typed.reason)
    updated["changed"] = True
    return ActionResult(updated)


def _thesis_status_approval_payload(model: BaseModel) -> dict[str, Any]:
    typed = cast(ChangeThesisStatusInput, model)
    return {"ticker": typed.ticker, "new_status": typed.status, "reason": typed.reason}


def _ticker_from_model(model: BaseModel) -> str | None:
    return str(getattr(model, "ticker", "") or "").strip().upper() or None


_ACTIONS: dict[ActionId, DomainAction] = {
    "update_portfolio_positions": DomainAction(
        action_id="update_portfolio_positions",
        input_model=UpdatePortfolioPositionsInput,
        handler=_update_portfolio_positions,
        approval_entity_type="portfolio_positions",
        approval_payload=_model_payload,
    ),
    "update_hedge_positions": DomainAction(
        action_id="update_hedge_positions",
        input_model=UpdateHedgePositionsInput,
        handler=_update_hedge_positions,
        approval_entity_type="hedge_positions",
        approval_payload=_model_payload,
    ),
    "change_thesis_status": DomainAction(
        action_id="change_thesis_status",
        input_model=ChangeThesisStatusInput,
        handler=_change_thesis_status,
        approval_entity_type="thesis_status",
        approval_payload=_thesis_status_approval_payload,
        approval_ticker=_ticker_from_model,
    ),
}


def get_action(action_id: ActionId) -> DomainAction:
    try:
        return _ACTIONS[action_id]
    except KeyError as exc:
        raise ActionValidationError(f"Unsupported action_id: {action_id}") from exc


def list_actions() -> list[dict[str, Any]]:
    return [
        {
            "action_id": action.action_id,
            "schema_version": action.schema_version,
            "approval_entity_type": action.approval_entity_type,
        }
        for action in _ACTIONS.values()
    ]
