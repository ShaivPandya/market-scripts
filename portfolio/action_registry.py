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

from api.audit import emit_audit_event, summarize_for_audit

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


ActionUpgradeAdapter = Callable[[dict[str, Any]], dict[str, Any]]
_ACTION_INPUT_MODELS: dict[tuple[ActionId, int], type[BaseModel]] = {}
_ACTION_UPGRADE_ADAPTERS: dict[tuple[ActionId, int, int], ActionUpgradeAdapter] = {}


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


class TickerMixin(BaseModel):
    ticker: str

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        ticker = str(value or "").strip().upper()
        if not ticker:
            raise ValueError("Ticker cannot be empty.")
        if not _TICKER_RE.match(ticker):
            raise ValueError(f"Invalid ticker format: '{ticker}'. Only letters, digits, and dots are allowed.")
        return ticker


class OptionalTickerMixin(BaseModel):
    ticker: str | None = None

    @field_validator("ticker")
    @classmethod
    def _normalize_optional_ticker(cls, value: str | None) -> str | None:
        ticker = str(value or "").strip().upper()
        if not ticker:
            return None
        if not _TICKER_RE.match(ticker):
            raise ValueError(f"Invalid ticker format: '{ticker}'. Only letters, digits, and dots are allowed.")
        return ticker


class CreateCatalystInput(TickerMixin):
    description: str
    category: Literal["fundamental", "technical", "macro", "event", "regulatory"] = "fundamental"
    target_date: str | None = None
    evidence: str | None = None

    @field_validator("description")
    @classmethod
    def _strip_description(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("Catalyst description cannot be empty.")
        return text


class UpdateCatalystStatusInput(OptionalTickerMixin):
    catalyst_id: int
    status: Literal["pending", "played_out", "failed", "superseded"]
    evidence: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _support_legacy_status_shape(cls, value: Any) -> Any:
        if isinstance(value, dict) and "status" not in value and "new_status" in value:
            value = {**value, "status": value.get("new_status")}
        return value


class CreateKillConditionInput(TickerMixin):
    condition: str
    metric: str | None = None
    threshold: str | None = None

    @field_validator("condition")
    @classmethod
    def _strip_condition(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("Kill condition cannot be empty.")
        return text


class UpdateKillConditionStatusInput(OptionalTickerMixin):
    kill_condition_id: int
    status: Literal["active", "triggered", "retired"]

    @model_validator(mode="before")
    @classmethod
    def _support_legacy_id_and_status_shape(cls, value: Any) -> Any:
        if isinstance(value, dict):
            updates = dict(value)
            if "kill_condition_id" not in updates and "kc_id" in updates:
                updates["kill_condition_id"] = updates.get("kc_id")
            if "status" not in updates and "new_status" in updates:
                updates["status"] = updates.get("new_status")
            return updates
        return value


class SourceRequirementActionInput(BaseModel):
    type: str = "custom"
    description: str
    required: bool = True
    freshness_days: int | None = Field(default=None, ge=0)


SourceRequirementActionValue = str | SourceRequirementActionInput


class CreateThesisClaimInput(TickerMixin):
    claim: str
    expected_evidence: str | None = None
    disconfirming_evidence: str | None = None
    source_requirements: list[SourceRequirementActionValue] = Field(default_factory=list)
    cadence: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    status: Literal["active", "supported", "challenged", "disconfirmed", "retired"] = "active"
    linked_catalyst_ids: list[int] = Field(default_factory=list)
    linked_kill_condition_ids: list[int] = Field(default_factory=list)
    source_type: Literal["workflow", "agent", "user"] | None = None
    source_id: str | None = None

    @field_validator("claim")
    @classmethod
    def _strip_claim(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("Thesis claim cannot be empty.")
        return text


class UpdateThesisClaimInput(BaseModel):
    claim_id: int
    claim: str | None = None
    expected_evidence: str | None = None
    disconfirming_evidence: str | None = None
    source_requirements: list[SourceRequirementActionValue] | None = None
    cadence: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    status: Literal["active", "supported", "challenged", "disconfirmed", "retired"] | None = None
    linked_catalyst_ids: list[int] | None = None
    linked_kill_condition_ids: list[int] | None = None


class CreateActionItemInput(OptionalTickerMixin):
    description: str
    action_type: Literal["review", "resize", "research", "exit", "enter", "hedge", "other"] = "review"
    urgency: Literal["low", "normal", "high", "urgent"] = "normal"

    @field_validator("description")
    @classmethod
    def _strip_description(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("Action item description cannot be empty.")
        return text


class CompleteActionItemInput(BaseModel):
    item_id: int
    resolution_note: str = ""

    @field_validator("resolution_note")
    @classmethod
    def _strip_resolution_note(cls, value: str) -> str:
        return str(value or "").strip()


class DismissActionItemInput(BaseModel):
    item_id: int


class CreateWatchTriggerInput(OptionalTickerMixin):
    condition: str
    trigger_type: Literal[
        "price_level",
        "technical",
        "fundamental",
        "fundamental_news",
        "event",
        "news_event",
        "macro",
        "custom",
    ] = "custom"
    expires_at: str | None = None
    definition: dict[str, Any] | None = None

    @field_validator("condition")
    @classmethod
    def _strip_condition(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("Watch trigger condition cannot be empty.")
        return text


class FireWatchTriggerInput(BaseModel):
    trigger_id: int
    result: dict[str, Any] | None = None
    evidence: str | None = None


class CancelWatchTriggerInput(BaseModel):
    trigger_id: int


class SaveThesisContentInput(TickerMixin):
    content: str
    preserve_exact_content: bool = False

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        text = str(value or "")
        if not text.strip():
            raise ValueError("Thesis content cannot be empty.")
        return text


class ResolveApprovalInput(BaseModel):
    approval_id: int
    status: Literal["approved", "rejected"]
    note: str | None = None


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
        action_schema_name=action.action_id,
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
    _emit_domain_audit(
        action,
        context,
        "domain.action.started",
        "started",
        action_run_id=run_id,
        metadata={
            "action_id": action.action_id,
            "action_schema_version": action.schema_version,
            "input_hash": input_hash,
            "input_summary": summarize_for_audit(raw_input),
        },
    )
    return run_id, input_hash


def _action_refs(
    action: DomainAction, context: ActionContext, action_run_id: int | None = None
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = [{"type": "domain_action", "id": action.action_id}]
    if action_run_id is not None:
        refs.append({"type": "action_run", "id": action_run_id})
    if context.approval_id is not None:
        refs.append({"type": "approval", "id": context.approval_id})
    if context.source_type and context.source_id:
        refs.append({"type": context.source_type, "id": context.source_id, "role": "source"})
    return refs


def _source_lineage(context: ActionContext) -> dict[str, Any]:
    return {
        "source_type": context.source_type,
        "source_id": context.source_id,
        "approval_id": context.approval_id,
        "parent_action_run_id": context.parent_action_run_id,
    }


def _emit_domain_audit(
    action: DomainAction,
    context: ActionContext,
    action_name: str,
    status: str,
    *,
    action_run_id: int | None = None,
    before_summary: Any | None = None,
    after_summary: Any | None = None,
    metadata: Any | None = None,
    error: str | None = None,
) -> None:
    emit_audit_event(
        action_name,
        "domain_action",
        status,
        actor=context,
        object_refs=_action_refs(action, context, action_run_id),
        before_summary=before_summary,
        after_summary=after_summary,
        source_lineage=_source_lineage(context),
        metadata=metadata,
        error=error,
    )


def _audit_fail(
    run_id: int,
    message: str,
    *,
    rolled_back: bool = False,
    action: DomainAction | None = None,
    context: ActionContext | None = None,
    audit_action_name: str = "domain.action.failed",
    audit_status: str = "failed",
) -> None:
    from portfolio import core_db

    core_db.record_action_event(run_id, "error", message=message)
    core_db.complete_action_run(run_id, status="rolled_back" if rolled_back else "failed", error=message)
    if action is not None and context is not None:
        _emit_domain_audit(
            action,
            context,
            audit_action_name,
            audit_status,
            action_run_id=run_id,
            after_summary={"status": audit_status},
            metadata={"action_id": action.action_id, "action_schema_version": action.schema_version},
            error=message,
        )


def execute_action(
    action_id: ActionId,
    raw_input: dict[str, Any],
    context: ActionContext | None = None,
    *,
    input_schema_version: int | None = None,
) -> ActionResult:
    action = get_action(action_id)
    context = context or ActionContext()
    audit_action = (
        action
        if input_schema_version in {None, action.schema_version}
        else replace(action, schema_version=int(input_schema_version))
    )
    run_id, _input_hash = _audit_start(audit_action, raw_input, context)
    context = replace(context, action_run_id=run_id)

    from portfolio import core_db

    try:
        try:
            typed_input = _validate_and_upgrade_action_input(
                action,
                raw_input,
                input_schema_version=input_schema_version,
            )
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
        _emit_domain_audit(
            action,
            context,
            "domain.action.succeeded",
            "succeeded",
            action_run_id=run_id,
            after_summary=result.output,
            metadata={"action_id": action.action_id, "action_schema_version": action.schema_version},
        )
        return result
    except ActionError as exc:
        if core_db.get_action_run(run_id).get("status") == "running":
            _audit_fail(
                run_id,
                exc.message,
                action=action,
                context=context,
                audit_action_name="domain.action.denied"
                if isinstance(exc, ActionAuthorizationError)
                else "domain.action.failed",
                audit_status="denied" if isinstance(exc, ActionAuthorizationError) else "failed",
            )
        raise
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        if core_db.get_action_run(run_id).get("status") == "running":
            _audit_fail(run_id, message, action=action, context=context)
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
            action_schema_name=action.action_id,
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
        _emit_domain_audit(
            proposal_action,
            context,
            "domain.action.succeeded",
            "succeeded",
            action_run_id=run_id,
            after_summary=output,
            metadata={
                "action_id": action.action_id,
                "proposal_action_id": proposal_action.action_id,
                "action_schema_version": action.schema_version,
                "input_hash": input_hash,
            },
        )
        return approval
    except ActionError as exc:
        _audit_fail(
            run_id,
            exc.message,
            action=proposal_action,
            context=context,
            audit_action_name="domain.action.denied"
            if isinstance(exc, ActionAuthorizationError)
            else "domain.action.failed",
            audit_status="denied" if isinstance(exc, ActionAuthorizationError) else "failed",
        )
        raise
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        _audit_fail(run_id, message, action=proposal_action, context=context)
        raise


def _ensure_unique_tickers(rows: list[BaseModel]) -> None:
    seen: set[str] = set()
    for row in rows:
        ticker = str(row.ticker)  # type: ignore[attr-defined]
        if ticker in seen:
            raise ActionValidationError(f"Duplicate ticker: '{ticker}'.")
        seen.add(ticker)


def register_action_schema_version(action_id: ActionId, schema_version: int, model: type[BaseModel]) -> None:
    _ACTION_INPUT_MODELS[(action_id, int(schema_version))] = model


def register_action_upgrade_adapter(
    action_id: ActionId,
    from_version: int,
    to_version: int,
    adapter: ActionUpgradeAdapter,
) -> None:
    _ACTION_UPGRADE_ADAPTERS[(action_id, int(from_version), int(to_version))] = adapter


def _validate_and_upgrade_action_input(
    action: DomainAction,
    raw_input: dict[str, Any],
    *,
    input_schema_version: int | None = None,
) -> BaseModel:
    supplied_version = int(input_schema_version or action.schema_version)
    if supplied_version == action.schema_version:
        return action.input_model.model_validate(raw_input)
    if supplied_version > action.schema_version:
        raise ActionValidationError(
            f"Unsupported future action schema version {supplied_version} for {action.action_id}"
        )

    model = _ACTION_INPUT_MODELS.get((action.action_id, supplied_version))
    if model is None:
        raise ActionValidationError(f"Missing action schema definition for {action.action_id} v{supplied_version}")
    payload = model.model_validate(raw_input).model_dump()
    current_version = supplied_version
    while current_version < action.schema_version:
        adapter = _ACTION_UPGRADE_ADAPTERS.get((action.action_id, current_version, current_version + 1))
        if adapter is None:
            raise ActionValidationError(
                f"Missing action schema upgrade adapter for {action.action_id} from "
                f"v{current_version} to v{current_version + 1}"
            )
        payload = adapter(payload)
        current_version += 1
    return action.input_model.model_validate(payload)


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


def _source_type_from_context(context: ActionContext) -> Literal["workflow", "agent", "user"]:
    if context.source_type in {"workflow", "agent", "user"}:
        return cast(Literal["workflow", "agent", "user"], context.source_type)
    if context.actor_type in {"workflow", "agent"}:
        return cast(Literal["workflow", "agent", "user"], context.actor_type)
    return "user"


def _source_requirements_payload(values: list[SourceRequirementActionValue] | None) -> list[Any]:
    result: list[Any] = []
    for value in values or []:
        if isinstance(value, str):
            result.append(value)
        else:
            result.append(value.model_dump())
    return result


def _sync_markdown_from_entities_callback(ticker: str | None) -> ActionCallback:
    def _sync() -> None:
        if not ticker:
            return
        from portfolio.thesis_sync import sync_markdown_from_entities

        sync_markdown_from_entities(ticker)

    return ActionCallback("sync_markdown_from_entities", _sync)


def _sync_entities_from_markdown_callback(ticker: str) -> ActionCallback:
    def _sync() -> None:
        from portfolio.thesis_sync import sync_entities_from_markdown

        sync_entities_from_markdown(ticker)

    return ActionCallback("sync_entities_from_markdown", _sync)


def _index_thesis_callback(ticker: str, content: str, source_path: str) -> ActionCallback:
    def _index() -> None:
        from api.retrieval import index_document

        index_document(
            doc_type="thesis",
            content=content,
            ticker=ticker,
            source_path=source_path,
            doc_id=f"thesis-{ticker}",
        )

    return ActionCallback("index_thesis", _index)


def _raise_not_found_or_validation(exc: ValueError, resource: str, identifier: int | str) -> None:
    message = str(exc)
    if message.lower().startswith(f"no {resource.lower()}"):
        raise ActionNotFoundError(resource, str(identifier)) from exc
    raise ActionValidationError(message) from exc


def _create_catalyst(input_model: BaseModel, context: ActionContext) -> ActionResult:
    typed = cast(CreateCatalystInput, input_model)
    from portfolio import core_db

    result = core_db.create_catalyst(
        ticker=typed.ticker,
        description=typed.description,
        category=typed.category,
        target_date=typed.target_date,
        evidence=typed.evidence,
        created_by=_source_type_from_context(context),
    )
    return ActionResult(result, (_sync_markdown_from_entities_callback(result["ticker"]),))


def _update_catalyst_status(input_model: BaseModel, _context: ActionContext) -> ActionResult:
    typed = cast(UpdateCatalystStatusInput, input_model)
    from portfolio import core_db

    try:
        result = core_db.update_catalyst_status(typed.catalyst_id, typed.status, typed.evidence)
    except ValueError as exc:
        _raise_not_found_or_validation(exc, "Catalyst", typed.catalyst_id)
    return ActionResult(result, (_sync_markdown_from_entities_callback(result["ticker"]),))


def _create_kill_condition(input_model: BaseModel, context: ActionContext) -> ActionResult:
    typed = cast(CreateKillConditionInput, input_model)
    from portfolio import core_db

    result = core_db.create_kill_condition(
        ticker=typed.ticker,
        condition=typed.condition,
        metric=typed.metric,
        threshold=typed.threshold,
        created_by=_source_type_from_context(context),
    )
    return ActionResult(result, (_sync_markdown_from_entities_callback(result["ticker"]),))


def _update_kill_condition_status(input_model: BaseModel, _context: ActionContext) -> ActionResult:
    typed = cast(UpdateKillConditionStatusInput, input_model)
    from portfolio import core_db

    try:
        result = core_db.update_kill_condition_status(typed.kill_condition_id, typed.status)
    except ValueError as exc:
        _raise_not_found_or_validation(exc, "Kill condition", typed.kill_condition_id)
    return ActionResult(result, (_sync_markdown_from_entities_callback(result["ticker"]),))


def _create_thesis_claim(input_model: BaseModel, context: ActionContext) -> ActionResult:
    typed = cast(CreateThesisClaimInput, input_model)
    from portfolio import core_db

    try:
        result = core_db.create_thesis_claim(
            {
                "ticker": typed.ticker,
                "claim": typed.claim,
                "expected_evidence": typed.expected_evidence,
                "disconfirming_evidence": typed.disconfirming_evidence,
                "source_requirements": _source_requirements_payload(typed.source_requirements),
                "cadence": typed.cadence,
                "confidence": typed.confidence,
                "status": typed.status,
                "linked_catalyst_ids": typed.linked_catalyst_ids,
                "linked_kill_condition_ids": typed.linked_kill_condition_ids,
                "source_type": typed.source_type or _source_type_from_context(context),
                "source_id": typed.source_id or context.source_id,
            }
        )
    except ValueError as exc:
        raise ActionValidationError(str(exc)) from exc
    return ActionResult(result, (_sync_markdown_from_entities_callback(result.get("ticker")),))


def _update_thesis_claim(input_model: BaseModel, _context: ActionContext) -> ActionResult:
    typed = cast(UpdateThesisClaimInput, input_model)
    from portfolio import core_db

    updates = typed.model_dump(exclude={"claim_id"}, exclude_unset=True)
    if "source_requirements" in updates:
        updates["source_requirements"] = _source_requirements_payload(typed.source_requirements)
    try:
        result = core_db.update_thesis_claim(typed.claim_id, updates)
    except ValueError as exc:
        _raise_not_found_or_validation(exc, "Thesis claim", typed.claim_id)
    return ActionResult(result, (_sync_markdown_from_entities_callback(result.get("ticker")),))


def _create_action_item(input_model: BaseModel, context: ActionContext) -> ActionResult:
    typed = cast(CreateActionItemInput, input_model)
    from portfolio import core_db

    result = core_db.create_action_item(
        description=typed.description,
        action_type=typed.action_type,
        ticker=typed.ticker,
        urgency=typed.urgency,
        source_type=_source_type_from_context(context),
        source_id=context.source_id,
    )
    return ActionResult(result)


def _complete_action_item(input_model: BaseModel, _context: ActionContext) -> ActionResult:
    typed = cast(CompleteActionItemInput, input_model)
    from portfolio import core_db

    try:
        result = core_db.complete_action_item(typed.item_id, typed.resolution_note)
    except ValueError as exc:
        _raise_not_found_or_validation(exc, "Action item", typed.item_id)
    return ActionResult(result)


def _dismiss_action_item(input_model: BaseModel, _context: ActionContext) -> ActionResult:
    typed = cast(DismissActionItemInput, input_model)
    from portfolio import core_db

    try:
        result = core_db.dismiss_action_item(typed.item_id)
    except ValueError as exc:
        _raise_not_found_or_validation(exc, "Action item", typed.item_id)
    return ActionResult(result)


def _create_watch_trigger(input_model: BaseModel, context: ActionContext) -> ActionResult:
    typed = cast(CreateWatchTriggerInput, input_model)
    from portfolio import core_db

    try:
        result = core_db.create_watch_trigger(
            condition=typed.condition,
            trigger_type=typed.trigger_type,
            ticker=typed.ticker,
            source_type=_source_type_from_context(context),
            source_id=context.source_id,
            expires_at=typed.expires_at,
            definition=typed.definition,
        )
    except ValueError as exc:
        raise ActionValidationError(str(exc)) from exc
    return ActionResult(result)


def _fire_watch_trigger(input_model: BaseModel, _context: ActionContext) -> ActionResult:
    typed = cast(FireWatchTriggerInput, input_model)
    from portfolio import core_db

    try:
        result = core_db.fire_watch_trigger(typed.trigger_id, result=typed.result, evidence=typed.evidence)
    except ValueError as exc:
        _raise_not_found_or_validation(exc, "Watch trigger", typed.trigger_id)
    return ActionResult(result)


def _cancel_watch_trigger(input_model: BaseModel, _context: ActionContext) -> ActionResult:
    typed = cast(CancelWatchTriggerInput, input_model)
    from portfolio import core_db

    try:
        result = core_db.cancel_watch_trigger(typed.trigger_id)
    except ValueError as exc:
        _raise_not_found_or_validation(exc, "Watch trigger", typed.trigger_id)
    return ActionResult(result)


def _save_thesis_content(input_model: BaseModel, _context: ActionContext) -> ActionResult:
    typed = cast(SaveThesisContentInput, input_model)
    from portfolio.thesis_content import save_thesis_content

    saved = save_thesis_content(
        typed.ticker,
        typed.content,
        preserve_exact_content=typed.preserve_exact_content,
    )
    return ActionResult(
        saved.output,
        (
            _index_thesis_callback(typed.ticker, saved.index_content, saved.source_path),
            _sync_entities_from_markdown_callback(typed.ticker),
        ),
    )


def _resolve_approval(input_model: BaseModel, context: ActionContext) -> ActionResult:
    typed = cast(ResolveApprovalInput, input_model)
    from portfolio import core_db

    try:
        result = core_db.apply_approval_resolution(
            typed.approval_id,
            typed.status,
            typed.note,
            parent_action_run_id=context.action_run_id,
        )
    except core_db.ApprovalApplicationError as exc:
        raise ActionConflictError(str(exc)) from exc
    except ValueError as exc:
        message = str(exc)
        if "not found" in message.lower() or "no pending" in message.lower():
            raise ActionNotFoundError("Approval", str(typed.approval_id)) from exc
        raise ActionValidationError(message) from exc
    return ActionResult(result)


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
    "create_catalyst": DomainAction(
        action_id="create_catalyst",
        input_model=CreateCatalystInput,
        handler=_create_catalyst,
        approval_entity_type="catalyst",
        approval_payload=_model_payload,
        approval_ticker=_ticker_from_model,
    ),
    "update_catalyst_status": DomainAction(
        action_id="update_catalyst_status",
        input_model=UpdateCatalystStatusInput,
        handler=_update_catalyst_status,
        approval_entity_type="catalyst_status",
        approval_payload=_model_payload,
        approval_ticker=_ticker_from_model,
    ),
    "create_kill_condition": DomainAction(
        action_id="create_kill_condition",
        input_model=CreateKillConditionInput,
        handler=_create_kill_condition,
        approval_entity_type="kill_condition",
        approval_payload=_model_payload,
        approval_ticker=_ticker_from_model,
    ),
    "update_kill_condition_status": DomainAction(
        action_id="update_kill_condition_status",
        input_model=UpdateKillConditionStatusInput,
        handler=_update_kill_condition_status,
        approval_entity_type="kill_condition_status",
        approval_payload=_model_payload,
        approval_ticker=_ticker_from_model,
    ),
    "create_thesis_claim": DomainAction(
        action_id="create_thesis_claim",
        input_model=CreateThesisClaimInput,
        handler=_create_thesis_claim,
        approval_entity_type="thesis_claim",
        approval_payload=_model_payload,
        approval_ticker=_ticker_from_model,
    ),
    "update_thesis_claim": DomainAction(
        action_id="update_thesis_claim",
        input_model=UpdateThesisClaimInput,
        handler=_update_thesis_claim,
        approval_entity_type="thesis_claim_update",
        approval_payload=_model_payload,
    ),
    "create_action_item": DomainAction(
        action_id="create_action_item",
        input_model=CreateActionItemInput,
        handler=_create_action_item,
        approval_entity_type="action_item",
        approval_payload=_model_payload,
        approval_ticker=_ticker_from_model,
    ),
    "complete_action_item": DomainAction(
        action_id="complete_action_item",
        input_model=CompleteActionItemInput,
        handler=_complete_action_item,
    ),
    "dismiss_action_item": DomainAction(
        action_id="dismiss_action_item",
        input_model=DismissActionItemInput,
        handler=_dismiss_action_item,
    ),
    "create_watch_trigger": DomainAction(
        action_id="create_watch_trigger",
        input_model=CreateWatchTriggerInput,
        handler=_create_watch_trigger,
        approval_entity_type="watch_trigger",
        approval_payload=_model_payload,
        approval_ticker=_ticker_from_model,
    ),
    "fire_watch_trigger": DomainAction(
        action_id="fire_watch_trigger",
        input_model=FireWatchTriggerInput,
        handler=_fire_watch_trigger,
    ),
    "cancel_watch_trigger": DomainAction(
        action_id="cancel_watch_trigger",
        input_model=CancelWatchTriggerInput,
        handler=_cancel_watch_trigger,
    ),
    "save_thesis_content": DomainAction(
        action_id="save_thesis_content",
        input_model=SaveThesisContentInput,
        handler=_save_thesis_content,
        approval_entity_type="thesis_content",
        approval_payload=_model_payload,
        approval_ticker=_ticker_from_model,
    ),
    "resolve_approval": DomainAction(
        action_id="resolve_approval",
        input_model=ResolveApprovalInput,
        handler=_resolve_approval,
    ),
}


def get_action(action_id: ActionId) -> DomainAction:
    try:
        return _ACTIONS[action_id]
    except KeyError as exc:
        raise ActionValidationError(f"Unsupported action_id: {action_id}") from exc


def iter_actions() -> list[DomainAction]:
    return list(_ACTIONS.values())


def list_actions() -> list[dict[str, Any]]:
    return [
        {
            "action_id": action.action_id,
            "schema_version": action.schema_version,
            "approval_entity_type": action.approval_entity_type,
        }
        for action in _ACTIONS.values()
    ]


for _action in _ACTIONS.values():
    register_action_schema_version(_action.action_id, _action.schema_version, _action.input_model)
