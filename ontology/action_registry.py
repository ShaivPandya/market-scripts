"""Canonical action and tool metadata registry.

This module owns two related registries:

* domain actions, which provide typed validation/audit/approval metadata for
  state-changing operations
* agent tool exposures, which provide the AI-safe callable surface derived from
  typed inputs plus access and approval rules

Despite living under ``ontology/``, state-changing domain actions execute
against the canonical backing stores in ``portfolio_db``, ``thesis_db``, and
``core_db``. Ontology snapshots only materialize those writes after ingestion.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    create_model,
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
_APPROVAL_APPLY_ACTORS = frozenset({"approval_apply"})


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
    provenance_event_id: str | None = None


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
PreconditionBuilder = Callable[[BaseModel], dict[str, Any]]
ToolActionInputAdapter = Callable[[BaseModel], dict[str, Any]]
ToolReasonBuilder = Callable[[BaseModel], str | None]
ToolEntityIdBuilder = Callable[[BaseModel], int | None]
ToolAccessMode = Literal["read", "compute", "proposal", "execute"]
ActionEffectKind = Literal["read_only", "approval_gated", "direct_mutation"]
ActionRiskClass = Literal["none", "low", "financial"]
ActionExecutionMode = Literal["direct", "approval_required", "break_glass"]


@dataclass(frozen=True)
class ApprovalSpec:
    entity_type: str
    reason_required: bool = False
    payload_builder: ApprovalPayloadBuilder | None = None
    ticker_extractor: TickerExtractor | None = None
    once: bool = False


@dataclass(frozen=True)
class AuditSpec:
    started_event: str = "domain.action.started"
    succeeded_event: str = "domain.action.succeeded"
    failed_event: str = "domain.action.failed"
    denied_event: str = "domain.action.denied"
    source_lineage_fields: tuple[str, ...] = ("source_type", "source_id", "approval_id", "parent_action_run_id")
    hash_input: bool = True


@dataclass(frozen=True)
class PolicySpec:
    allowed_actor_types: frozenset[str] | None = None
    ontology_actions: tuple[str, ...] = ()
    dynamic_ontology_actions: Callable[[dict[str, Any]], tuple[str, ...]] | None = None


@dataclass(frozen=True)
class OutputSchemaSpec:
    schema: dict[str, Any]
    strict: bool = False


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
    description: str = ""
    output_schema: dict[str, Any] | None = None
    effect_kind: ActionEffectKind | None = None
    approval_spec: ApprovalSpec | None = None
    audit_spec: AuditSpec = field(default_factory=AuditSpec)
    policy_spec: PolicySpec | None = None
    risk_class: ActionRiskClass = "financial"
    financial_effect: str | None = None
    default_execution_mode: ActionExecutionMode | None = None
    allow_self_apply: bool = True
    break_glass_allowed: bool = False
    reason_required: bool = True
    precondition_builder: PreconditionBuilder | None = None
    approval_summary_builder: ApprovalPayloadBuilder | None = None
    base_state_hash_fields: tuple[str, ...] = ()


ActionDefinition = DomainAction


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

    @model_validator(mode="after")
    def _validate_positions(self) -> UpdatePortfolioPositionsInput:
        if not self.positions:
            raise ValueError("At least one position is required.")
        tickers = [position.ticker for position in self.positions]
        if len(set(tickers)) != len(tickers):
            duplicate = next(ticker for ticker in tickers if tickers.count(ticker) > 1)
            raise ValueError(f"Duplicate ticker: '{duplicate}'.")
        return self


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

    @model_validator(mode="after")
    def _validate_positions(self) -> UpdateHedgePositionsInput:
        tickers = [position.ticker for position in self.positions]
        if len(set(tickers)) != len(tickers):
            duplicate = next(ticker for ticker in tickers if tickers.count(ticker) > 1)
            raise ValueError(f"Duplicate ticker: '{duplicate}'.")
        return self


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
    recommendation_id: int | None = None

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


class UpdateWatchTriggerCheckInput(BaseModel):
    trigger_id: int
    result: dict[str, Any] | None = None
    evidence: str | None = None


class UpdateWatchTriggerDefinitionInput(BaseModel):
    trigger_id: int
    definition: dict[str, Any]


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


class SaveEvaluationInput(TickerMixin):
    evaluated_at: str | None = None
    thesis_status: str
    technical_read: str = ""
    fundamental_read: str = ""
    action: str = ""
    confidence: str | float | None = None
    key_developments: list[str] = Field(default_factory=list)
    earnings_note: str | None = None
    risk_flag: str | None = None

    @field_validator("thesis_status")
    @classmethod
    def _strip_thesis_status(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("Evaluation thesis_status cannot be empty.")
        return text


class CreateResearchNoteInput(OptionalTickerMixin):
    title: str
    content: str
    note_type: Literal["general", "earnings", "catalyst_update", "risk_assessment", "workflow_output"] = "general"

    @field_validator("title", "content", "note_type")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("Research note fields cannot be empty.")
        return text


class DeletePortfolioNewsDigestInput(BaseModel):
    digest_id: str

    @field_validator("digest_id")
    @classmethod
    def _normalize_digest_id(cls, value: str) -> str:
        from portfolio.news_digests import validate_digest_id

        return validate_digest_id(str(value or "").strip())


class CreatePortfolioNewsDigestInput(BaseModel):
    content: str
    filename: str | None = None

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        text = str(value or "")
        if not text.strip():
            raise ValueError("News digest content cannot be empty.")
        return text


class CreateRecommendationInput(BaseModel):
    record: dict[str, Any]

    @field_validator("record")
    @classmethod
    def _validate_record(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict) or not value:
            raise ValueError("Recommendation record cannot be empty.")
        return value


class ResolveApprovalInput(BaseModel):
    approval_id: int
    status: Literal["approved", "rejected"]
    note: str | None = None


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _model_payload(model: BaseModel) -> dict[str, Any]:
    return model.model_dump()


def _model_payload_exclude_unset(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(exclude_unset=True)


def _hash_current_portfolio_book(_model: BaseModel) -> dict[str, Any]:
    from portfolio.portfolio_db import get_positions

    return {"positions": get_positions(include_hedges=False)}


def _hash_current_hedge_book(_model: BaseModel) -> dict[str, Any]:
    from portfolio.portfolio_db import get_hedge_positions

    return {"positions": get_hedge_positions()}


def _hash_current_thesis(model: BaseModel) -> dict[str, Any]:
    ticker = str(getattr(model, "ticker", "") or "").strip().upper()
    result: dict[str, Any] = {"ticker": ticker}
    if not ticker:
        return result
    try:
        from portfolio import thesis_content

        result["content_hash"] = _stable_hash({"content": thesis_content.read_thesis(ticker)})
    except Exception:
        result["content_hash"] = None
    try:
        from portfolio.thesis_db import get_thesis_meta

        result["meta"] = get_thesis_meta(ticker)
    except Exception:
        result["meta"] = None
    return result


def _hash_action_item_status(model: BaseModel) -> dict[str, Any]:
    item_id = int(getattr(model, "item_id", 0) or 0)
    from portfolio.core_db import get_action_items

    item = next((row for row in get_action_items(status=None) if int(row.get("id") or 0) == item_id), None)
    return {"item_id": item_id, "status": item.get("status") if item else None}


def _hash_watch_trigger_status(model: BaseModel) -> dict[str, Any]:
    trigger_id = int(getattr(model, "trigger_id", 0) or 0)
    from portfolio.core_db import get_watch_triggers

    trigger = next((row for row in get_watch_triggers(status=None) if int(row.get("id") or 0) == trigger_id), None)
    return {"trigger_id": trigger_id, "status": trigger.get("status") if trigger else None}


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
    try:
        from api import provenance

        event_id = provenance.deterministic_id("pv:action_run", run_id)
        provenance.start_event(
            event_id=event_id,
            event_type="action_run",
            event_name=action.action_id,
            actor=context,
            parent_event_id=context.provenance_event_id,
            action_run_id=run_id,
            approval_id=context.approval_id,
            input_value=raw_input,
            summary={
                "action_id": action.action_id,
                "action_schema_name": action.action_id,
                "action_schema_version": action.schema_version,
                "actor_type": context.actor_type,
            },
            metadata={
                "input_hash": input_hash,
                "source_type": context.source_type,
                "source_id": context.source_id,
                "parent_action_run_id": context.parent_action_run_id,
            },
        )
        core_db.set_action_run_provenance_event(run_id, event_id)
        provenance.link_refs(
            event_id=event_id,
            source_ref_type="domain_action",
            source_ref_id=action.action_id,
            source_ref_version=str(action.schema_version),
            target_ref_type="action_run",
            target_ref_id=str(run_id),
            link_type="executed_as",
        )
        if context.source_type and context.source_id:
            provenance.link_refs(
                event_id=event_id,
                source_ref_type=context.source_type,
                source_ref_id=str(context.source_id),
                target_ref_type="action_run",
                target_ref_id=str(run_id),
                link_type="triggered",
            )
        if context.approval_id is not None:
            provenance.link_refs(
                event_id=event_id,
                source_ref_type="approval",
                source_ref_id=str(context.approval_id),
                target_ref_type="action_run",
                target_ref_id=str(run_id),
                link_type="approved_execution",
            )
    except Exception:
        pass
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
    audit_event = emit_audit_event(
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
    if action_run_id is not None and audit_event:
        try:
            from api import provenance

            provenance.link_refs(
                event_id=_action_run_provenance_id(action_run_id),
                source_ref_type="action_run",
                source_ref_id=str(action_run_id),
                target_ref_type="audit_event",
                target_ref_id=str(audit_event.get("event_id") or audit_event.get("id")),
                link_type="audited_by",
                metadata={"action_name": action_name, "status": status},
            )
        except Exception:
            pass


def _action_run_provenance_id(run_id: int) -> str:
    try:
        from api import provenance

        return provenance.deterministic_id("pv:action_run", run_id)
    except Exception:
        return f"pv:action_run:{run_id}"


def _finish_action_provenance(
    run_id: int,
    *,
    status: str,
    output_value: Any | None = None,
    summary: Any | None = None,
    metadata: Any | None = None,
    error: str | None = None,
) -> None:
    try:
        from api import provenance

        provenance.finish_event(
            _action_run_provenance_id(run_id),
            status=status,
            output_value=output_value,
            summary=summary,
            metadata=metadata,
            error=error,
        )
    except Exception:
        pass


def _link_action_result_entities(
    action: DomainAction,
    context: ActionContext,
    run_id: int,
    output: dict[str, Any],
    input_payload: dict[str, Any],
) -> None:
    try:
        from api import provenance

        event_id = _action_run_provenance_id(run_id)
        for ref_type, ref_id, link_type in _action_result_refs(action.action_id, output, input_payload):
            provenance.link_refs(
                event_id=event_id,
                source_ref_type=provenance.REF_ACTION_RUN,
                source_ref_id=str(run_id),
                target_ref_type=ref_type,
                target_ref_id=ref_id,
                link_type=link_type,
                metadata={
                    "action_id": action.action_id,
                    "approval_id": context.approval_id,
                    "source_type": context.source_type,
                    "source_id": context.source_id,
                },
            )
    except Exception:
        logger.debug(
            "Failed to link action result entities run_id=%s action=%s", run_id, action.action_id, exc_info=True
        )


def _record_ontology_write_boundary_versions(
    *,
    action: DomainAction,
    context: ActionContext,
    run_id: int,
    input_hash: str | None,
    normalized_input: dict[str, Any],
    output: dict[str, Any],
) -> None:
    primary = False
    try:
        from ontology.domain_write_service import ontology_primary_writes_enabled, record_action_ontology_versions
        from portfolio import core_db

        primary = ontology_primary_writes_enabled()
        rows = record_action_ontology_versions(
            action_id=action.action_id,
            input_payload=normalized_input,
            output=output,
            context=context,
            input_hash=input_hash,
        )
        if rows:
            core_db.record_action_event(
                run_id,
                "ontology_versions_written",
                payload={
                    "action_id": action.action_id,
                    "count": len(rows),
                    "version_ids": [_ontology_version_id(row) for row in rows if _ontology_version_id(row)],
                },
            )
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        try:
            from portfolio import core_db

            core_db.record_action_event(run_id, "ontology_write_failed", message=message)
        except Exception:
            pass
        if primary:
            raise
        logger.warning("Ontology write-boundary mirror failed for %s", action.action_id, exc_info=True)


def _ontology_version_id(row: Mapping[str, Any]) -> str | None:
    meta = row.get("_meta")
    if isinstance(meta, Mapping):
        temporal = meta.get("temporal")
        if isinstance(temporal, Mapping) and temporal.get("version_id"):
            return str(temporal["version_id"])
    value = row.get("version_id")
    return str(value) if value else None


def _record_pending_approval_ontology_version(
    *,
    approval: Mapping[str, Any],
    context: ActionContext,
    run_id: int,
    input_hash: str | None,
) -> None:
    primary = False
    try:
        from ontology.domain_write_service import (
            ontology_primary_writes_enabled,
            record_pending_approval_ontology_version,
        )
        from portfolio import core_db

        primary = ontology_primary_writes_enabled()
        row = record_pending_approval_ontology_version(approval, context=context, input_hash=input_hash)
        if row:
            core_db.record_action_event(
                run_id,
                "ontology_approval_version_written",
                payload={"version_id": _ontology_version_id(row), "approval_id": approval.get("id")},
            )
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        try:
            from portfolio import core_db

            core_db.record_action_event(run_id, "ontology_write_failed", message=message)
        except Exception:
            pass
        if primary:
            raise
        logger.warning("Ontology approval mirror failed for approval=%s", approval.get("id"), exc_info=True)


def _action_result_refs(
    action_id: str,
    output: dict[str, Any],
    input_payload: dict[str, Any],
) -> list[tuple[str, str, str]]:
    try:
        from api import provenance

        produced = provenance.LINK_PRODUCED
        updated = provenance.LINK_UPDATED
        resolved = provenance.LINK_RESOLVED_BY
    except Exception:
        produced = "produced"
        updated = "updated"
        resolved = "resolved_by"

    def _id(*keys: str) -> str | None:
        for key in keys:
            value = output.get(key)
            if value is None:
                value = input_payload.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    def _ticker() -> str | None:
        value = output.get("ticker") or input_payload.get("ticker")
        return str(value).strip().upper() if value is not None and str(value).strip() else None

    refs: list[tuple[str, str, str]] = []
    if action_id == "update_portfolio_positions":
        refs.append(("portfolio_positions", "current", updated))
        for row in input_payload.get("positions") or []:
            if isinstance(row, dict) and row.get("ticker"):
                refs.append(("portfolio_position", str(row["ticker"]).strip().upper(), updated))
    elif action_id == "update_hedge_positions":
        refs.append(("hedge_positions", "current", updated))
        for row in input_payload.get("positions") or []:
            if isinstance(row, dict) and row.get("ticker"):
                refs.append(("hedge_position", str(row["ticker"]).strip().upper(), updated))
    elif action_id in {"change_thesis_status", "save_thesis_content"}:
        if ticker := _ticker():
            refs.append(("thesis", ticker, updated))
    elif action_id == "save_evaluation":
        if ticker := _ticker():
            evaluated_at = _id("evaluated_at") or "latest"
            refs.append(("thesis_evaluation", f"{ticker}:{evaluated_at}", produced))
            refs.append(("thesis", ticker, updated))
    elif action_id in {"create_catalyst", "update_catalyst_status"}:
        if ref_id := _id("id", "catalyst_id"):
            refs.append(("catalyst", ref_id, produced if action_id == "create_catalyst" else updated))
    elif action_id in {"create_kill_condition", "update_kill_condition_status"}:
        if ref_id := _id("id", "kill_condition_id"):
            refs.append(("kill_condition", ref_id, produced if action_id == "create_kill_condition" else updated))
    elif action_id in {"create_thesis_claim", "update_thesis_claim"}:
        if ref_id := _id("id", "claim_id"):
            refs.append(("thesis_claim", ref_id, produced if action_id == "create_thesis_claim" else updated))
    elif action_id in {"create_action_item", "complete_action_item", "dismiss_action_item"}:
        if ref_id := _id("id", "item_id"):
            refs.append(("action_item", ref_id, produced if action_id == "create_action_item" else updated))
    elif action_id in {
        "create_watch_trigger",
        "fire_watch_trigger",
        "cancel_watch_trigger",
        "update_watch_trigger_check",
        "update_watch_trigger_definition",
    }:
        if ref_id := _id("id", "trigger_id"):
            refs.append(("watch_trigger", ref_id, produced if action_id == "create_watch_trigger" else updated))
    elif action_id == "create_research_note":
        if ref_id := _id("id"):
            refs.append(("research_note", ref_id, produced))
    elif action_id == "create_portfolio_news_digest":
        if ref_id := _id("id"):
            refs.append(("news_digest", ref_id, produced))
    elif action_id == "delete_portfolio_news_digest":
        if ref_id := _id("digest_id"):
            refs.append(("news_digest", ref_id, updated))
    elif action_id == "create_recommendation":
        if ref_id := _id("id"):
            refs.append(("recommendation", ref_id, produced))
    elif action_id == "resolve_approval":
        if ref_id := _id("approval_id"):
            refs.append(("approval", ref_id, resolved))
    return refs


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
    _finish_action_provenance(
        run_id,
        status="rolled_back" if rolled_back else "failed",
        summary={"status": "rolled_back" if rolled_back else "failed"},
        metadata={"audit_action_name": audit_action_name},
        error=message,
    )
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
    audit_action = action
    if input_schema_version is not None and input_schema_version != action.schema_version:
        audit_action = replace(action, schema_version=int(input_schema_version))
    run_id, input_hash = _audit_start(audit_action, raw_input, context)
    context = replace(context, action_run_id=run_id, provenance_event_id=_action_run_provenance_id(run_id))

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
        if (
            action.effect_kind == "approval_gated"
            and context.actor_type == "approval_apply"
            and context.approval_id is None
        ):
            message = f"Approval-gated action {action.action_id} requires approval_id for execution"
            core_db.record_action_event(run_id, "authorization_denied", message=message)
            raise ActionAuthorizationError(message)
        core_db.record_action_event(run_id, "authorized", payload={"actor_type": context.actor_type})

        core_db.record_action_event(run_id, "mutation_started")
        if context.actor_type == "approval_apply":
            from ontology.domain_write_service import domain_write_scope

            with domain_write_scope(
                action_id=action.action_id,
                actor_type=context.actor_type,
                approval_id=context.approval_id,
                action_run_id=run_id,
                source_type=context.source_type,
                source_id=context.source_id,
            ):
                result = action.handler(typed_input, context)
        else:
            result = action.handler(typed_input, context)
        core_db.record_action_event(run_id, "mutation_completed", payload=result.output)
        _record_ontology_write_boundary_versions(
            action=action,
            context=context,
            run_id=run_id,
            input_hash=input_hash,
            normalized_input=normalized,
            output=result.output,
        )

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
        _link_action_result_entities(action, context, run_id, result.output, normalized)
        _finish_action_provenance(
            run_id,
            status="succeeded",
            output_value=result.output,
            summary={"status": "succeeded", **result.output},
            metadata={"action_id": action.action_id, "action_schema_version": action.schema_version},
        )
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
        action_run = core_db.get_action_run(run_id) or {}
        if action_run.get("status") == "running":
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
        action_run = core_db.get_action_run(run_id) or {}
        if action_run.get("status") == "running":
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
    approval_spec = action.approval_spec
    proposal_action = DomainAction(
        action_id=f"{action.action_id}:propose",
        input_model=action.input_model,
        handler=lambda _input, _context: ActionResult({"status": "pending_approval_created"}),
        schema_version=action.schema_version,
        execute_actor_types=action.propose_actor_types,
    )
    run_id, input_hash = _audit_start(proposal_action, raw_input, context)
    proposal_event_id = _action_run_provenance_id(run_id)

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
        if approval_spec is None:
            raise ActionValidationError(f"Action {action.action_id} cannot be proposed for approval")
        if approval_spec.reason_required and not str(reason or "").strip():
            raise ActionValidationError(f"Action {action.action_id} requires a proposal reason")

        approval_payload = (
            approval_spec.payload_builder(typed_input) if approval_spec.payload_builder else _model_payload(typed_input)
        )
        ticker = approval_spec.ticker_extractor(typed_input) if approval_spec.ticker_extractor else None
        base_state_hash = compute_action_base_state_hash(action.action_id, approval_payload)
        use_once = once or approval_spec.once
        create = core_db.create_pending_approval_once if use_once else core_db.create_pending_approval
        approval = create(
            entity_type=approval_spec.entity_type,
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
            risk_class=action.risk_class,
            approval_mode=action.default_execution_mode or "approval_required",
            base_state_hash=base_state_hash,
            requested_by_actor_id=context.actor_id,
            approval_note_required=action.reason_required,
        )
        try:
            from api import provenance

            core_db.set_pending_approval_provenance(
                int(approval["id"]),
                origin_provenance_event_id=proposal_event_id,
            )
            provenance.link_refs(
                event_id=proposal_event_id,
                source_ref_type="action_run",
                source_ref_id=str(run_id),
                target_ref_type="approval",
                target_ref_id=str(approval["id"]),
                link_type="proposed",
                metadata={"action_id": action.action_id, "entity_type": approval.get("entity_type")},
            )
        except Exception:
            pass
        output = {
            "status": "pending_approval_created",
            "approval_id": approval["id"],
            "entity_type": approval["entity_type"],
            "ticker": approval.get("ticker"),
        }
        _record_pending_approval_ontology_version(
            approval=approval,
            context=replace(context, action_run_id=run_id, provenance_event_id=proposal_event_id),
            run_id=run_id,
            input_hash=input_hash,
        )
        core_db.record_action_event(run_id, "approval_created", payload=output)
        core_db.complete_action_run(run_id, status="succeeded", output_payload=output)
        _finish_action_provenance(
            run_id,
            status="succeeded",
            output_value=output,
            summary=output,
            metadata={
                "action_id": action.action_id,
                "proposal_action_id": proposal_action.action_id,
                "action_schema_version": action.schema_version,
            },
        )
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


def compute_action_base_state_hash(action_id: ActionId, raw_input: dict[str, Any]) -> str | None:
    """Return the approval precondition hash for an action input."""

    action = get_action(action_id)
    if action.precondition_builder is None:
        return None
    typed_input = _validate_and_upgrade_action_input(action, raw_input)
    return _stable_hash(action.precondition_builder(typed_input))


def _ensure_unique_tickers(rows: Sequence[BaseModel]) -> None:
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


def _update_watch_trigger_check(input_model: BaseModel, _context: ActionContext) -> ActionResult:
    typed = cast(UpdateWatchTriggerCheckInput, input_model)
    from portfolio import core_db

    try:
        result = core_db.update_watch_trigger_check(typed.trigger_id, result=typed.result, evidence=typed.evidence)
    except ValueError as exc:
        _raise_not_found_or_validation(exc, "Watch trigger", typed.trigger_id)
    return ActionResult(result)


def _update_watch_trigger_definition(input_model: BaseModel, _context: ActionContext) -> ActionResult:
    typed = cast(UpdateWatchTriggerDefinitionInput, input_model)
    from portfolio import core_db

    try:
        result = core_db.update_watch_trigger_definition(typed.trigger_id, typed.definition)
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


def _save_evaluation(input_model: BaseModel, _context: ActionContext) -> ActionResult:
    typed = cast(SaveEvaluationInput, input_model)
    from portfolio.thesis_db import save_evaluations

    evaluated_at = typed.evaluated_at or datetime.now(UTC).isoformat()
    payload = typed.model_dump(exclude={"evaluated_at"})
    inserted = save_evaluations(evaluated_at, [payload])
    return ActionResult(
        {
            "status": "ok",
            "ticker": typed.ticker,
            "evaluated_at": evaluated_at,
            "count": inserted,
        }
    )


def _create_research_note(input_model: BaseModel, context: ActionContext) -> ActionResult:
    typed = cast(CreateResearchNoteInput, input_model)
    from portfolio import core_db

    result = core_db.create_research_note(
        typed.title,
        typed.content,
        ticker=typed.ticker,
        note_type=typed.note_type,
        source_type=_source_type_from_context(context),
        source_id=context.source_id,
    )
    return ActionResult(result)


def _delete_portfolio_news_digest(input_model: BaseModel, _context: ActionContext) -> ActionResult:
    typed = cast(DeletePortfolioNewsDigestInput, input_model)
    from api.routers.portfolio_news import _delete_digest_index_best_effort
    from portfolio.news_digests import delete_digest

    deleted = delete_digest(typed.digest_id)
    if not deleted:
        raise ActionNotFoundError("News digest", typed.digest_id)
    return ActionResult(
        {"status": "ok", "digest_id": typed.digest_id, "deleted": True},
        (ActionCallback("delete_digest_index", lambda: _delete_digest_index_best_effort(typed.digest_id)),),
    )


def _create_portfolio_news_digest(input_model: BaseModel, _context: ActionContext) -> ActionResult:
    typed = cast(CreatePortfolioNewsDigestInput, input_model)
    from api.routers.portfolio_news import _index_digest_best_effort
    from portfolio.news_digests import save_digest

    detail = save_digest(typed.content, filename=typed.filename)
    return ActionResult(
        {"status": "ok", "digest": detail, "id": detail.get("id")},
        (ActionCallback("index_digest", lambda: _index_digest_best_effort(detail)),),
    )


def _create_recommendation(input_model: BaseModel, _context: ActionContext) -> ActionResult:
    typed = cast(CreateRecommendationInput, input_model)
    from portfolio import core_db

    record = typed.record
    result = (
        core_db.upsert_recommendation(record)
        if record.get("idempotency_key")
        else core_db.create_recommendation(record)
    )
    if result.get("recommendation_status") == "clear" and result.get("action") in {
        "buy",
        "sell",
        "reduce",
        "exit",
        "rebalance",
        "hedge",
    }:
        description = f"{str(result.get('action') or '').replace('_', ' ').title()} {result.get('instrument') or result.get('ticker') or 'portfolio'}"
        if result.get("target_change"):
            description += f" ({result['target_change']})"
        approval = propose_action(
            "create_action_item",
            {
                "recommendation_id": result["id"],
                "ticker": result.get("ticker"),
                "description": description,
                "action_type": _recommendation_action_type(str(result.get("action") or "")),
                "urgency": "high" if result.get("action") in {"exit", "reduce"} else "normal",
            },
            ActionContext(
                actor_type="workflow",
                source_type="workflow",
                source_id=result.get("report_id") or f"{result.get('report_type')}:{result.get('as_of')}",
            ),
            reason=result.get("rationale", ""),
            once=True,
        )
        result = core_db.update_recommendation_approval(result["id"], approval["id"], "pending")
    return ActionResult(result)


def _recommendation_action_type(action: str) -> str:
    if action in {"buy", "enter"}:
        return "enter"
    if action in {"sell", "exit"}:
        return "exit"
    if action in {"reduce", "rebalance"}:
        return "resize"
    if action == "hedge":
        return "hedge"
    return "review"


def _resolve_approval(input_model: BaseModel, context: ActionContext) -> ActionResult:
    typed = cast(ResolveApprovalInput, input_model)
    from portfolio import core_db

    try:
        result = core_db.apply_approval_resolution(
            typed.approval_id,
            typed.status,
            typed.note,
            parent_action_run_id=context.action_run_id,
            resolved_by_actor_id=context.actor_id,
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
        precondition_builder=_hash_current_portfolio_book,
        base_state_hash_fields=("positions",),
    ),
    "update_hedge_positions": DomainAction(
        action_id="update_hedge_positions",
        input_model=UpdateHedgePositionsInput,
        handler=_update_hedge_positions,
        approval_entity_type="hedge_positions",
        approval_payload=_model_payload,
        precondition_builder=_hash_current_hedge_book,
        base_state_hash_fields=("positions",),
    ),
    "change_thesis_status": DomainAction(
        action_id="change_thesis_status",
        input_model=ChangeThesisStatusInput,
        handler=_change_thesis_status,
        approval_entity_type="thesis_status",
        approval_payload=_thesis_status_approval_payload,
        approval_ticker=_ticker_from_model,
        precondition_builder=_hash_current_thesis,
        base_state_hash_fields=("ticker", "content_hash", "meta"),
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
        approval_payload=_model_payload_exclude_unset,
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
        approval_entity_type="action_item_status",
        approval_payload=_model_payload,
        precondition_builder=_hash_action_item_status,
        base_state_hash_fields=("item_id", "status"),
    ),
    "dismiss_action_item": DomainAction(
        action_id="dismiss_action_item",
        input_model=DismissActionItemInput,
        handler=_dismiss_action_item,
        approval_entity_type="action_item_status",
        approval_payload=_model_payload,
        precondition_builder=_hash_action_item_status,
        base_state_hash_fields=("item_id", "status"),
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
        approval_entity_type="watch_trigger_status",
        approval_payload=_model_payload,
        precondition_builder=_hash_watch_trigger_status,
        base_state_hash_fields=("trigger_id", "status"),
    ),
    "cancel_watch_trigger": DomainAction(
        action_id="cancel_watch_trigger",
        input_model=CancelWatchTriggerInput,
        handler=_cancel_watch_trigger,
        approval_entity_type="watch_trigger_status",
        approval_payload=_model_payload,
        precondition_builder=_hash_watch_trigger_status,
        base_state_hash_fields=("trigger_id", "status"),
    ),
    "update_watch_trigger_check": DomainAction(
        action_id="update_watch_trigger_check",
        input_model=UpdateWatchTriggerCheckInput,
        handler=_update_watch_trigger_check,
        approval_entity_type="watch_trigger_check",
        approval_payload=_model_payload,
        precondition_builder=_hash_watch_trigger_status,
        base_state_hash_fields=("trigger_id", "status"),
    ),
    "update_watch_trigger_definition": DomainAction(
        action_id="update_watch_trigger_definition",
        input_model=UpdateWatchTriggerDefinitionInput,
        handler=_update_watch_trigger_definition,
        approval_entity_type="watch_trigger_definition",
        approval_payload=_model_payload,
        precondition_builder=_hash_watch_trigger_status,
        base_state_hash_fields=("trigger_id", "status"),
    ),
    "save_thesis_content": DomainAction(
        action_id="save_thesis_content",
        input_model=SaveThesisContentInput,
        handler=_save_thesis_content,
        approval_entity_type="thesis_content",
        approval_payload=_model_payload,
        approval_ticker=_ticker_from_model,
        effect_kind="approval_gated",
        precondition_builder=_hash_current_thesis,
        base_state_hash_fields=("ticker", "content_hash", "meta"),
    ),
    "save_evaluation": DomainAction(
        action_id="save_evaluation",
        input_model=SaveEvaluationInput,
        handler=_save_evaluation,
        approval_entity_type="evaluation",
        approval_payload=_model_payload,
        approval_ticker=_ticker_from_model,
        effect_kind="approval_gated",
    ),
    "create_research_note": DomainAction(
        action_id="create_research_note",
        input_model=CreateResearchNoteInput,
        handler=_create_research_note,
        approval_entity_type="research_note",
        approval_payload=_model_payload,
        approval_ticker=_ticker_from_model,
        effect_kind="approval_gated",
    ),
    "create_portfolio_news_digest": DomainAction(
        action_id="create_portfolio_news_digest",
        input_model=CreatePortfolioNewsDigestInput,
        handler=_create_portfolio_news_digest,
        approval_entity_type="news_digest_create",
        approval_payload=_model_payload,
        effect_kind="approval_gated",
    ),
    "delete_portfolio_news_digest": DomainAction(
        action_id="delete_portfolio_news_digest",
        input_model=DeletePortfolioNewsDigestInput,
        handler=_delete_portfolio_news_digest,
        approval_entity_type="news_digest_delete",
        approval_payload=_model_payload,
        effect_kind="approval_gated",
    ),
    "create_recommendation": DomainAction(
        action_id="create_recommendation",
        input_model=CreateRecommendationInput,
        handler=_create_recommendation,
        approval_entity_type="recommendation",
        approval_payload=_model_payload,
        effect_kind="approval_gated",
    ),
    "resolve_approval": DomainAction(
        action_id="resolve_approval",
        input_model=ResolveApprovalInput,
        handler=_resolve_approval,
        effect_kind="direct_mutation",
    ),
}


def _normalize_action_definition(action: DomainAction) -> DomainAction:
    approval_spec = action.approval_spec
    if approval_spec is None and action.approval_entity_type:
        approval_spec = ApprovalSpec(
            entity_type=action.approval_entity_type,
            reason_required=action.reason_required,
            payload_builder=action.approval_payload,
            ticker_extractor=action.approval_ticker,
        )
    effect_kind = action.effect_kind or ("approval_gated" if approval_spec is not None else "direct_mutation")
    description = action.description or action.action_id.replace("_", " ")
    execute_actor_types = action.execute_actor_types
    default_execution_mode = action.default_execution_mode
    if effect_kind == "approval_gated":
        execute_actor_types = _APPROVAL_APPLY_ACTORS
        default_execution_mode = default_execution_mode or "approval_required"
    else:
        default_execution_mode = default_execution_mode or "direct"
    return replace(
        action,
        approval_spec=approval_spec,
        effect_kind=effect_kind,
        description=description,
        execute_actor_types=execute_actor_types,
        default_execution_mode=default_execution_mode,
    )


_ACTIONS = {action_id: _normalize_action_definition(action) for action_id, action in _ACTIONS.items()}


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
            "effect_kind": action.effect_kind,
            "description": action.description,
        }
        for action in _ACTIONS.values()
    ]


for _action in _ACTIONS.values():
    register_action_schema_version(_action.action_id, _action.schema_version, _action.input_model)


WorkflowArtifactPayloadAdapter = Callable[[dict[str, Any], str | None], dict[str, Any]]

_GENERIC_TOOL_OUTPUT_SCHEMA: dict[str, Any] = {"type": "object", "additionalProperties": True}
_CONTROL_PLANE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"_meta": {"type": "object", "additionalProperties": True}},
    "additionalProperties": True,
}
_PROPOSAL_TOOL_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "approval_id": {"type": "integer"},
        "entity_type": {"type": "string"},
        "ticker": {"type": "string"},
        "message": {"type": "string"},
    },
    "required": ["status", "approval_id"],
    "additionalProperties": True,
}

_CONTROL_PLANE_TOOL_NAMES = {
    "query_ontology",
    "get_catalysts",
    "get_kill_conditions",
    "get_action_items",
    "get_watch_triggers",
    "get_pending_approvals",
    "get_dossier",
    "get_workflow_history",
    "get_workflow_run",
    "get_workspace",
    "get_research_notes",
    "get_portfolio_news",
    "get_portfolio_positions",
    "get_hedge_positions",
    "get_weekly_report",
    "search_agent_capabilities",
}


@dataclass(frozen=True)
class ToolExposure:
    tool_name: str
    access_mode: ToolAccessMode
    category: str
    description: str
    input_model: type[BaseModel]
    input_schema: dict[str, Any]
    output_spec: OutputSchemaSpec = field(
        default_factory=lambda: OutputSchemaSpec(schema=dict(_GENERIC_TOOL_OUTPUT_SCHEMA), strict=False)
    )
    aliases: tuple[str, ...] = ()
    selectable: bool = True
    agent_exposed: bool = True
    action_id: ActionId | None = None
    to_action_input: ToolActionInputAdapter | None = None
    reason_builder: ToolReasonBuilder | None = None
    entity_id_builder: ToolEntityIdBuilder | None = None
    once: bool = False
    policy_spec: PolicySpec | None = None

    @property
    def parameters(self) -> dict[str, Any]:
        return self.input_schema

    def to_tool_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.tool_name,
            "description": self.description,
            "parameters": self.input_schema,
        }


@dataclass(frozen=True)
class WorkflowArtifactBinding:
    artifact_key: str
    action_id: ActionId
    reason: str
    multiple: bool = False
    required_keys: tuple[str, ...] = ()
    entity_id_field: str | None = None
    payload_adapter: WorkflowArtifactPayloadAdapter | None = None


def _tool_model_name(tool_name: str) -> str:
    return "".join(part.capitalize() for part in tool_name.split("_")) + "ToolInput"


def _schema_annotation(schema: dict[str, Any] | None) -> Any:
    schema = schema or {}
    kind = str(schema.get("type") or "string")
    if kind == "integer":
        return int
    if kind == "number":
        return float
    if kind == "boolean":
        return bool
    if kind == "array":
        return list[Any]
    if kind == "object":
        return dict[str, Any]
    return str


def _input_model_from_schema(tool_name: str, schema: dict[str, Any]) -> type[BaseModel]:
    properties = schema.get("properties") if isinstance(schema, dict) else {}
    required = set(schema.get("required") or []) if isinstance(schema, dict) else set()
    fields: dict[str, tuple[Any, Any]] = {}
    for field_name, field_schema in (properties or {}).items():
        if not isinstance(field_schema, dict):
            field_schema = {}
        annotation = _schema_annotation(field_schema)
        if field_name not in required:
            annotation = annotation | None
            default = None
        else:
            default = ...
        description = field_schema.get("description")
        fields[field_name] = (annotation, Field(default=default, description=description))
    return cast(type[BaseModel], create_model(_tool_model_name(tool_name), **cast(Any, fields)))


def _output_spec_for_tool(tool_name: str, access_mode: ToolAccessMode) -> OutputSchemaSpec:
    if access_mode == "proposal":
        return OutputSchemaSpec(schema=dict(_PROPOSAL_TOOL_OUTPUT_SCHEMA), strict=True)
    if tool_name in _CONTROL_PLANE_TOOL_NAMES:
        return OutputSchemaSpec(schema=dict(_CONTROL_PLANE_OUTPUT_SCHEMA), strict=True)
    return OutputSchemaSpec(schema=dict(_GENERIC_TOOL_OUTPUT_SCHEMA), strict=False)


def _query_ontology_policy_actions(payload: dict[str, Any]) -> tuple[str, ...]:
    actions: list[str] = []
    if payload.get("include_graph"):
        actions.append("graph.read")
    if payload.get("refresh_snapshot"):
        actions.append("snapshot.refresh")
    return tuple(actions)


def _workflow_payload_with_ticker(item: dict[str, Any], ticker: str | None) -> dict[str, Any]:
    payload = dict(item)
    if ticker is not None and not payload.get("ticker"):
        payload["ticker"] = ticker
    return payload


def _workflow_thesis_status_payload(item: dict[str, Any], ticker: str | None) -> dict[str, Any]:
    resolved_ticker = str(item.get("ticker") or ticker or "").strip().upper()
    return {
        "ticker": resolved_ticker,
        "status": item.get("new_status"),
        "reason": str(item.get("reason") or ""),
    }


def _tool_reason(model: BaseModel) -> str | None:
    value = getattr(model, "reason", None)
    if value is None:
        return None
    reason = str(value).strip()
    return reason or None


def _proposal_identity_payload(model: BaseModel) -> dict[str, Any]:
    payload = model.model_dump()
    payload.pop("reason", None)
    return payload


def _proposal_thesis_status_payload(model: BaseModel) -> dict[str, Any]:
    payload = model.model_dump()
    return {"ticker": payload["ticker"], "status": payload["new_status"], "reason": payload["reason"]}


def _proposal_action_item_payload(model: BaseModel) -> dict[str, Any]:
    payload = model.model_dump(exclude_none=True)
    payload.pop("reason", None)
    return payload


def _proposal_catalyst_status_payload(model: BaseModel) -> dict[str, Any]:
    payload = model.model_dump()
    return {
        "ticker": payload["ticker"],
        "catalyst_id": payload["catalyst_id"],
        "status": payload["new_status"],
        "evidence": payload.get("evidence"),
    }


def _proposal_kill_condition_status_payload(model: BaseModel) -> dict[str, Any]:
    payload = model.model_dump()
    return {
        "ticker": payload["ticker"],
        "kill_condition_id": payload["kill_condition_id"],
        "status": payload["new_status"],
    }


def _proposal_watch_trigger_payload(model: BaseModel) -> dict[str, Any]:
    payload = model.model_dump(exclude_none=True)
    payload.pop("reason", None)
    return payload


def _proposal_positions_payload(model: BaseModel) -> dict[str, Any]:
    payload = model.model_dump()
    return {"positions": payload["positions"]}


def _proposal_thesis_content_payload(model: BaseModel) -> dict[str, Any]:
    payload = model.model_dump()
    return {"ticker": payload["ticker"], "content": payload["content"]}


def _proposal_catalyst_payload(model: BaseModel) -> dict[str, Any]:
    payload = model.model_dump(exclude_none=True)
    payload.pop("reason", None)
    return payload


def _proposal_kill_condition_payload(model: BaseModel) -> dict[str, Any]:
    payload = model.model_dump()
    payload.pop("reason", None)
    return payload


def _proposal_research_note_payload(model: BaseModel) -> dict[str, Any]:
    payload = model.model_dump(exclude_none=True)
    payload.pop("reason", None)
    return payload


def _proposal_news_digest_delete_payload(model: BaseModel) -> dict[str, Any]:
    from portfolio.news_digests import get_digest

    payload = model.model_dump()
    digest_id = str(payload["digest_id"])
    try:
        get_digest(digest_id)
    except FileNotFoundError as exc:
        raise ActionValidationError(f"Unknown news digest id: {digest_id}") from exc
    except ValueError as exc:
        raise ActionValidationError(str(exc) or "Invalid news digest id") from exc
    return {"digest_id": payload["digest_id"]}


_PROPOSAL_TOOL_BINDINGS: dict[str, dict[str, Any]] = {
    "propose_thesis_status_change": {"action_id": "change_thesis_status", "adapter": _proposal_thesis_status_payload},
    "propose_action_item": {"action_id": "create_action_item", "adapter": _proposal_action_item_payload},
    "propose_catalyst_status_change": {
        "action_id": "update_catalyst_status",
        "adapter": _proposal_catalyst_status_payload,
        "entity_id": lambda model: int(model.catalyst_id),
    },
    "propose_kill_condition_status_change": {
        "action_id": "update_kill_condition_status",
        "adapter": _proposal_kill_condition_status_payload,
        "entity_id": lambda model: int(model.kill_condition_id),
    },
    "propose_watch_trigger": {"action_id": "create_watch_trigger", "adapter": _proposal_watch_trigger_payload},
    "propose_portfolio_positions_update": {
        "action_id": "update_portfolio_positions",
        "adapter": _proposal_positions_payload,
    },
    "propose_hedge_positions_update": {"action_id": "update_hedge_positions", "adapter": _proposal_positions_payload},
    "propose_thesis_content_update": {"action_id": "save_thesis_content", "adapter": _proposal_thesis_content_payload},
    "propose_catalyst": {"action_id": "create_catalyst", "adapter": _proposal_catalyst_payload},
    "propose_kill_condition": {"action_id": "create_kill_condition", "adapter": _proposal_kill_condition_payload},
    "propose_research_note": {"action_id": "create_research_note", "adapter": _proposal_research_note_payload},
    "propose_news_digest_delete": {
        "action_id": "delete_portfolio_news_digest",
        "adapter": _proposal_news_digest_delete_payload,
    },
}

_WORKFLOW_ARTIFACT_BINDINGS: dict[str, WorkflowArtifactBinding] = {
    "evaluation_draft": WorkflowArtifactBinding(
        artifact_key="evaluation_draft",
        action_id="save_evaluation",
        reason="Workflow-generated evaluation",
        required_keys=("thesis_status",),
        payload_adapter=_workflow_payload_with_ticker,
    ),
    "action_items": WorkflowArtifactBinding(
        artifact_key="action_items",
        action_id="create_action_item",
        reason="Workflow-generated action item",
        multiple=True,
        required_keys=("description",),
        payload_adapter=_workflow_payload_with_ticker,
    ),
    "watch_triggers": WorkflowArtifactBinding(
        artifact_key="watch_triggers",
        action_id="create_watch_trigger",
        reason="Workflow-generated watch trigger",
        multiple=True,
        required_keys=("condition",),
        payload_adapter=_workflow_payload_with_ticker,
    ),
    "catalyst_updates": WorkflowArtifactBinding(
        artifact_key="catalyst_updates",
        action_id="update_catalyst_status",
        reason="Workflow-suggested catalyst status change",
        multiple=True,
        entity_id_field="catalyst_id",
        payload_adapter=_workflow_payload_with_ticker,
    ),
    "kill_condition_updates": WorkflowArtifactBinding(
        artifact_key="kill_condition_updates",
        action_id="update_kill_condition_status",
        reason="Workflow-suggested kill condition status change",
        multiple=True,
        entity_id_field="kill_condition_id",
        payload_adapter=_workflow_payload_with_ticker,
    ),
    "thesis_status_change": WorkflowArtifactBinding(
        artifact_key="thesis_status_change",
        action_id="change_thesis_status",
        reason="Workflow-suggested thesis status change",
        required_keys=("new_status",),
        payload_adapter=_workflow_thesis_status_payload,
    ),
    "research_notes": WorkflowArtifactBinding(
        artifact_key="research_notes",
        action_id="create_research_note",
        reason="Workflow-generated research note",
        multiple=True,
        required_keys=("title", "content"),
        payload_adapter=_workflow_payload_with_ticker,
    ),
    "news_digest_deletes": WorkflowArtifactBinding(
        artifact_key="news_digest_deletes",
        action_id="delete_portfolio_news_digest",
        reason="Workflow-suggested news digest delete",
        multiple=True,
        required_keys=("digest_id",),
    ),
}

_TOOL_EXPOSURE_SPECS_TEXT = r"""
{'name': 'get_liquidity', 'description': 'Fetch the global liquidity dashboard. Returns a composite liquidity score, regime (ample/normal/tight/stress), regional scores per major economy, individual component z-scores and contributions, and 1W/1M/3M changes. Use this to assess whether the global liquidity backdrop supports or hinders risk assets.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'macro', 'access_mode': 'read', 'aliases': ('liquidity', 'global liquidity', 'credit', 'get liquidity'), 'selectable': True}
{'name': 'get_market_breadth', 'description': 'Fetch S&P 500 market breadth data. Returns the percentage and count of stocks above their 200-day and 20-day moving averages, at 20-day / 52-week / 24-week highs and lows, and total analyzed. Use this to assess market participation and whether rallies or selloffs are broad-based or narrow.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'technical', 'access_mode': 'read', 'aliases': ('breadth', 'market breadth', 'participation', 'get market breadth'), 'selectable': True}
{'name': 'get_vix_term_structure', 'description': 'Fetch VIX term structure data. Returns the latest VIX, VIX3M (3-month VIX), the 3M/1M ratio, and a signal (Fear when ratio < 1.0, Complacency when > 1.25, else Neutral). Also includes recent ratio history and signal hit dates. Use this to gauge near-term vs longer-term volatility expectations.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'technical', 'access_mode': 'read', 'aliases': ('vix', 'volatility', 'term structure', 'get vix term structure'), 'selectable': True}
{'name': 'get_positioning', 'description': 'Fetch CFTC Commitments of Traders (COT) leveraged fund positioning data. Returns net % of open interest, positioning z-scores, deleveraging z-scores, and forced flow signals (long liquidation / short covering) for each instrument. Use this to assess crowded positions and squeeze risk.', 'parameters': {'type': 'object', 'properties': {'instruments': {'type': 'string', 'description': "Comma-separated instrument aliases. Available: SP500, NASDAQ, RUSSELL, US10Y, EUR, JPY, GBP, AUD, CAD, GOLD, OIL. Default: 'SP500,NASDAQ,RUSSELL,US10Y,EUR'"}}, 'required': []}, 'category': 'macro', 'access_mode': 'read', 'aliases': ('positioning', 'cftc', 'cot', 'crowded', 'get positioning'), 'selectable': True}
{'name': 'get_signal_aggregator', 'description': 'Fetch a unified cross-module market signal dashboard that combines VIX term structure, market breadth, liquidity, CFTC positioning, sector metrics, and momentum into a deterministic regime signal. Returns current regime label (risk-on/transitional/risk-off), factor scores, effective weights, failed modules, and historical weekly regime tracking with episodes.', 'parameters': {'type': 'object', 'properties': {'lookback_weeks': {'type': 'integer', 'description': 'Weekly history length for regime tracking. Default: 156 (about 3 years).'}, 'positioning_instruments': {'type': 'string', 'description': "Comma-separated CFTC instrument aliases for positioning input. Default: 'SP500,NASDAQ,RUSSELL,US10Y,EUR'."}, 'include_history': {'type': 'boolean', 'description': 'Include weekly historical regime series. Default false for faster chat responses.'}}, 'required': []}, 'category': 'macro', 'access_mode': 'read', 'aliases': ('signal aggregator', 'regime', 'risk on', 'risk off', 'get signal aggregator'), 'selectable': True}
{'name': 'get_economic_growth', 'description': 'Fetch cross-asset returns for growth regime assessment. Returns period returns (1D, 1W, 1M, 3M, 6M, YTD) for commodities (copper, oil, gold, CRB), equities (S&P 500, Russell 2000, transports, banks), and currency pairs. Use this to identify growth cycle signals from market prices.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'macro', 'access_mode': 'read', 'aliases': ('growth', 'economic growth', 'cross asset', 'get economic growth'), 'selectable': True}
{'name': 'get_labor_market', 'description': 'Fetch US labor market indicators. Returns time series and latest values for initial claims, continuing claims, unemployment rate, nonfarm payrolls, and wage growth. Use this to assess labor market tightness and recession risk.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'macro', 'access_mode': 'read', 'aliases': ('labor', 'jobs', 'claims', 'payrolls', 'get labor market'), 'selectable': True}
{'name': 'get_housing', 'description': 'Fetch US housing market indicators. Returns time series and latest values for housing starts, building permits, NAHB housing market index, and existing home sales. Use this to assess the residential construction cycle, builder sentiment, and housing demand.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'macro', 'access_mode': 'read', 'aliases': ('housing', 'starts', 'permits', 'nahb', 'get housing'), 'selectable': True}
{'name': 'get_sector_metrics', 'description': 'Fetch S&P 500 sector metrics. Returns sector weights, weight changes over 1M/3M/6M, relative performance vs SPY, and percentage of sector constituents above their 200-day moving average. Use this to identify sector rotation, concentration risk, and leadership changes.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'equities', 'access_mode': 'read', 'aliases': ('sector', 'rotation', 'sector metrics', 'get sector metrics'), 'selectable': True}
{'name': 'get_portfolio', 'description': "Fetch the user's portfolio dashboard. Returns current positions with their direction, cost basis, share quantity, conviction, P&L, contribution, and price data. Portfolio performance fields are direction-adjusted: price declines are favorable for short positions. Never judge a position from raw price moves alone; combine direction, quantity, cost basis, conviction, and P&L/return fields. Use this when the user asks about their portfolio, holdings, performance, or any specific position. Pair with get_thesis for investment reasoning context.", 'parameters': {'type': 'object', 'properties': {'timeframe': {'type': 'string', 'description': "Period for returns. Options: 'This Week', 'Daily', 'Weekly', 'Monthly'. Default: 'Daily'."}, 'include_hedges': {'type': 'boolean', 'description': 'Include hedge rows. Default false; use only for hedge, beta, or risk-exposure questions.'}}, 'required': []}, 'category': 'portfolio', 'access_mode': 'read', 'aliases': ('portfolio', 'holdings', 'positions', 'pnl', 'p&l', 'get portfolio'), 'selectable': True}
{'name': 'get_yield_curve', 'description': 'Fetch government bond yield curve data for the US, Germany, UK, and Japan. Returns current yields across tenors (3M through 30Y) and comparison vs a lookback period. Use this to assess yield curve shape, inversions, and changes in rate expectations.', 'parameters': {'type': 'object', 'properties': {'lookback_days': {'type': 'integer', 'description': 'Number of days to look back for comparison. Default: 90.'}}, 'required': []}, 'category': 'fixed_income', 'access_mode': 'read', 'aliases': ('yield curve', 'rates', 'bonds', 'get yield curve'), 'selectable': True}
{'name': 'get_bond_dashboard', 'description': 'Fetch government bond yield time-series for 2Y, 10Y, and 30Y tenors across US, UK, Germany, and Japan. Returns the past year of daily yields, latest values, and year-over-year changes in basis points per country and tenor. Use this to compare sovereign yield levels and trends across major economies.', 'parameters': {'type': 'object', 'properties': {'tenor': {'type': 'string', 'description': "Filter to a single tenor: '2Y', '10Y', or '30Y'. Default: return all tenors."}}, 'required': []}, 'category': 'fixed_income', 'access_mode': 'read', 'aliases': ('bond dashboard', 'sovereign yields', 'get bond dashboard'), 'selectable': True}
{'name': 'get_sentiment', 'description': 'Fetch market sentiment indicators. Returns put/call ratios (equity aggregate,SPY, QQQ, IWM), investor surveys (AAII bull/bear spread, NAAIM exposure index), and volatility indices (VIX, VXN, VVIX). Includes quality checks and latest-date validation metadata. If quality.ok is false, do not draw directional sentiment conclusions and treat sentiment as unavailable for this turn.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'macro', 'access_mode': 'read', 'aliases': ('sentiment', 'put call', 'aaii', 'naaim', 'get sentiment'), 'selectable': True}
{'name': 'get_central_banks', 'description': 'Fetch central bank news and recent publications. Returns articles and documents from the Fed, ECB, BoE, BoJ, SNB, RBA, and other major central banks, grouped by source with counts. Use this to check for recent policy signals or speeches.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'macro', 'access_mode': 'read', 'aliases': ('central banks', 'fed', 'ecb', 'boe', 'boj', 'get central banks'), 'selectable': True}
{'name': 'get_industry_monitor', 'description': 'Fetch what businesses and companies are actually saying from their earnings call transcripts. Covers leading (housing, trucking), coincident (banks, retail), and lagging (capital goods) industry sectors. Returns per-company sentiment (bullish/neutral/bearish), demand trends, pricing commentary, guidance outlook, macro quotes, and sector-level economic signals (expanding/stable/slowing/contracting). Use this when the user asks what businesses, companies, or management teams are saying about the economy, demand, or business conditions.', 'parameters': {'type': 'object', 'properties': {'refresh': {'type': 'boolean', 'description': 'If true, bypass cached data and recompute from source files. Default: false.'}}, 'required': []}, 'category': 'macro', 'access_mode': 'read', 'aliases': ('industry monitor', 'transcripts', 'management commentary', 'get industry monitor'), 'selectable': True}
{'name': 'get_breakout', 'description': 'Fetch macro breakout signals across asset classes. Returns recent breakouts with direction (up/down), date, market, and close price. Use this to identify which major markets are making significant technical moves.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'technical', 'access_mode': 'read', 'aliases': ('breakout', 'technical breakouts', 'get breakout'), 'selectable': True}
{'name': 'query_ontology', 'description': 'Run a cross-module ontology query that joins portfolio positions with macro/market signals (VIX, breadth, sector stress, liquidity, and other read-only data modules). Returns per-position risk scores with evidence. Use this when users ask about portfolio risk exposure, positions in deteriorating conditions, or entity-level context. Pair with get_thesis for the investment reasoning behind positions.', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string', 'description': "Natural-language query, e.g. 'Which positions are in deteriorating macro conditions?'"}, 'intent': {'type': 'string', 'description': 'Optional explicit intent. Allowed: portfolio_risk_exposure, positions_in_deteriorating_macro, entity_context.'}, 'filters': {'type': 'object', 'description': 'Optional filters: tickers, sectors, assets, min_risk_score.'}, 'page': {'type': 'integer', 'description': 'Optional 1-based results page. Defaults to 1.'}, 'page_size': {'type': 'integer', 'description': 'Optional page size from 1 to 100. Defaults to 25.'}, 'timeframe': {'type': 'string', 'description': 'Timeframe for portfolio-linked data. Options: This Week, Daily, Weekly, Monthly.'}, 'include_graph': {'type': 'boolean', 'description': 'If true, include ontology nodes and edges in output.'}, 'run_id': {'type': 'string', 'description': 'Optional ontology snapshot run_id for historical replay.'}, 'refresh_snapshot': {'type': 'boolean', 'description': 'If true, bypass latest snapshot reuse and force a fresh ontology snapshot build.'}}, 'required': []}, 'category': 'ontology', 'access_mode': 'read', 'aliases': ('ontology', 'risk exposure', 'portfolio risk', 'query ontology'), 'selectable': True}
{'name': 'get_thesis', 'description': "Fetch the investment thesis for a specific ticker. Returns the thesis markdown content (thesis statement, key catalysts, risk factors) and metadata (status, creation date, last update). Use this when the user asks about a position's investment reasoning, thesis, catalysts, kill conditions, or why they own something. Also useful for thesis pressure-tests and reviews.", 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string', 'description': "Ticker symbol (e.g. 'CRWD', 'AAPL'). Case-insensitive."}}, 'required': ['ticker']}, 'category': 'thesis', 'access_mode': 'read', 'aliases': ('thesis', 'investment thesis', 'get thesis'), 'selectable': True}
{'name': 'get_thesis_evaluations', 'description': "Fetch the monitoring evaluation history for a specific ticker's thesis. Returns weekly evaluations (thesis status, technical read, fundamental read, recommended action, confidence, key developments, earnings notes, risk flags) and status change history. Use this to understand how a thesis has evolved over time, whether conviction has increased or decreased, and what developments have occurred since the thesis was written.", 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string', 'description': "Ticker symbol (e.g. 'CRWD', 'AAPL'). Case-insensitive."}, 'limit': {'type': 'integer', 'description': 'Maximum number of evaluations to return (most recent first). Default: 10.'}}, 'required': ['ticker']}, 'category': 'thesis', 'access_mode': 'read', 'aliases': ('thesis evaluations', 'monitoring history', 'get thesis evaluations'), 'selectable': True}
{'name': 'search_knowledge_base', 'description': "Search across all indexed research documents — investment theses, uploaded news digests, weekly reports, daily reports, and past conversation summaries — using semantic similarity. Use this when the user asks what they wrote about a topic, references past research, wants to find previous analysis on a ticker or theme, or asks 'what did I say about X'. Returns ranked snippets with source attribution.", 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string', 'description': "Natural language search query (e.g. 'cloud security thesis for CRWD', 'liquidity tightening analysis')."}, 'doc_types': {'type': 'string', 'description': 'Comma-separated document types to search. Options: thesis, news_digest, weekly_report, daily_report, conversation_summary. Leave empty to search all.'}, 'tickers': {'type': 'string', 'description': "Comma-separated ticker filter (e.g. 'CRWD,AAPL'). Leave empty for all."}, 'top_k': {'type': 'integer', 'description': 'Number of results to return. Default: 5.'}}, 'required': ['query']}, 'category': 'research', 'access_mode': 'read', 'aliases': ('knowledge base', 'past research', 'notes', 'news digests', 'search knowledge base'), 'selectable': True}
{'name': 'get_ontology_diff', 'description': "Compare two ontology snapshots to show what changed in the portfolio's risk profile. Returns new/removed positions, risk score deltas, signal transitions (stable→deteriorating), and component score changes. Use this when the user asks 'what changed', 'how has my risk changed', 'what's different since last week', or any temporal comparison of portfolio risk.", 'parameters': {'type': 'object', 'properties': {'run_id_before': {'type': 'string', 'description': 'The older snapshot run_id to compare from. Leave empty to auto-select the most recent prior snapshot.'}, 'run_id_after': {'type': 'string', 'description': 'The newer snapshot run_id to compare to. Leave empty to use the latest/current snapshot.'}}, 'required': []}, 'category': 'ontology', 'access_mode': 'read', 'aliases': ('ontology diff', 'risk changes', 'get ontology diff'), 'selectable': True}
{'name': 'search_web', 'description': 'Search the web for recent news, events, or developments related to a ticker, company, sector, or macro topic. Uses trusted financial news sources (Bloomberg, CNBC, Reuters, WSJ, FT, etc.). Returns a summary of findings with source citations. Use this to verify catalyst status, check for breaking news, confirm regulatory actions, or validate thesis assumptions against real-world events.', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string', 'description': "Search query. Be specific — include ticker, company name, and what you're looking for."}}, 'required': ['query']}, 'category': 'research', 'access_mode': 'read', 'aliases': ('web', 'news', 'latest', 'recent', 'search web'), 'selectable': True}
{'name': 'get_catalysts', 'description': 'Fetch tracked catalysts for a given ticker. Returns a list of catalysts with their status (pending/played_out/failed/superseded), category, target date, and evidence.', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string', 'description': "Ticker symbol (e.g. 'AAPL')"}}, 'required': ['ticker']}, 'category': 'process', 'access_mode': 'read', 'aliases': ('catalysts', 'get catalysts'), 'selectable': True}
{'name': 'get_kill_conditions', 'description': 'Fetch kill conditions for a given ticker. Returns conditions that would invalidate the thesis, with status (active/triggered/retired), metric, and threshold.', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string', 'description': "Ticker symbol (e.g. 'AAPL')"}}, 'required': ['ticker']}, 'category': 'process', 'access_mode': 'read', 'aliases': ('kill conditions', 'invalidation', 'get kill conditions'), 'selectable': True}
{'name': 'get_action_items', 'description': 'Fetch open action items, optionally filtered by ticker. Returns tasks with urgency (low/normal/high/urgent), action type, and status.', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string', 'description': 'Optional ticker filter'}, 'status': {'type': 'string', 'description': "Filter by status. Default: 'open'"}}, 'required': []}, 'category': 'process', 'access_mode': 'read', 'aliases': ('action items', 'tasks', 'get action items'), 'selectable': True}
{'name': 'get_watch_triggers', 'description': 'Fetch active watch triggers, optionally filtered by ticker. Returns conditions the system is monitoring with trigger type and status.', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string', 'description': 'Optional ticker filter'}, 'status': {'type': 'string', 'description': "Filter by status. Default: 'active'"}}, 'required': []}, 'category': 'process', 'access_mode': 'read', 'aliases': ('watch triggers', 'monitoring', 'get watch triggers'), 'selectable': True}
{'name': 'get_pending_approvals', 'description': 'Fetch pending approval items. These are proposed changes from workflows or agent that require user approval before being applied.', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string', 'description': 'Optional ticker filter'}, 'status': {'type': 'string', 'description': "Filter by status. Default: 'pending'"}}, 'required': []}, 'category': 'approvals', 'access_mode': 'read', 'aliases': ('approvals', 'pending approvals', 'get pending approvals'), 'selectable': True}
{'name': 'get_dossier', 'description': 'Fetch the complete position dossier for a ticker. Returns thesis, catalysts, kill conditions, evaluations, ontology risk, workflow runs, action items, triggers, research notes, and pending approvals — all in one call.', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string', 'description': "Ticker symbol (e.g. 'MU')"}}, 'required': ['ticker']}, 'category': 'portfolio', 'access_mode': 'read', 'aliases': ('dossier', 'position dossier', 'get dossier'), 'selectable': True}
{'name': 'get_workflow_history', 'description': 'Fetch recent workflow run history, optionally filtered by workflow name or ticker.', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string', 'description': 'Optional ticker filter'}, 'workflow_name': {'type': 'string', 'description': 'Optional workflow name filter'}, 'limit': {'type': 'integer', 'description': 'Max results (default 10)'}}, 'required': []}, 'category': 'workflows', 'access_mode': 'read', 'aliases': ('workflow history', 'workflow runs', 'get workflow history'), 'selectable': True}
{'name': 'propose_thesis_status_change', 'description': 'Propose a thesis status change for a ticker. This creates a pending approval that the user must approve before the status is actually changed. Use this instead of directly modifying thesis status.', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string', 'description': 'Ticker symbol'}, 'new_status': {'type': 'string', 'description': 'Proposed new status: active|under_review|invalidated'}, 'reason': {'type': 'string', 'description': 'Explanation for the proposed change'}}, 'required': ['ticker', 'new_status', 'reason']}, 'category': 'thesis', 'access_mode': 'proposal', 'aliases': ('propose thesis status', 'thesis status', 'propose thesis status change'), 'selectable': True}
{'name': 'propose_action_item', 'description': 'Propose a new action item. This creates a pending approval that the user must approve before the action item is created. Use this for recommending trades, research tasks, or position adjustments.', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string', 'description': 'Ticker symbol (optional for non-ticker-specific actions)'}, 'description': {'type': 'string', 'description': 'What needs to be done'}, 'action_type': {'type': 'string', 'description': 'Type: review|resize|research|exit|enter|hedge|other'}, 'urgency': {'type': 'string', 'description': 'Urgency: low|normal|high|urgent'}, 'reason': {'type': 'string', 'description': 'Why this action is recommended'}}, 'required': ['description', 'action_type', 'reason']}, 'category': 'process', 'access_mode': 'proposal', 'aliases': ('propose action', 'action item', 'propose action item'), 'selectable': True}
{'name': 'propose_catalyst_status_change', 'description': 'Propose a catalyst status change. This creates a pending approval that the user must approve before the catalyst status is actually updated. Use this when evidence suggests a catalyst has played out, failed, or been superseded.', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string', 'description': 'Ticker symbol'}, 'catalyst_id': {'type': 'integer', 'description': 'ID of the catalyst to update'}, 'new_status': {'type': 'string', 'description': 'Proposed new status: pending|played_out|failed|superseded'}, 'evidence': {'type': 'string', 'description': 'Evidence supporting the status change'}, 'reason': {'type': 'string', 'description': 'Explanation for the proposed change'}}, 'required': ['ticker', 'catalyst_id', 'new_status', 'reason']}, 'category': 'process', 'access_mode': 'proposal', 'aliases': ('propose catalyst status', 'propose catalyst status change'), 'selectable': True}
{'name': 'propose_kill_condition_status_change', 'description': 'Propose a kill condition status change. This creates a pending approval that the user must approve before the kill condition status is actually updated. Use this when a kill condition has been triggered or should be retired.', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string', 'description': 'Ticker symbol'}, 'kill_condition_id': {'type': 'integer', 'description': 'ID of the kill condition to update'}, 'new_status': {'type': 'string', 'description': 'Proposed new status: active|triggered|retired'}, 'reason': {'type': 'string', 'description': 'Explanation for the proposed change'}}, 'required': ['ticker', 'kill_condition_id', 'new_status', 'reason']}, 'category': 'process', 'access_mode': 'proposal', 'aliases': ('propose kill condition status', 'propose kill condition status change'), 'selectable': True}
{'name': 'propose_watch_trigger', 'description': 'Propose a new watch trigger. This creates a pending approval that the user must approve before the trigger is activated. Use this to set up monitoring conditions.', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string', 'description': 'Ticker symbol (optional)'}, 'condition': {'type': 'string', 'description': "The condition to watch for (e.g. 'AAPL breaks below $180')"}, 'trigger_type': {'type': 'string', 'description': 'Type: price_level|technical|fundamental|fundamental_news|event|news_event|macro|custom'}, 'expires_at': {'type': 'string', 'description': 'Optional ISO timestamp when the trigger expires.'}, 'definition': {'type': 'object', 'description': 'Optional machine-readable executable trigger definition.'}, 'reason': {'type': 'string', 'description': 'Why this trigger matters'}}, 'required': ['condition', 'trigger_type', 'reason']}, 'category': 'process', 'access_mode': 'proposal', 'aliases': ('propose watch trigger', 'set trigger', 'propose watch trigger'), 'selectable': True}
{'name': 'search_agent_capabilities', 'description': "Search Stan's available app capabilities by natural-language query. Use when you need a tool that was not in the initially visible set.", 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string', 'description': 'Capability or app feature to find.'}, 'top_k': {'type': 'integer', 'description': 'Maximum matches to return. Default 8.'}}, 'required': ['query']}, 'category': 'agent', 'access_mode': 'read', 'aliases': ('capability search', 'available tools', 'what can you access'), 'selectable': True}
{'name': 'get_workspace', 'description': 'Fetch the Workspace landing page aggregate: regime, portfolio summary, thesis pressure, approvals, action items, triggers, and workflow runs.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'workspace', 'access_mode': 'read', 'aliases': ('workspace', 'dashboard home'), 'selectable': True}
{'name': 'get_portfolio_positions', 'description': 'Fetch editable portfolio positions, optionally including hedges.', 'parameters': {'type': 'object', 'properties': {'include_hedges': {'type': 'boolean'}}, 'required': []}, 'category': 'portfolio', 'access_mode': 'read', 'aliases': ('portfolio positions', 'editable holdings'), 'selectable': True}
{'name': 'get_hedge_positions', 'description': 'Fetch hedge positions from the portfolio editor.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'portfolio', 'access_mode': 'read', 'aliases': ('hedge positions', 'hedges'), 'selectable': True}
{'name': 'get_portfolio_news', 'description': 'List uploaded news digests, or fetch one digest when digest_id is provided.', 'parameters': {'type': 'object', 'properties': {'digest_id': {'type': 'string', 'description': 'Optional digest id for detail.'}}, 'required': []}, 'category': 'research', 'access_mode': 'read', 'aliases': ('news digests', 'portfolio news', 'uploaded news'), 'selectable': True}
{'name': 'get_research_notes', 'description': 'Fetch research notes, optionally filtered by ticker.', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string'}, 'limit': {'type': 'integer'}}, 'required': []}, 'category': 'research', 'access_mode': 'read', 'aliases': ('research notes', 'notes'), 'selectable': True}
{'name': 'get_workflow_run', 'description': 'Fetch one persisted workflow run by run_id.', 'parameters': {'type': 'object', 'properties': {'run_id': {'type': 'string'}}, 'required': ['run_id']}, 'category': 'workflows', 'access_mode': 'read', 'aliases': ('workflow run detail', 'run detail'), 'selectable': True}
{'name': 'get_weekly_report', 'description': 'Fetch the weekly report payload.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'reports', 'access_mode': 'read', 'aliases': ('weekly report', 'weekly'), 'selectable': True}
{'name': 'get_commodities', 'description': 'Fetch the commodity dashboard across major commodities for a timeframe.', 'parameters': {'type': 'object', 'properties': {'timeframe': {'type': 'string', 'description': 'This Week, Daily, Weekly, or Monthly. Default Daily.'}}, 'required': []}, 'category': 'commodities', 'access_mode': 'read', 'aliases': ('commodities dashboard', 'commodity prices'), 'selectable': True}
{'name': 'get_commodities_curve', 'description': 'Fetch futures curve data for CL, BZ, NG, or TTF.', 'parameters': {'type': 'object', 'properties': {'commodity': {'type': 'string'}, 'lookback_days': {'type': 'integer'}}, 'required': []}, 'category': 'commodities', 'access_mode': 'read', 'aliases': ('commodities curve', 'oil curve', 'gas curve', 'futures curve'), 'selectable': True}
{'name': 'get_commodity_research', 'description': 'Fetch the commodity proxy research screener.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'commodities', 'access_mode': 'read', 'aliases': ('commodity research', 'commodity proxy', 'aluminum research'), 'selectable': True}
{'name': 'get_country_dashboard', 'description': 'Fetch the country dashboard.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'macro', 'access_mode': 'read', 'aliases': ('country dashboard', 'countries'), 'selectable': True}
{'name': 'get_index_dashboard', 'description': 'Fetch the index dashboard.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'equities', 'access_mode': 'read', 'aliases': ('index dashboard', 'indices'), 'selectable': True}
{'name': 'get_fx_dashboard', 'description': 'Fetch the FX dashboard.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'fx', 'access_mode': 'read', 'aliases': ('fx dashboard', 'currencies'), 'selectable': True}
{'name': 'get_momentum', 'description': 'Fetch price momentum dashboard data.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'portfolio', 'access_mode': 'read', 'aliases': ('momentum', 'price momentum'), 'selectable': True}
{'name': 'get_top50_breadth', 'description': 'Fetch S&P 500 top-50 breadth data.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'technical', 'access_mode': 'read', 'aliases': ('top50 breadth', 'top 50 breadth'), 'selectable': True}
{'name': 'get_price_volume_signals', 'description': 'Fetch price-volume technical signals.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'technical', 'access_mode': 'read', 'aliases': ('price volume', 'volume signals'), 'selectable': True}
{'name': 'get_financials', 'description': 'Fetch single-company financial history and metrics.', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string'}}, 'required': ['ticker']}, 'category': 'equities', 'access_mode': 'read', 'aliases': ('financials', 'company financials', 'revenue', 'eps'), 'selectable': True}
{'name': 'get_dcf_historical', 'description': 'Fetch historical financials and multiples for DCF work.', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string'}}, 'required': ['ticker']}, 'category': 'equities', 'access_mode': 'read', 'aliases': ('dcf historical', 'valuation historical'), 'selectable': True}
{'name': 'run_dcf_valuation', 'description': 'Run a DCF valuation from explicit assumptions.', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string'}, 'revenue_growth_rates': {'type': 'array', 'items': {'type': 'number'}, 'description': 'Five annual revenue growth rates as decimals.'}, 'ebitda_margin': {'type': 'number'}, 'tax_rate': {'type': 'number'}, 'da_pct_revenue': {'type': 'number'}, 'nwc_pct_revenue': {'type': 'number'}, 'capex_pct_revenue': {'type': 'number'}, 'wacc': {'type': 'number'}, 'terminal_growth_rates': {'type': 'object'}, 'exit_ev_ebitda': {'type': 'object'}, 'exit_ev_revenue': {'type': 'object'}}, 'required': ['ticker', 'revenue_growth_rates', 'ebitda_margin', 'da_pct_revenue', 'nwc_pct_revenue', 'capex_pct_revenue', 'wacc', 'exit_ev_ebitda', 'exit_ev_revenue']}, 'category': 'equities', 'access_mode': 'compute', 'aliases': ('run dcf', 'dcf valuation', 'valuation'), 'selectable': True}
{'name': 'run_chart', 'description': 'Run technical analysis for a ticker.', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string'}, 'lookback': {'type': 'string'}}, 'required': ['ticker']}, 'category': 'technical', 'access_mode': 'compute', 'aliases': ('chart', 'technical analysis'), 'selectable': True}
{'name': 'run_ratio_chart', 'description': 'Run a ratio chart between two symbols.', 'parameters': {'type': 'object', 'properties': {'symbol_a': {'type': 'string'}, 'symbol_b': {'type': 'string'}, 'start_date': {'type': 'string'}, 'end_date': {'type': 'string'}, 'method': {'type': 'string'}}, 'required': ['symbol_a', 'symbol_b']}, 'category': 'technical', 'access_mode': 'compute', 'aliases': ('ratio chart', 'pair ratio'), 'selectable': True}
{'name': 'get_fx_model_pairs', 'description': 'List supported FX model pairs.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'fx', 'access_mode': 'read', 'aliases': ('fx model pairs', 'currency pairs'), 'selectable': True}
{'name': 'run_fx_model', 'description': 'Run the FX valuation/forecast model for a supported pair.', 'parameters': {'type': 'object', 'properties': {'pair': {'type': 'string'}, 'bootstrap': {'type': 'integer'}, 'skip_bis': {'type': 'boolean'}, 'horizons': {'type': 'string'}}, 'required': ['pair']}, 'category': 'fx', 'access_mode': 'compute', 'aliases': ('fx model', 'currency model'), 'selectable': True}
{'name': 'run_quality_screen', 'description': 'Run the quality equity screen.', 'parameters': {'type': 'object', 'properties': {'universe': {'type': 'string'}, 'tickers': {'type': 'string'}, 'benchmark': {'type': 'string'}, 'input_mode': {'type': 'string'}}, 'required': []}, 'category': 'screeners', 'access_mode': 'compute', 'aliases': ('quality screen', 'quality screener'), 'selectable': True}
{'name': 'run_short_screen', 'description': 'Start or reuse a short screen job.', 'parameters': {'type': 'object', 'properties': {'input_mode': {'type': 'string'}, 'universe': {'type': 'string'}, 'tickers': {'type': 'string'}, 'pb_threshold': {'type': 'number'}, 'loss_type': {'type': 'string'}, 'check_issuance': {'type': 'boolean'}, 'check_revenue': {'type': 'boolean'}, 'max_revenue_growth': {'type': 'number'}, 'check_eps': {'type': 'boolean'}, 'max_eps_growth': {'type': 'number'}, 'check_52w_positive': {'type': 'boolean'}, 'check_min_drawdown': {'type': 'boolean'}, 'min_drawdown_pct': {'type': 'number'}, 'check_max_drawdown': {'type': 'boolean'}, 'max_drawdown_pct': {'type': 'number'}, 'check_3m_neg_momentum': {'type': 'boolean'}, 'check_2m_neg_rel_momentum': {'type': 'boolean'}, 'rel_momentum_benchmark': {'type': 'string'}}, 'required': []}, 'category': 'screeners', 'access_mode': 'compute', 'aliases': ('short screen', 'short screener'), 'selectable': True}
{'name': 'run_long_screen', 'description': 'Start or reuse a long screen job.', 'parameters': {'type': 'object', 'properties': {'input_mode': {'type': 'string'}, 'universe': {'type': 'string'}, 'tickers': {'type': 'string'}, 'pb_threshold': {'type': 'number'}, 'profit_type': {'type': 'string'}, 'check_issuance': {'type': 'boolean'}, 'check_revenue': {'type': 'boolean'}, 'min_revenue_growth': {'type': 'number'}, 'check_eps': {'type': 'boolean'}, 'min_eps_growth': {'type': 'number'}, 'check_ebit_multiple': {'type': 'boolean'}, 'max_ebit_multiple': {'type': 'number'}, 'check_52w_positive': {'type': 'boolean'}, 'check_min_drawdown': {'type': 'boolean'}, 'min_drawdown_pct': {'type': 'number'}, 'check_max_drawdown': {'type': 'boolean'}, 'max_drawdown_pct': {'type': 'number'}, 'check_3m_pos_momentum': {'type': 'boolean'}, 'check_2m_pos_rel_momentum': {'type': 'boolean'}, 'rel_momentum_benchmark': {'type': 'string'}}, 'required': []}, 'category': 'screeners', 'access_mode': 'compute', 'aliases': ('long screen', 'long screener'), 'selectable': True}
{'name': 'run_fundamental_momentum', 'description': 'Start or reuse an EPS/revenue fundamental momentum screen.', 'parameters': {'type': 'object', 'properties': {'screen_type': {'type': 'string'}, 'universe': {'type': 'string'}, 'tickers': {'type': 'string'}, 'benchmark': {'type': 'string'}, 'input_mode': {'type': 'string'}}, 'required': []}, 'category': 'screeners', 'access_mode': 'compute', 'aliases': ('fundamental momentum', 'eps momentum', 'revenue momentum'), 'selectable': True}
{'name': 'run_portfolio_analyzer', 'description': 'Start or reuse the portfolio analyzer.', 'parameters': {'type': 'object', 'properties': {'book': {'type': 'number'}, 'target_leverage': {'type': 'number'}, 'beta_neutral': {'type': 'boolean'}}, 'required': []}, 'category': 'portfolio', 'access_mode': 'compute', 'aliases': ('portfolio analyzer', 'portfolio optimizer'), 'selectable': True}
{'name': 'run_portfolio_sizer', 'description': 'Start or reuse the portfolio sizer.', 'parameters': {'type': 'object', 'properties': {'book': {'type': 'number'}, 'target_leverage': {'type': 'number'}, 'positions': {'type': 'array', 'items': {'type': 'object'}}}, 'required': []}, 'category': 'portfolio', 'access_mode': 'compute', 'aliases': ('portfolio sizer', 'sizing'), 'selectable': True}
{'name': 'get_portfolio_sizer_prefill', 'description': 'Fetch portfolio sizer prefill positions.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'portfolio', 'access_mode': 'read', 'aliases': ('sizer prefill',), 'selectable': True}
{'name': 'run_hedging_tool', 'description': 'Start or reuse the hedging tool.', 'parameters': {'type': 'object', 'properties': {'book': {'type': 'number'}, 'positions': {'type': 'array', 'items': {'type': 'object'}}}, 'required': []}, 'category': 'portfolio', 'access_mode': 'compute', 'aliases': ('hedging tool', 'hedge analysis'), 'selectable': True}
{'name': 'get_hedging_tool_prefill', 'description': 'Fetch hedging tool prefill positions.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'category': 'portfolio', 'access_mode': 'read', 'aliases': ('hedging prefill',), 'selectable': True}
{'name': 'get_hedging_portfolio_weights', 'description': 'Derive hedging weights from the portfolio database.', 'parameters': {'type': 'object', 'properties': {'book': {'type': 'number'}}, 'required': []}, 'category': 'portfolio', 'access_mode': 'read', 'aliases': ('hedging weights', 'portfolio weights'), 'selectable': True}
{'name': 'run_hedging_recommendation', 'description': 'Generate LLM hedging recommendations from hedging analysis tables.', 'parameters': {'type': 'object', 'properties': {'net_beta_spy': {'type': 'number'}, 'net_beta_iwm': {'type': 'number'}, 'post_hedge_beta_spy': {'type': 'number'}, 'post_hedge_beta_iwm': {'type': 'number'}, 'gross_input': {'type': 'number'}, 'net_input': {'type': 'number'}, 'gross_after_hedging': {'type': 'number'}, 'volatility_after_hedging': {'type': 'number'}, 'hedge_spy_weight': {'type': 'number'}, 'hedge_iwm_weight': {'type': 'number'}, 'positions_df': {'type': 'array', 'items': {'type': 'object'}}, 'hedges_df': {'type': 'array', 'items': {'type': 'object'}}, 'book_size': {'type': 'number'}}, 'required': []}, 'category': 'portfolio', 'access_mode': 'compute', 'aliases': ('hedging recommendation', 'hedge recommendation'), 'selectable': True}
{'name': 'propose_portfolio_positions_update', 'description': 'Propose replacing editable portfolio positions. Creates a pending approval.', 'parameters': {'type': 'object', 'properties': {'positions': {'type': 'array', 'items': {'type': 'object'}}, 'reason': {'type': 'string'}}, 'required': ['positions', 'reason']}, 'category': 'portfolio', 'access_mode': 'proposal', 'aliases': ('propose portfolio edit', 'update portfolio positions'), 'selectable': True}
{'name': 'propose_hedge_positions_update', 'description': 'Propose replacing hedge positions. Creates a pending approval.', 'parameters': {'type': 'object', 'properties': {'positions': {'type': 'array', 'items': {'type': 'object'}}, 'reason': {'type': 'string'}}, 'required': ['positions', 'reason']}, 'category': 'portfolio', 'access_mode': 'proposal', 'aliases': ('propose hedge edit', 'update hedge positions'), 'selectable': True}
{'name': 'propose_thesis_content_update', 'description': "Propose replacing a ticker's thesis markdown. Creates a pending approval.", 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string'}, 'content': {'type': 'string'}, 'reason': {'type': 'string'}}, 'required': ['ticker', 'content', 'reason']}, 'category': 'thesis', 'access_mode': 'proposal', 'aliases': ('propose thesis edit', 'update thesis content'), 'selectable': True}
{'name': 'propose_catalyst', 'description': 'Propose creating a catalyst. Creates a pending approval.', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string'}, 'description': {'type': 'string'}, 'category': {'type': 'string'}, 'target_date': {'type': 'string'}, 'reason': {'type': 'string'}}, 'required': ['ticker', 'description', 'reason']}, 'category': 'process', 'access_mode': 'proposal', 'aliases': ('propose catalyst', 'create catalyst'), 'selectable': True}
{'name': 'propose_kill_condition', 'description': 'Propose creating a kill condition. Creates a pending approval.', 'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string'}, 'condition': {'type': 'string'}, 'metric': {'type': 'string'}, 'threshold': {'type': 'string'}, 'reason': {'type': 'string'}}, 'required': ['ticker', 'condition', 'reason']}, 'category': 'process', 'access_mode': 'proposal', 'aliases': ('propose kill condition', 'create kill condition'), 'selectable': True}
{'name': 'propose_research_note', 'description': 'Propose creating a research note. Creates a pending approval.', 'parameters': {'type': 'object', 'properties': {'title': {'type': 'string'}, 'content': {'type': 'string'}, 'ticker': {'type': 'string'}, 'note_type': {'type': 'string'}, 'reason': {'type': 'string'}}, 'required': ['title', 'content', 'reason']}, 'category': 'research', 'access_mode': 'proposal', 'aliases': ('propose research note', 'create research note'), 'selectable': True}
{'name': 'propose_news_digest_delete', 'description': 'Propose deleting an uploaded news digest. Creates a pending approval.', 'parameters': {'type': 'object', 'properties': {'digest_id': {'type': 'string'}, 'reason': {'type': 'string'}}, 'required': ['digest_id', 'reason']}, 'category': 'research', 'access_mode': 'proposal', 'aliases': ('delete news digest', 'remove digest'), 'selectable': True}
""".strip()


def _tool_policy_spec(tool_name: str) -> PolicySpec | None:
    if tool_name == "query_ontology":
        return PolicySpec(
            ontology_actions=("query",),
            dynamic_ontology_actions=_query_ontology_policy_actions,
        )
    return None


def _tool_binding(tool_name: str) -> dict[str, Any] | None:
    return _PROPOSAL_TOOL_BINDINGS.get(tool_name)


def _tool_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for line in _TOOL_EXPOSURE_SPECS_TEXT.splitlines():
        raw = line.strip()
        if not raw:
            continue
        specs.append(ast.literal_eval(raw))
    return specs


def _build_tool_exposure(spec: dict[str, Any]) -> ToolExposure:
    tool_name = str(spec["name"])
    binding = _tool_binding(tool_name)
    access_mode = cast(ToolAccessMode, str(spec.get("access_mode") or "read"))
    input_schema = dict(spec.get("parameters") or {"type": "object", "properties": {}, "required": []})
    return ToolExposure(
        tool_name=tool_name,
        access_mode=access_mode,
        category=str(spec.get("category") or "misc"),
        description=str(spec.get("description") or ""),
        input_model=_input_model_from_schema(tool_name, input_schema),
        input_schema=input_schema,
        output_spec=_output_spec_for_tool(tool_name, access_mode),
        aliases=tuple(spec.get("aliases") or ()),
        selectable=bool(spec.get("selectable", True)),
        agent_exposed=bool(spec.get("agent_exposed", True)),
        action_id=str(binding["action_id"]) if binding else None,
        to_action_input=binding.get("adapter") if binding else None,
        reason_builder=_tool_reason if binding else None,
        entity_id_builder=binding.get("entity_id") if binding else None,
        once=bool(binding.get("once", False)) if binding else False,
        policy_spec=_tool_policy_spec(tool_name),
    )


_TOOL_EXPOSURES: dict[str, ToolExposure] = {tool.tool_name: tool for tool in map(_build_tool_exposure, _tool_specs())}


def get_action_definition(action_id: ActionId) -> DomainAction:
    return get_action(action_id)


def iter_action_definitions() -> list[DomainAction]:
    return iter_actions()


def get_tool_exposure(tool_name: str) -> ToolExposure:
    try:
        return _TOOL_EXPOSURES[tool_name]
    except KeyError as exc:
        raise ActionValidationError(f"Unsupported tool_name: {tool_name}") from exc


def iter_tool_exposures(*, agent_exposed_only: bool = False) -> list[ToolExposure]:
    exposures = list(_TOOL_EXPOSURES.values())
    if not agent_exposed_only:
        return exposures
    return [tool for tool in exposures if tool.agent_exposed]


def agent_tool_names() -> set[str]:
    return {tool.tool_name for tool in iter_tool_exposures(agent_exposed_only=True)}


def is_agent_tool_exposed(tool_name: str) -> bool:
    exposure = _TOOL_EXPOSURES.get(tool_name)
    return bool(exposure and exposure.agent_exposed)


def list_tool_exposures(*, agent_exposed_only: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tool in iter_tool_exposures(agent_exposed_only=agent_exposed_only):
        rows.append(
            {
                "tool_name": tool.tool_name,
                "access_mode": tool.access_mode,
                "category": tool.category,
                "description": tool.description,
                "aliases": list(tool.aliases),
                "selectable": tool.selectable,
                "agent_exposed": tool.agent_exposed,
                "action_id": tool.action_id,
                "strict_output": tool.output_spec.strict,
            }
        )
    return rows


def tool_definition_rows(*, agent_exposed_only: bool = True) -> list[dict[str, Any]]:
    return [
        tool.to_tool_definition()
        for tool in iter_tool_exposures(agent_exposed_only=agent_exposed_only)
        if tool.agent_exposed or not agent_exposed_only
    ]


def validate_tool_input(tool_name: str, raw_input: dict[str, Any]) -> BaseModel:
    return get_tool_exposure(tool_name).input_model.model_validate(raw_input)


def propose_action_from_tool(tool_name: str, raw_input: dict[str, Any], context: ActionContext) -> dict[str, Any]:
    exposure = get_tool_exposure(tool_name)
    if exposure.access_mode != "proposal" or not exposure.action_id or exposure.to_action_input is None:
        raise ActionValidationError(f"Tool {tool_name} is not a proposal tool")
    try:
        typed_input = exposure.input_model.model_validate(raw_input)
    except PydanticValidationError as exc:
        raise ActionValidationError(_validation_message(exc)) from exc
    action_input = exposure.to_action_input(typed_input)
    reason = exposure.reason_builder(typed_input) if exposure.reason_builder else None
    entity_id = exposure.entity_id_builder(typed_input) if exposure.entity_id_builder else None
    approval = propose_action(
        exposure.action_id,
        action_input,
        context,
        reason=reason,
        entity_id=entity_id,
        once=exposure.once,
    )
    return approval


def workflow_artifact_keys() -> set[str]:
    return set(_WORKFLOW_ARTIFACT_BINDINGS)


def propose_workflow_artifact(artifact_key: str, artifact_value: Any, *, run_id: str, ticker: str | None) -> int:
    binding = _WORKFLOW_ARTIFACT_BINDINGS.get(artifact_key)
    if binding is None:
        return 0
    context = ActionContext(actor_type="workflow", source_type="workflow", source_id=run_id)
    items: list[dict[str, Any]]
    if binding.multiple:
        items = [item for item in artifact_value if isinstance(item, dict)] if isinstance(artifact_value, list) else []
    else:
        items = [artifact_value] if isinstance(artifact_value, dict) else []

    count = 0
    for item_index, item in enumerate(items):
        if binding.required_keys and any(not item.get(key) for key in binding.required_keys):
            continue
        payload = binding.payload_adapter(item, ticker) if binding.payload_adapter else dict(item)
        entity_id = (
            int(item[binding.entity_id_field])
            if binding.entity_id_field and item.get(binding.entity_id_field)
            else None
        )
        artifact_event_id: str | None = None
        try:
            from api import provenance

            artifact_event_id = provenance.deterministic_id("pv:workflow_artifact", run_id, artifact_key, item_index)
            provenance.start_event(
                event_id=artifact_event_id,
                event_type="workflow_artifact",
                event_name=artifact_key,
                actor=context,
                parent_event_id=provenance.deterministic_id("pv:workflow_run", run_id),
                workflow_run_id=run_id,
                input_value=item,
                summary={
                    "artifact_key": artifact_key,
                    "artifact_index": item_index,
                    "ticker": ticker,
                    "action_id": binding.action_id,
                    "entity_id": entity_id,
                },
                metadata={
                    "item_keys": sorted(str(key) for key in item.keys()),
                    "multiple": binding.multiple,
                },
            )
            provenance.link_refs(
                event_id=artifact_event_id,
                source_ref_type="workflow_run",
                source_ref_id=run_id,
                target_ref_type="workflow_artifact",
                target_ref_id=artifact_event_id,
                link_type="produced",
                metadata={"artifact_key": artifact_key, "artifact_index": item_index},
            )
        except Exception:
            artifact_event_id = None
        try:
            approval = propose_action(
                binding.action_id,
                payload,
                replace(context, provenance_event_id=artifact_event_id),
                reason=binding.reason,
                entity_id=entity_id,
            )
        except Exception as exc:
            try:
                from api import provenance

                provenance.finish_event(
                    artifact_event_id,
                    status="failed",
                    summary={
                        "artifact_key": artifact_key,
                        "artifact_index": item_index,
                        "action_id": binding.action_id,
                    },
                    error=str(exc) or exc.__class__.__name__,
                )
            except Exception:
                pass
            raise
        try:
            from api import provenance
            from portfolio import core_db

            record = provenance.record_workflow_artifact(
                workflow_run_id=run_id,
                artifact_key=artifact_key,
                artifact_index=item_index,
                artifact_value=item,
                approval_id=int(approval["id"]),
                provenance_event_id=artifact_event_id,
            )
            artifact_id = str(record.get("artifact_id")) if record and record.get("artifact_id") else artifact_event_id
            core_db.set_pending_approval_provenance(int(approval["id"]), origin_artifact_id=artifact_id)
            provenance.link_refs(
                event_id=artifact_event_id,
                source_ref_type="workflow_artifact",
                source_ref_id=artifact_id or artifact_key,
                target_ref_type="approval",
                target_ref_id=str(approval["id"]),
                link_type="proposed",
                metadata={"action_id": binding.action_id, "artifact_key": artifact_key},
            )
            provenance.finish_event(
                artifact_event_id,
                status="succeeded",
                output_value={"approval_id": approval["id"], "artifact_id": artifact_id},
                summary={
                    "artifact_key": artifact_key,
                    "artifact_index": item_index,
                    "approval_id": approval["id"],
                    "action_id": binding.action_id,
                    "status": "pending_approval_created",
                },
            )
        except Exception:
            pass
        count += 1
    return count
