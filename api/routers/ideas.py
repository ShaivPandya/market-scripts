"""Investment idea watchlist and evaluator endpoints."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from api.action_execution import stage_api_action
from api.async_job_runner import enqueue_registered_job, enqueue_response, poll_registered_job
from api.exceptions import NotFoundError, ValidationError
from api.routers.auth import ActorDep
from decision_quality import (
    ACTIONABLE_ACTIONS as DECISION_ACTIONABLE_ACTIONS,
)
from decision_quality import (
    CANONICAL_ACTIONS,
    DecisionQuality,
    apply_decision_quality_gates,
    decision_quality_schema,
    normalize_action,
    parse_decision_quality,
)
from ontology.object_service import OntologyObjectService
from ontology.policy import actor_to_dict, admin_actor
from ontology.runtime_read_service import OntologyRuntimeReadService
from portfolio.instruments import (
    default_contract_multiplier,
    futures_spec,
    is_continuous_future_symbol,
    normalize_asset,
    normalize_instrument_type,
    normalize_spot_fx_symbol,
    normalize_symbol,
    spot_fx_currencies,
)

router = APIRouter()
LOGGER = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "auto_report" / "prompts"
IDEA_EVALUATION_VERSION = "v4_decision_quality"
IDEA_EVALUATION_SCHEMA_VERSION = "idea_evaluator_v4_decision_quality"
IDEA_ACTIONS = set(CANONICAL_ACTIONS)
IDEA_ANALYZER_DIRECTIONS = {"inactive", "long", "short"}
CANONICAL_IDEA_FACTORS = (
    "macro_support",
    "industry_attractiveness",
    "business_quality",
    "management_quality",
    "valuation_asymmetry",
    "portfolio_fit",
)
type IdeaComparisonStatus = Literal["watching", "researching", "ready_for_review"]
type IdeaStatus = Literal["watching", "researching", "ready_for_review", "accepted", "rejected", "archived"]
type IdeaAsset = Literal["equity", "commodity", "fx", "bond"]
type IdeaInstrumentType = Literal["security", "future", "spot_fx"]
ACTIONABLE_IDEA_STATUSES: tuple[IdeaComparisonStatus, ...] = ("watching", "researching", "ready_for_review")
CRITICAL_MISSING_SEVERITIES = {"critical", "block"}
RECOMMENDATION_STATUSES = {"clear", "review_required", "blocked", "error"}
SOURCE_QUALITY_VALUES = {"ok", "degraded", "stale", "failed"}
OPTIONAL_JSON_BODY = Body(default=None)
IDEA_EVALUATION_OWNED_CHILD_RELATIONS = {
    "research_object_has_factor_score",
    "research_object_has_missing_information",
    "research_object_supported_by_evidence",
    "research_object_disconfirmed_by_evidence",
}
IDEA_EVIDENCE_OWNED_CHILD_RELATIONS = {"evidence_has_citation", "evidence_cites_citation"}
IDEA_INSTRUMENT_FIELDS = {
    "ticker",
    "asset",
    "instrument_type",
    "price_symbol",
    "contract_multiplier",
    "fx_base_currency",
    "fx_quote_currency",
    "currency",
    "country",
    "exchange",
}
IDEA_LIFECYCLE_TRACKED_FIELDS = (
    "status",
    "conviction",
    "user_notes",
    "tags",
    "analyzer_direction",
    "use_portfolio_context",
    *IDEA_INSTRUMENT_FIELDS,
    "rejection_note",
    "rejected_at",
)


class IdeaCreateRequest(BaseModel):
    ticker: str
    company_name: str | None = None
    asset: IdeaAsset | None = None
    instrument_type: IdeaInstrumentType | None = None
    price_symbol: str | None = None
    contract_multiplier: float | None = None
    fx_base_currency: str | None = None
    fx_quote_currency: str | None = None
    currency: str | None = None
    country: str | None = None
    exchange: str | None = None
    user_notes: str | None = None
    tags: list[str] = Field(default_factory=list)
    conviction: int | None = Field(default=None, ge=1, le=5)
    status: IdeaStatus = "watching"
    analyzer_direction: Literal["inactive", "long", "short"] = "inactive"
    use_portfolio_context: bool = True

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        ticker = str(value or "").strip().upper()
        if not ticker:
            raise ValueError("Ticker cannot be empty.")
        return ticker

    @model_validator(mode="after")
    def _normalize_instrument(self) -> IdeaCreateRequest:
        normalized = _normalized_idea_instrument(self.model_dump())
        for key, value in normalized.items():
            setattr(self, key, value)
        return self


class IdeaUpdateRequest(BaseModel):
    ticker: str | None = None
    company_name: str | None = None
    asset: IdeaAsset | None = None
    instrument_type: IdeaInstrumentType | None = None
    price_symbol: str | None = None
    contract_multiplier: float | None = None
    fx_base_currency: str | None = None
    fx_quote_currency: str | None = None
    currency: str | None = None
    country: str | None = None
    exchange: str | None = None
    user_notes: str | None = None
    tags: list[str] | None = None
    conviction: int | None = Field(default=None, ge=1, le=5)
    status: IdeaStatus | None = None
    analyzer_direction: Literal["inactive", "long", "short"] | None = None
    use_portfolio_context: bool | None = None

    @field_validator("ticker")
    @classmethod
    def _normalize_optional_ticker(cls, value: str | None) -> str | None:
        if value is None:
            return None
        ticker = str(value or "").strip().upper()
        if not ticker:
            raise ValueError("Ticker cannot be empty.")
        return ticker


class IdeaEvaluationRequest(BaseModel):
    idea_id: str
    force_refresh: bool = False
    use_portfolio_context: bool = True


class IdeaComparisonEvaluationRequest(BaseModel):
    scope_statuses: list[IdeaComparisonStatus] = Field(default_factory=lambda: list(ACTIONABLE_IDEA_STATUSES))
    use_portfolio_context: bool = True

    @field_validator("scope_statuses")
    @classmethod
    def _normalize_scope_statuses(cls, value: list[str]) -> list[IdeaComparisonStatus]:
        statuses: list[IdeaComparisonStatus] = []
        for status in value or []:
            normalized = str(status).strip().lower()
            if normalized not in ACTIONABLE_IDEA_STATUSES:
                raise ValueError(f"Unsupported idea comparison status: {status}")
            normalized_status = cast(IdeaComparisonStatus, normalized)
            if normalized_status not in statuses:
                statuses.append(normalized_status)
        return statuses or list(ACTIONABLE_IDEA_STATUSES)


class IdeaAcceptRequest(BaseModel):
    note: str | None = None


class IdeaRejectRequest(BaseModel):
    note: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _normalize_analyzer_direction(value: Any) -> str:
    direction = str(value or "inactive").strip().lower()
    return direction if direction in IDEA_ANALYZER_DIRECTIONS else "inactive"


def _normalize_use_portfolio_context(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    return bool(value)


def _normalized_idea_instrument(payload: dict[str, Any], *, base: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = {**(base or {}), **payload}
    ticker_raw = merged.get("ticker") or merged.get("price_symbol")
    price_symbol_raw = merged.get("price_symbol") or ticker_raw
    instrument_type = normalize_instrument_type(
        merged.get("instrument_type"),
        ticker=str(ticker_raw or ""),
        price_symbol=str(price_symbol_raw or ticker_raw or ""),
    )
    fx_base: str | None
    fx_quote: str | None
    currency: str | None
    exchange: str | None

    if instrument_type == "spot_fx":
        price_symbol = normalize_spot_fx_symbol(price_symbol_raw or ticker_raw, field_name="price_symbol")
        ticker = price_symbol
        fx_base, fx_quote = spot_fx_currencies(price_symbol)
        asset = "fx"
        currency = fx_quote
        exchange = str(merged.get("exchange") or "FX").strip().upper() or "FX"
    else:
        ticker = normalize_symbol(ticker_raw, field_name="ticker")
        price_symbol = normalize_symbol(price_symbol_raw or ticker, field_name="price_symbol")
        if instrument_type == "future" and not is_continuous_future_symbol(price_symbol):
            raise ValueError("Futures ideas require a continuous '=F' price_symbol.")
        fx_base = str(merged.get("fx_base_currency") or "").strip().upper() or None
        fx_quote = str(merged.get("fx_quote_currency") or "").strip().upper() or None
        spec = futures_spec(price_symbol) if instrument_type == "future" else None
        asset = (
            spec.asset
            if spec is not None
            else normalize_asset(merged.get("asset"), instrument_type=instrument_type, symbol=price_symbol)
        )
        currency = str(merged.get("currency") or "").strip().upper() or None
        exchange = str(merged.get("exchange") or "").strip().upper() or None

    return {
        "ticker": ticker,
        "asset": asset,
        "instrument_type": instrument_type,
        "price_symbol": price_symbol,
        "contract_multiplier": default_contract_multiplier(
            instrument_type=instrument_type,
            symbol=price_symbol,
            override=merged.get("contract_multiplier"),
        ),
        "fx_base_currency": fx_base,
        "fx_quote_currency": fx_quote,
        "currency": currency,
        "country": str(merged.get("country") or "").strip().upper() or None,
        "exchange": exchange,
    }


def _with_default_idea_instrument_fields(idea: dict[str, Any]) -> dict[str, Any]:
    if not idea:
        return idea
    try:
        normalized = _normalized_idea_instrument({}, base=idea)
    except Exception:
        normalized = {
            "asset": "equity",
            "instrument_type": "security",
            "price_symbol": str(idea.get("ticker") or "").strip().upper(),
            "contract_multiplier": 1.0,
            "fx_base_currency": None,
            "fx_quote_currency": None,
            "currency": idea.get("currency"),
            "country": idea.get("country"),
            "exchange": idea.get("exchange"),
        }
    out = dict(idea)
    for key, default in normalized.items():
        if key == "ticker":
            continue
        if out.get(key) in (None, ""):
            out[key] = default
    return out


def _idea_instrument_metadata(idea: dict[str, Any]) -> dict[str, Any]:
    normalized = _with_default_idea_instrument_fields(idea)
    return {
        "ticker": normalized.get("ticker"),
        "asset": normalized.get("asset"),
        "instrument_type": normalized.get("instrument_type"),
        "price_symbol": normalized.get("price_symbol"),
        "contract_multiplier": normalized.get("contract_multiplier"),
        "fx_base_currency": normalized.get("fx_base_currency"),
        "fx_quote_currency": normalized.get("fx_quote_currency"),
        "currency": normalized.get("currency"),
        "country": normalized.get("country"),
        "exchange": normalized.get("exchange"),
    }


def _is_equity_security_idea(idea: dict[str, Any]) -> bool:
    normalized = _with_default_idea_instrument_fields(idea)
    return (
        str(normalized.get("asset") or "equity").lower() == "equity"
        and str(normalized.get("instrument_type") or "security").lower() == "security"
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return cast(list[Any], value) if isinstance(value, list) else []


def _idea_analyzer_direction(idea: dict[str, Any]) -> str:
    metadata = idea.get("metadata") if isinstance(idea.get("metadata"), dict) else {}
    return _normalize_analyzer_direction(cast(dict[str, Any], metadata).get("analyzer_direction"))


def _idea_uses_portfolio_context(idea: dict[str, Any]) -> bool:
    metadata = idea.get("metadata") if isinstance(idea.get("metadata"), dict) else {}
    return _normalize_use_portfolio_context(cast(dict[str, Any], metadata).get("use_portfolio_context"), default=True)


def _with_analyzer_direction_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    direction = _normalize_analyzer_direction(payload.pop("analyzer_direction", "inactive"))
    use_portfolio_context = _normalize_use_portfolio_context(payload.pop("use_portfolio_context", True), default=True)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    payload["metadata"] = {
        **cast(dict[str, Any], metadata),
        "analyzer_direction": direction,
        "use_portfolio_context": use_portfolio_context,
    }
    return payload


def _fetch_company_name_yfinance(ticker: str) -> str | None:
    normalized = str(ticker or "").strip().upper()
    if not normalized:
        return None
    try:
        import yfinance as yf

        info = yf.Ticker(normalized).get_info() or {}
    except Exception:
        LOGGER.debug("yfinance company-name lookup failed for %s", normalized, exc_info=True)
        return None

    for key in ("longName", "shortName", "displayName"):
        value = str(info.get(key) or "").strip()
        if value:
            return value
    return None


def _resolve_company_name(ticker: str, provided: str | None) -> str | None:
    cleaned = str(provided or "").strip()
    if cleaned:
        return cleaned
    return _fetch_company_name_yfinance(ticker)


def _canonical_factor_score(factor_scores: dict[str, Any], key: str) -> float | None:
    row = factor_scores.get(key)
    if not isinstance(row, dict):
        return None
    return _numeric_or_none(row.get("score"), minimum=0, maximum=100)


def _canonical_score_from_factors(factor_scores: dict[str, Any]) -> float | None:
    scores = [
        score for key in CANONICAL_IDEA_FACTORS if (score := _canonical_factor_score(factor_scores, key)) is not None
    ]
    return round(sum(scores) / len(scores), 1) if scores else None


def _map_zscore_to_score(value: Any) -> float | None:
    signal = _numeric_or_none(value)
    if signal is None:
        return None
    clipped = max(-3.0, min(3.0, signal))
    return round(((clipped + 3.0) / 6.0) * 100.0, 1)


def _object_props(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    props = dict(row.get("properties") or row.get("properties_json") or {})
    uid = str(row.get("object_uid") or props.get("id") or "")
    props["id"] = uid
    props["object_uid"] = uid
    meta = row.get("_meta")
    if isinstance(meta, dict):
        props["_meta"] = meta
    return props


def _idea_uid(value: Any) -> str:
    text = str(value or "").strip()
    return text if text.startswith("investment_idea:") else f"investment_idea:{text}"


def _evaluation_uid(value: Any) -> str:
    text = str(value or "").strip()
    return text if text.startswith("idea_evaluation:") else f"idea_evaluation:{text}"


def _lifecycle_event_uid(value: Any) -> str:
    text = str(value or "").strip()
    return text if text.startswith("idea_lifecycle_event:") else f"idea_lifecycle_event:{text}"


def _idea_lifecycle_snapshot(idea: dict[str, Any]) -> dict[str, Any]:
    metadata_raw = idea.get("metadata")
    metadata: dict[str, Any] = metadata_raw if isinstance(metadata_raw, dict) else {}
    tags = idea.get("tags")
    return {
        "status": idea.get("status"),
        "conviction": idea.get("conviction"),
        "user_notes": idea.get("user_notes"),
        "tags": sorted(str(tag) for tag in (tags if isinstance(tags, list) else [])),
        "analyzer_direction": _idea_analyzer_direction(idea),
        "use_portfolio_context": _idea_uses_portfolio_context(idea),
        "ticker": idea.get("ticker"),
        "asset": idea.get("asset"),
        "instrument_type": idea.get("instrument_type"),
        "price_symbol": idea.get("price_symbol"),
        "contract_multiplier": idea.get("contract_multiplier"),
        "fx_base_currency": idea.get("fx_base_currency"),
        "fx_quote_currency": idea.get("fx_quote_currency"),
        "currency": idea.get("currency"),
        "country": idea.get("country"),
        "exchange": idea.get("exchange"),
        "rejection_note": metadata.get("rejection_note"),
        "rejected_at": metadata.get("rejected_at"),
    }


def _diff_idea_lifecycle_changes(
    before: dict[str, Any],
    after: dict[str, Any],
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    changed_fields: list[str] = []
    before_values: dict[str, Any] = {}
    after_values: dict[str, Any] = {}
    for field in IDEA_LIFECYCLE_TRACKED_FIELDS:
        before_value = before.get(field)
        after_value = after.get(field)
        if before_value != after_value:
            changed_fields.append(field)
            before_values[field] = before_value
            after_values[field] = after_value
    return changed_fields, before_values, after_values


def _lifecycle_event_type_for_changes(changed_fields: list[str]) -> str:
    field_type_map = {
        "status": "status_changed",
        "user_notes": "notes_edited",
        "tags": "tags_edited",
        "conviction": "conviction_changed",
        "analyzer_direction": "analyzer_direction_changed",
        "use_portfolio_context": "portfolio_context_changed",
    }
    if len(changed_fields) == 1 and changed_fields[0] in field_type_map:
        return field_type_map[changed_fields[0]]
    if set(changed_fields).issubset(IDEA_INSTRUMENT_FIELDS):
        return "instrument_changed"
    if {"rejection_note", "rejected_at"}.intersection(changed_fields):
        return "rejected"
    return "idea_updated"


def _write_idea_lifecycle_event(
    idea: dict[str, Any],
    *,
    event_type: str,
    changed_fields: list[str],
    before: dict[str, Any],
    after: dict[str, Any],
    reason: str | None = None,
    evaluation_id: str | None = None,
    recommendation_id: str | None = None,
    approval_id: str | None = None,
    action_approval_id: str | None = None,
    source_type: str = "user",
    source_id: str | None = None,
    changed_at: str | None = None,
) -> dict[str, Any]:
    if not changed_fields:
        return {}
    timestamp = changed_at or _now()
    idea_uid = _idea_uid(idea.get("id") or idea.get("object_uid") or idea.get("idea_id"))
    from ontology.schemas.identity import idea_lifecycle_event_id

    event_uid = idea_lifecycle_event_id(
        _stable_hash(
            {
                "idea_id": idea_uid,
                "event_type": event_type,
                "changed_at": timestamp,
                "changed_fields": changed_fields,
                "before": before,
                "after": after,
            }
        )
    )
    payload = {
        "event_id": event_uid,
        "idea_id": idea_uid,
        "ticker": str(idea.get("ticker") or ""),
        "event_type": event_type,
        "changed_at": timestamp,
        "changed_fields": changed_fields,
        "before": before,
        "after": after,
        "reason": reason,
        "evaluation_id": evaluation_id,
        "recommendation_id": recommendation_id,
        "approval_id": approval_id,
        "action_approval_id": action_approval_id,
        "source_type": source_type,
        "source_id": source_id,
        "metadata": {},
        "ontology_run_id": "operational",
    }
    return _write_runtime_object("IdeaLifecycleEvent", event_uid, payload)


def _record_idea_conviction_history(
    before_idea: dict[str, Any],
    after_idea: dict[str, Any],
    *,
    reason: str | None = None,
    evaluation_id: str | None = None,
    recommendation_id: str | None = None,
    approval_id: str | None = None,
    source_type: str = "user",
    source_id: str | None = None,
) -> dict[str, Any] | None:
    from ontology.conviction_history import record_conviction_change
    from ontology.object_service import OntologyObjectService

    before_conviction = before_idea.get("conviction")
    after_conviction = after_idea.get("conviction")
    if before_conviction == after_conviction:
        return None
    idea_uid = _idea_uid(after_idea.get("id") or after_idea.get("object_uid") or after_idea.get("idea_id"))
    ticker = str(after_idea.get("ticker") or "").strip().upper()
    if not ticker:
        return None
    return record_conviction_change(
        OntologyObjectService(),
        entity_type="investment_idea",
        entity_id=idea_uid,
        ticker=ticker,
        conviction_field="conviction",
        previous_conviction=before_conviction,
        new_conviction=after_conviction,
        changed_at=_now(),
        conviction_source_kind="idea_update",
        reason=reason,
        actor=actor_to_dict(admin_actor(source="ideas")),
        actor_type=source_type,
        source_type=source_type,
        source_id=source_id,
        evaluation_id=evaluation_id,
        recommendation_id=recommendation_id,
        approval_id=approval_id,
        provenance=f"pv:idea_conviction:{idea_uid}",
        input_hash=_stable_hash(
            {
                "idea_id": idea_uid,
                "before": before_conviction,
                "after": after_conviction,
                "changed_at": _now(),
            }
        ),
    )


def _record_idea_lifecycle_changes(
    before_idea: dict[str, Any],
    after_idea: dict[str, Any],
    *,
    event_type: str | None = None,
    reason: str | None = None,
    evaluation_id: str | None = None,
    recommendation_id: str | None = None,
    approval_id: str | None = None,
    action_approval_id: str | None = None,
    source_type: str = "user",
    source_id: str | None = None,
) -> dict[str, Any]:
    before = _idea_lifecycle_snapshot(before_idea)
    after = _idea_lifecycle_snapshot(after_idea)
    changed_fields, before_values, after_values = _diff_idea_lifecycle_changes(before, after)
    if not changed_fields:
        return {}
    if "conviction" in changed_fields:
        _record_idea_conviction_history(
            before_idea,
            after_idea,
            reason=reason,
            evaluation_id=evaluation_id,
            recommendation_id=recommendation_id,
            approval_id=approval_id,
            source_type=source_type,
            source_id=source_id,
        )
    resolved_type = event_type or _lifecycle_event_type_for_changes(changed_fields)
    return _write_idea_lifecycle_event(
        after_idea,
        event_type=resolved_type,
        changed_fields=changed_fields,
        before=before_values,
        after=after_values,
        reason=reason,
        evaluation_id=evaluation_id,
        recommendation_id=recommendation_id,
        approval_id=approval_id,
        action_approval_id=action_approval_id,
        source_type=source_type,
        source_id=source_id,
    )


def _comparison_uid(value: Any) -> str:
    text = str(value or "").strip()
    return text if text.startswith("idea_comparison_run:") else f"idea_comparison_run:{text}"


def _writeable_object_props(props: dict[str, Any]) -> dict[str, Any]:
    payload = dict(props)
    for key in ("_meta", "object_uid", "legacy_id"):
        payload.pop(key, None)
    return payload


def _write_runtime_object(object_type: str, uid: str, props: dict[str, Any]) -> dict[str, Any]:
    now = _now()
    payload = {**_writeable_object_props(props), "ontology_run_id": "operational"}
    service = OntologyObjectService()
    row = service.write_object(
        object_type,
        uid,
        payload,
        now,
        actor=actor_to_dict(admin_actor(source="ideas")),
        provenance=f"pv:{object_type}:{uid}",
        input_hash=_stable_hash(payload),
    )
    written = _object_props(row) or payload
    if object_type == "IdeaEvaluation":
        _write_idea_evaluation_graph(service, written, now=now)
    elif object_type == "IdeaComparisonRun":
        _write_idea_comparison_graph(service, written, now=now)
    return written


def _update_idea_refs(idea_id: Any, updates: dict[str, Any]) -> dict[str, Any]:
    idea = _get_idea(idea_id)
    if not idea:
        raise NotFoundError("Investment idea", str(idea_id))
    idea.update(updates)
    idea["updated_at"] = _now()
    idea.pop("_meta", None)
    idea.pop("id", None)
    idea.pop("object_uid", None)
    return _write_runtime_object("InvestmentIdea", _idea_uid(idea_id), idea)


def _object_uid_from_row(row: dict[str, Any]) -> str:
    props = _object_props(row) or {}
    return str(row.get("object_uid") or props.get("object_uid") or props.get("id") or "").strip()


def _relation_uid_from_row(row: dict[str, Any]) -> str:
    meta = _as_dict(row.get("_meta"))
    temporal = _as_dict(meta.get("temporal"))
    return str(row.get("relation_uid") or temporal.get("relation_uid") or "").strip()


def _expire_current_object_and_relations(service: OntologyObjectService, object_uid: str, *, now: str) -> int:
    uid = str(object_uid or "").strip()
    if not uid:
        return 0
    relation_uids: set[str] = set()
    for relation in [
        *service.query_relations(source_object_uid=uid, limit=1000),
        *service.query_relations(target_object_uid=uid, limit=1000),
    ]:
        relation_uid = _relation_uid_from_row(relation)
        if relation_uid:
            relation_uids.add(relation_uid)
    expired = service.expire_object(uid, tx_to=now)
    for relation_uid in relation_uids:
        service.expire_relation(relation_uid, tx_to=now)
    return expired


def _delete_ontology_runtime_idea(idea: dict[str, Any]) -> int:
    idea_uid = str(idea.get("object_uid") or idea.get("id") or "").strip()
    if not idea_uid:
        return 0
    service = OntologyObjectService()
    now = _now()
    evaluation_uids = {
        uid
        for row in service.query_objects("IdeaEvaluation", filters={"idea_id": idea_uid}, limit=1000)
        if (uid := _object_uid_from_row(row))
    }
    ranking_uids = {
        uid
        for row in service.query_objects("IdeaComparisonRanking", filters={"idea_id": idea_uid}, limit=1000)
        if (uid := _object_uid_from_row(row))
    }
    for evaluation_uid in evaluation_uids:
        ranking_uids.update(
            uid
            for row in service.query_objects(
                "IdeaComparisonRanking",
                filters={"evaluation_id": evaluation_uid},
                limit=1000,
            )
            if (uid := _object_uid_from_row(row))
        )

    owned_child_uids: set[str] = set()
    for evaluation_uid in evaluation_uids:
        for relation in service.query_relations(source_object_uid=evaluation_uid, limit=1000):
            relation_type = str(relation.get("relation_type") or "")
            if relation_type not in IDEA_EVALUATION_OWNED_CHILD_RELATIONS:
                continue
            child_uid = str(relation.get("target_object_uid") or "").strip()
            if not child_uid:
                continue
            owned_child_uids.add(child_uid)
            if relation_type in {"research_object_supported_by_evidence", "research_object_disconfirmed_by_evidence"}:
                for child_relation in service.query_relations(source_object_uid=child_uid, limit=1000):
                    child_relation_type = str(child_relation.get("relation_type") or "")
                    citation_uid = str(child_relation.get("target_object_uid") or "").strip()
                    if child_relation_type in IDEA_EVIDENCE_OWNED_CHILD_RELATIONS and citation_uid:
                        owned_child_uids.add(citation_uid)

    expired = 0
    for object_uid in sorted(owned_child_uids):
        expired += _expire_current_object_and_relations(service, object_uid, now=now)
    for object_uid in sorted(ranking_uids):
        expired += _expire_current_object_and_relations(service, object_uid, now=now)
    for object_uid in sorted(evaluation_uids):
        expired += _expire_current_object_and_relations(service, object_uid, now=now)
    expired += _expire_current_object_and_relations(service, idea_uid, now=now)
    return expired


def _delete_runtime_idea(idea_id: str, idea: dict[str, Any]) -> int:
    return _delete_ontology_runtime_idea(idea)


def _write_relation(
    service: OntologyObjectService,
    source_uid: Any,
    target_uid: Any,
    relation_type: str,
    *,
    now: str,
    properties: dict[str, Any] | None = None,
) -> None:
    source = str(source_uid or "").strip()
    target = str(target_uid or "").strip()
    if not source or not target:
        return
    service.write_relation(
        source,
        target,
        relation_type,
        {"ontology_run_id": "operational", **(properties or {})},
        now,
        actor=actor_to_dict(admin_actor(source="ideas")),
        provenance=f"pv:{relation_type}:{source}:{target}:{_stable_hash(properties or {})}",
    )


def _write_child_object(
    service: OntologyObjectService,
    object_type: str,
    business_key: str,
    props: dict[str, Any],
    *,
    now: str,
) -> dict[str, Any]:
    row = service.write_object(
        object_type,
        business_key,
        {**props, "ontology_run_id": "operational"},
        now,
        actor=actor_to_dict(admin_actor(source="ideas")),
        provenance=f"pv:{object_type}:{business_key}:{_stable_hash(props)}",
        input_hash=_stable_hash(props),
    )
    return _object_props(row) or props


def _write_evidence_graph(
    service: OntologyObjectService,
    parent_uid: str,
    rows: Any,
    *,
    relation_type: str,
    now: str,
) -> None:
    if not isinstance(rows, list):
        return
    for index, item in enumerate(rows, start=1):
        if not isinstance(item, dict):
            item = {"summary": str(item)}
        summary = str(item.get("summary") or item.get("text") or item.get("url") or "").strip()
        title = str(item.get("source") or item.get("title") or f"Evidence {index}").strip()
        evidence_key = f"{parent_uid}:{relation_type}:{index}:{_stable_hash(item)}"
        evidence = _write_child_object(
            service,
            "Evidence",
            evidence_key,
            {
                "evidence_id": evidence_key,
                "evidence_type": "idea_evaluator",
                "title": title,
                "summary": summary or title,
                "confidence": _numeric_or_none(item.get("confidence"), minimum=0, maximum=1),
                "observed_at": item.get("observed_at") or now,
            },
            now=now,
        )
        evidence_uid = str(evidence.get("id") or evidence.get("object_uid") or "")
        _write_relation(service, parent_uid, evidence_uid, relation_type, now=now)
        url = str(item.get("url") or "").strip()
        if url:
            citation_key = f"{evidence_uid}:{_stable_hash(url)}"
            citation = _write_child_object(
                service,
                "Citation",
                citation_key,
                {
                    "citation_id": citation_key,
                    "title": title,
                    "url": url,
                    "document_artifact_id": item.get("document_artifact_id"),
                    "source_record_id": item.get("source_record_id"),
                },
                now=now,
            )
            _write_relation(service, evidence_uid, citation.get("id"), "evidence_has_citation", now=now)


def _write_idea_evaluation_graph(service: OntologyObjectService, evaluation: dict[str, Any], *, now: str) -> None:
    evaluation_uid = str(evaluation.get("id") or evaluation.get("object_uid") or "")
    idea_uid = str(evaluation.get("idea_id") or "")
    if not evaluation_uid:
        return
    _write_relation(service, idea_uid, evaluation_uid, "idea_has_evaluation", now=now)

    factors = evaluation.get("factor_scores")
    if isinstance(factors, dict):
        for factor_name, raw_factor in factors.items():
            factor = raw_factor if isinstance(raw_factor, dict) else {"score": raw_factor}
            factor_key = f"{evaluation_uid}:factor:{factor_name}"
            factor_row = _write_child_object(
                service,
                "FactorScore",
                factor_key,
                {
                    "factor_score_id": factor_key,
                    "parent_uid": evaluation_uid,
                    "parent_type": "IdeaEvaluation",
                    "factor_name": str(factor_name),
                    "score": _numeric_or_none(factor.get("score"), minimum=0, maximum=100),
                    "status": factor.get("status"),
                    "rationale": factor.get("rationale"),
                    "missing": factor.get("missing") if isinstance(factor.get("missing"), list) else [],
                    "created_at": evaluation.get("created_at") or evaluation.get("evaluated_at") or now,
                },
                now=now,
            )
            _write_relation(service, evaluation_uid, factor_row.get("id"), "research_object_has_factor_score", now=now)

    missing = evaluation.get("missing_information")
    if isinstance(missing, list):
        for index, row in enumerate(missing, start=1):
            if not isinstance(row, dict):
                row = {"field": str(row), "severity": "medium", "reason": str(row)}
            field = str(row.get("field") or "unspecified")
            req_key = f"{evaluation_uid}:missing:{field}:{index}"
            req = _write_child_object(
                service,
                "MissingInformationRequirement",
                req_key,
                {
                    "requirement_id": req_key,
                    "parent_uid": evaluation_uid,
                    "parent_type": "IdeaEvaluation",
                    "field": field,
                    "severity": str(row.get("severity") or "medium"),
                    "reason": row.get("reason"),
                    "status": "open",
                    "created_at": evaluation.get("created_at") or evaluation.get("evaluated_at") or now,
                },
                now=now,
            )
            _write_relation(
                service,
                evaluation_uid,
                req.get("id"),
                "research_object_has_missing_information",
                now=now,
            )

    _write_evidence_graph(
        service,
        evaluation_uid,
        evaluation.get("evidence"),
        relation_type="research_object_supported_by_evidence",
        now=now,
    )
    _write_evidence_graph(
        service,
        evaluation_uid,
        evaluation.get("disconfirming_evidence"),
        relation_type="research_object_disconfirmed_by_evidence",
        now=now,
    )
    recommendation_id = _prefixed_uid(evaluation.get("recommendation_id"), "recommendation")
    if recommendation_id:
        _write_relation(
            service,
            evaluation_uid,
            recommendation_id,
            "research_object_links_recommendation",
            now=now,
        )
    for approval_value in (
        evaluation.get("approval_id"),
        evaluation.get("recommendation_approval_id"),
        evaluation.get("action_approval_id"),
    ):
        approval_uid = _prefixed_uid(approval_value, "approval")
        if approval_uid:
            _write_relation(service, evaluation_uid, approval_uid, "research_object_links_approval", now=now)
    action_item_uid = _prefixed_uid(evaluation.get("action_item_id"), "action_item")
    if action_item_uid:
        _write_relation(service, evaluation_uid, action_item_uid, "research_object_links_action_item", now=now)


def _prefixed_uid(value: Any, prefix: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text if text.startswith(f"{prefix}:") else f"{prefix}:{text}"


def _write_idea_comparison_graph(service: OntologyObjectService, run: dict[str, Any], *, now: str) -> None:
    run_uid = str(run.get("id") or run.get("object_uid") or run.get("run_id") or "")
    rankings = run.get("rankings")
    if not run_uid or not isinstance(rankings, list):
        return
    for index, row in enumerate(rankings, start=1):
        if not isinstance(row, dict):
            continue
        rank = int(row.get("rank") or index)
        idea_uid = _idea_uid(row.get("idea_id"))
        evaluation_uid = _evaluation_uid(row.get("evaluation_id"))
        ranking_key = f"{run_uid}:rank:{rank}:{idea_uid}"
        ranking = _write_child_object(
            service,
            "IdeaComparisonRanking",
            ranking_key,
            {
                "ranking_id": ranking_key,
                "comparison_run_id": run_uid,
                "run_id": run.get("run_id") or run_uid,
                "idea_id": idea_uid,
                "evaluation_id": evaluation_uid,
                "ticker": row.get("ticker"),
                "rank": rank,
                "action": row.get("action") or "watch",
                "score": _numeric_or_none(row.get("score"), minimum=0, maximum=100),
                "confidence": _numeric_or_none(row.get("confidence"), minimum=0, maximum=1),
                "confidence_level": row.get("confidence_level") or "low",
                "rationale": row.get("rationale"),
                "created_at": row.get("created_at") or run.get("created_at") or now,
            },
            now=now,
        )
        ranking_uid = ranking.get("id")
        row["id"] = ranking_uid
        row["run_id"] = run.get("run_id") or run_uid
        _write_relation(service, run_uid, ranking_uid, "comparison_run_has_ranking", now=now)
        _write_relation(service, ranking_uid, idea_uid, "ranking_targets_idea", now=now)
        _write_relation(service, ranking_uid, evaluation_uid, "ranking_uses_evaluation", now=now)


def _get_idea(idea_id: Any) -> dict[str, Any] | None:
    reads = OntologyRuntimeReadService()
    text = str(idea_id or "").strip()
    idea = reads.get(text) if text.startswith("investment_idea:") else reads.get(_idea_uid(text))
    return _with_default_idea_instrument_fields(idea) if idea else None


def _list_ideas(*, status: str | None = None, include_archived: bool = False, limit: int = 200) -> list[dict[str, Any]]:
    filters = {"status": status} if status else None
    ideas = OntologyRuntimeReadService().list_objects("InvestmentIdea", filters=filters, limit=limit)
    if not include_archived:
        ideas = [idea for idea in ideas if str(idea.get("status") or "").lower() != "archived"]
    return [_with_default_idea_instrument_fields(idea) for idea in ideas]


def _list_idea_evaluations(idea_id: Any | None = None, *, limit: int = 100) -> list[dict[str, Any]]:
    filters = {"idea_id": _idea_uid(idea_id)} if idea_id is not None else None
    rows = OntologyRuntimeReadService().list_objects("IdeaEvaluation", filters=filters, limit=limit)
    return sorted(rows, key=lambda row: str(row.get("evaluated_at") or ""), reverse=True)


def _list_idea_lifecycle_events(idea_id: Any | None = None, *, limit: int = 100) -> list[dict[str, Any]]:
    filters = {"idea_id": _idea_uid(idea_id)} if idea_id is not None else None
    rows = OntologyRuntimeReadService().list_objects("IdeaLifecycleEvent", filters=filters, limit=limit)
    return sorted(rows, key=lambda row: str(row.get("changed_at") or ""), reverse=True)


def _get_idea_evaluation(evaluation_id: Any) -> dict[str, Any] | None:
    text = str(evaluation_id or "").strip()
    reads = OntologyRuntimeReadService()
    return reads.get(text) if text.startswith("idea_evaluation:") else reads.get(_evaluation_uid(text))


def _write_idea_evaluation(
    idea: dict[str, Any], result: dict[str, Any], *, job_id: str | None = None
) -> dict[str, Any]:
    instrument = _idea_instrument_metadata(idea)
    data_quality = result.get("data_quality") if isinstance(result.get("data_quality"), dict) else {}
    uid = _evaluation_uid(_stable_hash({"idea_id": idea.get("id"), "result": result, "job_id": job_id}))
    payload = {
        **result,
        "data_quality": {**cast(dict[str, Any], data_quality), "instrument": instrument},
        "idea_id": idea.get("id"),
        "ticker": idea.get("ticker"),
        "job_id": job_id,
        "evaluated_at": result.get("evaluated_at") or _now(),
    }
    return _write_runtime_object("IdeaEvaluation", uid, payload)


def _read_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def _safe_text(value: Any, *, max_len: int = 80_000) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) > max_len:
        return text[:max_len] + "\n\n[truncated]"
    return text


def _normalize_missing_rows(value: Any) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, str) and row.strip():
            field = row.strip()
            out.append({"field": field, "severity": "medium", "reason": _missing_reason_from_field(field)})
        elif isinstance(row, dict):
            field = str(row.get("field") or row.get("name") or "unspecified").strip()
            if not field:
                field = "unspecified"
            severity = str(row.get("severity") or "medium").strip().lower()
            reason = str(
                row.get("reason") or row.get("description") or row.get("why_needed") or row.get("impact") or ""
            ).strip()
            if not reason or reason.lower() == field.lower():
                reason = _missing_reason_from_field(field)
            out.append(
                {
                    "field": field,
                    "severity": severity,
                    "reason": reason,
                }
            )
    return out


def _missing_reason_from_field(field: str) -> str:
    normalized = field.lower()
    if any(token in normalized for token in ("valuation", "multiple", "price", "p/e", "market cap")):
        return "Needed to judge valuation asymmetry, entry price, and downside if the thesis is crowded."
    if "capex" in normalized or "hyperscaler" in normalized:
        return "Needed to verify whether the demand driver is accelerating, stable, or starting to decelerate."
    if "portfolio" in normalized or "risk budget" in normalized or "concentration" in normalized:
        return "Needed to decide whether the idea fits current exposure and can be sized responsibly."
    if "customer" in normalized:
        return "Needed to test whether customer concentration is improving or becoming a larger thesis risk."
    if "short interest" in normalized or "insider" in normalized or "ownership" in normalized:
        return "Needed to assess positioning, squeeze risk, and whether consensus already owns the thesis."
    if "win/loss" in normalized or "spectrum-x" in normalized or "competitive" in normalized:
        return "Needed to validate the competitive threat and whether share is actually shifting."
    return "Needed before treating the idea as actionable."


def _has_critical_missing(rows: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(row, dict) and str(row.get("severity") or "").lower() in CRITICAL_MISSING_SEVERITIES for row in rows
    )


def _source_quality_from_missing(rows: list[dict[str, Any]], tool_errors: list[str]) -> dict[str, Any]:
    critical = _has_critical_missing(rows)
    degraded = critical or bool(tool_errors) or bool(rows)
    return {
        "critical_data_quality": "degraded" if degraded else "ok",
        "source_quality": "degraded" if degraded else "ok",
        "quality": "degraded" if degraded else "ok",
        "tool_errors": tool_errors,
        "missing_count": len(rows),
        "critical_missing_count": sum(
            1 for row in rows if str(row.get("severity") or "").lower() in CRITICAL_MISSING_SEVERITIES
        ),
    }


def _recommendation_status(value: Any, *, fallback: str = "clear") -> str:
    normalized = str(value or fallback).strip().lower()
    return normalized if normalized in RECOMMENDATION_STATUSES else fallback


def _quality_value(value: Any, *, fallback: str = "ok") -> str:
    normalized = str(value or fallback).strip().lower()
    return normalized if normalized in SOURCE_QUALITY_VALUES else fallback


def _as_json_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return cast(dict[str, Any], dumped) if isinstance(dumped, dict) else None
    return cast(dict[str, Any], value) if isinstance(value, dict) else None


def _idea_evaluator_json_schema() -> dict[str, Any]:
    nullable_string = {"type": ["string", "null"]}
    string_array = {"type": "array", "items": {"type": "string"}}
    factor_score_schema = {
        "type": "object",
        "description": "One factor score row. Always return an object, never a bare number.",
        "additionalProperties": False,
        "required": ["score", "status", "rationale", "source", "missing"],
        "properties": {
            "score": {"type": ["number", "null"], "minimum": 0, "maximum": 100},
            "status": {
                "type": "string",
                "description": "Short state such as supportive, mixed, challenged, incomplete, or reviewable.",
            },
            "rationale": {
                "type": "string",
                "description": "One concrete sentence explaining why this factor got this score; do not repeat the factor name.",
            },
            "source": {"type": "string", "description": "Evidence source or context used for the score."},
            "missing": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific missing inputs for this factor, or an empty list.",
            },
        },
    }
    missing_information_schema = {
        "type": "object",
        "description": "A missing input that would change actionability, confidence, or sizing.",
        "additionalProperties": False,
        "required": ["field", "severity", "reason"],
        "properties": {
            "field": {"type": "string", "description": "Short name of the missing input."},
            "severity": {"type": "string", "enum": ["low", "medium", "high", "critical", "block"]},
            "reason": {
                "type": "string",
                "description": "Why this input matters. Must not duplicate field.",
            },
        },
    }
    evidence_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["source", "url", "summary", "claim", "support"],
        "properties": {
            "source": {"type": "string"},
            "url": nullable_string,
            "summary": {"type": "string"},
            "claim": {"type": "string"},
            "support": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "thesis_statement",
            "action",
            "recommendation_status",
            "score",
            "confidence",
            "rationale",
            "factor_scores",
            "missing_information",
            "data_quality",
            "evidence",
            "disconfirming_evidence",
            "catalyst",
            "invalidation",
            "portfolio_fit",
            "decision_quality",
        ],
        "properties": {
            "thesis_statement": {"type": "string"},
            "action": {"type": "string", "enum": list(CANONICAL_ACTIONS)},
            "recommendation_status": {"type": "string", "enum": sorted(RECOMMENDATION_STATUSES)},
            "score": {"type": ["number", "null"], "minimum": 0, "maximum": 100},
            "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "rationale": {"type": "string"},
            "factor_scores": {
                "type": "object",
                "additionalProperties": False,
                "required": list(CANONICAL_IDEA_FACTORS),
                "properties": {name: factor_score_schema for name in CANONICAL_IDEA_FACTORS},
            },
            "missing_information": {"type": "array", "items": missing_information_schema},
            "data_quality": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "critical_data_quality",
                    "source_quality",
                    "quality",
                    "tool_errors",
                    "missing_count",
                    "critical_missing_count",
                    "portfolio_context_used",
                ],
                "properties": {
                    "critical_data_quality": {"type": "string", "enum": sorted(SOURCE_QUALITY_VALUES)},
                    "source_quality": {"type": "string", "enum": sorted(SOURCE_QUALITY_VALUES)},
                    "quality": {"type": "string", "enum": sorted(SOURCE_QUALITY_VALUES)},
                    "tool_errors": string_array,
                    "missing_count": {"type": "integer", "minimum": 0},
                    "critical_missing_count": {"type": "integer", "minimum": 0},
                    "portfolio_context_used": {"type": "boolean"},
                },
            },
            "evidence": {"type": "array", "items": evidence_schema},
            "disconfirming_evidence": {"type": "array", "items": evidence_schema},
            "catalyst": nullable_string,
            "invalidation": nullable_string,
            "portfolio_fit": {
                "type": "object",
                "additionalProperties": False,
                "required": ["status", "note"],
                "properties": {
                    "status": {"type": "string"},
                    "note": {"type": "string"},
                },
            },
            "decision_quality": decision_quality_schema(),
        },
    }


def _numeric_or_none(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if minimum is not None:
        numeric = max(minimum, numeric)
    if maximum is not None:
        numeric = min(maximum, numeric)
    return numeric


def _read_state_text(folder: str, ticker: str) -> tuple[str | None, str | None]:
    try:
        from api.state_storage import exists_text, read_text
        from paths import PROJECT_ROOT

        local_path = PROJECT_ROOT / folder / f"{ticker}.md"
        gcs_prefixes = {
            "investment_overviews": "live/overviews",
            "investment_theses": "live/theses",
            "investment_management_quality": "live/management_quality",
        }
        gcs_key = f"{gcs_prefixes.get(folder, folder)}/{ticker}.md"
        if not exists_text(local_path, gcs_key):
            return None, None
        return read_text(local_path, gcs_key, encoding="utf-8"), None
    except Exception as exc:
        return None, f"{folder}: {exc}"


def _read_management_quality_text(ticker: str) -> tuple[str | None, str | None]:
    try:
        from api.routers.management_quality import _read_markdown_projection, _render_management_quality_markdown

        assessment = OntologyRuntimeReadService().management_quality_assessment(ticker)
        if assessment:
            return (
                _read_markdown_projection(ticker) or _render_management_quality_markdown(ticker, assessment),
                None,
            )
    except Exception as exc:
        return None, f"investment_management_quality: {exc}"
    return None, None


def _safe_tool(name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        from api.agent_tools import execute_tool
        from ontology.policy import admin_actor

        raw = execute_tool(name, args or {}, actor=admin_actor(source="idea_evaluator"))
        try:
            data = json.loads(raw)
        except Exception:
            data = {"raw": raw}
        return {"ok": True, "data": data}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _records_from_table(value: Any) -> list[dict[str, Any]]:
    try:
        from api.serializers import serialize_dataframe

        if hasattr(value, "to_dict"):
            return cast(list[dict[str, Any]], serialize_dataframe(value))
    except Exception:
        pass
    if isinstance(value, list):
        return [cast(dict[str, Any], row) for row in value if isinstance(row, dict)]
    return []


ANALYZER_RISK_FLAG_KEYS = (
    "drawdown_risk",
    "drawdown_data_missing",
    "contrarian_not_eligible",
    "short_squeeze_risk",
    "short_squeeze_data_missing",
    "risk_data_missing",
)
ANALYZER_RISK_PART_KEYS = (
    "drawdown_risk_penalty",
    "contrarian_risk_pressure",
    "short_squeeze_cover_risk",
)
ANALYZER_RISK_NUMBER_KEYS = (
    "long_risk_penalty",
    "short_cover_risk",
)
ANALYZER_RISK_AVAILABILITY_KEYS = (
    "drawdown_metrics_available",
    "short_squeeze_metrics_available",
)


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return None


def _analyzer_risk_flags_from_sources(action: dict[str, Any], row: dict[str, Any]) -> dict[str, bool]:
    action_flags = _as_dict(action.get("risk_flags"))
    flags: dict[str, bool] = {}
    for key in ANALYZER_RISK_FLAG_KEYS:
        value = _bool_or_none(action_flags.get(key))
        if value is None:
            value = _bool_or_none(row.get(key))
        if value is not None:
            flags[key] = value
    return flags


def _analyzer_risk_parts_from_sources(action: dict[str, Any], row: dict[str, Any]) -> dict[str, float]:
    action_parts = _as_dict(action.get("risk_parts"))
    parts: dict[str, float] = {}
    for key in ANALYZER_RISK_PART_KEYS:
        value = _numeric_or_none(action_parts.get(key))
        if value is None:
            value = _numeric_or_none(row.get(key))
        if value is not None:
            parts[key] = value
    return parts


def _analyzer_numeric_from_sources(key: str, action: dict[str, Any], row: dict[str, Any]) -> float | None:
    value = _numeric_or_none(action.get(key))
    if value is None:
        value = _numeric_or_none(row.get(key))
    return value


def _compact_analyzer_risk_context(analyzer_context: Any) -> dict[str, Any]:
    context = _as_dict(analyzer_context)
    if not context:
        return {}
    compact: dict[str, Any] = {
        "status": context.get("status"),
        "action_label": context.get("action_label"),
        "direction": context.get("direction"),
        "scenario_score": context.get("scenario_score"),
        "score_delta": context.get("score_delta"),
        "risk_flags": context.get("risk_flags") if isinstance(context.get("risk_flags"), dict) else {},
        "risk_parts": context.get("risk_parts") if isinstance(context.get("risk_parts"), dict) else {},
        "long_risk_penalty": context.get("long_risk_penalty"),
        "short_cover_risk": context.get("short_cover_risk"),
    }
    warnings = context.get("warnings")
    if isinstance(warnings, list):
        compact["warnings"] = [str(warning) for warning in warnings[:4]]
    return compact


def _compute_portfolio_plus_ideas_analyzer_result() -> dict[str, Any]:
    try:
        from portfolio.portfolio_optimizer.portfolio_analyzer import get_data

        data = get_data(universe_mode="portfolio_plus_ideas")
        if data.get("error"):
            return {"status": "error", "error": str(data.get("error")), "raw_result": data}
        return {"status": "ok", "raw_result": data}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "raw_result": {}}


def _analyzer_contexts_from_result(analyzer_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if analyzer_result.get("status") != "ok":
        return {}
    raw_result = _as_dict(analyzer_result.get("raw_result"))
    weights = {
        str(row.get("ticker") or "").strip().upper(): row
        for row in _records_from_table(raw_result.get("weights_df"))
        if str(row.get("ticker") or "").strip()
    }
    course = _as_dict(raw_result.get("course_of_action"))
    action_rows = _as_list(course.get("action_queue"))
    actions = {
        str(row.get("ticker") or "").strip().upper(): cast(dict[str, Any], row)
        for row in action_rows
        if isinstance(row, dict) and str(row.get("ticker") or "").strip()
    }
    summary = _as_dict(course.get("summary"))
    source_timestamp = summary.get("as_of") or raw_result.get("timestamp")

    contexts: dict[str, dict[str, Any]] = {}
    for ticker, row in weights.items():
        action = actions.get(ticker, {})
        risk_flags = _analyzer_risk_flags_from_sources(action, row)
        risk_parts = _analyzer_risk_parts_from_sources(action, row)
        risk_numbers = {
            key: value
            for key in ANALYZER_RISK_NUMBER_KEYS
            if (value := _analyzer_numeric_from_sources(key, action, row)) is not None
        }
        metric_availability = {
            key: value for key in ANALYZER_RISK_AVAILABILITY_KEYS if (value := _bool_or_none(row.get(key))) is not None
        }
        contexts[ticker] = {
            "status": "available",
            "ticker": ticker,
            "source_timestamp": source_timestamp,
            "source_type": action.get("source_type") or row.get("source_type") or "portfolio",
            "action_label": action.get("action"),
            "scenario_score": action.get("scenario_score", row.get("scenario_score")),
            "baseline_score": action.get("baseline_score", row.get("baseline_score")),
            "score_delta": action.get("score_delta", row.get("score_delta")),
            "confidence": action.get("confidence"),
            "gate_status": action.get("gate_status"),
            "gate_reasons": action.get("gate_reasons") if isinstance(action.get("gate_reasons"), list) else [],
            "coverage": action.get("data_coverage") if isinstance(action.get("data_coverage"), dict) else {},
            "warnings": action.get("warnings") if isinstance(action.get("warnings"), list) else [],
            "factor_breakdown": action.get("factor_breakdown")
            if isinstance(action.get("factor_breakdown"), list)
            else [],
            "risk_flags": risk_flags,
            "risk_parts": risk_parts,
            **risk_numbers,
            **metric_availability,
            "row": row,
            "diagnostic_subfactors": {
                "fundamental_momentum": {
                    "signal": row.get("fundamental_momentum_signal"),
                    "mapped_score": _map_zscore_to_score(row.get("fundamental_momentum_signal")),
                },
                "price_momentum": {
                    "signal": row.get("price_mom_signal"),
                    "mapped_score": _map_zscore_to_score(row.get("price_mom_signal")),
                },
            },
            "qualitative_evidence": {
                "business_quality": row.get("business_quality_qual_evidence"),
                "industry_quality": row.get("industry_quality_evidence"),
                "management_quality": row.get("management_quality_evidence"),
            },
        }
    return contexts


def _analyzer_context_for_idea(
    idea: dict[str, Any],
    *,
    analyzer_result: dict[str, Any] | None = None,
    analyzer_contexts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    direction = _idea_analyzer_direction(idea)
    ticker = str(idea.get("ticker") or "").strip().upper()
    if direction == "inactive":
        return {"status": "inactive", "ticker": ticker, "direction": direction}

    result = analyzer_result or _compute_portfolio_plus_ideas_analyzer_result()
    if result.get("status") != "ok":
        return {
            "status": "error",
            "ticker": ticker,
            "direction": direction,
            "error": str(result.get("error") or "Analyzer unavailable."),
        }
    contexts = analyzer_contexts or _analyzer_contexts_from_result(result)
    context = dict(contexts.get(ticker) or {})
    if not context:
        return {
            "status": "missing",
            "ticker": ticker,
            "direction": direction,
            "reason": "No analyzer row was produced for this active idea.",
        }
    context["direction"] = direction
    return context


def _disabled_analyzer_context_for_idea(idea: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "disabled",
        "ticker": str(idea.get("ticker") or "").strip().upper(),
        "direction": _idea_analyzer_direction(idea),
        "reason": "Portfolio context excluded by evaluation setting.",
    }


def _build_context(
    idea: dict[str, Any],
    analyzer_context: dict[str, Any] | None = None,
    *,
    use_portfolio_context: bool = True,
) -> dict[str, Any]:
    idea = _with_default_idea_instrument_fields(idea)
    ticker = str(idea["ticker"]).upper()
    instrument = _idea_instrument_metadata(idea)
    equity_security = _is_equity_security_idea(idea)
    overview, overview_error = _read_state_text("investment_overviews", ticker)
    thesis, thesis_error = _read_state_text("investment_theses", ticker)
    management_quality, management_quality_error = (
        _read_management_quality_text(ticker) if equity_security else (None, None)
    )

    portfolio = _safe_tool("get_portfolio", {"include_hedges": True}) if use_portfolio_context else None
    signal_aggregator = _safe_tool("get_signal_aggregator", {"include_history": False, "lookback_weeks": 156})
    industry_monitor = (
        _safe_tool("get_industry_monitor", {"refresh": False})
        if equity_security
        else {"ok": True, "skipped": True, "reason": "Equity industry monitor does not apply to this instrument."}
    )
    dossier = (
        _safe_tool("get_dossier", {"ticker": ticker})
        if equity_security
        else {"ok": True, "skipped": True, "reason": "Equity dossier does not apply to this instrument."}
    )

    tool_payloads = {
        "signal_aggregator": signal_aggregator,
        "industry_monitor": industry_monitor,
        "dossier": dossier,
    }
    if use_portfolio_context and isinstance(portfolio, dict):
        tool_payloads["portfolio"] = portfolio
    tool_errors = [
        f"{label}: {payload.get('error')}" for label, payload in tool_payloads.items() if not payload.get("ok")
    ]
    if overview_error:
        tool_errors.append(overview_error)
    if thesis_error:
        tool_errors.append(thesis_error)
    if management_quality_error:
        tool_errors.append(management_quality_error)
    if isinstance(analyzer_context, dict):
        analyzer_payload = analyzer_context
    elif not use_portfolio_context:
        analyzer_payload = _disabled_analyzer_context_for_idea(idea)
    else:
        analyzer_payload = _analyzer_context_for_idea(idea)
    analyzer_status = str(analyzer_payload.get("status") or "")
    if analyzer_status in {"error", "missing"}:
        tool_errors.append(f"analyzer_context: {analyzer_payload.get('error') or analyzer_payload.get('reason')}")

    context = {
        "idea": idea,
        "ticker": ticker,
        "instrument": instrument,
        "asset": instrument.get("asset"),
        "instrument_type": instrument.get("instrument_type"),
        "analyzer_context": analyzer_payload,
        "use_portfolio_context": bool(use_portfolio_context),
        "overview_content": _safe_text(overview),
        "thesis_content": _safe_text(thesis),
        "management_quality_content": _safe_text(management_quality),
        "signal_aggregator": signal_aggregator,
        "industry_monitor": industry_monitor,
        "dossier": dossier,
        "tool_errors": tool_errors,
        "evaluated_at": _now(),
    }
    if use_portfolio_context and portfolio is not None:
        context["portfolio"] = portfolio
    return context


def _build_context_for_evaluation(
    idea: dict[str, Any],
    analyzer_context: dict[str, Any] | None = None,
    *,
    use_portfolio_context: bool = True,
) -> dict[str, Any]:
    import inspect

    params = inspect.signature(_build_context).parameters
    if "analyzer_context" in params and "use_portfolio_context" in params:
        return _build_context(
            idea,
            analyzer_context=analyzer_context,
            use_portfolio_context=use_portfolio_context,
        )
    if "analyzer_context" in params:
        context = _build_context(idea, analyzer_context=analyzer_context)
    else:
        context = _build_context(idea)
    context["use_portfolio_context"] = bool(use_portfolio_context)
    if not use_portfolio_context:
        context.pop("portfolio", None)
        context["analyzer_context"] = analyzer_context or _disabled_analyzer_context_for_idea(idea)
        return context
    context["analyzer_context"] = analyzer_context or {"status": "inactive", "ticker": context.get("ticker")}
    return context


def _factor(
    score: float,
    status: str,
    rationale: str,
    missing: list[str] | None = None,
    *,
    source: str = "evaluator",
) -> dict[str, Any]:
    return {
        "score": max(0, min(100, round(float(score), 1))),
        "status": status,
        "rationale": rationale,
        "missing": missing or [],
        "source": source,
    }


def _asset_label(context: dict[str, Any]) -> str:
    instrument = _as_dict(context.get("instrument"))
    asset = str(context.get("asset") or instrument.get("asset") or "equity").lower()
    instrument_type = str(context.get("instrument_type") or instrument.get("instrument_type") or "security").lower()
    if instrument_type == "spot_fx":
        base = str(instrument.get("fx_base_currency") or "").upper()
        quote = str(instrument.get("fx_quote_currency") or "").upper()
        return f"{base}/{quote} spot FX" if base and quote else "spot FX"
    if instrument_type == "future":
        return f"{asset} future"
    return f"{asset} security"


def _asset_factor_rationale(context: dict[str, Any], factor_name: str, *, has_equity_doc: bool = False) -> str:
    label = _asset_label(context)
    if factor_name == "industry_attractiveness":
        return f"{label} regime and market-structure evidence requires review."
    if factor_name == "business_quality":
        return (
            "Business quality is supported by company-specific overview material."
            if has_equity_doc
            else f"{label} thesis quality depends on macro, carry, curve, supply-demand, and liquidity evidence rather than company fundamentals."
        )
    if factor_name == "management_quality":
        return (
            "Management quality is supported by the uploaded management-quality assessment."
            if has_equity_doc
            else f"{label} has no issuer management-quality requirement; neutral score reflects non-equity applicability."
        )
    if factor_name == "valuation_asymmetry":
        return f"{label} valuation/asymmetry requires current level, carry or roll, curve, positioning, and downside scenario evidence."
    if factor_name == "portfolio_fit":
        return f"Portfolio fit uses current cross-asset exposure context for the {label} idea when available."
    return f"Macro support is scored against the current regime for the {label} idea."


def _factor_rationale_from_score(key: str, score: float) -> str:
    label = key.replace("_", " ")
    if score >= 75:
        posture = "strong support"
    elif score >= 60:
        posture = "moderate support"
    elif score >= 45:
        posture = "mixed support"
    else:
        posture = "weak support"
    return f"{label} scored as {posture}; evaluator did not provide a separate rationale."


def _ensure_canonical_factor_rows(factor_scores: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in CANONICAL_IDEA_FACTORS:
        row = factor_scores.get(key)
        if isinstance(row, dict):
            score = _numeric_or_none(row.get("score"), minimum=0, maximum=100)
            final_score = 50.0 if score is None else round(score, 1)
            rationale = str(row.get("rationale") or row.get("summary") or row.get("reason") or "").strip()
            if not rationale:
                rationale = _factor_rationale_from_score(key, final_score)
            normalized[key] = {
                **row,
                "score": final_score,
                "status": str(row.get("status") or "reviewable"),
                "rationale": rationale,
                "source": str(row.get("source") or "evaluator"),
                "missing": row.get("missing") if isinstance(row.get("missing"), list) else [],
            }
        else:
            normalized[key] = _factor(50, "missing", f"{key} was not returned by the evaluator.")
    return normalized


_FACTOR_ALIASES = {
    "macro": "macro_support",
    "macro_support": "macro_support",
    "industry": "industry_attractiveness",
    "industry_attractiveness": "industry_attractiveness",
    "business": "business_quality",
    "business_quality": "business_quality",
    "management": "management_quality",
    "management_quality": "management_quality",
    "valuation": "valuation_asymmetry",
    "valuation_asymmetry": "valuation_asymmetry",
    "portfolio": "portfolio_fit",
    "portfolio_fit": "portfolio_fit",
}


def _normalize_factor_scores(value: Any) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    rows: dict[str, Any] = {}
    for raw_key, raw_row in raw.items():
        key = _FACTOR_ALIASES.get(str(raw_key).strip().lower())
        if key is None:
            continue
        if isinstance(raw_row, dict):
            score = _numeric_or_none(
                raw_row.get("score") or raw_row.get("value") or raw_row.get("rating"),
                minimum=0,
                maximum=100,
            )
            rows[key] = _factor(
                50 if score is None else score,
                str(raw_row.get("status") or raw_row.get("label") or "reviewable"),
                str(
                    raw_row.get("rationale")
                    or raw_row.get("summary")
                    or raw_row.get("evidence")
                    or raw_row.get("reason")
                    or ""
                ),
                _as_list(raw_row.get("missing")),
                source=str(raw_row.get("source") or "evaluator"),
            )
        else:
            score = _numeric_or_none(raw_row, minimum=0, maximum=100)
            if score is not None:
                rows[key] = _factor(score, "reviewable", _factor_rationale_from_score(key, score))
    return _ensure_canonical_factor_rows(rows)


def _join_structured_text(value: Any, keys: Sequence[str]) -> str | None:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        parts: list[str] = []
        for key in keys:
            item = value.get(key)
            if item in (None, "", [], {}):
                continue
            label = key.replace("_", " ")
            if isinstance(item, list):
                text = "; ".join(str(child) for child in item if child not in (None, "", [], {}))
            else:
                text = str(item)
            if text:
                parts.append(f"{label}: {text}")
        if parts:
            return "; ".join(parts)
    return str(value)


def _normalize_evidence_rows(value: Any) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, str):
            row_summary = row.strip()
            if row_summary:
                normalized.append({"source": "evaluator", "summary": row_summary})
            continue
        if not isinstance(row, dict):
            continue
        source_refs = row.get("source_refs") if isinstance(row.get("source_refs"), list) else []
        source = (
            row.get("source")
            or row.get("title")
            or row.get("citation")
            or (", ".join(str(ref) for ref in source_refs if ref) if source_refs else None)
            or "evaluator"
        )
        summary: Any = (
            row.get("summary")
            or row.get("claim")
            or row.get("support")
            or row.get("detail")
            or row.get("rationale")
            or row.get("evidence")
        )
        if str(summary or "").strip().lower() == "evidence item":
            summary = row.get("claim") or row.get("support") or row.get("detail") or row.get("rationale")
        summary_text = str(summary or "").strip()
        if not summary_text:
            continue
        normalized_row = {
            "source": str(source),
            "summary": summary_text,
        }
        if row.get("url"):
            normalized_row["url"] = str(row["url"])
        if row.get("observed_at"):
            normalized_row["observed_at"] = str(row["observed_at"])
        normalized.append(normalized_row)
    return normalized


def _append_analyzer_evidence(result: dict[str, Any], analyzer_context: dict[str, Any]) -> None:
    evidence = result.setdefault("evidence", [])
    if not isinstance(evidence, list):
        result["evidence"] = evidence = []
    source_timestamp = analyzer_context.get("source_timestamp")
    qualitative = analyzer_context.get("qualitative_evidence")
    if isinstance(qualitative, dict):
        for label, value in qualitative.items():
            if value in (None, "", [], {}):
                continue
            evidence.append(
                {
                    "source": f"analyzer:{label}",
                    "summary": _safe_text(value, max_len=700),
                    "observed_at": source_timestamp,
                }
            )
    risk_flags = _as_dict(analyzer_context.get("risk_flags"))
    active_flags = [key for key in ANALYZER_RISK_FLAG_KEYS if risk_flags.get(key) is True]
    risk_parts = _as_dict(analyzer_context.get("risk_parts"))
    active_parts: list[str] = []
    for key in ANALYZER_RISK_PART_KEYS:
        value = _numeric_or_none(risk_parts.get(key))
        if value is not None and abs(value) > 1e-9:
            active_parts.append(f"{key}={value:.2f}")
    short_cover_risk = _numeric_or_none(analyzer_context.get("short_cover_risk"))
    long_risk_penalty = _numeric_or_none(analyzer_context.get("long_risk_penalty"))
    risk_summary_parts: list[str] = []
    if active_flags:
        risk_summary_parts.append("Flags: " + ", ".join(active_flags))
    if active_parts:
        risk_summary_parts.append("Parts: " + ", ".join(active_parts))
    if short_cover_risk is not None:
        risk_summary_parts.append(f"short_cover_risk={short_cover_risk:.2f}")
    if long_risk_penalty is not None:
        risk_summary_parts.append(f"long_risk_penalty={long_risk_penalty:.2f}")
    if risk_summary_parts:
        evidence.append(
            {
                "source": "analyzer_risk",
                "summary": "; ".join(risk_summary_parts),
                "observed_at": source_timestamp,
            }
        )
    warnings = analyzer_context.get("warnings")
    for warning in warnings if isinstance(warnings, list) else []:
        evidence.append(
            {
                "source": "analyzer_warning",
                "summary": str(warning),
                "observed_at": source_timestamp,
            }
        )


def _merge_analyzer_context_into_result(context: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    use_portfolio_context = _normalize_use_portfolio_context(context.get("use_portfolio_context"), default=True)
    analyzer_context = _as_dict(context.get("analyzer_context"))
    if not use_portfolio_context:
        analyzer_context = _disabled_analyzer_context_for_idea(_as_dict(context.get("idea")))
    result["evaluation_schema_version"] = IDEA_EVALUATION_SCHEMA_VERSION
    result["analyzer_context"] = analyzer_context
    data_quality = _as_dict(result.get("data_quality"))
    result["data_quality"] = {
        **data_quality,
        "portfolio_context_used": use_portfolio_context,
        "instrument": _as_dict(context.get("instrument")),
    }

    factor_scores_raw = result.get("factor_scores")
    factor_scores = _ensure_canonical_factor_rows(
        cast(dict[str, Any], factor_scores_raw) if isinstance(factor_scores_raw, dict) else {}
    )
    if not use_portfolio_context:
        factor_scores["portfolio_fit"] = _factor(
            50,
            "disabled",
            "Portfolio context excluded by evaluation setting; neutral score used for canonical average.",
        )
        portfolio_fit = _as_dict(result.get("portfolio_fit"))
        result["portfolio_fit"] = {
            **portfolio_fit,
            "status": "disabled",
            "note": "Portfolio context excluded by evaluation setting.",
        }
    elif analyzer_context.get("status") == "available":
        row = _as_dict(analyzer_context.get("row"))
        analyzer_factor_specs = {
            "industry_attractiveness": ("industry_quality_score", "Analyzer raw industry_quality_score"),
            "business_quality": ("business_quality_qual_score", "Analyzer raw business_quality_qual_score"),
            "management_quality": ("management_quality_score", "Analyzer raw management_quality_score"),
        }
        for factor_name, (column, label) in analyzer_factor_specs.items():
            score = _numeric_or_none(row.get(column), minimum=0, maximum=100)
            if score is None:
                continue
            factor_scores[factor_name] = _factor(
                score,
                "analyzer_raw_score",
                f"{label} is used directly on its native 0-100 scale, not the parallel z-scored signal.",
            )

        valuation_score = _map_zscore_to_score(row.get("valuation_signal"))
        if valuation_score is not None:
            factor_scores["valuation_asymmetry"] = _factor(
                valuation_score,
                "analyzer_mapped_signal",
                "Analyzer valuation_signal z-score is clipped to +/-3 and linearly mapped to 0-100 with 50 neutral.",
            )
        _append_analyzer_evidence(result, analyzer_context)
    elif analyzer_context.get("status") in {"error", "missing"}:
        missing = result.setdefault("missing_information", [])
        if isinstance(missing, list):
            missing.append(
                {
                    "field": "analyzer_context",
                    "severity": "medium",
                    "reason": str(
                        analyzer_context.get("error") or analyzer_context.get("reason") or "Analyzer unavailable."
                    ),
                }
            )
        confidence = _numeric_or_none(result.get("confidence"), minimum=0, maximum=1)
        result["confidence"] = min(confidence if confidence is not None else 0.45, 0.49)
        result["data_quality"] = {
            **_as_dict(result.get("data_quality")),
            "source_quality": "degraded",
            "quality": "degraded",
            "analyzer_context_quality": "failed",
        }

    result["factor_scores"] = factor_scores
    canonical_score = _canonical_score_from_factors(factor_scores)
    if canonical_score is not None:
        result["score"] = canonical_score
    if (
        _has_critical_missing(cast(list[dict[str, Any]], result.get("missing_information") or []))
        and result.get("action") == "buy"
    ):
        result["action"] = "watch"
    return result


def _deterministic_evaluation(context: dict[str, Any], *, reason: str | None = None) -> dict[str, Any]:
    idea = context["idea"]
    ticker = context["ticker"]
    equity_security = _is_equity_security_idea(_as_dict(idea))
    notes = str(idea.get("user_notes") or "").strip()
    overview = str(context.get("overview_content") or "").strip()
    thesis = str(context.get("thesis_content") or "").strip()
    management_quality = str(context.get("management_quality_content") or "").strip()
    tool_errors = list(context.get("tool_errors") or [])

    missing: list[dict[str, Any]] = []
    if equity_security and not overview:
        missing.append(
            {
                "field": "overview",
                "severity": "critical",
                "reason": "No uploaded business overview is available for the idea.",
            }
        )
    if not notes:
        missing.append(
            {
                "field": "user_notes",
                "severity": "medium",
                "reason": "No user rationale or reason-for-consideration notes are stored.",
            }
        )
    if not thesis:
        missing.append(
            {
                "field": "thesis",
                "severity": "medium",
                "reason": "No full thesis file is available for this non-position idea.",
            }
        )
    if equity_security and not management_quality:
        missing.append(
            {
                "field": "management_quality",
                "severity": "medium",
                "reason": "No explicit management-quality assessment is available for this idea.",
            }
        )
    if equity_security and context.get("industry_monitor", {}).get("ok") is not True:
        missing.append(
            {
                "field": "industry_management_commentary",
                "severity": "medium",
                "reason": "Industry monitor context was unavailable or failed.",
            }
        )

    signal_payload = context.get("signal_aggregator", {}).get("data")
    regime = ""
    if isinstance(signal_payload, dict):
        regime = str(signal_payload.get("regime") or signal_payload.get("regime_label") or "").lower()
    macro_score = 60
    macro_status = "mixed"
    if "risk-on" in regime or "risk_on" in regime:
        macro_score = 70
        macro_status = "supportive"
    elif "risk-off" in regime or "risk_off" in regime:
        macro_score = 35
        macro_status = "challenged"

    info_score = 50 + (12 if overview else 0) + (8 if notes else 0) + (6 if thesis else 0)
    factor_scores = {
        "macro_support": _factor(macro_score, macro_status, _asset_factor_rationale(context, "macro_support")),
        "industry_attractiveness": _factor(
            52 if context.get("industry_monitor", {}).get("ok") else 45,
            "mixed",
            _asset_factor_rationale(context, "industry_attractiveness"),
        ),
        "business_quality": _factor(
            min(info_score, 72),
            "incomplete" if not overview else "reviewable",
            _asset_factor_rationale(context, "business_quality", has_equity_doc=bool(overview)),
        ),
        "management_quality": _factor(
            62 if management_quality else 50 if not equity_security else 45,
            "reviewable" if management_quality else "not_applicable" if not equity_security else "incomplete",
            _asset_factor_rationale(context, "management_quality", has_equity_doc=bool(management_quality)),
        ),
        "valuation_asymmetry": _factor(
            45,
            "incomplete",
            _asset_factor_rationale(context, "valuation_asymmetry"),
            ["valuation", "expected upside/downside"]
            if equity_security
            else ["current level", "carry/roll", "downside scenario"],
        ),
        "portfolio_fit": _factor(
            55,
            "reviewable",
            _asset_factor_rationale(context, "portfolio_fit"),
        ),
    }
    score = round(sum(float(row["score"]) for row in factor_scores.values()) / len(factor_scores), 1)
    action = "watch"
    if _has_critical_missing(missing):
        action = "watch"
    elif score >= 72:
        action = "buy"
    elif score <= 40:
        action = "avoid"

    rationale_parts = [
        f"{ticker} is not investment-ready until the missing evidence is filled."
        if action == "watch"
        else f"{ticker} has enough stored evidence for review.",
    ]
    if reason:
        rationale_parts.append(f"Evaluator fallback reason: {reason}")
    if tool_errors:
        rationale_parts.append(
            "Some live/internal evidence sources failed, so the result is deliberately conservative."
        )

    data_quality = _source_quality_from_missing(missing, tool_errors)
    gate = apply_decision_quality_gates(
        None,
        current_action=action,
        recommendation_status="review_required" if data_quality["critical_data_quality"] != "ok" else "clear",
        data_quality=data_quality,
    )
    action = gate.final_action
    result = {
        "idea_id": idea["id"],
        "ticker": ticker,
        "evaluated_at": context["evaluated_at"],
        "action": action,
        "recommendation_status": gate.final_recommendation_status,
        "score": score,
        "confidence": 0.35 if missing else 0.55,
        "thesis_statement": f"{ticker} may be worth monitoring, but the evidence set is incomplete.",
        "rationale": " ".join(rationale_parts),
        "factor_scores": factor_scores,
        "missing_information": missing,
        "data_quality": data_quality,
        "evidence": [
            {"source": "user_notes", "summary": notes[:500]}
            if notes
            else {"source": "watchlist", "summary": "Idea exists."},
        ],
        "disconfirming_evidence": [
            {
                "source": "data_gap",
                "summary": "Critical overview, valuation, or management evidence may be missing.",
            }
        ],
        "catalyst": "Define a reason-now catalyst before treating this as actionable.",
        "invalidation": "Do not act if the missing overview, valuation, or management evidence cannot support the thesis.",
        "portfolio_fit": {"status": "needs_review", "note": "No position change is staged by evaluation."},
        "decision_quality": None,
        "decision_quality_gate": gate.model_dump(mode="json"),
    }
    result = _merge_analyzer_context_into_result(context, result)
    result["recommendation_record"] = _recommendation_record_from_result(idea, result)
    return result


def _normalize_llm_result(context: dict[str, Any], parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return _deterministic_evaluation(context, reason="model did not return JSON")

    idea = context["idea"]
    ticker = context["ticker"]
    action = normalize_action(parsed.get("action"), fallback="watch")
    missing = _normalize_missing_rows(parsed.get("missing_information"))
    tool_errors = list(context.get("tool_errors") or [])
    data_quality_raw = parsed.get("data_quality")
    data_quality: dict[str, Any] = cast(dict[str, Any], data_quality_raw) if isinstance(data_quality_raw, dict) else {}
    data_quality = {**_source_quality_from_missing(missing, tool_errors), **data_quality}
    data_quality["critical_data_quality"] = _quality_value(
        data_quality.get("critical_data_quality"), fallback="degraded" if missing else "ok"
    )
    data_quality["source_quality"] = _quality_value(
        data_quality.get("source_quality") or data_quality.get("quality"), fallback="degraded" if missing else "ok"
    )
    data_quality["quality"] = data_quality["source_quality"]
    if _has_critical_missing(missing) and action == "buy":
        action = "watch"

    factor_scores = _normalize_factor_scores(parsed.get("factor_scores"))
    score = _numeric_or_none(parsed.get("score"), minimum=0, maximum=100)
    confidence = _numeric_or_none(parsed.get("confidence"), minimum=0, maximum=1)
    decision_quality, decision_quality_errors = parse_decision_quality(parsed.get("decision_quality"))
    gate = apply_decision_quality_gates(
        decision_quality,
        current_action=action,
        recommendation_status=_recommendation_status(
            parsed.get("recommendation_status"),
            fallback="review_required" if _has_critical_missing(missing) else "clear",
        ),
        data_quality=data_quality,
        parse_errors=decision_quality_errors,
    )
    action = gate.final_action
    recommendation_status = gate.final_recommendation_status
    if confidence is not None and gate.confidence_cap is not None:
        confidence = min(confidence, gate.confidence_cap)
    result: dict[str, Any] = {
        "idea_id": idea["id"],
        "ticker": ticker,
        "evaluated_at": str(parsed.get("evaluated_at") or context["evaluated_at"]),
        "action": action,
        "recommendation_status": recommendation_status,
        "score": score,
        "confidence": confidence,
        "thesis_statement": str(parsed.get("thesis_statement") or f"{ticker} idea evaluation"),
        "rationale": str(parsed.get("rationale") or ""),
        "factor_scores": factor_scores,
        "missing_information": missing,
        "data_quality": data_quality,
        "evidence": _normalize_evidence_rows(parsed.get("evidence")),
        "disconfirming_evidence": _normalize_evidence_rows(parsed.get("disconfirming_evidence")),
        "catalyst": _join_structured_text(
            parsed.get("catalyst"),
            ("primary", "event_or_condition", "expected_timeframe", "why_now", "status", "source_evidence"),
        ),
        "invalidation": _join_structured_text(
            parsed.get("invalidation"),
            ("observable", "metric", "metric_or_event", "threshold", "timeframe", "implication"),
        ),
        "portfolio_fit": parsed.get("portfolio_fit") if isinstance(parsed.get("portfolio_fit"), dict) else {},
        "decision_quality": decision_quality.model_dump(mode="json") if decision_quality else None,
        "decision_quality_gate": gate.model_dump(mode="json"),
    }
    result = _merge_analyzer_context_into_result(context, result)
    if not result["rationale"]:
        result["rationale"] = "No rationale returned; review the factor scores and missing information before acting."
    if not result["disconfirming_evidence"]:
        result["disconfirming_evidence"] = [
            {"source": "required_review", "summary": "No explicit disconfirming evidence was returned."}
        ]
    if not result["invalidation"]:
        result["invalidation"] = "Reject the idea if the missing evidence cannot support the simple thesis."
    result["recommendation_record"] = _recommendation_record_from_result(idea, result)
    return result


def _attach_evaluator_diagnostics(result: dict[str, Any], diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    if not diagnostics:
        return result
    allowed = {
        "status",
        "provider",
        "model",
        "attempts",
        "web_search_status",
        "failure_reason",
    }
    safe = {key: diagnostics.get(key) for key in allowed if diagnostics.get(key) is not None}
    data_quality = _as_dict(result.get("data_quality"))
    data_quality["evaluator"] = safe
    result["data_quality"] = data_quality
    return result


def _call_llm_evaluator(context: dict[str, Any]) -> dict[str, Any]:
    from llm_utils import MODEL_HIGH, call_llm_json, has_llm_api_key

    if not has_llm_api_key():
        result = _deterministic_evaluation(context, reason="no configured LLM API key")
        return _attach_evaluator_diagnostics(
            result,
            {
                "status": "fallback",
                "attempts": 0,
                "web_search_status": "not_started",
                "failure_reason": "no configured LLM API key",
            },
        )

    system = "\n\n---\n\n".join(
        [
            _read_prompt("system.md"),
            _read_prompt("agent_system.md"),
            _read_prompt("recommendations_system.md"),
            _read_prompt("decision_quality.md"),
            (
                "You are evaluating independent watchlist ideas. Return only valid JSON. "
                "Do not invent missing evidence. If critical evidence is missing, action must be watch, avoid, or do_nothing. "
                "The canonical score denominator is exactly six factors. Do not add fundamental_momentum or "
                "price_momentum as top-level factors; they are analyzer diagnostics only. "
                "Use the idea asset and instrument_type from context: for non-equity ideas, interpret business_quality "
                "as thesis/instrument quality, management_quality as not-applicable unless an issuer/manager is central, "
                "industry_attractiveness as market-structure/regime attractiveness, and valuation_asymmetry as level, "
                "carry, roll, curve, spread, positioning, and scenario asymmetry. "
                "Each factor_scores entry must be an object with score, status, rationale, source, and missing. "
                "Never return a bare numeric factor score. Each missing_information reason must explain why the "
                "input matters and must not simply repeat the field."
            ),
        ]
    )
    prompt = (
        "Evaluate the investment idea below against the investment philosophy and recommendation contract. "
        "Use current web/news search only to fill high-level current context; cite sources inside evidence items when used. "
        "Return JSON with keys: thesis_statement, action, recommendation_status, score, confidence, rationale, "
        "factor_scores, missing_information, data_quality, evidence, disconfirming_evidence, catalyst, invalidation, "
        "portfolio_fit, decision_quality. "
        "factor_scores must include macro_support, industry_attractiveness, business_quality, management_quality, "
        "valuation_asymmetry, and portfolio_fit. Each factor must be an object: "
        "{score, status, rationale, source, missing}. Do not return numeric-only factor rows. "
        "missing_information rows must be {field, severity, reason}; reason must be a concrete explanation of "
        "how the missing input affects actionability, confidence, or sizing. "
        "Analyzer raw qualitative scores, when present, will override "
        "business/industry/management quality factors on the native 0-100 scale. Analyzer valuation_signal, when "
        "present, will be clipped to +/-3 and mapped to 0-100 with 50 neutral. action must use the shared "
        "canonical decision action vocabulary.\n\n"
        f"Context JSON:\n{json.dumps(context, default=str, sort_keys=True)}"
    )
    try:
        parsed, citations, _response, evaluator_diagnostics = call_llm_json(
            prompt=prompt,
            model=MODEL_HIGH,
            max_tokens=6000,
            system=system,
            enable_web_search=True,
            max_web_search_uses=4,
            json_schema=_idea_evaluator_json_schema(),
            json_schema_name="idea_evaluation_decision_quality",
        )
        if not isinstance(parsed, dict):
            reason = str(evaluator_diagnostics.get("failure_reason") or "model did not return JSON")
            result = _deterministic_evaluation(context, reason=reason)
            return _attach_evaluator_diagnostics(result, evaluator_diagnostics)
        result = _normalize_llm_result(context, parsed)
        result = _attach_evaluator_diagnostics(result, evaluator_diagnostics)
        if citations:
            evidence = result.setdefault("evidence", [])
            if isinstance(evidence, list):
                evidence.extend(
                    {"source": title, "url": url, "summary": "Live web source used by evaluator."}
                    for title, url in citations[:8]
                )
        return result
    except Exception as exc:
        result = _deterministic_evaluation(context, reason=str(exc))
        return _attach_evaluator_diagnostics(
            result,
            {
                "status": "fallback",
                "attempts": 0,
                "web_search_status": "error",
                "failure_reason": str(exc),
            },
        )


def _recommendation_record_from_result(idea: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    idea = _with_default_idea_instrument_fields(idea)
    action = str(result.get("action") or "watch").lower()
    now = str(result.get("evaluated_at") or _now())
    ticker = str(idea.get("ticker") or result.get("ticker") or "").upper()
    instrument = _idea_instrument_metadata(idea)
    data_quality_raw = result.get("data_quality")
    data_quality: dict[str, Any] = cast(dict[str, Any], data_quality_raw) if isinstance(data_quality_raw, dict) else {}
    missing_raw = result.get("missing_information")
    missing: list[Any] = missing_raw if isinstance(missing_raw, list) else []
    blocked_reasons = [
        str(row.get("reason") or row.get("field") or "Missing evidence")
        for row in missing
        if isinstance(row, dict) and str(row.get("severity") or "").lower() in CRITICAL_MISSING_SEVERITIES
    ]
    record = {
        "report_type": "daily",
        "as_of": now,
        "created_at": now,
        "source_report_path": "idea_watchlist",
        "source_json_path": f"idea:{idea.get('id')}",
        "stance": action,
        "recommendation_status": _recommendation_status(
            result.get("recommendation_status"),
            fallback="review_required" if blocked_reasons else "clear",
        ),
        "critical_data_quality": _quality_value(data_quality.get("critical_data_quality")),
        "blocked_reasons": blocked_reasons,
        "what_changed": ["Idea evaluator run accepted by user."],
        "do_nothing_rationale": result.get("rationale") if action == "do_nothing" else "",
        "action": action,
        "ticker": ticker,
        "instrument": instrument.get("price_symbol") or ticker,
        "asset": instrument.get("asset"),
        "instrument_type": instrument.get("instrument_type"),
        "price_symbol": instrument.get("price_symbol"),
        "contract_multiplier": instrument.get("contract_multiplier"),
        "fx_base_currency": instrument.get("fx_base_currency"),
        "fx_quote_currency": instrument.get("fx_quote_currency"),
        "currency": instrument.get("currency"),
        "country": instrument.get("country"),
        "exchange": instrument.get("exchange"),
        "horizon": "18-24 months",
        "target_change": "Initial one-third entry review" if action == "buy" else None,
        "rationale": str(result.get("rationale") or ""),
        "confidence": result.get("confidence"),
        "source_quality": _quality_value(data_quality.get("source_quality") or data_quality.get("quality")),
        "evidence": result.get("evidence") if isinstance(result.get("evidence"), list) else [],
        "disconfirming_evidence": (
            result.get("disconfirming_evidence") if isinstance(result.get("disconfirming_evidence"), list) else []
        ),
        "catalyst": result.get("catalyst"),
        "invalidation": result.get("invalidation"),
        "expected_onset_window": "18-24 months",
        "alternatives": [],
        "opportunity_cost": [],
        "source_quality_summary": data_quality,
        "decision_quality": result.get("decision_quality")
        if isinstance(result.get("decision_quality"), dict)
        else None,
        "decision_quality_gate": (
            result.get("decision_quality_gate") if isinstance(result.get("decision_quality_gate"), dict) else None
        ),
        "validation_status": IDEA_EVALUATION_SCHEMA_VERSION,
        "idempotency_key": f"idea:{idea.get('id')}:evaluation:{result.get('evaluated_at')}:action:{action}",
        "source_type": "idea_evaluator",
        "report_id": f"idea-evaluation-{idea.get('id')}-{_stable_hash(result)}",
    }
    try:
        from portfolio.policy_gate import attach_policy_gate_to_recommendation

        record, _gate = attach_policy_gate_to_recommendation(
            record,
            source_quality=data_quality,
            context={"source_type": "idea_evaluator", "source_id": str(idea.get("id"))},
        )
    except Exception:
        pass
    return record


def _cache_key(req: IdeaEvaluationRequest) -> str:
    idea = _get_idea(req.idea_id)
    if not idea:
        return f"idea_evaluation:{IDEA_EVALUATION_VERSION}:missing:{req.idea_id}"
    use_portfolio_context = bool(req.use_portfolio_context)
    token = {
        "id": idea.get("id"),
        "ticker": idea.get("ticker"),
        "instrument": _idea_instrument_metadata(idea),
        "company_name": idea.get("company_name"),
        "user_notes": idea.get("user_notes"),
        "tags": idea.get("tags"),
        "metadata": idea.get("metadata"),
        "use_portfolio_context": use_portfolio_context,
        "version": IDEA_EVALUATION_VERSION,
    }
    if not use_portfolio_context:
        token["analyzer_source"] = {"status": "disabled", "reason": "portfolio_context_disabled"}
        return f"idea_evaluation:{IDEA_EVALUATION_VERSION}:{_stable_hash(token)}"
    try:
        from portfolio.portfolio_optimizer.portfolio_analyzer import analyzer_source_cache_token

        try:
            token["analyzer_source"] = analyzer_source_cache_token(universe_mode="portfolio_plus_ideas")
        except TypeError:
            token["analyzer_source"] = analyzer_source_cache_token()
    except Exception:
        token["analyzer_source"] = {"status": "unavailable"}
    return f"idea_evaluation:{IDEA_EVALUATION_VERSION}:{_stable_hash(token)}"


def _confidence_level(confidence: float | None) -> str:
    value = float(confidence or 0)
    if value >= 0.75:
        return "high"
    if value >= 0.50:
        return "medium"
    return "low"


def _ranking_confidence(evaluation: dict[str, Any], value: Any | None = None) -> float:
    missing = evaluation.get("missing_information") if isinstance(evaluation.get("missing_information"), list) else []
    confidence = _numeric_or_none(value, minimum=0, maximum=1)
    if confidence is None:
        confidence = _numeric_or_none(evaluation.get("confidence"), minimum=0, maximum=1)
    if confidence is None:
        confidence = 0.35 if missing else 0.55
    if _has_critical_missing(cast(list[dict[str, Any]], missing)):
        confidence = min(confidence, 0.35)
    elif missing:
        confidence = min(confidence, 0.49)
    return round(float(confidence), 4)


def _action_priority(action: Any) -> int:
    return {
        "buy": 6,
        "add": 6,
        "short": 6,
        "sell": 6,
        "trim": 5,
        "reduce": 5,
        "exit": 5,
        "hedge": 4,
        "rebalance": 4,
        "research": 3,
        "watch": 2,
        "hold": 1,
        "do_nothing": 1,
        "avoid": 0,
    }.get(str(action or "").lower(), 1)


def _comparison_sort_key(evaluation: dict[str, Any]) -> tuple[int, float, float, str]:
    score = _numeric_or_none(evaluation.get("score"), minimum=0, maximum=100)
    confidence = _ranking_confidence(evaluation)
    return (
        -_action_priority(evaluation.get("action")),
        -(score if score is not None else -1),
        -confidence,
        str(evaluation.get("ticker") or ""),
    )


def _comparison_row_from_evaluation(
    evaluation: dict[str, Any], *, rank: int, rationale: str | None = None
) -> dict[str, Any]:
    confidence = _ranking_confidence(evaluation)
    return {
        "idea_id": str(evaluation["idea_id"]),
        "evaluation_id": str(evaluation["id"]),
        "ticker": str(evaluation.get("ticker") or "").upper(),
        "rank": rank,
        "action": str(evaluation.get("action") or "watch").lower(),
        "score": _numeric_or_none(evaluation.get("score"), minimum=0, maximum=100),
        "confidence": confidence,
        "confidence_level": _confidence_level(confidence),
        "rationale": rationale or str(evaluation.get("rationale") or "Ranked from the fresh idea evaluation."),
    }


def _deterministic_comparison_result(evaluations: list[dict[str, Any]], *, reason: str | None = None) -> dict[str, Any]:
    ranked = sorted(evaluations, key=_comparison_sort_key)
    rows = [_comparison_row_from_evaluation(evaluation, rank=index) for index, evaluation in enumerate(ranked, start=1)]
    summary = f"Ranked {len(rows)} actionable ideas by action, score, and evidence-adjusted confidence."
    if reason:
        summary = f"{summary} Ranking fallback reason: {reason}"
    return {"summary": summary, "rankings": rows}


def _normalize_comparison_result(evaluations: list[dict[str, Any]], parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return _deterministic_comparison_result(evaluations, reason="model did not return JSON")

    by_idea = {str(evaluation["idea_id"]): evaluation for evaluation in evaluations}
    by_ticker = {str(evaluation.get("ticker") or "").upper(): evaluation for evaluation in evaluations}
    rankings = parsed.get("rankings")
    raw_rows: list[Any] = rankings if isinstance(rankings, list) else []
    ordered: list[tuple[int, dict[str, Any]]] = []
    seen: set[str] = set()

    for fallback_rank, row in enumerate(raw_rows, start=1):
        if not isinstance(row, dict):
            continue
        row = cast(dict[str, Any], row)
        idea_id = str(row.get("idea_id") or "").strip() or None
        if idea_id is None:
            ticker = str(row.get("ticker") or "").upper()
            if ticker and ticker in by_ticker:
                idea_id = str(by_ticker[ticker]["idea_id"])
        if idea_id is None or idea_id in seen or idea_id not in by_idea:
            continue
        evaluation = by_idea[idea_id]
        try:
            rank = int(row.get("rank") or fallback_rank)
        except (TypeError, ValueError):
            rank = fallback_rank
        confidence = _ranking_confidence(evaluation, row.get("confidence"))
        ordered.append(
            (
                rank,
                {
                    "idea_id": str(evaluation["idea_id"]),
                    "evaluation_id": str(evaluation["id"]),
                    "ticker": str(evaluation.get("ticker") or "").upper(),
                    "rank": rank,
                    "action": str(evaluation.get("action") or "watch").lower(),
                    "score": _numeric_or_none(evaluation.get("score"), minimum=0, maximum=100),
                    "confidence": confidence,
                    "confidence_level": _confidence_level(confidence),
                    "rationale": str(row.get("rationale") or evaluation.get("rationale") or ""),
                },
            )
        )
        seen.add(idea_id)

    ordered.sort(key=lambda item: item[0])
    rows = [row for _rank, row in ordered]
    missing = [evaluation for evaluation in evaluations if str(evaluation["idea_id"]) not in seen]
    rows.extend(
        _comparison_row_from_evaluation(evaluation, rank=len(rows) + index)
        for index, evaluation in enumerate(sorted(missing, key=_comparison_sort_key), start=1)
    )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index

    summary = str(parsed.get("summary") or "").strip()
    if not summary:
        summary = f"Ranked {len(rows)} actionable ideas after fresh evaluations."
    return {"summary": summary, "rankings": rows}


def _call_llm_comparison_ranker(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    from llm_utils import MODEL_HIGH, call_llm_text, has_llm_api_key, parse_json_text

    if not evaluations:
        return _deterministic_comparison_result(evaluations)
    if not has_llm_api_key():
        return _deterministic_comparison_result(evaluations, reason="no configured LLM API key")

    system = "\n\n---\n\n".join(
        [
            _read_prompt("system.md"),
            _read_prompt("agent_system.md"),
            _read_prompt("recommendations_system.md"),
            (
                "You are ranking freshly evaluated watchlist ideas against one another. "
                "Return only valid JSON. Do not change the individual idea action or score."
            ),
        ]
    )
    compact = [
        {
            "idea_id": evaluation.get("idea_id"),
            "evaluation_id": evaluation.get("id"),
            "ticker": evaluation.get("ticker"),
            "action": evaluation.get("action"),
            "score": evaluation.get("score"),
            "confidence": evaluation.get("confidence"),
            "missing_information": evaluation.get("missing_information"),
            "thesis_statement": evaluation.get("thesis_statement"),
            "rationale": evaluation.get("rationale"),
            "factor_scores": evaluation.get("factor_scores"),
            "portfolio_fit": evaluation.get("portfolio_fit"),
            "analyzer_context": _compact_analyzer_risk_context(evaluation.get("analyzer_context")),
        }
        for evaluation in evaluations
    ]
    prompt = (
        "Rank these freshly evaluated actionable watchlist ideas relative to one another. "
        "Use the existing actions, scores, missing information, factor scores, and portfolio fit. "
        "Return JSON with keys: summary and rankings. rankings must be a list of objects with "
        "idea_id, rank, confidence, and rationale. confidence must be 0 to 1 and should reflect confidence "
        "in the comparative rank, not position sizing or trade execution certainty.\n\n"
        f"Fresh evaluations JSON:\n{json.dumps(compact, default=str, sort_keys=True)}"
    )
    try:
        text, _citations, _response = call_llm_text(
            prompt=prompt,
            model=MODEL_HIGH,
            max_tokens=2000,
            system=system,
            max_web_search_uses=0,
        )
        return _normalize_comparison_result(evaluations, parse_json_text(text))
    except Exception as exc:
        return _deterministic_comparison_result(evaluations, reason=str(exc))


def _compute_idea_evaluation_result(
    req: IdeaEvaluationRequest,
    *,
    job_id: str | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    idea = _get_idea(req.idea_id)
    if not idea:
        raise RuntimeError(f"No investment idea with id {req.idea_id}")
    use_portfolio_context = bool(req.use_portfolio_context)
    total = 5
    if callable(progress_callback):
        progress_callback("analyzer", 1, total)
    analyzer_result = (
        _compute_portfolio_plus_ideas_analyzer_result()
        if use_portfolio_context and _idea_analyzer_direction(idea) != "inactive"
        else None
    )
    analyzer_contexts = _analyzer_contexts_from_result(analyzer_result) if analyzer_result else {}
    analyzer_context = (
        _analyzer_context_for_idea(
            idea,
            analyzer_result=analyzer_result,
            analyzer_contexts=analyzer_contexts,
        )
        if use_portfolio_context
        else _disabled_analyzer_context_for_idea(idea)
    )
    if callable(progress_callback):
        progress_callback("context", 2, total)
    context = _build_context_for_evaluation(
        idea,
        analyzer_context=analyzer_context,
        use_portfolio_context=use_portfolio_context,
    )
    if callable(progress_callback):
        progress_callback("evaluation", 3, total)
    result = _call_llm_evaluator(context)
    if result.get("evaluation_schema_version") != IDEA_EVALUATION_SCHEMA_VERSION:
        result = _merge_analyzer_context_into_result(context, result)
        result["recommendation_record"] = _recommendation_record_from_result(idea, result)
    result["job_id"] = job_id
    if callable(progress_callback):
        progress_callback("persisting", 4, total)
    evaluation = _write_idea_evaluation(idea, result, job_id=job_id)
    idea = _update_idea_refs(req.idea_id, {"latest_evaluation_id": evaluation.get("id")})
    return {"idea": idea, "evaluation": evaluation, "result": result, "final_count": total}


def _actionable_ideas(scope_statuses: Sequence[str]) -> list[dict[str, Any]]:
    wanted = {str(status).lower() for status in scope_statuses} or set(ACTIONABLE_IDEA_STATUSES)
    return [
        idea
        for idea in _list_ideas(include_archived=False, limit=500)
        if str(idea.get("status") or "").lower() in wanted
    ]


def _compute_idea_comparison_evaluation_result(
    req: IdeaComparisonEvaluationRequest,
    *,
    job_id: str | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    scope_statuses = req.scope_statuses or list(ACTIONABLE_IDEA_STATUSES)
    ideas = _actionable_ideas(scope_statuses)
    total = max(len(ideas) + 3, 1)
    if callable(progress_callback):
        progress_callback("selecting", 0, total)
    if callable(progress_callback):
        progress_callback("analyzer", 1, total)
    use_portfolio_context = bool(req.use_portfolio_context)
    has_analyzer_ideas = use_portfolio_context and any(
        _idea_uses_portfolio_context(idea) and _idea_analyzer_direction(idea) != "inactive" for idea in ideas
    )
    analyzer_result = _compute_portfolio_plus_ideas_analyzer_result() if has_analyzer_ideas else None
    analyzer_contexts = _analyzer_contexts_from_result(analyzer_result) if analyzer_result else {}

    evaluations: list[dict[str, Any]] = []
    for index, idea in enumerate(ideas, start=1):
        if callable(progress_callback):
            progress_callback("evaluating", index + 1, total)
        idea_use_portfolio_context = use_portfolio_context and _idea_uses_portfolio_context(idea)
        analyzer_context = (
            _analyzer_context_for_idea(
                idea,
                analyzer_result=analyzer_result,
                analyzer_contexts=analyzer_contexts,
            )
            if idea_use_portfolio_context
            else _disabled_analyzer_context_for_idea(idea)
        )
        context = _build_context_for_evaluation(
            idea,
            analyzer_context=analyzer_context,
            use_portfolio_context=idea_use_portfolio_context,
        )
        result = _call_llm_evaluator(context)
        if result.get("evaluation_schema_version") != IDEA_EVALUATION_SCHEMA_VERSION:
            result = _merge_analyzer_context_into_result(context, result)
            result["recommendation_record"] = _recommendation_record_from_result(idea, result)
        result["job_id"] = job_id
        evaluation = _write_idea_evaluation(idea, result, job_id=job_id)
        _update_idea_refs(idea.get("id"), {"latest_evaluation_id": evaluation.get("id")})
        evaluations.append(evaluation)

    if callable(progress_callback):
        progress_callback("ranking", len(ideas) + 2, total)
    comparison = _call_llm_comparison_ranker(evaluations)

    if callable(progress_callback):
        progress_callback("persisting", len(ideas) + 3, total)
    run = _write_runtime_object(
        "IdeaComparisonRun",
        _comparison_uid(job_id or _stable_hash(comparison)),
        {
            "job_id": job_id,
            "scope_statuses": list(scope_statuses),
            "summary": str(comparison.get("summary") or ""),
            "rankings": cast(
                list[dict[str, Any]], comparison.get("rankings") if isinstance(comparison.get("rankings"), list) else []
            ),
            "raw_result": comparison,
        },
    )
    return {
        "run": run,
        "rankings": run.get("rankings", []),
        "evaluations": evaluations,
        "final_count": total,
    }


def _idea_detail(idea_id: str) -> dict[str, Any]:
    idea = _get_idea(idea_id)
    if not idea:
        raise NotFoundError("Investment idea", str(idea_id))
    equity_security = _is_equity_security_idea(idea)
    overview, overview_error = _read_state_text("investment_overviews", str(idea["ticker"]).upper())
    thesis, thesis_error = _read_state_text("investment_theses", str(idea["ticker"]).upper())
    management_quality, management_quality_error = (
        _read_management_quality_text(str(idea["ticker"]).upper()) if equity_security else (None, None)
    )
    overview_parsed = None
    if overview:
        try:
            from api.routers.overview import parse_overview_markdown

            overview_parsed = parse_overview_markdown(overview)
        except Exception:
            overview_parsed = None
    management_quality_parsed = None
    if management_quality:
        try:
            from api.routers.management_quality import parse_management_quality_markdown

            management_quality_parsed = parse_management_quality_markdown(management_quality)
        except Exception:
            management_quality_parsed = None
    idea_uid = _idea_uid(idea.get("id") or idea.get("object_uid") or idea.get("idea_id"))
    reads = OntologyRuntimeReadService()
    conviction_timeline = reads.conviction_history(
        str(idea.get("ticker") or ""),
        entity_type="investment_idea",
        entity_id=idea_uid,
        limit=20,
    )
    return {
        "idea": idea,
        "evaluations": _list_idea_evaluations(idea_id, limit=20),
        "lifecycle_history": _list_idea_lifecycle_events(idea_id, limit=20),
        "record_timeline": reads.record_timeline(
            context="idea",
            ticker=str(idea.get("ticker") or ""),
            idea_id=idea_id,
            limit=30,
        ),
        "conviction": {
            "current": idea.get("conviction"),
            "timeline": conviction_timeline,
        },
        "documents": {
            "overview_present": bool(overview),
            "overview_content": _safe_text(overview, max_len=120_000),
            "overview_parsed": overview_parsed,
            "overview_error": overview_error,
            "thesis_present": bool(thesis),
            "thesis_content": _safe_text(thesis, max_len=120_000),
            "thesis_error": thesis_error,
            "management_quality_present": bool(management_quality),
            "management_quality_content": _safe_text(management_quality, max_len=120_000),
            "management_quality_parsed": management_quality_parsed,
            "management_quality_error": management_quality_error,
        },
    }


@router.get("/ideas")
def list_ideas(status: str | None = None, include_archived: bool = False, limit: int = 200):
    ideas = _list_ideas(status=status, include_archived=include_archived, limit=limit)
    for idea in ideas:
        latest = _list_idea_evaluations(idea.get("id"), limit=1)
        idea["latest_evaluation"] = latest[0] if latest else None
    return {"ideas": ideas, "count": len(ideas)}


@router.post("/ideas")
def create_idea(body: IdeaCreateRequest):
    payload = _with_analyzer_direction_metadata(body.model_dump())
    payload["company_name"] = (
        _resolve_company_name(payload["ticker"], payload.get("company_name"))
        if _is_equity_security_idea(payload)
        else (str(payload.get("company_name") or "").strip() or None)
    )
    payload.update({"source_type": "user", "source_id": "ideas.create", "created_at": _now()})
    uid = _idea_uid(f"{payload['ticker']}:{_stable_hash(payload)}")
    idea = _write_runtime_object("InvestmentIdea", uid, payload)
    return _idea_detail(str(idea["id"]))


@router.post("/ideas/evaluate-all/async")
def start_idea_comparison_evaluation(body: dict[str, Any] | None = OPTIONAL_JSON_BODY):
    try:
        req = IdeaComparisonEvaluationRequest.model_validate(body or {})
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    ideas = _actionable_ideas(req.scope_statuses)
    if not ideas:
        raise ValidationError("No actionable ideas to evaluate.")

    row, _disposition = enqueue_registered_job(
        "idea_comparison_evaluation",
        req.model_dump(),
        cache_key=None,
        reuse_completed=False,
    )
    return enqueue_response(row, "/api/ideas/evaluate-all/async/{job_id}")


@router.get("/ideas/evaluate-all/async/{job_id}")
def get_idea_comparison_evaluation_job(job_id: str):
    try:
        return poll_registered_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job_id") from None


@router.get("/ideas/comparison-runs")
def list_idea_comparison_runs(limit: int = 20):
    runs = OntologyRuntimeReadService().list_objects("IdeaComparisonRun", limit=limit)
    return {"runs": runs, "count": len(runs)}


@router.get("/ideas/comparison-runs/{run_id}")
def get_idea_comparison_run(run_id: str):
    key = run_id if str(run_id).startswith("idea_comparison_run:") else _comparison_uid(run_id)
    run = OntologyRuntimeReadService().get(key)
    if not run:
        raise NotFoundError("Idea comparison run", run_id)
    return run


@router.get("/ideas/{idea_id}")
def get_idea(idea_id: str):
    return _idea_detail(idea_id)


@router.put("/ideas/{idea_id}")
def update_idea(idea_id: str, body: IdeaUpdateRequest):
    current = _get_idea(idea_id)
    if not current:
        raise NotFoundError("Investment idea", str(idea_id))
    before_idea = json.loads(json.dumps(current, default=str))
    updates = body.model_dump(exclude_unset=True)
    if "analyzer_direction" in updates or "use_portfolio_context" in updates:
        metadata = current.get("metadata") if isinstance(current.get("metadata"), dict) else {}
        next_metadata = {**cast(dict[str, Any], metadata)}
        if "analyzer_direction" in updates:
            next_metadata["analyzer_direction"] = _normalize_analyzer_direction(updates.pop("analyzer_direction"))
        if "use_portfolio_context" in updates:
            next_metadata["use_portfolio_context"] = _normalize_use_portfolio_context(
                updates.pop("use_portfolio_context"),
                default=_idea_uses_portfolio_context(current),
            )
        current["metadata"] = next_metadata
    if IDEA_INSTRUMENT_FIELDS.intersection(updates):
        instrument_updates = dict(updates)
        if (
            "ticker" in updates
            and "price_symbol" not in updates
            and str(current.get("price_symbol") or current.get("ticker") or "").upper()
            == str(current.get("ticker") or "").upper()
        ):
            instrument_updates["price_symbol"] = updates["ticker"]
        try:
            updates.update(_normalized_idea_instrument(instrument_updates, base=current))
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        if "company_name" not in updates and _is_equity_security_idea({**current, **updates}):
            updates["company_name"] = _resolve_company_name(updates["ticker"], current.get("company_name"))
    current.update(updates)
    current["updated_at"] = _now()
    current.pop("_meta", None)
    current.pop("id", None)
    current.pop("object_uid", None)
    _write_runtime_object("InvestmentIdea", _idea_uid(idea_id), current)
    _record_idea_lifecycle_changes(before_idea, current, source_id=f"ideas.update:{idea_id}")
    return _idea_detail(idea_id)


@router.delete("/ideas/{idea_id}")
def delete_idea(idea_id: str):
    idea = _get_idea(idea_id)
    if not idea:
        raise NotFoundError("Investment idea", str(idea_id))
    deleted_count = _delete_runtime_idea(idea_id, idea)
    return {
        "status": "deleted",
        "deleted": True,
        "idea_id": str(idea.get("id") or idea.get("object_uid") or idea_id),
        "deleted_count": deleted_count,
    }


@router.post("/ideas/{idea_id}/evaluate/async")
def start_idea_evaluation(idea_id: str, body: dict[str, Any] | None = OPTIONAL_JSON_BODY):
    idea = _get_idea(idea_id)
    if not idea:
        raise NotFoundError("Investment idea", str(idea_id))
    payload = body or {}
    force_refresh = bool(payload.get("force_refresh", False))
    use_portfolio_context = _normalize_use_portfolio_context(
        payload.get("use_portfolio_context") if "use_portfolio_context" in payload else None,
        default=_idea_uses_portfolio_context(idea),
    )
    req = IdeaEvaluationRequest(
        idea_id=idea_id,
        force_refresh=force_refresh,
        use_portfolio_context=use_portfolio_context,
    )
    row, _disposition = enqueue_registered_job(
        "idea_evaluation",
        req.model_dump(),
        cache_key=None if force_refresh else _cache_key(req),
        reuse_completed=not force_refresh,
    )
    before_idea = json.loads(json.dumps(idea, default=str))
    idea.update({"latest_job_id": str(row.get("job_id") or ""), "updated_at": _now()})
    if str(row.get("status") or "") in {"queued", "running"}:
        idea["status"] = "researching"
    idea.pop("_meta", None)
    idea.pop("id", None)
    idea.pop("object_uid", None)
    _write_runtime_object("InvestmentIdea", _idea_uid(idea_id), idea)
    if before_idea.get("status") != idea.get("status"):
        _record_idea_lifecycle_changes(
            before_idea,
            idea,
            event_type="evaluation_started",
            source_id=f"ideas.evaluate:{idea_id}",
        )
    return enqueue_response(row, "/api/ideas/evaluate/async/{job_id}")


@router.get("/ideas/evaluate/async/{job_id}")
def get_idea_evaluation_job(job_id: str):
    try:
        return poll_registered_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job_id") from None


@router.post("/ideas/{idea_id}/evaluations/{evaluation_id}/accept")
def accept_idea_evaluation(
    idea_id: str,
    evaluation_id: str,
    actor: ActorDep,
    body: IdeaAcceptRequest | None = None,
):
    idea = _get_idea(idea_id)
    evaluation = _get_idea_evaluation(evaluation_id)
    if not idea:
        raise NotFoundError("Investment idea", str(idea_id))
    if not evaluation or str(evaluation.get("idea_id") or "") != str(idea.get("id")):
        raise NotFoundError("Idea evaluation", str(evaluation_id))

    record = (
        evaluation.get("recommendation_record") if isinstance(evaluation.get("recommendation_record"), dict) else {}
    )
    if not record:
        record = _recommendation_record_from_result(idea, evaluation)
    from ontology.command_service import OntologyCommandContext, OntologyCommandService

    context = OntologyCommandContext(
        actor=actor,
        source_type="user",
        source_id=f"ideas.accept:{idea_id}:{evaluation_id}",
    )
    recommendation = OntologyCommandService().propose_action(
        "create_recommendation",
        {"record": record},
        context,
        reason=(body.note if body else None) or f"Accept idea evaluator recommendation for {idea.get('ticker')}",
    )
    recommendation_id = recommendation["id"]
    recommendation_approval_id = recommendation_id if str(recommendation_id).startswith("approval:") else None
    recommendation_object_id = recommendation_id if str(recommendation_id).startswith("recommendation:") else None
    accepted_recommendation_ref = recommendation_approval_id or recommendation_object_id or recommendation_id
    evaluation_uid = _evaluation_uid(evaluation_id)

    action_proposal = None
    action_error = None
    action = str(evaluation.get("action") or "").lower()
    if action in DECISION_ACTIONABLE_ACTIONS | {"watch", "research"}:
        if action in {"buy", "add", "short", "sell"}:
            action_type = "enter"
        elif action in {"trim", "reduce", "rebalance"}:
            action_type = "resize"
        elif action == "exit":
            action_type = "exit"
        elif action == "hedge":
            action_type = "hedge"
        else:
            action_type = "research"
        recommendation_ref = accepted_recommendation_ref
        recommendation_ref_label = "approval" if recommendation_approval_id else "record"
        description = (
            f"{'Evaluate position change' if action in DECISION_ACTIONABLE_ACTIONS else 'Research remaining evidence'} for "
            f"{idea['ticker']} from idea evaluator recommendation {recommendation_ref_label} {recommendation_ref}."
        )
        missing = (
            evaluation.get("missing_information") if isinstance(evaluation.get("missing_information"), list) else []
        )
        if missing:
            description += " Missing evidence: " + "; ".join(
                str(row.get("field") or row) for row in missing[:4] if isinstance(row, dict) or row
            )
        try:
            instrument = _idea_instrument_metadata(idea)
            action_proposal = stage_api_action(
                "create_action_item",
                {
                    "recommendation_id": recommendation_ref,
                    "ticker": idea.get("ticker"),
                    "asset": instrument.get("asset"),
                    "instrument_type": instrument.get("instrument_type"),
                    "price_symbol": instrument.get("price_symbol"),
                    "description": description,
                    "action_type": action_type,
                    "urgency": "normal" if action in DECISION_ACTIONABLE_ACTIONS else "low",
                },
                source_id=f"ideas.accept:{idea_id}:{evaluation_id}",
                actor=actor,
                reason=(body.note if body else None)
                or f"Accept idea evaluator recommendation for {idea.get('ticker')}",
            )
        except Exception as exc:
            action_error = str(exc)

    accepted_payload = {
        **evaluation,
        "accepted": True,
        "accepted_by": "user",
        "accepted_at": _now(),
        "recommendation_id": recommendation_object_id,
        "approval_id": recommendation_approval_id,
        "recommendation_approval_id": recommendation_approval_id,
        "action_approval_id": action_proposal["approval_id"] if action_proposal else None,
    }
    accepted_payload.pop("_meta", None)
    accepted_payload.pop("id", None)
    accepted_payload.pop("object_uid", None)
    accepted = _write_runtime_object("IdeaEvaluation", evaluation_uid, accepted_payload)
    _update_idea_refs(
        idea_id,
        {
            "accepted_recommendation_id": accepted_recommendation_ref,
            "latest_evaluation_id": evaluation_uid,
        },
    )
    _write_idea_lifecycle_event(
        idea,
        event_type="evaluation_accepted",
        changed_fields=["evaluation_accepted"],
        before={},
        after={
            "evaluation_id": evaluation_uid,
            "action": evaluation.get("action"),
            "recommendation_id": recommendation_object_id or recommendation_id,
        },
        reason=(body.note if body else None),
        evaluation_id=evaluation_uid,
        recommendation_id=str(recommendation_object_id or recommendation_id or ""),
        approval_id=str(recommendation_approval_id or "") or None,
        action_approval_id=str(action_proposal["approval_id"]) if action_proposal else None,
        source_id=f"ideas.accept:{idea_id}:{evaluation_id}",
    )
    return {
        "status": "accepted",
        "idea": _get_idea(idea_id),
        "evaluation": accepted,
        "recommendation": recommendation,
        "action_proposal": action_proposal,
        "action_error": action_error,
    }


@router.post("/ideas/{idea_id}/reject")
def reject_idea(idea_id: str, body: IdeaRejectRequest | None = None):
    idea = _get_idea(idea_id)
    if not idea:
        raise NotFoundError("Investment idea", str(idea_id))
    before_idea = json.loads(json.dumps(idea, default=str))
    metadata = idea.get("metadata") if isinstance(idea.get("metadata"), dict) else {}
    next_metadata = {
        **cast(dict[str, Any], metadata),
        "rejection_note": body.note if body else None,
        "rejected_at": _now(),
    }
    idea.update({"status": "rejected", "metadata": next_metadata})
    idea.pop("_meta", None)
    idea.pop("id", None)
    idea.pop("object_uid", None)
    idea = _write_runtime_object("InvestmentIdea", _idea_uid(idea_id), idea)
    _record_idea_lifecycle_changes(
        before_idea,
        idea,
        event_type="rejected",
        reason=body.note if body else None,
        source_id=f"ideas.reject:{idea_id}",
    )
    return {"status": "rejected", "idea": idea}
