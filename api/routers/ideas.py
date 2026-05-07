"""Investment idea watchlist and evaluator endpoints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field, field_validator

from api.action_execution import stage_api_action
from api.async_job_runner import enqueue_registered_job, enqueue_response, poll_registered_job
from api.exceptions import NotFoundError, ValidationError
from ontology.domain_write_service import ontology_primary_writes_enabled
from ontology.object_service import OntologyObjectService
from ontology.policy import actor_to_dict, admin_actor
from ontology.runtime_read_service import OntologyRuntimeReadService

router = APIRouter()

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "auto_report" / "prompts"
IDEA_EVALUATION_VERSION = "v2_analyzer_context"
IDEA_EVALUATION_SCHEMA_VERSION = "idea_evaluator_v2_analyzer_context"
IDEA_ACTIONS = {"buy", "watch", "avoid", "do_nothing"}
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


class IdeaCreateRequest(BaseModel):
    ticker: str
    company_name: str | None = None
    user_notes: str | None = None
    tags: list[str] = Field(default_factory=list)
    status: IdeaStatus = "watching"
    analyzer_direction: Literal["inactive", "long", "short"] = "inactive"

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        ticker = str(value or "").strip().upper()
        if not ticker:
            raise ValueError("Ticker cannot be empty.")
        return ticker


class IdeaUpdateRequest(BaseModel):
    ticker: str | None = None
    company_name: str | None = None
    user_notes: str | None = None
    tags: list[str] | None = None
    status: IdeaStatus | None = None
    analyzer_direction: Literal["inactive", "long", "short"] | None = None

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


class IdeaComparisonEvaluationRequest(BaseModel):
    scope_statuses: list[IdeaComparisonStatus] = Field(default_factory=lambda: list(ACTIONABLE_IDEA_STATUSES))

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


def _idea_analyzer_direction(idea: dict[str, Any]) -> str:
    metadata = idea.get("metadata") if isinstance(idea.get("metadata"), dict) else {}
    return _normalize_analyzer_direction(cast(dict[str, Any], metadata).get("analyzer_direction"))


def _with_analyzer_direction_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    direction = _normalize_analyzer_direction(payload.pop("analyzer_direction", "inactive"))
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    payload["metadata"] = {**cast(dict[str, Any], metadata), "analyzer_direction": direction}
    return payload


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


def _comparison_uid(value: Any) -> str:
    text = str(value or "").strip()
    return text if text.startswith("idea_comparison_run:") else f"idea_comparison_run:{text}"


def _write_runtime_object(object_type: str, uid: str, props: dict[str, Any]) -> dict[str, Any]:
    now = _now()
    payload = {**props, "ontology_run_id": "operational"}
    if not ontology_primary_writes_enabled():
        return _write_legacy_runtime_object(object_type, uid, payload)
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


def _object_uid_from_row(row: dict[str, Any]) -> str:
    props = _object_props(row) or {}
    return str(row.get("object_uid") or props.get("object_uid") or props.get("id") or "").strip()


def _relation_uid_from_row(row: dict[str, Any]) -> str:
    meta = row.get("_meta") if isinstance(row.get("_meta"), dict) else {}
    temporal = meta.get("temporal") if isinstance(meta.get("temporal"), dict) else {}
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
    if ontology_primary_writes_enabled():
        return _delete_ontology_runtime_idea(idea)
    from portfolio import core_db

    numeric_id = _required_legacy_numeric_id(idea.get("id") or idea_id)
    deleted = core_db.delete_investment_idea(numeric_id)
    return 1 if deleted else 0


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


def _write_legacy_runtime_object(object_type: str, uid: str, props: dict[str, Any]) -> dict[str, Any]:
    from portfolio import core_db

    if object_type == "InvestmentIdea":
        idea_id = _legacy_numeric_id(uid)
        if idea_id is None:
            return core_db.create_investment_idea(
                str(props.get("ticker") or ""),
                company_name=props.get("company_name"),
                user_notes=props.get("user_notes"),
                tags=cast(list[str] | None, props.get("tags") if isinstance(props.get("tags"), list) else None),
                status=str(props.get("status") or "watching"),
                source_type=str(props.get("source_type") or "user"),
                source_id=props.get("source_id"),
                metadata=cast(
                    dict[str, Any] | None, props.get("metadata") if isinstance(props.get("metadata"), dict) else None
                ),
            )
        return core_db.update_investment_idea(
            idea_id,
            ticker=props.get("ticker"),
            company_name=props.get("company_name"),
            status=props.get("status"),
            user_notes=props.get("user_notes"),
            tags=cast(list[str], props.get("tags") if isinstance(props.get("tags"), list) else []),
            latest_job_id=props.get("latest_job_id"),
            latest_evaluation_id=_legacy_numeric_id(props.get("latest_evaluation_id")),
            accepted_recommendation_id=_legacy_numeric_id(props.get("accepted_recommendation_id")),
            metadata=cast(dict[str, Any], props.get("metadata") if isinstance(props.get("metadata"), dict) else {}),
        )
    if object_type == "IdeaEvaluation":
        evaluation_id = _legacy_numeric_id(uid)
        if evaluation_id is not None and props.get("accepted_at"):
            return core_db.mark_idea_evaluation_accepted(
                evaluation_id,
                recommendation_id=_required_legacy_numeric_id(props.get("recommendation_id")),
                action_approval_id=_legacy_numeric_id(props.get("action_approval_id")),
                accepted_by=str(props.get("accepted_by") or "user"),
            )
        return core_db.create_idea_evaluation(
            _required_legacy_numeric_id(props.get("idea_id")),
            props,
            job_id=props.get("job_id"),
        )
    if object_type == "IdeaComparisonRun":
        rankings: list[dict[str, Any]] = []
        ranking_rows = props.get("rankings")
        for row in ranking_rows if isinstance(ranking_rows, list) else []:
            if not isinstance(row, dict):
                continue
            rankings.append(
                {
                    **row,
                    "idea_id": _required_legacy_numeric_id(row.get("idea_id")),
                    "evaluation_id": _required_legacy_numeric_id(row.get("evaluation_id")),
                }
            )
        return core_db.create_idea_comparison_run(
            job_id=props.get("job_id"),
            scope_statuses=cast(
                list[str] | None, props.get("scope_statuses") if isinstance(props.get("scope_statuses"), list) else None
            ),
            summary=str(props.get("summary") or ""),
            rankings=rankings,
            raw_result=cast(
                dict[str, Any] | None, props.get("raw_result") if isinstance(props.get("raw_result"), dict) else None
            ),
            run_id=_legacy_text_id(uid),
        )
    raise ValueError(f"Unsupported legacy idea runtime object type: {object_type}")


def _legacy_text_id(value: Any) -> str:
    text = str(value or "").strip()
    return text.split(":", 1)[1] if ":" in text else text


def _legacy_numeric_id(value: Any) -> int | None:
    text = _legacy_text_id(value)
    if not text:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _required_legacy_numeric_id(value: Any) -> int:
    numeric = _legacy_numeric_id(value)
    if numeric is None:
        raise ValueError(f"Expected numeric legacy id, got {value!r}")
    return numeric


def _get_idea(idea_id: Any) -> dict[str, Any] | None:
    reads = OntologyRuntimeReadService()
    text = str(idea_id or "").strip()
    if ontology_primary_writes_enabled():
        return reads.get(text) if text.startswith("investment_idea:") else None
    return reads.get(_idea_uid(text))


def _list_ideas(*, status: str | None = None, include_archived: bool = False, limit: int = 200) -> list[dict[str, Any]]:
    filters = {"status": status} if status else None
    ideas = OntologyRuntimeReadService().list_objects("InvestmentIdea", filters=filters, limit=limit)
    if not include_archived:
        ideas = [idea for idea in ideas if str(idea.get("status") or "").lower() != "archived"]
    return ideas


def _list_idea_evaluations(idea_id: Any | None = None, *, limit: int = 100) -> list[dict[str, Any]]:
    filters = {"idea_id": _idea_uid(idea_id)} if idea_id is not None else None
    rows = OntologyRuntimeReadService().list_objects("IdeaEvaluation", filters=filters, limit=limit)
    return sorted(rows, key=lambda row: str(row.get("evaluated_at") or ""), reverse=True)


def _get_idea_evaluation(evaluation_id: Any) -> dict[str, Any] | None:
    text = str(evaluation_id or "").strip()
    if ontology_primary_writes_enabled():
        return OntologyRuntimeReadService().get(text) if text.startswith("idea_evaluation:") else None
    return OntologyRuntimeReadService().get(_evaluation_uid(text))


def _write_idea_evaluation(
    idea: dict[str, Any], result: dict[str, Any], *, job_id: str | None = None
) -> dict[str, Any]:
    uid = _evaluation_uid(_stable_hash({"idea_id": idea.get("id"), "result": result, "job_id": job_id}))
    payload = {
        **result,
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
            out.append({"field": row.strip(), "severity": "medium", "reason": row.strip()})
        elif isinstance(row, dict):
            field = str(row.get("field") or row.get("name") or "unspecified").strip()
            if not field:
                field = "unspecified"
            severity = str(row.get("severity") or "medium").strip().lower()
            out.append(
                {
                    "field": field,
                    "severity": severity,
                    "reason": str(row.get("reason") or row.get("message") or "").strip(),
                }
            )
    return out


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
    if ontology_primary_writes_enabled():
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
    return _read_state_text("investment_management_quality", ticker)


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
    raw_result = analyzer_result.get("raw_result") if isinstance(analyzer_result.get("raw_result"), dict) else {}
    weights = {
        str(row.get("ticker") or "").strip().upper(): row
        for row in _records_from_table(cast(dict[str, Any], raw_result).get("weights_df"))
        if str(row.get("ticker") or "").strip()
    }
    course = raw_result.get("course_of_action") if isinstance(raw_result.get("course_of_action"), dict) else {}
    action_rows = course.get("action_queue") if isinstance(course.get("action_queue"), list) else []
    actions = {
        str(row.get("ticker") or "").strip().upper(): cast(dict[str, Any], row)
        for row in action_rows
        if isinstance(row, dict) and str(row.get("ticker") or "").strip()
    }
    summary = course.get("summary") if isinstance(course.get("summary"), dict) else {}
    source_timestamp = summary.get("as_of") or raw_result.get("timestamp")

    contexts: dict[str, dict[str, Any]] = {}
    for ticker, row in weights.items():
        action = actions.get(ticker, {})
        contexts[ticker] = {
            "status": "available",
            "ticker": ticker,
            "source_timestamp": source_timestamp,
            "source_type": row.get("source_type") or action.get("source_type") or "portfolio",
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


def _build_context(idea: dict[str, Any], analyzer_context: dict[str, Any] | None = None) -> dict[str, Any]:
    ticker = str(idea["ticker"]).upper()
    overview, overview_error = _read_state_text("investment_overviews", ticker)
    thesis, thesis_error = _read_state_text("investment_theses", ticker)
    management_quality, management_quality_error = _read_management_quality_text(ticker)

    portfolio = _safe_tool("get_portfolio", {"include_hedges": True})
    signal_aggregator = _safe_tool("get_signal_aggregator", {"include_history": False, "lookback_weeks": 156})
    industry_monitor = _safe_tool("get_industry_monitor", {"refresh": False})
    dossier = _safe_tool("get_dossier", {"ticker": ticker})

    tool_errors = [
        f"{label}: {payload.get('error')}"
        for label, payload in {
            "portfolio": portfolio,
            "signal_aggregator": signal_aggregator,
            "industry_monitor": industry_monitor,
            "dossier": dossier,
        }.items()
        if not payload.get("ok")
    ]
    if overview_error:
        tool_errors.append(overview_error)
    if thesis_error:
        tool_errors.append(thesis_error)
    if management_quality_error:
        tool_errors.append(management_quality_error)
    analyzer_payload = analyzer_context if isinstance(analyzer_context, dict) else _analyzer_context_for_idea(idea)
    analyzer_status = str(analyzer_payload.get("status") or "")
    if analyzer_status in {"error", "missing"}:
        tool_errors.append(f"analyzer_context: {analyzer_payload.get('error') or analyzer_payload.get('reason')}")

    return {
        "idea": idea,
        "ticker": ticker,
        "analyzer_context": analyzer_payload,
        "overview_content": _safe_text(overview),
        "thesis_content": _safe_text(thesis),
        "management_quality_content": _safe_text(management_quality),
        "portfolio": portfolio,
        "signal_aggregator": signal_aggregator,
        "industry_monitor": industry_monitor,
        "dossier": dossier,
        "tool_errors": tool_errors,
        "evaluated_at": _now(),
    }


def _build_context_for_evaluation(
    idea: dict[str, Any], analyzer_context: dict[str, Any] | None = None
) -> dict[str, Any]:
    import inspect

    if "analyzer_context" in inspect.signature(_build_context).parameters:
        return _build_context(idea, analyzer_context=analyzer_context)
    context = _build_context(idea)
    context["analyzer_context"] = analyzer_context or {"status": "inactive", "ticker": context.get("ticker")}
    return context


def _factor(score: float, status: str, rationale: str, missing: list[str] | None = None) -> dict[str, Any]:
    return {
        "score": max(0, min(100, round(float(score), 1))),
        "status": status,
        "rationale": rationale,
        "missing": missing or [],
    }


def _ensure_canonical_factor_rows(factor_scores: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in CANONICAL_IDEA_FACTORS:
        row = factor_scores.get(key)
        if isinstance(row, dict):
            score = _numeric_or_none(row.get("score"), minimum=0, maximum=100)
            normalized[key] = {
                **row,
                "score": 50.0 if score is None else round(score, 1),
                "status": str(row.get("status") or "reviewable"),
            }
        else:
            normalized[key] = _factor(50, "missing", f"{key} was not returned by the evaluator.")
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
    analyzer_context = context.get("analyzer_context") if isinstance(context.get("analyzer_context"), dict) else {}
    result["evaluation_schema_version"] = IDEA_EVALUATION_SCHEMA_VERSION
    result["analyzer_context"] = analyzer_context

    factor_scores_raw = result.get("factor_scores")
    factor_scores = _ensure_canonical_factor_rows(
        cast(dict[str, Any], factor_scores_raw) if isinstance(factor_scores_raw, dict) else {}
    )
    if analyzer_context.get("status") == "available":
        row = analyzer_context.get("row") if isinstance(analyzer_context.get("row"), dict) else {}
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
        data_quality = result.get("data_quality") if isinstance(result.get("data_quality"), dict) else {}
        result["data_quality"] = {
            **data_quality,
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
    notes = str(idea.get("user_notes") or "").strip()
    overview = str(context.get("overview_content") or "").strip()
    thesis = str(context.get("thesis_content") or "").strip()
    management_quality = str(context.get("management_quality_content") or "").strip()
    tool_errors = list(context.get("tool_errors") or [])

    missing: list[dict[str, Any]] = []
    if not overview:
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
    if not management_quality:
        missing.append(
            {
                "field": "management_quality",
                "severity": "medium",
                "reason": "No explicit management-quality assessment is available for this idea.",
            }
        )
    if context.get("industry_monitor", {}).get("ok") is not True:
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
        "macro_support": _factor(
            macro_score, macro_status, "Derived from the internal signal aggregator when available."
        ),
        "industry_attractiveness": _factor(
            52 if context.get("industry_monitor", {}).get("ok") else 45,
            "mixed",
            "Industry evidence requires manual review of monitor context and uploaded materials.",
        ),
        "business_quality": _factor(
            min(info_score, 72),
            "incomplete" if not overview else "reviewable",
            "Business quality is not fully scoreable without a complete overview and thesis.",
        ),
        "management_quality": _factor(
            62 if management_quality else 45,
            "reviewable" if management_quality else "incomplete",
            (
                "Management quality is supported by the uploaded management-quality assessment."
                if management_quality
                else "Management quality requires explicit track record, capital allocation, and transcript evidence."
            ),
        ),
        "valuation_asymmetry": _factor(
            45,
            "incomplete",
            "No dedicated valuation/asymmetry evidence was computed in deterministic fallback mode.",
            ["valuation", "expected upside/downside"],
        ),
        "portfolio_fit": _factor(
            55,
            "reviewable",
            "Portfolio fit uses current holdings context when available; concentration still needs human review.",
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
    result = {
        "idea_id": idea["id"],
        "ticker": ticker,
        "evaluated_at": context["evaluated_at"],
        "action": action,
        "recommendation_status": "review_required" if data_quality["critical_data_quality"] != "ok" else "clear",
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
    }
    result = _merge_analyzer_context_into_result(context, result)
    result["recommendation_record"] = _recommendation_record_from_result(idea, result)
    return result


def _normalize_llm_result(context: dict[str, Any], parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return _deterministic_evaluation(context, reason="model did not return JSON")

    idea = context["idea"]
    ticker = context["ticker"]
    action = str(parsed.get("action") or "watch").strip().lower()
    if action not in IDEA_ACTIONS:
        action = "watch"
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

    factor_scores_raw = parsed.get("factor_scores")
    factor_scores: dict[str, Any] = (
        cast(dict[str, Any], factor_scores_raw) if isinstance(factor_scores_raw, dict) else {}
    )
    score = _numeric_or_none(parsed.get("score"), minimum=0, maximum=100)
    confidence = _numeric_or_none(parsed.get("confidence"), minimum=0, maximum=1)
    result: dict[str, Any] = {
        "idea_id": idea["id"],
        "ticker": ticker,
        "evaluated_at": str(parsed.get("evaluated_at") or context["evaluated_at"]),
        "action": action,
        "recommendation_status": _recommendation_status(
            parsed.get("recommendation_status"),
            fallback="review_required" if _has_critical_missing(missing) else "clear",
        ),
        "score": score,
        "confidence": confidence,
        "thesis_statement": str(parsed.get("thesis_statement") or f"{ticker} idea evaluation"),
        "rationale": str(parsed.get("rationale") or ""),
        "factor_scores": factor_scores,
        "missing_information": missing,
        "data_quality": data_quality,
        "evidence": parsed.get("evidence") if isinstance(parsed.get("evidence"), list) else [],
        "disconfirming_evidence": (
            parsed.get("disconfirming_evidence") if isinstance(parsed.get("disconfirming_evidence"), list) else []
        ),
        "catalyst": parsed.get("catalyst"),
        "invalidation": parsed.get("invalidation"),
        "portfolio_fit": parsed.get("portfolio_fit") if isinstance(parsed.get("portfolio_fit"), dict) else {},
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


def _call_llm_evaluator(context: dict[str, Any]) -> dict[str, Any]:
    from llm_utils import MODEL_HIGH, call_llm_text, has_llm_api_key, parse_json_text

    if not has_llm_api_key():
        return _deterministic_evaluation(context, reason="no configured LLM API key")

    system = "\n\n---\n\n".join(
        [
            _read_prompt("system.md"),
            _read_prompt("agent_system.md"),
            _read_prompt("recommendations_system.md"),
            (
                "You are evaluating independent watchlist ideas. Return only valid JSON. "
                "Do not invent missing evidence. If critical evidence is missing, action must be watch, avoid, or do_nothing. "
                "The canonical score denominator is exactly six factors. Do not add fundamental_momentum or "
                "price_momentum as top-level factors; they are analyzer diagnostics only."
            ),
        ]
    )
    prompt = (
        "Evaluate the investment idea below against the investment philosophy and recommendation contract. "
        "Use current web/news search only to fill high-level current context; cite sources inside evidence items when used. "
        "Return JSON with keys: thesis_statement, action, recommendation_status, score, confidence, rationale, "
        "factor_scores, missing_information, data_quality, evidence, disconfirming_evidence, catalyst, invalidation, portfolio_fit. "
        "factor_scores must include macro_support, industry_attractiveness, business_quality, management_quality, "
        "valuation_asymmetry, and portfolio_fit. Analyzer raw qualitative scores, when present, will override "
        "business/industry/management quality factors on the native 0-100 scale. Analyzer valuation_signal, when "
        "present, will be clipped to +/-3 and mapped to 0-100 with 50 neutral. action must be one of buy, watch, "
        "avoid, do_nothing.\n\n"
        f"Context JSON:\n{json.dumps(context, default=str, sort_keys=True)}"
    )
    try:
        text, citations, _response = call_llm_text(
            prompt=prompt,
            model=MODEL_HIGH,
            max_tokens=3000,
            system=system,
            max_web_search_uses=4,
        )
        parsed = parse_json_text(text)
        result = _normalize_llm_result(context, parsed)
        if citations:
            evidence = result.setdefault("evidence", [])
            if isinstance(evidence, list):
                evidence.extend(
                    {"source": title, "url": url, "summary": "Live web source used by evaluator."}
                    for title, url in citations[:8]
                )
        return result
    except Exception as exc:
        return _deterministic_evaluation(context, reason=str(exc))


def _recommendation_record_from_result(idea: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    action = str(result.get("action") or "watch").lower()
    now = str(result.get("evaluated_at") or _now())
    ticker = str(idea.get("ticker") or result.get("ticker") or "").upper()
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
        "instrument": ticker,
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
    token = {
        "id": idea.get("id"),
        "ticker": idea.get("ticker"),
        "company_name": idea.get("company_name"),
        "user_notes": idea.get("user_notes"),
        "tags": idea.get("tags"),
        "metadata": idea.get("metadata"),
        "version": IDEA_EVALUATION_VERSION,
    }
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
    return {"buy": 3, "watch": 2, "do_nothing": 1, "avoid": 0}.get(str(action or "").lower(), 1)


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
    total = 5
    if callable(progress_callback):
        progress_callback("analyzer", 1, total)
    analyzer_result = (
        _compute_portfolio_plus_ideas_analyzer_result() if _idea_analyzer_direction(idea) != "inactive" else None
    )
    analyzer_contexts = _analyzer_contexts_from_result(analyzer_result) if analyzer_result else {}
    analyzer_context = _analyzer_context_for_idea(
        idea,
        analyzer_result=analyzer_result,
        analyzer_contexts=analyzer_contexts,
    )
    if callable(progress_callback):
        progress_callback("context", 2, total)
    context = _build_context_for_evaluation(idea, analyzer_context=analyzer_context)
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
    idea = _get_idea(req.idea_id)
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
    has_analyzer_ideas = any(_idea_analyzer_direction(idea) != "inactive" for idea in ideas)
    analyzer_result = _compute_portfolio_plus_ideas_analyzer_result() if has_analyzer_ideas else None
    analyzer_contexts = _analyzer_contexts_from_result(analyzer_result) if analyzer_result else {}

    evaluations: list[dict[str, Any]] = []
    for index, idea in enumerate(ideas, start=1):
        if callable(progress_callback):
            progress_callback("evaluating", index + 1, total)
        analyzer_context = _analyzer_context_for_idea(
            idea,
            analyzer_result=analyzer_result,
            analyzer_contexts=analyzer_contexts,
        )
        context = _build_context_for_evaluation(idea, analyzer_context=analyzer_context)
        result = _call_llm_evaluator(context)
        if result.get("evaluation_schema_version") != IDEA_EVALUATION_SCHEMA_VERSION:
            result = _merge_analyzer_context_into_result(context, result)
            result["recommendation_record"] = _recommendation_record_from_result(idea, result)
        result["job_id"] = job_id
        evaluations.append(_write_idea_evaluation(idea, result, job_id=job_id))

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
    overview, overview_error = _read_state_text("investment_overviews", str(idea["ticker"]).upper())
    thesis, thesis_error = _read_state_text("investment_theses", str(idea["ticker"]).upper())
    management_quality, management_quality_error = _read_management_quality_text(str(idea["ticker"]).upper())
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
    return {
        "idea": idea,
        "evaluations": _list_idea_evaluations(idea_id, limit=20),
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
    return enqueue_response(row, "/api/v1/ideas/evaluate-all/async/{job_id}")


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
    key = run_id if ontology_primary_writes_enabled() else _comparison_uid(run_id)
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
    updates = body.model_dump(exclude_unset=True)
    if "analyzer_direction" in updates:
        metadata = current.get("metadata") if isinstance(current.get("metadata"), dict) else {}
        current["metadata"] = {
            **cast(dict[str, Any], metadata),
            "analyzer_direction": _normalize_analyzer_direction(updates.pop("analyzer_direction")),
        }
    current.update(updates)
    current["updated_at"] = _now()
    current.pop("_meta", None)
    current.pop("id", None)
    current.pop("object_uid", None)
    _write_runtime_object("InvestmentIdea", _idea_uid(idea_id), current)
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
    force_refresh = bool((body or {}).get("force_refresh", False))
    req = IdeaEvaluationRequest(idea_id=idea_id, force_refresh=force_refresh)
    row, _disposition = enqueue_registered_job(
        "idea_evaluation",
        req.model_dump(),
        cache_key=None if force_refresh else _cache_key(req),
        reuse_completed=not force_refresh,
    )
    idea.update({"latest_job_id": str(row.get("job_id") or ""), "updated_at": _now()})
    if str(row.get("status") or "") in {"queued", "running"}:
        idea["status"] = "researching"
    idea.pop("_meta", None)
    idea.pop("id", None)
    idea.pop("object_uid", None)
    _write_runtime_object("InvestmentIdea", _idea_uid(idea_id), idea)
    return enqueue_response(row, "/api/v1/ideas/evaluate/async/{job_id}")


@router.get("/ideas/evaluate/async/{job_id}")
def get_idea_evaluation_job(job_id: str):
    try:
        return poll_registered_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job_id") from None


@router.post("/ideas/{idea_id}/evaluations/{evaluation_id}/accept")
def accept_idea_evaluation(idea_id: str, evaluation_id: str, body: IdeaAcceptRequest | None = None):
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
    if ontology_primary_writes_enabled():
        from ontology.command_service import OntologyCommandContext, OntologyCommandService

        context = OntologyCommandContext(
            actor=admin_actor(source="ideas"),
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
    else:
        from portfolio import core_db

        recommendation = (
            core_db.upsert_recommendation(record)
            if record.get("idempotency_key")
            else core_db.create_recommendation(record)
        )
        recommendation_id = recommendation["id"]

    action_proposal = None
    action_error = None
    action = str(evaluation.get("action") or "").lower()
    if action in {"buy", "watch"}:
        action_type = "enter" if action == "buy" else "research"
        description = (
            f"{'Evaluate initial entry' if action == 'buy' else 'Research remaining evidence'} for "
            f"{idea['ticker']} from idea evaluator recommendation approval {recommendation_id}."
        )
        missing = (
            evaluation.get("missing_information") if isinstance(evaluation.get("missing_information"), list) else []
        )
        if missing:
            description += " Missing evidence: " + "; ".join(
                str(row.get("field") or row) for row in missing[:4] if isinstance(row, dict) or row
            )
        try:
            action_proposal = stage_api_action(
                "create_action_item",
                {
                    "recommendation_id": recommendation_id,
                    "ticker": idea.get("ticker"),
                    "description": description,
                    "action_type": action_type,
                    "urgency": "normal" if action == "buy" else "low",
                },
                source_id=f"ideas.accept:{idea_id}:{evaluation_id}",
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
        "recommendation_id": recommendation_id if not ontology_primary_writes_enabled() else None,
        "recommendation_approval_id": recommendation_id,
        "action_approval_id": action_proposal["approval_id"] if action_proposal else None,
    }
    accepted_payload.pop("_meta", None)
    accepted_payload.pop("id", None)
    accepted_payload.pop("object_uid", None)
    accepted = _write_runtime_object("IdeaEvaluation", _evaluation_uid(evaluation_id), accepted_payload)
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
    idea.update(
        {"status": "rejected", "metadata": {"rejection_note": body.note if body else None, "rejected_at": _now()}}
    )
    idea.pop("_meta", None)
    idea.pop("id", None)
    idea.pop("object_uid", None)
    idea = _write_runtime_object("InvestmentIdea", _idea_uid(idea_id), idea)
    return {"status": "rejected", "idea": idea}
