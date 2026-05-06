"""Domain-oriented ontology writes for operational actions.

This module is the migration bridge between legacy domain actions and the
authoritative bitemporal ontology write boundary. In shadow mode, callers keep
their existing legacy writes and mirror the resulting operational objects here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from ontology.object_service import OntologyObjectService, object_uid_for
from ontology.read_model import TemporalReadModelRepository
from ontology.schemas.identity import action_run_id, object_version_ref_id, thesis_id

logger = logging.getLogger(__name__)

OPERATIONAL_ONTOLOGY_RUN_ID = "operational"
_APPROVED_DOMAIN_WRITE_SCOPE: ContextVar[dict[str, Any] | None] = ContextVar(
    "approved_domain_write_scope",
    default=None,
)


def ontology_shadow_writes_enabled() -> bool:
    return True


def ontology_primary_writes_enabled() -> bool:
    return True


def ontology_read_model_enabled() -> bool:
    return _env_flag("ONTOLOGY_READ_MODEL")


def legacy_write_guard_enabled() -> bool:
    return True


def approved_domain_write_scope() -> dict[str, Any] | None:
    """Return metadata for the current approved domain write, if any."""

    return _APPROVED_DOMAIN_WRITE_SCOPE.get()


@contextmanager
def domain_write_scope(
    *,
    action_id: str,
    actor_type: str,
    approval_id: int | None = None,
    action_run_id: int | None = None,
    source_type: str | None = None,
    source_id: str | None = None,
):
    """Mark a call stack as executing an approved financial mutation."""

    token = _APPROVED_DOMAIN_WRITE_SCOPE.set(
        {
            "action_id": action_id,
            "actor_type": actor_type,
            "approval_id": approval_id,
            "action_run_id": action_run_id,
            "source_type": source_type,
            "source_id": source_id,
        }
    )
    try:
        yield
    finally:
        _APPROVED_DOMAIN_WRITE_SCOPE.reset(token)


def assert_legacy_domain_write_allowed(surface: str) -> None:
    if approved_domain_write_scope() is not None and _env_flag("TALISMAN_ALLOW_LEGACY_PROJECTION_WRITE"):
        return
    raise RuntimeError(
        f"Legacy domain write blocked by ontology-primary runtime: {surface}. "
        "Use OntologyObjectService/OntologyCommandService, or run the isolated legacy backfill job."
    )


def _env_flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class OntologyMutation:
    object_type: str
    business_key: str
    properties: dict[str, Any]
    valid_from: str


class DomainOntologyWriteService:
    """Typed operational write facade above :class:`OntologyObjectService`."""

    def __init__(self, object_service: OntologyObjectService | None = None):
        self.object_service = object_service or OntologyObjectService()

    def write_object(
        self,
        mutation: OntologyMutation,
        *,
        actor: Any = None,
        provenance_event_id: str | None = None,
        action_run_id_value: int | None = None,
        approval_id: int | None = None,
        input_hash: str | None = None,
        temporal_confidence: str = "native",
        source_record_id: str | None = None,
    ) -> dict[str, Any]:
        properties = {**mutation.properties, "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID}
        return self.object_service.write_object(
            mutation.object_type,
            mutation.business_key,
            properties,
            mutation.valid_from,
            actor=actor,
            provenance=provenance_event_id,
            action_run_id=action_run_id_value,
            approval_id=approval_id,
            input_hash=input_hash,
            temporal_confidence=temporal_confidence,
            source_record_id=source_record_id,
        )

    def write_action_output(
        self,
        *,
        action_id: str,
        input_payload: Mapping[str, Any],
        output: Mapping[str, Any],
        context: Any,
        input_hash: str | None,
    ) -> list[dict[str, Any]]:
        now = _now()
        actor = _actor(context)
        action_run_id_value = _context_int(context, "action_run_id")
        approval_id = _context_int(context, "approval_id")
        provenance_event_id = _context_str(context, "provenance_event_id")
        rows: list[dict[str, Any]] = []

        if action_run_id_value is not None:
            action_run_row = self.write_object(
                OntologyMutation(
                    "ActionRun",
                    str(action_run_id_value),
                    {
                        "legacy_id": action_run_id_value,
                        "action_id": action_id,
                        "action_schema_version": 1,
                        "actor_type": actor.get("actor_type") or "unknown",
                        "actor_id": actor.get("actor_id"),
                        "source_type": _context_str(context, "source_type"),
                        "source_id": _context_str(context, "source_id"),
                        "approval_id": approval_id,
                        "parent_action_run_id": _context_int(context, "parent_action_run_id"),
                        "input_hash": input_hash,
                        "output_hash": _stable_hash(output),
                        "status": "succeeded",
                        "execution_state": "succeeded",
                        "completed_at": now,
                        "provenance_event_id": provenance_event_id,
                    },
                    now,
                ),
                actor=actor,
                provenance_event_id=provenance_event_id,
                action_run_id_value=action_run_id_value,
                approval_id=approval_id,
                input_hash=input_hash,
            )
            rows.append(action_run_row)
            _log_shadow_parity(
                action_id=action_id,
                row=action_run_row,
                action_run_id_value=action_run_id_value,
                approval_id=approval_id,
                provenance_event_id=provenance_event_id,
            )

        for mutation in action_mutations(action_id, input_payload, output, now=now):
            row = self.write_object(
                mutation,
                actor=actor,
                provenance_event_id=provenance_event_id,
                action_run_id_value=action_run_id_value,
                approval_id=approval_id,
                input_hash=input_hash,
            )
            rows.append(row)
            _log_shadow_parity(
                action_id=action_id,
                row=row,
                action_run_id_value=action_run_id_value,
                approval_id=approval_id,
                provenance_event_id=provenance_event_id,
            )
            if action_run_id_value is not None:
                self._link_action_run_to_version(
                    action_run_id_value,
                    row,
                    valid_from=mutation.valid_from,
                    actor=actor,
                    provenance_event_id=provenance_event_id,
                    approval_id=approval_id,
                    input_hash=input_hash,
                )
        _refresh_temporal_read_models_if_enabled()
        return rows

    def write_pending_approval(
        self,
        approval: Mapping[str, Any],
        *,
        context: Any,
        input_hash: str | None,
    ) -> dict[str, Any]:
        now = str(approval.get("created_at") or _now())
        actor = _actor(context)
        row = self.write_object(
            OntologyMutation(
                "Approval",
                str(approval.get("id") or approval.get("legacy_id") or _stable_hash(approval)),
                _approval_properties(approval),
                now,
            ),
            actor=actor,
            provenance_event_id=str(
                approval.get("provenance_event_id") or _context_str(context, "provenance_event_id") or ""
            ),
            action_run_id_value=_context_int(context, "action_run_id"),
            approval_id=_optional_int(approval.get("id")),
            input_hash=input_hash,
        )
        _log_shadow_parity(
            action_id="pending_approval",
            row=row,
            action_run_id_value=_context_int(context, "action_run_id"),
            approval_id=_optional_int(approval.get("id")),
            provenance_event_id=str(
                approval.get("provenance_event_id") or _context_str(context, "provenance_event_id") or ""
            ),
        )
        _refresh_temporal_read_models_if_enabled()
        return row

    def _link_action_run_to_version(
        self,
        action_run_id_value: int,
        row: Mapping[str, Any],
        *,
        valid_from: str,
        actor: Mapping[str, Any],
        provenance_event_id: str | None,
        approval_id: int | None,
        input_hash: str | None,
    ) -> None:
        temporal = _temporal(row)
        object_uid = str(row.get("object_uid") or temporal.get("object_uid") or "")
        version_id = str(temporal.get("version_id") or row.get("version_id") or "")
        if not object_uid or not version_id:
            return
        ref_id = f"{object_uid}:{version_id}"
        self.write_object(
            OntologyMutation(
                "ObjectVersionRef",
                ref_id,
                {
                    "ref_id": ref_id,
                    "object_uid": object_uid,
                    "object_type": row.get("object_type"),
                    "version_id": version_id,
                    "valid_from": temporal.get("valid_from"),
                    "tx_from": temporal.get("tx_from"),
                    "temporal_confidence": temporal.get("temporal_confidence"),
                },
                valid_from,
            ),
            actor=actor,
            provenance_event_id=provenance_event_id,
            action_run_id_value=action_run_id_value,
            approval_id=approval_id,
            input_hash=input_hash,
        )
        self.object_service.write_relation(
            action_run_id(action_run_id_value),
            object_version_ref_id(ref_id),
            "action_run_mutates_object_version",
            {
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                "object_uid": object_uid,
                "object_type": row.get("object_type"),
                "version_id": version_id,
            },
            valid_from,
            actor=actor,
            provenance=provenance_event_id,
            action_run_id=action_run_id_value,
            approval_id=approval_id,
            input_hash=input_hash,
        )


def record_action_ontology_versions(
    *,
    action_id: str,
    input_payload: Mapping[str, Any],
    output: Mapping[str, Any],
    context: Any,
    input_hash: str | None,
) -> list[dict[str, Any]]:
    if not ontology_shadow_writes_enabled():
        return []
    service = DomainOntologyWriteService()
    return service.write_action_output(
        action_id=action_id,
        input_payload=input_payload,
        output=output,
        context=context,
        input_hash=input_hash,
    )


def record_pending_approval_ontology_version(
    approval: Mapping[str, Any],
    *,
    context: Any,
    input_hash: str | None,
) -> dict[str, Any] | None:
    if not ontology_shadow_writes_enabled():
        return None
    return DomainOntologyWriteService().write_pending_approval(approval, context=context, input_hash=input_hash)


def _refresh_temporal_read_models_if_enabled() -> None:
    if not ontology_read_model_enabled():
        return
    try:
        TemporalReadModelRepository().refresh()
    except Exception:
        if ontology_primary_writes_enabled():
            raise
        logger.exception("ontology read model refresh failed during shadow write")


def _log_shadow_parity(
    *,
    action_id: str,
    row: Mapping[str, Any],
    action_run_id_value: int | None,
    approval_id: int | None,
    provenance_event_id: str | None,
) -> None:
    temporal = _temporal(row)
    logger.info(
        "ontology shadow parity action_id=%s object_type=%s business_key=%s object_uid=%s "
        "version_id=%s action_run_id=%s approval_id=%s provenance_event_id=%s",
        action_id,
        row.get("object_type"),
        row.get("business_key"),
        row.get("object_uid") or temporal.get("object_uid"),
        temporal.get("version_id") or row.get("version_id"),
        action_run_id_value,
        approval_id,
        provenance_event_id,
    )


def action_mutations(
    action_id: str,
    input_payload: Mapping[str, Any],
    output: Mapping[str, Any],
    *,
    now: str | None = None,
) -> list[OntologyMutation]:
    now = now or _now()
    if action_id == "update_portfolio_positions":
        return [
            OntologyMutation("Position", str(row.get("ticker") or ""), _position_properties(row, role="position"), now)
            for row in _dicts(input_payload.get("positions"))
            if row.get("ticker")
        ]
    if action_id == "update_hedge_positions":
        return [
            OntologyMutation(
                "HedgePosition",
                str(row.get("ticker") or ""),
                _hedge_properties(row),
                now,
            )
            for row in _dicts(input_payload.get("positions"))
            if row.get("ticker")
        ]
    if action_id == "change_thesis_status":
        if output.get("changed") is False:
            return []
        ticker = str(output.get("ticker") or input_payload.get("ticker") or "").upper()
        return [
            OntologyMutation(
                "Thesis",
                ticker,
                {
                    "ticker": ticker,
                    "status": str(output.get("new_status") or input_payload.get("status") or "active"),
                    "created_at": str(output.get("created_at") or output.get("updated_at") or now),
                    "updated_at": str(output.get("updated_at") or now),
                },
                str(output.get("updated_at") or now),
            )
        ]
    if action_id == "save_thesis_content":
        ticker = str(output.get("ticker") or input_payload.get("ticker") or "").upper()
        content = str(output.get("content") or input_payload.get("content") or "")
        return [
            OntologyMutation(
                "Thesis",
                ticker,
                {"ticker": ticker, "status": "active", "created_at": now, "updated_at": now},
                now,
            ),
            OntologyMutation(
                "DocumentArtifact",
                f"thesis:{ticker}",
                {
                    "document_type": "thesis",
                    "document_id": ticker,
                    "title": f"Thesis {ticker}",
                    "ticker": ticker,
                    "content_hash": _hash_text(content),
                    "status": "active",
                },
                now,
            ),
        ]
    if action_id == "save_evaluation":
        ticker = str(output.get("ticker") or input_payload.get("ticker") or "").upper()
        evaluated_at = str(output.get("evaluated_at") or input_payload.get("evaluated_at") or now)
        props = {
            "ticker": ticker,
            "evaluated_at": evaluated_at,
            "thesis_status": str(input_payload.get("thesis_status") or ""),
            "technical_read": str(input_payload.get("technical_read") or ""),
            "fundamental_read": str(input_payload.get("fundamental_read") or ""),
            "action": str(input_payload.get("action") or ""),
            "confidence": str(input_payload.get("confidence") or ""),
            "key_developments": _strings(input_payload.get("key_developments")),
            "earnings_note": input_payload.get("earnings_note"),
            "risk_flag": input_payload.get("risk_flag"),
        }
        return [OntologyMutation("Evaluation", f"{ticker}-{evaluated_at}", props, evaluated_at)]
    if action_id in {"create_catalyst", "update_catalyst_status"}:
        row = {**dict(input_payload), **dict(output)}
        return [
            OntologyMutation(
                "Catalyst",
                str(row.get("description") or row.get("id") or ""),
                _catalyst_properties(row),
                _row_time(row, now),
            )
        ]
    if action_id in {"create_kill_condition", "update_kill_condition_status"}:
        row = {**dict(input_payload), **dict(output)}
        return [
            OntologyMutation(
                "KillCondition",
                str(row.get("id") or row.get("kill_condition_id") or row.get("condition") or ""),
                _kill_condition_properties(row),
                _row_time(row, now),
            )
        ]
    if action_id in {"create_thesis_claim", "update_thesis_claim"}:
        row = {**dict(input_payload), **dict(output)}
        return [
            OntologyMutation(
                "ThesisClaim",
                str(row.get("id") or row.get("claim_id") or row.get("claim") or ""),
                _thesis_claim_properties(row),
                _row_time(row, now),
            )
        ]
    if action_id in {"create_action_item", "complete_action_item", "dismiss_action_item"}:
        row = {**dict(input_payload), **dict(output)}
        return [
            OntologyMutation(
                "ActionItem",
                str(row.get("id") or row.get("item_id") or row.get("description") or ""),
                _action_item_properties(row),
                _row_time(row, now),
            )
        ]
    if action_id == "create_recommendation":
        record_value = input_payload.get("record")
        record = cast(Mapping[str, Any], record_value) if isinstance(record_value, Mapping) else input_payload
        row = {**dict(record), **dict(output)}
        mutations = [
            OntologyMutation(
                "Recommendation",
                str(row.get("id") or row.get("legacy_id") or row.get("idempotency_key") or row.get("instrument") or ""),
                _recommendation_properties(row),
                _row_time(row, now),
            )
        ]
        gate = row.get("policy_gate_result")
        if isinstance(gate, Mapping):
            gate_id = str(
                row.get("policy_gate_result_id") or gate.get("id") or gate.get("evaluated_at") or _stable_hash(gate)
            )
            mutations.append(
                OntologyMutation(
                    "PolicyGateResult",
                    gate_id,
                    _policy_gate_result_properties(gate, gate_result_id=gate_id),
                    str(gate.get("evaluated_at") or now),
                )
            )
        return mutations
    if action_id in {"create_watch_trigger", "fire_watch_trigger", "cancel_watch_trigger"}:
        row = {**dict(input_payload), **dict(output)}
        return [
            OntologyMutation(
                "WatchTrigger",
                str(row.get("id") or row.get("trigger_id") or row.get("condition") or ""),
                _watch_trigger_properties(row),
                _row_time(row, now),
            )
        ]
    if action_id == "create_research_note":
        row = {**dict(input_payload), **dict(output)}
        return [
            OntologyMutation(
                "ResearchNote",
                str(row.get("id") or row.get("title") or ""),
                _research_note_properties(row),
                _row_time(row, now),
            )
        ]
    if action_id == "resolve_approval":
        return [
            OntologyMutation(
                "Approval",
                str(output.get("id") or output.get("approval_id") or input_payload.get("approval_id") or ""),
                _approval_properties(output or input_payload),
                str(output.get("resolved_at") or output.get("created_at") or now),
            )
        ]
    if action_id == "delete_portfolio_news_digest":
        digest_id = str(output.get("digest_id") or input_payload.get("digest_id") or "")
        return [
            OntologyMutation(
                "DocumentArtifact",
                f"news_digest:{digest_id}",
                {
                    "document_type": "news_digest",
                    "document_id": digest_id,
                    "status": "deleted",
                    "updated_at": now,
                },
                now,
            )
        ]
    return []


def _position_properties(row: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    ticker = str(row.get("ticker") or "").upper()
    return {
        "ticker": ticker,
        "asset": str(row.get("asset") or "equity").lower(),
        "direction": str(row.get("direction") or "long").lower(),
        "contrarian": bool(row.get("contrarian")),
        "conviction": _optional_int(row.get("conviction")) or 3,
        "cost_basis": _optional_float(row.get("cost_basis")),
        "shares": _optional_float(row.get("shares")),
        "quantity": _optional_float(row.get("quantity") if row.get("quantity") is not None else row.get("shares")),
        "instrument_type": str(row.get("instrument_type") or "security").lower(),
        "price_symbol": str(row.get("price_symbol") or ticker).upper(),
        "contract_multiplier": _optional_float(row.get("contract_multiplier")) or 1.0,
        "role": role,
    }


def _hedge_properties(row: Mapping[str, Any]) -> dict[str, Any]:
    ticker = str(row.get("ticker") or "").upper()
    return {
        "ticker": ticker,
        "direction": str(row.get("direction") or "long").lower(),
        "asset": str(row.get("asset") or "equity").lower(),
        "cost_basis": _optional_float(row.get("cost_basis")),
        "shares": _optional_float(row.get("shares")),
        "quantity": _optional_float(row.get("quantity") if row.get("quantity") is not None else row.get("shares")),
        "instrument_type": str(row.get("instrument_type") or "security").lower(),
        "price_symbol": str(row.get("price_symbol") or ticker).upper(),
        "contract_multiplier": _optional_float(row.get("contract_multiplier")) or 1.0,
    }


def _catalyst_properties(row: Mapping[str, Any]) -> dict[str, Any]:
    description = str(row.get("description") or "")
    return {
        "ticker": str(row.get("ticker") or "").upper(),
        "legacy_id": _optional_int(row.get("id") or row.get("catalyst_id")),
        "name": str(row.get("name") or description[:120] or "Catalyst"),
        "description": description or str(row.get("name") or "Catalyst"),
        "source": str(row.get("created_by") or row.get("source_type") or "domain_action"),
        "category": row.get("category"),
        "target_date": row.get("target_date"),
        "status": row.get("status") or "pending",
    }


def _kill_condition_properties(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ticker": str(row.get("ticker") or "").upper(),
        "legacy_id": _optional_int(row.get("id") or row.get("kill_condition_id")),
        "condition": str(row.get("condition") or ""),
        "metric": row.get("metric"),
        "threshold": row.get("threshold"),
        "status": row.get("status") or "active",
        "triggered_at": row.get("triggered_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "created_by": row.get("created_by"),
    }


def _thesis_claim_properties(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ticker": str(row.get("ticker") or "").upper(),
        "legacy_id": _optional_int(row.get("id") or row.get("claim_id")),
        "claim": str(row.get("claim") or ""),
        "expected_evidence": row.get("expected_evidence"),
        "disconfirming_evidence": row.get("disconfirming_evidence"),
        "source_requirements": _as_list(row.get("source_requirements")),
        "cadence": row.get("cadence"),
        "confidence": _optional_float(row.get("confidence")),
        "status": row.get("status") or "active",
        "linked_catalyst_ids": _ints(row.get("linked_catalyst_ids")),
        "linked_kill_condition_ids": _ints(row.get("linked_kill_condition_ids")),
        "source_type": row.get("source_type"),
        "source_id": row.get("source_id"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _action_item_properties(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "legacy_id": _optional_int(row.get("id") or row.get("item_id")),
        "ticker": _optional_ticker(row.get("ticker")),
        "description": str(row.get("description") or ""),
        "action_type": str(row.get("action_type") or "review"),
        "urgency": str(row.get("urgency") or "normal"),
        "status": str(row.get("status") or "open"),
        "source_type": row.get("source_type"),
        "source_id": row.get("source_id"),
        "created_at": row.get("created_at"),
        "completed_at": row.get("completed_at"),
        "resolution_note": row.get("resolution_note"),
    }


def _recommendation_properties(row: Mapping[str, Any]) -> dict[str, Any]:
    action = str(row.get("action") or "watch")
    is_actionable = action in {"buy", "sell", "reduce", "exit", "rebalance", "hedge"}
    return {
        "recommendation_id": row.get("id") or row.get("legacy_id") or row.get("idempotency_key"),
        "legacy_id": _optional_int(row.get("id")),
        "idempotency_key": row.get("idempotency_key"),
        "source_kind": "report",
        "report_type": row.get("report_type"),
        "as_of": row.get("as_of"),
        "action": action,
        "ticker": _optional_ticker(row.get("ticker")),
        "instrument": row.get("instrument"),
        "decision_state": "proposed" if row.get("approval_id") else "generated",
        "status": row.get("recommendation_status") or row.get("status"),
        "approval_id": _optional_int(row.get("approval_id")),
        "approval_required": bool(row.get("approval_required") or is_actionable),
        "approval_status": row.get("approval_status"),
        "outcome_status": row.get("outcome_status"),
        "supersedes_recommendation_id": row.get("supersedes_recommendation_id"),
        "account_id": row.get("account_id"),
        "portfolio_id": row.get("portfolio_id"),
        "policy_id": row.get("policy_id"),
        "policy_gate_result_id": _optional_int(row.get("policy_gate_result_id")),
        "policy_gate_decision": row.get("policy_gate_decision"),
        "policy_gate_review_required": bool(row.get("policy_gate_review_required")),
        "confidence": _optional_float(row.get("confidence")),
        "horizon": row.get("horizon"),
        "rationale_summary": str(row.get("rationale") or "")[:500] if row.get("rationale") else None,
        "rationale_hash": _hash_text(str(row.get("rationale") or "")) if row.get("rationale") else None,
        "source_quality": row.get("source_quality"),
        "payload": {key: _jsonable(value) for key, value in row.items() if key not in {"policy_gate_result"}},
    }


def _policy_gate_result_properties(row: Mapping[str, Any], *, gate_result_id: str) -> dict[str, Any]:
    return {
        "gate_result_id": gate_result_id,
        "decision": str(row.get("decision") or "error"),
        "review_required": bool(row.get("review_required")),
        "failure_reasons": _dicts(row.get("failure_reasons")),
        "warnings": _dicts(row.get("warnings")),
        "account_id": row.get("account_id"),
        "portfolio_id": row.get("portfolio_id"),
        "policy_id": row.get("policy_id"),
        "evaluated_at": row.get("evaluated_at"),
    }


def _watch_trigger_properties(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "legacy_id": _optional_int(row.get("id") or row.get("trigger_id")),
        "ticker": _optional_ticker(row.get("ticker")),
        "condition": str(row.get("condition") or ""),
        "trigger_type": str(row.get("trigger_type") or "custom"),
        "status": str(row.get("status") or "active"),
        "source_type": row.get("source_type"),
        "source_id": row.get("source_id"),
        "created_at": row.get("created_at"),
        "fired_at": row.get("fired_at"),
        "expires_at": row.get("expires_at"),
        "definition": _as_optional_dict(row.get("definition") or row.get("definition_json")),
        "last_checked_at": row.get("last_checked_at"),
        "last_result": _as_optional_dict(row.get("last_result") or row.get("last_result_json") or row.get("result")),
        "last_evidence": row.get("last_evidence") or row.get("evidence"),
    }


def _research_note_properties(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "legacy_id": _optional_int(row.get("id")),
        "ticker": _optional_ticker(row.get("ticker")),
        "title": str(row.get("title") or ""),
        "content": str(row.get("content") or ""),
        "note_type": str(row.get("note_type") or "general"),
        "source_type": row.get("source_type"),
        "source_id": row.get("source_id"),
        "created_at": row.get("created_at"),
    }


def _approval_properties(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "legacy_id": _optional_int(row.get("id") or row.get("approval_id")),
        "entity_type": str(row.get("entity_type") or "approval"),
        "entity_id": _optional_int(row.get("entity_id")),
        "ticker": _optional_ticker(row.get("ticker")),
        "target_object_uid": row.get("target_object_uid"),
        "target_object_type": row.get("target_object_type"),
        "action_id": row.get("action_id"),
        "action_schema_name": row.get("action_schema_name"),
        "action_schema_version": _optional_int(row.get("action_schema_version")),
        "action_input_hash": row.get("action_input_hash"),
        "proposed_change": _as_optional_dict(row.get("proposed_change")) or {},
        "reason": row.get("reason"),
        "source_type": row.get("source_type"),
        "source_id": row.get("source_id"),
        "status": str(row.get("status") or "pending"),
        "resolution_state": str(row.get("status") or "pending"),
        "application_state": str(row.get("application_status") or "pending"),
        "application_status": row.get("application_status"),
        "risk_class": row.get("risk_class"),
        "base_state_hash": row.get("base_state_hash"),
        "requested_by_actor_id": row.get("requested_by_actor_id"),
        "resolved_by_actor_id": row.get("resolved_by_actor_id"),
        "created_at": row.get("created_at"),
        "resolved_at": row.get("resolved_at"),
        "resolved_note": row.get("resolved_note"),
    }


def target_uid_for_ticker(ticker: str | None) -> str | None:
    ticker_s = _optional_ticker(ticker)
    return thesis_id(ticker_s) if ticker_s else None


def object_uid_for_mutation(mutation: OntologyMutation) -> str:
    return object_uid_for(mutation.object_type, mutation.business_key, mutation.properties)


def _actor(context: Any) -> dict[str, Any]:
    return {
        "actor_type": _context_str(context, "actor_type"),
        "actor_id": _context_str(context, "actor_id"),
    }


def _context_str(context: Any, name: str) -> str | None:
    value = getattr(context, name, None)
    if value is None and isinstance(context, Mapping):
        value = context.get(name)
    return str(value) if value is not None and str(value) else None


def _context_int(context: Any, name: str) -> int | None:
    value = getattr(context, name, None)
    if value is None and isinstance(context, Mapping):
        value = context.get(name)
    return _optional_int(value)


def _dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _strings(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value) if str(item).strip()]


def _ints(value: Any) -> list[int]:
    out: list[int] = []
    for item in _as_list(value):
        parsed = _optional_int(item)
        if parsed is not None:
            out.append(parsed)
    return out


def _as_optional_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return dict(parsed) if isinstance(parsed, Mapping) else None
    return None


def _optional_ticker(value: Any) -> str | None:
    ticker = str(value or "").strip().upper()
    return ticker or None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None and str(value).strip() else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None and str(value).strip() else None
    except (TypeError, ValueError):
        return None


def _row_time(row: Mapping[str, Any], fallback: str) -> str:
    for key in ("updated_at", "completed_at", "fired_at", "created_at", "evaluated_at"):
        value = row.get(key)
        if value:
            return str(value)
    return fallback


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _temporal(row: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = row.get("_meta")
    if isinstance(meta, Mapping):
        temporal = meta.get("temporal")
        if isinstance(temporal, Mapping):
            return temporal
    return {}
