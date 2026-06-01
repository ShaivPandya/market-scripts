"""OpportunityScout: proactive candidate builder, ranking, and queue helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from decision_quality.candidate_gates import apply_opportunity_candidate_gates
from decision_quality.opportunity_candidate import (
    OpportunityCandidate,
    parse_opportunity_candidate,
)
from ontology.decision_writeback import DecisionOntologyWriteback
from ontology.policy import system_actor
from ontology.schemas.identity import opportunity_candidate_id

ACTIVE_QUEUE_STATUSES = frozenset({"open", "generated", "watching", "research_requested"})
SEVERITY_SCORES = {"low": 1, "medium": 2, "high": 3}
HIT_TYPE_SCORES = {
    "triggered": 3,
    "needs_review": 2,
    "approaching": 2,
    "source_blocked": 2,
    "ok": 0,
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _stable_hash(value: Any, length: int = 16) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any, *, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _ticker(value: Any) -> str | None:
    text = _text(value).upper()
    return text or None


def _candidate_idempotency_key(*, source_kind: str, source_id: str, trigger: str, ticker: str | None) -> str:
    return _stable_hash(
        {
            "source_kind": source_kind,
            "source_id": source_id,
            "trigger": trigger,
            "ticker": ticker,
        },
        length=24,
    )


def _infer_opportunity_type(entity_type: str, hit_type: str) -> str:
    entity = entity_type.strip().lower()
    hit = hit_type.strip().lower()
    if entity == "kill_condition" or hit == "triggered":
        return "unsustainable_process"
    if entity == "catalyst":
        return "policy_inflection"
    if hit == "source_blocked":
        return "unclear"
    return "unclear"


def _default_missing_inputs(*, source_kind: str, ticker: str | None) -> list[str]:
    missing = ["decision_quality pressure-test", "sizing context"]
    if not ticker:
        missing.insert(0, "Ticker shortlist")
    if source_kind == "monitor_hit":
        missing.append("Monitor follow-up review")
    return missing


def build_candidate_from_monitor_hit(
    hit_payload: Mapping[str, Any],
    *,
    source_id: str | None = None,
) -> dict[str, Any]:
    """Normalize a monitor hit into an OpportunityCandidate persistence record."""

    ticker = _ticker(hit_payload.get("ticker"))
    if ticker == "UNKNOWN":
        ticker = None
    entity_type = _text(hit_payload.get("entity_type"), default="monitor_hit")
    hit_type = _text(hit_payload.get("hit_type"), default="needs_review")
    entity_label = _text(hit_payload.get("entity_label") or hit_payload.get("entity_id"))
    evidence = _text(hit_payload.get("evidence") or entity_label or "Monitor hit detected")
    severity = _text(hit_payload.get("severity"), default="medium").lower()
    trigger = f"{entity_type.replace('_', ' ')} monitor hit: {entity_label or hit_type}"
    why_now = evidence
    if severity == "high":
        why_now = f"High-severity monitor signal: {evidence}"
    source_kind = "monitor_hit"
    idempotency_key = _candidate_idempotency_key(
        source_kind=source_kind,
        source_id=_text(source_id or hit_payload.get("fingerprint") or hit_payload.get("entity_id")),
        trigger=trigger,
        ticker=ticker,
    )
    candidate, _ = parse_opportunity_candidate(
        {
            "ticker": ticker,
            "source": source_kind,
            "trigger": trigger,
            "opportunity_type": _infer_opportunity_type(entity_type, hit_type),
            "consensus": "Automated monitor surfaced a potential change; consensus not established.",
            "variant_view": entity_label or evidence,
            "why_now": why_now,
            "price_confirmation": "Not verified automatically; review price action before acting.",
            "crowding": "",
            "payoff_asymmetry": "",
            "missing_inputs": _default_missing_inputs(source_kind=source_kind, ticker=ticker),
            "source_refs": [
                {
                    "label": entity_label or hit_type,
                    "source_path": f"monitor_hit:{_text(hit_payload.get('fingerprint') or source_id)}",
                }
            ],
            "next_action": "research",
            "summary": evidence,
        }
    )
    assert candidate is not None
    gate = apply_opportunity_candidate_gates(candidate)
    return {
        "candidate_id": idempotency_key,
        "idempotency_key": idempotency_key,
        "source_kind": source_kind,
        "source_type": "workflow",
        "source_id": _text(source_id or hit_payload.get("fingerprint")),
        "status": "open",
        "decision_state": "generated",
        "monitor_hit_fingerprint": _text(hit_payload.get("fingerprint")),
        "severity": severity,
        "rank_signals": {
            "severity": severity,
            "hit_type": hit_type,
            "confidence": hit_payload.get("confidence"),
        },
        "opportunity_candidate": candidate.model_dump(mode="json"),
        "opportunity_candidate_gate": gate.model_dump(mode="json"),
        **candidate.model_dump(mode="json"),
    }


def build_candidate_from_change_event(
    change_row: Mapping[str, Any],
    *,
    source_id: str | None = None,
) -> dict[str, Any] | None:
    """Build a candidate from a what-changed row when it represents a monitorable shift."""

    object_type = _text(change_row.get("object_type") or change_row.get("category")).lower()
    if object_type not in {"catalyst", "kill_condition", "watch_trigger", "action_item"}:
        return None
    ticker = _ticker(change_row.get("ticker"))
    title = _text(change_row.get("title") or change_row.get("summary") or object_type)
    trigger = f"What changed: {title}"
    source_kind = "workflow"
    idempotency_key = _candidate_idempotency_key(
        source_kind=source_kind,
        source_id=_text(source_id or change_row.get("object_uid") or title),
        trigger=trigger,
        ticker=ticker,
    )
    candidate, _ = parse_opportunity_candidate(
        {
            "ticker": ticker,
            "source": source_kind,
            "trigger": trigger,
            "opportunity_type": _infer_opportunity_type(object_type, "needs_review"),
            "consensus": "Recent workspace change detected; market view not re-evaluated.",
            "variant_view": title,
            "why_now": _text(change_row.get("summary") or title),
            "price_confirmation": "Not verified from change feed alone.",
            "missing_inputs": _default_missing_inputs(source_kind=source_kind, ticker=ticker),
            "source_refs": [{"label": title, "source_path": f"what_changed:{object_type}"}],
            "next_action": "research",
            "summary": title,
        }
    )
    if candidate is None:
        return None
    gate = apply_opportunity_candidate_gates(candidate)
    return {
        "candidate_id": idempotency_key,
        "idempotency_key": idempotency_key,
        "source_kind": source_kind,
        "source_type": "workflow",
        "source_id": _text(source_id or change_row.get("object_uid")),
        "status": "open",
        "decision_state": "generated",
        "opportunity_candidate": candidate.model_dump(mode="json"),
        "opportunity_candidate_gate": gate.model_dump(mode="json"),
        **candidate.model_dump(mode="json"),
    }


def compute_candidate_rank_score(row: Mapping[str, Any]) -> float:
    """Deterministic, auditable rank score for queue ordering."""

    score = 0.0
    rank_signals = _as_dict(row.get("rank_signals"))
    severity = _text(rank_signals.get("severity") or row.get("severity"), default="medium").lower()
    hit_type = _text(rank_signals.get("hit_type"), default="needs_review").lower()
    score += SEVERITY_SCORES.get(severity, 2) * 10.0
    score += HIT_TYPE_SCORES.get(hit_type, 1) * 5.0

    gate = _as_dict(row.get("opportunity_candidate_gate"))
    gate_status = _text(gate.get("status"), default="pass")
    if gate_status == "blocked":
        score -= 20.0
    elif gate_status == "downgraded":
        score -= 5.0
    elif gate_status == "pass":
        score += 3.0

    missing_inputs = _as_list(row.get("missing_inputs"))
    score -= min(len(missing_inputs), 6) * 1.5

    confidence = rank_signals.get("confidence")
    if isinstance(confidence, (int, float)):
        score += float(confidence) * 4.0

    next_action = _text(row.get("next_action"), default="research").lower()
    if next_action == "graduate_to_decision_quality":
        score += 8.0
    elif next_action == "watch":
        score += 2.0

    updated_at = _text(row.get("updated_at") or row.get("created_at"))
    if updated_at:
        score += 0.001

    return round(score, 4)


def rank_opportunity_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        item = dict(row)
        item["rank_score"] = compute_candidate_rank_score(item)
        enriched.append(item)
    enriched.sort(
        key=lambda item: (
            float(item.get("rank_score") or 0.0),
            _text(item.get("updated_at") or item.get("created_at")),
            _text(item.get("candidate_id") or item.get("id")),
        ),
        reverse=True,
    )
    return enriched


def normalize_candidate_queue_item(row: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = _text(row.get("candidate_id") or row.get("id"))
    gate = _as_dict(row.get("opportunity_candidate_gate"))
    return {
        "id": candidate_id or _text(row.get("object_uid")),
        "object_uid": _text(row.get("object_uid") or opportunity_candidate_id(candidate_id)),
        "candidate_id": candidate_id,
        "ticker": _ticker(row.get("ticker")),
        "source_kind": _text(row.get("source_kind") or row.get("source"), default="other"),
        "trigger": _text(row.get("trigger")),
        "opportunity_type": _text(row.get("opportunity_type"), default="unclear"),
        "consensus": _text(row.get("consensus")),
        "variant_view": _text(row.get("variant_view")),
        "why_now": _text(row.get("why_now")),
        "price_confirmation": _text(row.get("price_confirmation")),
        "crowding": _text(row.get("crowding")),
        "payoff_asymmetry": _text(row.get("payoff_asymmetry")),
        "missing_inputs": [str(item) for item in _as_list(row.get("missing_inputs")) if str(item).strip()],
        "next_action": _text(row.get("next_action"), default="research"),
        "summary": _text(row.get("summary")),
        "status": _text(row.get("status") or row.get("decision_state"), default="generated"),
        "decision_state": _text(row.get("decision_state"), default="generated"),
        "gate_status": _text(gate.get("status"), default="pass"),
        "gate_final_action": _text(gate.get("final_action")),
        "should_graduate": bool(gate.get("should_graduate")),
        "rank_score": compute_candidate_rank_score(row),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "source_refs": _as_list(row.get("source_refs")),
    }


def persist_proactive_opportunity_candidate(
    record: Mapping[str, Any],
    *,
    actor_id: str = "opportunity_scout",
    source_id: str | None = None,
) -> list[dict[str, Any]]:
    """Persist a proactive candidate via ontology writeback."""

    now = _now()
    payload = dict(record)
    payload.setdefault("created_at", now)
    payload.setdefault("updated_at", now)
    payload.setdefault("status", "open")
    payload.setdefault("decision_state", "generated")
    if source_id:
        payload.setdefault("source_id", source_id)
    return DecisionOntologyWriteback().record_opportunity_candidate(
        record=payload,
        actor=system_actor(actor_id),
        provenance_id=f"pv:opportunity_scout:{payload.get('idempotency_key') or payload.get('candidate_id')}",
    )


def maybe_create_candidate_from_monitor_hit(
    hit_payload: Mapping[str, Any],
    *,
    source_id: str,
    existing_idempotency_keys: set[str] | None = None,
) -> dict[str, Any] | None:
    """Create a candidate from a monitor hit when it is actionable and not duplicate."""

    hit_type = _text(hit_payload.get("hit_type"), default="ok").lower()
    if hit_type in {"ok", "skipped"}:
        return None
    record = build_candidate_from_monitor_hit(hit_payload, source_id=source_id)
    idempotency_key = _text(record.get("idempotency_key"))
    if existing_idempotency_keys is not None and idempotency_key in existing_idempotency_keys:
        return None
    persist_proactive_opportunity_candidate(record, source_id=source_id)
    return record
