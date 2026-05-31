"""Shared evidence ledger write and read helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from ontology.domain_write_service import OPERATIONAL_ONTOLOGY_RUN_ID
from ontology.object_service import OntologyObjectService, source_record_object_uid_for
from ontology.runtime_read_service import OntologyRuntimeReadService, object_props
from ontology.schemas.identity import (
    citation_id,
    evidence_id,
    object_version_ref_id,
    source_record_object_id,
)


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _hash_text(value: str, *, length: int = 16) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _hash_value(value: Any, *, length: int = 16) -> str:
    raw = json.dumps(_jsonable(value), sort_keys=True, default=str, separators=(",", ":"))
    return _hash_text(raw, length=length)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                return {}
            return _as_dict(decoded) if isinstance(decoded, list) else _as_dict(decoded)
        if stripped.startswith("{"):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                return {}
            return _as_dict(decoded)
    return {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                return [stripped]
            return decoded if isinstance(decoded, list) else [decoded]
        return [stripped]
    return [value]


def _truncate(value: Any, max_chars: int) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _citation_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    citation = value.get("citation")
    if isinstance(citation, Mapping):
        return {str(key): item for key, item in citation.items()}
    if any(value.get(key) for key in ("url", "source_path", "document_artifact_id", "source_record_id")):
        return {
            "title": value.get("title") or value.get("source"),
            "url": value.get("url"),
            "source_path": value.get("source_path"),
            "document_artifact_id": value.get("document_artifact_id"),
            "source_record_id": value.get("source_record_id"),
            "quote": value.get("quote") or value.get("summary") or value.get("text"),
        }
    return {}


def _version_ref_payload(row: Mapping[str, Any]) -> dict[str, Any] | None:
    meta = row.get("_meta")
    temporal = meta.get("temporal") if isinstance(meta, Mapping) else {}
    if not isinstance(temporal, Mapping):
        temporal = {}
    object_uid = str(row.get("object_uid") or temporal.get("object_uid") or "")
    version_id = str(row.get("version_id") or temporal.get("version_id") or "")
    if not object_uid or not version_id:
        return None
    return {
        "ref_id": f"{object_uid}:{version_id}",
        "object_uid": object_uid,
        "object_type": row.get("object_type"),
        "version_id": version_id,
        "valid_from": temporal.get("valid_from") or row.get("valid_from"),
        "tx_from": temporal.get("tx_from") or row.get("tx_from"),
        "temporal_confidence": temporal.get("temporal_confidence") or row.get("temporal_confidence"),
        "source_record_id": row.get("source_record_id"),
    }


def parse_evidence_items(value: Any) -> list[Any]:
    """Normalize expected/disconfirming evidence payloads into writeable items."""
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item is not None and item != ""]
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                return [stripped]
            if isinstance(decoded, list):
                return [item for item in decoded if item is not None and item != ""]
            return [decoded]
        if stripped.startswith("{"):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                return [stripped]
            return [decoded]
        return [stripped]
    return [value]


def write_parent_evidence_graph(
    object_service: OntologyObjectService,
    *,
    parent_uid: str,
    parent_key: str,
    evidence_items: Sequence[Any],
    relation_type: str,
    role: str,
    valid_from: str,
    actor: Mapping[str, Any],
    provenance_id: str,
    approval_id: int | None = None,
    source_record_id: str | None = None,
    observed_at: str | None = None,
    input_hash: str | None = None,
) -> list[dict[str, Any]]:
    """Write Evidence (+ optional Citation) objects and link them to a parent."""
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(evidence_items):
        if item is None or item == "":
            continue
        evidence_payload = _as_dict(item) if isinstance(item, (Mapping, str)) else {}
        summary = (
            evidence_payload.get("summary")
            or evidence_payload.get("text")
            or evidence_payload.get("evidence")
            or evidence_payload.get("description")
            or (item if isinstance(item, str) else None)
        )
        if not str(summary or "").strip():
            continue
        evidence_key = str(evidence_payload.get("evidence_id") or f"{parent_key}:{role}:{index}:{_hash_value(item)}")
        item_source_record_id = evidence_payload.get("source_record_id") or source_record_id
        evidence_row = object_service.write_object(
            "Evidence",
            evidence_key,
            {
                "evidence_id": evidence_key,
                "evidence_type": str(evidence_payload.get("evidence_type") or role),
                "title": evidence_payload.get("title") or evidence_payload.get("source"),
                "summary": _truncate(summary, 2000),
                "source_record_id": item_source_record_id,
                "document_artifact_id": evidence_payload.get("document_artifact_id"),
                "confidence": _optional_float(evidence_payload.get("confidence")),
                "observed_at": evidence_payload.get("observed_at") or observed_at or valid_from,
                "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            },
            valid_from,
            actor=actor,
            provenance=provenance_id,
            approval_id=approval_id,
            source_record_id=item_source_record_id,
            input_hash=input_hash or _hash_value(item),
        )
        rows.append(evidence_row)
        evidence_uid_value = evidence_id(evidence_key)
        rows.append(
            object_service.write_relation(
                parent_uid,
                evidence_uid_value,
                relation_type,
                {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID, "relation_role": role},
                valid_from,
                actor=actor,
                provenance=provenance_id,
                approval_id=approval_id,
                source_record_id=item_source_record_id,
                input_hash=input_hash or _hash_value(item),
            )
        )
        citation_payload = _citation_payload(evidence_payload)
        if citation_payload:
            citation_key = str(
                citation_payload.get("citation_id")
                or f"{parent_key}:{role}:{index}:citation:{_hash_value(citation_payload)}"
            )
            citation_row = object_service.write_object(
                "Citation",
                citation_key,
                {
                    "citation_id": citation_key,
                    "source_record_id": citation_payload.get("source_record_id") or item_source_record_id,
                    "document_artifact_id": citation_payload.get("document_artifact_id")
                    or evidence_payload.get("document_artifact_id"),
                    "title": citation_payload.get("title") or evidence_payload.get("title"),
                    "url": citation_payload.get("url"),
                    "source_path": citation_payload.get("source_path"),
                    "quote_hash": citation_payload.get("quote_hash")
                    or _hash_text(str(citation_payload.get("quote") or summary), length=32),
                    "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
                },
                valid_from,
                actor=actor,
                provenance=provenance_id,
                approval_id=approval_id,
                source_record_id=item_source_record_id,
                input_hash=_hash_value(citation_payload),
            )
            rows.append(citation_row)
            citation_uid_value = citation_id(citation_key)
            for citation_relation in ("evidence_has_citation", "evidence_cites_citation"):
                rows.append(
                    object_service.write_relation(
                        evidence_uid_value,
                        citation_uid_value,
                        citation_relation,
                        {"ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID},
                        valid_from,
                        actor=actor,
                        provenance=provenance_id,
                        approval_id=approval_id,
                        source_record_id=item_source_record_id,
                        input_hash=_hash_value(citation_payload),
                    )
                )
        rows.extend(
            maybe_link_source_record_materialization(
                object_service,
                evidence_row=evidence_row,
                source_record_id=item_source_record_id,
                actor=actor,
                provenance_id=provenance_id,
                valid_from=valid_from,
                approval_id=approval_id,
            )
        )
    return rows


def write_claim_evidence_graph(
    object_service: OntologyObjectService,
    *,
    claim_uid: str,
    claim_key: str,
    expected_evidence: Any,
    disconfirming_evidence: Any,
    valid_from: str,
    actor: Mapping[str, Any],
    provenance_id: str,
    approval_id: int | None = None,
    input_hash: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(
        write_parent_evidence_graph(
            object_service,
            parent_uid=claim_uid,
            parent_key=claim_key,
            evidence_items=parse_evidence_items(expected_evidence),
            relation_type="claim_supported_by_evidence",
            role="supporting",
            valid_from=valid_from,
            actor=actor,
            provenance_id=provenance_id,
            approval_id=approval_id,
            input_hash=input_hash,
        )
    )
    rows.extend(
        write_parent_evidence_graph(
            object_service,
            parent_uid=claim_uid,
            parent_key=claim_key,
            evidence_items=parse_evidence_items(disconfirming_evidence),
            relation_type="claim_disconfirmed_by_evidence",
            role="disconfirming",
            valid_from=valid_from,
            actor=actor,
            provenance_id=provenance_id,
            approval_id=approval_id,
            input_hash=input_hash,
        )
    )
    return rows


def maybe_link_source_record_materialization(
    object_service: OntologyObjectService,
    *,
    evidence_row: Mapping[str, Any],
    source_record_id: str | None,
    actor: Mapping[str, Any],
    provenance_id: str,
    valid_from: str,
    approval_id: int | None = None,
) -> list[dict[str, Any]]:
    if not source_record_id:
        return []
    version_ref = _version_ref_payload(evidence_row)
    if not version_ref:
        return []
    source_uid = source_record_object_uid_for(source_record_id)
    target_uid = object_version_ref_id(version_ref["ref_id"])
    object_service.write_object(
        "ObjectVersionRef",
        version_ref["ref_id"],
        {
            "ref_id": version_ref["ref_id"],
            "object_uid": version_ref["object_uid"],
            "object_type": version_ref.get("object_type"),
            "version_id": version_ref["version_id"],
            "valid_from": version_ref.get("valid_from"),
            "tx_from": version_ref.get("tx_from"),
            "temporal_confidence": version_ref.get("temporal_confidence"),
            "source_record_id": source_record_id,
            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
        },
        valid_from,
        actor=actor,
        provenance=provenance_id,
        approval_id=approval_id,
        source_record_id=source_record_id,
    )
    relation = object_service.write_relation(
        source_uid,
        target_uid,
        "source_record_materializes_object_version",
        {
            "ontology_run_id": OPERATIONAL_ONTOLOGY_RUN_ID,
            "source_record_id": source_record_id,
            "object_uid": version_ref["object_uid"],
            "version_id": version_ref["version_id"],
        },
        valid_from,
        actor=actor,
        provenance=provenance_id,
        approval_id=approval_id,
        source_record_id=source_record_id,
    )
    return [relation]


def _citations_for_evidence(reads: OntologyRuntimeReadService, evidence_uid: str) -> list[dict[str, Any]]:
    relations = reads.objects.query_relations(
        relation_type="evidence_has_citation",
        source_object_uid=evidence_uid,
        limit=20,
    )
    citations: list[dict[str, Any]] = []
    for relation in relations:
        citation_uid = str(relation.get("target_object_uid") or "")
        if not citation_uid:
            continue
        row = reads.objects.get_object(citation_uid)
        if row:
            citations.append(object_props(row))
    return citations


def _source_record_for_evidence(
    reads: OntologyRuntimeReadService, evidence: Mapping[str, Any]
) -> dict[str, Any] | None:
    source_record_id = str(evidence.get("source_record_id") or "").strip()
    if not source_record_id:
        return None
    source_uid = source_record_object_id(source_record_id)
    row = reads.objects.get_object(source_uid)
    if row:
        return object_props(row)
    rows = reads.list_objects("SourceRecord", filters={"source_record_id": source_record_id}, limit=1)
    return rows[0] if rows else None


def _evidence_bundle_for_parent(
    reads: OntologyRuntimeReadService,
    parent_uid: str,
    relation_type: str,
) -> list[dict[str, Any]]:
    if not parent_uid:
        return []
    relations = reads.objects.query_relations(
        relation_type=relation_type,
        source_object_uid=parent_uid,
        limit=50,
    )
    bundles: list[dict[str, Any]] = []
    for relation in relations:
        evidence_uid = str(relation.get("target_object_uid") or "")
        row = reads.objects.get_object(evidence_uid)
        if not row:
            continue
        evidence = object_props(row)
        bundles.append(
            {
                "evidence": evidence,
                "citations": _citations_for_evidence(reads, evidence_uid),
                "source_record": _source_record_for_evidence(reads, evidence),
                "relation_role": (relation.get("properties") or {}).get("relation_role"),
            }
        )
    return bundles


def build_ticker_evidence_ledger(reads: OntologyRuntimeReadService, ticker: str) -> dict[str, Any]:
    normalized = str(ticker or "").strip().upper()
    claim_entries: list[dict[str, Any]] = []
    for claim in reads.thesis_claims(ticker=normalized):
        claim_uid = str(claim.get("object_uid") or claim.get("id") or "")
        claim_entries.append(
            {
                "claim_id": claim_uid,
                "claim": claim.get("claim"),
                "status": claim.get("status"),
                "expected_evidence_text": claim.get("expected_evidence"),
                "disconfirming_evidence_text": claim.get("disconfirming_evidence"),
                "supporting_evidence": _evidence_bundle_for_parent(reads, claim_uid, "claim_supported_by_evidence"),
                "disconfirming_evidence": _evidence_bundle_for_parent(
                    reads, claim_uid, "claim_disconfirmed_by_evidence"
                ),
            }
        )

    recommendation_entries: list[dict[str, Any]] = []
    for recommendation in reads.recommendations(ticker=normalized, limit=20):
        rec_uid = str(recommendation.get("object_uid") or recommendation.get("id") or "")
        recommendation_entries.append(
            {
                "recommendation_id": rec_uid,
                "action": recommendation.get("action"),
                "as_of": recommendation.get("as_of"),
                "status": recommendation.get("status"),
                "supporting_evidence": _evidence_bundle_for_parent(
                    reads, rec_uid, "recommendation_supported_by_evidence"
                ),
                "disconfirming_evidence": _evidence_bundle_for_parent(
                    reads, rec_uid, "recommendation_contradicted_by_evidence"
                ),
            }
        )

    return {
        "ticker": normalized,
        "generated_at": _now(),
        "claims": claim_entries,
        "recommendations": recommendation_entries,
        "counts": {
            "claims": len(claim_entries),
            "recommendations": len(recommendation_entries),
            "evidence_items": sum(
                len(entry.get("supporting_evidence") or []) + len(entry.get("disconfirming_evidence") or [])
                for entry in claim_entries + recommendation_entries
            ),
        },
    }
