"""Admin provenance trace API endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from api.exceptions import ValidationError
from api.routers.auth import require_actor
from ontology.domain_write_service import ontology_primary_writes_enabled
from ontology.policy import Actor, PolicyDenied
from ontology.runtime_read_service import OntologyRuntimeReadService
from ontology.schemas.relations import PROVENANCE_RELATION_TYPES

router = APIRouter()
ActorDep = Annotated[Actor, Depends(require_actor)]


def _require_admin(actor: Actor) -> None:
    roles = {role.lower() for role in actor.roles}
    if actor.actor_type != "system" and "admin" not in roles:
        raise PolicyDenied("Admin access is required for provenance traces.")


@router.get("/provenance/trace")
def get_provenance_trace(
    actor: ActorDep,
    workflow_run_id: str | None = None,
    ontology_run_id: str | None = None,
    object_version_id: str | None = None,
    relation_version_id: str | None = None,
    source_record_id: str | None = None,
    snapshot_id: str | None = None,
    approval_id: str | None = None,
    action_run_id: str | None = None,
    agent_session_id: str | None = None,
    event_id: str | None = None,
    ref_type: str | None = None,
    ref_id: str | None = None,
    max_depth: int = 3,
):
    _require_admin(actor)
    explicit_ref_selector = ref_type is not None and ref_id is not None
    if (ref_type is None) != (ref_id is None):
        raise ValidationError("ref_type and ref_id must be provided together.")
    typed_ref_selectors = {
        "ontology_object_version": object_version_id,
        "relation_version": relation_version_id,
        "source_record": source_record_id,
        "computed_snapshot_version": snapshot_id,
    }
    provided_typed_refs = [(key, value) for key, value in typed_ref_selectors.items() if value is not None]
    if provided_typed_refs and explicit_ref_selector:
        raise ValidationError("Provide exactly one provenance trace selector.")
    if provided_typed_refs:
        ref_type, ref_id = provided_typed_refs[0]
    selectors = [
        workflow_run_id,
        ontology_run_id,
        object_version_id,
        relation_version_id,
        source_record_id,
        snapshot_id,
        approval_id,
        action_run_id,
        agent_session_id,
        event_id,
        f"{ref_type}:{ref_id}" if explicit_ref_selector and not provided_typed_refs else None,
    ]
    if sum(value is not None for value in selectors) != 1:
        raise ValidationError("Provide exactly one provenance trace selector.")
    if max_depth < 1 or max_depth > 8:
        raise ValidationError("max_depth must be between 1 and 8.")

    return _ontology_trace(
        workflow_run_id=workflow_run_id,
        ontology_run_id=ontology_run_id,
        approval_id=approval_id,
        action_run_id=action_run_id,
        agent_session_id=agent_session_id,
        event_id=event_id,
        ref_type=ref_type,
        ref_id=ref_id,
        max_depth=max_depth,
    )


@router.get("/provenance/entity/{ref_type}/{ref_id}")
def get_entity_provenance(ref_type: str, ref_id: str, actor: ActorDep, max_depth: int = 3):
    _require_admin(actor)
    if not ref_type.strip() or not ref_id.strip():
        raise ValidationError("ref_type and ref_id are required.")
    if max_depth < 1 or max_depth > 8:
        raise ValidationError("max_depth must be between 1 and 8.")

    return _ontology_trace(ref_type=ref_type, ref_id=ref_id, max_depth=max_depth)


@router.get("/governance/lineage")
def get_governance_lineage_report(
    actor: ActorDep,
    recommendation_id: str | None = None,
    approval_id: str | None = None,
    action_run_id: str | None = None,
    workflow_run_id: str | None = None,
    object_version_id: str | None = None,
    relation_version_id: str | None = None,
    max_depth: int = 5,
):
    _require_admin(actor)
    if max_depth < 1 or max_depth > 8:
        raise ValidationError("max_depth must be between 1 and 8.")
    selectors = [recommendation_id, approval_id, action_run_id, workflow_run_id, object_version_id, relation_version_id]
    if sum(value is not None for value in selectors) != 1:
        raise ValidationError("Provide exactly one governance lineage selector.")

    if not ontology_primary_writes_enabled():
        from portfolio import core_db

        return core_db.get_decision_lineage_report(
            recommendation_id=_legacy_numeric_id(recommendation_id),
            approval_id=_legacy_numeric_id(approval_id),
            action_run_id=_legacy_numeric_id(action_run_id),
            workflow_run_id=workflow_run_id,
            object_version_id=object_version_id,
            relation_version_id=relation_version_id,
            max_depth=max_depth,
        )

    return _ontology_trace(
        recommendation_id=recommendation_id,
        approval_id=approval_id,
        action_run_id=action_run_id,
        workflow_run_id=workflow_run_id,
        object_version_id=object_version_id,
        relation_version_id=relation_version_id,
        max_depth=max_depth,
    )


def _legacy_numeric_id(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    try:
        return int(text)
    except (TypeError, ValueError):
        raise ValidationError(f"Legacy provenance selector must be numeric: {value}") from None


def _ontology_trace(max_depth: int = 3, **selector: str | None) -> dict:
    clean_selector = {key: value for key, value in selector.items() if value is not None}
    if not ontology_primary_writes_enabled():
        from portfolio import core_db

        return core_db.get_provenance_trace(
            workflow_run_id=clean_selector.get("workflow_run_id"),
            ontology_run_id=clean_selector.get("ontology_run_id"),
            approval_id=_legacy_numeric_id(clean_selector.get("approval_id")),
            action_run_id=_legacy_numeric_id(clean_selector.get("action_run_id")),
            agent_session_id=clean_selector.get("agent_session_id"),
            event_id=clean_selector.get("event_id"),
            ref_type=clean_selector.get("ref_type"),
            ref_id=clean_selector.get("ref_id"),
            max_depth=max_depth,
        )
    reads = OntologyRuntimeReadService()
    events = reads.list_objects("ProvenanceEvent", limit=500)
    reference_types = (
        "ObjectVersionRef",
        "RelationVersionRef",
        "SchemaDefinitionRef",
        "OntologyRunRef",
        "AgentSessionRef",
        "ModelCallRef",
        "ToolCallRef",
        "ComputedSnapshotRef",
        "SourceRecord",
        "WorkflowRun",
        "WorkflowArtifact",
        "Approval",
        "ActionRun",
        "AuditEvent",
    )
    references = []
    for object_type in reference_types:
        references.extend(reads.list_objects(object_type, limit=500))
    relations = []
    for relation_type in sorted(PROVENANCE_RELATION_TYPES):
        relations.extend(
            _relation_payload(row)
            for row in reads.objects.query_relations(relation_type=relation_type, include_history=True, limit=500)
        )
    if clean_selector:
        needle_values = {str(value) for value in clean_selector.values()}
        events = [event for event in events if any(value in str(event.values()) for value in needle_values)]
        references = [ref for ref in references if any(value in str(ref.values()) for value in needle_values)]
        relations = [
            relation for relation in relations if any(value in str(relation.values()) for value in needle_values)
        ]
    timeline = sorted(
        [
            {"kind": "event", "id": event.get("id"), "status": event.get("status"), "at": event.get("started_at")}
            for event in events
        ]
        + [
            {
                "kind": "relation",
                "id": relation.get("relation_uid"),
                "relation_type": relation.get("relation_type"),
                "at": ((relation.get("_meta") or {}).get("temporal") or {}).get("valid_from"),
            }
            for relation in relations
        ],
        key=lambda item: str(item.get("at") or ""),
    )
    return {
        "selector": clean_selector,
        "seed": _seed(clean_selector),
        "events": events,
        "references": references,
        "relations": relations,
        "timeline": timeline,
        "lineage_state": "ontology",
        "event_count": len(events),
        "reference_count": len(references),
        "relation_count": len(relations),
    }


def _relation_payload(row: dict) -> dict:
    properties = dict(row.get("properties") or row.get("properties_json") or {})
    relation_uid = str(row.get("relation_uid") or properties.get("id") or "")
    properties["id"] = relation_uid
    properties["relation_uid"] = relation_uid
    return {**row, "properties": properties}


def _seed(selector: dict[str, str]) -> dict[str, str] | None:
    if not selector:
        return None
    key, value = next(iter(selector.items()))
    if key == "ref_type":
        return {"ref_type": selector.get("ref_type", ""), "ref_id": selector.get("ref_id", "")}
    return {"selector_type": key, "selector_id": value}
