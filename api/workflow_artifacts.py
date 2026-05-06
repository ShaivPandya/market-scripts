"""
Extract and persist structured artifacts from workflow synthesis output.

Workflows instruct the LLM to include a fenced ```artifacts ... ``` JSON block
at the end of synthesis. This module parses that block and creates pending
approvals for each artifact.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from api.audit import emit_audit_event
from ontology.command_service import OntologyCommandContext, OntologyCommandService, OntologyCommandValidationError
from ontology.object_service import OntologyObjectService
from ontology.policy import system_actor

logger = logging.getLogger("api.workflow_artifacts")

_ARTIFACTS_PATTERN = re.compile(
    r"```artifacts\s*\n(.*?)```",
    re.DOTALL,
)


@dataclass(frozen=True)
class ArtifactBinding:
    action_id: str | None
    reason: str
    multiple: bool = False
    required_keys: tuple[str, ...] = ()


_ARTIFACT_BINDINGS: dict[str, ArtifactBinding] = {
    "evaluation_draft": ArtifactBinding(
        "save_evaluation", "Workflow-generated evaluation", required_keys=("thesis_status",)
    ),
    "action_items": ArtifactBinding(
        "create_action_item",
        "Workflow-generated action item",
        multiple=True,
        required_keys=("description",),
    ),
    "watch_triggers": ArtifactBinding(
        "create_watch_trigger",
        "Workflow-generated watch trigger",
        multiple=True,
        required_keys=("condition",),
    ),
    "catalyst_updates": ArtifactBinding(
        "update_catalyst_status",
        "Workflow-suggested catalyst status change",
        multiple=True,
    ),
    "kill_condition_updates": ArtifactBinding(
        "update_kill_condition_status",
        "Workflow-suggested kill condition status change",
        multiple=True,
    ),
    "thesis_status_change": ArtifactBinding(
        "change_thesis_status",
        "Workflow-suggested thesis status change",
        required_keys=("new_status",),
    ),
    "news_digest_deletes": ArtifactBinding(
        "delete_portfolio_news_digest",
        "Workflow-suggested news digest delete",
        multiple=True,
        required_keys=("digest_id",),
    ),
}


def extract_artifacts(synthesis_text: str, workflow_name: str) -> dict:
    """Parse the ```artifacts``` JSON block from synthesis text.

    Returns the parsed dict, or empty dict if not found or parse fails.
    """
    match = _ARTIFACTS_PATTERN.search(synthesis_text)
    if not match:
        emit_audit_event(
            "workflow.artifacts.parse",
            "workflow",
            "succeeded",
            object_refs=[{"type": "workflow", "id": workflow_name}],
            after_summary={"workflow_name": workflow_name, "artifact_block_found": False},
        )
        return {}

    raw = match.group(1).strip()
    try:
        artifacts = json.loads(raw)
        if not isinstance(artifacts, dict):
            logger.warning("Artifacts block is not a dict (workflow=%s)", workflow_name)
            emit_audit_event(
                "workflow.artifacts.parse",
                "workflow",
                "failed",
                object_refs=[{"type": "workflow", "id": workflow_name}],
                after_summary={"workflow_name": workflow_name, "artifact_block_found": True},
                error="Artifacts block is not a dict",
            )
            return {}
        emit_audit_event(
            "workflow.artifacts.parse",
            "workflow",
            "succeeded",
            object_refs=[{"type": "workflow", "id": workflow_name}],
            after_summary={
                "workflow_name": workflow_name,
                "artifact_block_found": True,
                "artifact_keys": sorted(artifacts.keys()),
            },
        )
        return artifacts
    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to parse artifacts JSON (workflow=%s): %s\nRaw: %s",
            workflow_name,
            exc,
            raw[:500],
        )
        emit_audit_event(
            "workflow.artifacts.parse",
            "workflow",
            "failed",
            object_refs=[{"type": "workflow", "id": workflow_name}],
            after_summary={
                "workflow_name": workflow_name,
                "artifact_block_found": True,
                "raw_hash": hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16],
            },
            error=str(exc),
        )
        return {}


def persist_artifacts(
    run_id: str,
    ticker: str | None,
    artifacts: dict,
) -> int:
    """Create ontology workflow artifacts and pending approvals for governed artifacts.

    Returns the number of approvals created.
    """
    from ontology.domain_write_service import ontology_primary_writes_enabled

    if not ontology_primary_writes_enabled():
        from ontology.action_registry import propose_workflow_artifact

        count = 0
        for artifact_key in _ARTIFACT_BINDINGS:
            count += propose_workflow_artifact(
                artifact_key,
                artifacts.get(artifact_key),
                run_id=run_id,
                ticker=ticker,
            )
        _emit_persisted_audit(run_id, ticker, artifacts, count)
        return count

    actor = system_actor("workflow_artifacts")
    context = OntologyCommandContext(actor=actor, source_type="workflow", source_id=run_id)
    command_service = OntologyCommandService()
    object_service = OntologyObjectService()
    count = 0
    for artifact_key, binding in _ARTIFACT_BINDINGS.items():
        for item_index, item in enumerate(_artifact_items(artifacts.get(artifact_key), multiple=binding.multiple)):
            if binding.required_keys and any(not item.get(key) for key in binding.required_keys):
                continue
            artifact_uid = (
                f"workflow_artifact:{hashlib.sha256(f'{run_id}:{artifact_key}:{item_index}'.encode()).hexdigest()[:24]}"
            )
            provenance_id = f"pv:workflow_artifact:{run_id}:{artifact_key}:{item_index}"
            now = datetime.now(UTC).isoformat()
            payload = _artifact_payload(artifact_key, item, ticker)
            object_service.write_object(
                "WorkflowArtifact",
                artifact_uid,
                {
                    "artifact_id": artifact_uid,
                    "workflow_run_id": run_id,
                    "artifact_key": artifact_key,
                    "artifact_index": item_index,
                    "artifact_value": item,
                    "artifact_hash": hashlib.sha256(
                        json.dumps(item, sort_keys=True, default=str).encode("utf-8")
                    ).hexdigest(),
                    "state": "proposed" if binding.action_id else "extracted",
                    "action_id": binding.action_id,
                    "provenance_event_id": provenance_id,
                    "metadata": {"ticker": ticker, "payload": payload},
                    "ontology_run_id": "operational",
                },
                now,
                actor=actor,
                provenance=provenance_id,
            )
            if not binding.action_id:
                continue
            try:
                command_service.propose_action(
                    binding.action_id,
                    payload,
                    context,
                    reason=binding.reason,
                    entity_id=artifact_uid,
                )
            except OntologyCommandValidationError:
                logger.warning(
                    "Skipping invalid workflow artifact proposal (run_id=%s artifact_key=%s)",
                    run_id,
                    artifact_key,
                    exc_info=True,
                )
                continue
            count += 1

    if count:
        logger.info("Persisted %d artifacts as pending approvals (run_id=%s)", count, run_id)

    _emit_persisted_audit(run_id, ticker, artifacts, count)
    return count


def _emit_persisted_audit(run_id: str, ticker: str | None, artifacts: dict, count: int) -> None:
    emit_audit_event(
        "workflow.artifacts.persisted",
        "workflow",
        "succeeded",
        object_refs=[{"type": "workflow_run", "id": run_id}],
        after_summary={
            "run_id": run_id,
            "ticker": ticker,
            "approval_count": count,
            "artifact_keys": sorted(artifacts.keys()),
        },
        source_lineage={"run_id": run_id, "ticker": ticker},
    )


def _artifact_items(value: Any, *, multiple: bool) -> list[dict[str, Any]]:
    if multiple:
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
    return [value] if isinstance(value, dict) else []


def _artifact_payload(artifact_key: str, item: dict[str, Any], ticker: str | None) -> dict[str, Any]:
    payload = dict(item)
    if ticker and not payload.get("ticker"):
        payload["ticker"] = ticker
    if artifact_key == "thesis_status_change":
        return {
            "ticker": str(payload.get("ticker") or ticker or "").strip().upper(),
            "status": payload.get("new_status") or payload.get("status"),
            "reason": str(payload.get("reason") or ""),
        }
    return payload
