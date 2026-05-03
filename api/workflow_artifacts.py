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

from api.audit import emit_audit_event

logger = logging.getLogger("api.workflow_artifacts")

_ARTIFACTS_PATTERN = re.compile(
    r"```artifacts\s*\n(.*?)```",
    re.DOTALL,
)


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
    """Create pending_approvals for each artifact in the extracted dict.

    Returns the number of approvals created.
    """
    from ontology.action_registry import propose_workflow_artifact, workflow_artifact_keys

    count = 0
    for artifact_key in workflow_artifact_keys():
        count += propose_workflow_artifact(
            artifact_key,
            artifacts.get(artifact_key),
            run_id=run_id,
            ticker=ticker,
        )

    if count:
        logger.info("Persisted %d artifacts as pending approvals (run_id=%s)", count, run_id)

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
    return count
