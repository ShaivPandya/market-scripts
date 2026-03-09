"""
Extract and persist structured artifacts from workflow synthesis output.

Workflows instruct the LLM to include a fenced ```artifacts ... ``` JSON block
at the end of synthesis. This module parses that block and creates pending
approvals for each artifact.
"""

from __future__ import annotations

import json
import logging
import re

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
        return {}

    raw = match.group(1).strip()
    try:
        artifacts = json.loads(raw)
        if not isinstance(artifacts, dict):
            logger.warning("Artifacts block is not a dict (workflow=%s)", workflow_name)
            return {}
        return artifacts
    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to parse artifacts JSON (workflow=%s): %s\nRaw: %s",
            workflow_name,
            exc,
            raw[:500],
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
    from portfolio.core_db import create_pending_approval

    count = 0

    # Evaluation draft -> pending approval for evaluation
    eval_draft = artifacts.get("evaluation_draft")
    if isinstance(eval_draft, dict):
        create_pending_approval(
            entity_type="evaluation",
            proposed_change=eval_draft,
            ticker=eval_draft.get("ticker", ticker),
            reason="Workflow-generated evaluation",
            source_type="workflow",
            source_id=run_id,
        )
        count += 1

    # Action items
    action_items = artifacts.get("action_items", [])
    if isinstance(action_items, list):
        for item in action_items:
            if not isinstance(item, dict) or not item.get("description"):
                continue
            create_pending_approval(
                entity_type="action_item",
                proposed_change=item,
                ticker=item.get("ticker", ticker),
                reason="Workflow-generated action item",
                source_type="workflow",
                source_id=run_id,
            )
            count += 1

    # Watch triggers
    watch_triggers = artifacts.get("watch_triggers", [])
    if isinstance(watch_triggers, list):
        for trigger in watch_triggers:
            if not isinstance(trigger, dict) or not trigger.get("condition"):
                continue
            create_pending_approval(
                entity_type="watch_trigger",
                proposed_change=trigger,
                ticker=trigger.get("ticker", ticker),
                reason="Workflow-generated watch trigger",
                source_type="workflow",
                source_id=run_id,
            )
            count += 1

    # Catalyst updates
    catalyst_updates = artifacts.get("catalyst_updates", [])
    if isinstance(catalyst_updates, list):
        for update in catalyst_updates:
            if not isinstance(update, dict):
                continue
            create_pending_approval(
                entity_type="catalyst_status",
                proposed_change=update,
                entity_id=update.get("catalyst_id"),
                ticker=ticker,
                reason="Workflow-suggested catalyst status change",
                source_type="workflow",
                source_id=run_id,
            )
            count += 1

    # Kill condition updates
    kc_updates = artifacts.get("kill_condition_updates", [])
    if isinstance(kc_updates, list):
        for update in kc_updates:
            if not isinstance(update, dict):
                continue
            create_pending_approval(
                entity_type="kill_condition_status",
                proposed_change=update,
                entity_id=update.get("kill_condition_id"),
                ticker=ticker,
                reason="Workflow-suggested kill condition status change",
                source_type="workflow",
                source_id=run_id,
            )
            count += 1

    # Thesis status change
    thesis_change = artifacts.get("thesis_status_change")
    if isinstance(thesis_change, dict) and thesis_change.get("new_status"):
        create_pending_approval(
            entity_type="thesis_status",
            proposed_change={
                "ticker": ticker,
                "new_status": thesis_change["new_status"],
                "reason": thesis_change.get("reason", ""),
            },
            ticker=ticker,
            reason="Workflow-suggested thesis status change",
            source_type="workflow",
            source_id=run_id,
        )
        count += 1

    if count:
        logger.info("Persisted %d artifacts as pending approvals (run_id=%s)", count, run_id)

    return count
