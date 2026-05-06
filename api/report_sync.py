"""Durable sync for GitHub Actions-generated daily/weekly report artifacts."""

from __future__ import annotations

import hashlib
from typing import Any

from auto_report.recommendations import persist_recommendations, stable_hash, validate_recommendations_payload
from ontology.domain_write_service import ontology_primary_writes_enabled
from ontology.policy import actor_to_dict, system_actor


def _hash_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _propose_report_action(action_id: str, payload: dict[str, Any], *, source_id: str, reason: str) -> dict[str, Any]:
    if ontology_primary_writes_enabled():
        from ontology.command_service import OntologyCommandContext, OntologyCommandService

        return OntologyCommandService().propose_action(
            action_id,
            payload,
            OntologyCommandContext(
                actor=system_actor("report_sync"),
                source_type="workflow",
                source_id=source_id,
            ),
            reason=reason,
        )
    from ontology.action_registry import ActionContext, propose_action

    return propose_action(
        action_id,
        payload,
        ActionContext(
            actor_type="workflow",
            actor_id="report_sync",
            source_type="workflow",
            source_id=source_id,
        ),
        reason=reason,
        once=True,
    )


def _report_id(report_type: str, as_of: str, payload: dict[str, Any]) -> str:
    return str(payload.get("report_id") or f"{report_type}:{as_of}")


def _extract_as_of(report_type: str, payload: dict[str, Any]) -> str:
    explicit = payload.get("as_of")
    if explicit:
        return str(explicit)
    recommendations = payload.get("recommendations")
    if isinstance(recommendations, dict) and recommendations.get("as_of"):
        return str(recommendations["as_of"])
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("as_of", "date", "report_date"):
            if summary.get(key):
                return str(summary[key])
    raise ValueError(f"{report_type} report sync payload is missing as_of.")


def _create_report_action_items(report_type: str, as_of: str, report_id: str, payload: dict[str, Any]) -> int:
    summary = _as_dict(payload.get("summary"))
    count = 0

    if report_type == "daily":
        for ticker in _as_list(summary.get("positions_flagged")):
            ticker_s = str(ticker).strip().upper()
            if not ticker_s:
                continue
            _propose_report_action(
                "create_action_item",
                {
                    "description": f"Review daily report flag for {ticker_s} ({as_of})",
                    "action_type": "review",
                    "ticker": ticker_s,
                    "urgency": "normal",
                },
                source_id=report_id,
                reason=f"Daily report flagged {ticker_s} ({as_of})",
            )
            count += 1
    thesis = _as_dict(summary.get("thesis_monitoring"))
    for ticker in _as_list(thesis.get("positions_needing_reassessment")):
        ticker_s = str(ticker).strip().upper()
        if not ticker_s:
            continue
        _propose_report_action(
            "create_action_item",
            {
                "description": f"Reassess thesis after {report_type} report for {ticker_s} ({as_of})",
                "action_type": "review",
                "ticker": ticker_s,
                "urgency": "high",
            },
            source_id=report_id,
            reason=f"{report_type.title()} report thesis reassessment for {ticker_s} ({as_of})",
        )
        count += 1
    return count


def _create_watch_trigger_approvals(report_type: str, as_of: str, report_id: str, payload: dict[str, Any]) -> int:
    summary = _as_dict(payload.get("summary"))
    count = 0
    for condition in _as_list(summary.get("watchlist_triggers")):
        condition_s = str(condition).strip()
        if not condition_s:
            continue
        _propose_report_action(
            "create_watch_trigger",
            {
                "condition": condition_s,
                "trigger_type": "macro",
                "ticker": None,
                "definition": None,
            },
            source_id=report_id,
            reason=f"{report_type.title()} report watch trigger ({as_of})",
        )
        count += 1
    return count


def _persist_weekly_thesis_evaluations(as_of: str, payload: dict[str, Any]) -> int:
    summary = _as_dict(payload.get("summary"))
    thesis = _as_dict(summary.get("thesis_monitoring"))
    evals = thesis.get("thesis_evaluations")
    if not isinstance(evals, list) or not evals:
        return 0
    report_id = str(payload.get("report_id") or f"weekly:{as_of}")
    count = 0
    for evaluation in evals:
        if not isinstance(evaluation, dict) or not evaluation.get("ticker"):
            continue
        _propose_report_action(
            "save_evaluation",
            {"evaluated_at": as_of, **evaluation},
            source_id=report_id,
            reason=f"Weekly thesis evaluation for {str(evaluation.get('ticker')).upper()} ({as_of})",
        )
        count += 1
    return count


def _persist_thesis_claims(report_id: str, payload: dict[str, Any]) -> int:
    count = 0
    for claim in _as_list(payload.get("thesis_claims")):
        if not isinstance(claim, dict) or not claim.get("ticker") or not claim.get("claim"):
            continue
        _propose_report_action(
            "create_thesis_claim",
            {**claim, "source_type": "workflow", "source_id": report_id},
            source_id=report_id,
            reason=f"Report thesis claim for {str(claim.get('ticker')).upper()}",
        )
        count += 1
    return count


def persist_report_sync(report_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    if report_type not in {"daily", "weekly"}:
        raise ValueError("report_type must be daily or weekly")

    as_of = _extract_as_of(report_type, payload)
    report_id = _report_id(report_type, as_of, payload)
    metadata = _as_dict(payload.get("metadata"))
    summary = _as_dict(payload.get("summary"))
    recommendations_payload = payload.get("recommendations")
    if not isinstance(recommendations_payload, dict):
        raise ValueError("report sync payload is missing recommendations.")
    recommendations_payload = validate_recommendations_payload(
        recommendations_payload,
        report_type=report_type,
        as_of=as_of,
        stance=str(recommendations_payload.get("stance") or "Neutral / Watchful"),
        data_quality=summary.get("data_quality", {}) if isinstance(summary.get("data_quality"), dict) else {},
    )

    report_md = str(payload.get("report_md") or payload.get("report") or "")
    bundle = _as_dict(payload.get("bundle"))
    report_hash = str(payload.get("report_hash") or metadata.get("report_hash") or _hash_text(report_md) or "")
    input_hash = str(
        payload.get("input_hash")
        or metadata.get("input_hash")
        or stable_hash({"summary": summary, "bundle": bundle, "recommendations": recommendations_payload})
    )

    report_run_payload = {
        "report_id": report_id,
        "report_type": report_type,
        "as_of": as_of,
        "source": metadata.get("source") or "github_actions",
        "source_run_id": metadata.get("github_run_id") or metadata.get("source_run_id"),
        "source_url": metadata.get("source_url"),
        "status": "completed",
        "report_hash": report_hash,
        "input_hash": input_hash,
        "summary": summary,
        "artifact_paths": _as_dict(payload.get("artifact_paths")),
        "issue_url": metadata.get("issue_url"),
        "ontology_run_id": "operational",
    }
    if ontology_primary_writes_enabled():
        from ontology.object_service import OntologyObjectService

        report_actor = system_actor("report_sync")
        report_row = OntologyObjectService().write_object(
            "ReportRun",
            report_id,
            report_run_payload,
            as_of,
            actor=actor_to_dict(report_actor),
            provenance=f"pv:report_sync:{report_id}",
            input_hash=input_hash,
        )
        report_run = {
            **report_run_payload,
            "id": report_row.get("object_uid"),
            "object_uid": report_row.get("object_uid"),
        }
    else:
        from portfolio import core_db

        report_row = core_db.upsert_report_run(report_run_payload)
        report_run = {
            **report_run_payload,
            **report_row,
            "id": report_row.get("report_id") or report_id,
            "object_uid": f"report_run:{report_row.get('report_id') or report_id}",
        }

    persisted_recommendations = persist_recommendations(
        recommendations_payload,
        source_report_path=str(_as_dict(payload.get("artifact_paths")).get("recommendations_md") or ""),
        source_json_path=str(_as_dict(payload.get("artifact_paths")).get("recommendations_json") or ""),
        prompt_metadata={
            "report_id": report_id,
            "model": metadata.get("model"),
            "prompt_hash": metadata.get("prompt_hash"),
            "input_hash": input_hash,
            "validation_status": "ok" if recommendations_payload.get("recommendation_status") != "error" else "error",
            "source_quality_summary": summary.get("data_quality", {}),
        },
    )
    try:
        from ontology.decision_writeback import record_report_output

        record_report_output(
            report_type=report_type,
            payload=payload,
            report_run=report_run,
            persisted_recommendations=persisted_recommendations,
            actor={"actor_type": "workflow", "actor_id": "report_sync"},
            provenance=f"pv:report_sync:{report_id}",
        )
    except Exception:
        if ontology_primary_writes_enabled():
            raise

    thesis_evaluations = _persist_weekly_thesis_evaluations(as_of, payload) if report_type == "weekly" else 0
    action_items = _create_report_action_items(report_type, as_of, report_id, payload)
    watch_trigger_approvals = _create_watch_trigger_approvals(report_type, as_of, report_id, payload)
    thesis_claims = _persist_thesis_claims(report_id, payload)

    return {
        "status": "synced",
        "report_run": report_run,
        "counts": {
            "recommendations": len(persisted_recommendations),
            "thesis_evaluations": thesis_evaluations,
            "action_items": action_items,
            "watch_trigger_approvals": watch_trigger_approvals,
            "thesis_claims": thesis_claims,
        },
    }
