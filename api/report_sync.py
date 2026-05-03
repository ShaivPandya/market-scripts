"""Durable sync for GitHub Actions-generated daily/weekly report artifacts."""

from __future__ import annotations

import hashlib
from typing import Any

from auto_report.recommendations import persist_recommendations, stable_hash, validate_recommendations_payload


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


def _create_report_notes(report_type: str, as_of: str, report_id: str, payload: dict[str, Any]) -> int:
    from portfolio.core_db import create_research_note_once

    count = 0
    report_md = str(payload.get("report_md") or payload.get("report") or "").strip()
    if report_md:
        create_research_note_once(
            title=f"{report_type.title()} Report - {as_of}",
            content=report_md[:20000],
            note_type="workflow_output",
            source_type="workflow",
            source_id=report_id,
        )
        count += 1

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    thesis = summary.get("thesis_monitoring") if isinstance(summary.get("thesis_monitoring"), dict) else {}
    for item in _as_list(thesis.get("material_developments")):
        if not isinstance(item, dict) or not item.get("summary"):
            continue
        ticker = str(item.get("ticker") or "").upper() or None
        create_research_note_once(
            title=f"{ticker or 'Portfolio'} thesis development - {as_of}",
            content=str(item["summary"]),
            ticker=ticker,
            note_type="risk_assessment" if item.get("type") in {"contradicts_thesis", "new_risk"} else "general",
            source_type="workflow",
            source_id=report_id,
        )
        count += 1
    return count


def _create_report_action_items(report_type: str, as_of: str, report_id: str, payload: dict[str, Any]) -> int:
    from portfolio.core_db import create_action_item_once

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    count = 0

    if report_type == "daily":
        for ticker in _as_list(summary.get("positions_flagged")):
            ticker_s = str(ticker).strip().upper()
            if not ticker_s:
                continue
            create_action_item_once(
                description=f"Review daily report flag for {ticker_s} ({as_of})",
                action_type="review",
                ticker=ticker_s,
                urgency="normal",
                source_type="workflow",
                source_id=report_id,
            )
            count += 1
    thesis = summary.get("thesis_monitoring") if isinstance(summary.get("thesis_monitoring"), dict) else {}
    for ticker in _as_list(thesis.get("positions_needing_reassessment")):
        ticker_s = str(ticker).strip().upper()
        if not ticker_s:
            continue
        create_action_item_once(
            description=f"Reassess thesis after {report_type} report for {ticker_s} ({as_of})",
            action_type="review",
            ticker=ticker_s,
            urgency="high",
            source_type="workflow",
            source_id=report_id,
        )
        count += 1
    return count


def _create_watch_trigger_approvals(report_type: str, as_of: str, report_id: str, payload: dict[str, Any]) -> int:
    from portfolio.action_registry import ActionContext, propose_action

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    count = 0
    for condition in _as_list(summary.get("watchlist_triggers")):
        condition_s = str(condition).strip()
        if not condition_s:
            continue
        propose_action(
            "create_watch_trigger",
            {
                "condition": condition_s,
                "trigger_type": "macro",
                "ticker": None,
                "definition": None,
            },
            ActionContext(actor_type="workflow", source_type="workflow", source_id=report_id),
            reason=f"{report_type.title()} report watch trigger ({as_of})",
            once=True,
        )
        count += 1
    return count


def _persist_weekly_thesis_evaluations(as_of: str, payload: dict[str, Any]) -> int:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    thesis = summary.get("thesis_monitoring") if isinstance(summary.get("thesis_monitoring"), dict) else {}
    evals = thesis.get("thesis_evaluations") if isinstance(thesis, dict) else None
    if not isinstance(evals, list) or not evals:
        return 0
    from portfolio.thesis_db import save_evaluations, upsert_thesis_meta

    saved = save_evaluations(as_of, evals)
    for ticker in thesis.get("positions_reviewed", []):
        if ticker:
            upsert_thesis_meta(str(ticker).upper())
    return saved


def _persist_thesis_claims(report_id: str, payload: dict[str, Any]) -> int:
    from portfolio.core_db import create_thesis_claim_once

    count = 0
    for claim in _as_list(payload.get("thesis_claims")):
        if not isinstance(claim, dict) or not claim.get("ticker") or not claim.get("claim"):
            continue
        create_thesis_claim_once({**claim, "source_type": "workflow", "source_id": report_id})
        count += 1
    return count


def persist_report_sync(report_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    if report_type not in {"daily", "weekly"}:
        raise ValueError("report_type must be daily or weekly")

    as_of = _extract_as_of(report_type, payload)
    report_id = _report_id(report_type, as_of, payload)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
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
    bundle = payload.get("bundle") if isinstance(payload.get("bundle"), dict) else {}
    report_hash = str(payload.get("report_hash") or metadata.get("report_hash") or _hash_text(report_md) or "")
    input_hash = str(
        payload.get("input_hash")
        or metadata.get("input_hash")
        or stable_hash({"summary": summary, "bundle": bundle, "recommendations": recommendations_payload})
    )

    from portfolio.core_db import upsert_report_run

    report_run = upsert_report_run(
        {
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
            "artifact_paths": payload.get("artifact_paths") or {},
            "issue_url": metadata.get("issue_url"),
        }
    )

    persisted_recommendations = persist_recommendations(
        recommendations_payload,
        source_report_path=str((payload.get("artifact_paths") or {}).get("recommendations_md") or ""),
        source_json_path=str((payload.get("artifact_paths") or {}).get("recommendations_json") or ""),
        prompt_metadata={
            "report_id": report_id,
            "model": metadata.get("model"),
            "prompt_hash": metadata.get("prompt_hash"),
            "input_hash": input_hash,
            "validation_status": "ok" if recommendations_payload.get("recommendation_status") != "error" else "error",
            "source_quality_summary": summary.get("data_quality", {}),
        },
    )

    thesis_evaluations = _persist_weekly_thesis_evaluations(as_of, payload) if report_type == "weekly" else 0
    research_notes = _create_report_notes(report_type, as_of, report_id, payload)
    action_items = _create_report_action_items(report_type, as_of, report_id, payload)
    watch_trigger_approvals = _create_watch_trigger_approvals(report_type, as_of, report_id, payload)
    thesis_claims = _persist_thesis_claims(report_id, payload)

    return {
        "status": "synced",
        "report_run": report_run,
        "counts": {
            "recommendations": len(persisted_recommendations),
            "thesis_evaluations": thesis_evaluations,
            "research_notes": research_notes,
            "action_items": action_items,
            "watch_trigger_approvals": watch_trigger_approvals,
            "thesis_claims": thesis_claims,
        },
    }
