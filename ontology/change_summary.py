"""Deterministic operational change summaries for workspace and dossier views."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 7
MAX_CHANGE_ROWS_PER_TYPE = 500


@dataclass(frozen=True)
class ObjectChangeConfig:
    category: str
    title_fields: tuple[str, ...]
    fields: tuple[str, ...]


CHANGE_OBJECT_CONFIGS: dict[str, ObjectChangeConfig] = {
    "Evaluation": ObjectChangeConfig(
        category="evaluation",
        title_fields=("ticker",),
        fields=("ticker", "thesis_status", "action", "confidence", "risk_flag", "evaluated_at"),
    ),
    "Catalyst": ObjectChangeConfig(
        category="catalyst",
        title_fields=("description", "name"),
        fields=("ticker", "description", "category", "status", "target_date", "evidence"),
    ),
    "KillCondition": ObjectChangeConfig(
        category="kill_condition",
        title_fields=("condition", "description"),
        fields=("ticker", "condition", "metric", "threshold", "status", "triggered_at"),
    ),
    "ThesisClaim": ObjectChangeConfig(
        category="thesis_claim",
        title_fields=("claim", "summary", "title"),
        fields=("ticker", "claim", "summary", "status", "confidence", "updated_at"),
    ),
    "Approval": ObjectChangeConfig(
        category="approval",
        title_fields=("reason", "action_id", "entity_type"),
        fields=("ticker", "status", "application_status", "action_id", "entity_type", "reason"),
    ),
    "Recommendation": ObjectChangeConfig(
        category="recommendation",
        title_fields=("ticker", "report_type", "action"),
        fields=(
            "ticker",
            "report_type",
            "action",
            "recommendation_status",
            "critical_data_quality",
            "approval_status",
            "outcome_status",
            "as_of",
        ),
    ),
    "ActionItem": ObjectChangeConfig(
        category="action_item",
        title_fields=("description", "action_type"),
        fields=("ticker", "description", "action_type", "urgency", "status", "created_at"),
    ),
    "WatchTrigger": ObjectChangeConfig(
        category="watch_trigger",
        title_fields=("condition", "trigger_type"),
        fields=("ticker", "condition", "trigger_type", "status", "last_checked_at", "last_evidence"),
    ),
    "WorkflowRun": ObjectChangeConfig(
        category="workflow",
        title_fields=("workflow_name", "run_id"),
        fields=("ticker", "workflow_name", "status", "started_at", "completed_at", "updated_at"),
    ),
    "ReportRun": ObjectChangeConfig(
        category="report",
        title_fields=("ticker", "report_type", "run_id", "id"),
        fields=("ticker", "report_type", "status", "as_of", "synced_at", "updated_at"),
    ),
    "MonitorHit": ObjectChangeConfig(
        category="monitor_hit",
        title_fields=("entity_label", "hit_type", "ticker"),
        fields=("ticker", "entity_type", "entity_id", "entity_label", "hit_type", "severity", "status", "evidence"),
    ),
    "OpportunityCandidate": ObjectChangeConfig(
        category="opportunity_candidate",
        title_fields=("trigger", "ticker", "candidate_id"),
        fields=(
            "ticker",
            "trigger",
            "opportunity_type",
            "why_now",
            "next_action",
            "status",
            "decision_state",
            "updated_at",
        ),
    ),
}


class ChangeSummaryInputError(ValueError):
    """Raised when a caller supplies an invalid change-summary parameter."""


def build_workspace_change_summary(
    bundle: dict[str, Any],
    *,
    since: str | None = None,
    object_service: Any | None = None,
    now: datetime | str | None = None,
) -> dict[str, Any]:
    return OntologyChangeSummaryService(object_service=object_service, now=now).workspace_summary(bundle, since=since)


def build_dossier_change_summary(
    bundle: dict[str, Any],
    ticker: str,
    *,
    since: str | None = None,
    object_service: Any | None = None,
    now: datetime | str | None = None,
) -> dict[str, Any]:
    return OntologyChangeSummaryService(object_service=object_service, now=now).dossier_summary(
        bundle,
        ticker,
        since=since,
    )


class OntologyChangeSummaryService:
    def __init__(self, *, object_service: Any | None = None, now: datetime | str | None = None):
        self._object_service = object_service
        self.now = _parse_datetime(now) or datetime.now(UTC)

    def workspace_summary(self, bundle: dict[str, Any] | None, *, since: str | None = None) -> dict[str, Any]:
        baseline = self._workspace_baseline(bundle or {}, since=since)
        return self._summary(
            baseline=baseline,
            object_types=tuple(CHANGE_OBJECT_CONFIGS),
            ticker=None,
        )

    def dossier_summary(
        self,
        bundle: dict[str, Any] | None,
        ticker: str,
        *,
        since: str | None = None,
    ) -> dict[str, Any]:
        baseline = self._dossier_baseline(bundle or {}, ticker=ticker, since=since)
        return self._summary(
            baseline=baseline,
            object_types=tuple(CHANGE_OBJECT_CONFIGS),
            ticker=_ticker(ticker),
        )

    def _workspace_baseline(self, bundle: dict[str, Any], *, since: str | None) -> dict[str, Any]:
        override = _parse_override(since)
        if override:
            return override

        report = _latest_completed(bundle.get("recent_report_runs"), object_type="ReportRun")
        if report:
            return report
        workflow = _latest_completed(bundle.get("recent_workflow_runs"), object_type="WorkflowRun")
        if workflow:
            return workflow
        return self._fallback_baseline()

    def _dossier_baseline(self, bundle: dict[str, Any], *, ticker: str, since: str | None) -> dict[str, Any]:
        override = _parse_override(since)
        if override:
            return override

        workflow = _latest_completed(
            _filter_rows_by_ticker(bundle.get("workflow_runs"), ticker), object_type="WorkflowRun"
        )
        if workflow:
            return workflow
        return self._fallback_baseline()

    def _fallback_baseline(self) -> dict[str, Any]:
        return {
            "kind": "lookback",
            "source_type": None,
            "source_id": None,
            "at": _iso(self.now - timedelta(days=DEFAULT_LOOKBACK_DAYS)),
            "days": DEFAULT_LOOKBACK_DAYS,
        }

    def _summary(
        self,
        *,
        baseline: dict[str, Any],
        object_types: tuple[str, ...],
        ticker: str | None,
    ) -> dict[str, Any]:
        generated_at = _iso(self.now)
        baseline_at = _parse_datetime(baseline.get("at")) or self.now - timedelta(days=DEFAULT_LOOKBACK_DAYS)
        try:
            items = self._change_items(
                baseline_at=baseline_at,
                object_types=object_types,
                ticker=ticker,
            )
        except Exception:
            logger.debug("Failed to build ontology change summary", exc_info=True)
            items = []

        return {
            "baseline": baseline,
            "generated_at": generated_at,
            "items": items,
            "counts": _counts(items),
        }

    def _change_items(
        self,
        *,
        baseline_at: datetime,
        object_types: tuple[str, ...],
        ticker: str | None,
    ) -> list[dict[str, Any]]:
        rows_by_uid: dict[str, list[dict[str, Any]]] = {}
        object_service = self._objects()
        for object_type in object_types:
            filters = {"ticker": ticker} if ticker and _object_type_accepts_ticker(object_type) else None
            for row in object_service.query_objects(
                object_type,
                filters=filters,
                include_history=True,
                limit=MAX_CHANGE_ROWS_PER_TYPE,
            ):
                uid = _object_uid(row)
                if uid:
                    rows_by_uid.setdefault(uid, []).append(row)

        items: list[dict[str, Any]] = []
        for uid, rows in rows_by_uid.items():
            ordered = sorted(rows, key=_history_sort_key, reverse=True)
            if not ordered:
                continue
            current = ordered[0]
            object_type = str(current.get("object_type") or "")
            config = CHANGE_OBJECT_CONFIGS.get(object_type)
            if config is None:
                continue
            changed_at = _changed_at(current)
            if changed_at is None or changed_at <= baseline_at:
                continue
            previous = ordered[1] if len(ordered) > 1 else None
            item = _build_item(uid, object_type, config, current, previous, changed_at)
            if item is not None:
                items.append(item)

        return sorted(items, key=lambda item: (str(item["changed_at"]), str(item["object_uid"])), reverse=True)

    def _objects(self) -> Any:
        if self._object_service is None:
            from ontology.object_service import OntologyObjectService

            self._object_service = OntologyObjectService()
        return self._object_service


def _build_item(
    uid: str,
    object_type: str,
    config: ObjectChangeConfig,
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    changed_at: datetime,
) -> dict[str, Any] | None:
    current_fields = _project_fields(current, config.fields)
    previous_fields = _project_fields(previous, config.fields) if previous else {}
    if previous is None:
        change_kind = "created"
        before: dict[str, Any] = {}
        after = {key: value for key, value in current_fields.items() if value not in (None, "", [], {})}
    else:
        diff_fields = [
            key
            for key in config.fields
            if _normalize_value(current_fields.get(key)) != _normalize_value(previous_fields.get(key))
        ]
        if not diff_fields:
            return None
        change_kind = "updated"
        before = {key: previous_fields.get(key) for key in diff_fields}
        after = {key: current_fields.get(key) for key in diff_fields}

    title = _title(object_type, config, current_fields, uid)
    return {
        "object_type": object_type,
        "object_uid": uid,
        "ticker": _ticker(current_fields.get("ticker")),
        "category": config.category,
        "change_kind": change_kind,
        "severity": _severity(object_type, current_fields),
        "changed_at": _iso(changed_at),
        "title": title,
        "summary": _summary_text(object_type, change_kind, title, before, after),
        "before": before,
        "after": after,
    }


def _project_fields(row: dict[str, Any] | None, fields: tuple[str, ...]) -> dict[str, Any]:
    if not row:
        return {}
    props = _props(row)
    return {field: _json_scalar(props.get(field, row.get(field))) for field in fields}


def _summary_text(
    object_type: str,
    change_kind: str,
    title: str,
    before: dict[str, Any],
    after: dict[str, Any],
) -> str:
    label = _object_label(object_type)
    if change_kind == "created":
        return f"New {label}: {title}"
    pieces: list[str] = []
    for field, after_value in list(after.items())[:3]:
        before_value = before.get(field)
        pieces.append(f"{_field_label(field)} changed from {_display(before_value)} to {_display(after_value)}")
    return "; ".join(pieces) if pieces else f"{label} updated"


def _severity(object_type: str, fields: dict[str, Any]) -> str:
    status = str(fields.get("status") or "").strip().lower()
    application_status = str(fields.get("application_status") or "").strip().lower()
    recommendation_status = str(fields.get("recommendation_status") or "").strip().lower()
    critical_quality = str(fields.get("critical_data_quality") or "").strip().lower()
    urgency = str(fields.get("urgency") or "").strip().lower()
    action = str(fields.get("action") or "").strip().lower()
    risk_flag = str(fields.get("risk_flag") or "").strip()

    if object_type == "KillCondition" and status == "triggered":
        return "critical"
    if object_type == "Recommendation" and (
        recommendation_status == "blocked" or critical_quality in {"stale", "failed"}
    ):
        return "critical"
    if object_type == "Approval" and application_status == "failed":
        return "critical"
    if status in {"failed", "blocked", "triggered"}:
        return "critical"
    if urgency in {"urgent", "high"} or risk_flag or action in {"trim", "reduce", "exit", "sell"}:
        return "warning"
    if object_type == "Approval" and status == "pending":
        return "warning"
    return "info"


def _latest_completed(rows: Any, *, object_type: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    completed = [row for row in candidates if _completed(row, object_type=object_type)]
    if not completed:
        return None
    row = max(completed, key=lambda item: _completed_at(item) or datetime.min.replace(tzinfo=UTC))
    at = _completed_at(row)
    if at is None:
        return None
    return {
        "kind": "last_report_run" if object_type == "ReportRun" else "last_workflow_run",
        "source_type": object_type,
        "source_id": _object_uid(row) or _identity(row),
        "at": _iso(at),
    }


def _completed(row: dict[str, Any], *, object_type: str) -> bool:
    props = _props(row)
    status = str(props.get("status") or row.get("status") or "").strip().lower()
    if status in {"failed", "error", "running", "started", "queued", "canceled", "cancelled"}:
        return False
    if status in {"completed", "succeeded", "success", "done", "ok"}:
        return True
    if object_type == "ReportRun":
        return _completed_at(row) is not None
    return bool(props.get("completed_at") or row.get("completed_at"))


def _filter_rows_by_ticker(rows: Any, ticker: str) -> list[dict[str, Any]]:
    target = _ticker(ticker)
    candidates = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    if not target:
        return candidates
    return [row for row in candidates if _ticker(_props(row).get("ticker", row.get("ticker"))) == target]


def _completed_at(row: dict[str, Any]) -> datetime | None:
    props = _props(row)
    for key in ("completed_at", "synced_at", "as_of", "updated_at", "created_at", "started_at"):
        parsed = _parse_datetime(props.get(key, row.get(key)))
        if parsed:
            return parsed
    return _changed_at(row)


def _history_sort_key(row: dict[str, Any]) -> tuple[int, datetime]:
    return (1 if _is_current(row) else 0, _changed_at(row) or datetime.min.replace(tzinfo=UTC))


def _is_current(row: dict[str, Any]) -> bool:
    temporal = _temporal(row)
    return (
        not temporal.get("tx_to") and not temporal.get("valid_to") and not row.get("tx_to") and not row.get("valid_to")
    )


def _changed_at(row: dict[str, Any]) -> datetime | None:
    props = _props(row)
    for key in ("updated_at", "completed_at", "synced_at", "evaluated_at", "created_at", "started_at", "as_of"):
        parsed = _parse_datetime(props.get(key, row.get(key)))
        if parsed:
            return parsed
    temporal = _temporal(row)
    return _parse_datetime(
        temporal.get("tx_from") or temporal.get("valid_from") or row.get("tx_from") or row.get("valid_from")
    )


def _props(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("properties", "properties_json"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _temporal(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("_meta")
    if isinstance(meta, dict):
        temporal = meta.get("temporal")
        if isinstance(temporal, dict):
            return temporal
    return {}


def _object_uid(row: dict[str, Any]) -> str:
    props = _props(row)
    return str(row.get("object_uid") or props.get("object_uid") or props.get("id") or "").strip()


def _identity(row: dict[str, Any]) -> str | None:
    props = _props(row)
    for key in ("run_id", "id", "business_key"):
        value = str(props.get(key) or row.get(key) or "").strip()
        if value:
            return value
    return None


def _title(object_type: str, config: ObjectChangeConfig, fields: dict[str, Any], uid: str) -> str:
    if object_type == "Evaluation" and fields.get("ticker"):
        return f"{fields['ticker']} evaluation"
    if object_type == "Recommendation":
        pieces = [str(fields.get(key) or "").strip() for key in ("ticker", "report_type", "action")]
        text = " ".join(piece for piece in pieces if piece)
        if text:
            return text
    for field in config.title_fields:
        value = str(fields.get(field) or "").strip()
        if value:
            return _truncate(value, 96)
    return uid or _object_label(object_type)


def _object_label(object_type: str) -> str:
    label = []
    for idx, char in enumerate(object_type):
        if idx and char.isupper() and object_type[idx - 1].islower():
            label.append(" ")
        label.append(char.lower())
    return "".join(label)


def _field_label(field: str) -> str:
    return field.replace("_", " ")


def _display(value: Any) -> str:
    if value in (None, "", [], {}):
        return "empty"
    text = str(value)
    return _truncate(text, 72)


def _truncate(value: str, max_len: int) -> str:
    return value if len(value) <= max_len else f"{value[: max_len - 1]}..."


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_json_scalar(item) for item in value if _json_scalar(item) not in (None, "", [], {})][:5]
    if isinstance(value, dict):
        return {
            str(key): _json_scalar(item) for key, item in value.items() if _json_scalar(item) not in (None, "", [], {})
        }
    return str(value)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def _counts(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(items),
        "by_category": dict(Counter(str(item.get("category") or "unknown") for item in items)),
        "by_severity": dict(Counter(str(item.get("severity") or "info") for item in items)),
        "by_change_kind": dict(Counter(str(item.get("change_kind") or "updated") for item in items)),
    }


def _parse_override(value: str | None) -> dict[str, Any] | None:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = _parse_datetime(text)
    if parsed is None:
        raise ChangeSummaryInputError("since must be an ISO date or datetime")
    return {
        "kind": "override",
        "source_type": None,
        "source_id": None,
        "at": _iso(parsed),
    }


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(f"{text}T00:00:00+00:00")
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _ticker(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return text or None


def _object_type_accepts_ticker(object_type: str) -> bool:
    return object_type in CHANGE_OBJECT_CONFIGS
