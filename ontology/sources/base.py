from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal, Protocol

RawSourcePayload = Any
SourceStatus = Literal["ok", "partial", "error"]
SourceQuality = Literal["ok", "degraded", "missing", "schema_drift"]
SchemaDriftSeverity = Literal["info", "warning", "error"]

log = logging.getLogger(__name__)


@dataclass(slots=True)
class SchemaDriftIssue:
    severity: SchemaDriftSeverity
    path: str
    expected: str
    actual: str
    action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LineageMetadata:
    raw_module: str
    raw_function: str
    adapter: str
    adapter_version: str
    parameters: dict[str, Any] = field(default_factory=dict)
    cache_hint: str | None = None
    snapshot_hint: str | None = None
    payload_fingerprint: str | None = None
    provenance_event_id: str | None = None
    coverage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return {key: value for key, value in out.items() if value not in (None, {}, [])}


@dataclass(slots=True)
class SourceResult[T]:
    data: T | None
    status: SourceStatus
    quality: SourceQuality
    fetched_at: str
    as_of: str | None
    lineage: LineageMetadata
    schema_drift: list[SchemaDriftIssue] = field(default_factory=list)
    detail: str | None = None

    def to_status_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": self.status,
            "quality": self.quality,
            "source_name": self.lineage.adapter,
            "source_version": self.lineage.adapter_version,
            "fetched_at": self.fetched_at,
            "lineage": self.lineage.to_dict(),
        }
        if self.as_of:
            out["as_of"] = self.as_of
        if self.detail:
            out["detail"] = self.detail
        if self.schema_drift:
            out["schema_drift"] = [issue.to_dict() for issue in self.schema_drift]
        return out


class SourceAdapter[T](Protocol):
    source_name: str
    source_version: str
    required: bool
    raw_module: str
    raw_function: str
    parameters: dict[str, Any]

    def fetch(self) -> RawSourcePayload: ...

    def normalize(self, raw: RawSourcePayload) -> SourceResult[T]: ...


def run_source_adapter[T](
    adapter: SourceAdapter[T],
    *,
    provenance_parent_event_id: str | None = None,
    ontology_run_id: str | None = None,
) -> SourceResult[T]:
    started = time.perf_counter()
    provenance_event_id: str | None = None
    try:
        from api import provenance

        provenance_event_id = provenance.deterministic_id(
            "pv:adapter",
            ontology_run_id or "standalone",
            adapter.source_name,
        )
        provenance.start_event(
            event_id=provenance_event_id,
            event_type="source_adapter_run",
            event_name=adapter.source_name,
            parent_event_id=provenance_parent_event_id,
            ontology_run_id=ontology_run_id,
            summary={
                "source_name": adapter.source_name,
                "source_version": adapter.source_version,
                "raw_module": getattr(adapter, "raw_module", ""),
                "raw_function": getattr(adapter, "raw_function", ""),
            },
            metadata={
                "parameters": dict(getattr(adapter, "parameters", {}) or {}),
                "required": bool(getattr(adapter, "required", False)),
            },
        )
    except Exception:
        provenance_event_id = None
    try:
        raw = adapter.fetch()
        result = adapter.normalize(raw)
    except Exception as exc:
        result = error_result(adapter, _sanitize_detail(str(exc)))

    duration_ms = (time.perf_counter() - started) * 1000.0
    result.lineage.provenance_event_id = provenance_event_id
    if provenance_event_id:
        try:
            from api import provenance

            provenance.finish_event(
                provenance_event_id,
                status="succeeded" if result.status != "error" else "failed",
                output_value={
                    "status": result.status,
                    "quality": result.quality,
                    "payload_fingerprint": result.lineage.payload_fingerprint,
                },
                summary=result.to_status_dict(),
                metadata={
                    "duration_ms": round(duration_ms, 1),
                    "schema_drift_count": len(result.schema_drift),
                },
                error=result.detail if result.status == "error" else None,
            )
        except Exception:
            pass
    log.info(
        "ontology_source_adapter source=%s version=%s status=%s quality=%s duration_ms=%.1f as_of=%s drift_count=%d detail=%s",
        adapter.source_name,
        adapter.source_version,
        result.status,
        result.quality,
        duration_ms,
        result.as_of,
        len(result.schema_drift),
        result.detail,
    )
    return result


def build_source_result[T](
    adapter: SourceAdapter[T],
    raw: RawSourcePayload,
    data: T | None,
    *,
    status: SourceStatus,
    quality: SourceQuality,
    as_of: str | None,
    schema_drift: list[SchemaDriftIssue] | None = None,
    detail: str | None = None,
    coverage: dict[str, Any] | None = None,
    cache_hint: str | None = None,
    snapshot_hint: str | None = None,
    fingerprint_payload: Any | None = None,
) -> SourceResult[T]:
    fetched_at = now_iso()
    lineage = LineageMetadata(
        raw_module=adapter.raw_module,
        raw_function=adapter.raw_function,
        adapter=adapter.source_name,
        adapter_version=adapter.source_version,
        parameters=dict(getattr(adapter, "parameters", {}) or {}),
        cache_hint=cache_hint,
        snapshot_hint=snapshot_hint,
        payload_fingerprint=payload_fingerprint(fingerprint_payload if fingerprint_payload is not None else raw),
        coverage=dict(coverage or {}),
    )
    return SourceResult(
        data=data,
        status=status,
        quality=quality,
        fetched_at=fetched_at,
        as_of=as_of,
        lineage=lineage,
        schema_drift=list(schema_drift or []),
        detail=_sanitize_detail(detail),
    )


def error_result[T](adapter: SourceAdapter[T], detail: str) -> SourceResult[T]:
    fetched_at = now_iso()
    lineage = LineageMetadata(
        raw_module=getattr(adapter, "raw_module", ""),
        raw_function=getattr(adapter, "raw_function", ""),
        adapter=adapter.source_name,
        adapter_version=adapter.source_version,
        parameters=dict(getattr(adapter, "parameters", {}) or {}),
    )
    return SourceResult(
        data=None,
        status="error",
        quality="missing",
        fetched_at=fetched_at,
        as_of=None,
        lineage=lineage,
        detail=_sanitize_detail(detail),
    )


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def payload_fingerprint(payload: Any) -> str:
    normalized = _fingerprintable(payload)
    encoded = json.dumps(normalized, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def schema_issue(
    severity: SchemaDriftSeverity,
    path: str,
    expected: str,
    actual: Any,
    action: str,
) -> SchemaDriftIssue:
    return SchemaDriftIssue(
        severity=severity,
        path=path,
        expected=expected,
        actual=_actual_type(actual),
        action=action,
    )


def unknown_fields(raw: Mapping[str, Any], expected: set[str], *, path: str = "$") -> list[SchemaDriftIssue]:
    return [
        schema_issue("info", f"{path}.{key}", "known field", value, "ignored")
        for key, value in sorted(raw.items())
        if key not in expected
    ]


def status_for_drift(
    *,
    base_status: SourceStatus,
    base_quality: SourceQuality,
    drift: list[SchemaDriftIssue],
) -> tuple[SourceStatus, SourceQuality]:
    if any(issue.severity in {"warning", "error"} for issue in drift):
        return "partial", "schema_drift"
    return base_status, base_quality


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [{str(k): v for k, v in row.items()} for row in value if isinstance(row, dict)]

    if hasattr(value, "reset_index") and hasattr(value, "to_dict"):
        try:
            value = value.reset_index()
        except Exception:
            pass

    if hasattr(value, "to_dict"):
        try:
            records = value.to_dict(orient="records")
        except TypeError:
            records = None
        if isinstance(records, list):
            return [{str(k): v for k, v in row.items()} for row in records if isinstance(row, dict)]

    return []


def first_row(value: Any) -> dict[str, Any]:
    rows = as_rows(value)
    if rows:
        return rows[0]
    return value if isinstance(value, dict) else {}


def to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if out != out:
            return None
        return out
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def iso_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime | date):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def latest_series_value(series: Any) -> float | None:
    if isinstance(series, list) and series:
        last = series[-1]
        if isinstance(last, dict):
            return to_float(last.get("value"))
        return to_float(last)
    if hasattr(series, "iloc"):
        try:
            return to_float(series.iloc[-1])
        except Exception:
            return None
    return None


def series_point_count(series: Any) -> int:
    if isinstance(series, list):
        return len(series)
    try:
        return int(len(series))
    except Exception:
        return 0


def _sanitize_detail(detail: str | None) -> str | None:
    if detail is None:
        return None
    text = str(detail).strip().replace("\n", " ")
    return text[:500] if text else None


def _actual_type(value: Any) -> str:
    if value is None:
        return "missing"
    return type(value).__name__


def _fingerprintable(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return type(value).__name__
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if is_dataclass(value):
        return _fingerprintable(asdict(value), depth=depth + 1)
    if isinstance(value, Mapping):
        return {
            str(k): _fingerprintable(v, depth=depth + 1)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))[:80]
            if str(k) not in {"df_weekly", "composite_series", "raw_df"}
        }
    if isinstance(value, list | tuple):
        return [_fingerprintable(item, depth=depth + 1) for item in list(value)[:40]]
    if hasattr(value, "shape") and hasattr(value, "head") and hasattr(value, "to_dict"):
        try:
            head = value.head(3).to_dict()
        except Exception:
            head = repr(value)[:200]
        return {
            "type": type(value).__name__,
            "shape": tuple(getattr(value, "shape", ())),
            "head": _fingerprintable(head, depth=depth + 1),
        }
    return repr(value)[:200]
