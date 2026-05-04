"""Source record version normalization for temporal ontology ingestion."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from api.provenance import redacted_summary, stable_hash
from ontology.sources.base import SourceResult, payload_fingerprint
from ontology.temporal_repository import SourceRecordWrite, TemporalOntologyRepository


def write_source_record(
    *,
    vendor: str,
    source_name: str,
    source_version: str,
    dataset: str,
    record_kind: str,
    record_key: str,
    payload: Any,
    valid_from: datetime | str,
    valid_to: datetime | str | None = None,
    status: str = "ok",
    quality: str = "ok",
    as_of: datetime | str | None = None,
    load_time: datetime | str | None = None,
    artifact_uri: str | None = None,
    provenance_event_id: str | None = None,
    repository: TemporalOntologyRepository | None = None,
) -> dict[str, Any]:
    repo = repository or TemporalOntologyRepository()
    return repo.write_source_record_version(
        SourceRecordWrite(
            vendor=vendor,
            source_name=source_name,
            source_version=source_version,
            dataset=dataset,
            record_kind=record_kind,
            record_key=record_key,
            record_key_hash=stable_hash(record_key, length=32),
            payload_hash=payload_fingerprint(payload),
            payload=redacted_summary(payload),
            artifact_uri=artifact_uri,
            status=status,
            quality=quality,
            as_of=as_of,
            load_time=load_time,
            valid_from=valid_from,
            valid_to=valid_to,
            provenance_event_id=provenance_event_id,
        )
    )


def write_source_result_records(
    source_name: str,
    result: SourceResult[Any],
    *,
    dataset: str | None = None,
    vendor: str = "internal",
    repository: TemporalOntologyRepository | None = None,
) -> list[dict[str, Any]]:
    """Persist a normalized source adapter result as one or more source records."""
    repo = repository or TemporalOntologyRepository()
    data = getattr(result, "data", None)
    lineage = getattr(result, "lineage", None)
    source_version = str(getattr(lineage, "adapter_version", "") or "unknown")
    provenance_event_id = getattr(lineage, "provenance_event_id", None)
    as_of = result.as_of
    fetched_at = result.fetched_at
    valid_from: datetime | str = as_of or fetched_at
    status = str(getattr(result, "status", "ok") or "ok")
    quality = str(getattr(result, "quality", "ok") or "ok")
    dataset_name = dataset or source_name

    rows: list[dict[str, Any]] = []
    for record_kind, record_key, payload in _records_from_data(source_name, data):
        rows.append(
            repo.write_source_record_version(
                SourceRecordWrite(
                    vendor=vendor,
                    source_name=source_name,
                    source_version=source_version,
                    dataset=dataset_name,
                    record_kind=record_kind,
                    record_key=record_key,
                    record_key_hash=stable_hash(record_key, length=32),
                    payload_hash=payload_fingerprint(payload),
                    payload=redacted_summary(payload),
                    status=status,
                    quality=quality,
                    as_of=as_of,
                    load_time=fetched_at,
                    valid_from=valid_from,
                    provenance_event_id=provenance_event_id,
                )
            )
        )
    return rows


def _records_from_data(source_name: str, data: Any) -> list[tuple[str, str, Any]]:
    if data is None:
        return [("snapshot", source_name, {"source_name": source_name, "data": None})]
    positions = getattr(data, "positions", None)
    if isinstance(positions, Mapping):
        return [
            ("portfolio_position", str(ticker).upper(), _dataclass_or_mapping(position))
            for ticker, position in positions.items()
        ]
    rows = getattr(data, "rows", None)
    if isinstance(rows, list):
        out: list[tuple[str, str, Any]] = []
        for idx, row in enumerate(rows):
            payload = _dataclass_or_mapping(row)
            key = str(payload.get("ticker") or payload.get("sector") or payload.get("name") or idx)
            out.append((f"{source_name}_row", key, payload))
        return out
    return [("snapshot", source_name, _dataclass_or_mapping(data))]


def _dataclass_or_mapping(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return value
