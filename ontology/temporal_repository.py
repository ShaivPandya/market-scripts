"""Postgres repository for authoritative bitemporal ontology state."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from api.postgres import connect

ConnectionFactory = Callable[[], Any]


@dataclass(frozen=True, slots=True)
class TemporalActor:
    actor_type: str | None = None
    actor_id: str | None = None


@dataclass(frozen=True, slots=True)
class ObjectVersionWrite:
    object_uid: str
    object_type: str
    business_key: str
    schema_name: str
    schema_version: int
    properties: dict[str, Any]
    valid_from: datetime | str
    valid_to: datetime | str | None = None
    source_record_id: uuid.UUID | str | None = None
    provenance_event_id: str | None = None
    action_run_id: int | None = None
    approval_id: str | int | None = None
    actor_type: str | None = None
    actor_id: str | None = None
    input_hash: str | None = None
    supersedes_version_id: uuid.UUID | str | None = None
    temporal_confidence: str = "native"
    tx_from: datetime | str | None = None


@dataclass(frozen=True, slots=True)
class RelationVersionWrite:
    relation_uid: str
    source_object_uid: str
    target_object_uid: str
    relation_type: str
    relation_schema_name: str
    relation_schema_version: int
    properties: dict[str, Any]
    valid_from: datetime | str
    valid_to: datetime | str | None = None
    source_record_id: uuid.UUID | str | None = None
    provenance_event_id: str | None = None
    action_run_id: int | None = None
    approval_id: str | int | None = None
    actor_type: str | None = None
    actor_id: str | None = None
    input_hash: str | None = None
    supersedes_version_id: uuid.UUID | str | None = None
    temporal_confidence: str = "native"
    tx_from: datetime | str | None = None


@dataclass(frozen=True, slots=True)
class SourceRecordWrite:
    vendor: str
    source_name: str
    source_version: str
    dataset: str
    record_kind: str
    record_key: str
    payload_hash: str
    valid_from: datetime | str
    record_key_hash: str | None = None
    payload: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    artifact_uri: str | None = None
    status: str = "ok"
    quality: str = "ok"
    as_of: datetime | str | None = None
    load_time: datetime | str | None = None
    valid_to: datetime | str | None = None
    provenance_event_id: str | None = None
    tx_from: datetime | str | None = None


@dataclass(frozen=True, slots=True)
class SnapshotVersionWrite:
    snapshot_key: str
    payload_hash: str
    valid_from: datetime | str
    payload: dict[str, Any] | None = None
    artifact_uri: str | None = None
    as_of: datetime | str | None = None
    load_time: datetime | str | None = None
    valid_to: datetime | str | None = None
    status: str = "ok"
    quality: str = "ok"
    error: str | None = None
    source_record_ids: list[uuid.UUID | str] | None = None
    provenance_event_id: str | None = None
    tx_from: datetime | str | None = None


class TemporalOntologyRepository:
    """Authoritative Postgres repository for ontology temporal versions."""

    def __init__(self, connection_factory: ConnectionFactory | None = None):
        self._connection_factory = connection_factory or connect

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        with self._connection_factory() as conn:
            yield conn

    def get_object(
        self,
        object_uid: str,
        *,
        as_of: datetime | str | None = None,
        tx_as_of: datetime | str | None = None,
    ) -> dict[str, Any] | None:
        where, params = _temporal_where("ontology_object_versions", as_of=as_of, tx_as_of=tx_as_of)
        sql = f"""
        SELECT *
        FROM ontology_object_versions
        WHERE object_uid = %s
          AND {where}
        ORDER BY tx_from DESC, valid_from DESC
        LIMIT 1
        """
        with self._connect() as conn:
            row = conn.execute(sql, (object_uid, *params)).fetchone()
        return _normalize_row(row)

    def query_objects(
        self,
        object_type: str | None = None,
        *,
        filters: Mapping[str, Any] | None = None,
        as_of: datetime | str | None = None,
        tx_as_of: datetime | str | None = None,
        include_history: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        where_parts: list[str] = []
        params: list[Any] = []
        if object_type:
            where_parts.append("object_type = %s")
            params.append(object_type)
        if not include_history:
            temporal_where, temporal_params = _temporal_where(
                "ontology_object_versions", as_of=as_of, tx_as_of=tx_as_of
            )
            where_parts.append(temporal_where)
            params.extend(temporal_params)
        _append_object_filters(where_parts, params, filters)
        where_sql = " AND ".join(where_parts) if where_parts else "TRUE"
        sql = f"""
        SELECT *
        FROM ontology_object_versions
        WHERE {where_sql}
        ORDER BY object_type, business_key, valid_from DESC, tx_from DESC
        LIMIT %s OFFSET %s
        """
        params.extend([max(1, min(int(limit), 500)), max(0, int(offset))])
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return _normalize_rows(rows)

    def write_object_version(self, write: ObjectVersionWrite) -> dict[str, Any]:
        version_id = uuid.uuid4()
        valid_from = _parse_ts(write.valid_from)
        valid_to = _parse_optional_ts(write.valid_to)
        tx_from = _parse_optional_ts(write.tx_from) or _now()
        payload = _jsonable(write.properties)
        existing = self._find_equivalent_current_object(
            write, valid_from=valid_from, valid_to=valid_to, payload=payload
        )
        if existing is not None:
            return existing

        with self._connect() as conn:
            self._close_overlapping_object_versions(
                conn,
                object_uid=write.object_uid,
                valid_from=valid_from,
                valid_to=valid_to,
                tx_to=tx_from,
            )
            row = conn.execute(
                """
                INSERT INTO ontology_object_versions (
                    version_id,
                    object_uid,
                    object_type,
                    business_key,
                    schema_name,
                    schema_version,
                    properties_json,
                    valid_from,
                    valid_to,
                    tx_from,
                    tx_to,
                    source_record_id,
                    provenance_event_id,
                    action_run_id,
                    approval_id,
                    actor_type,
                    actor_id,
                    input_hash,
                    supersedes_version_id,
                    temporal_confidence
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING *
                """,
                (
                    version_id,
                    write.object_uid,
                    write.object_type,
                    write.business_key,
                    write.schema_name,
                    int(write.schema_version),
                    _jsonb(payload),
                    valid_from,
                    valid_to,
                    tx_from,
                    _uuid_or_none(write.source_record_id),
                    write.provenance_event_id,
                    write.action_run_id,
                    write.approval_id,
                    write.actor_type,
                    write.actor_id,
                    write.input_hash,
                    _uuid_or_none(write.supersedes_version_id),
                    write.temporal_confidence or "native",
                ),
            ).fetchone()
            out = _normalize_required_row(row)
            _link_version_provenance_tx(
                conn,
                "ontology_object_version",
                out.get("version_id"),
                write.provenance_event_id,
            )
            conn.commit()
        return out

    def correct_object_version(
        self,
        version_id: uuid.UUID | str,
        *,
        properties: Mapping[str, Any],
        actor: TemporalActor | None = None,
        provenance_event_id: str | None = None,
        input_hash: str | None = None,
    ) -> dict[str, Any]:
        tx_from = _now()
        with self._connect() as conn:
            current = conn.execute(
                """
                SELECT *
                FROM ontology_object_versions
                WHERE version_id = %s
                  AND tx_to IS NULL
                """,
                (_uuid(version_id),),
            ).fetchone()
            if current is None:
                raise KeyError(f"Current object version not found: {version_id}")
            current_row = _normalize_required_row(current)
            conn.execute(
                "UPDATE ontology_object_versions SET tx_to = %s WHERE version_id = %s AND tx_to IS NULL",
                (tx_from, _uuid(version_id)),
            )
            replacement = ObjectVersionWrite(
                object_uid=str(current_row["object_uid"]),
                object_type=str(current_row["object_type"]),
                business_key=str(current_row["business_key"]),
                schema_name=str(current_row["schema_name"]),
                schema_version=int(current_row["schema_version"]),
                properties=dict(properties),
                valid_from=current_row["valid_from"],
                valid_to=current_row.get("valid_to"),
                source_record_id=current_row.get("source_record_id"),
                provenance_event_id=provenance_event_id or current_row.get("provenance_event_id"),
                action_run_id=current_row.get("action_run_id"),
                approval_id=current_row.get("approval_id"),
                actor_type=actor.actor_type if actor else current_row.get("actor_type"),
                actor_id=actor.actor_id if actor else current_row.get("actor_id"),
                input_hash=input_hash or current_row.get("input_hash"),
                supersedes_version_id=_uuid(version_id),
                temporal_confidence=str(current_row.get("temporal_confidence") or "native"),
            )
            row = self._insert_object_version_without_closing(conn, replacement, tx_from=tx_from).fetchone()
            out = _normalize_required_row(row)
            _link_version_provenance_tx(
                conn,
                "ontology_object_version",
                out.get("version_id"),
                replacement.provenance_event_id,
            )
            conn.commit()
        return out

    def expire_object_versions(
        self,
        object_uid: str,
        *,
        valid_from: datetime | str | None = None,
        valid_to: datetime | str | None = None,
        tx_to: datetime | str | None = None,
    ) -> int:
        """Close current transaction-time object versions for an object UID."""
        tx_to_ts = _parse_optional_ts(tx_to) or _now()
        valid_from_ts = _parse_optional_ts(valid_from) or datetime.min.replace(tzinfo=UTC)
        valid_to_ts = _parse_optional_ts(valid_to)
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE ontology_object_versions
                SET tx_to = %s
                WHERE object_uid = %s
                  AND tx_to IS NULL
                  AND valid_from < COALESCE(%s, 'infinity'::timestamptz)
                  AND COALESCE(valid_to, 'infinity'::timestamptz) > %s
                """,
                (tx_to_ts, object_uid, valid_to_ts, valid_from_ts),
            )
            conn.commit()
        return int(getattr(result, "rowcount", 0) or 0)

    def write_relation_version(self, write: RelationVersionWrite) -> dict[str, Any]:
        version_id = uuid.uuid4()
        valid_from = _parse_ts(write.valid_from)
        valid_to = _parse_optional_ts(write.valid_to)
        tx_from = _parse_optional_ts(write.tx_from) or _now()
        payload = _jsonable(write.properties or {})
        existing = self._find_equivalent_current_relation(
            write,
            valid_from=valid_from,
            valid_to=valid_to,
            payload=payload,
        )
        if existing is not None:
            return existing

        with self._connect() as conn:
            self._close_overlapping_relation_versions(
                conn,
                relation_uid=write.relation_uid,
                valid_from=valid_from,
                valid_to=valid_to,
                tx_to=tx_from,
            )
            row = conn.execute(
                """
                INSERT INTO ontology_relation_versions (
                    version_id,
                    relation_uid,
                    source_object_uid,
                    target_object_uid,
                    relation_type,
                    relation_schema_name,
                    relation_schema_version,
                    properties_json,
                    valid_from,
                    valid_to,
                    tx_from,
                    tx_to,
                    source_record_id,
                    provenance_event_id,
                    action_run_id,
                    approval_id,
                    actor_type,
                    actor_id,
                    input_hash,
                    supersedes_version_id,
                    temporal_confidence
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING *
                """,
                (
                    version_id,
                    write.relation_uid,
                    write.source_object_uid,
                    write.target_object_uid,
                    write.relation_type,
                    write.relation_schema_name,
                    int(write.relation_schema_version),
                    _jsonb(payload),
                    valid_from,
                    valid_to,
                    tx_from,
                    _uuid_or_none(write.source_record_id),
                    write.provenance_event_id,
                    write.action_run_id,
                    write.approval_id,
                    write.actor_type,
                    write.actor_id,
                    write.input_hash,
                    _uuid_or_none(write.supersedes_version_id),
                    write.temporal_confidence or "native",
                ),
            ).fetchone()
            out = _normalize_required_row(row)
            _link_version_provenance_tx(conn, "relation_version", out.get("version_id"), write.provenance_event_id)
            conn.commit()
        return out

    def expire_relation_versions(
        self,
        relation_uid: str,
        *,
        valid_from: datetime | str | None = None,
        valid_to: datetime | str | None = None,
        tx_to: datetime | str | None = None,
    ) -> int:
        """Close current transaction-time relation versions for a relation UID."""
        tx_to_ts = _parse_optional_ts(tx_to) or _now()
        valid_from_ts = _parse_optional_ts(valid_from) or datetime.min.replace(tzinfo=UTC)
        valid_to_ts = _parse_optional_ts(valid_to)
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE ontology_relation_versions
                SET tx_to = %s
                WHERE relation_uid = %s
                  AND tx_to IS NULL
                  AND valid_from < COALESCE(%s, 'infinity'::timestamptz)
                  AND COALESCE(valid_to, 'infinity'::timestamptz) > %s
                """,
                (tx_to_ts, relation_uid, valid_to_ts, valid_from_ts),
            )
            conn.commit()
        return int(getattr(result, "rowcount", 0) or 0)

    def query_relations(
        self,
        relation_type: str | None = None,
        *,
        source_object_uid: str | None = None,
        target_object_uid: str | None = None,
        as_of: datetime | str | None = None,
        tx_as_of: datetime | str | None = None,
        include_history: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        where_parts: list[str] = []
        params: list[Any] = []
        if relation_type:
            where_parts.append("relation_type = %s")
            params.append(relation_type)
        if source_object_uid:
            where_parts.append("source_object_uid = %s")
            params.append(source_object_uid)
        if target_object_uid:
            where_parts.append("target_object_uid = %s")
            params.append(target_object_uid)
        if not include_history:
            temporal_where, temporal_params = _temporal_where(
                "ontology_relation_versions", as_of=as_of, tx_as_of=tx_as_of
            )
            where_parts.append(temporal_where)
            params.extend(temporal_params)
        where_sql = " AND ".join(where_parts) if where_parts else "TRUE"
        sql = f"""
        SELECT *
        FROM ontology_relation_versions
        WHERE {where_sql}
        ORDER BY relation_type, source_object_uid, target_object_uid, valid_from DESC, tx_from DESC
        LIMIT %s OFFSET %s
        """
        params.extend([max(1, min(int(limit), 500)), max(0, int(offset))])
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return _normalize_rows(rows)

    def write_source_record_version(self, write: SourceRecordWrite) -> dict[str, Any]:
        valid_from = _parse_ts(write.valid_from)
        valid_to = _parse_optional_ts(write.valid_to)
        load_time = _parse_optional_ts(write.load_time) or _now()
        as_of = _parse_optional_ts(write.as_of)
        record_key_hash = write.record_key_hash or _hash_text(write.record_key, length=32)
        payload = _jsonable(write.payload) if write.payload is not None else None
        existing = self._find_equivalent_current_source_record(
            write,
            record_key_hash=record_key_hash,
            valid_from=valid_from,
            valid_to=valid_to,
            as_of=as_of,
        )
        if existing is not None:
            return existing

        source_record_id = uuid.uuid4()
        tx_from = _parse_optional_ts(write.tx_from) or _now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE source_record_versions
                SET tx_to = %s
                WHERE vendor = %s
                  AND source_name = %s
                  AND dataset = %s
                  AND record_kind = %s
                  AND record_key_hash = %s
                  AND tx_to IS NULL
                  AND valid_from < COALESCE(%s, 'infinity'::timestamptz)
                  AND COALESCE(valid_to, 'infinity'::timestamptz) > %s
                """,
                (
                    tx_from,
                    write.vendor,
                    write.source_name,
                    write.dataset,
                    write.record_kind,
                    record_key_hash,
                    valid_to,
                    valid_from,
                ),
            )
            row = conn.execute(
                """
                INSERT INTO source_record_versions (
                    source_record_id,
                    vendor,
                    source_name,
                    source_version,
                    dataset,
                    record_kind,
                    record_key,
                    record_key_hash,
                    payload_hash,
                    payload_json,
                    artifact_uri,
                    status,
                    quality,
                    as_of,
                    load_time,
                    valid_from,
                    valid_to,
                    tx_from,
                    tx_to,
                    provenance_event_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, %s)
                RETURNING *
                """,
                (
                    source_record_id,
                    write.vendor,
                    write.source_name,
                    write.source_version,
                    write.dataset,
                    write.record_kind,
                    write.record_key,
                    record_key_hash,
                    write.payload_hash,
                    _jsonb(payload) if payload is not None else None,
                    write.artifact_uri,
                    write.status,
                    write.quality,
                    as_of,
                    load_time,
                    valid_from,
                    valid_to,
                    tx_from,
                    write.provenance_event_id,
                ),
            ).fetchone()
            conn.commit()
        return _normalize_required_row(row)

    def query_source_records(
        self,
        *,
        vendor: str | None = None,
        source_name: str | None = None,
        record_kind: str | None = None,
        as_of: datetime | str | None = None,
        tx_as_of: datetime | str | None = None,
        include_history: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        where_parts: list[str] = []
        params: list[Any] = []
        if vendor:
            where_parts.append("vendor = %s")
            params.append(vendor)
        if source_name:
            where_parts.append("source_name = %s")
            params.append(source_name)
        if record_kind:
            where_parts.append("record_kind = %s")
            params.append(record_kind)
        if not include_history:
            temporal_where, temporal_params = _temporal_where("source_record_versions", as_of=as_of, tx_as_of=tx_as_of)
            where_parts.append(temporal_where)
            params.extend(temporal_params)
        where_sql = " AND ".join(where_parts) if where_parts else "TRUE"
        sql = f"""
        SELECT *
        FROM source_record_versions
        WHERE {where_sql}
        ORDER BY source_name, record_kind, record_key_hash, valid_from DESC, tx_from DESC
        LIMIT %s OFFSET %s
        """
        params.extend([max(1, min(int(limit), 500)), max(0, int(offset))])
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return _normalize_rows(rows)

    def write_computed_snapshot_version(self, write: SnapshotVersionWrite) -> dict[str, Any]:
        valid_from = _parse_ts(write.valid_from)
        valid_to = _parse_optional_ts(write.valid_to)
        as_of = _parse_optional_ts(write.as_of)
        load_time = _parse_optional_ts(write.load_time) or _now()
        payload = _jsonable(write.payload) if write.payload is not None else None
        existing = self._find_equivalent_current_snapshot(
            write,
            valid_from=valid_from,
            valid_to=valid_to,
            as_of=as_of,
            payload=payload,
        )
        if existing is not None:
            return existing

        snapshot_id = uuid.uuid4()
        tx_from = _parse_optional_ts(write.tx_from) or _now()
        source_record_ids = [_uuid(item) for item in write.source_record_ids or []]
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE computed_snapshot_versions
                SET tx_to = %s
                WHERE snapshot_key = %s
                  AND tx_to IS NULL
                  AND valid_from < COALESCE(%s, 'infinity'::timestamptz)
                  AND COALESCE(valid_to, 'infinity'::timestamptz) > %s
                """,
                (tx_from, write.snapshot_key, valid_to, valid_from),
            )
            row = conn.execute(
                """
                INSERT INTO computed_snapshot_versions (
                    snapshot_id,
                    snapshot_key,
                    payload_hash,
                    payload_json,
                    artifact_uri,
                    as_of,
                    load_time,
                    valid_from,
                    valid_to,
                    tx_from,
                    tx_to,
                    status,
                    quality,
                    error,
                    source_record_ids,
                    provenance_event_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    snapshot_id,
                    write.snapshot_key,
                    write.payload_hash,
                    _jsonb(payload) if payload is not None else None,
                    write.artifact_uri,
                    as_of,
                    load_time,
                    valid_from,
                    valid_to,
                    tx_from,
                    write.status,
                    write.quality,
                    write.error,
                    source_record_ids,
                    write.provenance_event_id,
                ),
            ).fetchone()
            conn.commit()
        return _normalize_required_row(row)

    def read_computed_snapshot_version(
        self,
        snapshot_key: str,
        *,
        as_of: datetime | str | None = None,
        tx_as_of: datetime | str | None = None,
    ) -> dict[str, Any] | None:
        where, params = _temporal_where("computed_snapshot_versions", as_of=as_of, tx_as_of=tx_as_of)
        sql = f"""
        SELECT *
        FROM computed_snapshot_versions
        WHERE snapshot_key = %s
          AND {where}
        ORDER BY load_time DESC, tx_from DESC
        LIMIT 1
        """
        with self._connect() as conn:
            row = conn.execute(sql, (snapshot_key, *params)).fetchone()
        return _normalize_row(row)

    def _find_equivalent_current_object(
        self,
        write: ObjectVersionWrite,
        *,
        valid_from: datetime,
        valid_to: datetime | None,
        payload: Any,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM ontology_object_versions
                WHERE object_uid = %s
                  AND tx_to IS NULL
                  AND valid_from = %s
                  AND valid_to IS NOT DISTINCT FROM %s
                ORDER BY tx_from DESC
                LIMIT 1
                """,
                (write.object_uid, valid_from, valid_to),
            ).fetchone()
        normalized = _normalize_row(row)
        if not normalized:
            return None
        candidate = {
            "object_type": normalized.get("object_type"),
            "business_key": normalized.get("business_key"),
            "schema_name": normalized.get("schema_name"),
            "schema_version": int(normalized.get("schema_version") or 0),
            "properties_json": normalized.get("properties_json"),
            "source_record_id": str(normalized["source_record_id"]) if normalized.get("source_record_id") else None,
            "provenance_event_id": normalized.get("provenance_event_id"),
            "action_run_id": normalized.get("action_run_id"),
            "approval_id": normalized.get("approval_id"),
            "input_hash": normalized.get("input_hash"),
            "temporal_confidence": normalized.get("temporal_confidence"),
        }
        wanted = {
            "object_type": write.object_type,
            "business_key": write.business_key,
            "schema_name": write.schema_name,
            "schema_version": int(write.schema_version),
            "properties_json": payload,
            "source_record_id": str(write.source_record_id) if write.source_record_id else None,
            "provenance_event_id": write.provenance_event_id,
            "action_run_id": write.action_run_id,
            "approval_id": write.approval_id,
            "input_hash": write.input_hash,
            "temporal_confidence": write.temporal_confidence or "native",
        }
        return normalized if _stable_hash(candidate) == _stable_hash(wanted) else None

    def _find_equivalent_current_relation(
        self,
        write: RelationVersionWrite,
        *,
        valid_from: datetime,
        valid_to: datetime | None,
        payload: Any,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM ontology_relation_versions
                WHERE relation_uid = %s
                  AND tx_to IS NULL
                  AND valid_from = %s
                  AND valid_to IS NOT DISTINCT FROM %s
                ORDER BY tx_from DESC
                LIMIT 1
                """,
                (write.relation_uid, valid_from, valid_to),
            ).fetchone()
        normalized = _normalize_row(row)
        if not normalized:
            return None
        candidate = {
            "source_object_uid": normalized.get("source_object_uid"),
            "target_object_uid": normalized.get("target_object_uid"),
            "relation_type": normalized.get("relation_type"),
            "relation_schema_name": normalized.get("relation_schema_name"),
            "relation_schema_version": int(normalized.get("relation_schema_version") or 0),
            "properties_json": normalized.get("properties_json"),
            "source_record_id": str(normalized["source_record_id"]) if normalized.get("source_record_id") else None,
            "provenance_event_id": normalized.get("provenance_event_id"),
            "action_run_id": normalized.get("action_run_id"),
            "approval_id": normalized.get("approval_id"),
            "input_hash": normalized.get("input_hash"),
            "temporal_confidence": normalized.get("temporal_confidence"),
        }
        wanted = {
            "source_object_uid": write.source_object_uid,
            "target_object_uid": write.target_object_uid,
            "relation_type": write.relation_type,
            "relation_schema_name": write.relation_schema_name,
            "relation_schema_version": int(write.relation_schema_version),
            "properties_json": payload,
            "source_record_id": str(write.source_record_id) if write.source_record_id else None,
            "provenance_event_id": write.provenance_event_id,
            "action_run_id": write.action_run_id,
            "approval_id": write.approval_id,
            "input_hash": write.input_hash,
            "temporal_confidence": write.temporal_confidence or "native",
        }
        return normalized if _stable_hash(candidate) == _stable_hash(wanted) else None

    def _find_equivalent_current_source_record(
        self,
        write: SourceRecordWrite,
        *,
        record_key_hash: str,
        valid_from: datetime,
        valid_to: datetime | None,
        as_of: datetime | None,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM source_record_versions
                WHERE vendor = %s
                  AND source_name = %s
                  AND source_version = %s
                  AND dataset = %s
                  AND record_kind = %s
                  AND record_key_hash = %s
                  AND payload_hash = %s
                  AND status = %s
                  AND quality = %s
                  AND as_of IS NOT DISTINCT FROM %s
                  AND valid_from = %s
                  AND valid_to IS NOT DISTINCT FROM %s
                  AND tx_to IS NULL
                ORDER BY tx_from DESC
                LIMIT 1
                """,
                (
                    write.vendor,
                    write.source_name,
                    write.source_version,
                    write.dataset,
                    write.record_kind,
                    record_key_hash,
                    write.payload_hash,
                    write.status,
                    write.quality,
                    as_of,
                    valid_from,
                    valid_to,
                ),
            ).fetchone()
        return _normalize_row(row)

    def _find_equivalent_current_snapshot(
        self,
        write: SnapshotVersionWrite,
        *,
        valid_from: datetime,
        valid_to: datetime | None,
        as_of: datetime | None,
        payload: Any,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM computed_snapshot_versions
                WHERE snapshot_key = %s
                  AND payload_hash = %s
                  AND status = %s
                  AND quality = %s
                  AND error IS NOT DISTINCT FROM %s
                  AND as_of IS NOT DISTINCT FROM %s
                  AND valid_from = %s
                  AND valid_to IS NOT DISTINCT FROM %s
                  AND tx_to IS NULL
                ORDER BY tx_from DESC
                LIMIT 1
                """,
                (
                    write.snapshot_key,
                    write.payload_hash,
                    write.status,
                    write.quality,
                    write.error,
                    as_of,
                    valid_from,
                    valid_to,
                ),
            ).fetchone()
        normalized = _normalize_row(row)
        if not normalized:
            return None
        if _stable_hash(normalized.get("payload_json")) != _stable_hash(payload):
            return None
        return normalized

    def _close_overlapping_object_versions(
        self,
        conn: Any,
        *,
        object_uid: str,
        valid_from: datetime,
        valid_to: datetime | None,
        tx_to: datetime,
    ) -> None:
        conn.execute(
            """
            UPDATE ontology_object_versions
            SET tx_to = %s
            WHERE object_uid = %s
              AND tx_to IS NULL
              AND valid_from < COALESCE(%s, 'infinity'::timestamptz)
              AND COALESCE(valid_to, 'infinity'::timestamptz) > %s
            """,
            (tx_to, object_uid, valid_to, valid_from),
        )

    def _close_overlapping_relation_versions(
        self,
        conn: Any,
        *,
        relation_uid: str,
        valid_from: datetime,
        valid_to: datetime | None,
        tx_to: datetime,
    ) -> None:
        conn.execute(
            """
            UPDATE ontology_relation_versions
            SET tx_to = %s
            WHERE relation_uid = %s
              AND tx_to IS NULL
              AND valid_from < COALESCE(%s, 'infinity'::timestamptz)
              AND COALESCE(valid_to, 'infinity'::timestamptz) > %s
            """,
            (tx_to, relation_uid, valid_to, valid_from),
        )

    def _insert_object_version_without_closing(
        self,
        conn: Any,
        write: ObjectVersionWrite,
        *,
        tx_from: datetime,
    ) -> Any:
        return conn.execute(
            """
            INSERT INTO ontology_object_versions (
                version_id,
                object_uid,
                object_type,
                business_key,
                schema_name,
                schema_version,
                properties_json,
                valid_from,
                valid_to,
                tx_from,
                tx_to,
                source_record_id,
                provenance_event_id,
                action_run_id,
                approval_id,
                actor_type,
                actor_id,
                input_hash,
                supersedes_version_id,
                temporal_confidence
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL,
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING *
            """,
            (
                uuid.uuid4(),
                write.object_uid,
                write.object_type,
                write.business_key,
                write.schema_name,
                int(write.schema_version),
                _jsonb(_jsonable(write.properties)),
                _parse_ts(write.valid_from),
                _parse_optional_ts(write.valid_to),
                tx_from,
                _uuid_or_none(write.source_record_id),
                write.provenance_event_id,
                write.action_run_id,
                write.approval_id,
                write.actor_type,
                write.actor_id,
                write.input_hash,
                _uuid_or_none(write.supersedes_version_id),
                write.temporal_confidence or "native",
            ),
        )


def _temporal_where(
    _table_name: str,
    *,
    as_of: datetime | str | None,
    tx_as_of: datetime | str | None,
) -> tuple[str, list[Any]]:
    valid_time = _parse_optional_ts(as_of) or _now()
    params: list[Any] = [valid_time, valid_time]
    parts = [
        "valid_from <= %s",
        "(valid_to IS NULL OR valid_to > %s)",
    ]
    tx_time = _parse_optional_ts(tx_as_of)
    if tx_time is None:
        parts.append("tx_to IS NULL")
    else:
        parts.extend(["tx_from <= %s", "(tx_to IS NULL OR tx_to > %s)"])
        params.extend([tx_time, tx_time])
    return " AND ".join(parts), params


def _append_object_filters(where_parts: list[str], params: list[Any], filters: Mapping[str, Any] | None) -> None:
    if not filters:
        return
    business_key = filters.get("business_key")
    if business_key is not None:
        where_parts.append("business_key = %s")
        params.append(str(business_key))
    object_uid = filters.get("object_uid")
    if object_uid is not None:
        where_parts.append("object_uid = %s")
        params.append(str(object_uid))
    property_filters = filters.get("properties")
    if isinstance(property_filters, Mapping) and property_filters:
        where_parts.append("properties_json @> %s")
        params.append(_jsonb(dict(property_filters)))
    for key, value in filters.items():
        if key in {"business_key", "object_uid", "properties"}:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            where_parts.append("properties_json @> %s")
            params.append(_jsonb({str(key): value}))


def _normalize_row(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    if isinstance(row, Mapping):
        out = dict(row)
    elif hasattr(row, "keys"):
        out = {key: row[key] for key in row.keys()}
    else:
        return dict(row)
    for key in ("properties_json", "payload_json"):
        if isinstance(out.get(key), str):
            try:
                out[key] = json.loads(out[key])
            except Exception:
                pass
    return out


def _normalize_required_row(row: Any) -> dict[str, Any]:
    normalized = _normalize_row(row)
    if normalized is None:
        raise RuntimeError("Expected database statement to return a row.")
    return normalized


def _normalize_rows(rows: Iterable[Any]) -> list[dict[str, Any]]:
    return [_normalize_required_row(row) for row in rows]


def _parse_ts(value: datetime | date | str) -> datetime:
    parsed = _parse_optional_ts(value)
    if parsed is None:
        raise ValueError("timestamp is required")
    return parsed


def _parse_optional_ts(value: datetime | date | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _now() -> datetime:
    return datetime.now(UTC)


def _uuid(value: uuid.UUID | str) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _uuid_or_none(value: uuid.UUID | str | None) -> uuid.UUID | None:
    return _uuid(value) if value else None


def _jsonb(value: Any) -> Any:
    try:
        from psycopg.types.json import Jsonb

        return Jsonb(value)
    except Exception:
        return value


def _jsonable(value: Any) -> Any:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return json.loads(raw)


def _stable_hash(value: Any, *, length: int = 32) -> str:
    raw = json.dumps(_jsonable(value), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _hash_text(value: str, *, length: int = 32) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:length]


def payload_hash(value: Any, *, length: int = 32) -> str:
    return _stable_hash(value, length=length)


def _link_version_provenance_tx(conn: Any, ref_type: str, ref_id: Any, event_id: str | None) -> None:
    if not event_id or not ref_id:
        return
    lineage_root_id = f"{ref_type}:{ref_id}"
    link_id = "pvlink:" + _stable_hash(
        {
            "event_id": event_id,
            "source_ref_type": "source_adapter_run",
            "source_ref_id": str(event_id),
            "target_ref_type": ref_type,
            "target_ref_id": str(ref_id),
            "link_type": "produced",
        },
        length=32,
    )
    now = _now().isoformat()
    conn.execute(
        """
        INSERT INTO provenance_links (
            id,
            event_id,
            source_ref_type,
            source_ref_id,
            source_ref_version,
            target_ref_type,
            target_ref_id,
            target_ref_version,
            link_type,
            metadata_json,
            lineage_root_id,
            created_at
        )
        VALUES (%s, %s, %s, %s, NULL, %s, %s, NULL, %s, NULL, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            event_id = EXCLUDED.event_id,
            source_ref_type = EXCLUDED.source_ref_type,
            source_ref_id = EXCLUDED.source_ref_id,
            target_ref_type = EXCLUDED.target_ref_type,
            target_ref_id = EXCLUDED.target_ref_id,
            link_type = EXCLUDED.link_type,
            lineage_root_id = EXCLUDED.lineage_root_id
        """,
        (
            link_id,
            event_id,
            "source_adapter_run",
            str(event_id),
            ref_type,
            str(ref_id),
            "produced",
            lineage_root_id,
            now,
        ),
    )


def _link_version_provenance(ref_type: str, ref_id: Any, event_id: str | None) -> None:
    if not event_id or not ref_id:
        return
    try:
        from api import provenance

        provenance.link_refs(
            event_id=event_id,
            source_ref_type=provenance.REF_SOURCE_ADAPTER_RUN,
            source_ref_id=str(event_id),
            target_ref_type=ref_type,
            target_ref_id=str(ref_id),
            link_type=provenance.LINK_PRODUCED,
            lineage_root_id=f"{ref_type}:{ref_id}",
            fail_closed=True,
        )
    except Exception:
        raise
