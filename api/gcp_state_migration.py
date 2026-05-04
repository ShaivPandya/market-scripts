"""Offline state migration tooling for the GCP production stack.

Typical cutover flow:

1. Freeze writers.
2. Run ``python -m api.gcp_state_migration snapshot --output source.tar.zst``.
3. Upload the snapshot to ``gs://$GCS_STATE_BUCKET/backups/pre-migration/$MIGRATION_RUN_ID/source.tar.zst``.
4. Run the Cloud Run migration job with ``python -m api.gcp_state_migration migrate``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from api.postgres import require_database_url

SOURCE_DBS: dict[str, str] = {
    "portfolio": "portfolio/portfolio.db",
    "thesis": "portfolio/thesis.db",
    "core": "portfolio/core.db",
    "memory": "data_cache/memory/memory.db",
    "ontology": "data_cache/ontology/ontology.sqlite3",
    "retrieval": "data_cache/retrieval/embeddings.db",
    "central_banks": "macro/central_banks/centralbank_summaries.sqlite3",
    "industry": "macro/industry/industry_transcripts.sqlite3",
}

OBJECT_DIRS: tuple[str, ...] = (
    "investment_theses",
    "investment_overviews",
    "outputs",
    "auto_report/outputs",
    "data_cache/aluminum/processed",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _copy_sqlite_family(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = destination_dir / source.name
    shutil.copy2(source, copied)
    for suffix in ("-wal", "-shm"):
        sibling = source.with_name(source.name + suffix)
        if sibling.exists():
            shutil.copy2(sibling, copied.with_name(copied.name + suffix))
    return copied


def snapshot_sqlite_db(source: Path, destination: Path) -> Path:
    """Create a compact SQLite snapshot that includes uncheckpointed WAL rows."""
    with tempfile.TemporaryDirectory(prefix="sqlite-snapshot-") as tmp:
        tmp_dir = Path(tmp)
        copied = _copy_sqlite_family(source, tmp_dir)
        compact = tmp_dir / "snapshot.db"
        with sqlite3.connect(str(copied)) as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            escaped = str(compact).replace("'", "''")
            conn.execute(f"VACUUM INTO '{escaped}'")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(compact, destination)
    return destination


def _tar_zst(source_dir: Path, output: Path) -> None:
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise RuntimeError("zstandard is required to create .tar.zst snapshots.") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=10)
    with output.open("wb") as raw:
        with cctx.stream_writer(raw) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|") as tar:
                tar.add(source_dir, arcname=".")


def _extract_tar_zst(tarball: Path, destination: Path) -> None:
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise RuntimeError("zstandard is required to extract .tar.zst snapshots.") from exc

    dctx = zstd.ZstdDecompressor()
    with tarball.open("rb") as raw:
        with dctx.stream_reader(raw) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                tar.extractall(destination, filter="data")


def create_source_snapshot(project_root: Path, output: Path) -> dict[str, Any]:
    """Snapshot DBs and object source dirs into a zstd-compressed tarball."""
    manifest: dict[str, Any] = {"created_at": datetime.now(UTC).isoformat(), "dbs": {}, "objects": {}}
    with tempfile.TemporaryDirectory(prefix="market-state-snapshot-") as tmp:
        snapshot_root = Path(tmp) / "source"
        snapshot_root.mkdir()

        for source_name, rel in SOURCE_DBS.items():
            source = project_root / rel
            if not source.exists():
                continue
            destination = snapshot_root / rel
            snapshot_sqlite_db(source, destination)
            manifest["dbs"][source_name] = {"path": rel, "sha256": sha256_file(destination)}

        for rel in OBJECT_DIRS:
            src_dir = project_root / rel
            if not src_dir.exists():
                continue
            dst_dir = snapshot_root / rel
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            files = {}
            for file_path in sorted(p for p in dst_dir.rglob("*") if p.is_file()):
                files[str(file_path.relative_to(snapshot_root))] = sha256_file(file_path)
            manifest["objects"][rel] = files

        (snapshot_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        _tar_zst(snapshot_root, output)
    return manifest


def _gcs_client():
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise RuntimeError("google-cloud-storage is required for GCS migration operations.") from exc
    return storage.Client()


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {uri}")
    bucket, _, key = uri[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid gs:// URI: {uri}")
    return bucket, key


def download_gcs(uri: str, destination: Path) -> Path:
    bucket_name, key = _parse_gs_uri(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _gcs_client().bucket(bucket_name).blob(key).download_to_filename(str(destination))
    return destination


def _upload_gcs_file(bucket_name: str, key: str, path: Path, *, metadata: dict[str, str]) -> str:
    bucket = _gcs_client().bucket(bucket_name)
    blob = bucket.blob(key)
    existing = None
    if blob.exists():
        blob.reload()
        existing = (blob.metadata or {}).get("source_sha256")
    source_hash = metadata.get("source_sha256")
    if existing == source_hash:
        return f"gs://{bucket_name}/{key}"
    blob.metadata = metadata
    blob.upload_from_filename(str(path))
    return f"gs://{bucket_name}/{key}"


def _sqlite_rows(db_path: Path, table: str) -> list[dict[str, Any]]:
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table}").fetchall()]


def _sqlite_count(db_path: Path, table: str) -> int:
    with sqlite3.connect(str(db_path)) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _normalize_pending_approval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        status = row.get("status")
        application_status = row.get("application_status")
        if application_status is None:
            if status == "approved":
                application_status = "applied"
            elif status in {"rejected", "expired"}:
                application_status = "not_applicable"
            else:
                application_status = "pending"
            row["application_status"] = application_status
        if row.get("application_attempts") is None:
            row["application_attempts"] = 0
        if application_status in {"applied", "not_applicable"} and row.get("application_completed_at") is None:
            row["application_completed_at"] = row.get("resolved_at") or row.get("created_at")
    return rows


def _sqlite_table_exists(db_path: Path, table: str) -> bool:
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        return row is not None


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(UTC)
    cleaned = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return datetime.now(UTC)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _embedding_blob_to_vector(blob: bytes) -> list[float]:
    import struct

    if len(blob) != 384 * 4:
        raise ValueError(f"Expected 1536-byte all-MiniLM-L6-v2 embedding blob, got {len(blob)} bytes")
    return list(struct.unpack("384f", blob))


@dataclass(frozen=True)
class SourceResult:
    source_name: str
    source_sha256: str
    row_counts: dict[str, int]
    object_manifest: dict[str, Any] | None = None


class StateMigrator:
    def __init__(self, *, snapshot_root: Path, run_id: str, gcs_bucket: str | None = None):
        self.snapshot_root = snapshot_root
        self.run_id = run_id
        self.gcs_bucket = gcs_bucket
        self.conn: Any | None = None

    def __enter__(self) -> StateMigrator:
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg.rows import dict_row

        self.conn = psycopg.connect(require_database_url(), row_factory=dict_row)
        register_vector(self.conn)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.conn is not None:
            self.conn.close()

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        assert self.conn is not None
        return self.conn.execute(sql, params)

    def _commit(self) -> None:
        assert self.conn is not None
        self.conn.commit()

    def _rollback(self) -> None:
        assert self.conn is not None
        self.conn.rollback()

    def _source_completed(self, source_name: str, source_sha256: str) -> bool:
        row = self._execute(
            """
            SELECT 1 FROM migration_sources
            WHERE run_id = %s AND source_name = %s AND source_sha256 = %s AND status = 'completed'
            """,
            (self.run_id, source_name, source_sha256),
        ).fetchone()
        return row is not None

    def _record_run_started(self, manifest_hash: str) -> None:
        self._execute(
            """
            INSERT INTO migration_runs (run_id, source_manifest_sha256, started_at, status)
            VALUES (%s, %s, %s, 'running')
            ON CONFLICT (run_id) DO UPDATE SET
                source_manifest_sha256 = EXCLUDED.source_manifest_sha256,
                status = 'running'
            """,
            (self.run_id, manifest_hash, datetime.now(UTC)),
        )
        self._commit()

    def _record_source(self, result: SourceResult, status: str = "completed") -> None:
        from psycopg.types.json import Jsonb

        self._execute(
            """
            INSERT INTO migration_sources
                (run_id, source_name, source_sha256, row_counts_json, object_manifest_json, status, completed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id, source_name) DO UPDATE SET
                source_sha256 = EXCLUDED.source_sha256,
                row_counts_json = EXCLUDED.row_counts_json,
                object_manifest_json = EXCLUDED.object_manifest_json,
                status = EXCLUDED.status,
                completed_at = EXCLUDED.completed_at
            """,
            (
                self.run_id,
                result.source_name,
                result.source_sha256,
                Jsonb(result.row_counts),
                Jsonb(result.object_manifest or {}),
                status,
                datetime.now(UTC),
            ),
        )
        self._commit()

    def _finish_run(self, status: str) -> None:
        self._execute(
            "UPDATE migration_runs SET status = %s, completed_at = %s WHERE run_id = %s",
            (status, datetime.now(UTC), self.run_id),
        )
        self._commit()

    def _db_path(self, source_name: str) -> Path:
        return self.snapshot_root / SOURCE_DBS[source_name]

    def _upsert_rows(self, table: str, columns: list[str], conflict: list[str], rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        placeholders = ", ".join(["%s"] * len(columns))
        quoted_cols = ", ".join(columns)
        conflict_cols = ", ".join(conflict)
        updates = ", ".join(f"{col} = EXCLUDED.{col}" for col in columns if col not in conflict)
        action = f"DO UPDATE SET {updates}" if updates else "DO NOTHING"
        sql = f"INSERT INTO {table} ({quoted_cols}) VALUES ({placeholders}) ON CONFLICT ({conflict_cols}) {action}"
        assert self.conn is not None
        with self.conn.cursor() as cur:
            cur.executemany(sql, [tuple(row.get(col) for col in columns) for row in rows])

    def _reset_identity(self, table: str, column: str = "id") -> None:
        self._execute(
            f"""
            SELECT setval(
                pg_get_serial_sequence('{table}', '{column}'),
                GREATEST(COALESCE((SELECT MAX({column}) FROM {table}), 1), 1),
                true
            )
            """
        )

    def migrate_portfolio(self) -> SourceResult:
        db = self._db_path("portfolio")
        source_hash = sha256_file(db)
        if self._source_completed("portfolio", source_hash):
            return SourceResult("portfolio", source_hash, {"positions": _sqlite_count(db, "positions")})
        rows = _sqlite_rows(db, "positions")
        for row in rows:
            row["contrarian"] = bool(row.get("contrarian"))
        self._upsert_rows(
            "positions",
            ["ticker", "asset", "direction", "contrarian", "conviction", "cost_basis", "shares", "role"],
            ["ticker"],
            rows,
        )
        result = SourceResult("portfolio", source_hash, {"positions": len(rows)})
        self._record_source(result)
        return result

    def migrate_thesis(self) -> SourceResult:
        db = self._db_path("thesis")
        source_hash = sha256_file(db)
        counts = {
            "thesis_meta": _sqlite_count(db, "thesis_meta"),
            "thesis_status_history": _sqlite_count(db, "thesis_status_history"),
            "thesis_evaluations": _sqlite_count(db, "thesis_evaluations"),
        }
        if self._source_completed("thesis", source_hash):
            return SourceResult("thesis", source_hash, counts)
        self._upsert_rows(
            "thesis_meta", ["ticker", "status", "created_at", "updated_at"], ["ticker"], _sqlite_rows(db, "thesis_meta")
        )
        self._upsert_rows(
            "thesis_status_history",
            ["id", "ticker", "old_status", "new_status", "reason", "changed_at"],
            ["id"],
            _sqlite_rows(db, "thesis_status_history"),
        )
        self._upsert_rows(
            "thesis_evaluations",
            [
                "id",
                "ticker",
                "evaluated_at",
                "thesis_status",
                "technical_read",
                "fundamental_read",
                "action",
                "confidence",
                "key_developments",
                "earnings_note",
                "risk_flag",
            ],
            ["ticker", "evaluated_at"],
            _sqlite_rows(db, "thesis_evaluations"),
        )
        self._reset_identity("thesis_status_history")
        self._reset_identity("thesis_evaluations")
        result = SourceResult("thesis", source_hash, counts)
        self._record_source(result)
        return result

    def migrate_core(self) -> SourceResult:
        db = self._db_path("core")
        source_hash = sha256_file(db)
        tables = {
            "catalysts": [
                "id",
                "ticker",
                "description",
                "category",
                "status",
                "target_date",
                "evidence",
                "created_at",
                "updated_at",
                "created_by",
            ],
            "kill_conditions": [
                "id",
                "ticker",
                "condition",
                "metric",
                "threshold",
                "status",
                "triggered_at",
                "created_at",
                "updated_at",
                "created_by",
            ],
            "workflow_runs": [
                "run_id",
                "workflow_name",
                "ticker",
                "status",
                "started_at",
                "completed_at",
                "tool_sections",
                "synthesis",
                "artifacts",
                "provenance_event_id",
                "lineage_completeness",
                "error",
            ],
            "report_runs": [
                "report_id",
                "report_type",
                "as_of",
                "source",
                "source_run_id",
                "source_url",
                "status",
                "report_hash",
                "input_hash",
                "summary_json",
                "artifact_paths_json",
                "issue_url",
                "created_at",
                "updated_at",
                "synced_at",
                "error",
            ],
            "action_items": [
                "id",
                "ticker",
                "action_type",
                "description",
                "urgency",
                "status",
                "source_type",
                "source_id",
                "created_at",
                "completed_at",
                "resolution_note",
            ],
            "watch_triggers": [
                "id",
                "ticker",
                "trigger_type",
                "condition",
                "status",
                "source_type",
                "source_id",
                "created_at",
                "fired_at",
                "expires_at",
                "definition_json",
                "last_checked_at",
                "last_result_json",
                "last_evidence",
            ],
            "thesis_claims": [
                "id",
                "ticker",
                "claim",
                "expected_evidence",
                "disconfirming_evidence",
                "source_requirements_json",
                "cadence",
                "confidence",
                "status",
                "linked_catalyst_ids_json",
                "linked_kill_condition_ids_json",
                "source_type",
                "source_id",
                "created_at",
                "updated_at",
            ],
            "research_notes": [
                "id",
                "ticker",
                "title",
                "content",
                "note_type",
                "source_type",
                "source_id",
                "created_at",
            ],
            "pending_approvals": [
                "id",
                "entity_type",
                "entity_id",
                "ticker",
                "action_id",
                "action_schema_name",
                "action_schema_version",
                "action_input_hash",
                "request_schema_name",
                "request_schema_version",
                "proposed_change",
                "reason",
                "source_type",
                "source_id",
                "status",
                "created_at",
                "resolved_at",
                "resolved_note",
                "application_status",
                "application_attempts",
                "application_started_at",
                "application_completed_at",
                "application_error",
                "provenance_event_id",
                "origin_provenance_event_id",
                "origin_artifact_id",
                "lineage_completeness",
            ],
            "action_runs": [
                "id",
                "action_id",
                "action_schema_name",
                "action_schema_version",
                "request_schema_name",
                "request_schema_version",
                "actor_type",
                "actor_id",
                "source_type",
                "source_id",
                "approval_id",
                "parent_action_run_id",
                "input_hash",
                "input_json",
                "output_json",
                "status",
                "error",
                "started_at",
                "completed_at",
                "provenance_event_id",
                "lineage_completeness",
            ],
            "action_events": [
                "id",
                "action_run_id",
                "event_type",
                "message",
                "payload_json",
                "created_at",
            ],
            "audit_events": [
                "id",
                "event_id",
                "occurred_at",
                "received_at",
                "request_id",
                "actor_id",
                "actor_type",
                "parent_actor_id",
                "action_name",
                "action_category",
                "status",
                "object_type",
                "object_id",
                "object_refs_json",
                "before_summary_json",
                "after_summary_json",
                "source_lineage_json",
                "metadata_json",
                "error",
                "schema_version",
                "criticality",
                "lineage_root_id",
                "idempotency_key",
                "producer_name",
                "producer_version",
                "redaction_policy",
                "retention_class",
            ],
            "recommendations": [
                "id",
                "report_type",
                "as_of",
                "created_at",
                "source_report_path",
                "source_json_path",
                "stance",
                "recommendation_status",
                "critical_data_quality",
                "blocked_reasons_json",
                "what_changed_json",
                "do_nothing_rationale",
                "action",
                "ticker",
                "instrument",
                "horizon",
                "target_change",
                "rationale",
                "confidence",
                "source_quality",
                "status",
                "evidence_json",
                "disconfirming_evidence_json",
                "catalyst",
                "invalidation",
                "expected_onset_window",
                "alternatives_json",
                "opportunity_cost_json",
                "approval_id",
                "approval_status",
                "outcome_status",
                "outcome_json",
                "model",
                "prompt_hash",
                "input_hash",
                "validation_status",
                "source_quality_summary_json",
                "report_id",
                "idempotency_key",
                "provenance_event_id",
                "lineage_root_id",
                "lineage_completeness",
                "policy_gate_result_id",
                "policy_gate_status",
                "policy_gate_decision",
                "policy_gate_review_required",
                "policy_gate_failures_json",
                "policy_gate_warnings_json",
                "policy_gate_disclosures_json",
                "account_id",
                "portfolio_id",
                "policy_id",
                "trade_proposal_json",
                "risk_snapshot_id",
                "portfolio_risk_snapshot_id",
                "risk_quality",
                "risk_confidence",
                "risk_score",
                "risk_level",
                "risk_source_status_json",
                "risk_bindings_json",
            ],
            "recommendation_risk_bindings": [
                "id",
                "recommendation_id",
                "created_at",
                "ticker",
                "risk_snapshot_id",
                "portfolio_risk_snapshot_id",
                "risk_quality",
                "risk_confidence",
                "risk_score",
                "risk_level",
                "source_status_json",
                "binding_json",
            ],
            "policy_gate_results": [
                "id",
                "created_at",
                "decision",
                "review_required",
                "override_acknowledged",
                "account_id",
                "portfolio_id",
                "policy_id",
                "mandate_id",
                "action_id",
                "source_type",
                "source_id",
                "target_type",
                "target_id",
                "payload_hash",
                "provenance_event_id",
                "lineage_root_id",
                "lineage_completeness",
                "result_json",
            ],
            "provenance_events": [
                "id",
                "event_type",
                "event_name",
                "status",
                "started_at",
                "completed_at",
                "actor_type",
                "actor_id",
                "parent_actor_id",
                "request_id",
                "parent_event_id",
                "workflow_run_id",
                "ontology_run_id",
                "agent_session_id",
                "action_run_id",
                "approval_id",
                "audit_event_id",
                "input_hash",
                "output_hash",
                "summary_json",
                "metadata_json",
                "schema_version",
                "criticality",
                "lineage_root_id",
                "idempotency_key",
                "producer_name",
                "producer_version",
                "redaction_policy",
                "retention_class",
                "error",
            ],
            "provenance_links": [
                "id",
                "event_id",
                "source_ref_type",
                "source_ref_id",
                "source_ref_version",
                "target_ref_type",
                "target_ref_id",
                "target_ref_version",
                "link_type",
                "metadata_json",
                "lineage_root_id",
                "created_at",
            ],
            "governance_outbox": [
                "id",
                "idempotency_key",
                "event_bundle_json",
                "status",
                "attempt_count",
                "next_attempt_at",
                "locked_at",
                "last_error",
                "dead_lettered_at",
                "lineage_root_id",
                "retention_class",
                "created_at",
                "updated_at",
            ],
            "source_record_refs": [
                "record_ref_id",
                "adapter_run_event_id",
                "source_name",
                "record_kind",
                "record_key_hash",
                "record_hash",
                "as_of",
                "summary_json",
                "redaction_policy",
                "retention_class",
                "created_at",
            ],
            "workflow_artifact_records": [
                "artifact_id",
                "workflow_run_id",
                "artifact_key",
                "artifact_index",
                "artifact_hash",
                "summary_json",
                "approval_id",
                "provenance_event_id",
                "redaction_policy",
                "retention_class",
                "created_at",
            ],
        }
        tables = {table: columns for table, columns in tables.items() if _sqlite_table_exists(db, table)}
        counts = {table: _sqlite_count(db, table) for table in tables}
        if self._source_completed("core", source_hash):
            return SourceResult("core", source_hash, counts)
        for table, columns in tables.items():
            conflict = (
                ["run_id"]
                if table == "workflow_runs"
                else ["report_id"]
                if table == "report_runs"
                else ["event_id"]
                if table == "audit_events"
                else ["idempotency_key"]
                if table == "governance_outbox"
                else ["record_ref_id"]
                if table == "source_record_refs"
                else ["artifact_id"]
                if table == "workflow_artifact_records"
                else ["id"]
            )
            rows = _sqlite_rows(db, table)
            if table == "pending_approvals":
                rows = _normalize_pending_approval_rows(rows)
            self._upsert_rows(table, columns, conflict, rows)
        for table in [
            t
            for t in tables
            if t
            not in {
                "workflow_runs",
                "report_runs",
                "provenance_events",
                "provenance_links",
                "source_record_refs",
                "workflow_artifact_records",
            }
        ]:
            self._reset_identity(table)
        result = SourceResult("core", source_hash, counts)
        self._record_source(result)
        return result

    def migrate_memory(self) -> SourceResult:
        db = self._db_path("memory")
        source_hash = sha256_file(db)
        counts = {"conversation_sessions": _sqlite_count(db, "conversation_sessions")}
        if self._source_completed("memory", source_hash):
            return SourceResult("memory", source_hash, counts)
        self._upsert_rows(
            "conversation_sessions",
            [
                "session_id",
                "started_at",
                "ended_at",
                "message_count",
                "key_tickers",
                "key_topics",
                "summary",
                "transcript",
                "rolling_summary",
                "server_messages",
            ],
            ["session_id"],
            _sqlite_rows(db, "conversation_sessions"),
        )
        result = SourceResult("memory", source_hash, counts)
        self._record_source(result)
        return result

    def migrate_ontology(self) -> SourceResult:
        db = self._db_path("ontology")
        source_hash = sha256_file(db)
        table_map = {
            "nodes": (
                "ontology_nodes",
                ["id", "type", "label", "properties_json", "schema_name", "schema_version", "updated_at"],
                ["id"],
            ),
            "edges": (
                "ontology_edges",
                [
                    "source_id",
                    "target_id",
                    "relation_type",
                    "properties_json",
                    "schema_name",
                    "schema_version",
                    "relation_schema_name",
                    "relation_schema_version",
                    "updated_at",
                ],
                ["source_id", "target_id", "relation_type"],
            ),
            "ontology_runs": (
                "ontology_runs",
                [
                    "run_id",
                    "as_of",
                    "source_status_json",
                    "required_modules_json",
                    "optional_modules_json",
                    "component_scores_json",
                    "created_at",
                ],
                ["run_id"],
            ),
            "snapshot_nodes": (
                "ontology_snapshot_nodes",
                ["run_id", "id", "type", "label", "properties_json", "schema_name", "schema_version", "updated_at"],
                ["run_id", "id"],
            ),
            "snapshot_edges": (
                "ontology_snapshot_edges",
                [
                    "run_id",
                    "source_id",
                    "target_id",
                    "relation_type",
                    "properties_json",
                    "schema_name",
                    "schema_version",
                    "relation_schema_name",
                    "relation_schema_version",
                    "updated_at",
                ],
                ["run_id", "source_id", "target_id", "relation_type"],
            ),
            "schema_definitions": (
                "schema_definitions",
                [
                    "schema_kind",
                    "schema_name",
                    "schema_version",
                    "definition_json",
                    "definition_hash",
                    "compatibility_json",
                    "status",
                    "created_at",
                    "deprecated_at",
                ],
                ["schema_kind", "schema_name", "schema_version"],
            ),
            "ontology_run_schema_bindings": (
                "ontology_run_schema_bindings",
                ["run_id", "schema_kind", "schema_name", "schema_version", "definition_hash"],
                ["run_id", "schema_kind", "schema_name", "schema_version"],
            ),
        }
        table_map = {table: value for table, value in table_map.items() if _sqlite_table_exists(db, table)}
        counts = {table: _sqlite_count(db, table) for table in table_map}
        if self._source_completed("ontology", source_hash):
            return SourceResult("ontology", source_hash, counts)
        for source_table, (target_table, columns, conflict) in table_map.items():
            rows = _sqlite_rows(db, source_table)
            if source_table in {"nodes", "edges", "snapshot_nodes", "snapshot_edges"}:
                for row in rows:
                    row["schema_name"] = row.get("schema_name") or "legacy"
                    row["schema_version"] = int(row.get("schema_version") or 0)
                    if source_table in {"edges", "snapshot_edges"}:
                        row["relation_schema_name"] = row.get("relation_schema_name") or "legacy"
                        row["relation_schema_version"] = int(row.get("relation_schema_version") or 0)
            self._upsert_rows(target_table, columns, conflict, rows)
        if "schema_definitions" not in table_map:
            from ontology.schema_definitions import domain_action_schema_definitions, ontology_schema_definitions

            now = datetime.now(UTC).isoformat()
            definition_rows = []
            for definition in [*ontology_schema_definitions(), *domain_action_schema_definitions()]:
                definition_row = definition.row()
                definition_rows.append(
                    {
                        "schema_kind": definition_row[0],
                        "schema_name": definition_row[1],
                        "schema_version": definition_row[2],
                        "definition_json": definition_row[3],
                        "definition_hash": definition_row[4],
                        "compatibility_json": definition_row[5],
                        "status": definition_row[6],
                        "created_at": now,
                        "deprecated_at": definition_row[7],
                    }
                )
            self._upsert_rows(
                "schema_definitions",
                [
                    "schema_kind",
                    "schema_name",
                    "schema_version",
                    "definition_json",
                    "definition_hash",
                    "compatibility_json",
                    "status",
                    "created_at",
                    "deprecated_at",
                ],
                ["schema_kind", "schema_name", "schema_version"],
                definition_rows,
            )
        result = SourceResult("ontology", source_hash, counts)
        self._record_source(result)
        return result

    def migrate_retrieval(self) -> SourceResult:
        db = self._db_path("retrieval")
        source_hash = sha256_file(db)
        counts = {"documents": _sqlite_count(db, "documents"), "chunks": _sqlite_count(db, "chunks")}
        if self._source_completed("retrieval", source_hash):
            return SourceResult("retrieval", source_hash, counts)

        docs = _sqlite_rows(db, "documents")
        for row in docs:
            row["created_at"] = _parse_datetime(row.get("created_at"))
            row["updated_at"] = _parse_datetime(row.get("updated_at"))
        self._upsert_rows(
            "retrieval_documents",
            ["doc_id", "doc_type", "source_path", "ticker", "content", "created_at", "updated_at"],
            ["doc_id"],
            docs,
        )

        chunks = _sqlite_rows(db, "chunks")
        for row in chunks:
            row["embedding"] = _embedding_blob_to_vector(row["embedding"])
        self._upsert_rows(
            "retrieval_chunks",
            ["chunk_id", "doc_id", "chunk_index", "content", "heading", "embedding"],
            ["chunk_id"],
            chunks,
        )
        result = SourceResult("retrieval", source_hash, counts)
        self._record_source(result)
        return result

    def migrate_central_banks(self) -> SourceResult:
        db = self._db_path("central_banks")
        source_hash = sha256_file(db)
        counts = {"items": _sqlite_count(db, "items")}
        if self._source_completed("central_banks", source_hash):
            return SourceResult("central_banks", source_hash, counts)
        self._upsert_rows(
            "central_bank_items",
            [
                "guid",
                "source",
                "kind",
                "title",
                "url",
                "published_at",
                "content_sha256",
                "content_text",
                "summary_json",
                "content_url",
            ],
            ["guid"],
            _sqlite_rows(db, "items"),
        )
        result = SourceResult("central_banks", source_hash, counts)
        self._record_source(result)
        return result

    def migrate_industry(self) -> SourceResult:
        db = self._db_path("industry")
        source_hash = sha256_file(db)
        counts = {"transcripts": _sqlite_count(db, "transcripts")}
        if self._source_completed("industry", source_hash):
            return SourceResult("industry", source_hash, counts)
        rows = _sqlite_rows(db, "transcripts")
        for row in rows:
            row["is_stale"] = bool(row.get("is_stale"))
        self._upsert_rows(
            "industry_transcripts",
            [
                "id",
                "ticker",
                "company_name",
                "sector",
                "sector_type",
                "sub_sector",
                "quarter",
                "year",
                "transcript_text",
                "content_sha256",
                "summary_json",
                "fetched_at",
                "summarized_at",
                "transcript_date",
                "is_stale",
                "price_reaction_2d",
            ],
            ["id"],
            rows,
        )
        result = SourceResult("industry", source_hash, counts)
        self._record_source(result)
        return result

    def migrate_objects(self) -> SourceResult:
        files: list[tuple[Path, str]] = []

        def add_files(source_dir: str, key_fn: Callable[[Path], str]) -> None:
            root = self.snapshot_root / source_dir
            if not root.exists():
                return
            for path in sorted(p for p in root.rglob("*") if p.is_file()):
                files.append((path, key_fn(path.relative_to(root))))

        add_files("investment_theses", lambda rel: f"live/theses/{rel.as_posix()}")
        add_files("investment_overviews", lambda rel: f"live/overviews/{rel.as_posix()}")
        add_files("outputs", lambda rel: f"live/reports/weekly/{rel.as_posix()}")
        add_files("auto_report/outputs", lambda rel: f"live/reports/daily/{rel.as_posix()}")
        add_files("data_cache/aluminum/processed", lambda rel: f"live/snapshots/aluminum/{rel.as_posix()}")

        manifest = {key: sha256_file(path) for path, key in files}
        source_hash = sha256_json(manifest)
        if self._source_completed("objects", source_hash):
            return SourceResult("objects", source_hash, {"objects": len(files)}, manifest)
        if not self.gcs_bucket:
            raise RuntimeError("GCS_STATE_BUCKET is required to migrate object files.")
        uploaded = {}
        for path, key in files:
            source_hash_for_object = sha256_file(path)
            uploaded[key] = _upload_gcs_file(
                self.gcs_bucket,
                key,
                path,
                metadata={"source_sha256": source_hash_for_object, "migration_run_id": self.run_id},
            )
        result = SourceResult("objects", source_hash, {"objects": len(files)}, uploaded)
        self._record_source(result)
        return result

    def run(self) -> list[SourceResult]:
        manifest_path = self.snapshot_root / "manifest.json"
        manifest_hash = (
            sha256_file(manifest_path)
            if manifest_path.exists()
            else sha256_json({"snapshot_root": str(self.snapshot_root)})
        )
        self._record_run_started(manifest_hash)
        results: list[SourceResult] = []
        try:
            for method in [
                self.migrate_portfolio,
                self.migrate_thesis,
                self.migrate_core,
                self.migrate_memory,
                self.migrate_ontology,
                self.migrate_retrieval,
                self.migrate_central_banks,
                self.migrate_industry,
                self.migrate_objects,
            ]:
                results.append(method())
                self._commit()
            self._finish_run("completed")
            return results
        except Exception:
            self._rollback()
            self._finish_run("failed")
            raise


def _resolve_source_tarball(run_id: str, explicit_uri: str | None) -> str:
    if explicit_uri:
        return explicit_uri
    bucket = os.getenv("GCS_STATE_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError("GCS_STATE_BUCKET is required when --source-uri is not provided.")
    return f"gs://{bucket}/backups/pre-migration/{run_id}/source.tar.zst"


def _cmd_snapshot(args: argparse.Namespace) -> None:
    manifest = create_source_snapshot(Path(args.project_root).resolve(), Path(args.output).resolve())
    print(json.dumps({"output": str(Path(args.output).resolve()), "manifest": manifest}, indent=2, sort_keys=True))


def _cmd_migrate(args: argparse.Namespace) -> None:
    run_id = args.run_id or os.getenv("MIGRATION_RUN_ID") or datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    gcs_bucket = args.gcs_bucket or os.getenv("GCS_STATE_BUCKET") or None
    with tempfile.TemporaryDirectory(prefix="market-state-migrate-") as tmp:
        root = Path(tmp)
        if args.source_dir:
            snapshot_root = Path(args.source_dir).resolve()
        else:
            source_uri = _resolve_source_tarball(run_id, args.source_uri)
            tarball = download_gcs(source_uri, root / "source.tar.zst")
            snapshot_root = root / "source"
            snapshot_root.mkdir()
            _extract_tar_zst(tarball, snapshot_root)
        with StateMigrator(snapshot_root=snapshot_root, run_id=run_id, gcs_bucket=gcs_bucket) as migrator:
            results = migrator.run()
    print(
        json.dumps(
            {
                "run_id": run_id,
                "sources": [
                    {
                        "source_name": r.source_name,
                        "source_sha256": r.source_sha256,
                        "row_counts": r.row_counts,
                    }
                    for r in results
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GCP production state migration tools")
    sub = parser.add_subparsers(dest="command", required=True)

    snapshot = sub.add_parser("snapshot", help="Create a WAL-safe source snapshot tarball")
    snapshot.add_argument("--project-root", default=".", help="Repository root to snapshot")
    snapshot.add_argument("--output", required=True, help="Output .tar.zst path")
    snapshot.set_defaults(func=_cmd_snapshot)

    migrate = sub.add_parser("migrate", help="Migrate a source snapshot into Cloud SQL and GCS")
    migrate.add_argument("--run-id", default=None, help="Migration run id; defaults to MIGRATION_RUN_ID")
    migrate.add_argument("--source-uri", default=None, help="gs:// source tarball URI")
    migrate.add_argument("--source-dir", default=None, help="Already-extracted source snapshot directory")
    migrate.add_argument("--gcs-bucket", default=None, help="Destination GCS state bucket")
    migrate.set_defaults(func=_cmd_migrate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
