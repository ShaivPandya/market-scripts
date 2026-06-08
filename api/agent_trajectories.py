"""Training-grade agent trajectory contract and storage.

This module owns the durable boundary between operational agent turns and
future training datasets. Raw trajectory records are retained for audit/replay;
exports only return the deterministic sanitized view.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from api.agent_governance import classify_model_payload, redact_secrets
from api.postgres import connect, use_postgres_state
from api.provenance import redacted_summary, stable_hash

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SQLITE_PATH = _REPO_ROOT / "data_cache" / "agent_trajectories.sqlite3"

TRAJECTORY_SCHEMA_VERSION = 1
TRAJECTORY_REDACTION_POLICY = "agent_trajectory_training_v1"
TRAJECTORY_RETENTION_CLASS = "agent_trajectory_365d"
TRAINING_VIEW_RETENTION_CLASS = "agent_training_view_365d"

ConsentState = Literal["not_requested", "granted", "denied"]
FinalDisposition = Literal["succeeded", "failed", "blocked", "timeout", "cancelled", "handoff", "unknown"]

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS agent_trajectories (
    trajectory_id TEXT PRIMARY KEY,
    session_id TEXT,
    client_turn_id TEXT,
    captured_at TEXT NOT NULL,
    completed_at TEXT,
    schema_version INTEGER NOT NULL DEFAULT 1,
    final_disposition TEXT NOT NULL DEFAULT 'unknown',
    provider TEXT,
    model TEXT,
    prompt_version TEXT,
    code_version TEXT,
    sensitivity TEXT NOT NULL DEFAULT 'operational_private',
    consent_state TEXT NOT NULL DEFAULT 'not_requested',
    training_eligible INTEGER NOT NULL DEFAULT 0,
    exclusion_reasons_json TEXT NOT NULL DEFAULT '[]',
    dataset_split_group TEXT NOT NULL,
    redaction_policy TEXT NOT NULL DEFAULT 'agent_trajectory_training_v1',
    retention_class TEXT NOT NULL DEFAULT 'agent_trajectory_365d',
    redaction_manifest_json TEXT NOT NULL DEFAULT '{}',
    source_provenance_json TEXT NOT NULL DEFAULT '{}',
    provenance_json TEXT NOT NULL DEFAULT '{}',
    raw_payload_json TEXT NOT NULL,
    sanitized_payload_json TEXT NOT NULL,
    tombstoned_at TEXT,
    deletion_reason TEXT
)
"""

_CREATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_agent_trajectories_captured_at ON agent_trajectories(captured_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_agent_trajectories_session_turn ON agent_trajectories(session_id, client_turn_id)",
    "CREATE INDEX IF NOT EXISTS idx_agent_trajectories_training ON agent_trajectories(training_eligible, captured_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_agent_trajectories_split_group ON agent_trajectories(dataset_split_group)",
)

_HASH_ONLY_PAYLOAD_KEYS = {
    "args",
    "arguments",
    "raw",
    "raw_payload",
    "result",
    "results",
    "tool_result",
    "tool_results",
    "output",
    "outputs",
}


class TrajectoryStep(BaseModel):
    """One ordered event in an agent turn."""

    step_id: str
    index: int = Field(ge=0)
    kind: str = Field(min_length=1, max_length=80)
    name: str | None = Field(default=None, max_length=160)
    status: str | None = Field(default=None, max_length=80)
    started_at: str | None = None
    completed_at: str | None = None
    elapsed_ms: float | None = Field(default=None, ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentTrajectory(BaseModel):
    """Versioned persisted trajectory contract."""

    trajectory_id: str
    schema_version: int = TRAJECTORY_SCHEMA_VERSION
    session_id: str | None = None
    client_turn_id: str | None = None
    captured_at: str
    completed_at: str | None = None
    final_disposition: FinalDisposition = "unknown"
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    code_version: str | None = None
    route: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[TrajectoryStep] = Field(default_factory=list)
    gate_outcomes: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float | None = Field(default=None, ge=0)
    provenance: dict[str, Any] = Field(default_factory=dict)
    sensitivity: str = "operational_private"
    consent_state: ConsentState = "not_requested"
    training_eligible: bool = False
    exclusion_reasons: list[str] = Field(default_factory=list)
    redaction_policy: str = TRAJECTORY_REDACTION_POLICY
    retention_class: str = TRAJECTORY_RETENTION_CLASS
    dataset_split_group: str
    source_provenance: dict[str, Any] = Field(default_factory=dict)
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _supported_schema(cls, value: int) -> int:
        if value != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported trajectory schema version: {value}")
        return value

    @field_validator("steps")
    @classmethod
    def _steps_are_ordered(cls, value: list[TrajectoryStep]) -> list[TrajectoryStep]:
        indexes = [step.index for step in value]
        if indexes != sorted(indexes):
            raise ValueError("Trajectory steps must be ordered by index")
        ids = [step.step_id for step in value]
        if len(ids) != len(set(ids)):
            raise ValueError("Trajectory step IDs must be unique")
        return value


class TrajectoryExportError(ValueError):
    """Raised when a trajectory cannot be exported for training."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=str, sort_keys=True)


def _json_load(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str) and value.strip():
        return json.loads(value)
    return default


def _sqlite_connect() -> sqlite3.Connection:
    _SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_SQLITE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE)
    for statement in _CREATE_INDEXES:
        conn.execute(statement)
    conn.commit()
    return conn


def trajectory_id_for(*, session_id: str | None, client_turn_id: str | None, captured_at: str | None = None) -> str:
    if session_id and client_turn_id:
        return (
            f"agent_trajectory:{stable_hash({'session_id': session_id, 'client_turn_id': client_turn_id}, length=24)}"
        )
    return (
        f"agent_trajectory:{stable_hash({'captured_at': captured_at or _now(), 'uuid': str(uuid.uuid4())}, length=24)}"
    )


def dataset_split_group_for(
    *, session_id: str | None, client_turn_id: str | None, messages: list[dict[str, Any]]
) -> str:
    group_source = {"session_id": session_id, "client_turn_id": client_turn_id}
    if not session_id:
        group_source["messages_hash"] = stable_hash(messages, length=24)
    return f"agent_turn:{stable_hash(group_source, length=16)}"


def _default_exclusion_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("consent_state", "not_requested") != "granted":
        reasons.append("missing_training_consent")
    if not payload.get("messages"):
        reasons.append("missing_messages")
    if not payload.get("steps"):
        reasons.append("missing_steps")
    if payload.get("final_disposition") not in {"succeeded", "blocked"}:
        reasons.append("non_success_terminal_state")
    return reasons


def build_trajectory(payload: dict[str, Any]) -> AgentTrajectory:
    """Build and validate a trajectory record from operational turn metadata."""

    captured_at = str(payload.get("captured_at") or _now())
    messages = list(payload.get("messages") or [])
    session_id = payload.get("session_id")
    client_turn_id = payload.get("client_turn_id")
    trajectory_id = str(
        payload.get("trajectory_id")
        or trajectory_id_for(session_id=session_id, client_turn_id=client_turn_id, captured_at=captured_at)
    )
    dataset_split_group = str(
        payload.get("dataset_split_group")
        or dataset_split_group_for(session_id=session_id, client_turn_id=client_turn_id, messages=messages)
    )
    raw_payload = dict(payload)
    raw_payload.setdefault("trajectory_id", trajectory_id)
    raw_payload.setdefault("captured_at", captured_at)

    sensitivity = str(payload.get("sensitivity") or classify_model_payload(raw_payload))
    exclusion_reasons = list(
        payload.get("exclusion_reasons") or _default_exclusion_reasons({**payload, "messages": messages})
    )
    training_eligible = bool(payload.get("training_eligible")) and not exclusion_reasons

    return AgentTrajectory(
        trajectory_id=trajectory_id,
        schema_version=int(payload.get("schema_version") or TRAJECTORY_SCHEMA_VERSION),
        session_id=str(session_id) if session_id else None,
        client_turn_id=str(client_turn_id) if client_turn_id else None,
        captured_at=captured_at,
        completed_at=payload.get("completed_at"),
        final_disposition=payload.get("final_disposition") or "unknown",
        provider=payload.get("provider"),
        model=payload.get("model"),
        prompt_version=payload.get("prompt_version"),
        code_version=payload.get("code_version"),
        route=dict(payload.get("route") or {}),
        messages=messages,
        steps=list(payload.get("steps") or []),
        gate_outcomes=list(payload.get("gate_outcomes") or []),
        usage=dict(payload.get("usage") or {}),
        latency_ms=payload.get("latency_ms"),
        provenance=dict(payload.get("provenance") or {}),
        sensitivity=sensitivity,
        consent_state=payload.get("consent_state") or "not_requested",
        training_eligible=training_eligible,
        exclusion_reasons=exclusion_reasons,
        redaction_policy=payload.get("redaction_policy") or TRAJECTORY_REDACTION_POLICY,
        retention_class=payload.get("retention_class") or TRAJECTORY_RETENTION_CLASS,
        dataset_split_group=dataset_split_group,
        source_provenance=dict(payload.get("source_provenance") or {"source": "agent_chat"}),
        raw_payload=raw_payload,
    )


def _hash_only(value: Any) -> dict[str, Any]:
    if value is None:
        return {"redacted": True, "type": "none"}
    if isinstance(value, str):
        return {"redacted": True, "type": "text", "length": len(value), "sha256": stable_hash(value)}
    if isinstance(value, list):
        return {"redacted": True, "type": "list", "count": len(value), "sha256": stable_hash(value)}
    if isinstance(value, dict):
        return {
            "redacted": True,
            "type": "dict",
            "field_names": sorted(str(key) for key in value.keys())[:24],
            "sha256": stable_hash(value),
        }
    return {"redacted": True, "type": type(value).__name__, "sha256": stable_hash(value)}


def _sanitize_step_payload(payload: dict[str, Any], *, hash_paths: list[str], prefix: str) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in payload.items():
        key_str = str(key)
        if key_str.lower() in _HASH_ONLY_PAYLOAD_KEYS:
            sanitized[key_str] = _hash_only(value)
            hash_paths.append(f"{prefix}.{key_str}")
        else:
            sanitized[key_str] = value
    return sanitized


def sanitize_trajectory(record: AgentTrajectory) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the training-safe payload and redaction manifest."""

    hash_paths: list[str] = []
    steps: list[dict[str, Any]] = []
    for step in record.steps:
        step_payload = step.model_dump(mode="json")
        step_payload["payload"] = _sanitize_step_payload(
            dict(step_payload.get("payload") or {}),
            hash_paths=hash_paths,
            prefix=f"steps[{step.index}].payload",
        )
        steps.append(step_payload)

    sanitized: dict[str, Any] = {
        "trajectory_id": record.trajectory_id,
        "schema_version": record.schema_version,
        "session_id": record.session_id,
        "client_turn_id": record.client_turn_id,
        "captured_at": record.captured_at,
        "completed_at": record.completed_at,
        "final_disposition": record.final_disposition,
        "provider": record.provider,
        "model": record.model,
        "prompt_version": record.prompt_version,
        "code_version": record.code_version,
        "route": record.route,
        "messages": record.messages,
        "steps": steps,
        "gate_outcomes": record.gate_outcomes,
        "usage": record.usage,
        "latency_ms": record.latency_ms,
        "provenance": record.provenance,
        "sensitivity": record.sensitivity,
        "consent_state": record.consent_state,
        "training_eligible": record.training_eligible,
        "exclusion_reasons": record.exclusion_reasons,
        "redaction_policy": record.redaction_policy,
        "retention_class": TRAINING_VIEW_RETENTION_CLASS,
        "dataset_split_group": record.dataset_split_group,
        "source_provenance": record.source_provenance,
        "raw_payload_ref": _hash_only(record.raw_payload),
    }
    redacted, findings = redact_secrets(sanitized)
    manifest = {
        "policy": record.redaction_policy,
        "findings": findings,
        "hash_only_paths": hash_paths + ["raw_payload_ref"],
        "generated_at": _now(),
    }
    return redacted, manifest


def _row_to_dict(row: Any) -> dict[str, Any]:
    data = dict(row)
    return {
        "trajectory_id": data.get("trajectory_id"),
        "session_id": data.get("session_id"),
        "client_turn_id": data.get("client_turn_id"),
        "captured_at": str(data.get("captured_at")),
        "completed_at": str(data.get("completed_at")) if data.get("completed_at") else None,
        "schema_version": int(data.get("schema_version") or 0),
        "final_disposition": data.get("final_disposition"),
        "provider": data.get("provider"),
        "model": data.get("model"),
        "prompt_version": data.get("prompt_version"),
        "code_version": data.get("code_version"),
        "sensitivity": data.get("sensitivity"),
        "consent_state": data.get("consent_state"),
        "training_eligible": bool(data.get("training_eligible")),
        "exclusion_reasons": _json_load(data.get("exclusion_reasons_json"), []),
        "dataset_split_group": data.get("dataset_split_group"),
        "redaction_policy": data.get("redaction_policy"),
        "retention_class": data.get("retention_class"),
        "redaction_manifest": _json_load(data.get("redaction_manifest_json"), {}),
        "source_provenance": _json_load(data.get("source_provenance_json"), {}),
        "provenance": _json_load(data.get("provenance_json"), {}),
        "raw_payload": _json_load(data.get("raw_payload_json"), {}),
        "sanitized_payload": _json_load(data.get("sanitized_payload_json"), {}),
        "tombstoned_at": str(data.get("tombstoned_at")) if data.get("tombstoned_at") else None,
        "deletion_reason": data.get("deletion_reason"),
    }


def insert_trajectory(payload: dict[str, Any]) -> str | None:
    """Persist a trajectory. Returns the trajectory ID or None on duplicate/failure."""

    try:
        record = build_trajectory(payload)
        sanitized, manifest = sanitize_trajectory(record)
        if use_postgres_state():
            return _insert_postgres(record, sanitized=sanitized, manifest=manifest)
        return _insert_sqlite(record, sanitized=sanitized, manifest=manifest)
    except (ValidationError, ValueError):
        logger.exception("agent_trajectory_validation_failed")
        return None
    except Exception:
        logger.exception("agent_trajectory_insert_failed")
        return None


def _insert_sqlite(record: AgentTrajectory, *, sanitized: dict[str, Any], manifest: dict[str, Any]) -> str | None:
    with _sqlite_connect() as conn:
        try:
            conn.execute(
                """
                INSERT INTO agent_trajectories (
                    trajectory_id, session_id, client_turn_id, captured_at, completed_at,
                    schema_version, final_disposition, provider, model, prompt_version, code_version,
                    sensitivity, consent_state, training_eligible, exclusion_reasons_json,
                    dataset_split_group, redaction_policy, retention_class, redaction_manifest_json,
                    source_provenance_json, provenance_json, raw_payload_json, sanitized_payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _insert_values(record, sanitized=sanitized, manifest=manifest, bool_as_int=True),
            )
            conn.commit()
            return record.trajectory_id
        except sqlite3.IntegrityError:
            return None


def _insert_postgres(record: AgentTrajectory, *, sanitized: dict[str, Any], manifest: dict[str, Any]) -> str | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO agent_trajectories (
                trajectory_id, session_id, client_turn_id, captured_at, completed_at,
                schema_version, final_disposition, provider, model, prompt_version, code_version,
                sensitivity, consent_state, training_eligible, exclusion_reasons_json,
                dataset_split_group, redaction_policy, retention_class, redaction_manifest_json,
                source_provenance_json, provenance_json, raw_payload_json, sanitized_payload_json
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s::jsonb,
                %s, %s, %s, %s::jsonb,
                %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb
            )
            ON CONFLICT (trajectory_id) DO NOTHING
            """,
            _insert_values(record, sanitized=sanitized, manifest=manifest, bool_as_int=False),
        )
        conn.commit()
        return record.trajectory_id if cur.rowcount else None


def _insert_values(
    record: AgentTrajectory,
    *,
    sanitized: dict[str, Any],
    manifest: dict[str, Any],
    bool_as_int: bool,
) -> tuple[Any, ...]:
    return (
        record.trajectory_id,
        record.session_id,
        record.client_turn_id,
        record.captured_at,
        record.completed_at,
        record.schema_version,
        record.final_disposition,
        record.provider,
        record.model,
        record.prompt_version,
        record.code_version,
        record.sensitivity,
        record.consent_state,
        int(record.training_eligible) if bool_as_int else record.training_eligible,
        _json_dump(record.exclusion_reasons),
        record.dataset_split_group,
        record.redaction_policy,
        record.retention_class,
        _json_dump(manifest),
        _json_dump(record.source_provenance),
        _json_dump(record.provenance),
        _json_dump(record.raw_payload),
        _json_dump(sanitized),
    )


def get_trajectory(trajectory_id: str) -> dict[str, Any] | None:
    if use_postgres_state():
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM agent_trajectories WHERE trajectory_id = %s", (trajectory_id,))
            row = cur.fetchone()
            return _row_to_dict(row) if row else None
    with _sqlite_connect() as conn:
        row = conn.execute("SELECT * FROM agent_trajectories WHERE trajectory_id = ?", (trajectory_id,)).fetchone()
        return _row_to_dict(row) if row else None


def get_trajectory_by_session_turn(*, session_id: str, client_turn_id: str) -> dict[str, Any] | None:
    trajectory_id = trajectory_id_for(session_id=session_id, client_turn_id=client_turn_id)
    return get_trajectory(trajectory_id)


def list_trajectories(*, limit: int = 100, training_eligible_only: bool = False) -> list[dict[str, Any]]:
    clauses = ["tombstoned_at IS NULL"]
    if training_eligible_only:
        clauses.append("training_eligible = TRUE" if use_postgres_state() else "training_eligible = 1")
    if use_postgres_state():
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT * FROM agent_trajectories
                WHERE {" AND ".join(clauses)}
                ORDER BY captured_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return [_row_to_dict(row) for row in cur.fetchall()]
    with _sqlite_connect() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM agent_trajectories
            WHERE {" AND ".join(clauses)}
            ORDER BY captured_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]


def grant_trajectory_preference_export_consent(
    trajectory_id: str,
    *,
    reviewer_actor_id: str,
    consent_reason: str = "human_feedback_preference",
) -> dict[str, Any] | None:
    """Grant preference-dataset export consent without SFT promotion."""

    row = get_trajectory(trajectory_id)
    if not row or row.get("tombstoned_at"):
        return None

    raw_payload = dict(row.get("raw_payload") or {})
    raw_payload["consent_state"] = "granted"
    source_provenance = dict(row.get("source_provenance") or {})
    source_provenance.update(
        {
            "preference_export_consent": True,
            "preference_export_by": reviewer_actor_id,
            "preference_export_at": _now(),
            "preference_export_reason": consent_reason,
        }
    )
    raw_payload["source_provenance"] = source_provenance

    record = build_trajectory({**raw_payload, "trajectory_id": trajectory_id})
    sanitized, manifest = sanitize_trajectory(record)
    if use_postgres_state():
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE agent_trajectories
                SET consent_state = %s,
                    source_provenance_json = %s::jsonb,
                    sanitized_payload_json = %s::jsonb,
                    redaction_manifest_json = %s::jsonb
                WHERE trajectory_id = %s AND tombstoned_at IS NULL
                """,
                (
                    record.consent_state,
                    _json_dump(record.source_provenance),
                    _json_dump(sanitized),
                    _json_dump(manifest),
                    trajectory_id,
                ),
            )
            conn.commit()
            if not cur.rowcount:
                return None
    else:
        with _sqlite_connect() as conn:
            cur = conn.execute(
                """
                UPDATE agent_trajectories
                SET consent_state = ?,
                    source_provenance_json = ?,
                    sanitized_payload_json = ?,
                    redaction_manifest_json = ?
                WHERE trajectory_id = ? AND tombstoned_at IS NULL
                """,
                (
                    record.consent_state,
                    _json_dump(record.source_provenance),
                    _json_dump(sanitized),
                    _json_dump(manifest),
                    trajectory_id,
                ),
            )
            conn.commit()
            if not cur.rowcount:
                return None
    return get_trajectory(trajectory_id)


def promote_trajectory_for_training(
    trajectory_id: str,
    *,
    reviewer_actor_id: str,
    promotion_reason: str = "human_review_approved",
) -> dict[str, Any] | None:
    """Promote one reviewed trajectory to exportable training eligibility."""

    row = get_trajectory(trajectory_id)
    if not row or row.get("tombstoned_at"):
        return None

    raw_payload = dict(row.get("raw_payload") or {})
    raw_payload["consent_state"] = "granted"
    raw_payload["training_eligible"] = True
    raw_payload["exclusion_reasons"] = [
        reason
        for reason in list(raw_payload.get("exclusion_reasons") or row.get("exclusion_reasons") or [])
        if reason != "missing_training_consent"
    ]
    source_provenance = dict(row.get("source_provenance") or {})
    source_provenance.update(
        {
            "promoted_by": reviewer_actor_id,
            "promoted_at": _now(),
            "promotion_reason": promotion_reason,
        }
    )
    raw_payload["source_provenance"] = source_provenance

    record = build_trajectory({**raw_payload, "trajectory_id": trajectory_id})
    sanitized, manifest = sanitize_trajectory(record)
    if use_postgres_state():
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE agent_trajectories
                SET consent_state = %s,
                    training_eligible = %s,
                    exclusion_reasons_json = %s::jsonb,
                    source_provenance_json = %s::jsonb,
                    sanitized_payload_json = %s::jsonb,
                    redaction_manifest_json = %s::jsonb
                WHERE trajectory_id = %s AND tombstoned_at IS NULL
                """,
                (
                    record.consent_state,
                    record.training_eligible,
                    _json_dump(record.exclusion_reasons),
                    _json_dump(record.source_provenance),
                    _json_dump(sanitized),
                    _json_dump(manifest),
                    trajectory_id,
                ),
            )
            conn.commit()
            if not cur.rowcount:
                return None
    else:
        with _sqlite_connect() as conn:
            cur = conn.execute(
                """
                UPDATE agent_trajectories
                SET consent_state = ?,
                    training_eligible = ?,
                    exclusion_reasons_json = ?,
                    source_provenance_json = ?,
                    sanitized_payload_json = ?,
                    redaction_manifest_json = ?
                WHERE trajectory_id = ? AND tombstoned_at IS NULL
                """,
                (
                    record.consent_state,
                    int(record.training_eligible),
                    _json_dump(record.exclusion_reasons),
                    _json_dump(record.source_provenance),
                    _json_dump(sanitized),
                    _json_dump(manifest),
                    trajectory_id,
                ),
            )
            conn.commit()
            if not cur.rowcount:
                return None
    return get_trajectory(trajectory_id)


def tombstone_trajectory(trajectory_id: str, *, reason: str = "deletion_requested") -> bool:
    """Remove a trajectory from training eligibility while preserving audit lineage."""

    tombstoned_at = _now()
    if use_postgres_state():
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE agent_trajectories
                SET tombstoned_at = %s,
                    deletion_reason = %s,
                    training_eligible = FALSE,
                    sanitized_payload_json = jsonb_set(
                        COALESCE(sanitized_payload_json, '{}'::jsonb),
                        '{training_eligible}',
                        'false'::jsonb,
                        true
                    )
                WHERE trajectory_id = %s AND tombstoned_at IS NULL
                """,
                (tombstoned_at, reason, trajectory_id),
            )
            conn.commit()
            updated = bool(cur.rowcount)
    else:
        with _sqlite_connect() as conn:
            row = conn.execute(
                "SELECT sanitized_payload_json FROM agent_trajectories WHERE trajectory_id = ? AND tombstoned_at IS NULL",
                (trajectory_id,),
            ).fetchone()
            if not row:
                return False
            sanitized = _json_load(row["sanitized_payload_json"], {})
            if isinstance(sanitized, dict):
                sanitized["training_eligible"] = False
                reasons = list(sanitized.get("exclusion_reasons") or [])
                if "deletion_requested" not in reasons:
                    reasons.append("deletion_requested")
                sanitized["exclusion_reasons"] = reasons
            cur = conn.execute(
                """
                UPDATE agent_trajectories
                SET tombstoned_at = ?,
                    deletion_reason = ?,
                    training_eligible = 0,
                    sanitized_payload_json = ?
                WHERE trajectory_id = ? AND tombstoned_at IS NULL
                """,
                (tombstoned_at, reason, _json_dump(sanitized), trajectory_id),
            )
            conn.commit()
            updated = bool(cur.rowcount)

    if updated:
        try:
            from api.agent_response_feedback import tombstone_feedback_for_trajectory

            tombstone_feedback_for_trajectory(trajectory_id, reason=reason)
        except Exception:
            logger.exception("agent_feedback_tombstone_failed trajectory_id=%s", trajectory_id)
    return updated


def export_sanitized_trajectories(*, limit: int = 1000) -> list[dict[str, Any]]:
    """Return training-safe trajectory payloads, rejecting invalid eligible rows."""

    rows = list_trajectories(limit=limit, training_eligible_only=True)
    exported: list[dict[str, Any]] = []
    for row in rows:
        exported.append(_exportable_payload(row))
    return exported


def _has_preference_export_consent(row: dict[str, Any]) -> bool:
    if row.get("consent_state") != "granted":
        return False
    if row.get("training_eligible"):
        return True
    source_provenance = row.get("source_provenance")
    if isinstance(source_provenance, dict) and source_provenance.get("preference_export_consent"):
        return True
    return False


def _validated_sanitized_payload(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
        raise TrajectoryExportError(f"Unsupported trajectory schema version for {row.get('trajectory_id')}")
    if row.get("tombstoned_at"):
        raise TrajectoryExportError(f"Tombstoned trajectory cannot be exported: {row.get('trajectory_id')}")
    manifest = row.get("redaction_manifest")
    if not isinstance(manifest, dict) or manifest.get("policy") != TRAJECTORY_REDACTION_POLICY:
        raise TrajectoryExportError(f"Missing redaction manifest for {row.get('trajectory_id')}")
    payload = row.get("sanitized_payload")
    if not isinstance(payload, dict) or not payload:
        raise TrajectoryExportError(f"Missing sanitized payload for {row.get('trajectory_id')}")
    rerendered, _findings = redact_secrets(payload)
    if rerendered != payload:
        raise TrajectoryExportError(
            f"Sanitized payload still contains restricted fields for {row.get('trajectory_id')}"
        )
    if not payload.get("messages") or not payload.get("steps"):
        raise TrajectoryExportError(f"Incomplete trajectory cannot be exported: {row.get('trajectory_id')}")
    return payload


def _exportable_payload(row: dict[str, Any]) -> dict[str, Any]:
    if not row.get("training_eligible"):
        raise TrajectoryExportError(f"Ineligible trajectory cannot be exported: {row.get('trajectory_id')}")
    return _validated_sanitized_payload(row)


def _exportable_preference_payload(row: dict[str, Any]) -> dict[str, Any]:
    if not _has_preference_export_consent(row):
        raise TrajectoryExportError(f"Trajectory lacks preference export consent: {row.get('trajectory_id')}")
    return _validated_sanitized_payload(row)


def reset_agent_trajectory_store_for_tests() -> None:
    if _SQLITE_PATH.exists():
        _SQLITE_PATH.unlink()
