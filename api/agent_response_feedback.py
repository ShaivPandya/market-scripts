"""Human-reviewed agent response feedback and labeling storage.

Explicit labels are stored separately from trajectory capture and inferred
behavioral signals. Approved trajectories can be promoted for sanitized export
only when a reviewer explicitly opts in.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from api.agent_governance import redact_secrets
from api.postgres import connect, use_postgres_state
from api.provenance import stable_hash

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SQLITE_PATH = _REPO_ROOT / "data_cache" / "agent_response_feedback.sqlite3"

FEEDBACK_SCHEMA_VERSION = 1
FEEDBACK_RETENTION_CLASS = "agent_feedback_365d"
HUMAN_REVIEWED_SIGNAL = "human_reviewed"

FeedbackDecision = Literal["approve", "reject", "correct"]

FAILURE_TAG_CATEGORIES = frozenset(
    {
        "routing",
        "tools",
        "source_quality",
        "synthesis",
        "calibration",
        "policy_boundary",
    }
)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS agent_response_feedback (
    feedback_id TEXT PRIMARY KEY,
    trajectory_id TEXT NOT NULL,
    session_id TEXT,
    client_turn_id TEXT,
    response_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    decision TEXT NOT NULL,
    reviewer_actor_id TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    model TEXT,
    provider TEXT,
    corrected_response TEXT,
    failure_tags_json TEXT NOT NULL DEFAULT '[]',
    notes TEXT,
    training_eligible INTEGER NOT NULL DEFAULT 0,
    signal_source TEXT NOT NULL DEFAULT 'human_reviewed',
    provenance_json TEXT NOT NULL DEFAULT '{}',
    retention_class TEXT NOT NULL DEFAULT 'agent_feedback_365d',
    tombstoned_at TEXT,
    deletion_reason TEXT
)
"""

_CREATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_agent_response_feedback_trajectory ON agent_response_feedback(trajectory_id)",
    "CREATE INDEX IF NOT EXISTS idx_agent_response_feedback_session_turn ON agent_response_feedback(session_id, client_turn_id)",
    "CREATE INDEX IF NOT EXISTS idx_agent_response_feedback_reviewed_at ON agent_response_feedback(reviewed_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_agent_response_feedback_training ON agent_response_feedback(training_eligible, reviewed_at DESC)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_response_feedback_reviewer_turn ON agent_response_feedback(trajectory_id, reviewer_actor_id, response_version)",
)


class AgentResponseFeedback(BaseModel):
    """Versioned human label tied to one immutable trajectory response."""

    feedback_id: str
    schema_version: int = FEEDBACK_SCHEMA_VERSION
    trajectory_id: str
    session_id: str | None = None
    client_turn_id: str | None = None
    response_version: str
    decision: FeedbackDecision
    reviewer_actor_id: str
    reviewed_at: str
    model: str | None = None
    provider: str | None = None
    corrected_response: str | None = None
    failure_tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    training_eligible: bool = False
    signal_source: str = HUMAN_REVIEWED_SIGNAL
    provenance: dict[str, Any] = Field(default_factory=dict)
    retention_class: str = FEEDBACK_RETENTION_CLASS
    tombstoned_at: str | None = None
    deletion_reason: str | None = None

    @field_validator("schema_version")
    @classmethod
    def _supported_schema(cls, value: int) -> int:
        if value != FEEDBACK_SCHEMA_VERSION:
            raise ValueError(f"Unsupported feedback schema version: {value}")
        return value

    @field_validator("failure_tags")
    @classmethod
    def _validate_failure_tags(cls, value: list[str]) -> list[str]:
        invalid = [tag for tag in value if tag not in FAILURE_TAG_CATEGORIES]
        if invalid:
            raise ValueError(f"Unsupported failure tags: {', '.join(invalid)}")
        return value

    @model_validator(mode="after")
    def _correct_requires_payload(self) -> AgentResponseFeedback:
        if self.decision == "correct" and not self.corrected_response:
            raise ValueError("corrected_response is required when decision is correct")
        return self

    @field_validator("corrected_response")
    @classmethod
    def _strip_corrected_response(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class FeedbackStoreError(ValueError):
    """Raised when feedback cannot be stored or exported."""


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
        try:
            conn.execute(statement)
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return conn


def feedback_id_for(*, trajectory_id: str, reviewer_actor_id: str, response_version: str) -> str:
    return (
        "agent_feedback:"
        f"{stable_hash({'trajectory_id': trajectory_id, 'reviewer_actor_id': reviewer_actor_id, 'response_version': response_version}, length=24)}"
    )


def response_version_for_trajectory(trajectory: dict[str, Any]) -> str:
    """Immutable response identity from model metadata and assistant content."""

    payload = trajectory.get("sanitized_payload")
    if not isinstance(payload, dict):
        payload = trajectory.get("raw_payload") if isinstance(trajectory.get("raw_payload"), dict) else {}
    messages = list(payload.get("messages") or [])
    assistant_content = ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            assistant_content = str(message.get("content") or "")
            break
    return stable_hash(
        {
            "model": trajectory.get("model"),
            "provider": trajectory.get("provider"),
            "prompt_version": trajectory.get("prompt_version"),
            "code_version": trajectory.get("code_version"),
            "assistant_content": assistant_content,
        },
        length=16,
    )


def _row_to_dict(row: Any) -> dict[str, Any]:
    data = dict(row)
    return {
        "feedback_id": data.get("feedback_id"),
        "trajectory_id": data.get("trajectory_id"),
        "session_id": data.get("session_id"),
        "client_turn_id": data.get("client_turn_id"),
        "response_version": data.get("response_version"),
        "schema_version": int(data.get("schema_version") or 0),
        "decision": data.get("decision"),
        "reviewer_actor_id": data.get("reviewer_actor_id"),
        "reviewed_at": str(data.get("reviewed_at")),
        "model": data.get("model"),
        "provider": data.get("provider"),
        "corrected_response": data.get("corrected_response"),
        "failure_tags": _json_load(data.get("failure_tags_json"), []),
        "notes": data.get("notes"),
        "training_eligible": bool(data.get("training_eligible")),
        "signal_source": data.get("signal_source") or HUMAN_REVIEWED_SIGNAL,
        "provenance": _json_load(data.get("provenance_json"), {}),
        "retention_class": data.get("retention_class"),
        "tombstoned_at": str(data.get("tombstoned_at")) if data.get("tombstoned_at") else None,
        "deletion_reason": data.get("deletion_reason"),
    }


def _public_feedback(row: dict[str, Any], *, include_corrected: bool = False) -> dict[str, Any]:
    payload = dict(row)
    if not include_corrected:
        payload.pop("corrected_response", None)
    payload["human_reviewed"] = payload.get("signal_source") == HUMAN_REVIEWED_SIGNAL
    return payload


def upsert_feedback(payload: dict[str, Any]) -> dict[str, Any]:
    """Create or update one human-reviewed label for a trajectory response."""

    try:
        record = AgentResponseFeedback.model_validate(payload)
    except ValidationError as exc:
        raise FeedbackStoreError(str(exc)) from exc

    if use_postgres_state():
        return _upsert_postgres(record)
    return _upsert_sqlite(record)


def _upsert_values(record: AgentResponseFeedback, *, bool_as_int: bool) -> tuple[Any, ...]:
    return (
        record.feedback_id,
        record.trajectory_id,
        record.session_id,
        record.client_turn_id,
        record.response_version,
        record.schema_version,
        record.decision,
        record.reviewer_actor_id,
        record.reviewed_at,
        record.model,
        record.provider,
        record.corrected_response,
        _json_dump(record.failure_tags),
        record.notes,
        int(record.training_eligible) if bool_as_int else record.training_eligible,
        record.signal_source,
        _json_dump(record.provenance),
        record.retention_class,
    )


def _upsert_sqlite(record: AgentResponseFeedback) -> dict[str, Any]:
    with _sqlite_connect() as conn:
        conn.execute(
            """
            INSERT INTO agent_response_feedback (
                feedback_id, trajectory_id, session_id, client_turn_id, response_version,
                schema_version, decision, reviewer_actor_id, reviewed_at, model, provider,
                corrected_response, failure_tags_json, notes, training_eligible,
                signal_source, provenance_json, retention_class
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trajectory_id, reviewer_actor_id, response_version) DO UPDATE SET
                decision = excluded.decision,
                reviewed_at = excluded.reviewed_at,
                corrected_response = excluded.corrected_response,
                failure_tags_json = excluded.failure_tags_json,
                notes = excluded.notes,
                training_eligible = excluded.training_eligible,
                provenance_json = excluded.provenance_json,
                tombstoned_at = NULL,
                deletion_reason = NULL
            """,
            _upsert_values(record, bool_as_int=True),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM agent_response_feedback WHERE feedback_id = ?",
            (record.feedback_id,),
        ).fetchone()
        if not row:
            raise FeedbackStoreError("Feedback upsert failed")
        return _row_to_dict(row)


def _upsert_postgres(record: AgentResponseFeedback) -> dict[str, Any]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO agent_response_feedback (
                feedback_id, trajectory_id, session_id, client_turn_id, response_version,
                schema_version, decision, reviewer_actor_id, reviewed_at, model, provider,
                corrected_response, failure_tags_json, notes, training_eligible,
                signal_source, provenance_json, retention_class
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s::jsonb, %s, %s,
                %s, %s::jsonb, %s
            )
            ON CONFLICT (trajectory_id, reviewer_actor_id, response_version) DO UPDATE SET
                decision = EXCLUDED.decision,
                reviewed_at = EXCLUDED.reviewed_at,
                corrected_response = EXCLUDED.corrected_response,
                failure_tags_json = EXCLUDED.failure_tags_json,
                notes = EXCLUDED.notes,
                training_eligible = EXCLUDED.training_eligible,
                provenance_json = EXCLUDED.provenance_json,
                tombstoned_at = NULL,
                deletion_reason = NULL
            RETURNING *
            """,
            _upsert_values(record, bool_as_int=False),
        )
        row = cur.fetchone()
        conn.commit()
        if not row:
            raise FeedbackStoreError("Feedback upsert failed")
        return _row_to_dict(row)


def get_feedback(feedback_id: str) -> dict[str, Any] | None:
    if use_postgres_state():
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM agent_response_feedback WHERE feedback_id = %s", (feedback_id,))
            row = cur.fetchone()
            return _row_to_dict(row) if row else None
    with _sqlite_connect() as conn:
        row = conn.execute(
            "SELECT * FROM agent_response_feedback WHERE feedback_id = ?",
            (feedback_id,),
        ).fetchone()
        return _row_to_dict(row) if row else None


def list_feedback_for_trajectory(trajectory_id: str, *, include_tombstoned: bool = False) -> list[dict[str, Any]]:
    clauses = ["trajectory_id = ?" if not use_postgres_state() else "trajectory_id = %s"]
    if not include_tombstoned:
        clauses.append("tombstoned_at IS NULL")
    if use_postgres_state():
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT * FROM agent_response_feedback
                WHERE {" AND ".join(clauses)}
                ORDER BY reviewed_at DESC
                """,
                (trajectory_id,),
            )
            return [_row_to_dict(row) for row in cur.fetchall()]
    with _sqlite_connect() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM agent_response_feedback
            WHERE {" AND ".join(clauses)}
            ORDER BY reviewed_at DESC
            """,
            (trajectory_id,),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]


def list_review_queue(*, limit: int = 50) -> list[dict[str, Any]]:
    """Return recent trajectories that have no active human feedback yet."""

    from api.agent_trajectories import list_trajectories

    trajectories = list_trajectories(limit=limit)
    queue: list[dict[str, Any]] = []
    for trajectory in trajectories:
        labels = list_feedback_for_trajectory(str(trajectory["trajectory_id"]))
        if labels:
            continue
        if trajectory.get("final_disposition") not in {"succeeded", "blocked"}:
            continue
        queue.append(
            {
                "trajectory_id": trajectory.get("trajectory_id"),
                "session_id": trajectory.get("session_id"),
                "client_turn_id": trajectory.get("client_turn_id"),
                "captured_at": trajectory.get("captured_at"),
                "final_disposition": trajectory.get("final_disposition"),
                "provider": trajectory.get("provider"),
                "model": trajectory.get("model"),
                "response_version": response_version_for_trajectory(trajectory),
            }
        )
    return queue


def list_feedback(
    *,
    limit: int = 100,
    training_eligible_only: bool = False,
    trajectory_id: str | None = None,
) -> list[dict[str, Any]]:
    clauses: list[str] = ["tombstoned_at IS NULL"]
    params: list[Any] = []
    if trajectory_id:
        clauses.append("trajectory_id = %s" if use_postgres_state() else "trajectory_id = ?")
        params.append(trajectory_id)
    if training_eligible_only:
        clauses.append("training_eligible = TRUE" if use_postgres_state() else "training_eligible = 1")
    placeholder = "%s" if use_postgres_state() else "?"
    params.append(limit)
    if use_postgres_state():
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT * FROM agent_response_feedback
                WHERE {" AND ".join(clauses)}
                ORDER BY reviewed_at DESC
                LIMIT {placeholder}
                """,
                tuple(params),
            )
            return [_row_to_dict(row) for row in cur.fetchall()]
    with _sqlite_connect() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM agent_response_feedback
            WHERE {" AND ".join(clauses)}
            ORDER BY reviewed_at DESC
            LIMIT {placeholder}
            """,
            tuple(params),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]


def tombstone_feedback_for_trajectory(trajectory_id: str, *, reason: str = "trajectory_deleted") -> int:
    """Exclude all feedback labels for a trajectory from future exports."""

    tombstoned_at = _now()
    if use_postgres_state():
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE agent_response_feedback
                SET tombstoned_at = %s,
                    deletion_reason = %s,
                    training_eligible = FALSE
                WHERE trajectory_id = %s AND tombstoned_at IS NULL
                """,
                (tombstoned_at, reason, trajectory_id),
            )
            conn.commit()
            return int(cur.rowcount or 0)
    with _sqlite_connect() as conn:
        cur = conn.execute(
            """
            UPDATE agent_response_feedback
            SET tombstoned_at = ?,
                deletion_reason = ?,
                training_eligible = 0
            WHERE trajectory_id = ? AND tombstoned_at IS NULL
            """,
            (tombstoned_at, reason, trajectory_id),
        )
        conn.commit()
        return int(cur.rowcount or 0)


def export_human_reviewed_feedback(*, limit: int = 1000) -> list[dict[str, Any]]:
    """Return training-safe human-reviewed labels for preference dataset builders."""

    rows = list_feedback(limit=limit, training_eligible_only=True)
    exported: list[dict[str, Any]] = []
    for row in rows:
        if row.get("signal_source") != HUMAN_REVIEWED_SIGNAL:
            continue
        payload = {
            "feedback_id": row.get("feedback_id"),
            "trajectory_id": row.get("trajectory_id"),
            "session_id": row.get("session_id"),
            "client_turn_id": row.get("client_turn_id"),
            "response_version": row.get("response_version"),
            "decision": row.get("decision"),
            "reviewer_actor_id": row.get("reviewer_actor_id"),
            "reviewed_at": row.get("reviewed_at"),
            "model": row.get("model"),
            "provider": row.get("provider"),
            "failure_tags": row.get("failure_tags") or [],
            "notes": row.get("notes"),
            "training_eligible": True,
            "signal_source": HUMAN_REVIEWED_SIGNAL,
            "human_reviewed": True,
        }
        if row.get("decision") == "correct" and row.get("corrected_response"):
            redacted, _findings = redact_secrets({"corrected_response": row["corrected_response"]})
            payload["corrected_response"] = redacted.get("corrected_response")
        exported.append(payload)
    return exported


def reset_agent_response_feedback_store_for_tests() -> None:
    if _SQLITE_PATH.exists():
        _SQLITE_PATH.unlink()
