"""Add conversation session titles.

Revision ID: 20260515_0002
Revises: 20260515_0001
Create Date: 2026-05-15
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from alembic import op

revision: str = "20260515_0002"
down_revision: str | None = "20260515_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TITLE_MAX_CHARS = 80


def upgrade() -> None:
    op.add_column("conversation_sessions", sa.Column("title", sa.Text(), nullable=True))
    op.add_column("conversation_sessions", sa.Column("title_source", sa.Text(), nullable=True))
    op.add_column("conversation_sessions", sa.Column("title_updated_at", sa.Text(), nullable=True))
    _backfill_titles()


def downgrade() -> None:
    op.drop_column("conversation_sessions", "title_updated_at")
    op.drop_column("conversation_sessions", "title_source")
    op.drop_column("conversation_sessions", "title")


def _backfill_titles() -> None:
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            """
            SELECT session_id, transcript, server_messages
            FROM conversation_sessions
            WHERE title IS NULL OR trim(title) = ''
            """
        )
    ).fetchall()
    now = datetime.now(UTC).isoformat()
    for row in rows:
        record = row._mapping
        title = _title_from_messages(_messages_from_record(record))
        if not title:
            continue
        bind.execute(
            sa.text(
                """
                UPDATE conversation_sessions
                SET title = :title,
                    title_source = :title_source,
                    title_updated_at = :title_updated_at
                WHERE session_id = :session_id
                  AND (title IS NULL OR trim(title) = '')
                """
            ),
            {
                "title": title,
                "title_source": "deterministic",
                "title_updated_at": now,
                "session_id": record["session_id"],
            },
        )


def _messages_from_record(record: Any) -> list[dict[str, Any]]:
    for field in ("transcript", "server_messages"):
        try:
            parsed = json.loads(record[field]) if record[field] else []
        except Exception:
            parsed = []
        if isinstance(parsed, list) and parsed:
            return [item for item in parsed if isinstance(item, dict)]
    return []


def _title_from_messages(messages: list[dict[str, Any]]) -> str | None:
    for message in messages:
        if message.get("role") != "user":
            continue
        title = _deterministic_title(str(message.get("content") or ""))
        if title:
            return title
    return None


def _deterministic_title(value: str) -> str | None:
    text = re.sub(r"\s+", " ", value).strip()
    if not text:
        return None
    match = re.match(r"^/workflow:([A-Za-z0-9_]+)(?::([A-Za-z0-9._=-]+))?(?:\s+(.*))?$", text)
    if match:
        workflow = match.group(1).replace("_", " ").strip().title()
        ticker = (match.group(2) or "").strip().upper()
        trailing = (match.group(3) or "").strip()
        text = trailing or (f"{ticker} {workflow}" if ticker else workflow)
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n-:,.!?")
    if not text:
        return None
    if len(text) <= TITLE_MAX_CHARS:
        return text
    candidate = text[:TITLE_MAX_CHARS].rstrip()
    boundary = candidate.rfind(" ")
    if boundary >= 40:
        candidate = candidate[:boundary]
    return candidate.rstrip(" -:,.!?")
