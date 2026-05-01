from __future__ import annotations

import json

import pytest

from api import memory_db


@pytest.fixture
def temp_memory_db(tmp_path, monkeypatch):
    if memory_db._conn is not None:
        try:
            memory_db._conn.close()
        except Exception:
            pass
    monkeypatch.setenv("STATE_DB_BACKEND", "sqlite")
    monkeypatch.setattr(memory_db, "_DB_PATH", tmp_path / "memory.db")
    monkeypatch.setattr(memory_db, "_conn", None)
    yield
    if memory_db._conn is not None:
        try:
            memory_db._conn.close()
        except Exception:
            pass
    memory_db._conn = None


def test_append_messages_updates_transcript_for_v2_sessions(temp_memory_db):
    session = memory_db.get_or_create_session(None)
    sid = session["session_id"]
    messages = [
        {"role": "user", "content": "hello", "timestamp": 1.0},
        {"role": "assistant", "content": "Hey.", "timestamp": 2.0},
    ]

    total = memory_db.append_messages(sid, messages)
    loaded = memory_db.get_session(sid)

    assert total == 2
    assert loaded is not None
    assert loaded["transcript"] == messages
    assert loaded["message_count"] == 2


def test_get_session_falls_back_to_server_messages_for_existing_v2_sessions(temp_memory_db):
    session = memory_db.get_or_create_session(None)
    sid = session["session_id"]
    server_messages = [
        {"role": "user", "content": "portfolio?", "timestamp": 1.0},
        {"role": "assistant", "content": "Use direction-adjusted P&L.", "timestamp": 2.0},
    ]

    conn = memory_db._get_conn()
    with memory_db._lock:
        conn.execute(
            """
            UPDATE conversation_sessions
            SET transcript = '[]', server_messages = ?, message_count = ?
            WHERE session_id = ?
            """,
            (json.dumps(server_messages), len(server_messages), sid),
        )
        conn.commit()

    loaded = memory_db.get_session(sid)

    assert loaded is not None
    assert loaded["transcript"] == server_messages
