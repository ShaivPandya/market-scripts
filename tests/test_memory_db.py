from __future__ import annotations

import json

import pytest

from api import memory_db, memory_manager


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


def test_begin_turn_and_complete_turn_messages(temp_memory_db):
    session = memory_db.get_or_create_session(None)
    sid = session["session_id"]
    turn_id = "turn-begin-complete"
    user = {"role": "user", "content": "hello", "timestamp": 1.0, "client_turn_id": turn_id}
    assistant_placeholder = {
        "role": "assistant",
        "content": "",
        "timestamp": 2.0,
        "client_turn_id": turn_id,
        "is_streaming": True,
    }

    first = memory_db.begin_turn(sid, user, assistant_placeholder)
    second = memory_db.begin_turn(sid, user, assistant_placeholder)
    assert first == 2
    assert second == 2

    memory_db.update_assistant_message(sid, turn_id, {"content": "partial", "is_streaming": True})
    loaded = memory_db.get_session(sid)
    assert loaded is not None
    assert loaded["transcript"][1]["content"] == "partial"

    final_assistant = {
        "role": "assistant",
        "content": "done",
        "timestamp": 3.0,
        "client_turn_id": turn_id,
    }
    total = memory_db.complete_turn_messages(sid, turn_id, user, final_assistant)
    loaded = memory_db.get_session(sid)
    assert total == 2
    assert loaded is not None
    assert loaded["transcript"][1]["content"] == "done"
    assert loaded["transcript"][1].get("is_streaming") is False


def test_fail_turn_marks_assistant_cancelled(temp_memory_db):
    session = memory_db.get_or_create_session(None)
    sid = session["session_id"]
    turn_id = "turn-fail"
    user = {"role": "user", "content": "hi", "timestamp": 1.0, "client_turn_id": turn_id}
    assistant_placeholder = {
        "role": "assistant",
        "content": "partial",
        "timestamp": 2.0,
        "client_turn_id": turn_id,
        "is_streaming": True,
    }
    memory_db.begin_turn(sid, user, assistant_placeholder)
    assert memory_db.fail_turn(sid, turn_id, status="cancelled", content="partial") is True
    loaded = memory_db.get_session(sid)
    assert loaded is not None
    assert loaded["transcript"][1]["status"] == "cancelled"
    assert loaded["transcript"][1].get("is_streaming") is False


def test_append_messages_dedupes_client_turn_id(temp_memory_db):
    session = memory_db.get_or_create_session(None)
    sid = session["session_id"]
    messages = [
        {"role": "user", "content": "hello", "timestamp": 1.0, "client_turn_id": "turn-1"},
        {"role": "assistant", "content": "Hey.", "timestamp": 2.0, "client_turn_id": "turn-1"},
    ]

    first_total = memory_db.append_messages(sid, messages)
    second_total = memory_db.append_messages(sid, messages)
    loaded = memory_db.get_session(sid)

    assert first_total == 2
    assert second_total == 2
    assert loaded is not None
    assert loaded["transcript"] == messages


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


def test_deterministic_title_strips_workflow_command(temp_memory_db):
    session = memory_db.get_or_create_session(None)
    sid = session["session_id"]

    did_update = memory_db.set_deterministic_title_if_missing(
        sid,
        "/workflow:position_risk_review:NVDA",
    )
    loaded = memory_db.get_session(sid)

    assert did_update is True
    assert loaded is not None
    assert loaded["title"] == "NVDA Position Risk Review"
    assert loaded["title_source"] == "deterministic"


def test_manual_title_is_not_overwritten_by_generated_title(temp_memory_db):
    session = memory_db.get_or_create_session(None)
    sid = session["session_id"]

    renamed = memory_db.rename_session(sid, "My Manual Title")
    deterministic = memory_db.set_deterministic_title_if_missing(sid, "Should not replace manual")
    generated = memory_db.update_generated_title(sid, "Generated Title")
    loaded = memory_db.get_session(sid)

    assert renamed is not None
    assert deterministic is False
    assert generated is False
    assert loaded is not None
    assert loaded["title"] == "My Manual Title"
    assert loaded["title_source"] == "manual"


def test_finalize_turn_sets_title_and_schedules_refinement(temp_memory_db, monkeypatch):
    session = memory_db.get_or_create_session(None)
    sid = session["session_id"]
    scheduled: list[tuple[str, dict, dict]] = []
    monkeypatch.setattr(
        memory_manager,
        "_refine_session_title_async",
        lambda session_id, user_msg, assistant_msg: scheduled.append((session_id, user_msg, assistant_msg)),
    )

    memory_manager.finalize_turn(
        sid,
        {"role": "user", "content": "Analyze NVDA earnings setup", "timestamp": 1.0},
        {"role": "assistant", "content": "NVDA earnings setup looks constructive.", "timestamp": 2.0},
    )
    loaded = memory_db.get_session(sid)

    assert loaded is not None
    assert loaded["title"] == "Analyze NVDA earnings setup"
    assert loaded["title_source"] == "deterministic"
    assert [item[0] for item in scheduled] == [sid]


def test_patch_session_title_updates_metadata(temp_memory_db, auth_client):
    session = memory_db.get_or_create_session(None)
    sid = session["session_id"]

    resp = auth_client.patch(f"/api/memory/sessions/{sid}", json={"title": "  Renamed   chat  "})

    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Renamed chat"
    assert data["title_source"] == "manual"
    assert data["title_updated_at"]

    listed = auth_client.get("/api/memory/sessions").json()
    assert listed[0]["title"] == "Renamed chat"


def test_patch_session_title_validates_empty_and_missing(temp_memory_db, auth_client):
    session = memory_db.get_or_create_session(None)
    sid = session["session_id"]

    empty = auth_client.patch(f"/api/memory/sessions/{sid}", json={"title": "   "})
    missing = auth_client.patch("/api/memory/sessions/missing-session", json={"title": "New title"})

    assert empty.status_code == 400
    assert missing.status_code == 404
