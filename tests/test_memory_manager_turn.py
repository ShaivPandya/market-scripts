from __future__ import annotations

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


def test_build_conversation_context_skips_duplicate_user_turn(temp_memory_db):
    session = memory_db.get_or_create_session(None)
    sid = session["session_id"]
    turn_id = "turn-context"
    memory_manager.begin_turn(
        sid,
        {"role": "user", "content": "Review NVDA", "timestamp": 1.0, "client_turn_id": turn_id},
        {"role": "assistant", "content": "", "timestamp": 2.0, "client_turn_id": turn_id, "is_streaming": True},
    )

    conversation, _ = memory_manager.build_conversation_context(
        sid,
        "Review NVDA",
        enable_retrieval=False,
        client_turn_id=turn_id,
    )

    user_messages = [item for item in conversation if item["role"] == "user" and item["content"] == "Review NVDA"]
    assert len(user_messages) == 1


def test_build_conversation_context_excludes_streaming_assistant(temp_memory_db):
    session = memory_db.get_or_create_session(None)
    sid = session["session_id"]
    turn_id = "turn-streaming"
    memory_manager.begin_turn(
        sid,
        {"role": "user", "content": "hello", "timestamp": 1.0, "client_turn_id": turn_id},
        {"role": "assistant", "content": "partial", "timestamp": 2.0, "client_turn_id": turn_id, "is_streaming": True},
    )

    conversation, _ = memory_manager.build_conversation_context(
        sid,
        "follow up",
        enable_retrieval=False,
        client_turn_id="turn-2",
    )

    assert not any(item.get("content") == "partial" for item in conversation)
