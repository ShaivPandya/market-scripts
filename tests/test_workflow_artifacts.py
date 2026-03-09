"""Tests for workflow artifact extraction and persistence."""

from __future__ import annotations

import pytest

import portfolio.core_db as core_db
from api.workflow_artifacts import extract_artifacts, persist_artifacts


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "test_core.db")
    monkeypatch.setattr(core_db, "_conn", None)
    yield
    if core_db._conn:
        try:
            core_db._conn.close()
        except Exception:
            pass
    monkeypatch.setattr(core_db, "_conn", None)


SAMPLE_SYNTHESIS = """Here is my analysis of MU post-earnings...

The thesis remains intact. Revenue beat by 5%.

```artifacts
{
  "evaluation_draft": {
    "ticker": "MU",
    "thesis_status": "active",
    "technical_read": "Bullish — breakout above 200d MA",
    "fundamental_read": "Strong — beat on revenue and margins",
    "action": "hold",
    "confidence": "high",
    "key_developments": ["HBM3 revenue exceeded expectations", "NAND margins improving"],
    "earnings_note": "Beat by 5% on revenue"
  },
  "action_items": [
    {"description": "Review position size after strong beat", "action_type": "resize", "urgency": "normal"}
  ],
  "watch_triggers": [
    {"condition": "MU breaks $150", "trigger_type": "price_level", "ticker": "MU"}
  ]
}
```
"""

SYNTHESIS_NO_ARTIFACTS = """Here is my analysis. The thesis is intact.

No changes recommended at this time."""

SYNTHESIS_BAD_JSON = """Analysis complete.

```artifacts
{this is not valid json}
```
"""


def test_extract_artifacts_success():
    artifacts = extract_artifacts(SAMPLE_SYNTHESIS, "post_earnings_review")
    assert "evaluation_draft" in artifacts
    assert artifacts["evaluation_draft"]["ticker"] == "MU"
    assert len(artifacts["action_items"]) == 1
    assert len(artifacts["watch_triggers"]) == 1


def test_extract_artifacts_no_block():
    artifacts = extract_artifacts(SYNTHESIS_NO_ARTIFACTS, "morning_brief")
    assert artifacts == {}


def test_extract_artifacts_bad_json():
    artifacts = extract_artifacts(SYNTHESIS_BAD_JSON, "thesis_review")
    assert artifacts == {}


def test_persist_artifacts_creates_approvals():
    artifacts = extract_artifacts(SAMPLE_SYNTHESIS, "post_earnings_review")
    count = persist_artifacts("test-run-123", "MU", artifacts)
    assert count == 3  # 1 eval + 1 action + 1 trigger

    approvals = core_db.get_pending_approvals()
    assert len(approvals) == 3

    types = {a["entity_type"] for a in approvals}
    assert "evaluation" in types
    assert "action_item" in types
    assert "watch_trigger" in types


def test_persist_empty_artifacts():
    count = persist_artifacts("test-run-456", "MU", {})
    assert count == 0


def test_persist_thesis_status_change():
    artifacts = {
        "thesis_status_change": {"new_status": "under_review", "reason": "Multiple kill conditions approaching"}
    }
    count = persist_artifacts("test-run-789", "MU", artifacts)
    assert count == 1

    approvals = core_db.get_pending_approvals()
    assert len(approvals) == 1
    assert approvals[0]["entity_type"] == "thesis_status"


def test_persist_kill_condition_updates():
    artifacts = {
        "kill_condition_updates": [
            {"kill_condition_id": 1, "condition": "Revenue miss", "status": "triggered"},
            {"kill_condition_id": 2, "condition": "Old risk", "status": "retired"},
        ]
    }
    count = persist_artifacts("test-run-abc", "MU", artifacts)
    assert count == 2
