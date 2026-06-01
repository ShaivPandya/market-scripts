from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.intent_router_training_store import (
    insert_training_row,
    list_training_rows,
    update_opportunity_candidate_metadata,
)
from decision_quality.intent_router import (
    RouteDecision,
    should_capture_training_row,
    training_row_from_telemetry,
)
from decision_quality.intent_router_training import export_training_dataset, train_baseline_classifier


@pytest.fixture(autouse=True)
def _isolate_training_store(tmp_path, monkeypatch):
    db_path = tmp_path / "intent_router_training.sqlite3"
    monkeypatch.setenv("STATE_DB_BACKEND", "sqlite")
    monkeypatch.setenv("DATABASE_URL", "")
    import api.intent_router_training_store as store

    monkeypatch.setattr(store, "_SQLITE_PATH", db_path)
    yield


def test_training_row_from_telemetry_includes_versioned_fields():
    applied = RouteDecision(
        intent_class="thesis_review",
        run_hidden_dq=True,
        run_opportunity_preflight=True,
        workflow_name=None,
        workflow_ticker=None,
        tool_names=["get_thesis"],
        confidence=1.0,
        source="regex",
    )
    row = training_row_from_telemetry(
        user_text="what do you think about nvidia as a long?",
        route_meta={
            "regex_baseline": applied.to_meta(),
            "llm_candidate": {"intent_class": "general_research"},
            "shadow_comparison": {"intent_match": False},
            "applied_source": "regex_shadow",
            "confidence_threshold": 0.7,
        },
        session_id="sess-1",
        client_turn_id="turn-1",
        screen_context={"ticker": "NVDA"},
        recent_session_features=[{"role": "user", "content": "prior"}],
        applied_route=applied.to_meta(),
        sampling_reason="shadow_all",
    )
    assert row["schema_version"] == 1
    assert row["session_id"] == "sess-1"
    assert row["recent_session_features"][0]["role"] == "user"
    assert row["applied_route"]["intent_class"] == "thesis_review"


def test_should_capture_training_row_respects_flags(monkeypatch):
    monkeypatch.delenv("AGENT_INTENT_ROUTER_TRAINING_CAPTURE_ENABLED", raising=False)
    should, reason = should_capture_training_row(route_meta={"shadow_comparison": {"intent_match": False}})
    assert should is False
    assert reason == "capture_disabled"

    monkeypatch.setenv("AGENT_INTENT_ROUTER_TRAINING_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("AGENT_INTENT_ROUTER_TRAINING_CAPTURE_MISMATCH_ONLY", "true")
    should, reason = should_capture_training_row(
        route_meta={
            "shadow_comparison": {
                "intent_match": True,
                "hidden_dq_match": True,
                "opportunity_preflight_match": True,
                "workflow_match": True,
            }
        }
    )
    assert should is False
    assert reason == "mismatch_only_match"


def test_insert_and_list_training_rows():
    row = {
        "session_id": "sess-2",
        "client_turn_id": "turn-2",
        "user_text": "Scan semiconductors",
        "regex_baseline": {"intent_class": "opportunity_discovery"},
        "applied_source": "regex_shadow",
    }
    row_id = insert_training_row(row)
    assert row_id
    rows = list_training_rows(limit=10)
    assert len(rows) == 1
    assert rows[0]["session_id"] == "sess-2"


def test_update_opportunity_candidate_metadata():
    insert_training_row(
        {
            "session_id": "sess-3",
            "client_turn_id": "turn-3",
            "user_text": "NVDA thesis",
        }
    )
    updated = update_opportunity_candidate_metadata(
        session_id="sess-3",
        client_turn_id="turn-3",
        opportunity_candidate_metadata={"trigger": "earnings", "opportunity_type": "undervalued_asset"},
    )
    assert updated is True
    rows = list_training_rows(limit=1)
    assert rows[0]["opportunity_candidate_metadata"]["trigger"] == "earnings"


def test_active_learning_export_excludes_drafts_and_includes_failure_metadata(tmp_path, monkeypatch):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "draft_capture.json").write_text(
        json.dumps(
            {
                "id": "draft_capture",
                "status": "draft",
                "user_message": "draft only",
                "failure_tags": ["wrong_routing"],
                "routing_expectations": {"intent_class": "thesis_review"},
            }
        ),
        encoding="utf-8",
    )
    (cases_dir / "review_capture.json").write_text(
        json.dumps(
            {
                "id": "review_capture",
                "status": "review",
                "user_message": "review me",
                "failure_tags": ["wrong_routing"],
                "failure_type": "wrong_routing",
                "corpus_tags": ["routing_tool_use"],
                "source_session_id": "sess-review",
                "routing_expectations": {
                    "intent_class": "thesis_review",
                    "run_hidden_dq": True,
                    "required_tool_names": ["get_thesis"],
                },
            }
        ),
        encoding="utf-8",
    )
    from decision_quality import intent_router_training as training_module

    original_load_cases = training_module.load_cases

    def _load_cases(*args, **kwargs):
        kwargs.setdefault("cases_dir", cases_dir)
        return original_load_cases(*args, **kwargs)

    monkeypatch.setattr(training_module, "load_cases", _load_cases)

    manifest = export_training_dataset(
        output_dir=tmp_path / "exports",
        include_db_rows=False,
        active_learning_only=True,
    )
    dataset_path = Path(manifest["dataset_path"])
    assert dataset_path.name == "active_learning_router.jsonl"
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["case_id"] == "review_capture"
    assert row["failure_tags"] == ["wrong_routing"]
    assert row["eval_status"] == "review"
    assert row["source_session_id"] == "sess-review"
    assert row["label_intent_class"] == "thesis_review"


def test_export_training_dataset_from_fixtures(tmp_path):
    manifest = export_training_dataset(
        output_dir=tmp_path,
        include_db_rows=False,
        fixture_prefix="routing_",
    )
    assert manifest["row_count"] >= 5
    dataset_path = Path(manifest["dataset_path"])
    assert dataset_path.exists()
    first = json.loads(dataset_path.read_text(encoding="utf-8").splitlines()[0])
    assert first["label_intent_class"]


def test_train_baseline_classifier_from_export(tmp_path):
    export_training_dataset(
        output_dir=tmp_path / "exports",
        include_db_rows=False,
        fixture_prefix="routing_",
    )
    dataset_path = sorted((tmp_path / "exports").glob("*/dataset.jsonl"))[-1]
    result = train_baseline_classifier(
        dataset_path=dataset_path,
        output_dir=tmp_path / "models",
        holdout_ratio=0.2,
    )
    assert Path(result["artifact_path"]).exists()
    assert result["metrics"]["train_rows"] >= 1


def test_supervised_route_prediction(tmp_path, monkeypatch):
    export_training_dataset(
        output_dir=tmp_path / "exports",
        include_db_rows=False,
        fixture_prefix="routing_",
    )
    dataset_path = sorted((tmp_path / "exports").glob("*/dataset.jsonl"))[-1]
    result = train_baseline_classifier(
        dataset_path=dataset_path,
        output_dir=tmp_path / "models",
        holdout_ratio=0.2,
    )

    import api.routers.agent as agent_router
    from decision_quality.intent_router import build_route_context, resolve_agent_route
    from decision_quality.intent_router_supervised import predict_route_decision

    baseline = agent_router.build_regex_route_decision(
        user_text="what do you think about nvidia as a long?",
        select_tool_names=agent_router._select_tool_names,
        detect_workflow=agent_router._detect_workflow,
        should_run_hidden_dq=agent_router._should_run_decision_quality_chat,
        should_run_opportunity_preflight=agent_router._should_run_opportunity_candidate_preflight,
        screen_context=None,
    )
    context = build_route_context(user_text="what do you think about nvidia as a long?")
    decision = predict_route_decision(
        context=context,
        regex_baseline=baseline,
        model_path=Path(result["artifact_path"]),
    )
    assert decision is not None
    assert decision.source == "supervised"
    assert decision.run_hidden_dq is True

    monkeypatch.setenv("AGENT_INTENT_ROUTER_SUPERVISED_ENABLED", "true")
    monkeypatch.setenv("AGENT_INTENT_ROUTER_SUPERVISED_MODEL_PATH", result["artifact_path"])
    monkeypatch.setenv("AGENT_INTENT_ROUTER_ENABLED", "false")
    monkeypatch.setenv("AGENT_INTENT_ROUTER_SHADOW_MODE", "false")

    effective, meta = resolve_agent_route(context=context, regex_baseline=baseline)
    assert meta.get("supervised_candidate")
    assert effective.source in {"supervised", "regex"}
