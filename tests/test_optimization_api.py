from __future__ import annotations

from fastapi.responses import JSONResponse

import portfolio.core_db as core_db


def _reset_core_db(tmp_path, monkeypatch):
    if core_db._conn:
        core_db._conn.close()
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "optimization_api.db")
    monkeypatch.setattr(core_db, "_conn", None)


def test_admin_scheduler_endpoint_enqueues_continuous_optimizer(client, monkeypatch):
    from api.routers import admin_jobs

    monkeypatch.setenv("SCHEDULER_SECRET", "scheduler-secret")
    captured = {}

    def fake_enqueue_registered_job(job_type, payload, **kwargs):
        captured["job_type"] = job_type
        captured["payload"] = payload
        captured["kwargs"] = kwargs
        return (
            {
                "job_id": "continuous-optimizer-job",
                "job_type": job_type,
                "status": "queued",
                "payload_json": payload,
                "result_json": None,
                "error": None,
                "progress_json": None,
            },
            "created",
        )

    monkeypatch.setattr(admin_jobs, "enqueue_registered_job", fake_enqueue_registered_job)
    monkeypatch.setattr(
        admin_jobs,
        "enqueue_response",
        lambda row, poll_path: JSONResponse({"job_id": row["job_id"], "status": "queued"}, status_code=202),
    )

    resp = client.post(
        "/api/v1/admin/jobs/enqueue-continuous-optimizer",
        headers={"X-Scheduler-Secret": "scheduler-secret"},
    )

    assert resp.status_code == 202
    assert captured["job_type"] == "continuous_optimizer"
    assert captured["payload"] == {"source": "scheduler"}
    assert captured["kwargs"]["reuse_completed"] is False


def test_manual_optimization_run_endpoint_enqueues_selected_mission(auth_client, tmp_path, monkeypatch):
    _reset_core_db(tmp_path, monkeypatch)
    mission = core_db.ensure_default_optimization_mission()
    from api.routers import optimization

    captured = {}

    def fake_enqueue_registered_job(job_type, payload, **kwargs):
        captured["job_type"] = job_type
        captured["payload"] = payload
        captured["kwargs"] = kwargs
        return (
            {
                "job_id": "manual-optimizer-job",
                "job_type": job_type,
                "status": "queued",
                "payload_json": payload,
                "result_json": None,
                "error": None,
                "progress_json": None,
            },
            "created",
        )

    monkeypatch.setattr(optimization, "enqueue_registered_job", fake_enqueue_registered_job)
    monkeypatch.setattr(
        optimization,
        "enqueue_response",
        lambda row, poll_path: JSONResponse({"job_id": row["job_id"], "status": "queued"}, status_code=202),
    )

    resp = auth_client.post(f"/api/v1/optimization/missions/{mission['id']}/run", json={"source": "manual"})

    assert resp.status_code == 202
    assert captured["job_type"] == "continuous_optimizer"
    assert captured["payload"]["mission_id"] == mission["id"]
    assert captured["payload"]["source"] == "manual"


def test_workspace_payload_includes_optimizer_alerts(auth_client, tmp_path, monkeypatch):
    _reset_core_db(tmp_path, monkeypatch)
    mission = core_db.ensure_default_optimization_mission()
    run = core_db.create_optimization_run(mission)
    snapshot = core_db.create_optimization_action_snapshot(
        {
            "run_id": run["run_id"],
            "mission_id": mission["id"],
            "ticker": "TSM",
            "asset": "equity",
            "direction": "long",
            "action": "Trim Long",
            "conviction_band": "medium",
            "priority_score": 2.0,
            "confidence": 0.7,
            "gate_status": "pass",
            "severity": "high",
            "state_hash": "state-1",
            "evidence": {"material_state": {"action": "Trim Long"}},
        }
    )
    core_db.create_optimization_alert(
        {
            "mission_id": mission["id"],
            "run_id": run["run_id"],
            "ticker": "TSM",
            "alert_type": "action_changed",
            "severity": "high",
            "current_snapshot_id": snapshot["id"],
            "change_summary": "TSM: action Hold Long -> Trim Long.",
            "evidence": {},
        }
    )
    import api.agent_tools as agent_tools

    monkeypatch.setattr(agent_tools, "execute_tool", lambda name, args: {"positions": []})
    monkeypatch.setattr(
        "api.signal_snapshot.get_signal_aggregator_snapshot_or_module_response",
        lambda **kwargs: {"regime": {"label": "neutral", "score": 0}},
    )

    resp = auth_client.get("/api/v1/workspace")

    assert resp.status_code == 200
    data = resp.json()
    assert data["continuous_optimization"]["open_alert_count"] == 1
    assert data["continuous_optimization"]["open_alerts"][0]["ticker"] == "TSM"
