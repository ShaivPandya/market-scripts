from fastapi.responses import JSONResponse


def test_scheduler_secret_can_enqueue_maintenance_job(client, monkeypatch):
    from api.routers import admin_jobs

    monkeypatch.setenv("SCHEDULER_SECRET", "scheduler-secret")

    def fake_enqueue_registered_job(job_type, payload, **kwargs):
        return (
            {
                "job_id": "maintenance-job",
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
        "/api/admin/jobs/enqueue-cache-warm",
        headers={"X-Scheduler-Secret": "scheduler-secret"},
    )

    assert resp.status_code == 202
    assert resp.json() == {"job_id": "maintenance-job", "status": "queued"}


def test_scheduler_secret_can_enqueue_macro_snapshot_job(client, monkeypatch):
    from api.routers import admin_jobs

    monkeypatch.setenv("SCHEDULER_SECRET", "scheduler-secret")
    seen: dict[str, object] = {}

    def fake_enqueue_registered_job(job_type, payload, **kwargs):
        seen.update({"job_type": job_type, "payload": payload, **kwargs})
        return (
            {
                "job_id": "macro-snapshot-job",
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
        "/api/admin/jobs/enqueue-macro-snapshot-refresh",
        headers={"X-Scheduler-Secret": "scheduler-secret"},
    )

    assert resp.status_code == 202
    assert resp.json() == {"job_id": "macro-snapshot-job", "status": "queued"}
    assert seen["job_type"] == "macro_snapshot_refresh"
    assert seen["cache_key"] == "maintenance:macro_snapshot_refresh:v1"
    assert seen["reuse_completed"] is False


def test_scheduler_secret_can_enqueue_workspace_source_refresh_job(client, monkeypatch):
    from api.routers import admin_jobs

    monkeypatch.setenv("SCHEDULER_SECRET", "scheduler-secret")
    seen: dict[str, object] = {}

    def fake_enqueue_registered_job(job_type, payload, **kwargs):
        seen.update({"job_type": job_type, "payload": payload, **kwargs})
        return (
            {
                "job_id": "workspace-source-job",
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
        "/api/admin/jobs/enqueue-workspace-source-refresh",
        headers={"X-Scheduler-Secret": "scheduler-secret"},
    )

    assert resp.status_code == 202
    assert resp.json() == {"job_id": "workspace-source-job", "status": "queued"}
    assert seen["job_type"] == "workspace_source_refresh"
    assert seen["cache_key"] == "maintenance:workspace_source_refresh:v1"
    assert seen["reuse_completed"] is False


def test_scheduler_secret_can_enqueue_monitor_mission_runner(client, monkeypatch):
    from api.routers import admin_jobs

    monkeypatch.setenv("SCHEDULER_SECRET", "scheduler-secret")
    seen: dict[str, object] = {}

    def fake_enqueue_registered_job(job_type, payload, **kwargs):
        seen.update({"job_type": job_type, "payload": payload, **kwargs})
        return (
            {
                "job_id": "monitor-mission-job",
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
        "/api/admin/jobs/enqueue-monitor-mission-runner",
        headers={"X-Scheduler-Secret": "scheduler-secret"},
    )

    assert resp.status_code == 202
    assert resp.json() == {"job_id": "monitor-mission-job", "status": "queued"}
    assert seen["job_type"] == "monitor_mission_runner"
    assert seen["cache_key"] == "maintenance:monitor_mission_runner:v1"
    assert seen["reuse_completed"] is False


def test_scheduler_secret_cannot_poll_arbitrary_job_result(client, monkeypatch):
    from api.job_queue import clear_memory_jobs, complete_job, create_job

    monkeypatch.setenv("SCHEDULER_SECRET", "scheduler-secret")
    clear_memory_jobs()
    create_job("analyzer", payload={"private": "payload"}, job_id="known-job")
    complete_job("known-job", {"secret_result": "value"})

    resp = client.get(
        "/api/admin/jobs/known-job",
        headers={"X-Scheduler-Secret": "scheduler-secret"},
    )

    assert resp.status_code == 401


def test_authenticated_admin_can_poll_job_result(auth_client):
    from api.job_queue import clear_memory_jobs, complete_job, create_job

    clear_memory_jobs()
    create_job("analyzer", payload={"private": "payload"}, job_id="known-job")
    complete_job("known-job", {"secret_result": "value"})

    resp = auth_client.get("/api/admin/jobs/known-job")

    assert resp.status_code == 200
    assert resp.json()["result"] == {"secret_result": "value"}


def test_admin_enqueue_dispatch_error_returns_structured_503(auth_client, monkeypatch):
    from api.exceptions import AsyncJobDispatchError
    from api.routers import admin_jobs

    def fail_enqueue(*_args, **_kwargs):
        raise AsyncJobDispatchError("run api unavailable")

    monkeypatch.setattr(admin_jobs, "enqueue_registered_job", fail_enqueue)

    resp = auth_client.post("/api/admin/jobs/enqueue-market-snapshot-refresh")

    assert resp.status_code == 503
    # Sentinel: The message is now generic to prevent leakage, details are in 'detail' (redacted in prod)
    assert resp.json()["error"] == "Async job dispatch failed"
    assert resp.json()["detail"] == "run api unavailable"
