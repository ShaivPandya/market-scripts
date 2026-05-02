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
        "/api/v1/admin/jobs/enqueue-cache-warm",
        headers={"X-Scheduler-Secret": "scheduler-secret"},
    )

    assert resp.status_code == 202
    assert resp.json() == {"job_id": "maintenance-job", "status": "queued"}


def test_scheduler_secret_cannot_poll_arbitrary_job_result(client, monkeypatch):
    from api.job_queue import clear_memory_jobs, complete_job, create_job

    monkeypatch.setenv("SCHEDULER_SECRET", "scheduler-secret")
    clear_memory_jobs()
    create_job("analyzer", payload={"private": "payload"}, job_id="known-job")
    complete_job("known-job", {"secret_result": "value"})

    resp = client.get(
        "/api/v1/admin/jobs/known-job",
        headers={"X-Scheduler-Secret": "scheduler-secret"},
    )

    assert resp.status_code == 401


def test_authenticated_admin_can_poll_job_result(auth_client):
    from api.job_queue import clear_memory_jobs, complete_job, create_job

    clear_memory_jobs()
    create_job("analyzer", payload={"private": "payload"}, job_id="known-job")
    complete_job("known-job", {"secret_result": "value"})

    resp = auth_client.get("/api/v1/admin/jobs/known-job")

    assert resp.status_code == 200
    assert resp.json()["result"] == {"secret_result": "value"}
