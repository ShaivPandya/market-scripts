from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path


def test_async_jobs_migration_contract():
    migration = Path("migrations/versions/20260429_0002_async_jobs_rq.py").read_text(encoding="utf-8")
    event_migration = Path("migrations/versions/20260503_0014_async_job_events.py").read_text(encoding="utf-8")
    queue = Path("api/job_queue.py").read_text(encoding="utf-8")

    assert 'down_revision: str | None = "20260429_0001"' in migration
    assert "cache_key" in migration
    assert "progress_json" in migration
    assert "result_expires_at" in migration
    assert "uq_async_jobs_active_dedupe" in migration
    assert "status IN ('queued', 'running') AND cache_key IS NOT NULL" in migration
    assert "ON CONFLICT (job_type, cache_key)" in queue
    assert "DO NOTHING" in queue
    assert "async_job_events" in event_migration
    assert 'PrimaryKeyConstraint("job_id", "seq")' in event_migration


def test_async_job_storage_stays_local_when_backend_is_local(monkeypatch):
    from api.job_queue import postgres_jobs_enabled

    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "local")

    assert postgres_jobs_enabled() is False


def test_async_job_storage_defaults_local_in_development(monkeypatch):
    from api.job_queue import postgres_jobs_enabled

    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("ASYNC_JOB_BACKEND", raising=False)

    assert postgres_jobs_enabled() is False


def test_async_job_storage_defaults_postgres_in_production(monkeypatch):
    from api.job_queue import postgres_jobs_enabled

    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("ASYNC_JOB_BACKEND", raising=False)

    assert postgres_jobs_enabled() is True


def test_async_job_execution_defaults_local_without_cloud_run_opt_in(monkeypatch):
    from api import async_job_runner

    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("ASYNC_JOB_BACKEND", raising=False)
    monkeypatch.delenv("CLOUD_RUN_JOBS_ENABLED", raising=False)

    assert async_job_runner._env_backend() == "local"


def test_async_job_execution_uses_cloud_run_when_enabled(monkeypatch):
    from api import async_job_runner

    monkeypatch.delenv("ASYNC_JOB_BACKEND", raising=False)
    monkeypatch.setenv("CLOUD_RUN_JOBS_ENABLED", "true")

    assert async_job_runner._env_backend() == "cloud_run_jobs"


def test_cloud_run_jobs_enabled_matches_explicit_opt_in(monkeypatch):
    from api.cloud_run_jobs import cloud_run_jobs_enabled

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("ASYNC_JOB_BACKEND", raising=False)
    monkeypatch.delenv("CLOUD_RUN_JOBS_ENABLED", raising=False)

    assert cloud_run_jobs_enabled() is False

    monkeypatch.setenv("ASYNC_JOB_BACKEND", "cloud_run_jobs")
    monkeypatch.setenv("CLOUD_RUN_JOBS_ENABLED", "false")

    assert cloud_run_jobs_enabled() is True


def test_cloud_run_enqueue_dispatches_existing_job_once(monkeypatch):
    from api import async_job_runner, cache
    from api.job_queue import get_job

    cache.invalidate_all()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "cloud_run_jobs")
    dispatched: list[tuple[str, str]] = []
    monkeypatch.setattr(
        async_job_runner,
        "_enqueue_cloud_run_job",
        lambda job_type, job_id: dispatched.append((job_type, job_id)),
    )

    row, disposition = async_job_runner.enqueue_registered_job("analyzer", {}, cache_key="cloud-run-dispatch")

    assert disposition == "created"
    assert dispatched == [("analyzer", row["job_id"])]
    persisted = get_job(row["job_id"])
    assert persisted is not None
    assert persisted["status"] == "queued"


def test_agent_chat_warm_worker_dispatch_leaves_job_queued(monkeypatch):
    from api import async_job_runner, cache
    from api.job_queue import get_job

    cache.invalidate_all()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "cloud_run_jobs")
    monkeypatch.setenv("AGENT_CHAT_DISPATCH_BACKEND", "warm_worker")
    monkeypatch.setattr(
        async_job_runner,
        "_enqueue_cloud_run_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no Cloud Run dispatch")),
    )

    row, disposition = async_job_runner.enqueue_registered_job(
        "agent_chat_turn",
        {"session_id": "warm-session", "message": "hello", "client_turn_id": "warm-turn"},
        cache_key="agent-chat-warm-worker",
    )

    assert disposition == "created"
    persisted = get_job(row["job_id"])
    assert persisted is not None
    assert persisted["status"] == "queued"
    assert persisted["queue_name"] == "agent"


def test_claim_queued_agent_job_is_exclusive(monkeypatch):
    from api import cache
    from api.job_queue import claim_queued_job, create_or_reuse_job, get_job

    cache.invalidate_all()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "local")

    row, _disposition = create_or_reuse_job(
        "agent_chat_turn",
        payload={"session_id": "claim-session", "message": "hello"},
        cache_key="agent-claim",
        queue_name="agent",
    )

    first = claim_queued_job("agent_chat_turn", queue_name="agent")
    second = claim_queued_job("agent_chat_turn", queue_name="agent")

    assert first is not None
    assert first["job_id"] == row["job_id"]
    assert second is None
    assert get_job(row["job_id"])["status"] == "running"


def test_agent_worker_loop_claims_and_completes_one_job(monkeypatch):
    import api.agent_chat_worker as agent_chat_worker
    from api import async_job_runner, cache
    from api.agent_worker_loop import run_once
    from api.job_queue import get_job

    cache.invalidate_all()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "local")
    monkeypatch.setenv("AGENT_CHAT_DISPATCH_BACKEND", "warm_worker")

    monkeypatch.setattr(
        agent_chat_worker,
        "_run_agent_chat_turn_job",
        lambda req, *, job_id: {"status": "done", "session_id": req.session_id, "job_id": job_id},
    )
    row, _disposition = async_job_runner.enqueue_registered_job(
        "agent_chat_turn",
        {"session_id": "worker-session", "message": "hello"},
        cache_key="agent-worker-loop",
    )

    assert run_once() is True
    persisted = get_job(row["job_id"])
    assert persisted is not None
    assert persisted["status"] == "completed"


def test_cloud_run_dedupe_reuses_active_and_completed_jobs(monkeypatch):
    from api import async_job_runner, cache
    from api.job_queue import complete_job

    cache.invalidate_all()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "cloud_run_jobs")
    dispatched: list[str] = []
    monkeypatch.setattr(
        async_job_runner,
        "_enqueue_cloud_run_job",
        lambda _job_type, job_id: dispatched.append(job_id),
    )

    first, first_disposition = async_job_runner.enqueue_registered_job("analyzer", {}, cache_key="cloud-run-dedupe")
    second, second_disposition = async_job_runner.enqueue_registered_job("analyzer", {}, cache_key="cloud-run-dedupe")

    assert first_disposition == "created"
    assert second_disposition == "active"
    assert second["job_id"] == first["job_id"]
    assert dispatched == [first["job_id"]]

    complete_job(first["job_id"], {"ok": True})
    completed, completed_disposition = async_job_runner.enqueue_registered_job(
        "analyzer",
        {},
        cache_key="cloud-run-dedupe",
    )

    assert completed_disposition == "completed"
    assert completed["job_id"] == first["job_id"]
    assert completed["result_json"] == {"ok": True}
    assert dispatched == [first["job_id"]]


def test_cloud_run_dispatch_failure_marks_job_failed(monkeypatch):
    import pytest

    from api import async_job_runner, cache
    from api.exceptions import AsyncJobDispatchError
    from api.job_queue import get_job

    cache.invalidate_all()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "cloud_run_jobs")
    attempted: list[str] = []

    def fail_dispatch(_job_type, job_id):
        attempted.append(job_id)
        raise RuntimeError("run api unavailable")

    monkeypatch.setattr(async_job_runner, "_enqueue_cloud_run_job", fail_dispatch)

    with pytest.raises(AsyncJobDispatchError, match="run api unavailable"):
        async_job_runner.enqueue_registered_job("analyzer", {}, cache_key="cloud-run-fail")

    failed = get_job(attempted[0])
    assert failed is not None
    assert failed["status"] == "failed"
    assert "run api unavailable" in failed["error"]


def test_perform_job_marks_running_before_request_parse(monkeypatch):
    import pytest

    from api import async_job_runner, cache
    from api.job_queue import create_or_reuse_job, get_job

    cache.invalidate_all()
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "local")
    row, _disposition = create_or_reuse_job("analyzer", payload={}, cache_key="parse-order")
    observed: dict[str, str] = {}

    def fail_parse(_spec, _payload):
        observed["status_during_parse"] = str((get_job(row["job_id"]) or {}).get("status") or "")
        raise RuntimeError("parse boom")

    monkeypatch.setattr(async_job_runner, "parse_request", fail_parse)

    with pytest.raises(RuntimeError, match="parse boom"):
        async_job_runner.perform_job(row["job_id"])

    assert observed["status_during_parse"] == "running"
    assert get_job(row["job_id"])["status"] == "failed"


def test_dispatch_cloud_run_job_invokes_generic_runner(monkeypatch):
    from api import cache
    from api.cloud_run_jobs import dispatch_cloud_run_job
    from api.job_queue import create_or_reuse_job, get_job

    cache.invalidate_all()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "local")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("CLOUD_RUN_REGION", "us-central1")
    monkeypatch.setenv("ASYNC_CLOUD_RUN_JOB", "talisman-async-job")
    row, _disposition = create_or_reuse_job("analyzer", payload={}, cache_key="cloud-run-helper")
    calls: list[dict] = []

    class FakeResponse:
        def raise_for_status(self):
            return None

    class FakeSession:
        def __init__(self, credentials):
            self.credentials = credentials

        def post(self, url, *, json, timeout):
            calls.append({"url": url, "json": json, "timeout": timeout, "credentials": self.credentials})
            return FakeResponse()

    monkeypatch.setattr("google.auth.default", lambda scopes: ("creds", "ignored-project"))
    monkeypatch.setattr("google.auth.transport.requests.AuthorizedSession", FakeSession)

    job_name = dispatch_cloud_run_job("analyzer", row["job_id"])

    assert job_name == "talisman-async-job"
    assert get_job(row["job_id"])["cloud_run_job_name"] == "talisman-async-job"
    assert calls[0]["url"] == (
        "https://run.googleapis.com/v2/projects/test-project/locations/us-central1/jobs/talisman-async-job:run"
    )
    env = calls[0]["json"]["overrides"]["containerOverrides"][0]["env"]
    assert {"name": "ASYNC_JOB_ID", "value": row["job_id"]} in env
    assert {"name": "ASYNC_JOB_TYPE", "value": "analyzer"} in env


def test_dispatch_cloud_run_job_uses_adc_project_when_env_unset(monkeypatch):
    from api import cache
    from api.cloud_run_jobs import dispatch_cloud_run_job
    from api.job_queue import create_or_reuse_job

    cache.invalidate_all()
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "local")
    monkeypatch.setenv("CLOUD_RUN_REGION", "us-central1")
    monkeypatch.setenv("ASYNC_CLOUD_RUN_JOB", "talisman-async-job")
    row, _disposition = create_or_reuse_job("analyzer", payload={}, cache_key="cloud-run-adc-project")
    calls: list[dict] = []

    class FakeResponse:
        def raise_for_status(self):
            return None

    class FakeSession:
        def __init__(self, credentials):
            self.credentials = credentials

        def post(self, url, *, json, timeout):
            calls.append({"url": url, "json": json, "timeout": timeout})
            return FakeResponse()

    monkeypatch.setattr("google.auth.default", lambda scopes: ("creds", "adc-project"))
    monkeypatch.setattr("google.auth.transport.requests.AuthorizedSession", FakeSession)

    dispatch_cloud_run_job("analyzer", row["job_id"])

    assert calls[0]["url"] == (
        "https://run.googleapis.com/v2/projects/adc-project/locations/us-central1/jobs/talisman-async-job:run"
    )


def test_async_job_runner_cli_completes_job(monkeypatch):
    from api import async_job_runner, cache
    from api.job_queue import create_or_reuse_job, get_job
    from api.routers import analyzer

    cache.invalidate_all()
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "local")
    monkeypatch.setattr(analyzer, "_compute_analyzer_result", lambda _req: {"ok": True})
    row, _disposition = create_or_reuse_job("analyzer", payload={}, cache_key="cli-complete", queue_name="default")

    assert async_job_runner.main(["run", row["job_id"]]) == 0

    persisted = get_job(row["job_id"])
    assert persisted is not None
    assert persisted["status"] == "completed"
    assert persisted["result_json"] == {"ok": True}


def test_async_job_runner_cli_marks_failure(monkeypatch):
    from api import async_job_runner, cache
    from api.job_queue import create_or_reuse_job, get_job
    from api.routers import analyzer

    cache.invalidate_all()
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "local")

    def fail_compute(_req):
        raise RuntimeError("worker crashed")

    monkeypatch.setattr(analyzer, "_compute_analyzer_result", fail_compute)
    row, _disposition = create_or_reuse_job("analyzer", payload={}, cache_key="cli-fail", queue_name="default")

    assert async_job_runner.main(["run", row["job_id"]]) == 1

    persisted = get_job(row["job_id"])
    assert persisted is not None
    assert persisted["status"] == "failed"
    assert "worker crashed" in persisted["error"]


def test_stale_active_job_is_failed_and_no_longer_blocks_dedupe(monkeypatch):
    from api import async_job_runner, cache
    from api.job_queue import get_job
    from api.job_registry import get_job_spec

    cache.invalidate_all()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "cloud_run_jobs")
    monkeypatch.setenv("ASYNC_JOB_STALE_GRACE_SECONDS", "0")
    dispatched: list[str] = []
    monkeypatch.setattr(
        async_job_runner,
        "_enqueue_cloud_run_job",
        lambda _job_type, job_id: dispatched.append(job_id),
    )

    first, _disposition = async_job_runner.enqueue_registered_job("analyzer", {}, cache_key="stale-job")
    stale_count = async_job_runner.fail_stale_active_jobs(
        datetime.now(UTC) + timedelta(seconds=get_job_spec("analyzer").timeout_s + 1)
    )

    assert stale_count == 1
    assert get_job(first["job_id"])["status"] == "failed"

    second, second_disposition = async_job_runner.enqueue_registered_job("analyzer", {}, cache_key="stale-job")

    assert second_disposition == "created"
    assert second["job_id"] != first["job_id"]
    assert dispatched == [first["job_id"], second["job_id"]]


def test_local_async_jobs_dedupe_concurrent_active(monkeypatch):
    from api import cache
    from api.async_job_runner import enqueue_registered_job, poll_registered_job
    from api.routers import analyzer

    cache.invalidate_all()
    started = threading.Event()
    release = threading.Event()

    def slow_compute(_req):
        started.set()
        assert release.wait(timeout=2)
        return {"ok": True}

    monkeypatch.setattr(analyzer, "_compute_analyzer_result", slow_compute)

    body = {}
    cache_key = analyzer._cache_key(analyzer.AnalyzerRequest())
    rows: list[dict] = []

    def enqueue():
        row, _disposition = enqueue_registered_job("analyzer", body, cache_key=cache_key)
        rows.append(row)

    threads = [threading.Thread(target=enqueue) for _ in range(4)]
    for thread in threads:
        thread.start()

    assert started.wait(timeout=2)
    for thread in threads:
        thread.join(timeout=2)

    job_ids = {row["job_id"] for row in rows}
    assert len(job_ids) == 1

    release.set()
    job_id = next(iter(job_ids))
    deadline = time.time() + 2
    while time.time() < deadline:
        payload = poll_registered_job(job_id)
        if payload["status"] == "done":
            assert payload["result"] == {"ok": True}
            return
        time.sleep(0.05)
    raise AssertionError("job did not complete")


def test_sweep_expired_local_jobs():
    from api import cache
    from api.job_queue import complete_job, create_or_reuse_job, get_job, sweep_expired_jobs

    cache.invalidate_all()
    row, disposition = create_or_reuse_job(
        "analyzer",
        payload={},
        cache_key="expired-job",
        queue_name="default",
    )
    assert disposition == "created"
    complete_job(row["job_id"], {"ok": True}, result_ttl_seconds=0)

    deleted = sweep_expired_jobs(datetime.now(UTC) + timedelta(seconds=1))

    assert deleted == 1
    assert get_job(row["job_id"]) is None


def test_core_async_endpoints_use_persisted_job_contract(auth_client, monkeypatch):
    from api import cache
    from api.job_registry import get_job_spec
    from api.routers import analyzer, fundamental_momentum, hedging, sizer

    cache.invalidate_all()
    cases = [
        (analyzer, "analyzer", "_compute_analyzer_result", "/api/v1/portfolio-analyzer/async", {}),
        (
            hedging,
            "hedging",
            "_compute_hedging_result",
            "/api/v1/hedging-tool/async",
            {"book": 100000, "positions": [{"ticker": "AAA", "weight": 0.1}]},
        ),
        (
            sizer,
            "sizer",
            "_compute_sizer_result",
            "/api/v1/portfolio-sizer/async",
            {"book": 100000, "target_leverage": 2.0, "positions": [{"ticker": "AAA", "conviction": 3}]},
        ),
        (
            fundamental_momentum,
            "fundamental_momentum",
            "_compute_fundamental_momentum",
            "/api/v1/fundamental-momentum/async",
            {"input_mode": "Custom Tickers", "tickers": "AAA", "screen_type": "EPS"},
        ),
    ]

    for module, job_type, attr, path, body in cases:
        monkeypatch.setattr(module, attr, lambda _req, label=attr: {"ok": label})
        started = auth_client.post(path, json=body)
        assert started.status_code in (200, 202)
        payload = started.json()
        job_id = payload["job_id"]
        assert payload["timeout_s"] == get_job_spec(job_type).timeout_s

        deadline = time.time() + 2
        while time.time() < deadline:
            done = auth_client.get(f"{path}/{job_id}").json()
            assert done["timeout_s"] == get_job_spec(job_type).timeout_s
            if done["status"] == "done":
                assert done["result"]["ok"] == attr
                break
            time.sleep(0.05)
        else:
            raise AssertionError(f"{path} did not complete")


def test_fundamental_momentum_dispatch_error_returns_structured_503(auth_client, monkeypatch):
    from api.exceptions import AsyncJobDispatchError
    from api.routers import fundamental_momentum

    def fail_enqueue(*_args, **_kwargs):
        raise AsyncJobDispatchError("run api unavailable")

    monkeypatch.setattr(fundamental_momentum, "enqueue_registered_job", fail_enqueue)

    resp = auth_client.post(
        "/api/v1/fundamental-momentum/async",
        json={
            "input_mode": "Custom Tickers",
            "tickers": "FTI, INVX",
            "screen_type": "Both",
            "benchmark": "Same as Input",
        },
    )

    assert resp.status_code == 503
    assert resp.json()["error"] == "Async job dispatch failed: run api unavailable"
