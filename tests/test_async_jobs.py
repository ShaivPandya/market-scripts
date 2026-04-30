from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path


def test_async_jobs_migration_contract():
    migration = Path("migrations/versions/20260429_0002_async_jobs_rq.py").read_text(encoding="utf-8")
    queue = Path("api/job_queue.py").read_text(encoding="utf-8")

    assert 'down_revision: str | None = "20260429_0001"' in migration
    assert "cache_key" in migration
    assert "progress_json" in migration
    assert "result_expires_at" in migration
    assert "uq_async_jobs_active_dedupe" in migration
    assert "status IN ('queued', 'running') AND cache_key IS NOT NULL" in migration
    assert "ON CONFLICT (job_type, cache_key)" in queue
    assert "DO NOTHING" in queue


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
    from api.routers import analyzer, fundamental_momentum, hedging, sizer

    cache.invalidate_all()
    cases = [
        (analyzer, "_compute_analyzer_result", "/api/v1/portfolio-analyzer/async", {}),
        (
            hedging,
            "_compute_hedging_result",
            "/api/v1/hedging-tool/async",
            {"book": 100000, "positions": [{"ticker": "AAA", "weight": 0.1}]},
        ),
        (
            sizer,
            "_compute_sizer_result",
            "/api/v1/portfolio-sizer/async",
            {"book": 100000, "target_leverage": 2.0, "positions": [{"ticker": "AAA", "conviction": 3}]},
        ),
        (
            fundamental_momentum,
            "_compute_fundamental_momentum",
            "/api/v1/fundamental-momentum/async",
            {"input_mode": "Custom Tickers", "tickers": "AAA", "screen_type": "EPS"},
        ),
    ]

    for module, attr, path, body in cases:
        monkeypatch.setattr(module, attr, lambda _req, label=attr: {"ok": label})
        started = auth_client.post(path, json=body)
        assert started.status_code in (200, 202)
        payload = started.json()
        job_id = payload["job_id"]

        deadline = time.time() + 2
        while time.time() < deadline:
            done = auth_client.get(f"{path}/{job_id}").json()
            if done["status"] == "done":
                assert done["result"]["ok"] == attr
                break
            time.sleep(0.05)
        else:
            raise AssertionError(f"{path} did not complete")
