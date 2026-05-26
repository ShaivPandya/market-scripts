from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path


def test_async_jobs_migration_contract():
    migration = Path("migrations/versions/20260429_0002_async_jobs_rq.py").read_text(encoding="utf-8")
    event_migration = Path("migrations/versions/20260503_0014_async_job_events.py").read_text(encoding="utf-8")
    freshness_migration = Path("migrations/versions/20260505_0008_async_job_freshness_ttls.py").read_text(
        encoding="utf-8"
    )
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
    assert "idx_ontology_object_versions_watermark" in freshness_migration
    assert "idx_ontology_relation_versions_watermark" in freshness_migration
    assert "idx_computed_snapshot_versions_watermark" in freshness_migration
    assert "idx_source_record_versions_watermark" in freshness_migration
    assert "job_type IN ('analyzer', 'sizer', 'hedging')" in freshness_migration
    assert "payload_json->>'run_id'" in freshness_migration
    assert "payload_json->>'as_of'" in freshness_migration
    assert "payload_json->>'tx_as_of'" in freshness_migration


def test_idea_evaluation_job_registered_with_progress():
    from api.job_registry import get_job_spec

    spec = get_job_spec("idea_evaluation")

    assert spec.request_model == "api.routers.ideas.IdeaEvaluationRequest"
    assert spec.compute_func == "api.routers.ideas._compute_idea_evaluation_result"
    assert spec.cache_key_func == "api.routers.ideas._cache_key"
    assert spec.supports_progress is True


def test_idea_comparison_evaluation_job_registered_with_progress():
    from api.job_registry import get_job_spec

    spec = get_job_spec("idea_comparison_evaluation")

    assert spec.request_model == "api.routers.ideas.IdeaComparisonEvaluationRequest"
    assert spec.compute_func == "api.routers.ideas._compute_idea_comparison_evaluation_result"
    assert spec.cache_key_func is None
    assert spec.supports_progress is True


def test_p0_async_job_completed_ttl_policy_defaults():
    from api.job_registry import completed_ttl_for_request, get_job_spec
    from api.routers.ontology import OntologyQueryJobRequest

    assert get_job_spec("analyzer").completed_ttl_s == 24 * 60 * 60
    assert get_job_spec("sizer").completed_ttl_s == 300
    assert get_job_spec("sizer").timeout_s == 3 * 60
    assert get_job_spec("sizer").stale_grace_s is None
    assert get_job_spec("hedging").completed_ttl_s == 300

    ontology_spec = get_job_spec("ontology")
    current = OntologyQueryJobRequest(schema_mode="upgraded", actor={})
    replay = OntologyQueryJobRequest(schema_mode="upgraded", actor={}, run_id="historical-run")

    assert completed_ttl_for_request(ontology_spec, current) == 60
    assert completed_ttl_for_request(ontology_spec, replay) == 24 * 60 * 60


def test_p0_async_job_completed_ttl_policy_env_overrides(monkeypatch):
    import importlib

    import api.job_registry as registry
    from api.routers.ontology import OntologyQueryJobRequest

    monkeypatch.setenv("ASYNC_ANALYZER_COMPLETED_TTL_SECONDS", "11")
    monkeypatch.setenv("ASYNC_SIZER_COMPLETED_TTL_SECONDS", "22")
    monkeypatch.setenv("ASYNC_TIMEOUT_SIZER_SECONDS", "222")
    monkeypatch.setenv("ASYNC_STALE_GRACE_SIZER_SECONDS", "66")
    monkeypatch.setenv("ASYNC_HEDGING_COMPLETED_TTL_SECONDS", "33")
    monkeypatch.setenv("ASYNC_ONTOLOGY_CURRENT_COMPLETED_TTL_SECONDS", "44")
    monkeypatch.setenv("ASYNC_ONTOLOGY_REPLAY_COMPLETED_TTL_SECONDS", "55")
    registry = importlib.reload(registry)
    try:
        assert registry.get_job_spec("analyzer").completed_ttl_s == 11
        assert registry.get_job_spec("sizer").completed_ttl_s == 22
        assert registry.get_job_spec("sizer").timeout_s == 222
        assert registry.get_job_spec("sizer").stale_grace_s == 66
        assert registry.get_job_spec("hedging").completed_ttl_s == 33

        ontology_spec = registry.get_job_spec("ontology")
        current = OntologyQueryJobRequest(schema_mode="upgraded", actor={})
        replay = OntologyQueryJobRequest(schema_mode="upgraded", actor={}, tx_as_of="2026-05-01T00:00:00Z")

        assert registry.completed_ttl_for_request(ontology_spec, current) == 44
        assert registry.completed_ttl_for_request(ontology_spec, replay) == 55
    finally:
        monkeypatch.delenv("ASYNC_ANALYZER_COMPLETED_TTL_SECONDS", raising=False)
        monkeypatch.delenv("ASYNC_SIZER_COMPLETED_TTL_SECONDS", raising=False)
        monkeypatch.delenv("ASYNC_TIMEOUT_SIZER_SECONDS", raising=False)
        monkeypatch.delenv("ASYNC_STALE_GRACE_SIZER_SECONDS", raising=False)
        monkeypatch.delenv("ASYNC_HEDGING_COMPLETED_TTL_SECONDS", raising=False)
        monkeypatch.delenv("ASYNC_ONTOLOGY_CURRENT_COMPLETED_TTL_SECONDS", raising=False)
        monkeypatch.delenv("ASYNC_ONTOLOGY_REPLAY_COMPLETED_TTL_SECONDS", raising=False)
        importlib.reload(registry)


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


def test_poll_registered_job_suppresses_success_read_audit_by_default(monkeypatch):
    from api import async_job_runner

    audits: list[str] = []
    row = {"job_id": "job-1", "job_type": "analyzer", "status": "running", "progress_json": {}}

    monkeypatch.delenv("ASYNC_JOB_SUCCESS_READ_AUDIT_ENABLED", raising=False)
    monkeypatch.setattr(async_job_runner, "get_job", lambda job_id: row if job_id == "job-1" else None)
    monkeypatch.setattr(async_job_runner, "_sync_stale_active_job", lambda loaded: loaded)
    monkeypatch.setattr(
        async_job_runner,
        "_emit_job_audit",
        lambda action_name, **_kwargs: audits.append(action_name),
    )

    payload = async_job_runner.poll_registered_job("job-1")

    assert payload["status"] == "running"
    assert audits == []


def test_poll_registered_job_success_read_audit_can_be_reenabled(monkeypatch):
    from api import async_job_runner

    audits: list[str] = []
    row = {"job_id": "job-1", "job_type": "analyzer", "status": "running", "progress_json": {}}

    monkeypatch.setenv("ASYNC_JOB_SUCCESS_READ_AUDIT_ENABLED", "true")
    monkeypatch.setattr(async_job_runner, "get_job", lambda job_id: row if job_id == "job-1" else None)
    monkeypatch.setattr(async_job_runner, "_sync_stale_active_job", lambda loaded: loaded)
    monkeypatch.setattr(
        async_job_runner,
        "_emit_job_audit",
        lambda action_name, **_kwargs: audits.append(action_name),
    )

    payload = async_job_runner.poll_registered_job("job-1")

    assert payload["status"] == "running"
    assert audits == ["async_job.read"]


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


def test_claim_queued_job_postgres_queue_filter_uses_typed_comparison(monkeypatch):
    from api import job_queue

    calls: list[tuple[str, tuple[object, ...]]] = []

    class _Cursor:
        def fetchone(self):
            return {
                "job_id": "job-1",
                "job_type": "agent_chat_turn",
                "queue_name": "agent",
                "status": "running",
            }

    class _Conn:
        def execute(self, sql: str, params: tuple[object, ...]):
            calls.append((sql, params))
            return _Cursor()

        def commit(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(job_queue, "postgres_jobs_enabled", lambda: True)
    monkeypatch.setattr(job_queue, "connect", lambda: _Conn())

    claimed = job_queue.claim_queued_job("agent_chat_turn", queue_name="agent")

    assert claimed is not None
    sql, params = calls[-1]
    assert "%s IS NULL" not in sql
    assert "queue_name = %s" in sql
    assert len(params) == 4
    assert params[:2] == ("agent_chat_turn", "agent")


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


def test_analyzer_warm_worker_dispatch_leaves_job_queued(monkeypatch):
    from api import async_job_runner, cache
    from api.job_queue import get_job

    cache.invalidate_all()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "cloud_run_jobs")
    monkeypatch.setenv("ASYNC_DISPATCH_BACKEND_ANALYZER", "warm_worker")
    monkeypatch.setattr(
        async_job_runner,
        "_enqueue_cloud_run_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no Cloud Run dispatch")),
    )

    row, disposition = async_job_runner.enqueue_registered_job(
        "analyzer",
        {},
        cache_key="analyzer-warm-worker",
    )

    assert disposition == "created"
    persisted = get_job(row["job_id"])
    assert persisted is not None
    assert persisted["status"] == "queued"
    assert persisted["queue_name"] == "analyzer"


def test_sizer_inline_dispatch_completes_without_cloud_run_job(monkeypatch):
    from api import async_job_runner, cache
    from api.job_queue import get_job
    from api.routers import sizer

    cache.invalidate_all()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "cloud_run_jobs")
    monkeypatch.setenv("ASYNC_DISPATCH_BACKEND_SIZER", "inline")
    monkeypatch.setattr(sizer, "_compute_sizer_result", lambda req: {"ok": req.positions[0].ticker})
    monkeypatch.setattr(
        async_job_runner,
        "_enqueue_cloud_run_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no Cloud Run dispatch")),
    )

    row, disposition = async_job_runner.enqueue_registered_job(
        "sizer",
        {"book": 100000, "target_leverage": 2.0, "positions": [{"ticker": "AAA", "conviction": 3}]},
        cache_key="sizer-warm-worker",
    )

    assert disposition == "created"
    persisted = get_job(row["job_id"])
    assert persisted is not None
    assert persisted["status"] == "completed"
    assert persisted["result_json"] == {"ok": "AAA"}
    assert persisted["queue_name"] == "sizer"


def test_ontology_warm_worker_dispatch_leaves_job_queued(monkeypatch):
    from api import async_job_runner, cache
    from api.job_queue import get_job

    cache.invalidate_all()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "cloud_run_jobs")
    monkeypatch.setenv("ASYNC_DISPATCH_BACKEND_ONTOLOGY", "warm_worker")
    monkeypatch.setattr(
        async_job_runner,
        "_enqueue_cloud_run_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no Cloud Run dispatch")),
    )

    row, disposition = async_job_runner.enqueue_registered_job(
        "ontology",
        {"query": "risk", "schema_mode": "stored"},
        cache_key="ontology-warm-worker",
    )

    assert disposition == "created"
    persisted = get_job(row["job_id"])
    assert persisted is not None
    assert persisted["status"] == "queued"
    assert persisted["queue_name"] == "ontology"


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


def test_generic_worker_loop_can_claim_and_complete_sizer_job(monkeypatch):
    from api import cache
    from api.job_queue import create_or_reuse_job, get_job
    from api.job_worker_loop import run_once
    from api.routers import sizer

    cache.invalidate_all()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(sizer, "_compute_sizer_result", lambda req: {"ok": req.positions[0].ticker})

    row, _disposition = create_or_reuse_job(
        "sizer",
        payload={"book": 100000, "target_leverage": 2.0, "positions": [{"ticker": "AAA", "conviction": 3}]},
        cache_key="sizer-worker-loop",
        queue_name="sizer",
    )

    assert run_once(job_type="sizer", queue_name="sizer") is True
    persisted = get_job(row["job_id"])
    assert persisted is not None
    assert persisted["status"] == "completed"
    assert persisted["result_json"] == {"ok": "AAA"}


def test_generic_worker_loop_claims_and_completes_analyzer_job(monkeypatch):
    from api import async_job_runner, cache
    from api.job_queue import get_job
    from api.job_worker_loop import run_once
    from api.routers import analyzer

    cache.invalidate_all()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "cloud_run_jobs")
    monkeypatch.setenv("ASYNC_DISPATCH_BACKEND_ANALYZER", "warm_worker")
    monkeypatch.setattr(analyzer, "_compute_analyzer_result", lambda req: {"ok": req.scenario is None})

    row, _disposition = async_job_runner.enqueue_registered_job(
        "analyzer",
        {},
        cache_key="analyzer-worker-loop",
    )

    assert run_once(job_type="analyzer", queue_name="analyzer") is True
    persisted = get_job(row["job_id"])
    assert persisted is not None
    assert persisted["status"] == "completed"
    assert persisted["result_json"] == {"ok": True}


def test_generic_worker_loop_claims_and_completes_ontology_job(monkeypatch):
    from api import async_job_runner, cache
    from api.job_queue import get_job
    from api.job_worker_loop import run_once
    from api.routers import ontology

    cache.invalidate_all()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "cloud_run_jobs")
    monkeypatch.setenv("ASYNC_DISPATCH_BACKEND_ONTOLOGY", "warm_worker")
    monkeypatch.setattr(ontology, "_execute_query", lambda req: {"ok": req.query})

    row, _disposition = async_job_runner.enqueue_registered_job(
        "ontology",
        {"query": "risk", "schema_mode": "stored"},
        cache_key="ontology-worker-loop",
    )

    assert run_once(job_type="ontology", queue_name="ontology") is True
    persisted = get_job(row["job_id"])
    assert persisted is not None
    assert persisted["status"] == "completed"
    assert persisted["result_json"] == {"ok": "risk"}


def test_generic_worker_loop_parser_reads_job_defaults_from_env(monkeypatch):
    from api.job_worker_loop import _parser

    monkeypatch.setenv("JOB_WORKER_JOB_TYPE", "sizer")
    monkeypatch.setenv("JOB_WORKER_QUEUE", "sizer")

    args = _parser().parse_args(["run"])

    assert args.job_type == "sizer"
    assert args.queue_name == "sizer"


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

    with pytest.raises(AsyncJobDispatchError) as exc_info:
        async_job_runner.enqueue_registered_job("analyzer", {}, cache_key="cloud-run-fail")
    assert exc_info.value.detail == "run api unavailable"

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


def test_perform_job_uses_p0_completed_result_ttl(monkeypatch):
    from api import async_job_runner, cache
    from api.job_queue import clear_memory_jobs, create_or_reuse_job, get_job
    from api.routers import analyzer

    cache.invalidate_all()
    clear_memory_jobs()
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "local")
    monkeypatch.setattr(analyzer, "_compute_analyzer_result", lambda _req: {"ok": True})

    row, disposition = create_or_reuse_job("analyzer", payload={}, cache_key="p0-ttl")

    assert disposition == "created"
    async_job_runner.perform_job(str(row["job_id"]))

    completed = get_job(str(row["job_id"]))
    assert completed is not None
    assert completed["status"] == "completed"
    delta = completed["result_expires_at"] - completed["completed_at"]
    assert 86399 <= delta.total_seconds() <= 86401


def test_analyzer_ttl_follows_generic_completed_ttl_when_unset(monkeypatch):
    import importlib

    import api.job_registry as registry

    monkeypatch.delenv("ASYNC_ANALYZER_COMPLETED_TTL_SECONDS", raising=False)
    monkeypatch.setenv("ASYNC_JOB_COMPLETED_TTL_SECONDS", "43200")
    reloaded = importlib.reload(registry)

    assert reloaded.get_job_spec("analyzer").completed_ttl_s == 43200

    monkeypatch.delenv("ASYNC_JOB_COMPLETED_TTL_SECONDS", raising=False)
    importlib.reload(registry)


def test_analyzer_ttl_specific_env_overrides_generic(monkeypatch):
    import importlib

    import api.job_registry as registry

    monkeypatch.setenv("ASYNC_JOB_COMPLETED_TTL_SECONDS", "43200")
    monkeypatch.setenv("ASYNC_ANALYZER_COMPLETED_TTL_SECONDS", "86400")
    reloaded = importlib.reload(registry)

    assert reloaded.get_job_spec("analyzer").completed_ttl_s == 86400

    monkeypatch.delenv("ASYNC_JOB_COMPLETED_TTL_SECONDS", raising=False)
    monkeypatch.delenv("ASYNC_ANALYZER_COMPLETED_TTL_SECONDS", raising=False)
    importlib.reload(registry)


def test_completed_job_reuse_respects_expiry(monkeypatch):
    from api import cache
    from api.job_queue import clear_memory_jobs, complete_job, create_or_reuse_job

    cache.invalidate_all()
    clear_memory_jobs()
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "local")

    row, disposition = create_or_reuse_job("analyzer", payload={}, cache_key="reuse-before-expiry")
    assert disposition == "created"
    complete_job(str(row["job_id"]), {"ok": True}, result_ttl_seconds=300)

    reused, reused_disposition = create_or_reuse_job("analyzer", payload={}, cache_key="reuse-before-expiry")

    assert reused_disposition == "completed"
    assert reused["job_id"] == row["job_id"]

    expired, disposition = create_or_reuse_job("analyzer", payload={}, cache_key="reuse-after-expiry")
    assert disposition == "created"
    complete_job(str(expired["job_id"]), {"ok": True}, result_ttl_seconds=0)

    fresh, fresh_disposition = create_or_reuse_job("analyzer", payload={}, cache_key="reuse-after-expiry")

    assert fresh_disposition == "created"
    assert fresh["job_id"] != expired["job_id"]


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
        (analyzer, "analyzer", "_compute_analyzer_result", "/api/portfolio-analyzer/async", {}),
        (
            hedging,
            "hedging",
            "_compute_hedging_result",
            "/api/hedging-tool/async",
            {"book": 100000, "positions": [{"ticker": "AAA", "weight": 0.1}]},
        ),
        (
            sizer,
            "sizer",
            "_compute_sizer_result",
            "/api/portfolio-sizer/async",
            {"book": 100000, "target_leverage": 2.0, "positions": [{"ticker": "AAA", "conviction": 3}]},
        ),
        (
            fundamental_momentum,
            "fundamental_momentum",
            "_compute_fundamental_momentum",
            "/api/fundamental-momentum/async",
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


def test_analyzer_async_cancel_marks_job_cancelled(auth_client, monkeypatch):
    from api import cache
    from api.job_queue import get_job
    from api.routers import analyzer

    cache.invalidate_all()
    started = threading.Event()
    release = threading.Event()

    def slow_analyzer_job(_req, *, job_id=None):
        assert job_id
        started.set()
        release.wait(timeout=2)
        return {"ok": True}

    monkeypatch.setattr(analyzer, "_compute_analyzer_result", slow_analyzer_job)

    started_resp = auth_client.post(
        "/api/portfolio-analyzer/async",
        json={"scenario": {"preset": f"cancel-test-{time.time_ns()}"}},
    )
    assert started_resp.status_code == 202
    job_id = started_resp.json()["job_id"]
    assert started.wait(timeout=2)

    cancel_resp = auth_client.post(f"/api/portfolio-analyzer/async/{job_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"

    release.set()
    deadline = time.time() + 2
    while time.time() < deadline:
        row = get_job(job_id)
        if row and row["status"] == "cancelled":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("cancelled analyzer job was overwritten")

    poll_resp = auth_client.get(f"/api/portfolio-analyzer/async/{job_id}")
    assert poll_resp.status_code == 200
    assert poll_resp.json()["status"] == "cancelled"


def test_sizer_async_endpoint_runs_inline_without_cloud_run_job(auth_client, monkeypatch):
    from api import async_job_runner, cache
    from api.routers import sizer

    cache.invalidate_all()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "cloud_run_jobs")
    monkeypatch.setenv("ASYNC_DISPATCH_BACKEND_SIZER", "inline")
    monkeypatch.setattr(sizer, "_compute_sizer_result", lambda req: {"ok": req.positions[0].ticker})
    monkeypatch.setattr(
        async_job_runner,
        "_enqueue_cloud_run_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no Cloud Run dispatch")),
    )

    started = auth_client.post(
        "/api/portfolio-sizer/async",
        json={"book": 100000, "target_leverage": 2.0, "positions": [{"ticker": "AAA", "conviction": 3}]},
    )

    assert started.status_code == 200
    payload = started.json()
    job_id = payload["job_id"]
    assert payload["status"] == "done"
    assert payload["result"] == {"ok": "AAA"}

    completed = auth_client.get(f"/api/portfolio-sizer/async/{job_id}").json()
    assert completed["status"] == "done"
    assert completed["result"] == {"ok": "AAA"}


def test_fundamental_momentum_dispatch_error_returns_structured_503(auth_client, monkeypatch):
    from api.exceptions import AsyncJobDispatchError
    from api.routers import fundamental_momentum

    def fail_enqueue(*_args, **_kwargs):
        raise AsyncJobDispatchError("run api unavailable")

    monkeypatch.setattr(fundamental_momentum, "enqueue_registered_job", fail_enqueue)

    resp = auth_client.post(
        "/api/fundamental-momentum/async",
        json={
            "input_mode": "Custom Tickers",
            "tickers": "FTI, INVX",
            "screen_type": "Both",
            "benchmark": "Same as Input",
        },
    )

    assert resp.status_code == 503
    assert resp.json()["error"] == "Async job dispatch failed"
    # In development mode (default for tests), detail is present
    assert resp.json()["detail"] == "run api unavailable"
