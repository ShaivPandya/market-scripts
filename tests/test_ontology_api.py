from __future__ import annotations

import time
import uuid

import pytest

import portfolio.core_db as core_db
from api.main import app
from api.routers.auth import require_actor
from ontology.policy import Actor, DefaultOntologyPolicy, OntologyAction, PolicyDecision, admin_actor
from ontology.service import OntologyRunNotFoundError


@pytest.fixture(autouse=True)
def _use_temp_state(tmp_path, monkeypatch):
    from api import job_queue

    if core_db._conn:
        try:
            core_db._conn.close()
        except Exception:
            pass
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "test_core.db")
    monkeypatch.setattr(core_db, "_conn", None)
    job_queue._memory_jobs.clear()
    yield
    job_queue._memory_jobs.clear()
    if core_db._conn:
        try:
            core_db._conn.close()
        except Exception:
            pass
    monkeypatch.setattr(core_db, "_conn", None)
    app.dependency_overrides.pop(require_actor, None)


class _Policy(DefaultOntologyPolicy):
    def __init__(self, *, denied_actions: set[str] | None = None):
        self.denied_actions = denied_actions or set()

    def check_action(self, actor, action: str, context=None):
        if action in self.denied_actions:
            return PolicyDecision(False, f"denied action: {action}")
        return PolicyDecision(True)

    def check_object(self, actor, node, action: str = "read"):
        return PolicyDecision(True)

    def check_relationship(self, actor, edge, source=None, target=None, action: str = "read"):
        return PolicyDecision(True)

    def allowed_fields(self, actor, resource):
        return None


class _FakeService:
    def __init__(self, payload):
        self.payload = payload

    def query(
        self,
        query,
        intent,
        filters,
        timeframe,
        include_graph,
        run_id,
        refresh_snapshot=False,
        page=1,
        page_size=25,
        schema_mode="upgraded",
    ):
        out = dict(self.payload)
        out.setdefault("run_id", run_id or "run-1")
        out.setdefault("intent", intent or "portfolio_risk_exposure")
        out.setdefault(
            "interpreted_query",
            {
                "source": "structured" if intent else "deterministic_fallback",
                "query": query,
                "entity": None,
                "filters": filters,
            },
        )
        out.setdefault("as_of", "2026-03-08T00:00:00Z")
        out.setdefault("source_status", {"portfolio": {"status": "ok"}})
        out.setdefault("results", [])
        out.setdefault(
            "aggregate",
            {
                "position_count": 0,
                "risk_buckets": {"high": 0, "medium": 0, "low": 0},
                "asset_exposure_counts": {},
                "average_risk_score": 0.0,
                "confidence": 1.0,
            },
        )
        return out


def _resolve_ontology_result(auth_client, resp):
    assert resp.status_code in (200, 202)
    data = resp.json()
    if "job_id" not in data:
        return data
    done = _poll_ontology_job(auth_client, data["job_id"])
    assert done["status"] == "done"
    return done["result"]


def test_ontology_query_requires_auth(client):
    resp = client.post("/api/v1/ontology/query", json={"intent": "portfolio_risk_exposure", "schema_mode": "upgraded"})
    assert resp.status_code == 401


def test_ontology_query_structured_returns_schema(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    fake = _FakeService(
        {
            "results": [
                {
                    "ticker": "MU",
                    "asset": "equity",
                    "direction": "long",
                    "sector": "Information Technology",
                    "risk_score": 0.72,
                    "risk_level": "medium",
                    "evidence": [],
                }
            ]
        }
    )
    monkeypatch.setattr(ontology_router, "_service", fake)

    resp = auth_client.post(
        "/api/v1/ontology/query",
        json={
            "intent": "portfolio_risk_exposure",
            "filters": {"tickers": ["MU"], "min_risk_score": 0.5},
            "timeframe": "Daily",
            "schema_mode": "upgraded",
        },
    )

    data = _resolve_ontology_result(auth_client, resp)
    assert data["intent"] == "portfolio_risk_exposure"
    assert data["interpreted_query"]["source"] == "structured"
    assert isinstance(data["source_status"], dict)
    assert isinstance(data["results"], list)
    assert data["run_id"] == "run-1"
    assert "aggregate" in data


def test_ontology_query_nl_path_returns_interpreted_source(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    fake = _FakeService(
        {
            "intent": "positions_in_deteriorating_macro",
            "interpreted_query": {
                "source": "deterministic_fallback",
                "query": "Which positions are in deteriorating macro conditions?",
                "entity": None,
                "filters": {},
            },
        }
    )
    monkeypatch.setattr(ontology_router, "_service", fake)

    resp = auth_client.post(
        "/api/v1/ontology/query",
        json={"query": "Which positions are in deteriorating macro conditions?", "schema_mode": "upgraded"},
    )

    data = _resolve_ontology_result(auth_client, resp)
    assert data["intent"] == "positions_in_deteriorating_macro"
    assert data["interpreted_query"]["source"] in {"llm", "deterministic_fallback", "structured"}


def test_ontology_query_partial_failure_returns_200_with_degraded_confidence(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    fake = _FakeService(
        {
            "source_status": {
                "portfolio": {"status": "ok"},
                "vix_term_structure": {"status": "error", "detail": "timeout"},
            },
            "aggregate": {
                "position_count": 1,
                "risk_buckets": {"high": 1, "medium": 0, "low": 0},
                "asset_exposure_counts": {"equity": 1},
                "average_risk_score": 0.81,
                "confidence": 0.62,
            },
        }
    )
    monkeypatch.setattr(ontology_router, "_service", fake)

    resp = auth_client.post(
        "/api/v1/ontology/query",
        json={"intent": "portfolio_risk_exposure", "schema_mode": "upgraded"},
    )

    data = _resolve_ontology_result(auth_client, resp)
    assert data["aggregate"]["confidence"] < 1.0
    assert data["source_status"]["vix_term_structure"]["status"] in {"error", "partial"}


def test_ontology_query_unknown_run_id_returns_404(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    class _MissingRunService:
        def query(
            self,
            query,
            intent,
            filters,
            timeframe,
            include_graph,
            run_id,
            refresh_snapshot=False,
            page=1,
            page_size=25,
            schema_mode="upgraded",
        ):
            raise OntologyRunNotFoundError(str(run_id))

    monkeypatch.setattr(ontology_router, "_service", _MissingRunService())

    resp = auth_client.post(
        "/api/v1/ontology/query",
        json={"intent": "portfolio_risk_exposure", "run_id": "missing-run", "schema_mode": "upgraded"},
    )

    assert resp.status_code in (200, 202)
    body = resp.json()
    done = _poll_ontology_job(auth_client, body["job_id"])
    assert done["status"] == "error"
    msg = str(done.get("error", ""))
    assert "Ontology run" in msg


def test_ontology_query_passes_run_id(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    captured: dict[str, str | None] = {"run_id": None}

    class _CaptureRunService:
        def query(
            self,
            query,
            intent,
            filters,
            timeframe,
            include_graph,
            run_id,
            refresh_snapshot=False,
            page=1,
            page_size=25,
            schema_mode="upgraded",
        ):
            captured["run_id"] = run_id
            return {
                "run_id": run_id or "run-1",
                "intent": "portfolio_risk_exposure",
                "interpreted_query": {"source": "structured", "query": query, "entity": None, "filters": filters},
                "as_of": "2026-03-08T00:00:00Z",
                "source_status": {"portfolio": {"status": "ok"}},
                "results": [],
                "aggregate": {
                    "position_count": 0,
                    "risk_buckets": {"high": 0, "medium": 0, "low": 0},
                    "asset_exposure_counts": {},
                    "average_risk_score": 0.0,
                    "confidence": 1.0,
                },
            }

    monkeypatch.setattr(ontology_router, "_service", _CaptureRunService())

    resp = auth_client.post(
        "/api/v1/ontology/query",
        json={"intent": "portfolio_risk_exposure", "run_id": "run-abc", "schema_mode": "upgraded"},
    )

    data = _resolve_ontology_result(auth_client, resp)
    assert captured["run_id"] == "run-abc"
    assert data["run_id"] == "run-abc"


def test_ontology_query_passes_refresh_snapshot(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    captured: dict[str, bool] = {"refresh_snapshot": False}

    class _CaptureRefreshService:
        def query(
            self,
            query,
            intent,
            filters,
            timeframe,
            include_graph,
            run_id,
            refresh_snapshot=False,
            page=1,
            page_size=25,
            schema_mode="upgraded",
        ):
            captured["refresh_snapshot"] = bool(refresh_snapshot)
            return {
                "run_id": "run-1",
                "intent": "portfolio_risk_exposure",
                "interpreted_query": {"source": "structured", "query": query, "entity": None, "filters": filters},
                "as_of": "2026-03-08T00:00:00Z",
                "source_status": {"portfolio": {"status": "ok"}},
                "results": [],
                "aggregate": {
                    "position_count": 0,
                    "risk_buckets": {"high": 0, "medium": 0, "low": 0},
                    "asset_exposure_counts": {},
                    "average_risk_score": 0.0,
                    "confidence": 1.0,
                },
            }

    monkeypatch.setattr(ontology_router, "_service", _CaptureRefreshService())

    resp = auth_client.post(
        "/api/v1/ontology/query",
        json={"intent": "portfolio_risk_exposure", "refresh_snapshot": True, "schema_mode": "upgraded"},
    )
    _resolve_ontology_result(auth_client, resp)
    assert captured["refresh_snapshot"] is True


def test_ontology_query_rejects_removed_filters_max_results(auth_client):
    resp = auth_client.post(
        "/api/v1/ontology/query",
        json={
            "intent": "portfolio_risk_exposure",
            "filters": {"tickers": ["MU"], "max_results": 5},
            "schema_mode": "upgraded",
        },
    )

    assert resp.status_code == 422
    assert "filters.max_results has been removed; use top-level page_size instead" in str(resp.json())


def test_ontology_query_passes_pagination(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    captured: dict[str, int] = {"page": 0, "page_size": 0}

    class _CapturePaginationService:
        def query(
            self,
            query,
            intent,
            filters,
            timeframe,
            include_graph,
            run_id,
            refresh_snapshot=False,
            page=1,
            page_size=25,
            schema_mode="upgraded",
        ):
            captured["page"] = page
            captured["page_size"] = page_size
            return {
                "run_id": "run-1",
                "intent": "portfolio_risk_exposure",
                "interpreted_query": {"source": "structured", "query": query, "entity": None, "filters": filters},
                "as_of": "2026-03-08T00:00:00Z",
                "source_status": {"portfolio": {"status": "ok"}},
                "results": [],
                "aggregate": {
                    "position_count": 0,
                    "risk_buckets": {"high": 0, "medium": 0, "low": 0},
                    "asset_exposure_counts": {},
                    "average_risk_score": 0.0,
                    "confidence": 1.0,
                },
                "_meta": {
                    "pagination": {
                        "page": page,
                        "page_size": page_size,
                        "returned_results": 0,
                        "total_results": 0,
                        "total_pages": 0,
                        "has_prev": False,
                        "has_next": False,
                        "sort": "risk_score_desc_then_position_id_asc",
                        "exact_total": True,
                    }
                },
            }

    monkeypatch.setattr(ontology_router, "_service", _CapturePaginationService())

    resp = auth_client.post(
        "/api/v1/ontology/query",
        json={"intent": "portfolio_risk_exposure", "page": 3, "page_size": 7, "schema_mode": "upgraded"},
    )

    data = _resolve_ontology_result(auth_client, resp)
    assert captured == {"page": 3, "page_size": 7}
    assert data["_meta"]["pagination"]["page"] == 3
    assert data["_meta"]["pagination"]["page_size"] == 7


def _poll_ontology_job(auth_client, job_id: str, timeout_s: float = 4.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = auth_client.get(f"/api/v1/ontology/query/async/{job_id}")
        assert resp.status_code == 200
        payload = resp.json()
        if payload.get("status") in {"done", "error"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout_s}s")


def test_ontology_query_async_returns_done_result(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    query_text = f"async-run-{uuid.uuid4().hex}"

    class _AsyncService:
        def query(
            self,
            query,
            intent,
            filters,
            timeframe,
            include_graph,
            run_id,
            refresh_snapshot=False,
            page=1,
            page_size=25,
            schema_mode="upgraded",
        ):
            time.sleep(0.1)
            return {
                "run_id": "run-async",
                "intent": intent or "portfolio_risk_exposure",
                "interpreted_query": {"source": "structured", "query": query, "entity": None, "filters": filters},
                "as_of": "2026-03-08T00:00:00Z",
                "source_status": {"portfolio": {"status": "ok"}},
                "results": [],
                "aggregate": {
                    "position_count": 0,
                    "risk_buckets": {"high": 0, "medium": 0, "low": 0},
                    "asset_exposure_counts": {},
                    "average_risk_score": 0.0,
                    "confidence": 1.0,
                },
            }

    monkeypatch.setattr(ontology_router, "_service", _AsyncService())

    started = auth_client.post(
        "/api/v1/ontology/query/async", json={"query": query_text, "timeframe": "Daily", "schema_mode": "upgraded"}
    )
    assert started.status_code in (200, 202)
    started_payload = started.json()
    assert started_payload["status"] in {"queued", "running", "done"}
    job_id = started_payload["job_id"]

    if started_payload["status"] == "done":
        done = started_payload
    else:
        done = _poll_ontology_job(auth_client, job_id)
    assert done["status"] == "done"
    assert done["result"]["run_id"] == "run-async"


def test_ontology_query_async_dedupes_running_job(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    query_text = f"async-dedupe-{uuid.uuid4().hex}"

    class _SlowService:
        def query(
            self,
            query,
            intent,
            filters,
            timeframe,
            include_graph,
            run_id,
            refresh_snapshot=False,
            page=1,
            page_size=25,
            schema_mode="upgraded",
        ):
            time.sleep(0.25)
            return {
                "run_id": "run-dedupe",
                "intent": "portfolio_risk_exposure",
                "interpreted_query": {"source": "structured", "query": query, "entity": None, "filters": filters},
                "as_of": "2026-03-08T00:00:00Z",
                "source_status": {"portfolio": {"status": "ok"}},
                "results": [],
                "aggregate": {
                    "position_count": 0,
                    "risk_buckets": {"high": 0, "medium": 0, "low": 0},
                    "asset_exposure_counts": {},
                    "average_risk_score": 0.0,
                    "confidence": 1.0,
                },
            }

    monkeypatch.setattr(ontology_router, "_service", _SlowService())

    req = {"query": query_text, "timeframe": "Daily", "schema_mode": "upgraded"}
    first = auth_client.post("/api/v1/ontology/query/async", json=req)
    second = auth_client.post("/api/v1/ontology/query/async", json=req)
    assert first.status_code in (200, 202)
    assert second.status_code in (200, 202)
    p1 = first.json()
    p2 = second.json()
    assert p2["job_id"] == p1["job_id"]
    assert p2["status"] in {"queued", "running", "done"}

    done = _poll_ontology_job(auth_client, p1["job_id"], timeout_s=6.0)
    assert done["status"] == "done"


def test_ontology_query_async_surfaces_worker_error(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    query_text = f"async-error-{uuid.uuid4().hex}"

    class _ErrorService:
        def query(
            self,
            query,
            intent,
            filters,
            timeframe,
            include_graph,
            run_id,
            refresh_snapshot=False,
            page=1,
            page_size=25,
            schema_mode="upgraded",
        ):
            raise RuntimeError("boom")

    monkeypatch.setattr(ontology_router, "_service", _ErrorService())

    started = auth_client.post("/api/v1/ontology/query/async", json={"query": query_text, "schema_mode": "upgraded"})
    assert started.status_code in (200, 202)
    job_id = started.json()["job_id"]

    result = _poll_ontology_job(auth_client, job_id)
    assert result["status"] == "error"
    assert "boom" in str(result.get("error", ""))


def test_query_preflight_denies_include_graph_without_enqueuing_job(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    called = {"enqueued": False}

    class _Service:
        policy = _Policy(denied_actions={OntologyAction.GRAPH_READ})

    monkeypatch.setattr(ontology_router, "_service", _Service())
    monkeypatch.setattr(
        ontology_router,
        "enqueue_registered_job",
        lambda *args, **kwargs: called.__setitem__("enqueued", True),
    )

    resp = auth_client.post(
        "/api/v1/ontology/query",
        json={"intent": "portfolio_risk_exposure", "include_graph": True, "schema_mode": "upgraded"},
    )

    assert resp.status_code == 403
    assert called["enqueued"] is False


def test_query_preflight_denies_refresh_snapshot_without_enqueuing_job(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    called = {"enqueued": False}

    class _Service:
        policy = _Policy(denied_actions={OntologyAction.SNAPSHOT_REFRESH})

    monkeypatch.setattr(ontology_router, "_service", _Service())
    monkeypatch.setattr(
        ontology_router,
        "enqueue_registered_job",
        lambda *args, **kwargs: called.__setitem__("enqueued", True),
    )

    resp = auth_client.post(
        "/api/v1/ontology/query",
        json={"intent": "portfolio_risk_exposure", "refresh_snapshot": True, "schema_mode": "upgraded"},
    )

    assert resp.status_code == 403
    assert called["enqueued"] is False


def test_refresh_snapshot_does_not_reuse_completed_ontology_job(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    calls: list[dict] = []

    def fake_enqueue(job_type, payload, **kwargs):
        calls.append({"job_type": job_type, "payload": payload, **kwargs})
        return (
            {
                "job_id": f"job-{len(calls)}",
                "job_type": job_type,
                "status": "queued",
                "progress_json": None,
            },
            "created",
        )

    monkeypatch.setattr(ontology_router, "enqueue_registered_job", fake_enqueue)

    resp = auth_client.post(
        "/api/v1/ontology/query/async",
        json={"intent": "portfolio_risk_exposure", "refresh_snapshot": True, "schema_mode": "upgraded"},
    )

    assert resp.status_code == 202
    assert calls[0]["reuse_completed"] is False


def test_cached_ontology_reads_can_reuse_completed_job(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    calls: list[dict] = []

    def fake_enqueue(job_type, payload, **kwargs):
        calls.append({"job_type": job_type, "payload": payload, **kwargs})
        return (
            {
                "job_id": f"job-{len(calls)}",
                "job_type": job_type,
                "status": "queued",
                "progress_json": None,
            },
            "created",
        )

    monkeypatch.setattr(ontology_router, "enqueue_registered_job", fake_enqueue)

    resp = auth_client.post(
        "/api/v1/ontology/query/async",
        json={"intent": "portfolio_risk_exposure", "schema_mode": "upgraded"},
    )

    assert resp.status_code == 202
    assert calls[0]["reuse_completed"] is True


def test_async_job_read_allows_submitter_and_admin_but_denies_other_actor(client, monkeypatch):
    import api.routers.ontology as ontology_router

    class _PermissivePolicy(_Policy):
        def check_action(self, actor, action: str, context=None):
            return PolicyDecision(True)

    class _Service:
        policy = _PermissivePolicy()

        def query(
            self,
            query,
            intent,
            filters,
            timeframe,
            include_graph,
            run_id,
            refresh_snapshot=False,
            page=1,
            page_size=25,
            schema_mode="upgraded",
            actor: Actor | None = None,
        ):
            return {
                "run_id": "run-owner",
                "intent": intent or "portfolio_risk_exposure",
                "interpreted_query": {"source": "structured", "query": query, "entity": None, "filters": filters},
                "as_of": "2026-03-08T00:00:00Z",
                "source_status": {"portfolio": {"status": "ok"}},
                "results": [],
                "aggregate": {"position_count": 0},
                "actor_id": actor.actor_id if actor is not None else None,
            }

    monkeypatch.setattr(ontology_router, "_service", _Service())

    submitter = Actor(actor_id="submitter", actor_type="user", roles=(), source="test")
    other = Actor(actor_id="other-user", actor_type="user", roles=(), source="test")
    admin = admin_actor("admin", source="test")

    app.dependency_overrides[require_actor] = lambda: submitter
    started = client.post(
        "/api/v1/ontology/query/async",
        json={"query": f"owner-{uuid.uuid4().hex}", "schema_mode": "upgraded"},
    )
    assert started.status_code in (200, 202)
    job_id = started.json()["job_id"]

    own_read = client.get(f"/api/v1/ontology/query/async/{job_id}")
    assert own_read.status_code == 200

    app.dependency_overrides[require_actor] = lambda: other
    other_read = client.get(f"/api/v1/ontology/query/async/{job_id}")
    assert other_read.status_code == 403

    app.dependency_overrides[require_actor] = lambda: admin
    admin_read = client.get(f"/api/v1/ontology/query/async/{job_id}")
    assert admin_read.status_code == 200


def test_schema_mode_stored_surfaces_as_terminal_async_error(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    class _StoredModeService:
        def query(
            self,
            query,
            intent,
            filters,
            timeframe,
            include_graph,
            run_id,
            refresh_snapshot=False,
            page=1,
            page_size=25,
            schema_mode="upgraded",
        ):
            raise ValueError("Ontology semantic queries require schema_mode='upgraded'")

    monkeypatch.setattr(ontology_router, "_service", _StoredModeService())

    started = auth_client.post(
        "/api/v1/ontology/query/async",
        json={"intent": "portfolio_risk_exposure", "schema_mode": "stored"},
    )
    assert started.status_code in (200, 202)

    result = _poll_ontology_job(auth_client, started.json()["job_id"])
    assert result["status"] == "error"
    assert "schema_mode='upgraded'" in str(result.get("error", ""))


def test_query_preflight_and_job_read_emit_expected_audit_rows(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    class _DeniedService:
        policy = _Policy(denied_actions={OntologyAction.GRAPH_READ})

    class _PermissivePolicy(_Policy):
        def check_action(self, actor, action: str, context=None):
            return PolicyDecision(True)

    class _PermissiveService:
        policy = _PermissivePolicy()

        def query(
            self,
            query,
            intent,
            filters,
            timeframe,
            include_graph,
            run_id,
            refresh_snapshot=False,
            page=1,
            page_size=25,
            schema_mode="upgraded",
        ):
            return {
                "run_id": "run-audit",
                "intent": intent or "portfolio_risk_exposure",
                "interpreted_query": {"source": "structured", "query": query, "entity": None, "filters": filters},
                "as_of": "2026-03-08T00:00:00Z",
                "source_status": {"portfolio": {"status": "ok"}},
                "results": [],
                "aggregate": {"position_count": 0},
            }

    monkeypatch.setattr(ontology_router, "_service", _DeniedService())
    denied = auth_client.post(
        "/api/v1/ontology/query",
        json={"intent": "portfolio_risk_exposure", "include_graph": True, "schema_mode": "upgraded"},
    )
    assert denied.status_code == 403

    monkeypatch.setattr(ontology_router, "_service", _PermissiveService())
    started = auth_client.post(
        "/api/v1/ontology/query/async",
        json={"intent": "portfolio_risk_exposure", "schema_mode": "upgraded"},
    )
    job_id = started.json()["job_id"]
    read = auth_client.get(f"/api/v1/ontology/query/async/{job_id}")
    assert read.status_code == 200

    preflight = core_db.get_audit_events(action_name="ontology.query.preflight", limit=5)[0]
    job_read = core_db.get_audit_events(action_name="ontology.job.read", limit=5)[0]
    assert preflight["status"] == "denied"
    assert preflight["metadata"]["include_graph"] is True
    assert job_read["status"] == "succeeded"
    assert job_read["object_refs"] == [{"type": "async_job", "id": job_id}]


def test_unknown_job_id_returns_404_without_leaking_actor_context(client):
    actor = Actor(actor_id="analyst-user", actor_type="user", roles=("admin",), source="test")
    app.dependency_overrides[require_actor] = lambda: actor

    resp = client.get("/api/v1/ontology/query/async/missing-job")

    assert resp.status_code == 404
    assert "analyst-user" not in str(resp.json())
