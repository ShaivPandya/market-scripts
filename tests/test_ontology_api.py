from __future__ import annotations

import time
import uuid

from ontology.service import OntologyRunNotFoundError


class _FakeService:
    def __init__(self, payload):
        self.payload = payload

    def query(self, query, intent, filters, timeframe, include_graph, run_id, refresh_snapshot=False):
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
    resp = client.post("/api/v1/ontology/query", json={"intent": "portfolio_risk_exposure"})
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
        json={"query": "Which positions are in deteriorating macro conditions?"},
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
        json={"intent": "portfolio_risk_exposure"},
    )

    data = _resolve_ontology_result(auth_client, resp)
    assert data["aggregate"]["confidence"] < 1.0
    assert data["source_status"]["vix_term_structure"]["status"] in {"error", "partial"}


def test_ontology_query_unknown_run_id_returns_404(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    class _MissingRunService:
        def query(self, query, intent, filters, timeframe, include_graph, run_id, refresh_snapshot=False):
            raise OntologyRunNotFoundError(str(run_id))

    monkeypatch.setattr(ontology_router, "_service", _MissingRunService())

    resp = auth_client.post(
        "/api/v1/ontology/query",
        json={"intent": "portfolio_risk_exposure", "run_id": "missing-run"},
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
        def query(self, query, intent, filters, timeframe, include_graph, run_id, refresh_snapshot=False):
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
        json={"intent": "portfolio_risk_exposure", "run_id": "run-abc"},
    )

    data = _resolve_ontology_result(auth_client, resp)
    assert captured["run_id"] == "run-abc"
    assert data["run_id"] == "run-abc"


def test_ontology_query_passes_refresh_snapshot(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    captured: dict[str, bool] = {"refresh_snapshot": False}

    class _CaptureRefreshService:
        def query(self, query, intent, filters, timeframe, include_graph, run_id, refresh_snapshot=False):
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
        json={"intent": "portfolio_risk_exposure", "refresh_snapshot": True},
    )
    _resolve_ontology_result(auth_client, resp)
    assert captured["refresh_snapshot"] is True


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
        def query(self, query, intent, filters, timeframe, include_graph, run_id, refresh_snapshot=False):
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

    started = auth_client.post("/api/v1/ontology/query/async", json={"query": query_text, "timeframe": "Daily"})
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
        def query(self, query, intent, filters, timeframe, include_graph, run_id, refresh_snapshot=False):
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

    req = {"query": query_text, "timeframe": "Daily"}
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
        def query(self, query, intent, filters, timeframe, include_graph, run_id, refresh_snapshot=False):
            raise RuntimeError("boom")

    monkeypatch.setattr(ontology_router, "_service", _ErrorService())

    started = auth_client.post("/api/v1/ontology/query/async", json={"query": query_text})
    assert started.status_code in (200, 202)
    job_id = started.json()["job_id"]

    result = _poll_ontology_job(auth_client, job_id)
    assert result["status"] == "error"
    assert "boom" in str(result.get("error", ""))
