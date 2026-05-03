from __future__ import annotations

import json
from typing import Any

import ontology.service as svc
from api.agent_tools import execute_tool
from ontology.models import InterpretedQuery
from ontology.policy import (
    Actor,
    DefaultOntologyPolicy,
    EdgeResource,
    NodeResource,
    OntologyAction,
    PolicyDecision,
    PolicyDenied,
    admin_actor,
    agent_actor,
)
from ontology.service import OntologyQueryService


class _Policy(DefaultOntologyPolicy):
    def __init__(
        self,
        *,
        denied_actions: set[str] | None = None,
        denied_objects: set[str] | None = None,
        denied_relations: set[str] | None = None,
        fields: dict[str, set[str]] | None = None,
    ):
        self.denied_actions = denied_actions or set()
        self.denied_objects = denied_objects or set()
        self.denied_relations = denied_relations or set()
        self.fields = fields or {}

    def check_action(self, actor, action: str, context=None):
        if action in self.denied_actions:
            return PolicyDecision(False, f"denied action: {action}")
        return PolicyDecision(True)

    def check_object(self, actor, node: NodeResource, action: str = "read"):
        if node.id in self.denied_objects:
            return PolicyDecision(False, f"denied object: {node.id}")
        return PolicyDecision(True)

    def check_relationship(self, actor, edge: EdgeResource, source=None, target=None, action: str = "read"):
        if edge.relation_type in self.denied_relations:
            return PolicyDecision(False, f"denied relation: {edge.relation_type}")
        return PolicyDecision(True)

    def allowed_fields(self, actor, resource):
        key = (
            getattr(resource, "relation_type", None)
            or getattr(resource, "type", None)
            or getattr(resource, "owner_type", "")
        )
        return self.fields.get(str(key))


class _Repo:
    def get_run(self, run_id: str):
        return {
            "run_id": run_id,
            "as_of": "2026-03-08T00:00:00Z",
            "source_status": {"portfolio": {"status": "ok"}},
            "required_modules": ["portfolio"],
            "component_scores": {},
            "created_at": "2026-03-08 00:00:00",
        }

    def list_runs(self, limit: int = 100):
        return [{"run_id": "run-1"}, {"run_id": "run-0"}]

    def fetch_snapshot_position_asset_sector_rows(self, run_id: str, *, schema_mode="upgraded"):
        return [
            self._row("MU", 0.72, "medium", "Information Technology"),
            self._row("NVDA", 0.81, "high", "Information Technology"),
        ]

    def query_snapshot_positions_page(self, run_id: str, *, filters=None, page=1, page_size=25, schema_mode="upgraded"):
        rows = self.fetch_snapshot_position_asset_sector_rows(run_id, schema_mode=schema_mode)
        start = max(0, (page - 1) * page_size)
        end = start + page_size
        return {
            "rows": rows[start:end],
            "total_results": len(rows),
            "page": page,
            "page_size": page_size,
        }

    def aggregate_snapshot_positions(self, run_id: str, *, filters=None):
        rows = self.fetch_snapshot_position_asset_sector_rows(run_id)
        scores = [float(row["position_props"].get("risk_score") or 0.0) for row in rows]
        return {
            "position_count": len(rows),
            "risk_buckets": {
                "high": sum(1 for score in scores if score >= 0.75),
                "medium": sum(1 for score in scores if 0.5 <= score < 0.75),
                "low": sum(1 for score in scores if score < 0.5),
            },
            "asset_exposure_counts": {"equity": len(rows)},
            "average_risk_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        }

    def fetch_snapshot_all_position_signal_evidence(self, run_id: str, *, schema_mode="upgraded"):
        return {
            "position:MU": [self._evidence("MU")],
            "position:NVDA": [self._evidence("NVDA")],
        }

    def fetch_snapshot_position_signal_evidence_batch(self, run_id: str, position_ids, *, schema_mode="upgraded"):
        grouped = self.fetch_snapshot_all_position_signal_evidence(run_id, schema_mode=schema_mode)
        return {position_id: list(grouped.get(position_id, [])) for position_id in position_ids}

    def fetch_snapshot_position_signal_evidence(self, run_id: str, position_id: str, *, schema_mode="upgraded"):
        ticker = position_id.split(":")[-1] if ":" in position_id else "MU"
        return [self._evidence(ticker)]

    def fetch_snapshot_position_thesis_context_batch(self, run_id: str, position_ids, *, schema_mode="upgraded"):
        return {position_id: {} for position_id in position_ids}

    def snapshot_has_positions(self, run_id: str) -> bool:
        return True

    def fetch_snapshot_graph(self, run_id: str, *, schema_mode="upgraded"):
        return {
            "nodes": [
                {"id": "position:MU", "type": "Position", "label": "MU", "properties": {"ticker": "MU"}},
                {"id": "position:NVDA", "type": "Position", "label": "NVDA", "properties": {"ticker": "NVDA"}},
                {"id": "asset:MU", "type": "Asset", "label": "MU", "properties": {"ticker": "MU", "asset": "equity"}},
            ],
            "edges": [
                {
                    "source_id": "position:MU",
                    "target_id": "asset:MU",
                    "relation_type": "references_asset",
                    "properties": {"ontology_run_id": run_id},
                },
                {
                    "source_id": "position:NVDA",
                    "target_id": "asset:MU",
                    "relation_type": "references_asset",
                    "properties": {"ontology_run_id": run_id},
                },
            ],
        }

    def _row(self, ticker: str, risk_score: float, risk_level: str, sector: str):
        return {
            "position_id": f"position:{ticker}",
            "position_label": ticker,
            "position_props": {
                "ticker": ticker,
                "asset": "equity",
                "direction": "long",
                "risk_score": risk_score,
                "risk_level": risk_level,
                "ontology_run_id": "run-1",
            },
            "asset_id": f"asset:{ticker}",
            "asset_label": ticker,
            "asset_props": {"ticker": ticker, "asset": "equity"},
            "sector_id": "sector:information_technology",
            "sector_label": sector,
            "sector_props": {"name": sector},
        }

    def _evidence(self, ticker: str):
        return {
            "position_id": f"position:{ticker}",
            "signal_id": f"signal:{ticker}",
            "signal_label": f"{ticker} signal",
            "signal_props": {"name": f"{ticker} signal"},
            "edge_props": {
                "source": "test",
                "name": "risk signal",
                "value": 1.2,
                "threshold": "> 1",
                "direction": "deteriorating",
                "contribution": 0.5,
                "ontology_run_id": "run-1",
            },
        }


def _stub_parse(*_args, **_kwargs):
    return InterpretedQuery(
        intent="portfolio_risk_exposure",
        source="structured",
        filters={},
        entity=None,
        original_query=None,
    )


def _query(policy: _Policy, monkeypatch):
    monkeypatch.setattr(svc, "parse_hybrid_query", _stub_parse)
    service = OntologyQueryService(repository=_Repo(), policy=policy)
    return service.query(
        query=None,
        intent="portfolio_risk_exposure",
        filters={},
        run_id="run-1",
        include_graph=False,
        actor=admin_actor(),
    )


def test_policy_denies_object_and_recomputes_aggregate(monkeypatch):
    resp = _query(_Policy(denied_objects={"position:NVDA"}), monkeypatch)

    assert [row["ticker"] for row in resp["results"]] == ["MU"]
    assert resp["aggregate"]["exact"] is False
    assert resp["_meta"]["pagination"]["exact_total"] is False
    assert resp["_meta"]["authorization"]["filtered_objects"] == 1


def test_policy_denies_relationships_and_redacts_edge_fields(monkeypatch):
    resp = _query(
        _Policy(
            denied_relations={"belongs_to_sector"},
            fields={"exposed_to_signal": {"source", "name"}},
        ),
        monkeypatch,
    )

    first = resp["results"][0]
    assert first["sector"] is None
    assert first["evidence"][0]["name"] == "risk signal"
    assert first["evidence"][0]["value"] is None
    assert first["evidence"][0]["threshold"] is None
    assert first["evidence"][0]["contribution"] is None


def test_policy_denied_exposed_to_signal_removes_evidence(monkeypatch):
    resp = _query(_Policy(denied_relations={"exposed_to_signal"}), monkeypatch)

    assert resp["results"][0]["evidence"] == []
    assert resp["_meta"]["authorization"]["filtered_relationships"] >= 1


def test_policy_redacts_risk_score_and_suppresses_risk_deltas(monkeypatch):
    monkeypatch.setattr(svc, "parse_hybrid_query", _stub_parse)
    service = OntologyQueryService(
        repository=_Repo(),
        policy=_Policy(
            fields={
                "Position": {"ticker", "asset", "direction", "risk_level", "ontology_run_id"},
                "Sector": {"name"},
                "Signal": {"name"},
            }
        ),
    )

    resp = service.query(
        query=None,
        intent="portfolio_risk_exposure",
        filters={},
        run_id="run-1",
        actor=admin_actor(),
    )
    assert resp["results"][0]["risk_score"] is None
    assert resp["aggregate"]["average_risk_score"] == 0.0

    diff = service.compare_snapshots("run-0", "run-1", actor=admin_actor())
    assert diff["risk_changes"] == []


def test_policy_filters_graph_nodes_and_connected_edges(monkeypatch):
    monkeypatch.setattr(svc, "parse_hybrid_query", _stub_parse)
    service = OntologyQueryService(repository=_Repo(), policy=_Policy(denied_objects={"position:NVDA"}))

    resp = service.query(
        query=None,
        intent="portfolio_risk_exposure",
        filters={},
        run_id="run-1",
        include_graph=True,
        actor=admin_actor(),
    )

    graph = resp["graph"]
    assert {node["id"] for node in graph["nodes"]} == {
        "position:MU",
        "asset:MU",
        "sector:information_technology",
        "signal:MU",
    }
    assert all(edge["source_id"] != "position:NVDA" for edge in graph["edges"])


def test_ontology_api_denied_action_returns_403(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    service = OntologyQueryService(repository=_Repo(), policy=_Policy(denied_actions={OntologyAction.QUERY}))
    monkeypatch.setattr(ontology_router, "_service", service)

    resp = auth_client.post(
        "/api/v1/ontology/query",
        json={"intent": "portfolio_risk_exposure", "schema_mode": "upgraded"},
    )

    assert resp.status_code == 403


def test_agent_ontology_denied_action_returns_structured_error(monkeypatch):
    monkeypatch.setattr("api.agent_tools.get_cached", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("api.agent_tools.set_cached", lambda *_args, **_kwargs: None)

    def fake_query(self, *args, **kwargs):
        raise PolicyDenied("query denied")

    monkeypatch.setattr("ontology.service.OntologyQueryService.query", fake_query)

    payload = json.loads(execute_tool("query_ontology", {}, actor=agent_actor(admin_actor())))

    assert payload["type"] == "PolicyDenied"
    assert payload["_meta"]["status"] == "denied"


def test_ontology_api_and_agent_propagate_actor(auth_client, monkeypatch):
    import api.routers.ontology as ontology_router

    captured: dict[str, Actor] = {}

    class _Service:
        policy = _Policy()

        def query(self, *args, actor: Actor | None = None, **kwargs):
            captured["api"] = actor
            return {
                "run_id": "run-1",
                "intent": "portfolio_risk_exposure",
                "interpreted_query": {"source": "structured", "filters": {}},
                "as_of": "2026-03-08T00:00:00Z",
                "source_status": {"portfolio": {"status": "ok"}},
                "results": [],
                "aggregate": {"position_count": 0},
            }

    monkeypatch.setattr(ontology_router, "_service", _Service())
    resp = auth_client.post(
        "/api/v1/ontology/query",
        json={"intent": "portfolio_risk_exposure", "query": "actor propagation check", "schema_mode": "upgraded"},
    )
    job = resp.json()
    done = auth_client.get(f"/api/v1/ontology/query/async/{job['job_id']}").json()

    assert done["status"] == "done"
    assert captured["api"].actor_id == "admin"

    def fake_agent_query(self, *args, actor: Actor | None = None, **kwargs):
        captured["agent"] = actor
        return {
            "run_id": "run-1",
            "intent": "portfolio_risk_exposure",
            "interpreted_query": {"source": "structured", "filters": {}},
            "as_of": "2026-03-08T00:00:00Z",
            "source_status": {"portfolio": {"status": "ok"}},
            "results": [],
            "aggregate": {"position_count": 0},
        }

    monkeypatch.setattr("api.agent_tools.get_cached", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("api.agent_tools.set_cached", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("ontology.service.OntologyQueryService.query", fake_agent_query)
    execute_tool("query_ontology", {"query": "actor propagation check"}, actor=agent_actor(admin_actor()))

    assert captured["agent"].actor_type == "agent"
    assert captured["agent"].parent_actor_id == "admin"


def test_ontology_job_payload_includes_actor_and_cache_key_differs():
    import api.routers.ontology as ontology_router

    req = ontology_router.OntologyQueryRequest(intent="portfolio_risk_exposure", schema_mode="upgraded")
    first = ontology_router._job_request(req, admin_actor("admin"))
    second = ontology_router._job_request(req, admin_actor("other"))

    assert first.actor["actor_id"] == "admin"
    assert ontology_router._job_cache_key(first) != ontology_router._job_cache_key(second)
