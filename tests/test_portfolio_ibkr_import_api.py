from __future__ import annotations

from io import BytesIO

from tests.test_ibkr_flex_import import FLEX_WITH_INDEX_ETFS, SAMPLE_FLEX_XML


def test_import_ibkr_flex_route_stages_replacement_proposal(auth_client, monkeypatch):
    from api.routers import portfolio_edit

    calls: list[dict] = []

    def fake_stage(action_id, payload, **kwargs):
        calls.append({"action_id": action_id, "payload": payload, **kwargs})
        return {
            "status": "pending_approval_created",
            "approval_id": "approval:portfolio-import",
            "application_status": "pending",
            "action_id": action_id,
            "entity_type": "portfolio_positions",
            "ticker": None,
            "proposed_change": payload,
        }

    monkeypatch.setattr(portfolio_edit, "stage_api_action", fake_stage)
    monkeypatch.setattr(
        portfolio_edit.OntologyRuntimeReadService,
        "positions",
        lambda self, include_hedges=False: [
            {
                "ticker": "MU",
                "instrument_type": "security",
                "direction": "long",
                "shares": 5,
                "conviction": 5,
                "contrarian": True,
                "group_name": "Semis",
                "group_conviction": 4,
                "role": "position",
            }
        ],
    )

    response = auth_client.post(
        "/api/portfolio-positions/import/ibkr-flex",
        files={"file": ("open_positions.xml", BytesIO(SAMPLE_FLEX_XML.encode("utf-8")), "application/xml")},
    )

    assert response.status_code == 200
    assert calls[0]["action_id"] == "update_portfolio_positions"
    positions = calls[0]["payload"]["positions"]
    assert len(positions) == 5
    mu = next(row for row in positions if row["ticker"] == "MU")
    assert mu["shares"] == 10
    assert mu["conviction"] == 5
    assert mu["contrarian"] is True
    assert mu["group_name"] == "Semis"
    meta_call = next(row for row in positions if row.get("option_contract_symbol") == "META270319C00620000")
    assert meta_call["instrument_type"] == "option"
    body = response.json()
    assert body["import_summary"]["imported_count"] == 5
    assert body["import_summary"]["source"] == "ibkr_flex"


def test_import_ibkr_flex_route_stages_hedge_proposal_for_short_index_etfs(auth_client, monkeypatch):
    from api.routers import portfolio_edit

    calls: list[dict] = []

    def fake_stage(action_id, payload, **kwargs):
        calls.append({"action_id": action_id, "payload": payload, **kwargs})
        return {
            "status": "pending_approval_created",
            "approval_id": f"approval:{action_id}",
            "application_status": "pending",
            "action_id": action_id,
            "entity_type": "portfolio_positions" if action_id == "update_portfolio_positions" else "hedge_positions",
            "ticker": None,
            "proposed_change": payload,
        }

    monkeypatch.setattr(portfolio_edit, "stage_api_action", fake_stage)
    monkeypatch.setattr(
        portfolio_edit.OntologyRuntimeReadService,
        "positions",
        lambda self, include_hedges=False: [],
    )
    monkeypatch.setattr(
        portfolio_edit.OntologyRuntimeReadService,
        "list_objects",
        lambda self, object_type, limit=1000: (
            [
                {
                    "ticker": "TLT",
                    "instrument_type": "security",
                    "direction": "short",
                    "shares": 100,
                    "position_id": "TLT",
                }
            ]
            if object_type == "HedgePosition"
            else []
        ),
    )

    response = auth_client.post(
        "/api/portfolio-positions/import/ibkr-flex",
        files={"file": ("open_positions.xml", BytesIO(FLEX_WITH_INDEX_ETFS.encode("utf-8")), "application/xml")},
    )

    assert response.status_code == 200
    assert len(calls) == 2
    assert calls[0]["action_id"] == "update_portfolio_positions"
    assert calls[1]["action_id"] == "update_hedge_positions"

    portfolio_tickers = {row["ticker"] for row in calls[0]["payload"]["positions"]}
    assert "SPY" not in portfolio_tickers
    assert "IWM" not in portfolio_tickers
    assert "QQQ" in portfolio_tickers
    assert "PII" in portfolio_tickers

    hedge_positions = calls[1]["payload"]["positions"]
    hedge_tickers = {row["ticker"] for row in hedge_positions}
    assert hedge_tickers == {"SPY", "IWM", "TLT"}
    spy = next(row for row in hedge_positions if row["ticker"] == "SPY")
    assert spy["shares"] == 30

    body = response.json()
    assert body["import_summary"]["hedge_imported_count"] == 2
    assert body["import_summary"]["hedge_tickers"] == ["IWM", "SPY"]
    assert len(body["staged_proposals"]) == 2


def test_import_ibkr_flex_route_rejects_non_xml(auth_client):
    response = auth_client.post(
        "/api/portfolio-positions/import/ibkr-flex",
        files={"file": ("positions.txt", BytesIO(b"not xml"), "text/plain")},
    )
    assert response.status_code == 400
