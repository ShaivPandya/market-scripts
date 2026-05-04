from __future__ import annotations

import pytest

import portfolio.core_db as core_db
import portfolio.portfolio_db as portfolio_db


@pytest.fixture(autouse=True)
def _use_temp_portfolio_db(tmp_path, monkeypatch):
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "core.db")
    monkeypatch.setattr(portfolio_db, "DB_PATH", tmp_path / "portfolio.db")
    if core_db._conn:
        try:
            core_db._conn.close()
        except Exception:
            pass
    if portfolio_db._conn:
        try:
            portfolio_db._conn.close()
        except Exception:
            pass
    monkeypatch.setattr(core_db, "_conn", None)
    monkeypatch.setattr(portfolio_db, "_conn", None)
    yield
    if core_db._conn:
        try:
            core_db._conn.close()
        except Exception:
            pass
    if portfolio_db._conn:
        try:
            portfolio_db._conn.close()
        except Exception:
            pass
    monkeypatch.setattr(core_db, "_conn", None)
    monkeypatch.setattr(portfolio_db, "_conn", None)


def test_save_and_get_hedge_positions(auth_client):
    create_resp = auth_client.put(
        "/api/v1/hedge-positions",
        json={
            "positions": [
                {"ticker": "spy", "direction": "short", "cost_basis": 510.25, "shares": 12},
            ],
            "apply": True,
            "approval_note": "Apply in test",
        },
    )
    assert create_resp.status_code == 200
    assert create_resp.json()["status"] == "applied"

    fetch_resp = auth_client.get("/api/v1/hedge-positions")
    assert fetch_resp.status_code == 200
    assert fetch_resp.json()["positions"] == [
        {
            "ticker": "SPY",
            "asset": "equity",
            "direction": "short",
            "contrarian": False,
            "conviction": 3,
            "cost_basis": 510.25,
            "shares": 12.0,
            "role": "hedge",
        }
    ]

    update_resp = auth_client.put(
        "/api/v1/hedge-positions",
        json={
            "positions": [
                {"ticker": "SPY", "direction": "long", "cost_basis": 501.0, "shares": 8.5},
            ],
            "apply": True,
            "approval_note": "Apply in test",
        },
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["status"] == "applied"

    updated_fetch = auth_client.get("/api/v1/hedge-positions")
    assert updated_fetch.status_code == 200
    assert updated_fetch.json()["positions"][0]["direction"] == "long"
    assert updated_fetch.json()["positions"][0]["cost_basis"] == 501.0
    assert updated_fetch.json()["positions"][0]["shares"] == 8.5


def test_clear_all_hedge_positions(auth_client):
    auth_client.put(
        "/api/v1/hedge-positions",
        json={
            "positions": [
                {"ticker": "SPY", "direction": "short", "cost_basis": None, "shares": 5},
            ],
            "apply": True,
            "approval_note": "Apply in test",
        },
    )

    clear_resp = auth_client.put(
        "/api/v1/hedge-positions",
        json={"positions": [], "apply": True, "approval_note": "Apply in test"},
    )
    assert clear_resp.status_code == 200
    assert clear_resp.json()["status"] == "applied"

    fetch_resp = auth_client.get("/api/v1/hedge-positions")
    assert fetch_resp.status_code == 200
    assert fetch_resp.json() == {"positions": []}


def test_duplicate_hedge_tickers_fail(auth_client):
    resp = auth_client.put(
        "/api/v1/hedge-positions",
        json={
            "positions": [
                {"ticker": "SPY", "direction": "short", "cost_basis": None, "shares": 2},
                {"ticker": "spy", "direction": "long", "cost_basis": None, "shares": 1},
            ],
            "apply": True,
            "approval_note": "Apply in test",
        },
    )
    assert resp.status_code == 400
    assert "Duplicate ticker" in resp.json()["detail"]


def test_hedge_ticker_collision_with_position_returns_409(auth_client):
    position_resp = auth_client.put(
        "/api/v1/portfolio-positions",
        json={
            "positions": [
                {
                    "ticker": "SPY",
                    "asset": "equity",
                    "direction": "long",
                    "contrarian": False,
                    "conviction": 3,
                    "cost_basis": None,
                    "shares": 10,
                }
            ],
            "apply": True,
            "approval_note": "Apply in test",
        },
    )
    assert position_resp.status_code == 200

    hedge_resp = auth_client.put(
        "/api/v1/hedge-positions",
        json={
            "positions": [
                {"ticker": "SPY", "direction": "short", "cost_basis": None, "shares": 3},
            ]
        },
    )
    assert hedge_resp.status_code == 200
    approval_id = hedge_resp.json()["approval_id"]
    approve_resp = auth_client.post(f"/api/v1/approvals/{approval_id}/approve", json={"note": "Apply in test"})
    assert approve_resp.status_code == 409
    assert "already exist as portfolio positions" in str(approve_resp.json())


def test_hedge_position_update_stages_by_default(auth_client):
    resp = auth_client.put(
        "/api/v1/hedge-positions",
        json={
            "positions": [
                {"ticker": "spy", "direction": "short", "cost_basis": 510.25, "shares": 12},
            ]
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "pending_approval_created"
    assert portfolio_db.get_hedge_positions() == []


def test_normalized_rows_use_bool_for_contrarian():
    rows = portfolio_db._normalize_position_rows(
        [
            {
                "ticker": "NVDA",
                "asset": "equity",
                "direction": "long",
                "contrarian": True,
                "conviction": 4,
                "cost_basis": 210.5,
                "shares": 20,
            }
        ],
        role="position",
    )

    assert rows[0][3] is True
