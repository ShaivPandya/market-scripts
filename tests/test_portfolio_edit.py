from __future__ import annotations

import sqlite3

import pytest

import portfolio.core_db as core_db
import portfolio.portfolio_db as portfolio_db
import portfolio.valuation as valuation


@pytest.fixture(autouse=True)
def _use_temp_portfolio_db(tmp_path, monkeypatch):
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "core.db")
    monkeypatch.setattr(portfolio_db, "DB_PATH", tmp_path / "portfolio.db")
    monkeypatch.setattr(
        valuation,
        "_cached_yfinance_metadata",
        lambda symbol: {"currency": "USD", "country": "United States", "exchange": None},
    )
    monkeypatch.setattr(
        valuation,
        "fx_rate_to_base",
        lambda currency, base_currency="USD": {"rate": 1.0, "as_of": "2026-05-05"},
    )
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
            "quantity": 12.0,
            "instrument_type": "security",
            "price_symbol": "SPY",
            "contract_multiplier": 1.0,
            "currency": "USD",
            "country": "United States",
            "exchange": None,
            "base_currency": "USD",
            "fx_rate_to_base": 1.0,
            "fx_rate_as_of": "2026-05-05",
            "cost_basis_base": 510.25,
            "notional_base": 6123.0,
            "valuation_status": "ok",
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
    assert updated_fetch.json()["positions"][0]["quantity"] == 8.5


def test_portfolio_settings_book_size_persists(auth_client):
    default_resp = auth_client.get("/api/v1/portfolio-settings")
    assert default_resp.status_code == 200
    assert default_resp.json()["book_size"] == 100_000.0
    assert default_resp.json()["configured"] is False

    update_resp = auth_client.put("/api/v1/portfolio-settings", json={"book_size": 125_000})
    assert update_resp.status_code == 200
    assert update_resp.json()["book_size"] == 125_000.0
    assert update_resp.json()["configured"] is True

    fetch_resp = auth_client.get("/api/v1/portfolio-settings")
    assert fetch_resp.status_code == 200
    assert fetch_resp.json()["book_size"] == 125_000.0
    assert fetch_resp.json()["configured"] is True


def test_save_and_get_futures_position(auth_client):
    resp = auth_client.put(
        "/api/v1/portfolio-positions",
        json={
            "positions": [
                {
                    "ticker": "ES=F",
                    "instrument_type": "future",
                    "direction": "long",
                    "contrarian": False,
                    "conviction": 4,
                    "cost_basis": 5000,
                    "quantity": 2,
                }
            ],
            "apply": True,
            "approval_note": "Apply in test",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "applied"

    fetch_resp = auth_client.get("/api/v1/portfolio-positions")
    assert fetch_resp.status_code == 200
    position = fetch_resp.json()["positions"][0]
    assert position["ticker"] == "ES=F"
    assert position["price_symbol"] == "ES=F"
    assert position["instrument_type"] == "future"
    assert position["asset"] == "equity"
    assert position["quantity"] == 2.0
    assert position["shares"] == 2.0
    assert position["contract_multiplier"] == 50.0


def test_unsafe_futures_symbol_is_rejected(auth_client):
    resp = auth_client.put(
        "/api/v1/portfolio-positions",
        json={
            "positions": [
                {
                    "ticker": "../ES=F",
                    "instrument_type": "future",
                    "direction": "long",
                    "contrarian": False,
                    "conviction": 4,
                }
            ],
            "apply": True,
            "approval_note": "Apply in test",
        },
    )

    assert resp.status_code == 422


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


def test_portfolio_update_detects_currency_and_persists_base_valuation(auth_client, monkeypatch):
    monkeypatch.setattr(
        valuation,
        "_cached_yfinance_metadata",
        lambda symbol: {"currency": "JPY", "country": "Japan", "exchange": "Tokyo Stock Exchange"},
    )
    monkeypatch.setattr(
        valuation,
        "fx_rate_to_base",
        lambda currency, base_currency="USD": {"rate": 1 / 155, "as_of": "2026-05-05"},
    )

    resp = auth_client.put(
        "/api/v1/portfolio-positions",
        json={
            "positions": [
                {
                    "ticker": "8001.T",
                    "asset": "equity",
                    "direction": "long",
                    "contrarian": False,
                    "conviction": 3,
                    "cost_basis": 8001,
                    "shares": 100,
                }
            ],
        },
    )

    assert resp.status_code == 200
    proposed = resp.json()["proposed_change"]["positions"][0]
    assert proposed["currency"] == "JPY"
    assert proposed["country"] == "Japan"
    assert proposed["valuation_status"] == "ok"
    assert proposed["notional_base"] == pytest.approx(8001 * 100 / 155)

    def _fail_fx_lookup(*_args, **_kwargs):
        raise AssertionError("approval apply must preserve the reviewed FX valuation")

    monkeypatch.setattr(valuation, "fx_rate_to_base", _fail_fx_lookup)
    approval_id = resp.json()["approval_id"]
    approved = auth_client.post(f"/api/v1/approvals/{approval_id}/approve", json={"note": "Apply in test"})
    assert approved.status_code == 200
    stored = portfolio_db.get_positions()[0]
    assert stored["currency"] == "JPY"
    assert stored["notional_base"] == pytest.approx(8001 * 100 / 155)


def test_explicit_currency_override_is_preserved(auth_client, monkeypatch):
    monkeypatch.setattr(
        valuation,
        "fx_rate_to_base",
        lambda currency, base_currency="USD": {"rate": 0.75, "as_of": "2026-05-05"},
    )

    resp = auth_client.put(
        "/api/v1/portfolio-positions",
        json={
            "positions": [
                {
                    "ticker": "TEST.L",
                    "asset": "equity",
                    "direction": "long",
                    "contrarian": False,
                    "conviction": 3,
                    "cost_basis": 100,
                    "shares": 10,
                    "currency": "GBP",
                    "country": "United Kingdom",
                    "exchange": "Manual Exchange",
                }
            ],
        },
    )

    assert resp.status_code == 200
    proposed = resp.json()["proposed_change"]["positions"][0]
    assert proposed["currency"] == "GBP"
    assert proposed["country"] == "United Kingdom"
    assert proposed["exchange"] == "Manual Exchange"
    assert proposed["notional_base"] == pytest.approx(750)


def test_sqlite_init_backfills_legacy_position_columns():
    conn = sqlite3.connect(portfolio_db.DB_PATH)
    conn.execute(
        """
        CREATE TABLE positions (
            ticker TEXT PRIMARY KEY,
            asset TEXT NOT NULL,
            direction TEXT NOT NULL,
            contrarian INTEGER NOT NULL DEFAULT 0,
            conviction INTEGER NOT NULL DEFAULT 3,
            cost_basis REAL,
            shares REAL,
            role TEXT NOT NULL DEFAULT 'position'
        )
        """
    )
    conn.execute(
        """
        INSERT INTO positions (
            ticker, asset, direction, contrarian, conviction, cost_basis, shares, role
        )
        VALUES ('MU', 'equity', 'long', 0, 4, 100.0, 12.0, 'position')
        """
    )
    conn.commit()
    conn.close()

    rows = portfolio_db.get_positions()

    assert rows == [
        {
            "ticker": "MU",
            "asset": "equity",
            "direction": "long",
            "contrarian": 0,
            "conviction": 4,
            "cost_basis": 100.0,
            "shares": 12.0,
            "quantity": 12.0,
            "instrument_type": "security",
            "price_symbol": "MU",
            "contract_multiplier": 1.0,
            "currency": None,
            "country": None,
            "exchange": None,
            "base_currency": "USD",
            "fx_rate_to_base": None,
            "fx_rate_as_of": None,
            "cost_basis_base": None,
            "notional_base": None,
            "valuation_status": "missing_position_inputs",
            "role": "position",
        }
    ]
