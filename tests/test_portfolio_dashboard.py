import pandas as pd


def _import_dashboard(monkeypatch):
    import sys

    from ontology import runtime_read_service

    monkeypatch.setattr(
        runtime_read_service,
        "get_positions_df",
        lambda include_hedges=False: pd.DataFrame(columns=["ticker", "asset", "direction"]),
    )
    monkeypatch.setattr(runtime_read_service, "get_positions", lambda include_hedges=False: [])
    sys.modules.pop("portfolio.portfolio_dashboard", None)
    import portfolio.portfolio_dashboard as dashboard

    return dashboard, runtime_read_service


def test_dashboard_uses_runtime_portfolio_read_adapter(monkeypatch):
    dashboard, runtime_read_service = _import_dashboard(monkeypatch)

    assert dashboard.get_positions is runtime_read_service.get_positions
    assert dashboard.get_positions_df is runtime_read_service.get_positions_df


def test_runtime_positions_use_ontology_source_in_primary_mode(monkeypatch):
    from ontology import runtime_read_service

    monkeypatch.setattr(
        runtime_read_service.OntologyRuntimeReadService,
        "positions",
        lambda self, include_hedges=False: [{"ticker": "NVDA"}],
    )

    assert runtime_read_service.get_positions() == [{"ticker": "NVDA"}]


def test_empty_portfolio_returns_empty_payload_without_yfinance(monkeypatch):
    dashboard, _runtime_read_service = _import_dashboard(monkeypatch)
    called = False

    def fake_download(*args, **kwargs):
        nonlocal called
        called = True
        return pd.DataFrame()

    monkeypatch.setattr(dashboard, "POSITIONS", {})
    monkeypatch.setattr(dashboard, "POSITION_ORDER", [])
    monkeypatch.setattr(dashboard, "POSITION_META", {})
    monkeypatch.setattr(dashboard, "DISPLAY_META", {})
    monkeypatch.setattr(dashboard, "get_positions", lambda: [])
    monkeypatch.setattr(dashboard, "yf_download", fake_download)

    data = dashboard.fetch_portfolio_data("Daily")

    assert "error" not in data
    assert data["positions"] == {}
    assert data["warning"] == "No portfolio positions configured."
    assert called is False


def test_yfinance_empty_result_returns_warning_payload(monkeypatch):
    dashboard, _runtime_read_service = _import_dashboard(monkeypatch)

    monkeypatch.setattr(dashboard, "POSITIONS", {"MU": "MU"})
    monkeypatch.setattr(dashboard, "POSITION_ORDER", ["MU"])
    monkeypatch.setattr(
        dashboard,
        "POSITION_META",
        {"MU": {"asset": "equity", "direction": "long", "position_id": "MU", "display_ticker": "MU"}},
    )
    monkeypatch.setattr(
        dashboard,
        "DISPLAY_META",
        {"MU": {"asset": "equity", "direction": "long", "display_ticker": "MU", "legs": ["MU"]}},
    )
    monkeypatch.setattr(
        dashboard,
        "get_positions",
        lambda: [{"ticker": "MU", "asset": "equity", "direction": "long", "cost_basis": 100.0, "shares": 1.0}],
    )
    monkeypatch.setattr(dashboard, "yf_download", lambda *args, **kwargs: pd.DataFrame())

    data = dashboard.fetch_portfolio_data("Daily")

    assert "error" not in data
    assert data["positions"] == {}
    assert data["metadata"]["MU"]["asset"] == "equity"
    assert data["analytics"]["per_position"]["MU"]["current_price"] is None
    assert data["warning"] == "No data returned from yfinance."


def test_futures_price_symbol_maps_back_to_portfolio_label(monkeypatch):
    dashboard, _runtime_read_service = _import_dashboard(monkeypatch)

    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    raw = pd.DataFrame({("ES=F", "Close"): [5000.0, 5020.0]}, index=dates)
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)
    holding = {
        "ticker": "ES",
        "asset": "equity",
        "direction": "long",
        "cost_basis": 5000.0,
        "quantity": 2.0,
        "shares": 2.0,
        "instrument_type": "future",
        "price_symbol": "ES=F",
        "contract_multiplier": 50.0,
        "position_id": "ES",
        "display_ticker": "ES",
    }

    monkeypatch.setattr(dashboard, "POSITIONS", {"ES": "ES=F"})
    monkeypatch.setattr(dashboard, "POSITION_ORDER", ["ES"])
    monkeypatch.setattr(dashboard, "POSITION_META", {"ES": holding})
    monkeypatch.setattr(dashboard, "DISPLAY_META", {"ES": {**holding, "legs": ["ES"]}})
    monkeypatch.setattr(dashboard, "get_positions", lambda: [holding])
    monkeypatch.setattr(dashboard, "yf_download", lambda *args, **kwargs: raw)

    data = dashboard.fetch_portfolio_data("Daily")

    assert list(data["positions"].keys()) == ["ES"]
    assert data["metadata"]["ES"]["price_symbol"] == "ES=F"
    assert data["metadata"]["ES"]["current_notional"] == 502000.0
    assert data["analytics"]["per_position"]["ES"]["current_price"] == 5020.0
    assert data["analytics"]["per_position"]["ES"]["current_notional"] == 502000.0
    assert data["analytics"]["per_position"]["ES"]["unrealized_pnl_dollar"] == 2000.0
    assert "roll P&L is not modeled" in data["warning"]


def test_underlying_exposures_nets_and_directs_option_legs(monkeypatch):
    dashboard, _ = _import_dashboard(monkeypatch)

    leg_metadata = {
        "META260116C00500000": {
            "display_ticker": "META",
            "ticker": "META",
            "instrument_type": "option",
            "option_type": "call",
            "direction": "long",
            "current_notional": 1200.0,
            "cost_notional": 1000.0,
        },
        "META260116C00520000": {
            "display_ticker": "META",
            "ticker": "META",
            "instrument_type": "option",
            "option_type": "call",
            "direction": "short",
            "current_notional": 500.0,
            "cost_notional": 400.0,
        },
    }

    row = dashboard._underlying_exposures(leg_metadata)[0]

    assert row["underlying_ticker"] == "META"
    assert row["current_notional"] == 1700.0  # gross preserved for diagnostics
    assert row["option_gross_current_notional"] == 1700.0
    assert row["option_net_current_notional"] == 700.0
    assert row["option_net_size_current_notional"] == 700.0
    assert row["net_direction"] == "long"
    assert row["near_zero_net"] is False


def test_underlying_exposures_mixed_equity_option_kept_separate(monkeypatch):
    dashboard, _ = _import_dashboard(monkeypatch)

    leg_metadata = {
        "AAPL": {
            "display_ticker": "AAPL",
            "ticker": "AAPL",
            "instrument_type": "security",
            "direction": "long",
            "current_notional": 1600.0,
            "cost_notional": 1000.0,
        },
        "AAPL260116P00150000": {
            "display_ticker": "AAPL",
            "ticker": "AAPL",
            "instrument_type": "option",
            "option_type": "put",
            "direction": "long",
            "current_notional": 300.0,
            "cost_notional": 250.0,
        },
    }

    row = dashboard._underlying_exposures(leg_metadata)[0]

    # Equity market value and option premium are reported separately, not merged.
    assert row["equity_current_notional"] == 1600.0
    assert row["option_gross_current_notional"] == 300.0
    assert row["option_net_current_notional"] == -300.0  # long put = short exposure
    # A real share leg dictates direction.
    assert row["net_direction"] == "long"
    assert row["near_zero_net"] is False


def test_underlying_exposures_flags_offsetting_straddle(monkeypatch):
    dashboard, _ = _import_dashboard(monkeypatch)

    leg_metadata = {
        "TSLA260116C00250000": {
            "display_ticker": "TSLA",
            "ticker": "TSLA",
            "instrument_type": "option",
            "option_type": "call",
            "direction": "long",
            "current_notional": 1000.0,
            "cost_notional": 1000.0,
        },
        "TSLA260116P00250000": {
            "display_ticker": "TSLA",
            "ticker": "TSLA",
            "instrument_type": "option",
            "option_type": "put",
            "direction": "long",
            "current_notional": 1000.0,
            "cost_notional": 1000.0,
        },
    }

    row = dashboard._underlying_exposures(leg_metadata)[0]

    assert row["option_net_current_notional"] == 0.0
    assert row["near_zero_net"] is True
    assert row["net_direction"] == "neutral"


def test_option_legs_group_under_underlying_chart_tile(monkeypatch):
    dashboard, _runtime_read_service = _import_dashboard(monkeypatch)

    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    raw = pd.DataFrame({("META", "Close"): [500.0, 520.0]}, index=dates)
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)

    share_leg = {
        "ticker": "META",
        "asset": "equity",
        "direction": "long",
        "cost_basis": 400.0,
        "quantity": 10.0,
        "shares": 10.0,
        "instrument_type": "security",
        "position_id": "META",
        "display_ticker": "META",
    }
    call_leg = {
        "ticker": "META",
        "asset": "equity",
        "direction": "long",
        "cost_basis": 12.0,
        "quantity": 2.0,
        "shares": 2.0,
        "instrument_type": "option",
        "underlying_ticker": "META",
        "option_contract_symbol": "META260116C00500000",
        "option_expiration": "2026-01-16",
        "option_strike": 500.0,
        "option_type": "call",
        "price_symbol": "META260116C00500000",
        "contract_multiplier": 100.0,
        "position_id": "META260116C00500000",
        "display_ticker": "META",
    }

    monkeypatch.setattr(dashboard, "POSITIONS", {"META": "META"})
    monkeypatch.setattr(dashboard, "POSITION_ORDER", ["META"])
    monkeypatch.setattr(
        dashboard,
        "POSITION_META",
        {
            "META": share_leg,
            "META260116C00500000": call_leg,
        },
    )
    monkeypatch.setattr(
        dashboard,
        "DISPLAY_META",
        {
            "META": {
                **share_leg,
                "legs": ["META", "META260116C00500000"],
            }
        },
    )
    monkeypatch.setattr(dashboard, "get_positions", lambda: [share_leg, call_leg])
    monkeypatch.setattr(
        dashboard,
        "compute_analytics",
        lambda positions, holdings: {
            "per_position": {
                "META": {"current_price": 520.0, "current_notional": 5200.0, "cost_notional": 4000.0},
                "META260116C00500000": {"current_price": 15.0, "current_notional": 3000.0, "cost_notional": 2400.0},
            },
            "portfolio": {"position_count": 2},
        },
    )
    monkeypatch.setattr(dashboard, "yf_download", lambda *args, **kwargs: raw)

    data = dashboard.fetch_portfolio_data("Daily")

    assert list(data["positions"].keys()) == ["META"]
    assert "META260116C00500000" not in data["positions"]
    assert data["position_order"] == ["META"]
    assert data["metadata"]["META"]["legs"] == ["META", "META260116C00500000"]
    assert data["underlying_exposures"][0]["underlying_ticker"] == "META"
    assert set(data["underlying_exposures"][0]["legs"]) == {"META", "META260116C00500000"}
    assert "charts show the underlying stock" in data["warning"]
