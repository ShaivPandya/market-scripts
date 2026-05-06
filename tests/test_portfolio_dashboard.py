import pandas as pd


def test_dashboard_uses_runtime_portfolio_read_adapter():
    import portfolio.portfolio_dashboard as dashboard
    from ontology import runtime_read_service

    assert dashboard.get_positions is runtime_read_service.get_positions
    assert dashboard.get_positions_df is runtime_read_service.get_positions_df


def test_runtime_positions_use_ontology_source_in_primary_mode(monkeypatch):
    from ontology import runtime_read_service
    from portfolio import portfolio_db

    monkeypatch.setenv("ONTOLOGY_PRIMARY_WRITES", "true")
    monkeypatch.setattr(portfolio_db, "get_positions", lambda **kwargs: [{"ticker": "NUAI"}])
    monkeypatch.setattr(
        runtime_read_service.OntologyRuntimeReadService,
        "positions",
        lambda self, include_hedges=False: [{"ticker": "NVDA"}],
    )

    assert runtime_read_service.get_positions() == [{"ticker": "NVDA"}]


def test_empty_portfolio_returns_empty_payload_without_yfinance(monkeypatch):
    import portfolio.portfolio_dashboard as dashboard

    called = False

    def fake_download(*args, **kwargs):
        nonlocal called
        called = True
        return pd.DataFrame()

    monkeypatch.setattr(dashboard, "POSITIONS", {})
    monkeypatch.setattr(dashboard, "POSITION_ORDER", [])
    monkeypatch.setattr(dashboard, "POSITION_META", {})
    monkeypatch.setattr(dashboard, "get_positions", lambda: [])
    monkeypatch.setattr(dashboard, "yf_download", fake_download)

    data = dashboard.fetch_portfolio_data("Daily")

    assert "error" not in data
    assert data["positions"] == {}
    assert data["warning"] == "No portfolio positions configured."
    assert called is False


def test_yfinance_empty_result_returns_warning_payload(monkeypatch):
    import portfolio.portfolio_dashboard as dashboard

    monkeypatch.setattr(dashboard, "POSITIONS", {"MU": "MU"})
    monkeypatch.setattr(dashboard, "POSITION_ORDER", ["MU"])
    monkeypatch.setattr(dashboard, "POSITION_META", {"MU": {"asset": "equity", "direction": "long"}})
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
    import portfolio.portfolio_dashboard as dashboard

    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    raw = pd.DataFrame(
        {("ES=F", "Close"): [5000.0, 5020.0]},
        index=dates,
    )
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
    }

    monkeypatch.setattr(dashboard, "POSITIONS", {"ES": "ES=F"})
    monkeypatch.setattr(dashboard, "POSITION_ORDER", ["ES"])
    monkeypatch.setattr(dashboard, "POSITION_META", {"ES": holding})
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
