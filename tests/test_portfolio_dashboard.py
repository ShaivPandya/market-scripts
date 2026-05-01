import pandas as pd


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
