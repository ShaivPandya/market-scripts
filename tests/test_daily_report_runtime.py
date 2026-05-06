from __future__ import annotations

import pandas as pd
import pytest

import portfolio.portfolio_db as portfolio_db
from auto_report import auto_daily_report
from macro.economic_growth.economic_growth import calculate_equity_relative_returns, calculate_return
from portfolio.portfolio_optimizer import portfolio_sizer as portfolio_sizer_module


def test_run_sizer_imports_packaged_module(monkeypatch):
    captured = {}

    def fake_size_portfolio(*, positions, book, target_leverage):
        captured["positions"] = positions
        captured["book"] = book
        captured["target_leverage"] = target_leverage
        return {"weights_df": None}

    monkeypatch.setattr(portfolio_sizer_module, "size_portfolio", fake_size_portfolio)

    portfolio_df = pd.DataFrame(
        [
            {"ticker": "MU", "direction": "long", "conviction": 3},
            {"ticker": "OKLO", "direction": "short", "conviction": 5},
        ]
    )

    result = auto_daily_report.run_sizer(portfolio_df, 100_000.0, target_leverage=1.25)

    assert result == {"weights_df": None}
    assert captured == {
        "positions": [{"ticker": "MU", "conviction": 3}, {"ticker": "OKLO", "conviction": 5}],
        "book": 100_000.0,
        "target_leverage": 1.25,
    }


def test_load_configured_book_size_prefers_github_env(monkeypatch):
    monkeypatch.setenv("TALISMAN_BOOK_SIZE", "125000")

    assert auto_daily_report.load_configured_book_size() == 125_000.0


def test_compute_adjustments_uses_current_hedge_shares():
    weights_df = pd.DataFrame(
        [
            {"ticker": "CRWD", "shares": 9, "price": 422.8, "direction": "long"},
        ]
    )
    hedges_df = pd.DataFrame(
        [
            {"ticker": "SPY", "shares": -34, "price": 718.0, "current_shares": -107},
            {"ticker": "IWM", "shares": 24, "price": 278.0, "current_shares": 0},
        ]
    )
    portfolio_df = pd.DataFrame(
        [
            {"ticker": "CRWD", "direction": "long", "shares": 18},
        ]
    )

    result = auto_daily_report.compute_adjustments(
        {"weights_df": weights_df, "hedges_df": hedges_df},
        portfolio_df,
    )
    by_ticker = result.set_index("ticker")

    assert by_ticker.loc["SPY", "current_shares"] == -107
    assert by_ticker.loc["SPY", "delta"] == 73
    assert by_ticker.loc["SPY", "action"] == "BUY"
    assert by_ticker.loc["IWM", "current_shares"] == 0


def test_get_positions_df_falls_back_to_csv(tmp_path):
    csv_path = tmp_path / "portfolio.csv"
    csv_path.write_text("ticker,asset,direction,contrarian,conviction\nMU,equity,long,false,3\n", encoding="utf-8")

    original_db_path = portfolio_db.DB_PATH
    original_csv_path = portfolio_db.CSV_PATH
    original_conn = portfolio_db._conn

    if original_conn is not None:
        try:
            original_conn.close()
        except Exception:
            pass

    try:
        portfolio_db.DB_PATH = tmp_path / "portfolio.db"
        portfolio_db.CSV_PATH = csv_path
        portfolio_db._conn = None

        df = portfolio_db.get_positions_df(fallback_to_csv=True)

        assert list(df["ticker"]) == ["MU"]
        assert list(df["asset"]) == ["equity"]
        assert list(df["direction"]) == ["long"]
        assert list(df["contrarian"]) == [False]
        assert list(df["role"]) == ["position"]
    finally:
        if portfolio_db._conn is not None:
            try:
                portfolio_db._conn.close()
            except Exception:
                pass
        portfolio_db._conn = original_conn
        portfolio_db.DB_PATH = original_db_path
        portfolio_db.CSV_PATH = original_csv_path


def test_calculate_return_handles_lower_precision_datetime_index():
    index = pd.date_range("2024-01-01", periods=3, freq="D").as_unit("s")
    close_series = pd.Series([100.0, 110.0, 121.0], index=index)

    result = calculate_return(
        close_series,
        1,
        reference_time=pd.Timestamp("2024-01-03 12:34:56.123456"),
    )

    assert result == pytest.approx(10.0)


def test_calculate_equity_relative_returns_uses_row_benchmarks():
    results = {
        "S&P 500": {"1-mo": 5.0, "3-mo": 4.0},
        "Russell 2000": {"1-mo": 7.5, "3-mo": 3.0},
        "STOXX 600": {"1-mo": 2.0, "3-mo": 1.0},
        "Europe Banks": {"1-mo": 6.0, "3-mo": -1.0},
    }

    relative = calculate_equity_relative_returns(results, ["1-mo", "3-mo"])

    assert relative["S&P 500"] == {"1-mo": None, "3-mo": None}
    assert relative["STOXX 600"] == {"1-mo": None, "3-mo": None}
    assert relative["Russell 2000"] == {"1-mo": 2.5, "3-mo": -1.0}
    assert relative["Europe Banks"] == {"1-mo": 4.0, "3-mo": -2.0}
