from __future__ import annotations

import pandas as pd

from api.routers import sizer


def test_sizer_request_defaults_and_cache_key_include_beta_hedge_mode():
    base_body = {
        "book": 100000,
        "target_leverage": 2.0,
        "positions": [{"ticker": "AAA", "conviction": 3}],
    }

    default_req = sizer.SizerRequest(**base_body)
    spy_req = sizer.SizerRequest(**base_body, beta_hedge_mode="spy")
    qqq_req = sizer.SizerRequest(**base_body, beta_hedge_mode="qqq")
    spy_qqq_req = sizer.SizerRequest(**base_body, beta_hedge_mode="spy_qqq")
    all_three_req = sizer.SizerRequest(**base_body, beta_hedge_mode="spy_iwm_qqq")

    assert default_req.beta_hedge_mode == "spy_iwm"
    assert spy_req.beta_hedge_mode == "spy"
    assert qqq_req.beta_hedge_mode == "qqq"
    assert spy_qqq_req.beta_hedge_mode == "spy_qqq"
    assert all_three_req.beta_hedge_mode == "spy_iwm_qqq"
    assert sizer._cache_key(default_req) != sizer._cache_key(spy_req)
    assert sizer._cache_key(qqq_req) != sizer._cache_key(spy_qqq_req)
    assert sizer._cache_key(spy_qqq_req) != sizer._cache_key(all_three_req)
    assert "beta_hedge_mode=spy_iwm" in sizer._cache_key(default_req)
    assert "beta_hedge_mode=spy" in sizer._cache_key(spy_req)
    assert "beta_hedge_mode=qqq" in sizer._cache_key(qqq_req)
    assert "beta_hedge_mode=spy_qqq" in sizer._cache_key(spy_qqq_req)
    assert "beta_hedge_mode=spy_iwm_qqq" in sizer._cache_key(all_three_req)


def test_sizer_cache_key_is_group_sensitive():
    ungrouped = sizer.SizerRequest(
        book=100000,
        target_leverage=2.0,
        positions=[{"ticker": "AAA", "conviction": 3}],
    )
    grouped = sizer.SizerRequest(
        book=100000,
        target_leverage=2.0,
        positions=[{"ticker": "AAA", "conviction": 3, "group_name": " Memory ", "group_conviction": 5}],
    )

    assert sizer._cache_key(ungrouped) != sizer._cache_key(grouped)
    assert "group=Memory:5" in sizer._cache_key(grouped)


def test_compute_sizer_result_forwards_beta_hedge_mode(monkeypatch):
    from portfolio.portfolio_optimizer import portfolio_sizer

    captured: dict = {}

    def fake_get_data(**kwargs):
        captured.update(kwargs)
        return {
            "error": None,
            "status": "ok",
            "beta_hedge_mode": kwargs["beta_hedge_mode"],
        }

    monkeypatch.setattr(portfolio_sizer, "get_data", fake_get_data)

    req = sizer.SizerRequest(
        book=100000,
        target_leverage=2.0,
        beta_hedge_mode="spy_iwm_qqq",
        positions=[{"ticker": "AAA", "conviction": 3}],
    )

    result = sizer._compute_sizer_result(req)

    assert captured["beta_hedge_mode"] == "spy_iwm_qqq"
    assert result["beta_hedge_mode"] == "spy_iwm_qqq"


def test_compute_sizer_result_forwards_group_fields(monkeypatch):
    from portfolio.portfolio_optimizer import portfolio_sizer

    captured: dict = {}

    def fake_get_data(**kwargs):
        captured.update(kwargs)
        return {"error": None, "status": "ok"}

    monkeypatch.setattr(portfolio_sizer, "get_data", fake_get_data)

    req = sizer.SizerRequest(
        book=100000,
        target_leverage=2.0,
        positions=[
            {"ticker": "SKH", "conviction": 4, "group_name": " Memory ", "group_conviction": 5},
            {"ticker": "MU", "conviction": 3, "group_name": "memory", "group_conviction": 5},
        ],
    )

    sizer._compute_sizer_result(req)

    assert captured["positions"] == [
        {"ticker": "SKH", "conviction": 4, "group_name": "Memory", "group_conviction": 5},
        {"ticker": "MU", "conviction": 3, "group_name": "Memory", "group_conviction": 5},
    ]


def test_sizer_prefill_only_returns_equity_positions(monkeypatch):
    class FakeRuntimeReadService:
        def positions_df(self):
            return pd.DataFrame(
                [
                    {"ticker": "NVDA", "direction": "long", "conviction": 5, "asset": "equity"},
                    {
                        "ticker": "BZ=F",
                        "direction": "long",
                        "conviction": 4,
                        "asset": "commodity",
                        "instrument_type": "future",
                    },
                    {"ticker": "EURUSD=X", "direction": "short", "conviction": 3, "asset": "fx"},
                ]
            )

    monkeypatch.setattr(sizer, "OntologyRuntimeReadService", lambda: FakeRuntimeReadService())
    monkeypatch.setattr(sizer, "get_portfolio_book_size", lambda: 100000.0)

    result = sizer.get_sizer_prefill()

    assert [row["ticker"] for row in result["positions"]] == ["NVDA"]
    assert result["count"] == 1
