from __future__ import annotations

from api.routers import sizer


def test_sizer_request_defaults_and_cache_key_include_beta_hedge_mode():
    base_body = {
        "book": 100000,
        "target_leverage": 2.0,
        "positions": [{"ticker": "AAA", "conviction": 3}],
    }

    default_req = sizer.SizerRequest(**base_body)
    spy_req = sizer.SizerRequest(**base_body, beta_hedge_mode="spy")

    assert default_req.beta_hedge_mode == "spy_iwm"
    assert spy_req.beta_hedge_mode == "spy"
    assert sizer._cache_key(default_req) != sizer._cache_key(spy_req)
    assert "beta_hedge_mode=spy_iwm" in sizer._cache_key(default_req)
    assert "beta_hedge_mode=spy" in sizer._cache_key(spy_req)


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
        beta_hedge_mode="spy",
        positions=[{"ticker": "AAA", "conviction": 3}],
    )

    result = sizer._compute_sizer_result(req)

    assert captured["beta_hedge_mode"] == "spy"
    assert result["beta_hedge_mode"] == "spy"
