import threading
import time

import pandas as pd
import pytest


def _poll(client, path: str, job_id: str, timeout_s: float = 3.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = client.get(f"{path}/{job_id}")
        assert resp.status_code == 200
        payload = resp.json()
        if payload["status"] in ("done", "error"):
            return payload
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


def test_short_screen_async_returns_result_and_cache(auth_client, monkeypatch):
    from api import cache
    from api.routers import short_screen as router

    cache.invalidate_all()
    calls = {"n": 0}

    def fake_compute(req, progress_callback=None):
        calls["n"] += 1
        if progress_callback:
            progress_callback("prices", 1, 2)
        return {
            "results_df": [{"Ticker": "AAA"}],
            "failed_tickers": [],
            "phase1_count": 2,
            "phase1_pass_count": 1,
            "final_count": 1,
        }

    monkeypatch.setattr(router, "_compute_short_screen", fake_compute)

    body = {
        "input_mode": "Custom Tickers",
        "tickers": "AAA,BBB",
        "universe": "Russell 2000",
        "pb_threshold": None,
        "loss_type": None,
        "check_issuance": False,
        "check_revenue": False,
        "max_revenue_growth": 0,
        "check_eps": False,
        "max_eps_growth": 0,
        "check_52w_positive": False,
        "check_min_drawdown": False,
        "min_drawdown_pct": 25,
        "check_max_drawdown": False,
        "max_drawdown_pct": 60,
        "check_3m_neg_momentum": False,
        "check_2m_neg_rel_momentum": False,
        "rel_momentum_benchmark": "IWM",
    }

    started = auth_client.post("/api/v1/short-screen/async", json=body)
    assert started.status_code in (200, 202)
    job_id = started.json()["job_id"]
    done = _poll(auth_client, "/api/v1/short-screen/async", job_id)
    assert done["status"] == "done"
    assert done["result"]["results_df"] == [{"Ticker": "AAA"}]

    cached = auth_client.post("/api/v1/short-screen/async", json=body)
    assert cached.status_code in (200, 202)
    assert cached.json()["status"] == "done"
    assert cached.json()["result"]["results_df"] == [{"Ticker": "AAA"}]
    assert calls["n"] == 1


def test_short_screen_async_dedupes_running_job(auth_client, monkeypatch):
    from api import cache
    from api.routers import short_screen as router

    cache.invalidate_all()
    started_compute = threading.Event()
    release_compute = threading.Event()

    def slow_compute(req, progress_callback=None):
        started_compute.set()
        assert release_compute.wait(timeout=2)
        return {
            "results_df": [],
            "failed_tickers": [],
            "phase1_count": 1,
            "phase1_pass_count": 0,
            "final_count": 0,
        }

    monkeypatch.setattr(router, "_compute_short_screen", slow_compute)

    body = {
        "input_mode": "Custom Tickers",
        "tickers": "AAA",
        "universe": "Russell 2000",
        "pb_threshold": None,
        "loss_type": None,
        "check_issuance": False,
        "check_revenue": False,
        "max_revenue_growth": 0,
        "check_eps": False,
        "max_eps_growth": 0,
        "check_52w_positive": False,
        "check_min_drawdown": False,
        "min_drawdown_pct": 25,
        "check_max_drawdown": False,
        "max_drawdown_pct": 60,
        "check_3m_neg_momentum": False,
        "check_2m_neg_rel_momentum": False,
        "rel_momentum_benchmark": "IWM",
    }

    first = auth_client.post("/api/v1/short-screen/async", json=body)
    assert first.status_code in (200, 202)
    assert started_compute.wait(timeout=2)

    second = auth_client.post("/api/v1/short-screen/async", json=body)
    assert second.status_code in (200, 202)
    assert second.json()["job_id"] == first.json()["job_id"]

    release_compute.set()
    done = _poll(auth_client, "/api/v1/short-screen/async", first.json()["job_id"])
    assert done["status"] == "done"


def test_short_screen_async_surfaces_worker_error(auth_client, monkeypatch):
    from api import cache
    from api.routers import short_screen as router

    cache.invalidate_all()

    def failing_compute(req, progress_callback=None):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(router, "_compute_short_screen", failing_compute)

    body = {
        "input_mode": "Custom Tickers",
        "tickers": "AAA",
        "universe": "Russell 2000",
        "pb_threshold": None,
        "loss_type": None,
        "check_issuance": False,
        "check_revenue": False,
        "max_revenue_growth": 0,
        "check_eps": False,
        "max_eps_growth": 0,
        "check_52w_positive": False,
        "check_min_drawdown": False,
        "min_drawdown_pct": 25,
        "check_max_drawdown": False,
        "max_drawdown_pct": 60,
        "check_3m_neg_momentum": False,
        "check_2m_neg_rel_momentum": False,
        "rel_momentum_benchmark": "IWM",
    }

    started = auth_client.post("/api/v1/short-screen/async", json=body)
    assert started.status_code in (200, 202)
    done = _poll(auth_client, "/api/v1/short-screen/async", started.json()["job_id"])
    assert done["status"] == "error"
    assert "rate limited" in done["error"]


def test_price_filter_prefilters_before_short_fundamentals(monkeypatch):
    from equities.short_screen import short_screen as screen

    seen_fundamental_universe = []

    def fake_price_filter(passers, **kwargs):
        return passers[:1], {passers[0]["ticker"]: {"return_3m": -5.0}}

    def fake_fundamentals(universe, **kwargs):
        seen_fundamental_universe.extend(universe)
        return (
            [
                {
                    "ticker": universe[0],
                    "company_name": "",
                    "price_to_book": 4.0,
                    "gross_profit": -1.0,
                    "operating_income": -1.0,
                    "market_cap": 100.0,
                }
            ],
            [],
        )

    monkeypatch.setattr(screen, "_apply_price_filters", fake_price_filter)
    monkeypatch.setattr(screen, "_screen_short_fundamentals", fake_fundamentals)
    monkeypatch.setattr(screen, "_throttled_yf_call", lambda label, fn: 1.0)

    result = screen.get_data(
        ["AAA", "BBB", "CCC"],
        pb_threshold=3.0,
        loss_type="Gross Loss",
        check_3m_neg_momentum=True,
    )

    assert seen_fundamental_universe == ["AAA"]
    assert result["phase3_pass_count"] == 1
    assert result["final_count"] == 1


def test_price_only_short_screen_skips_fundamentals(monkeypatch):
    from equities.short_screen import short_screen as screen

    def fake_price_filter(passers, **kwargs):
        return passers[:2], {p["ticker"]: {"return_3m": -3.0} for p in passers[:2]}

    monkeypatch.setattr(screen, "_apply_price_filters", fake_price_filter)
    monkeypatch.setattr(screen, "_screen_short_fundamentals", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(screen, "_throttled_yf_call", lambda label, fn: 1.0)

    result = screen.get_data(
        ["AAA", "BBB", "CCC"],
        pb_threshold=None,
        loss_type=None,
        check_3m_neg_momentum=True,
    )

    assert result["phase1_pass_count"] is None
    assert result["phase3_pass_count"] == 2
    assert result["final_count"] == 2


def test_price_momentum_analyze_ticker_returns_raw_roc63():
    from portfolio.momentum.price_momentum.momentum import analyze_ticker

    dates = pd.bdate_range("2025-01-01", periods=100)
    prices = pd.Series(range(100, 200), index=dates, dtype="float64")
    benchmark = pd.Series(100.0, index=dates)

    result = analyze_ticker("AAA", benchmark, years=1, ticker_prices=prices)

    assert result is not None
    expected = (prices.iloc[-1] / prices.shift(63).iloc[-1] - 1.0) * 100.0
    assert result["roc63"] == pytest.approx(expected)


def test_price_momentum_router_resolves_benchmark_and_custom_tickers():
    from api.routers import price_momentum as router

    custom = router.PriceMomentumRequest(
        input_mode="Custom Tickers",
        tickers="bbb, AAA, aaa",
        benchmark="Same as Input",
    )
    sector = router.PriceMomentumRequest(
        input_mode="Universe",
        universe="VGT — Technology",
        benchmark="Same as Input",
    )
    selected = router.PriceMomentumRequest(benchmark="Russell 2000")

    assert router._resolve_tickers(custom) == ["BBB", "AAA"]
    assert router._resolve_benchmark_ticker(custom.benchmark, custom.universe, custom.input_mode) == "SPY"
    assert router._resolve_benchmark_ticker(sector.benchmark, sector.universe, sector.input_mode) == "VGT"
    assert router._resolve_benchmark_ticker(selected.benchmark, selected.universe, selected.input_mode) == "IWM"
    assert '"tickers":"AAA,BBB"' in router._cache_key(custom)


def test_price_momentum_async_returns_result_and_cache(auth_client, monkeypatch):
    from api import cache
    from api.routers import price_momentum as router

    cache.invalidate_all()
    calls = {"n": 0}

    def fake_compute(req, progress_callback=None):
        calls["n"] += 1
        if progress_callback:
            progress_callback("prices", 1, 1)
        return {
            "results_df": [{"ticker": "AAA", "roc63": 12.3, "benchmark": "SPY"}],
            "failed_tickers": [],
            "input_count": 1,
            "scored_count": 1,
            "benchmark_name": "SPY",
            "date": "2026-05-01",
            "final_count": 1,
        }

    monkeypatch.setattr(router, "_compute_price_momentum", fake_compute)

    body = {
        "input_mode": "Custom Tickers",
        "tickers": "AAA",
        "universe": "S&P 500",
        "benchmark": "Same as Input",
    }

    started = auth_client.post("/api/v1/price-momentum/async", json=body)
    assert started.status_code in (200, 202)
    job_id = started.json()["job_id"]
    done = _poll(auth_client, "/api/v1/price-momentum/async", job_id)
    assert done["status"] == "done"
    assert done["result"]["results_df"] == [{"ticker": "AAA", "roc63": 12.3, "benchmark": "SPY"}]

    cached = auth_client.post("/api/v1/price-momentum/async", json=body)
    assert cached.status_code in (200, 202)
    assert cached.json()["status"] == "done"
    assert cached.json()["result"]["scored_count"] == 1
    assert calls["n"] == 1


def test_existing_momentum_endpoint_hides_raw_roc63(monkeypatch):
    from api import cache
    from api.routers import momentum as router
    from portfolio.momentum.price_momentum import momentum

    cache.invalidate_all()
    monkeypatch.setattr(
        momentum,
        "get_data",
        lambda: {
            "results": [
                {
                    "ticker": "AAA",
                    "roc63": 9.0,
                    "avg20_roc63": 7.0,
                    "avg20_vol_roc63": 5.0,
                }
            ]
        },
    )

    result = router.get_momentum()

    assert result["results"] == [{"ticker": "AAA", "avg20_roc63": 7.0}]


def test_quarterly_financials_only_requested_for_growth_filters(monkeypatch):
    from equities.short_screen import short_screen as screen

    calls = []

    def fake_fetch(ticker, *, include_quarterly=False, **kwargs):
        calls.append(include_quarterly)
        return {
            "ticker": ticker,
            "price_to_book": 4.0,
            "gross_profit": -1.0,
            "operating_income": -1.0,
            "market_cap": 100.0,
            "rev_yoy_avg": -5.0,
            "eps_yoy_avg": -5.0,
        }

    monkeypatch.setattr(screen, "fetch_yf_data", fake_fetch)

    screen.screen_ticker("AAA", None, None)
    screen.screen_ticker("AAA", None, None, check_revenue=True)
    screen.screen_ticker("AAA", None, None, check_eps=True)

    assert calls == [False, True, True]


def test_long_screen_ebit_multiple_filter(monkeypatch):
    from equities.long_screen import long_screen as screen

    calls = []
    records = {
        "PASS": {"market_cap": 1_000.0, "operating_income": 100.0},
        "RICH": {"market_cap": 2_500.0, "operating_income": 100.0},
        "ZERO": {"market_cap": 1_000.0, "operating_income": 0.0},
        "NEG": {"market_cap": 1_000.0, "operating_income": -100.0},
        "MISSING": {"market_cap": 1_000.0, "operating_income": float("nan")},
    }

    def fake_fetch(ticker, **kwargs):
        calls.append(kwargs)
        return {"ticker": ticker, **records[ticker]}

    monkeypatch.setattr(screen, "fetch_yf_data", fake_fetch)

    passes, data = screen.screen_ticker_long(
        "PASS",
        pb_threshold=None,
        profit_type=None,
        check_ebit_multiple=True,
        max_ebit_multiple=20.0,
    )
    assert passes is True
    assert data["ebit_multiple"] == pytest.approx(10.0)
    assert calls[-1]["need_profit"] is True
    assert calls[-1]["need_market_cap"] is True

    for ticker in ("RICH", "ZERO", "NEG", "MISSING"):
        passes, _ = screen.screen_ticker_long(
            ticker,
            pb_threshold=None,
            profit_type=None,
            check_ebit_multiple=True,
            max_ebit_multiple=20.0,
        )
        assert passes is False


def test_long_screen_ebit_multiple_requires_fundamentals():
    from equities.long_screen import long_screen as screen

    assert screen._needs_long_fundamentals(
        pb_threshold=None,
        profit_type=None,
        check_revenue=False,
        check_eps=False,
        check_ebit_multiple=True,
    )


def test_long_screen_router_forwards_ebit_multiple(monkeypatch):
    from api.routers import long_screen as router
    from equities.long_screen import long_screen as screen

    captured = {}

    def fake_get_data(**kwargs):
        captured.update(kwargs)
        return {
            "results_df": [],
            "failed_tickers": [],
            "phase1_count": 1,
            "phase1_pass_count": 0,
            "final_count": 0,
        }

    monkeypatch.setattr(router, "_resolve_tickers", lambda req: ["AAA"])
    monkeypatch.setattr(screen, "get_data", fake_get_data)

    req = router.LongScreenRequest(
        input_mode="Custom Tickers",
        tickers="AAA",
        pb_threshold=None,
        profit_type=None,
        check_ebit_multiple=True,
        max_ebit_multiple=20.0,
    )
    result = router._compute_long_screen(req)

    assert result["final_count"] == 0
    assert captured["check_ebit_multiple"] is True
    assert captured["max_ebit_multiple"] == 20.0
