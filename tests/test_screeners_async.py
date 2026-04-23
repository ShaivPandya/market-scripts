import threading
import time


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
    assert started.status_code == 200
    job_id = started.json()["job_id"]
    done = _poll(auth_client, "/api/v1/short-screen/async", job_id)
    assert done["status"] == "done"
    assert done["result"]["results_df"] == [{"Ticker": "AAA"}]

    cached = auth_client.post("/api/v1/short-screen/async", json=body)
    assert cached.status_code == 200
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
    assert first.status_code == 200
    assert started_compute.wait(timeout=2)

    second = auth_client.post("/api/v1/short-screen/async", json=body)
    assert second.status_code == 200
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
    assert started.status_code == 200
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
