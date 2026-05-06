from __future__ import annotations

from contextlib import contextmanager

import pytest


@pytest.fixture(autouse=True)
def _position_risk_state(tmp_path, monkeypatch):
    import portfolio.portfolio_db as portfolio_db
    from api import position_risk_store, snapshot_store

    monkeypatch.setenv("STATE_DB_BACKEND", "sqlite")
    monkeypatch.setattr(snapshot_store, "_SQLITE_PATH", tmp_path / "computed_snapshots.sqlite3")
    monkeypatch.setattr(position_risk_store, "_SQLITE_PATH", tmp_path / "position_risk.sqlite3")
    if portfolio_db._conn is not None:
        portfolio_db._conn.close()
    monkeypatch.setattr(portfolio_db, "DB_PATH", tmp_path / "portfolio.db")
    monkeypatch.setattr(portfolio_db, "_conn", None)
    monkeypatch.setattr(
        "ontology.sources.macro.EconomicGrowthAdapter.fetch",
        lambda self: _economic_growth_payload("2099-01-01"),
    )
    monkeypatch.setattr(
        "ontology.sources.macro.LaborMarketAdapter.fetch",
        lambda self: _labor_market_payload("2099-01-01"),
    )
    monkeypatch.setattr(
        "ontology.sources.macro.SentimentAdapter.fetch",
        lambda self: _sentiment_payload("2099-01-01"),
    )
    monkeypatch.setattr(
        "ontology.sources.macro.PositioningAdapter.fetch",
        lambda self: _positioning_payload("2099-01-01"),
    )
    portfolio_db.save_positions(
        [
            {
                "ticker": "MU",
                "asset": "equity",
                "direction": "long",
                "shares": 42,
                "conviction": 4,
            }
        ]
    )
    yield
    if portfolio_db._conn is not None:
        portfolio_db._conn.close()
    monkeypatch.setattr(portfolio_db, "_conn", None)


def test_position_risk_accepts_fresh_cached_modules(monkeypatch):
    from api import position_risk as pr
    from api.position_risk_store import read_latest_position_risk

    _seed_required_snapshots(as_of="2099-01-01")
    monkeypatch.setattr(
        pr.SectorMapper, "resolve_sector", lambda self, ticker, asset: _Sector("Information Technology")
    )

    snapshot = pr.refresh_position_risk("MU")

    assert snapshot["ticker"] == "MU"
    assert snapshot["risk_level"] in {"low", "medium", "high"}
    assert snapshot["source_status"]["liquidity"]["accepted"] is True
    assert snapshot["source_status"]["market_breadth"].get("refreshed") is not True
    assert read_latest_position_risk("MU")["result_id"] == snapshot["result_id"]


def test_position_risk_refresh_returns_fresh_cached_snapshot(monkeypatch):
    from api import position_risk as pr

    _seed_required_snapshots(as_of="2099-01-01")
    monkeypatch.setattr(
        pr.SectorMapper, "resolve_sector", lambda self, ticker, asset: _Sector("Information Technology")
    )

    snapshot = pr.refresh_position_risk("MU")

    def fail_bundle_load(*args, **kwargs):
        raise AssertionError("fresh position risk snapshot should be reused")

    monkeypatch.setattr(pr, "load_global_risk_input_bundle", fail_bundle_load)

    cached = pr.refresh_position_risk("MU")

    assert cached["result_id"] == snapshot["result_id"]
    assert cached["_meta"]["cache_status"] == "hit"


def test_position_risk_accepts_legacy_sector_metrics_without_sector(monkeypatch):
    from api import position_risk as pr
    from api.snapshot_keys import SNAPSHOT_SECTOR_METRICS
    from api.snapshot_store import write_snapshot_success

    _seed_required_snapshots(as_of="2099-01-01")
    write_snapshot_success(
        SNAPSHOT_SECTOR_METRICS,
        {
            "timestamp": "2099-01-01T21:00:00",
            "weights_df": [{"Weight_Now": 17.8, "RelPerf_3M_pp": -6.9, "Pct_Above_200DMA": 2.4}],
        },
        as_of_date="2099-01-01",
    )
    monkeypatch.setattr(
        pr.SectorMapper, "resolve_sector", lambda self, ticker, asset: _Sector("Communication Services")
    )
    monkeypatch.setattr(
        "ontology.sources.sector_metrics.SectorMetricsAdapter.fetch",
        lambda self: (_ for _ in ()).throw(AssertionError("sector metrics live fetch should not run")),
    )

    snapshot = pr.refresh_position_risk("MU")

    assert snapshot["source_status"]["sector_metrics"]["accepted"] is True
    assert snapshot["source_status"]["sector_metrics"].get("refreshed") is not True


def test_stale_required_liquidity_triggers_targeted_refresh(monkeypatch):
    from api import position_risk as pr

    _seed_required_snapshots(as_of="2099-01-01")
    _write_liquidity({"regime": "tight", "latest_date": "2000-01-01"}, as_of="2000-01-01")
    monkeypatch.setattr(
        pr.SectorMapper, "resolve_sector", lambda self, ticker, asset: _Sector("Information Technology")
    )
    monkeypatch.setattr(
        "ontology.sources.liquidity.LiquidityAdapter.fetch",
        lambda self: {"regime": "normal", "latest_date": "2099-01-01", "components": [], "regional_scores": {}},
    )

    snapshot = pr.refresh_position_risk("MU")

    state = snapshot["source_status"]["liquidity"]
    assert state["status"] == "ok"
    assert state["accepted"] is True
    assert state["refreshed"] is True
    assert state["freshness"]["observed_as_of_date"] == "2099-01-01"


def test_failed_liquidity_heal_persists_degraded_result(monkeypatch):
    from api import position_risk as pr
    from api.position_risk_store import read_latest_position_risk

    _seed_required_snapshots(as_of="2099-01-01")
    _write_liquidity({"regime": "tight", "latest_date": "2000-01-01"}, as_of="2000-01-01")
    monkeypatch.setattr(
        pr.SectorMapper, "resolve_sector", lambda self, ticker, asset: _Sector("Information Technology")
    )

    def fail_liquidity(self):
        raise RuntimeError("liquidity unavailable")

    monkeypatch.setattr("ontology.sources.liquidity.LiquidityAdapter.fetch", fail_liquidity)

    snapshot = pr.refresh_position_risk("MU")

    state = snapshot["source_status"]["liquidity"]
    assert snapshot["quality"] == "degraded"
    assert snapshot["confidence"] < 1
    assert state["status"] == "error"
    assert state["fallback_used"] is True
    assert any(item["module"] == "liquidity" for item in snapshot["degraded_modules"])
    assert read_latest_position_risk("MU")["quality"] == "degraded"


def test_invalid_liquidity_cache_is_not_used_as_fallback(monkeypatch):
    from api import position_risk as pr

    _seed_required_snapshots(as_of="2099-01-01")
    _write_liquidity({"latest_date": "2000-01-01"}, as_of="2000-01-01")
    monkeypatch.setattr(
        pr.SectorMapper, "resolve_sector", lambda self, ticker, asset: _Sector("Information Technology")
    )
    monkeypatch.setattr(
        "ontology.sources.liquidity.LiquidityAdapter.fetch",
        lambda self: (_ for _ in ()).throw(RuntimeError("liquidity unavailable")),
    )

    snapshot = pr.refresh_position_risk("MU")

    state = snapshot["source_status"]["liquidity"]
    assert state["status"] == "error"
    assert state["fallback_used"] is False
    assert state["used"] is False
    assert "liquidity" in snapshot["missing_modules"]


def test_optional_module_failed_refresh_reduces_confidence_without_blocking(monkeypatch):
    from api import position_risk as pr

    _seed_required_snapshots(as_of="2099-01-01")
    monkeypatch.setattr(
        pr.SectorMapper, "resolve_sector", lambda self, ticker, asset: _Sector("Information Technology")
    )
    monkeypatch.setattr(
        "ontology.sources.macro.SentimentAdapter.fetch",
        lambda self: (_ for _ in ()).throw(RuntimeError("sentiment unavailable")),
    )

    snapshot = pr.refresh_position_risk("MU")

    assert snapshot["risk_score"] is not None
    assert snapshot["source_status"]["sentiment"]["status"] == "error"
    assert snapshot["confidence"] < 1
    assert any(item["module"] == "sentiment" and not item["required"] for item in snapshot["degraded_modules"])


def test_missing_required_economic_growth_snapshot_is_refreshed(monkeypatch):
    from api import position_risk as pr

    calls = {"count": 0}

    def fetch_growth(self):
        calls["count"] += 1
        return _economic_growth_payload("2099-01-01")

    _seed_required_snapshots(as_of="2099-01-01")
    monkeypatch.setattr(
        pr.SectorMapper, "resolve_sector", lambda self, ticker, asset: _Sector("Information Technology")
    )
    monkeypatch.setattr("ontology.sources.macro.EconomicGrowthAdapter.fetch", fetch_growth)

    snapshot = pr.refresh_position_risk("MU")

    state = snapshot["source_status"]["economic_growth"]
    assert calls["count"] == 1
    assert state["status"] == "ok"
    assert state["required"] is True
    assert state["accepted"] is True
    assert state["used"] is True
    assert state["refreshed"] is True
    assert not any(item["module"] == "economic_growth" for item in snapshot["degraded_modules"])


def test_missing_labor_market_optional_snapshot_is_refreshed(monkeypatch):
    from api import position_risk as pr

    calls = {"count": 0}

    def fetch_labor(self):
        calls["count"] += 1
        return _labor_market_payload("2099-01-01")

    _seed_required_snapshots(as_of="2099-01-01")
    monkeypatch.setattr(
        pr.SectorMapper, "resolve_sector", lambda self, ticker, asset: _Sector("Information Technology")
    )
    monkeypatch.setattr("ontology.sources.macro.LaborMarketAdapter.fetch", fetch_labor)

    snapshot = pr.refresh_position_risk("MU")

    state = snapshot["source_status"]["labor_market"]
    assert calls["count"] == 1
    assert state["status"] == "ok"
    assert state["required"] is False
    assert state["accepted"] is True
    assert state["used"] is True
    assert state["refreshed"] is True
    assert not any(item["module"] == "labor_market" for item in snapshot["degraded_modules"])


def test_missing_sentiment_optional_snapshot_is_refreshed(monkeypatch):
    from api import position_risk as pr

    calls = {"count": 0}

    def fetch_sentiment(self):
        calls["count"] += 1
        return _sentiment_payload("2099-01-01")

    _seed_required_snapshots(as_of="2099-01-01")
    monkeypatch.setattr(
        pr.SectorMapper, "resolve_sector", lambda self, ticker, asset: _Sector("Information Technology")
    )
    monkeypatch.setattr("ontology.sources.macro.SentimentAdapter.fetch", fetch_sentiment)

    snapshot = pr.refresh_position_risk("MU")

    state = snapshot["source_status"]["sentiment"]
    assert calls["count"] == 1
    assert state["status"] == "ok"
    assert state["required"] is False
    assert state["accepted"] is True
    assert state["used"] is True
    assert state["refreshed"] is True
    assert not any(item["module"] == "sentiment" for item in snapshot["degraded_modules"])


def test_missing_positioning_optional_snapshot_is_refreshed(monkeypatch):
    from api import position_risk as pr

    calls = {"count": 0}

    def fetch_positioning(self):
        calls["count"] += 1
        return _positioning_payload("2099-01-01")

    _seed_required_snapshots(as_of="2099-01-01")
    monkeypatch.setattr(
        pr.SectorMapper, "resolve_sector", lambda self, ticker, asset: _Sector("Information Technology")
    )
    monkeypatch.setattr("ontology.sources.macro.PositioningAdapter.fetch", fetch_positioning)

    snapshot = pr.refresh_position_risk("MU")

    state = snapshot["source_status"]["positioning_summary"]
    assert calls["count"] == 1
    assert state["status"] == "ok"
    assert state["required"] is False
    assert state["accepted"] is True
    assert state["used"] is True
    assert state["refreshed"] is True
    assert not any(item["module"] == "positioning_summary" for item in snapshot["degraded_modules"])


def test_positioning_weekly_report_cache_is_fresh_within_window():
    from datetime import UTC, datetime

    from api import position_risk as pr
    from api.snapshot_keys import SNAPSHOT_POSITIONING_SUMMARY
    from api.snapshot_store import write_snapshot_success

    record = write_snapshot_success(
        SNAPSHOT_POSITIONING_SUMMARY,
        {"rows": _positioning_payload("2026-04-28")},
        as_of_date="2026-04-28",
    )

    state, _ = pr._evaluate_record(
        pr._MODULES["positioning_summary"],
        record,
        now=datetime(2026, 5, 6, 14, 58, tzinfo=UTC),
    )

    assert state["status"] == "ok"
    assert state["accepted"] is True
    assert state["freshness"]["policy"] == "weekly_report"
    assert state["freshness"]["fresh"] is True

    stale_record = write_snapshot_success(
        SNAPSHOT_POSITIONING_SUMMARY,
        {"rows": _positioning_payload("2026-04-21")},
        as_of_date="2026-04-21",
    )
    stale_state, _ = pr._evaluate_record(
        pr._MODULES["positioning_summary"],
        stale_record,
        now=datetime(2026, 5, 6, 14, 58, tzinfo=UTC),
    )

    assert stale_state["status"] == "stale"
    assert stale_state["accepted"] is False
    assert stale_state["freshness"]["fresh"] is False


def test_position_risk_refresh_endpoint_does_not_enqueue_ontology_job(auth_client, monkeypatch):
    from api import position_risk as pr

    _seed_required_snapshots(as_of="2099-01-01")
    monkeypatch.setattr(
        pr.SectorMapper, "resolve_sector", lambda self, ticker, asset: _Sector("Information Technology")
    )

    def fail_enqueue(*args, **kwargs):
        raise AssertionError("position risk refresh must not enqueue ontology jobs")

    monkeypatch.setattr("api.routers.ontology.enqueue_registered_job", fail_enqueue)

    resp = auth_client.post("/api/v1/risk/positions/MU/refresh")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "MU"
    assert data["_meta"]["intent"] == "position_risk_refresh"

    latest = auth_client.get("/api/v1/risk/positions/MU/latest")
    assert latest.status_code == 200
    assert latest.json()["result_id"] == data["result_id"]


def test_portfolio_risk_refresh_reuses_one_global_input_bundle(monkeypatch):
    import portfolio.portfolio_db as portfolio_db
    from api import position_risk as pr
    from api.position_risk_store import read_latest_portfolio_risk

    portfolio_db.save_positions(
        [
            {"ticker": "MU", "asset": "equity", "direction": "long", "shares": 42, "conviction": 4},
            {"ticker": "CRWD", "asset": "equity", "direction": "long", "shares": 5, "conviction": 3},
        ]
    )
    _seed_required_snapshots(as_of="2099-01-01")
    monkeypatch.setattr(
        pr.SectorMapper, "resolve_sector", lambda self, ticker, asset: _Sector("Information Technology")
    )
    original_loader = pr.load_global_risk_input_bundle
    calls = {"count": 0}

    def counted_loader(*args, **kwargs):
        calls["count"] += 1
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(pr, "load_global_risk_input_bundle", counted_loader)

    snapshot = pr.refresh_portfolio_risk()

    assert calls["count"] == 1
    assert snapshot["position_count"] == 2
    assert snapshot["aggregate"]["position_count"] == 2
    assert set(snapshot["position_snapshot_ids"]) == {"MU", "CRWD"}
    assert all(row["portfolio_risk_snapshot_id"] == snapshot["result_id"] for row in snapshot["position_snapshots"])
    assert read_latest_portfolio_risk()["result_id"] == snapshot["result_id"]


def test_portfolio_risk_api_returns_latest_without_ontology_refresh(auth_client, monkeypatch):
    import portfolio.portfolio_db as portfolio_db
    from api import position_risk as pr

    portfolio_db.save_positions(
        [
            {"ticker": "MU", "asset": "equity", "direction": "long", "shares": 42, "conviction": 4},
            {"ticker": "CRWD", "asset": "equity", "direction": "long", "shares": 5, "conviction": 3},
        ]
    )
    _seed_required_snapshots(as_of="2099-01-01")
    monkeypatch.setattr(
        pr.SectorMapper, "resolve_sector", lambda self, ticker, asset: _Sector("Information Technology")
    )

    def fail_enqueue(*args, **kwargs):
        raise AssertionError("portfolio risk refresh must not enqueue ontology jobs")

    monkeypatch.setattr("api.routers.ontology.enqueue_registered_job", fail_enqueue)

    refreshed = auth_client.post("/api/v1/risk/portfolio/refresh")
    assert refreshed.status_code == 200
    data = refreshed.json()
    assert data["_meta"]["intent"] == "portfolio_risk_refresh"
    assert data["position_count"] == 2

    latest = auth_client.get("/api/v1/risk/portfolio/latest")
    assert latest.status_code == 200
    assert latest.json()["result_id"] == data["result_id"]


def test_postgres_position_risk_store_does_not_run_runtime_ddl(monkeypatch):
    from api import position_risk_store as store

    conn = _FakePostgresConnection(
        [
            {"payload_json": {"result_id": "position-risk:MU:cached", "ticker": "MU"}},
            {"payload_json": {"result_id": "portfolio-risk:cached"}},
        ]
    )

    @contextmanager
    def fake_connect():
        yield conn

    monkeypatch.setattr(store, "use_postgres_state", lambda: True)
    monkeypatch.setattr(store, "connect", fake_connect)

    assert store.read_latest_position_risk("MU") == {"result_id": "position-risk:MU:cached", "ticker": "MU"}
    store.write_position_risk_snapshot(
        {
            "result_id": "position-risk:MU:fresh",
            "ticker": "MU",
            "computed_at": "2099-01-01T00:00:00+00:00",
            "risk_score": 0.2,
            "risk_level": "low",
            "confidence": 1.0,
            "quality": "ok",
        }
    )
    assert store.read_latest_portfolio_risk() == {"result_id": "portfolio-risk:cached"}
    store.write_portfolio_risk_snapshot(
        {
            "result_id": "portfolio-risk:fresh",
            "computed_at": "2099-01-01T00:00:00+00:00",
            "average_risk_score": 0.2,
            "max_risk_score": 0.3,
            "confidence": 1.0,
            "quality": "ok",
            "position_count": 1,
        }
    )

    ddl_verbs = ("CREATE ", "ALTER ", "DROP ", "GRANT ", "REVOKE ")
    ddl_statements = [sql for sql in conn.sql if sql.lstrip().upper().startswith(ddl_verbs)]
    assert ddl_statements == []


class _Sector:
    def __init__(self, sector: str):
        self.sector = sector
        self.source = "test"


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakePostgresConnection:
    def __init__(self, rows):
        self._rows = list(rows)
        self.sql = []
        self.commits = 0

    def execute(self, sql, params=None):
        self.sql.append(str(sql))
        statement = str(sql).lstrip().upper()
        if statement.startswith("SELECT "):
            return _FakeResult(self._rows.pop(0) if self._rows else None)
        return _FakeResult(None)

    def commit(self):
        self.commits += 1


def _seed_required_snapshots(*, as_of: str) -> None:
    from api.snapshot_keys import (
        SNAPSHOT_MARKET_BREADTH,
        SNAPSHOT_SECTOR_METRICS,
        SNAPSHOT_TOP50_BREADTH,
        SNAPSHOT_VIX_TERM_STRUCTURE,
    )
    from api.snapshot_store import write_snapshot_success

    write_snapshot_success(
        SNAPSHOT_MARKET_BREADTH,
        {
            "pct_above_200dma": 62.0,
            "pct_above_20dma": 58.0,
            "pct_at_20day_low": 8.0,
            "pct_at_52wk_low": 2.0,
            "total_analyzed": 503,
            "as_of_date": as_of,
        },
        as_of_date=as_of,
    )
    write_snapshot_success(
        SNAPSHOT_TOP50_BREADTH,
        {
            "pct_below_50dma": 20.0,
            "pct_3plus_dist": 12.0,
            "pct_broke_20low": 5.0,
            "universe_size": 50,
        },
        as_of_date=as_of,
    )
    write_snapshot_success(
        SNAPSHOT_VIX_TERM_STRUCTURE,
        {"latest_df": [{"Date": as_of, "VIX": 18.0, "VIX3M": 21.0, "Ratio": 1.16, "Signal": "Neutral"}]},
        as_of_date=as_of,
    )
    write_snapshot_success(
        SNAPSHOT_SECTOR_METRICS,
        {
            "timestamp": f"{as_of}T21:00:00",
            "weights_df": [
                {
                    "Sector": "Information Technology",
                    "RelPerf_3M_pp": 2.0,
                    "Chg_3M_pp": 0.2,
                    "Pct_Above_200DMA": 4.0,
                }
            ],
        },
        as_of_date=as_of,
    )
    _write_liquidity({"regime": "normal", "latest_date": as_of, "components": [], "regional_scores": {}}, as_of=as_of)


def _write_liquidity(payload: dict, *, as_of: str) -> None:
    from api.snapshot_keys import SNAPSHOT_LIQUIDITY
    from api.snapshot_store import write_snapshot_success

    write_snapshot_success(SNAPSHOT_LIQUIDITY, payload, as_of_date=as_of)


def _economic_growth_payload(as_of: str) -> dict:
    periods = {"1-mo": 1.2, "3-mo": 2.4, "6-mo": 4.8, "1-yr": 8.0}
    return {
        "timestamp": f"{as_of}T21:00:00+00:00",
        "commodities": {
            "Copper": dict(periods),
            "GS Commodity Index": dict(periods),
        },
        "equities": {
            "S&P 500": dict(periods),
            "Russell 2000": dict(periods),
        },
        "equity_relative_returns": {},
        "currencies": {
            "AUD/JPY": dict(periods),
            "CAD/JPY": dict(periods),
        },
    }


def _labor_market_payload(as_of: str) -> dict:
    return {
        "timestamp": f"{as_of}T21:00:00+00:00",
        "series": {
            "initial_claims": {"label": "Initial Claims", "unit": "thousands"},
            "continuing_claims": {"label": "Continuing Claims", "unit": "thousands"},
        },
        "latest": {
            "initial_claims": {"value": 230, "date": as_of, "change": 2.5},
            "continuing_claims": {"value": 1800, "date": as_of, "change": -4.0},
        },
    }


def _sentiment_payload(as_of: str) -> dict:
    return {
        "put_call": {},
        "surveys": {"aaii": [], "naaim": [], "errors": {}},
        "volatility": [{"date": as_of, "vvix": 95.0}],
    }


def _positioning_payload(as_of: str) -> list[dict]:
    return [
        {
            "instrument": "SP500",
            "report_date": as_of,
            "lf_net": 1200,
            "lf_net_pct_oi": 4.2,
            "lf_z": 0.4,
            "lf_deleveraging_z": 0.1,
            "lf_forced": "",
        }
    ]
