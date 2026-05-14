from __future__ import annotations

from datetime import datetime, timedelta

from api.snapshot_store import SnapshotRecord
from api.source_health import build_workspace_source_health


def _snapshot(
    key: str,
    *,
    fetched_at: datetime,
    status: str = "ok",
    quality: str = "ok",
    error: str | None = None,
) -> SnapshotRecord:
    return SnapshotRecord(
        snapshot_key=key,
        payload={"value": 1},
        as_of_date=fetched_at.date().isoformat(),
        fetched_at=fetched_at.isoformat(),
        status=status,
        error=error,
        version=1,
        artifact_uri=None,
        quality=quality,
    )


def _sources(payload: dict) -> dict[str, dict]:
    return {source["id"]: source for domain in payload["domains"] for source in domain["sources"]}


def _portfolio_risk(now: datetime) -> dict:
    source_status = {
        "portfolio": {
            "status": "ok",
            "quality": "ok",
            "required": True,
            "as_of": now.isoformat(),
            "fetched_at": now.isoformat(),
            "freshness": {"fresh": True, "observed_as_of_date": now.date().isoformat()},
        }
    }
    for module, snapshot_key in {
        "market_breadth": "market_breadth:sp500:1y",
        "top50_breadth": "top50_breadth:sp500:2y",
        "vix_term_structure": "vix_term_structure:current:v1",
        "sector_metrics": "sector_metrics:sp500:2y",
        "liquidity": "liquidity:current:v1",
        "economic_growth": "economic_growth:current:v1",
    }.items():
        source_status[module] = {
            "status": "ok",
            "quality": "ok",
            "required": True,
            "snapshot_key": snapshot_key,
            "as_of": now.isoformat(),
            "fetched_at": now.isoformat(),
            "freshness": {"fresh": True, "observed_as_of_date": now.date().isoformat()},
        }
    return {"source_status": source_status}


def test_source_health_marks_fresh_snapshot_ok():
    now = datetime(2026, 5, 14, 18, 0)
    payload = build_workspace_source_health(
        now=now,
        portfolio_risk=_portfolio_risk(now),
        snapshot_records=[
            _snapshot("market_breadth:sp500:1y", fetched_at=now - timedelta(hours=1)),
            _snapshot("signal_aggregator:current:v1", fetched_at=now - timedelta(hours=1)),
        ],
    )

    sources = _sources(payload)
    assert sources["market_breadth:sp500:1y"]["status"] == "ok"
    assert sources["market_breadth:sp500:1y"]["required"] is True


def test_source_health_stale_required_affects_overall_quality():
    now = datetime(2026, 5, 14, 18, 0)
    payload = build_workspace_source_health(
        now=now,
        portfolio_risk=_portfolio_risk(now),
        snapshot_records=[
            _snapshot("market_breadth:sp500:1y", fetched_at=now - timedelta(days=3)),
            _snapshot("signal_aggregator:current:v1", fetched_at=now - timedelta(hours=1)),
        ],
    )

    sources = _sources(payload)
    assert sources["market_breadth:sp500:1y"]["status"] == "stale"
    assert payload["overall_quality"] == "stale"
    assert payload["counts"]["required_stale"] == 1


def test_source_health_missing_required_source_does_not_throw():
    now = datetime(2026, 5, 14, 18, 0)
    payload = build_workspace_source_health(now=now, snapshot_records=[])

    sources = _sources(payload)
    assert sources["portfolio"]["status"] == "missing"
    assert payload["overall_quality"] == "failed"
    assert payload["counts"]["required_failed"] > 0


def test_source_health_degraded_optional_is_not_required_failure():
    now = datetime(2026, 5, 14, 18, 0)
    payload = build_workspace_source_health(
        now=now,
        portfolio_risk=_portfolio_risk(now),
        snapshot_records=[
            _snapshot("housing:current:v1", fetched_at=now - timedelta(hours=1), status="error", error="timeout"),
            _snapshot("signal_aggregator:current:v1", fetched_at=now - timedelta(hours=1)),
        ],
    )

    sources = _sources(payload)
    assert sources["housing:current:v1"]["required"] is False
    assert sources["housing:current:v1"]["status"] == "degraded"
    assert payload["counts"]["optional_degraded"] == 1
