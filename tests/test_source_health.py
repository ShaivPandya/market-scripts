from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from api.snapshot_store import SnapshotRecord
from api.source_health import build_approval_source_health_review, build_workspace_source_health


def _snapshot(
    key: str,
    *,
    fetched_at: datetime,
    as_of_date: str | None = None,
    status: str = "ok",
    quality: str = "ok",
    error: str | None = None,
) -> SnapshotRecord:
    return SnapshotRecord(
        snapshot_key=key,
        payload={"value": 1},
        as_of_date=as_of_date if as_of_date is not None else fetched_at.date().isoformat(),
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
    assert sources["market_breadth:sp500:1y"]["source_registry"]["source_id"] == "market_breadth"


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
    assert sources["market_breadth:sp500:1y"]["reliability_tier"] == "critical"
    assert sources["market_breadth:sp500:1y"]["sla_breach"] is True
    assert payload["overall_quality"] == "stale"
    assert payload["counts"]["required_stale"] == 1
    assert payload["counts"]["critical_stale"] == 1
    assert payload["tier_counts"]["critical"] >= 1


def test_source_health_market_snapshot_fresh_on_sunday_after_friday_close():
    eastern = ZoneInfo("America/New_York")
    now = datetime(2026, 5, 31, 12, 0, tzinfo=eastern)
    fetched = datetime(2026, 5, 29, 23, 30, tzinfo=ZoneInfo("UTC"))
    payload = build_workspace_source_health(
        now=now,
        portfolio_risk=_portfolio_risk(now),
        snapshot_records=[
            _snapshot("market_breadth:sp500:1y", fetched_at=fetched, as_of_date="2026-05-29"),
            _snapshot("signal_aggregator:current:v1", fetched_at=fetched, as_of_date="2026-05-29"),
        ],
    )

    sources = _sources(payload)
    assert sources["market_breadth:sp500:1y"]["status"] == "ok"
    assert sources["market_breadth:sp500:1y"]["stale"] is False
    assert sources["market_breadth:sp500:1y"]["expected_as_of_date"] == "2026-05-29"


def test_source_health_market_snapshot_fresh_monday_before_close_and_stale_after_close():
    eastern = ZoneInfo("America/New_York")
    fetched = datetime(2026, 5, 29, 23, 30, tzinfo=ZoneInfo("UTC"))

    before_close = build_workspace_source_health(
        now=datetime(2026, 6, 1, 15, 30, tzinfo=eastern),
        portfolio_risk=_portfolio_risk(datetime(2026, 6, 1, 15, 30, tzinfo=eastern)),
        snapshot_records=[
            _snapshot("market_breadth:sp500:1y", fetched_at=fetched, as_of_date="2026-05-29"),
            _snapshot("signal_aggregator:current:v1", fetched_at=fetched, as_of_date="2026-05-29"),
        ],
    )
    after_close = build_workspace_source_health(
        now=datetime(2026, 6, 1, 16, 30, tzinfo=eastern),
        portfolio_risk=_portfolio_risk(datetime(2026, 6, 1, 16, 30, tzinfo=eastern)),
        snapshot_records=[
            _snapshot("market_breadth:sp500:1y", fetched_at=fetched, as_of_date="2026-05-29"),
            _snapshot("signal_aggregator:current:v1", fetched_at=fetched, as_of_date="2026-05-29"),
        ],
    )

    assert _sources(before_close)["market_breadth:sp500:1y"]["status"] == "ok"
    assert _sources(after_close)["market_breadth:sp500:1y"]["status"] == "stale"
    assert _sources(after_close)["market_breadth:sp500:1y"]["expected_as_of_date"] == "2026-06-01"


def test_source_health_macro_cadence_windows_do_not_use_wall_clock_sla():
    now = datetime(2026, 5, 31, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    fetched = datetime(2026, 5, 29, 23, 30, tzinfo=ZoneInfo("UTC"))
    payload = build_workspace_source_health(
        now=now,
        portfolio_risk=_portfolio_risk(now),
        snapshot_records=[
            _snapshot("liquidity:current:v1", fetched_at=fetched, as_of_date="2026-05-22"),
            _snapshot("labor_market:current:v1", fetched_at=fetched, as_of_date="2026-05-22"),
            _snapshot("positioning_summary:current:v1", fetched_at=fetched, as_of_date="2026-05-22"),
            _snapshot("housing:current:v1", fetched_at=fetched, as_of_date="2026-04-20"),
            _snapshot("economic_growth:current:v1", fetched_at=fetched, as_of_date="2026-04-20"),
            _snapshot("signal_aggregator:current:v1", fetched_at=fetched, as_of_date="2026-05-29"),
        ],
    )

    sources = _sources(payload)
    assert sources["liquidity:current:v1"]["status"] == "ok"
    assert sources["labor_market:current:v1"]["status"] == "ok"
    assert sources["positioning_summary:current:v1"]["status"] == "ok"
    assert sources["housing:current:v1"]["status"] == "ok"
    assert sources["economic_growth:current:v1"]["status"] == "ok"


def test_source_health_missing_required_source_does_not_throw():
    now = datetime(2026, 5, 14, 18, 0)
    payload = build_workspace_source_health(now=now, snapshot_records=[])

    sources = _sources(payload)
    assert sources["portfolio"]["status"] == "missing"
    assert payload["overall_quality"] == "failed"
    assert payload["counts"]["required_failed"] > 0


def test_source_health_uses_workspace_runtime_sources_for_missing_freshness_records():
    now = datetime(2026, 5, 14, 18, 0)
    payload = build_workspace_source_health(
        now=now,
        portfolio_data={"positions": [{"ticker": "MU", "as_of": now.isoformat()}]},
        regime_data={
            "status": "ok",
            "as_of": now.date().isoformat(),
            "_meta": {
                "snapshot": {
                    "key": "signal_aggregator:current:v1",
                    "as_of": now.date().isoformat(),
                    "fetched_at": now.isoformat(),
                    "refresh_status": "ok",
                    "stale": False,
                }
            },
            "module_status": {
                "liquidity": {
                    "status": "ok",
                    "detail": "live fallback",
                }
            },
        },
        snapshot_records=[
            _snapshot("market_breadth:sp500:1y", fetched_at=now - timedelta(hours=1)),
            _snapshot("top50_breadth:sp500:2y", fetched_at=now - timedelta(hours=1)),
            _snapshot("vix_term_structure:current:v1", fetched_at=now - timedelta(hours=1)),
            _snapshot("sector_metrics:sp500:2y", fetched_at=now - timedelta(hours=1)),
            _snapshot("economic_growth:current:v1", fetched_at=now - timedelta(hours=1)),
        ],
    )

    sources = _sources(payload)
    assert sources["portfolio"]["status"] == "ok"
    assert sources["signal_aggregator:current:v1"]["status"] == "ok"
    assert sources["liquidity:current:v1"]["status"] == "ok"
    assert payload["overall_quality"] == "ok"
    assert payload["counts"]["required_failed"] == 0


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


def test_approval_source_health_blocks_required_missing_source():
    now = datetime(2026, 5, 14, 18, 0)
    payload = build_workspace_source_health(now=now, snapshot_records=[])

    review = build_approval_source_health_review({"id": "approval:1", "proposed_change": {}}, payload)

    assert review["status"] == "blocked"
    assert any(row["status"] == "missing" and row["required"] is True for row in review["blockers"])


def test_approval_source_health_blocks_critical_stale_source():
    now = datetime(2026, 5, 14, 18, 0)
    payload = build_workspace_source_health(
        now=now,
        portfolio_risk=_portfolio_risk(now),
        snapshot_records=[
            _snapshot("market_breadth:sp500:1y", fetched_at=now - timedelta(days=3)),
            _snapshot("signal_aggregator:current:v1", fetched_at=now - timedelta(hours=1)),
        ],
    )

    review = build_approval_source_health_review({"id": "approval:1", "proposed_change": {}}, payload)

    assert review["status"] == "blocked"
    assert review["blockers"]
    assert review["blockers"][0]["id"] == "market_breadth:sp500:1y"
    assert review["blockers"][0]["reliability_tier"] == "critical"


def test_approval_source_health_warns_on_standard_stale_source():
    now = datetime(2026, 5, 14, 18, 0)
    payload = build_workspace_source_health(
        now=now,
        portfolio_risk=_portfolio_risk(now),
        snapshot_records=[
            _snapshot("market_breadth:sp500:1y", fetched_at=now - timedelta(hours=1)),
            _snapshot("signal_aggregator:current:v1", fetched_at=now - timedelta(days=3)),
        ],
    )

    review = build_approval_source_health_review({"id": "approval:1", "proposed_change": {}}, payload)

    assert review["status"] == "warning"
    assert review["blockers"] == []
    assert any(row["id"] == "signal_aggregator:current:v1" for row in review["warnings"])
    assert review["warnings"][0]["reliability_tier"] == "standard"


def test_approval_source_health_blocks_explicit_stale_dependency():
    now = datetime(2026, 5, 14, 18, 0)
    payload = build_workspace_source_health(
        now=now,
        portfolio_risk=_portfolio_risk(now),
        snapshot_records=[
            _snapshot("housing:current:v1", fetched_at=now - timedelta(days=60)),
            _snapshot("signal_aggregator:current:v1", fetched_at=now - timedelta(hours=1)),
        ],
    )

    review = build_approval_source_health_review(
        {
            "id": "approval:1",
            "proposed_change": {"source_dependencies": ["housing:current:v1"]},
        },
        payload,
    )

    assert review["status"] == "blocked"
    assert review["blockers"][0]["id"] == "housing:current:v1"


def test_approval_source_health_warns_on_optional_degraded_source():
    now = datetime(2026, 5, 14, 18, 0)
    payload = build_workspace_source_health(
        now=now,
        portfolio_risk=_portfolio_risk(now),
        snapshot_records=[
            _snapshot("housing:current:v1", fetched_at=now - timedelta(hours=1), status="error", error="timeout"),
            _snapshot("signal_aggregator:current:v1", fetched_at=now - timedelta(hours=1)),
        ],
    )

    review = build_approval_source_health_review({"id": "approval:1", "proposed_change": {}}, payload)

    assert review["status"] == "warning"
    assert review["blockers"] == []
    assert review["warnings"][0]["id"] == "housing:current:v1"
