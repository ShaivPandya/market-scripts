from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from macro.liquidity.liquidity import (
    add_japan_derived_series,
    build_component_as_of,
    build_data_quality,
    build_weekly_panel,
    cap_weekly_panel_to_completed_week,
)


def test_completed_week_cap_drops_partial_future_bucket():
    raw = pd.DataFrame(
        {"on_rrp": [3719.0, 2034.0]},
        index=pd.to_datetime(["2026-05-13", "2026-05-14"]),
    )

    weekly = build_weekly_panel(raw, week_ending="W-WED")
    assert weekly.index[-1] == pd.Timestamp("2026-05-20")

    capped, suppressed = cap_weekly_panel_to_completed_week(
        weekly,
        now=datetime(2026, 5, 15, 12, 0, tzinfo=UTC),
        week_ending="W-WED",
    )

    assert capped.index[-1] == pd.Timestamp("2026-05-13")
    assert suppressed == pd.Timestamp("2026-05-20")


def test_japan_yoy_uses_native_frequency_before_weekly_alignment():
    raw_index = pd.DatetimeIndex(
        sorted(
            {
                *pd.date_range("2025-04-01", "2026-04-01", freq="MS").to_pydatetime().tolist(),
                *pd.to_datetime(["2024-07-01", "2024-10-01", "2025-01-01", "2025-04-01", "2025-07-01"]).to_pydatetime(),
                *pd.date_range("2026-05-08", "2026-05-14", freq="D").to_pydatetime().tolist(),
            }
        )
    )
    raw = pd.DataFrame(index=raw_index, columns=["boj_assets", "jpn_m3_yoy", "jpn_credit_private"], dtype=float)

    for month in pd.date_range("2025-04-01", "2026-03-01", freq="MS"):
        raw.loc[month, "boj_assets"] = 100.0
    raw.loc[pd.Timestamp("2026-04-01"), "boj_assets"] = 90.0

    raw.loc[pd.Timestamp("2025-11-01"), "jpn_m3_yoy"] = 1.2

    for date, value in {
        "2024-07-01": 100.0,
        "2024-10-01": 101.0,
        "2025-01-01": 102.0,
        "2025-04-01": 103.0,
        "2025-07-01": 80.0,
    }.items():
        raw.loc[pd.Timestamp(date), "jpn_credit_private"] = value

    weekly = pd.DataFrame(index=pd.date_range("2025-07-02", "2026-05-13", freq="W-WED"))
    result = add_japan_derived_series(weekly, raw, week_ending="W-WED")

    assert result.loc[pd.Timestamp("2026-05-13"), "boj_assets_yoy"] == pytest.approx(-10.0)
    assert result.loc[pd.Timestamp("2026-05-13"), "jpn_credit_yoy"] == pytest.approx(-20.0)
    assert result.loc[pd.Timestamp("2026-05-13"), "jpn_m3_yoy"] == pytest.approx(1.2)


def test_data_quality_warns_for_suppressed_future_bucket_and_lagged_component():
    quality = build_data_quality(
        pd.Timestamp("2026-05-13"),
        {
            "net_liquidity": "2026-05-13",
            "jpn_m3_yoy": "2025-11-01",
        },
        suppressed_future_date=pd.Timestamp("2026-05-20"),
    )

    assert quality["status"] == "degraded"
    assert any("Suppressed partial weekly bucket ending 2026-05-20" in warning for warning in quality["warnings"])
    assert any("M3 YoY is lagged" in warning for warning in quality["warnings"])


def test_quarterly_source_dates_use_period_end_for_freshness():
    raw = pd.DataFrame(
        {"jpn_credit_private": [1_135_794.3]},
        index=pd.to_datetime(["2025-07-01"]),
    )

    component_as_of = build_component_as_of(raw, None, pd.Timestamp("2026-05-27"))

    assert component_as_of["jpn_credit_yoy"] == "2025-09-30"

    quality = build_data_quality(pd.Timestamp("2026-05-27"), component_as_of)

    assert quality == {"status": "ok", "warnings": []}
