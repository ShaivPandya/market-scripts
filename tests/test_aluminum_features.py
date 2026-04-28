from __future__ import annotations

import pandas as pd


def _prices(values: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=len(values), freq="ME")
    return pd.DataFrame(
        {
            "date": dates,
            "aluminum_price_usd_tonne": values,
            "source": "world_bank_pink_sheet",
        }
    )


def test_feature_lagging_uses_prior_month_data():
    from commodities.aluminum.features import build_monthly_features

    prices = _prices([100.0, 110.0, 121.0, 100.0, 150.0, 120.0])
    features = build_monthly_features(world_bank_prices=prices, drop_missing_target=False)
    row = features[features["date"] == pd.Timestamp("2020-04-30")].iloc[0]

    # Row 2020-04 feature is shifted: it uses the 2020-03 return, not the
    # contemporaneous 2020-04 return and not the 2020-05 target return.
    expected_prior_return = 121.0 / 110.0 - 1.0
    contemporaneous_return = 100.0 / 121.0 - 1.0
    forward_target = 150.0 / 100.0 - 1.0

    assert row["aluminum_return_1m"] == expected_prior_return
    assert row["aluminum_return_1m"] != contemporaneous_return
    assert row["aluminum_return_1m"] != forward_target


def test_target_is_next_month_return_and_last_row_dropped():
    from commodities.aluminum.features import build_monthly_features

    prices = _prices([100.0, 105.0, 120.0, 90.0])
    features = build_monthly_features(world_bank_prices=prices)

    first = features.iloc[0]
    assert first["date"] == pd.Timestamp("2020-01-31")
    assert first["target_return_1m_forward"] == 105.0 / 100.0 - 1.0
    assert features["date"].max() == pd.Timestamp("2020-03-31")


def test_feature_metadata_marks_unavailable_optional_sources():
    from commodities.aluminum.features import build_monthly_features, feature_metadata

    features = build_monthly_features(world_bank_prices=_prices([100.0 + i for i in range(18)]))
    meta = feature_metadata(features).set_index("feature")

    assert meta.loc["aluminum_return_1m", "category"] == "price_technical"
    assert meta.loc["inventory_change_1m", "category"] == "unavailable"
    assert meta.loc["power_proxy_change_1m", "category"] == "unavailable"
    assert bool(meta.loc["aluminum_return_1m", "is_lagged"])


def test_inventory_and_power_proxy_features_when_data_exists():
    from commodities.aluminum.features import build_monthly_features

    dates = pd.date_range("2020-01-31", periods=8, freq="ME")
    eia = pd.DataFrame(
        {
            "date": dates,
            "eia_series_id_or_route": "proxy",
            "value": [10, 11, 12, 13, 14, 15, 16, 17],
            "unit": "cents/kwh",
            "source": "eia_api_v2",
        }
    )
    shfe = pd.DataFrame(
        {
            "date": dates,
            "contract_or_product": "Aluminum",
            "inventory_tonnes": [100, 105, 110, 100, 95, 90, 85, 80],
            "source": "shfe_public_html",
        }
    )

    features = build_monthly_features(
        world_bank_prices=_prices([100.0 + i for i in range(8)]), eia_power_proxy=eia, shfe_inventory=shfe
    )

    assert features["inventory_change_1m"].notna().any()
    assert features["power_proxy_change_1m"].notna().any()
    assert features["has_eia_power_proxy"].any()
    assert features["has_shfe_inventory"].any()
