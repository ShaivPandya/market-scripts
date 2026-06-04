import pandas as pd

from portfolio.economic_exposure import (
    exposure_group_key,
    resolve_economic_exposure,
    scale_gross_notional_for_exposure,
    scale_signed_notional_for_exposure,
)
from portfolio.portfolio_analytics import compute_analytics
from portfolio.portfolio_dashboard import _underlying_exposures
from portfolio.policy_gate import evaluate_policy_gate
from portfolio.portfolio_optimizer import portfolio_sizer
from portfolio.scenario_simulator import _economic_scenario_pnl, _normalize_position


def test_static_metu_maps_to_meta_2x():
    exposure = resolve_economic_exposure({"ticker": "METU", "instrument_type": "security", "direction": "long"})
    assert exposure.underlying_ticker == "META"
    assert exposure.factor == 2.0
    assert exposure.source == "static"


def test_static_inverse_metd_maps_negative_factor():
    exposure = resolve_economic_exposure({"ticker": "METD", "instrument_type": "security", "direction": "long"})
    assert exposure.underlying_ticker == "META"
    assert exposure.factor == -1.0


def test_unknown_etf_returns_identity():
    exposure = resolve_economic_exposure({"ticker": "XYZ", "instrument_type": "security"})
    assert exposure.underlying_ticker == "XYZ"
    assert exposure.factor == 1.0
    assert exposure.source == "identity"


def test_metadata_parser_inverse_from_long_name():
    exposure = resolve_economic_exposure(
        {"ticker": "FOO", "instrument_type": "security"},
        metadata={"longName": "Daily NVDA Inverse -1X Shares", "underlying_ticker": "NVDA"},
    )
    assert exposure.underlying_ticker == "NVDA"
    assert exposure.factor == -1.0
    assert exposure.source == "metadata"


def test_exposure_group_key_collapses_metu_to_meta():
    assert exposure_group_key({"ticker": "METU", "instrument_type": "security"}) == "META"
    assert exposure_group_key({"ticker": "META", "instrument_type": "security"}) == "META"


def test_scale_signed_notional_applies_leverage_factor():
    row = {"ticker": "METU", "instrument_type": "security", "direction": "long"}
    assert scale_signed_notional_for_exposure(1000.0, row) == 2000.0
    assert scale_gross_notional_for_exposure(1000.0, row) == 2000.0


def test_analytics_groups_meta_and_metu_economic_exposure():
    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    prices = {
        "META": pd.Series([500.0, 520.0], index=dates),
        "METU": pd.Series([40.0, 42.0], index=dates),
    }
    holdings = [
        {
            "ticker": "META",
            "asset": "equity",
            "direction": "long",
            "cost_basis": 500.0,
            "quantity": 1.0,
            "shares": 1.0,
            "instrument_type": "security",
            "position_id": "META",
        },
        {
            "ticker": "METU",
            "asset": "equity",
            "direction": "long",
            "cost_basis": 40.0,
            "quantity": 10.0,
            "shares": 10.0,
            "instrument_type": "security",
            "position_id": "METU",
        },
    ]
    analytics = compute_analytics(prices, holdings)
    meta_leg = analytics["per_position"]["META"]
    metu_leg = analytics["per_position"]["METU"]
    assert meta_leg["exposure_group_key"] == "META"
    assert metu_leg["exposure_group_key"] == "META"
    assert metu_leg["exposure_multiplier"] == 2.0
    assert meta_leg["weight"] + metu_leg["weight"] == 1.0


def test_underlying_exposures_scales_metu_under_meta():
    leg_metadata = {
        "META": {
            "display_ticker": "META",
            "ticker": "META",
            "instrument_type": "security",
            "direction": "long",
            "exposure_group_key": "META",
            "current_notional": 520.0,
            "cost_notional": 500.0,
        },
        "METU": {
            "display_ticker": "METU",
            "ticker": "METU",
            "instrument_type": "security",
            "direction": "long",
            "exposure_group_key": "META",
            "economic_underlying_ticker": "META",
            "exposure_multiplier": 2.0,
            "current_notional": 420.0,
            "cost_notional": 400.0,
        },
    }
    row = _underlying_exposures(leg_metadata)[0]
    assert row["underlying_ticker"] == "META"
    assert row["equity_current_notional"] == 520.0 + 840.0


def test_policy_concentration_aggregates_meta_and_metu():
    from api.financial_policy_settings import set_financial_policy_matrix_setting
    from portfolio.policy_matrix import default_financial_policy_matrix

    set_financial_policy_matrix_setting(default_financial_policy_matrix())
    gate = evaluate_policy_gate(
        "update_portfolio_positions",
        {
            "book_size": 1000,
            "positions": [
                {
                    "ticker": "META",
                    "asset": "equity",
                    "direction": "long",
                    "notional_base": 100,
                },
                {
                    "ticker": "METU",
                    "asset": "equity",
                    "direction": "long",
                    "notional_base": 100,
                },
            ],
        },
    )
    concentration = next(
        check for check in gate["check_results"] if check.get("check") == "concentration.position"
    )
    assert concentration is not None
    assert concentration["observed"] == 0.3
    assert gate["decision"] in {"review_required", "blocked"}


def test_sizer_current_exposure_uses_economic_group_and_factor():
    meta = pd.DataFrame(
        [
            {
                "ticker": "METU",
                "instrument_type": "security",
                "direction": "long",
                "quantity": 10.0,
                "shares": 10.0,
                "price_symbol": "METU",
                "contract_multiplier": 1.0,
                "asset": "equity",
            }
        ]
    )
    exposure = portfolio_sizer._compute_current_underlying_dollar_exposure(meta, {"METU": 42.0})
    assert exposure["META"] == 840.0


def test_scenario_pnl_scales_leveraged_etf_move():
    position = _normalize_position(
        {
            "ticker": "METU",
            "direction": "long",
            "notional_base": 1000,
            "instrument_type": "security",
        }
    )
    pnl = _economic_scenario_pnl(position, 1000.0, 0.10)
    assert pnl == 200.0


def test_scenario_pnl_inverse_etf_negative_on_underlying_rally():
    position = _normalize_position(
        {
            "ticker": "METD",
            "direction": "long",
            "notional_base": 1000,
            "instrument_type": "security",
        }
    )
    pnl = _economic_scenario_pnl(position, 1000.0, 0.10)
    assert pnl == -100.0
