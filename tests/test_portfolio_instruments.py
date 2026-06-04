import pytest

from portfolio.economic_exposure import resolve_economic_exposure
from portfolio.instruments import (
    build_option_contract_symbol,
    display_ticker,
    normalize_portfolio_instrument_row,
    parse_occ_symbol,
    position_row_id,
)


def test_parse_occ_symbol_meta_call():
    parsed = parse_occ_symbol("META260116C00500000")
    assert parsed is not None
    assert parsed.underlying_ticker == "META"
    assert parsed.option_expiration == "2026-01-16"
    assert parsed.option_type == "call"
    assert parsed.option_strike == 500.0
    assert parsed.option_contract_symbol == "META260116C00500000"


def test_build_option_contract_symbol_from_structured_fields():
    symbol = build_option_contract_symbol("META", "2026-01-16", "call", 500)
    assert symbol == "META260116C00500000"


def test_normalize_portfolio_option_row_sets_display_ticker_and_position_id():
    row = normalize_portfolio_instrument_row(
        {
            "ticker": "META",
            "asset": "equity",
            "direction": "long",
            "shares": 2,
            "instrument_type": "option",
            "underlying_ticker": "META",
            "option_expiration": "2026-01-16",
            "option_strike": 500,
            "option_type": "call",
        }
    )
    assert row["ticker"] == "META"
    assert row["underlying_ticker"] == "META"
    assert row["option_contract_symbol"] == "META260116C00500000"
    assert row["position_id"] == "META260116C00500000"
    assert row["contract_multiplier"] == 100.0
    assert display_ticker(row) == "META"
    assert position_row_id(row) == "META260116C00500000"


def test_normalize_portfolio_option_row_rejects_incomplete_fields():
    with pytest.raises(ValueError, match="Option positions require"):
        normalize_portfolio_instrument_row(
            {
                "ticker": "META",
                "asset": "equity",
                "direction": "long",
                "instrument_type": "option",
                "underlying_ticker": "META",
            }
        )


def test_multiple_meta_option_legs_have_distinct_position_ids():
    call = normalize_portfolio_instrument_row(
        {
            "ticker": "META",
            "asset": "equity",
            "direction": "long",
            "shares": 1,
            "instrument_type": "option",
            "underlying_ticker": "META",
            "option_expiration": "2026-01-16",
            "option_strike": 500,
            "option_type": "call",
        }
    )
    put = normalize_portfolio_instrument_row(
        {
            "ticker": "META",
            "asset": "equity",
            "direction": "long",
            "shares": 1,
            "instrument_type": "option",
            "underlying_ticker": "META",
            "option_expiration": "2026-01-16",
            "option_strike": 450,
            "option_type": "put",
        }
    )
    shares = normalize_portfolio_instrument_row(
        {
            "ticker": "META",
            "asset": "equity",
            "direction": "long",
            "shares": 10,
            "instrument_type": "security",
        }
    )
    assert position_row_id(call) != position_row_id(put)
    assert position_row_id(shares) == "META"
    assert display_ticker(call) == display_ticker(put) == display_ticker(shares) == "META"


def test_metu_economic_exposure_resolves_to_meta_2x():
    exposure = resolve_economic_exposure({"ticker": "METU", "instrument_type": "security"})
    assert exposure.underlying_ticker == "META"
    assert exposure.factor == 2.0
