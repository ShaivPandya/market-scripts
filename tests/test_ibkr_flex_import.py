from __future__ import annotations

import pytest

from portfolio.ibkr_flex_import import (
    is_ibkr_flex_index_hedge_row,
    merge_ibkr_flex_hedge_replacement,
    merge_preserved_portfolio_metadata,
    parse_ibkr_flex_open_positions_xml,
    split_ibkr_flex_import_rows,
)

SAMPLE_FLEX_XML = """<?xml version="1.0" encoding="UTF-8"?>
<FlexQueryResponse queryName="open_positions" type="AF">
<FlexStatements count="1">
<FlexStatement accountId="U18542639" fromDate="20260602" toDate="20260602" period="LastBusinessDay" whenGenerated="20260603;091519">
<OpenPositions>
<OpenPosition assetCategory="STK" symbol="MU" position="10" side="Long" currency="USD" fxRateToBase="1" listingExchange="NASDAQ" costBasisPrice="938.5139801" costBasisMoney="9385.139801" positionValue="10641" reportDate="20260602" />
<OpenPosition assetCategory="STK" symbol="PII" position="-20" side="Short" currency="USD" fxRateToBase="1" listingExchange="NYSE" costBasisPrice="63.86696425" costBasisMoney="-1277.339285" positionValue="-1365" reportDate="20260602" />
<OpenPosition assetCategory="OPT" symbol="META  270319C00620000" underlyingSymbol="META" expiry="20270319" strike="620" putCall="C" multiplier="100" position="2" side="Long" currency="USD" fxRateToBase="1" listingExchange="CBOE" costBasisPrice="85.7505075" costBasisMoney="17150.1015" positionValue="16280" reportDate="20260602" />
<OpenPosition assetCategory="OPT" symbol="META  270319C00660000" underlyingSymbol="META" expiry="20270319" strike="660" putCall="C" multiplier="100" position="-7" side="Short" currency="USD" fxRateToBase="1" listingExchange="CBOE" costBasisPrice="68.836698491" costBasisMoney="-48185.688944" positionValue="-46532.5" reportDate="20260602" />
<OpenPosition assetCategory="OPT" symbol="NVDA  270319P00195000" underlyingSymbol="NVDA" expiry="20270319" strike="195" putCall="P" multiplier="100" position="-2" side="Short" currency="USD" fxRateToBase="1" listingExchange="CBOE" costBasisPrice="21.13903641" costBasisMoney="-4227.807282" positionValue="-3870" reportDate="20260602" />
<OpenPosition assetCategory="FUT" symbol="ESZ5" position="1" side="Long" currency="USD" />
</OpenPositions>
</FlexStatement>
</FlexStatements>
</FlexQueryResponse>
"""


def test_parse_long_stock_row():
    rows = parse_ibkr_flex_open_positions_xml(SAMPLE_FLEX_XML)
    mu = next(row for row in rows if row["ticker"] == "MU")
    assert mu["instrument_type"] == "security"
    assert mu["direction"] == "long"
    assert mu["shares"] == 10
    assert mu["cost_basis"] == pytest.approx(938.5139801)
    assert mu["position_id"] == "MU"


def test_parse_short_stock_row():
    rows = parse_ibkr_flex_open_positions_xml(SAMPLE_FLEX_XML)
    pii = next(row for row in rows if row["ticker"] == "PII")
    assert pii["direction"] == "short"
    assert pii["shares"] == 20
    assert pii["cost_basis_base"] == pytest.approx(-1277.339285)


def test_parse_long_call_option_row():
    rows = parse_ibkr_flex_open_positions_xml(SAMPLE_FLEX_XML)
    call = next(row for row in rows if row.get("option_strike") == 620.0)
    assert call["instrument_type"] == "option"
    assert call["ticker"] == "META"
    assert call["underlying_ticker"] == "META"
    assert call["option_type"] == "call"
    assert call["option_contract_symbol"] == "META270319C00620000"
    assert call["contract_multiplier"] == 100
    assert call["direction"] == "long"
    assert call["shares"] == 2


def test_parse_short_call_and_put_option_rows():
    rows = parse_ibkr_flex_open_positions_xml(SAMPLE_FLEX_XML)
    short_call = next(row for row in rows if row.get("option_strike") == 660.0)
    short_put = next(row for row in rows if row.get("option_strike") == 195.0)
    assert short_call["direction"] == "short"
    assert short_call["option_type"] == "call"
    assert short_put["direction"] == "short"
    assert short_put["option_type"] == "put"
    assert short_put["ticker"] == "NVDA"


def test_parser_ignores_unsupported_asset_categories():
    rows = parse_ibkr_flex_open_positions_xml(SAMPLE_FLEX_XML)
    assert all(row.get("ticker") != "ESZ5" for row in rows)


def test_merge_preserved_portfolio_metadata():
    imported = parse_ibkr_flex_open_positions_xml(SAMPLE_FLEX_XML)
    existing = [
        {
            "ticker": "MU",
            "instrument_type": "security",
            "direction": "long",
            "shares": 5,
            "conviction": 5,
            "contrarian": True,
            "group_name": "Semis",
            "group_conviction": 4,
            "role": "position",
        }
    ]
    merged = merge_preserved_portfolio_metadata(imported, existing)
    mu = next(row for row in merged if row["ticker"] == "MU")
    assert mu["shares"] == 10
    assert mu["conviction"] == 5
    assert mu["contrarian"] is True
    assert mu["group_name"] == "Semis"
    assert mu["group_conviction"] == 4


def test_parse_rejects_empty_xml():
    with pytest.raises(ValueError, match="empty"):
        parse_ibkr_flex_open_positions_xml("")


FLEX_WITH_INDEX_ETFS = """<?xml version="1.0" encoding="UTF-8"?>
<FlexQueryResponse queryName="open_positions" type="AF">
<FlexStatements count="1">
<FlexStatement accountId="U18542639" fromDate="20260603" toDate="20260603" period="LastBusinessDay" whenGenerated="20260604;051310">
<OpenPositions>
<OpenPosition assetCategory="STK" symbol="SPY" position="-30" side="Short" currency="USD" fxRateToBase="1" listingExchange="ARCA" costBasisPrice="714.25" costBasisMoney="-21427.7" positionValue="-22627.2" reportDate="20260603" />
<OpenPosition assetCategory="STK" symbol="IWM" position="-10" side="Short" currency="USD" fxRateToBase="1" listingExchange="ARCA" costBasisPrice="200" costBasisMoney="-2000" positionValue="-2100" reportDate="20260603" />
<OpenPosition assetCategory="STK" symbol="QQQ" position="5" side="Long" currency="USD" fxRateToBase="1" listingExchange="NASDAQ" costBasisPrice="400" costBasisMoney="2000" positionValue="2100" reportDate="20260603" />
<OpenPosition assetCategory="STK" symbol="PII" position="-20" side="Short" currency="USD" fxRateToBase="1" listingExchange="NYSE" costBasisPrice="63.86" costBasisMoney="-1277.33" positionValue="-1366.4" reportDate="20260603" />
<OpenPosition assetCategory="OPT" symbol="NVDA  270319C00215000" underlyingSymbol="NVDA" expiry="20270319" strike="215" putCall="C" multiplier="100" position="23" side="Long" currency="USD" fxRateToBase="1" listingExchange="CBOE" costBasisPrice="38.45" costBasisMoney="88455.38" positionValue="84795.25" reportDate="20260603" />
</OpenPositions>
</FlexStatement>
</FlexStatements>
</FlexQueryResponse>
"""


def test_split_short_index_etfs_to_hedges():
    rows = parse_ibkr_flex_open_positions_xml(FLEX_WITH_INDEX_ETFS)
    portfolio_rows, hedge_rows = split_ibkr_flex_import_rows(rows)
    hedge_tickers = {row["ticker"] for row in hedge_rows}
    portfolio_tickers = {row["ticker"] for row in portfolio_rows}

    assert hedge_tickers == {"SPY", "IWM"}
    assert "QQQ" in portfolio_tickers
    assert "PII" in portfolio_tickers
    assert "NVDA" in portfolio_tickers
    assert all(row["direction"] == "short" for row in hedge_rows)
    assert is_ibkr_flex_index_hedge_row(next(row for row in hedge_rows if row["ticker"] == "SPY"))


def test_merge_ibkr_flex_hedge_replacement_preserves_unmatched_existing():
    imported = [
        {"ticker": "SPY", "instrument_type": "security", "direction": "short", "shares": 30, "position_id": "SPY"}
    ]
    existing = [
        {"ticker": "IWM", "instrument_type": "security", "direction": "short", "shares": 5, "position_id": "IWM"},
        {"ticker": "SPY", "instrument_type": "security", "direction": "short", "shares": 10, "position_id": "SPY"},
    ]
    merged = merge_ibkr_flex_hedge_replacement(imported, existing)
    tickers = [row["ticker"] for row in merged]
    assert tickers == ["IWM", "SPY"]
    spy = next(row for row in merged if row["ticker"] == "SPY")
    assert spy["shares"] == 30
