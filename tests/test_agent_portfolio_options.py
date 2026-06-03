from api.agent_tools import _build_agent_portfolio_payload


def test_agent_portfolio_payload_includes_underlying_exposure_for_option_legs():
    holdings = [
        {
            "ticker": "META",
            "asset": "equity",
            "direction": "long",
            "shares": 10,
            "instrument_type": "security",
            "position_id": "META",
        },
        {
            "ticker": "META",
            "asset": "equity",
            "direction": "long",
            "shares": 2,
            "instrument_type": "option",
            "underlying_ticker": "META",
            "option_contract_symbol": "META260116C00500000",
            "option_expiration": "2026-01-16",
            "option_strike": 500.0,
            "option_type": "call",
            "contract_multiplier": 100.0,
            "position_id": "META260116C00500000",
        },
    ]
    raw = {
        "positions": {"META": [{"date": "2026-01-01", "value": 500.0}]},
        "analytics": {
            "per_position": {
                "META": {"current_price": 520.0, "current_notional": 5200.0},
                "META260116C00500000": {"current_price": 15.0, "current_notional": 3000.0},
            },
            "portfolio": {"position_count": 2},
        },
        "underlying_exposures": [
            {
                "underlying_ticker": "META",
                "tickers": ["META"],
                "legs": ["META", "META260116C00500000"],
                "current_notional": 8200.0,
                "cost_notional": 6400.0,
            }
        ],
    }

    payload = _build_agent_portfolio_payload(raw, holdings, include_hedges=False)

    assert payload["summary"]["position_count"] == 2
    assert payload["underlying_exposures"][0]["underlying_ticker"] == "META"
    assert set(payload["underlying_exposures"][0]["legs"]) == {"META", "META260116C00500000"}
    option_rows = [row for row in payload["positions"] if row.get("instrument_type") == "option"]
    assert len(option_rows) == 1
    assert option_rows[0]["option_contract_symbol"] == "META260116C00500000"
    assert option_rows[0]["display_ticker"] == "META"
