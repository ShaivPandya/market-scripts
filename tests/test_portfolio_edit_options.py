import pytest

from api.routers.portfolio_edit import PortfolioPosition, PortfolioUpdateRequest
from ontology.action_registry import PortfolioPositionInput, UpdatePortfolioPositionsInput


def _option_row(**overrides):
    base = {
        "ticker": "META",
        "asset": "equity",
        "direction": "long",
        "contrarian": False,
        "conviction": 3,
        "cost_basis": 12.5,
        "shares": 2,
        "instrument_type": "option",
        "underlying_ticker": "META",
        "option_expiration": "2026-01-16",
        "option_strike": 500,
        "option_type": "call",
    }
    base.update(overrides)
    return base


def test_portfolio_position_normalizes_valid_option_row():
    row = PortfolioPosition(**_option_row())
    assert row.instrument_type == "option"
    assert row.option_contract_symbol == "META260116C00500000"
    assert row.position_id == "META260116C00500000"
    assert row.contract_multiplier == 100.0


def test_portfolio_update_request_accepts_multiple_meta_legs():
    payload = PortfolioUpdateRequest(
        positions=[
            PortfolioPosition(
                **{
                    "ticker": "META",
                    "asset": "equity",
                    "direction": "long",
                    "contrarian": False,
                    "conviction": 4,
                    "cost_basis": 500.0,
                    "shares": 10,
                    "instrument_type": "security",
                }
            ),
            PortfolioPosition(**_option_row()),
            PortfolioPosition(**_option_row(option_strike=450, option_type="put")),
        ]
    )
    ids = [position.position_id for position in payload.positions]
    assert len(set(ids)) == 3


def test_portfolio_update_accepts_mixed_direction_group():
    rows = [
        {
            "ticker": "NVDA",
            "asset": "equity",
            "direction": "long",
            "contrarian": False,
            "conviction": 5,
            "cost_basis": 100.0,
            "shares": 10,
            "instrument_type": "security",
            "group_name": "Semiconductors",
            "group_conviction": 5,
        },
        {
            "ticker": "AMD",
            "asset": "equity",
            "direction": "short",
            "contrarian": False,
            "conviction": 3,
            "cost_basis": 50.0,
            "shares": 5,
            "instrument_type": "security",
            "group_name": "semiconductors",
            "group_conviction": 5,
        },
    ]

    request = PortfolioUpdateRequest(positions=[PortfolioPosition(**row) for row in rows])
    proposal = UpdatePortfolioPositionsInput(positions=[PortfolioPositionInput(**row) for row in rows])

    assert [position.group_name for position in request.positions] == ["Semiconductors", "Semiconductors"]
    assert [position.group_name for position in proposal.positions] == ["Semiconductors", "Semiconductors"]


def test_portfolio_update_payload_rejects_duplicate_position_ids():
    row = _option_row()
    with pytest.raises(ValueError, match="Duplicate position_id"):
        UpdatePortfolioPositionsInput(
            positions=[
                PortfolioPositionInput(**row),
                PortfolioPositionInput(**row),
            ]
        )
