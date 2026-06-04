from __future__ import annotations

import pytest

from portfolio.position_groups import (
    canonicalize_position_group_rows,
    group_key,
    normalize_group_name,
    validate_position_groups,
)


def test_group_name_normalization_and_matching_key():
    assert normalize_group_name("Memory") == "Memory"
    assert normalize_group_name("Memory ") == "Memory"
    assert normalize_group_name("Memory   Cycle") == "Memory Cycle"
    assert normalize_group_name("Cafe\u0301") == "Café"
    assert normalize_group_name(float("nan")) is None
    assert normalize_group_name("N/A") is None
    assert normalize_group_name("none") is None
    assert normalize_group_name("Ungrouped") is None
    assert group_key("Memory") == group_key(" memory ")


def test_group_validation_rejects_conflicting_conviction_and_direction():
    with pytest.raises(ValueError, match="inconsistent group convictions"):
        validate_position_groups(
            [
                {"ticker": "MU", "direction": "long", "group_name": "Memory", "group_conviction": 5},
                {"ticker": "SKH", "direction": "long", "group_name": "memory", "group_conviction": 4},
            ]
        )


def test_canonicalize_position_group_rows_preserves_first_display_name():
    rows = canonicalize_position_group_rows(
        [
            {"ticker": "SKH", "direction": "long", "group_name": "Memory   Cycle", "group_conviction": 5},
            {"ticker": "MU", "direction": "long", "group_name": "memory cycle", "group_conviction": 5},
        ]
    )

    assert [row["group_name"] for row in rows] == ["Memory Cycle", "Memory Cycle"]

    with pytest.raises(ValueError, match="cannot mix long and short"):
        validate_position_groups(
            [
                {"ticker": "MU", "direction": "long", "group_name": "Memory", "group_conviction": 5},
                {"ticker": "SSNLF", "direction": "short", "group_name": "Memory", "group_conviction": 5},
            ]
        )


def test_group_validation_uses_option_exposure_direction():
    validate_position_groups(
        [
            {"ticker": "MRVL", "direction": "long", "group_name": "Semiconductors", "group_conviction": 5},
            {
                "ticker": "MRVL",
                "direction": "short",
                "instrument_type": "option",
                "option_type": "put",
                "group_name": "Semiconductors",
                "group_conviction": 5,
            },
        ]
    )

    with pytest.raises(ValueError, match="cannot mix long and short"):
        validate_position_groups(
            [
                {"ticker": "MRVL", "direction": "long", "group_name": "Semiconductors", "group_conviction": 5},
                {
                    "ticker": "MRVL",
                    "direction": "long",
                    "instrument_type": "option",
                    "option_type": "put",
                    "group_name": "Semiconductors",
                    "group_conviction": 5,
                },
            ]
        )
