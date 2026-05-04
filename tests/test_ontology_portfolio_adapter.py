from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from ontology.sources.portfolio import PortfolioAdapter


def test_portfolio_adapter_normalizes_pandas_series_position():
    raw = {
        "positions": {
            "NVDA": pd.Series([100.0, 101.5], index=pd.date_range("2026-05-01", periods=2)),
        },
        "metadata": {
            "NVDA": {"asset": "equity", "direction": "long"},
        },
        "position_order": ["NVDA"],
        "timeframe": "Daily",
        "timestamp": datetime(2026, 5, 4, 20, 23, tzinfo=UTC),
        "analytics": {},
    }

    result = PortfolioAdapter(timeframe="Daily").normalize(raw)

    assert result.status == "ok"
    assert result.data is not None
    position = result.data.positions["NVDA"]
    assert position.latest_price == 101.5
    assert position.series_points == 2
