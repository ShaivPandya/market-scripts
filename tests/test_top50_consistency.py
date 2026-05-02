from __future__ import annotations

import pandas as pd


def test_top50_breadth_prices_df_uses_shared_126_session_ranking(monkeypatch):
    from equities.market_technicals import top50_breadth
    from equities.market_technicals.get_top50 import compute_top50_from_close

    dates = pd.bdate_range("2025-10-01", periods=130)
    close = pd.DataFrame(
        {
            "AAA": range(100, 230),
            "BBB": [100 + i * 0.5 for i in range(130)],
            "CCC": [100] * 130,
            "DDD": range(230, 100, -1),
        },
        index=dates,
    )
    prices = pd.concat(
        {
            "Close": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Volume": close * 1000,
        },
        axis=1,
    )
    expected = compute_top50_from_close(close)["ticker"].tolist()
    captured: dict[str, list[str]] = {}

    def fake_compute_metrics(tickers, period="2y", prices_df=None):
        captured["tickers"] = list(tickers)
        return pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "rows": 30,
                    "below_50dma": False,
                    "dist_days_last20": 0,
                    "has_3plus_dist_days": False,
                    "broke_prior20_low_last_week": False,
                }
                for ticker in tickers
            ]
        )

    monkeypatch.setattr(top50_breadth, "compute_metrics", fake_compute_metrics)
    top50_breadth.get_data(prices_df=prices)

    assert captured["tickers"] == expected
