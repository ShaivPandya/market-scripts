from __future__ import annotations

import pandas as pd


def test_fetch_full_price_history_formats_close_prices(monkeypatch):
    import portfolio.technical_analysis.technical_analysis as technical_analysis

    def fake_yf_download(*args, **kwargs):
        assert args[0] == "MU"
        assert kwargs["period"] == "max"
        return pd.DataFrame(
            {"Close": [10.125, None, 11.5]},
            index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
        )

    monkeypatch.setattr(technical_analysis, "yf_download", fake_yf_download)

    df = technical_analysis.fetch_full_price_history("MU")

    assert list(df.columns) == ["Date", "Close"]
    assert df.to_dict("records") == [
        {"Date": "2020-01-02", "Close": 10.125},
        {"Date": "2020-01-06", "Close": 11.5},
    ]


def test_download_price_history_csv(auth_client, monkeypatch):
    import portfolio.technical_analysis.technical_analysis as technical_analysis

    def fake_fetch_full_price_history(ticker: str):
        assert ticker == "BRK-B"
        return pd.DataFrame(
            [
                {"Date": "2020-01-02", "Close": 10.5},
                {"Date": "2020-01-03", "Close": 11.25},
            ]
        )

    monkeypatch.setattr(technical_analysis, "fetch_full_price_history", fake_fetch_full_price_history)

    resp = auth_client.get("/api/chart/price-history/brk-b")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert 'filename="BRK_B_price_history.csv"' in resp.headers["content-disposition"]
    assert resp.text == "Date,Close\n2020-01-02,10.5\n2020-01-03,11.25\n"
