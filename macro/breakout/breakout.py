"""
Congestion-regime breakout detector (daily).

Formula:
  atr = ATR(14) using Wilder smoothing (RMA)
  atr <= lowest(atr, 100) * 1.25
  range_high = highest(high, 30)
  range_low = lowest(low, 30)
  (range_high - range_low) / atr <= 6
  abs(close - SMA(20)) <= atr
  congestion = all conditions true

Breakout rules (no stops/exits):
  long_breakout = congestion[1] and close > range_high[1]
  short_breakout = congestion[1] and close < range_low[1]

`get_data()` response schema:
  {
    "latest": [ ... ],
    "events": [ ... ],
    "history": {ticker: {market, name, ticker, rows:[ ... ]}},
    "params": { ... },
    "error": "..."  # only on failure
  }
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class Params:
    atr_period: int = 14
    congestion_lookback: int = 30
    atr_compression_lookback: int = 100
    sma_period: int = 20
    atr_compression_mult: float = 1.25
    range_atr_max: float = 6.0


P = Params()


UNIVERSE: Dict[str, Dict[str, str]] = {
    "FX": {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "AUDUSD": "AUDUSD=X",
        "NZDUSD": "NZDUSD=X",
        "USDCAD": "CAD=X",
        "USDCHF": "CHF=X",
        "USDJPY": "JPY=X",
    },
    "COMMODITIES": {
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Copper": "HG=F",
        "Platinum": "PL=F",
        "Aluminum": "ALI=F",
        "Palladium": "PA=F",
    },
}


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def wilder_rma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def compute_features(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    df = df.copy().sort_index()

    df["TR"] = true_range(df)
    df["ATR"] = wilder_rma(df["TR"], p.atr_period)
    df["ATRMin100"] = df["ATR"].rolling(
        p.atr_compression_lookback,
        min_periods=p.atr_compression_lookback,
    ).min()

    df["RangeHigh30"] = df["High"].rolling(
        p.congestion_lookback,
        min_periods=p.congestion_lookback,
    ).max()
    df["RangeLow30"] = df["Low"].rolling(
        p.congestion_lookback,
        min_periods=p.congestion_lookback,
    ).min()

    df["SMA20"] = df["Close"].rolling(p.sma_period, min_periods=p.sma_period).mean()

    range_width = df["RangeHigh30"] - df["RangeLow30"]
    df["RangeATRRatio"] = np.where(df["ATR"] > 0, range_width / df["ATR"], np.nan)

    df["CondATRCompression"] = df["ATR"] <= (df["ATRMin100"] * p.atr_compression_mult)
    df["CondRangeATR"] = df["RangeATRRatio"] <= p.range_atr_max
    df["CondSMADistance"] = (df["Close"] - df["SMA20"]).abs() <= df["ATR"]

    df["Congestion"] = (
        df["CondATRCompression"]
        & df["CondRangeATR"]
        & df["CondSMADistance"]
    )

    df["PrevCongestion"] = df["Congestion"].shift(1).eq(True)
    df["RangeHigh30Prev"] = df["RangeHigh30"].shift(1)
    df["RangeLow30Prev"] = df["RangeLow30"].shift(1)

    df["LongBreakout"] = df["PrevCongestion"] & (df["Close"] > df["RangeHigh30Prev"])
    df["ShortBreakout"] = df["PrevCongestion"] & (df["Close"] < df["RangeLow30Prev"])

    overlap = df["LongBreakout"] & df["ShortBreakout"]
    if overlap.any():
        df.loc[overlap, ["LongBreakout", "ShortBreakout"]] = False

    return df


def download_daily(tickers: List[str], period: str = "3y") -> Dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    out: Dict[str, pd.DataFrame] = {}

    if raw is None or raw.empty:
        return out

    if not isinstance(raw.columns, pd.MultiIndex):
        out[tickers[0]] = raw.dropna(how="all")
        return out

    for ticker in tickers:
        if ticker not in raw.columns.get_level_values(0):
            continue
        df = raw[ticker].dropna(how="all")
        if not df.empty:
            out[ticker] = df

    return out


def _universe_meta() -> Tuple[List[str], List[Tuple[str, str, str]]]:
    tickers: List[str] = []
    meta: List[Tuple[str, str, str]] = []
    for market, items in UNIVERSE.items():
        for name, ticker in items.items():
            tickers.append(ticker)
            meta.append((market, name, ticker))
    return tickers, meta


def _to_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _to_bool(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return bool(value)


def _asset_latest_row(
    market: str,
    name: str,
    ticker: str,
    feats: pd.DataFrame,
) -> Dict[str, Any]:
    row = feats.iloc[-1]
    dt = feats.index[-1]

    long_breakout = _to_bool(row["LongBreakout"])
    short_breakout = _to_bool(row["ShortBreakout"])

    direction = None
    if long_breakout:
        direction = "LONG"
    elif short_breakout:
        direction = "SHORT"

    return {
        "market": market,
        "name": name,
        "ticker": ticker,
        "date": pd.Timestamp(dt).date().isoformat(),
        "close": _to_float(row["Close"]),
        "atr": _to_float(row["ATR"]),
        "atr_min100": _to_float(row["ATRMin100"]),
        "range_high30": _to_float(row["RangeHigh30"]),
        "range_low30": _to_float(row["RangeLow30"]),
        "sma20": _to_float(row["SMA20"]),
        "range_atr_ratio": _to_float(row["RangeATRRatio"]),
        "cond_atr_compression": _to_bool(row["CondATRCompression"]),
        "cond_range_atr": _to_bool(row["CondRangeATR"]),
        "cond_sma_distance": _to_bool(row["CondSMADistance"]),
        "congestion": _to_bool(row["Congestion"]),
        "long_breakout": long_breakout,
        "short_breakout": short_breakout,
        "direction": direction,
    }


def _asset_events(
    market: str,
    name: str,
    ticker: str,
    feats: pd.DataFrame,
) -> List[Dict[str, Any]]:
    event_mask = feats["LongBreakout"] | feats["ShortBreakout"]
    if not event_mask.any():
        return []

    events: List[Dict[str, Any]] = []
    for dt, row in feats.loc[event_mask].iterrows():
        direction = "LONG" if _to_bool(row["LongBreakout"]) else "SHORT"
        events.append(
            {
                "market": market,
                "name": name,
                "ticker": ticker,
                "date": pd.Timestamp(dt).date().isoformat(),
                "direction": direction,
                "close": _to_float(row["Close"]),
                "atr": _to_float(row["ATR"]),
                "range_high30_prev": _to_float(row["RangeHigh30Prev"]),
                "range_low30_prev": _to_float(row["RangeLow30Prev"]),
                "congestion_prev": _to_bool(row["PrevCongestion"]),
            }
        )

    return events


def _asset_history_rows(feats: pd.DataFrame) -> List[Dict[str, Any]]:
    cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "ATR",
        "ATRMin100",
        "RangeHigh30",
        "RangeLow30",
        "SMA20",
        "RangeATRRatio",
        "CondATRCompression",
        "CondRangeATR",
        "CondSMADistance",
        "Congestion",
        "LongBreakout",
        "ShortBreakout",
    ]

    rows: List[Dict[str, Any]] = []
    for dt, row in feats[cols].iterrows():
        rows.append(
            {
                "date": pd.Timestamp(dt).date().isoformat(),
                "open": _to_float(row["Open"]),
                "high": _to_float(row["High"]),
                "low": _to_float(row["Low"]),
                "close": _to_float(row["Close"]),
                "atr": _to_float(row["ATR"]),
                "atr_min100": _to_float(row["ATRMin100"]),
                "range_high30": _to_float(row["RangeHigh30"]),
                "range_low30": _to_float(row["RangeLow30"]),
                "sma20": _to_float(row["SMA20"]),
                "range_atr_ratio": _to_float(row["RangeATRRatio"]),
                "cond_atr_compression": _to_bool(row["CondATRCompression"]),
                "cond_range_atr": _to_bool(row["CondRangeATR"]),
                "cond_sma_distance": _to_bool(row["CondSMADistance"]),
                "congestion": _to_bool(row["Congestion"]),
                "long_breakout": _to_bool(row["LongBreakout"]),
                "short_breakout": _to_bool(row["ShortBreakout"]),
            }
        )

    return rows


def _analyze(period: str = "3y") -> Dict[str, Any]:
    tickers, meta = _universe_meta()
    raw = download_daily(tickers, period=period)

    latest: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    history: Dict[str, Dict[str, Any]] = {}

    for market, name, ticker in meta:
        df = raw.get(ticker)
        if df is None or df.empty:
            continue

        df = df.copy()
        df.index = pd.to_datetime(df.index)

        needed = {"Open", "High", "Low", "Close"}
        if not needed.issubset(df.columns):
            continue

        feats = compute_features(df, P)

        latest.append(_asset_latest_row(market, name, ticker, feats))
        events.extend(_asset_events(market, name, ticker, feats))
        history[ticker] = {
            "market": market,
            "name": name,
            "ticker": ticker,
            "rows": _asset_history_rows(feats),
        }

    latest.sort(key=lambda x: (x["market"], x["name"]))
    events.sort(key=lambda x: (x["date"], x["market"], x["name"]), reverse=True)

    return {
        "latest": latest,
        "events": events,
        "history": history,
        "params": asdict(P),
    }


def get_data() -> Dict[str, Any]:
    try:
        return _analyze(period="3y")
    except Exception as e:
        return {"error": str(e)}


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None or pd.isna(v):
        return "N/A"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _print_latest_breakout_section(latest: List[Dict[str, Any]], direction: str) -> None:
    items = [x for x in latest if x.get("direction") == direction]
    title = "LATEST LONG BREAKOUTS" if direction == "LONG" else "LATEST SHORT BREAKOUTS"

    print(f"\n{title}")
    print("-" * len(title))
    if not items:
        print("None")
        return

    for item in items:
        print(
            f"{item['date']}  {item['name']} ({item['market']})  "
            f"close={_fmt(item['close'])}  "
            f"range=[{_fmt(item['range_low30'])}, {_fmt(item['range_high30'])}]"
        )


def _print_congestion_section(latest: List[Dict[str, Any]]) -> None:
    items = [x for x in latest if x.get("congestion")]

    print("\nCURRENT CONGESTION REGIME")
    print("-" * 25)
    if not items:
        print("None")
        return

    for item in items:
        print(
            f"{item['date']}  {item['name']} ({item['market']})  "
            f"close={_fmt(item['close'])}  atr={_fmt(item['atr'])}  "
            f"ratio={_fmt(item['range_atr_ratio'], digits=3)}"
        )


def _print_recent_events_section(events: List[Dict[str, Any]], limit: int = 20) -> None:
    print(f"\nRECENT HISTORICAL BREAKOUT EVENTS (last {limit})")
    print("-" * 45)

    if not events:
        print("None")
        return

    for ev in events[:limit]:
        print(
            f"{ev['date']}  {ev['name']} ({ev['market']})  {ev['direction']}  "
            f"close={_fmt(ev['close'])}"
        )


def main() -> None:
    result = get_data()

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    latest = result.get("latest", [])
    events = result.get("events", [])

    print("\nCONGESTION-REGIME BREAKOUT SCAN")
    print("=" * 31)
    print(
        "Formula: atr<=min(atr,100)*1.25, (range30/atr)<=6, |close-sma20|<=atr; "
        "breakout uses prior-bar congestion + prior range boundaries."
    )

    _print_latest_breakout_section(latest, "LONG")
    _print_latest_breakout_section(latest, "SHORT")
    _print_congestion_section(latest)
    _print_recent_events_section(events, limit=20)


if __name__ == "__main__":
    main()
