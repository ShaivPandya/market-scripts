"""
Kovner-style breakout detector (daily): tight congestion box -> close-based breakout + volume expansion confirmation.

Universe:
  FX (hybrid box): EURUSD, GBPUSD, AUDUSD, NZDUSD, USDCAD, USDCHF, USDJPY
  Commodities (frozen box): Gold, Silver, Copper, Platinum

Data source: Yahoo Finance via yfinance (daily bars).
Note on FX "volume": Yahoo spot FX often has Volume=0. If so, this script falls back to a participation proxy
(True Range vs its moving average) for the "volume expansion" confirmation and labels it accordingly.

Usage:
  pip install yfinance pandas numpy
  python3 breakout.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
# Configuration
# ----------------------------

@dataclass
class Params:
    # Congestion definition
    Wc: int = 20                 # Donchian window
    atr_n: int = 20
    bb_n: int = 20
    bb_k: float = 2.0
    bb_q: float = 0.20           # 20th percentile threshold
    bb_lookback: int = 252       # approx 1 trading year
    k_min: int = 8               # >= 8 of last 20 days meet tightness filters
    range_atr_mult: float = 2.0  # box_range <= 2.0 * ATR

    # Breakout trigger
    buffer_k: float = 0.20       # buffer = 0.20 * ATR
    vol_ma_n: int = 20           # volume MA window
    vol_mult: float = 1.20       # Volume_t > 1.20 * VolMA_t

    # Box lifecycle
    miss_limit: int = 3          # end box if tightness fails this many consecutive days (no breakout)
    max_box_len: int = 60        # safety cap

    # FX hybrid tolerance (boundary updates allowed only within tol * ATR)
    fx_tol_atr: float = 0.15


P = Params()


# ----------------------------
# Universe / Yahoo tickers
# ----------------------------

UNIVERSE: Dict[str, Dict[str, str]] = {
    "FX": {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "AUDUSD": "AUDUSD=X",
        "NZDUSD": "NZDUSD=X",
        # Yahoo commonly uses "CAD=X" for USD/CAD, "CHF=X" for USD/CHF, "JPY=X" for USD/JPY
        "USDCAD": "CAD=X",
        "USDCHF": "CHF=X",
        "USDJPY": "JPY=X",
    },
    "COMMODITIES": {
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Copper": "HG=F",
        "Platinum": "PL=F",
    }
}


# ----------------------------
# Indicators
# ----------------------------

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def compute_features(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    df = df.copy()

    # TR / ATR
    df["TR"] = true_range(df)
    df["ATR"] = df["TR"].rolling(p.atr_n, min_periods=p.atr_n).mean()

    # Donchian box over Wc
    df["DonchianUpper"] = df["High"].rolling(p.Wc, min_periods=p.Wc).max()
    df["DonchianLower"] = df["Low"].rolling(p.Wc, min_periods=p.Wc).min()
    df["BoxRange"] = df["DonchianUpper"] - df["DonchianLower"]

    # Bollinger width
    ma = df["Close"].rolling(p.bb_n, min_periods=p.bb_n).mean()
    sd = df["Close"].rolling(p.bb_n, min_periods=p.bb_n).std(ddof=0)
    upper_bb = ma + p.bb_k * sd
    lower_bb = ma - p.bb_k * sd
    df["BBWidth"] = (upper_bb - lower_bb) / ma.replace(0, np.nan)

    # Rolling 20th percentile of BBWidth over ~1y lookback
    df["BBWidthQ"] = df["BBWidth"].rolling(p.bb_lookback, min_periods=p.bb_lookback).quantile(p.bb_q)

    # Tightness condition per day
    df["Tight"] = (
        (df["BoxRange"] <= p.range_atr_mult * df["ATR"]) &
        (df["BBWidth"] <= df["BBWidthQ"])
    )

    # Duration rule: at least k_min of last Wc days are tight
    df["TightCount"] = df["Tight"].rolling(p.Wc, min_periods=p.Wc).sum()
    df["CongestionConfirmed"] = df["TightCount"] >= p.k_min

    # Volume confirmation precompute
    if "Volume" in df.columns:
        df["VolMA"] = df["Volume"].rolling(p.vol_ma_n, min_periods=p.vol_ma_n).mean()
    else:
        df["Volume"] = np.nan
        df["VolMA"] = np.nan

    return df


# ----------------------------
# Box / breakout detection
# ----------------------------

@dataclass
class Signal:
    market: str
    name: str
    ticker: str
    date: pd.Timestamp
    direction: str               # "UP" or "DOWN"
    close: float
    box_upper: float
    box_lower: float
    buffer: float
    confirm: bool
    vol_ratio: Optional[float]
    vol_method: str              # "VOLUME" or "TR_PROXY"


def _volume_confirmation(row: pd.Series, p: Params, use_tr_proxy: bool) -> Tuple[bool, Optional[float], str]:
    if use_tr_proxy:
        # "Participation" proxy using TR instead of volume
        proxy = row.get("TR", np.nan)
        proxy_ma = row.get("TR_MA", np.nan)
        if pd.isna(proxy) or pd.isna(proxy_ma) or proxy_ma == 0:
            return False, None, "TR_PROXY"
        ratio = float(proxy / proxy_ma)
        return ratio > p.vol_mult, ratio, "TR_PROXY"

    vol = row.get("Volume", np.nan)
    vol_ma = row.get("VolMA", np.nan)
    if pd.isna(vol) or pd.isna(vol_ma) or vol_ma == 0:
        return False, None, "VOLUME"
    ratio = float(vol / vol_ma)
    return ratio > p.vol_mult, ratio, "VOLUME"


def detect_latest_breakout(
    df: pd.DataFrame,
    market: str,
    name: str,
    ticker: str,
    p: Params
) -> Optional[Signal]:
    """
    Walk forward, maintain latest active congestion box, and return a signal only if the latest bar breaks out.
    Commodities: frozen box.
    FX: hybrid (allow small boundary updates within fx_tol_atr * ATR).
    """

    if df.empty:
        return None

    is_fx = (market == "FX")

    # Decide whether to use TR proxy for "volume expansion"
    # If recent volume is mostly zeros/NaN, treat as unavailable.
    recent_vol = df["Volume"].tail(60)
    vol_unusable = recent_vol.isna().all() or (recent_vol.fillna(0).median() == 0)
    use_tr_proxy = is_fx and vol_unusable

    if use_tr_proxy:
        df = df.copy()
        df["TR_MA"] = df["TR"].rolling(p.vol_ma_n, min_periods=p.vol_ma_n).mean()

    in_box = False
    box_upper = np.nan
    box_lower = np.nan
    box_start_idx = None
    miss_count = 0

    # Iterate in time order
    for i in range(len(df)):
        row = df.iloc[i]
        dt = df.index[i]

        # Skip until indicators are available
        if pd.isna(row["ATR"]) or pd.isna(row["DonchianUpper"]) or pd.isna(row["BBWidthQ"]):
            continue

        confirmed = bool(row["CongestionConfirmed"])

        if not in_box:
            # Start a new box when congestion is confirmed
            if confirmed:
                in_box = True
                box_upper = float(row["DonchianUpper"])
                box_lower = float(row["DonchianLower"])
                box_start_idx = i
                miss_count = 0
            continue

        # If box is active: optionally update boundaries (FX hybrid only)
        if is_fx:
            tol = p.fx_tol_atr * float(row["ATR"])
            d_up = float(row["DonchianUpper"]) - box_upper
            d_dn = box_lower - float(row["DonchianLower"])

            if d_up > 0 and d_up <= tol:
                box_upper = float(row["DonchianUpper"])
            if d_dn > 0 and d_dn <= tol:
                box_lower = float(row["DonchianLower"])

        # Evaluate breakout first (even if congestion isn't confirmed today)
        buffer = p.buffer_k * float(row["ATR"])
        close = float(row["Close"])

        up_break = close > (box_upper + buffer)
        dn_break = close < (box_lower - buffer)

        if up_break or dn_break:
            confirm, vol_ratio, vol_method = _volume_confirmation(row, p, use_tr_proxy)
            # Only return a signal if the breakout is on the latest bar
            if i == len(df) - 1 and confirm:
                direction = "UP" if up_break else "DOWN"
                return Signal(
                    market=market,
                    name=name,
                    ticker=ticker,
                    date=dt,
                    direction=direction,
                    close=close,
                    box_upper=box_upper,
                    box_lower=box_lower,
                    buffer=buffer,
                    confirm=True,
                    vol_ratio=vol_ratio,
                    vol_method=vol_method,
                )
            # Box resolves regardless of confirmation once price exits materially
            in_box = False
            box_upper = box_lower = np.nan
            box_start_idx = None
            miss_count = 0
            continue

        # No breakout: manage lifecycle using "misses" of tightness confirmation
        if confirmed:
            miss_count = 0
        else:
            miss_count += 1

        # End box if no longer tight for too long or too old
        if miss_count >= p.miss_limit:
            in_box = False
            box_upper = box_lower = np.nan
            box_start_idx = None
            miss_count = 0
            continue

        if box_start_idx is not None and (i - box_start_idx) >= p.max_box_len:
            in_box = False
            box_upper = box_lower = np.nan
            box_start_idx = None
            miss_count = 0
            continue

    return None


# ----------------------------
# Data retrieval
# ----------------------------

def download_daily(tickers: List[str], period: str = "3y") -> Dict[str, pd.DataFrame]:
    """
    Download daily OHLCV for all tickers. Returns dict[ticker] -> DataFrame
    """
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

    # If only one ticker, yfinance returns single-level columns
    if isinstance(raw.columns, pd.Index):
        # Single ticker case
        df = raw.dropna(how="all")
        out[tickers[0]] = df
        return out

    # Multi-ticker case: top level = ticker
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            continue
        df = raw[t].dropna(how="all")
        if not df.empty:
            out[t] = df

    return out


# ----------------------------
# Main
# ----------------------------

def main():
    # Flatten tickers list
    tickers = []
    meta = []  # (market, name, ticker)
    for market, items in UNIVERSE.items():
        for name, t in items.items():
            tickers.append(t)
            meta.append((market, name, t))

    data = download_daily(tickers, period="3y")

    signals: List[Signal] = []

    for market, name, ticker in meta:
        df = data.get(ticker)
        if df is None or df.empty:
            continue

        # Standardize columns and index
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        needed = {"Open", "High", "Low", "Close"}
        if not needed.issubset(df.columns):
            continue

        feats = compute_features(df, P)
        sig = detect_latest_breakout(feats, market, name, ticker, P)
        if sig:
            signals.append(sig)

    if not signals:
        print("No confirmed breakouts found on the latest daily bar for the defined universe.")
        return

    # Print results
    rows = []
    for s in signals:
        rows.append({
            "Market": s.market,
            "Name": s.name,
            "Ticker": s.ticker,
            "Date": s.date.date().isoformat(),
            "Direction": s.direction,
            "Close": round(s.close, 6),
            "BoxUpper": round(s.box_upper, 6),
            "BoxLower": round(s.box_lower, 6),
            "Buffer": round(s.buffer, 6),
            "Confirmed": s.confirm,
            "VolMethod": s.vol_method,
            "VolRatio": None if s.vol_ratio is None else round(s.vol_ratio, 3),
        })

    out_df = pd.DataFrame(rows).sort_values(["Market", "Name"])
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
"""
Breakout detector (daily): tight congestion box -> close-based breakout + volume expansion confirmation.

Universe:
  FX (hybrid box): EURUSD, GBPUSD, AUDUSD, NZDUSD, USDCAD, USDCHF, USDJPY
  Commodities (frozen box): Gold, Silver, Copper, Platinum

Data source: Yahoo Finance via yfinance (daily bars).
Note on FX "volume": Yahoo spot FX often has Volume=0. If so, this script falls back to a participation proxy
(True Range vs its moving average) for the "volume expansion" confirmation and labels it accordingly.

Usage:
  pip install yfinance pandas numpy
  python3 breakout.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
# Configuration
# ----------------------------

@dataclass
class Params:
    # Congestion definition
    Wc: int = 20                 # Donchian window
    atr_n: int = 20
    bb_n: int = 20
    bb_k: float = 2.0
    bb_q: float = 0.20           # 20th percentile threshold
    bb_lookback: int = 252       # approx 1 trading year
    k_min: int = 8               # >= 8 of last 20 days meet tightness filters
    range_atr_mult: float = 2.0  # box_range <= 2.0 * ATR

    # Breakout trigger
    buffer_k: float = 0.20       # buffer = 0.20 * ATR
    vol_ma_n: int = 20           # volume MA window
    vol_mult: float = 1.20       # Volume_t > 1.20 * VolMA_t

    # Box lifecycle
    miss_limit: int = 3          # end box if tightness fails this many consecutive days (no breakout)
    max_box_len: int = 60        # safety cap

    # FX hybrid tolerance (boundary updates allowed only within tol * ATR)
    fx_tol_atr: float = 0.15


P = Params()


# ----------------------------
# Universe / Yahoo tickers
# ----------------------------

UNIVERSE: Dict[str, Dict[str, str]] = {
    "FX": {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "AUDUSD": "AUDUSD=X",
        "NZDUSD": "NZDUSD=X",
        # Yahoo commonly uses "CAD=X" for USD/CAD, "CHF=X" for USD/CHF, "JPY=X" for USD/JPY
        "USDCAD": "CAD=X",
        "USDCHF": "CHF=X",
        "USDJPY": "JPY=X",
    },
    "COMMODITIES": {
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Copper": "HG=F",
        "Platinum": "PL=F",
    }
}


# ----------------------------
# Indicators
# ----------------------------

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def compute_features(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    df = df.copy()

    # TR / ATR
    df["TR"] = true_range(df)
    df["ATR"] = df["TR"].rolling(p.atr_n, min_periods=p.atr_n).mean()

    # Donchian box over Wc
    df["DonchianUpper"] = df["High"].rolling(p.Wc, min_periods=p.Wc).max()
    df["DonchianLower"] = df["Low"].rolling(p.Wc, min_periods=p.Wc).min()
    df["BoxRange"] = df["DonchianUpper"] - df["DonchianLower"]

    # Bollinger width
    ma = df["Close"].rolling(p.bb_n, min_periods=p.bb_n).mean()
    sd = df["Close"].rolling(p.bb_n, min_periods=p.bb_n).std(ddof=0)
    upper_bb = ma + p.bb_k * sd
    lower_bb = ma - p.bb_k * sd
    df["BBWidth"] = (upper_bb - lower_bb) / ma.replace(0, np.nan)

    # Rolling 20th percentile of BBWidth over ~1y lookback
    df["BBWidthQ"] = df["BBWidth"].rolling(p.bb_lookback, min_periods=p.bb_lookback).quantile(p.bb_q)

    # Tightness condition per day
    df["Tight"] = (
        (df["BoxRange"] <= p.range_atr_mult * df["ATR"]) &
        (df["BBWidth"] <= df["BBWidthQ"])
    )

    # Duration rule: at least k_min of last Wc days are tight
    df["TightCount"] = df["Tight"].rolling(p.Wc, min_periods=p.Wc).sum()
    df["CongestionConfirmed"] = df["TightCount"] >= p.k_min

    # Volume confirmation precompute
    if "Volume" in df.columns:
        df["VolMA"] = df["Volume"].rolling(p.vol_ma_n, min_periods=p.vol_ma_n).mean()
    else:
        df["Volume"] = np.nan
        df["VolMA"] = np.nan

    return df


# ----------------------------
# Box / breakout detection
# ----------------------------

@dataclass
class Signal:
    market: str
    name: str
    ticker: str
    date: pd.Timestamp
    direction: str               # "UP" or "DOWN"
    close: float
    box_upper: float
    box_lower: float
    buffer: float
    confirm: bool
    vol_ratio: Optional[float]
    vol_method: str              # "VOLUME" or "TR_PROXY"


def _volume_confirmation(row: pd.Series, p: Params, use_tr_proxy: bool) -> Tuple[bool, Optional[float], str]:
    if use_tr_proxy:
        # "Participation" proxy using TR instead of volume
        proxy = row.get("TR", np.nan)
        proxy_ma = row.get("TR_MA", np.nan)
        if pd.isna(proxy) or pd.isna(proxy_ma) or proxy_ma == 0:
            return False, None, "TR_PROXY"
        ratio = float(proxy / proxy_ma)
        return ratio > p.vol_mult, ratio, "TR_PROXY"

    vol = row.get("Volume", np.nan)
    vol_ma = row.get("VolMA", np.nan)
    if pd.isna(vol) or pd.isna(vol_ma) or vol_ma == 0:
        return False, None, "VOLUME"
    ratio = float(vol / vol_ma)
    return ratio > p.vol_mult, ratio, "VOLUME"


def detect_latest_breakout(
    df: pd.DataFrame,
    market: str,
    name: str,
    ticker: str,
    p: Params
) -> Optional[Signal]:
    """
    Walk forward, maintain latest active congestion box, and return a signal only if the latest bar breaks out.
    Commodities: frozen box.
    FX: hybrid (allow small boundary updates within fx_tol_atr * ATR).
    """

    if df.empty:
        return None

    is_fx = (market == "FX")

    # Decide whether to use TR proxy for "volume expansion"
    # If recent volume is mostly zeros/NaN, treat as unavailable.
    recent_vol = df["Volume"].tail(60)
    vol_unusable = recent_vol.isna().all() or (recent_vol.fillna(0).median() == 0)
    use_tr_proxy = is_fx and vol_unusable

    if use_tr_proxy:
        df = df.copy()
        df["TR_MA"] = df["TR"].rolling(p.vol_ma_n, min_periods=p.vol_ma_n).mean()

    in_box = False
    box_upper = np.nan
    box_lower = np.nan
    box_start_idx = None
    miss_count = 0

    # Iterate in time order
    for i in range(len(df)):
        row = df.iloc[i]
        dt = df.index[i]

        # Skip until indicators are available
        if pd.isna(row["ATR"]) or pd.isna(row["DonchianUpper"]) or pd.isna(row["BBWidthQ"]):
            continue

        confirmed = bool(row["CongestionConfirmed"])

        if not in_box:
            # Start a new box when congestion is confirmed
            if confirmed:
                in_box = True
                box_upper = float(row["DonchianUpper"])
                box_lower = float(row["DonchianLower"])
                box_start_idx = i
                miss_count = 0
            continue

        # If box is active: optionally update boundaries (FX hybrid only)
        if is_fx:
            tol = p.fx_tol_atr * float(row["ATR"])
            d_up = float(row["DonchianUpper"]) - box_upper
            d_dn = box_lower - float(row["DonchianLower"])

            if d_up > 0 and d_up <= tol:
                box_upper = float(row["DonchianUpper"])
            if d_dn > 0 and d_dn <= tol:
                box_lower = float(row["DonchianLower"])

        # Evaluate breakout first (even if congestion isn't confirmed today)
        buffer = p.buffer_k * float(row["ATR"])
        close = float(row["Close"])

        up_break = close > (box_upper + buffer)
        dn_break = close < (box_lower - buffer)

        if up_break or dn_break:
            confirm, vol_ratio, vol_method = _volume_confirmation(row, p, use_tr_proxy)
            # Only return a signal if the breakout is on the latest bar
            if i == len(df) - 1 and confirm:
                direction = "UP" if up_break else "DOWN"
                return Signal(
                    market=market,
                    name=name,
                    ticker=ticker,
                    date=dt,
                    direction=direction,
                    close=close,
                    box_upper=box_upper,
                    box_lower=box_lower,
                    buffer=buffer,
                    confirm=True,
                    vol_ratio=vol_ratio,
                    vol_method=vol_method,
                )
            # Box resolves regardless of confirmation once price exits materially
            in_box = False
            box_upper = box_lower = np.nan
            box_start_idx = None
            miss_count = 0
            continue

        # No breakout: manage lifecycle using "misses" of tightness confirmation
        if confirmed:
            miss_count = 0
        else:
            miss_count += 1

        # End box if no longer tight for too long or too old
        if miss_count >= p.miss_limit:
            in_box = False
            box_upper = box_lower = np.nan
            box_start_idx = None
            miss_count = 0
            continue

        if box_start_idx is not None and (i - box_start_idx) >= p.max_box_len:
            in_box = False
            box_upper = box_lower = np.nan
            box_start_idx = None
            miss_count = 0
            continue

    return None


# ----------------------------
# Data retrieval
# ----------------------------

def download_daily(tickers: List[str], period: str = "3y") -> Dict[str, pd.DataFrame]:
    """
    Download daily OHLCV for all tickers. Returns dict[ticker] -> DataFrame
    """
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

    # If only one ticker, yfinance returns single-level columns
    if isinstance(raw.columns, pd.Index):
        # Single ticker case
        df = raw.dropna(how="all")
        out[tickers[0]] = df
        return out

    # Multi-ticker case: top level = ticker
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            continue
        df = raw[t].dropna(how="all")
        if not df.empty:
            out[t] = df

    return out


# ----------------------------
# Main
# ----------------------------

def main():
    # Flatten tickers list
    tickers = []
    meta = []  # (market, name, ticker)
    for market, items in UNIVERSE.items():
        for name, t in items.items():
            tickers.append(t)
            meta.append((market, name, t))

    data = download_daily(tickers, period="3y")

    signals: List[Signal] = []

    for market, name, ticker in meta:
        df = data.get(ticker)
        if df is None or df.empty:
            continue

        # Standardize columns and index
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        needed = {"Open", "High", "Low", "Close"}
        if not needed.issubset(df.columns):
            continue

        feats = compute_features(df, P)
        sig = detect_latest_breakout(feats, market, name, ticker, P)
        if sig:
            signals.append(sig)

    if not signals:
        print("No confirmed breakouts found on the latest daily bar for the defined universe.")
        return

    # Print results
    rows = []
    for s in signals:
        rows.append({
            "Market": s.market,
            "Name": s.name,
            "Ticker": s.ticker,
            "Date": s.date.date().isoformat(),
            "Direction": s.direction,
            "Close": round(s.close, 6),
            "BoxUpper": round(s.box_upper, 6),
            "BoxLower": round(s.box_lower, 6),
            "Buffer": round(s.buffer, 6),
            "Confirmed": s.confirm,
            "VolMethod": s.vol_method,
            "VolRatio": None if s.vol_ratio is None else round(s.vol_ratio, 3),
        })

    out_df = pd.DataFrame(rows).sort_values(["Market", "Name"])
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
