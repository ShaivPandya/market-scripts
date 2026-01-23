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


@dataclass
class DiagnosticInfo:
    """Diagnostic information about an asset's breakout status."""
    market: str
    name: str
    ticker: str
    in_box: bool
    box_upper: Optional[float]
    box_lower: Optional[float]
    days_in_box: Optional[int]
    close: float
    atr: float
    buffer: float
    dist_to_up_breakout_pct: Optional[float]    # % distance to upper breakout level
    dist_to_down_breakout_pct: Optional[float]  # % distance to lower breakout level
    vol_ratio: Optional[float]
    vol_threshold: float
    vol_method: str
    status: str  # "in_box", "near_miss_up", "near_miss_down", "not_in_box"
    near_miss_reason: Optional[str]  # e.g., "volume ratio 1.08x < 1.20x threshold"
    # Extended diagnostics for detailed output
    bb_width: Optional[float] = None              # Current Bollinger Band width
    bb_width_percentile: Optional[float] = None   # Where current width sits vs historical (0-100)
    tight_count: Optional[int] = None             # How many of last 20 days were "tight"
    tight_threshold: int = 8                      # k_min threshold
    pct_change_5d: Optional[float] = None         # 5-day price change %
    pct_change_20d: Optional[float] = None        # 20-day price change %
    high_20d: Optional[float] = None              # 20-day high
    low_20d: Optional[float] = None               # 20-day low
    dist_from_high_pct: Optional[float] = None    # % distance from 20-day high
    dist_from_low_pct: Optional[float] = None     # % distance from 20-day low


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


def _compute_extended_diagnostics(df: pd.DataFrame, p: Params) -> Dict:
    """Compute extended diagnostic metrics from the dataframe."""
    if df.empty or len(df) < 20:
        return {}

    last_row = df.iloc[-1]
    close = float(last_row["Close"])

    result = {}

    # BB Width and percentile
    bb_width = last_row.get("BBWidth", np.nan)
    if not pd.isna(bb_width):
        result["bb_width"] = float(bb_width)
        # Calculate percentile of current BB width in last 252 days
        bb_hist = df["BBWidth"].dropna().tail(252)
        if len(bb_hist) > 0:
            percentile = (bb_hist < bb_width).sum() / len(bb_hist) * 100
            result["bb_width_percentile"] = float(percentile)

    # Tight count
    tight_count = last_row.get("TightCount", np.nan)
    if not pd.isna(tight_count):
        result["tight_count"] = int(tight_count)
    result["tight_threshold"] = p.k_min

    # Price changes
    if len(df) >= 6:
        close_5d_ago = df["Close"].iloc[-6]
        if not pd.isna(close_5d_ago) and close_5d_ago != 0:
            result["pct_change_5d"] = ((close - close_5d_ago) / close_5d_ago) * 100

    if len(df) >= 21:
        close_20d_ago = df["Close"].iloc[-21]
        if not pd.isna(close_20d_ago) and close_20d_ago != 0:
            result["pct_change_20d"] = ((close - close_20d_ago) / close_20d_ago) * 100

    # 20-day high/low
    recent_20 = df.tail(20)
    high_20d = recent_20["High"].max()
    low_20d = recent_20["Low"].min()

    if not pd.isna(high_20d):
        result["high_20d"] = float(high_20d)
        if close != 0:
            result["dist_from_high_pct"] = ((high_20d - close) / close) * 100

    if not pd.isna(low_20d):
        result["low_20d"] = float(low_20d)
        if close != 0:
            result["dist_from_low_pct"] = ((close - low_20d) / close) * 100

    return result


def detect_latest_breakout(
    df: pd.DataFrame,
    market: str,
    name: str,
    ticker: str,
    p: Params
) -> Tuple[Optional[Signal], DiagnosticInfo]:
    """
    Walk forward, maintain latest active congestion box, and return a signal only if the latest bar breaks out.
    Commodities: frozen box.
    FX: hybrid (allow small boundary updates within fx_tol_atr * ATR).

    Returns:
        Tuple of (Signal or None, DiagnosticInfo with current state)
    """

    # Default diagnostic for empty dataframe
    if df.empty:
        return None, DiagnosticInfo(
            market=market, name=name, ticker=ticker, in_box=False,
            box_upper=None, box_lower=None, days_in_box=None,
            close=0.0, atr=0.0, buffer=0.0,
            dist_to_up_breakout_pct=None, dist_to_down_breakout_pct=None,
            vol_ratio=None, vol_threshold=p.vol_mult, vol_method="UNKNOWN",
            status="not_in_box", near_miss_reason=None
        )

    is_fx = (market == "FX")

    # Decide whether to use TR proxy for "volume expansion"
    # If recent volume is mostly zeros/NaN, treat as unavailable.
    recent_vol = df["Volume"].tail(60)
    vol_unusable = recent_vol.isna().all() or (recent_vol.fillna(0).median() == 0)
    use_tr_proxy = is_fx and vol_unusable

    if use_tr_proxy:
        df = df.copy()
        df["TR_MA"] = df["TR"].rolling(p.vol_ma_n, min_periods=p.vol_ma_n).mean()

    # Compute extended diagnostics once
    ext_diag = _compute_extended_diagnostics(df, p)

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
            direction = "UP" if up_break else "DOWN"

            # Only return a signal if the breakout is on the latest bar AND volume confirms
            if i == len(df) - 1:
                days_in = i - box_start_idx if box_start_idx is not None else 0
                atr_val = float(row["ATR"])

                if confirm:
                    # Confirmed breakout
                    diag = DiagnosticInfo(
                        market=market, name=name, ticker=ticker, in_box=True,
                        box_upper=box_upper, box_lower=box_lower, days_in_box=days_in,
                        close=close, atr=atr_val, buffer=buffer,
                        dist_to_up_breakout_pct=0.0 if up_break else None,
                        dist_to_down_breakout_pct=0.0 if dn_break else None,
                        vol_ratio=vol_ratio, vol_threshold=p.vol_mult, vol_method=vol_method,
                        status=f"breakout_{direction.lower()}", near_miss_reason=None,
                        **ext_diag
                    )
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
                    ), diag
                else:
                    # Near miss: price broke out but volume didn't confirm
                    ratio_str = f"{vol_ratio:.2f}x" if vol_ratio else "N/A"
                    near_miss_reason = f"volume ratio {ratio_str} < {p.vol_mult:.2f}x threshold"
                    diag = DiagnosticInfo(
                        market=market, name=name, ticker=ticker, in_box=True,
                        box_upper=box_upper, box_lower=box_lower, days_in_box=days_in,
                        close=close, atr=atr_val, buffer=buffer,
                        dist_to_up_breakout_pct=0.0 if up_break else None,
                        dist_to_down_breakout_pct=0.0 if dn_break else None,
                        vol_ratio=vol_ratio, vol_threshold=p.vol_mult, vol_method=vol_method,
                        status=f"near_miss_{direction.lower()}", near_miss_reason=near_miss_reason,
                        **ext_diag
                    )
                    return None, diag

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

    # Build diagnostic info from the final state (latest bar)
    last_row = df.iloc[-1]
    last_close = float(last_row["Close"])
    last_atr = float(last_row["ATR"]) if not pd.isna(last_row["ATR"]) else 0.0
    last_buffer = p.buffer_k * last_atr

    # Get volume info for latest bar
    _, last_vol_ratio, last_vol_method = _volume_confirmation(last_row, p, use_tr_proxy)

    if in_box and not math.isnan(box_upper) and not math.isnan(box_lower):
        # Currently in a box but no breakout on latest bar
        days_in = (len(df) - 1 - box_start_idx) if box_start_idx is not None else 0
        up_level = box_upper + last_buffer
        down_level = box_lower - last_buffer
        dist_up_pct = ((up_level - last_close) / last_close) * 100 if last_close > 0 else None
        dist_down_pct = ((last_close - down_level) / last_close) * 100 if last_close > 0 else None

        return None, DiagnosticInfo(
            market=market, name=name, ticker=ticker, in_box=True,
            box_upper=box_upper, box_lower=box_lower, days_in_box=days_in,
            close=last_close, atr=last_atr, buffer=last_buffer,
            dist_to_up_breakout_pct=dist_up_pct,
            dist_to_down_breakout_pct=dist_down_pct,
            vol_ratio=last_vol_ratio, vol_threshold=p.vol_mult, vol_method=last_vol_method,
            status="in_box", near_miss_reason=None,
            **ext_diag
        )
    else:
        # Not currently in a congestion box
        return None, DiagnosticInfo(
            market=market, name=name, ticker=ticker, in_box=False,
            box_upper=None, box_lower=None, days_in_box=None,
            close=last_close, atr=last_atr, buffer=last_buffer,
            dist_to_up_breakout_pct=None, dist_to_down_breakout_pct=None,
            vol_ratio=last_vol_ratio, vol_threshold=p.vol_mult, vol_method=last_vol_method,
            status="not_in_box", near_miss_reason=None,
            **ext_diag
        )


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

    # If only one ticker, yfinance returns single-level columns (not MultiIndex)
    if not isinstance(raw.columns, pd.MultiIndex):
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
# Terminal colors and formatting
# ----------------------------

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Colors
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    WHITE = "\033[37m"

    # Bright variants
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"


def _colorize(text: str, color: str) -> str:
    """Wrap text with color codes."""
    return f"{color}{text}{Colors.RESET}"


def _progress_bar(value: float, max_value: float, width: int = 20, fill_char: str = "█", empty_char: str = "░") -> str:
    """Create a simple progress bar."""
    if max_value <= 0:
        return empty_char * width
    ratio = min(value / max_value, 1.0)
    filled = int(ratio * width)
    return fill_char * filled + empty_char * (width - filled)


def _position_bar(pct_from_low: float, width: int = 20) -> str:
    """Create a position indicator bar showing where price is in range."""
    pct = max(0, min(100, pct_from_low))
    pos = int((pct / 100) * (width - 1))
    bar = "─" * pos + "●" + "─" * (width - 1 - pos)
    return f"L [{bar}] H"


# ----------------------------
# Diagnostic output
# ----------------------------

def _volatility_description(percentile: Optional[float]) -> str:
    """Return a human-readable volatility description based on BB width percentile."""
    if percentile is None:
        return "N/A"
    if percentile <= 20:
        return "very tight"
    elif percentile <= 40:
        return "relatively tight"
    elif percentile <= 60:
        return "normal"
    elif percentile <= 80:
        return "elevated"
    else:
        return "high"


def _ordinal(n: int) -> str:
    """Return ordinal string for a number (1st, 2nd, 3rd, 4th, etc.)."""
    if 11 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def print_diagnostic_summary(diagnostics: List[DiagnosticInfo]) -> None:
    """Print a comprehensive summary of breakout analysis for all assets."""
    C = Colors

    # Header
    print()
    print(f"{C.BOLD}{C.CYAN}╔{'═' * 68}╗{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}║{C.RESET}{C.BOLD}{'BREAKOUT ANALYSIS SUMMARY':^68}{C.RESET}{C.BOLD}{C.CYAN}║{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}╚{'═' * 68}╝{C.RESET}")

    # Categorize assets
    in_box = [d for d in diagnostics if d.status == "in_box"]
    near_misses = [d for d in diagnostics if d.status.startswith("near_miss")]
    not_in_box = [d for d in diagnostics if d.status == "not_in_box"]

    # Sort in_box by closest to breakout (minimum of up/down distance)
    def min_distance(d: DiagnosticInfo) -> float:
        up = d.dist_to_up_breakout_pct if d.dist_to_up_breakout_pct is not None else float('inf')
        dn = d.dist_to_down_breakout_pct if d.dist_to_down_breakout_pct is not None else float('inf')
        return min(abs(up), abs(dn))

    in_box.sort(key=min_distance)

    # Print assets in consolidation boxes
    if in_box:
        print(f"\n{C.BOLD}{C.YELLOW}◆ CONSOLIDATION BOXES{C.RESET} {C.DIM}({len(in_box)} potential breakout{'s' if len(in_box) != 1 else ''}){C.RESET}")
        print(f"{C.DIM}{'─' * 68}{C.RESET}")

        for d in in_box:
            box_str = f"{d.box_lower:.4f} → {d.box_upper:.4f}" if d.box_lower and d.box_upper else "N/A"
            day_bar = _progress_bar(d.days_in_box or 0, 60, width=10)

            print(f"\n  {C.BOLD}{C.WHITE}{d.name}{C.RESET} {C.DIM}({d.market}){C.RESET}")
            print(f"  {C.DIM}├─{C.RESET} Box: {C.CYAN}{box_str}{C.RESET}")
            print(f"  {C.DIM}├─{C.RESET} Duration: {day_bar} {C.DIM}Day {d.days_in_box}/60{C.RESET}")
            print(f"  {C.DIM}├─{C.RESET} Close: {C.WHITE}{d.close:.4f}{C.RESET}")

            # Distance indicators with color coding
            if d.dist_to_up_breakout_pct is not None:
                up_color = C.BRIGHT_GREEN if d.dist_to_up_breakout_pct < 1 else C.GREEN
                up_str = f"{up_color}▲ {d.dist_to_up_breakout_pct:+.2f}%{C.RESET}"
            else:
                up_str = f"{C.DIM}▲ N/A{C.RESET}"

            if d.dist_to_down_breakout_pct is not None:
                dn_color = C.BRIGHT_RED if d.dist_to_down_breakout_pct < 1 else C.RED
                dn_str = f"{dn_color}▼ {d.dist_to_down_breakout_pct:+.2f}%{C.RESET}"
            else:
                dn_str = f"{C.DIM}▼ N/A{C.RESET}"

            print(f"  {C.DIM}├─{C.RESET} Distance: {up_str}  {dn_str}")

            # Volume with threshold indicator
            vol_str = f"{d.vol_ratio:.2f}x" if d.vol_ratio is not None else "N/A"
            vol_color = C.BRIGHT_GREEN if d.vol_ratio and d.vol_ratio >= d.vol_threshold else C.DIM
            print(f"  {C.DIM}└─{C.RESET} Volume: {vol_color}{vol_str}{C.RESET} {C.DIM}(need {d.vol_threshold:.2f}x) [{d.vol_method}]{C.RESET}")
    else:
        print(f"\n{C.BOLD}{C.YELLOW}◆ CONSOLIDATION BOXES{C.RESET} {C.DIM}(none){C.RESET}")

    # Print near misses
    if near_misses:
        print(f"\n{C.BOLD}{C.MAGENTA}◇ NEAR MISSES{C.RESET} {C.DIM}({len(near_misses)} - price broke but volume didn't confirm){C.RESET}")
        print(f"{C.DIM}{'─' * 68}{C.RESET}")

        for d in near_misses:
            direction = "UP" if "up" in d.status else "DOWN"
            dir_color = C.BRIGHT_GREEN if direction == "UP" else C.BRIGHT_RED
            dir_symbol = "▲" if direction == "UP" else "▼"

            print(f"\n  {C.BOLD}{C.WHITE}{d.name}{C.RESET} {C.DIM}({d.market}){C.RESET}")
            print(f"  {C.DIM}├─{C.RESET} Direction: {dir_color}{dir_symbol} {direction}{C.RESET}  Close: {C.WHITE}{d.close:.4f}{C.RESET}")
            print(f"  {C.DIM}└─{C.RESET} {C.YELLOW}⚠ {d.near_miss_reason}{C.RESET}")
    else:
        print(f"\n{C.BOLD}{C.MAGENTA}◇ NEAR MISSES{C.RESET} {C.DIM}(none){C.RESET}")

    # Print detailed info for assets not in consolidation
    if not_in_box:
        # Sort by how close they are to forming a congestion box (tight_count descending)
        not_in_box_sorted = sorted(
            not_in_box,
            key=lambda d: d.tight_count if d.tight_count is not None else 0,
            reverse=True
        )

        print(f"\n{C.BOLD}{C.BLUE}○ NOT IN CONSOLIDATION{C.RESET} {C.DIM}({len(not_in_box)}){C.RESET}")
        print(f"{C.DIM}{'─' * 68}{C.RESET}")

        for d in not_in_box_sorted:
            print(f"\n  {C.BOLD}{C.WHITE}{d.name}{C.RESET} {C.DIM}({d.market}){C.RESET}")

            # Price info
            print(f"  {C.DIM}├─{C.RESET} Price: {C.WHITE}{d.close:.4f}{C.RESET}  ATR: {C.CYAN}{d.atr:.4f}{C.RESET}")

            # Range and position
            if d.high_20d is not None and d.low_20d is not None:
                if d.dist_from_low_pct is not None and d.dist_from_high_pct is not None:
                    total_range = d.dist_from_low_pct + d.dist_from_high_pct
                    if total_range > 0:
                        pct_from_low = (d.dist_from_low_pct / total_range) * 100
                        pos_bar = _position_bar(pct_from_low, width=16)
                        print(f"  {C.DIM}├─{C.RESET} 20d Range: {pos_bar} {C.DIM}{d.low_20d:.4f} - {d.high_20d:.4f}{C.RESET}")

            # Trend indicators
            chg_5d = d.pct_change_5d
            chg_20d = d.pct_change_20d
            chg_5d_color = C.BRIGHT_GREEN if chg_5d and chg_5d > 0 else C.BRIGHT_RED if chg_5d and chg_5d < 0 else C.DIM
            chg_20d_color = C.BRIGHT_GREEN if chg_20d and chg_20d > 0 else C.BRIGHT_RED if chg_20d and chg_20d < 0 else C.DIM
            chg_5d_str = f"{chg_5d:+.2f}%" if chg_5d is not None else "N/A"
            chg_20d_str = f"{chg_20d:+.2f}%" if chg_20d is not None else "N/A"
            print(f"  {C.DIM}├─{C.RESET} Trend: 5d {chg_5d_color}{chg_5d_str}{C.RESET}  20d {chg_20d_color}{chg_20d_str}{C.RESET}")

            # Volatility
            vol_desc = _volatility_description(d.bb_width_percentile)
            bb_str = f"{d.bb_width:.2%}" if d.bb_width is not None else "N/A"
            pctl_str = _ordinal(int(d.bb_width_percentile)) if d.bb_width_percentile is not None else "N/A"
            vol_color = C.BRIGHT_CYAN if vol_desc in ["very tight", "relatively tight"] else C.DIM
            print(f"  {C.DIM}├─{C.RESET} Volatility: {vol_color}{vol_desc}{C.RESET} {C.DIM}(BB: {bb_str}, {pctl_str} pctl){C.RESET}")

            # Congestion progress
            tight = d.tight_count if d.tight_count is not None else 0
            threshold = d.tight_threshold
            tight_bar = _progress_bar(tight, threshold, width=10)
            needed = max(0, threshold - tight)
            if needed == 0:
                status_str = f"{C.BRIGHT_YELLOW}watching for box{C.RESET}"
            else:
                status_str = f"{C.DIM}{needed} more needed{C.RESET}"
            print(f"  {C.DIM}└─{C.RESET} Congestion: {tight_bar} {C.DIM}{tight}/{threshold}{C.RESET} {status_str}")

    # Footer
    print()
    print(f"{C.DIM}{'─' * 68}{C.RESET}")
    print(f"{C.DIM}No confirmed breakouts on latest bar.{C.RESET}")
    print()


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
    diagnostics: List[DiagnosticInfo] = []

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
        sig, diag = detect_latest_breakout(feats, market, name, ticker, P)
        diagnostics.append(diag)
        if sig:
            signals.append(sig)

    if not signals:
        print_diagnostic_summary(diagnostics)
        return

    # Print confirmed breakouts with nice formatting
    C = Colors

    print()
    print(f"{C.BOLD}{C.BRIGHT_GREEN}╔{'═' * 68}╗{C.RESET}")
    print(f"{C.BOLD}{C.BRIGHT_GREEN}║{C.RESET}{C.BOLD}{'🚀 CONFIRMED BREAKOUTS':^68}{C.RESET}{C.BOLD}{C.BRIGHT_GREEN}║{C.RESET}")
    print(f"{C.BOLD}{C.BRIGHT_GREEN}╚{'═' * 68}╝{C.RESET}")

    # Sort by market then name
    signals_sorted = sorted(signals, key=lambda s: (s.market, s.name))

    for s in signals_sorted:
        dir_color = C.BRIGHT_GREEN if s.direction == "UP" else C.BRIGHT_RED
        dir_symbol = "▲" if s.direction == "UP" else "▼"
        box_str = f"{s.box_lower:.4f} → {s.box_upper:.4f}"

        print()
        print(f"  {C.BOLD}{C.WHITE}{s.name}{C.RESET} {C.DIM}({s.market}){C.RESET}  {dir_color}{C.BOLD}{dir_symbol} {s.direction} BREAKOUT{C.RESET}")
        print(f"  {C.DIM}{'─' * 50}{C.RESET}")
        print(f"  {C.DIM}├─{C.RESET} Date:     {C.WHITE}{s.date.date().isoformat()}{C.RESET}")
        print(f"  {C.DIM}├─{C.RESET} Close:    {dir_color}{C.BOLD}{s.close:.6f}{C.RESET}")
        print(f"  {C.DIM}├─{C.RESET} Box:      {C.CYAN}{box_str}{C.RESET}")
        print(f"  {C.DIM}├─{C.RESET} Buffer:   {C.DIM}{s.buffer:.6f}{C.RESET}")

        vol_str = f"{s.vol_ratio:.2f}x" if s.vol_ratio else "N/A"
        print(f"  {C.DIM}└─{C.RESET} Volume:   {C.BRIGHT_GREEN}{vol_str}{C.RESET} {C.DIM}[{s.vol_method}]{C.RESET}")

    print()
    print(f"{C.DIM}{'─' * 68}{C.RESET}")
    print(f"{C.BOLD}{C.WHITE}Total: {len(signals)} confirmed breakout{'s' if len(signals) != 1 else ''}{C.RESET}")
    print()


if __name__ == "__main__":
    main()
