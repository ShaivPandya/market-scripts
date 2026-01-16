#!/usr/bin/env python3
"""
Momentum + Relative Strength Regime Script

Computes:
- Rate of Change (ROC) + acceleration
- Log-price regression slope + R² (trend strength & quality)
- Relative Price vs benchmark

Produces:
- Composite momentum score (weighted rolling z-scores)
- Daily regime labels: bullish / caution / bearish / neutral

Regime      Action
--------    ------
bullish     Hold / add (positive trend, clean structure)
caution     Tighten stop, reduce (warning signs present)
bearish     Avoid / exit (clear downtrend)
neutral     Wait (no clear signal)

Caution triggers (any ONE fires the signal):
- Decelerating momentum (roc > 0 but accel < 0 and score falling)
- Near highs but choppy (R² < 0.6) — potential topping
- Significant relative weakness (lagging benchmark by 1%+)

Dependencies: pandas, numpy
Optional: yfinance (for fetching), matplotlib (for plotting)

Examples:
  python3 momentum_regime.py --ticker AAPL --benchmark SPY --start 2020-01-01
  python3 momentum_regime.py --ticker AAPL --benchmark SPY --start 2020-01-01 --plot
  python3 momentum_regime.py --ticker MSFT --benchmark QQQ --roc 20 --zwin 60 --plot
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def try_import_yfinance():
    try:
        import yfinance as yf  # type: ignore
        return yf
    except Exception:
        return None


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


def rolling_slope(s: pd.Series, window: int) -> pd.Series:
    """
    Rolling OLS slope vs time index 0..window-1.
    Returns slope per day (units of s per day).
    """
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        if np.any(np.isnan(y)):
            return np.nan
        y_mean = y.mean()
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    return s.rolling(window, min_periods=window).apply(lambda v: _slope(v.to_numpy()), raw=False)


def rolling_log_slope_r2(s: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
    """
    Rolling OLS of log(price) vs time index 0..window-1.
    Returns (slope_per_day, r_squared).

    slope: average log-return per day over the window
    r²: goodness of fit (1.0 = perfect linear trend, 0.0 = no trend)
    """
    log_s = np.log(s)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    ss_x = ((x - x_mean) ** 2).sum()

    slopes = []
    r2s = []

    for i in range(len(log_s)):
        if i < window - 1:
            slopes.append(np.nan)
            r2s.append(np.nan)
            continue

        y = log_s.iloc[i - window + 1 : i + 1].values
        if np.any(np.isnan(y)):
            slopes.append(np.nan)
            r2s.append(np.nan)
            continue

        y_mean = y.mean()
        ss_y = ((y - y_mean) ** 2).sum()

        if ss_y == 0:
            # Flat line - perfect fit but zero slope
            slopes.append(0.0)
            r2s.append(1.0)
            continue

        cov_xy = ((x - x_mean) * (y - y_mean)).sum()
        slope = cov_xy / ss_x
        r2 = (cov_xy ** 2) / (ss_x * ss_y)

        slopes.append(slope)
        r2s.append(r2)

    return (
        pd.Series(slopes, index=s.index, name="log_slope"),
        pd.Series(r2s, index=s.index, name="log_r2"),
    )


@dataclass
class Config:
    ticker: str
    benchmark: str
    start: str
    end: Optional[str]
    roc_period: int
    ewm_span: int
    zwin: int
    trend_win: int
    flat_win: int
    ma_fast: int
    ma_slow: int
    w_roc: float
    w_accel: float
    w_rel: float
    flat_roc_thresh: float
    near_high_pct: float
    rel_lag_thresh: float
    log_slope_win: int
    r2_thresh: float
    plot: bool
    csv_ticker: Optional[str]
    csv_benchmark: Optional[str]


def fetch_adj_close_yf(ticker: str, start: str, end: Optional[str]) -> pd.Series:
    yf = try_import_yfinance()
    if yf is None:
        raise RuntimeError(
            "yfinance is not installed/available. Install with: pip install yfinance\n"
            "Or provide --csv-ticker/--csv-benchmark pointing to CSVs with a Date column."
        )
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    s = df[col].copy()
    s.name = ticker
    return s


def fetch_from_csv(path: str, series_name: str) -> pd.Series:
    df = pd.read_csv(path)
    # Expect Date + (Adj Close or Close) or a single price column
    date_col = None
    for c in df.columns:
        if c.lower() in ("date", "datetime", "time"):
            date_col = c
            break
    if date_col is None:
        raise RuntimeError(f"{path}: CSV must contain a Date column.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    price_col = None
    for c in df.columns:
        if c.lower() in ("adj close", "adj_close", "adjusted_close", "close"):
            price_col = c
            break
    if price_col is None:
        # fallback: if exactly one non-date column exists
        if df.shape[1] == 1:
            price_col = df.columns[0]
        else:
            raise RuntimeError(
                f"{path}: Could not find a price column. Expected Adj Close/Close or a single column."
            )

    s = df[price_col].astype(float).copy()
    s.name = series_name
    return s


def align_series(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    df = pd.concat([a, b], axis=1).dropna()
    return df.iloc[:, 0], df.iloc[:, 1]


def compute_features(price: pd.Series, bench: pd.Series, cfg: Config) -> pd.DataFrame:
    # Base series
    px = price.astype(float)
    bx = bench.astype(float)

    # ROC + smoothing
    roc = px.pct_change(cfg.roc_period)
    roc_s = roc.ewm(span=cfg.ewm_span, adjust=False).mean()

    # Acceleration: change in smoothed ROC
    accel = roc_s.diff()

    # Relative price + relative ROC
    rel = px / bx
    rel_roc = rel.pct_change(cfg.roc_period)
    rel_roc_s = rel_roc.ewm(span=cfg.ewm_span, adjust=False).mean()

    # Moving averages (for “flag vs top” structure)
    ma_fast = px.rolling(cfg.ma_fast, min_periods=cfg.ma_fast).mean()
    ma_slow = px.rolling(cfg.ma_slow, min_periods=cfg.ma_slow).mean()

    # Rolling highs to check “near highs”
    roll_max = px.rolling(cfg.flat_win, min_periods=cfg.flat_win).max()
    near_high = (px / roll_max) >= (1.0 - cfg.near_high_pct)

    # Z-scores for composite momentum
    roc_z = rolling_zscore(roc_s, cfg.zwin)
    accel_z = rolling_zscore(accel, cfg.zwin)
    rel_z = rolling_zscore(rel_roc_s, cfg.zwin)

    score = cfg.w_roc * roc_z + cfg.w_accel * accel_z + cfg.w_rel * rel_z

    # Trends (slopes)
    score_slope = rolling_slope(score, cfg.trend_win)
    roc_slope = rolling_slope(roc_s, cfg.flat_win)
    rel_slope = rolling_slope(rel_roc_s, cfg.flat_win)

    # Log-price regression slope + R² (trend strength & quality)
    log_slope, log_r2 = rolling_log_slope_r2(px, cfg.log_slope_win)

    out = pd.DataFrame(
        {
            "price": px,
            "benchmark": bx,
            "roc": roc,
            "roc_s": roc_s,
            "accel": accel,
            "rel": rel,
            "rel_roc": rel_roc,
            "rel_roc_s": rel_roc_s,
            "ma_fast": ma_fast,
            "ma_slow": ma_slow,
            "near_high": near_high.astype(float),
            "roc_z": roc_z,
            "accel_z": accel_z,
            "rel_z": rel_z,
            "score": score,
            "score_slope": score_slope,
            "roc_slope": roc_slope,
            "rel_slope": rel_slope,
            "log_slope": log_slope,
            "log_r2": log_r2,
        }
    )
    return out


def classify_regime(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    Simplified 4-state regime classification:

    bullish:  Positive trend with clean structure (hold/add)
    caution:  Warning signs present (tighten stop, reduce)
    bearish:  Clear downtrend (avoid/exit)
    neutral:  No clear signal (wait)
    """
    roc_s = df["roc_s"]
    accel = df["accel"]
    rel_roc_s = df["rel_roc_s"]
    score_slope = df["score_slope"]
    log_slope = df["log_slope"]
    log_r2 = df["log_r2"]
    px = df["price"]
    ma_slow = df["ma_slow"]
    near_high = df["near_high"] > 0.6

    # Trend quality threshold (0.6 = moderate quality)
    clean_trend = log_r2 >= 0.6
    choppy = log_r2 < 0.6

    # BULLISH: Positive trend with clean structure
    bullish = (
        (log_slope > 0)      # Regression trend positive
        & clean_trend        # Reasonably clean trend (R² >= 0.6)
        & (px >= ma_slow)    # Above trend support
    )

    # CAUTION: Any ONE warning sign triggers caution (OR logic)
    caution = (
        # Trigger 1: Decelerating momentum while still positive
        ((roc_s > 0) & (accel < 0) & (score_slope < 0))
        |
        # Trigger 2: Near highs but choppy (low R²) — potential topping
        (near_high & choppy)
        |
        # Trigger 3: Significant relative weakness (lagging benchmark by 1%+)
        (rel_roc_s < -0.01)
    )

    # BEARISH: Clear downtrend
    bearish = (
        (log_slope < 0)      # Regression trend negative
        & (roc_s < 0)        # Momentum negative
        & (px < ma_slow)     # Below trend support
    )

    # Priority: caution > bullish > bearish > neutral
    # (caution overrides bullish if warning signs present)
    regime = pd.Series("neutral", index=df.index)
    regime[bearish] = "bearish"
    regime[bullish] = "bullish"
    regime[caution] = "caution"  # Caution last = highest priority

    return regime


def _r2_label(r2: float, thresh: float) -> str:
    """Return trend quality label based on R²."""
    if r2 >= thresh:
        return "strong trend"
    elif r2 >= 0.4:
        return "moderate"
    else:
        return "choppy"


def summarize_latest(df: pd.DataFrame, regime: pd.Series, cfg: Config) -> str:
    last = df.dropna().iloc[-1]
    last_reg = regime.loc[last.name]
    lines = []
    lines.append(f"Date: {last.name.date()}")
    lines.append(f"{cfg.ticker} price: {last['price']:.2f} | {cfg.benchmark}: {last['benchmark']:.2f}")
    lines.append(f"ROC({cfg.roc_period}) smoothed: {last['roc_s']*100:.2f}% | Accel: {last['accel']*100:.2f}%")

    # Log-price regression slope: daily % and annualized %
    log_slope_daily_pct = (np.exp(last['log_slope']) - 1) * 100
    log_slope_ann_pct = (np.exp(last['log_slope'] * 252) - 1) * 100
    r2_val = last['log_r2']
    r2_label = _r2_label(r2_val, cfg.r2_thresh)
    lines.append(f"Log slope: {log_slope_daily_pct:.2f}%/day (~{log_slope_ann_pct:.0f}% ann) | R²: {r2_val:.2f} ({r2_label})")

    lines.append(f"Relative ({cfg.ticker}/{cfg.benchmark}): {last['rel']:.4f}")
    lines.append(f"Rel ROC smoothed: {last['rel_roc_s']*100:.2f}%")
    lines.append(f"Momentum score: {last['score']:.2f} | Score slope({cfg.trend_win}): {last['score_slope']:.4f}")
    lines.append(f"Regime: {last_reg}")
    return "\n".join(lines)


def plot_df(df: pd.DataFrame, regime: pd.Series, cfg: Config) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not available; skipping plot. Install with: pip install matplotlib", file=sys.stderr)
        return

    # 1) Price + MAs
    plt.figure()
    plt.plot(df.index, df["price"], label=f"{cfg.ticker} price")
    plt.plot(df.index, df["ma_fast"], label=f"MA{cfg.ma_fast}")
    plt.plot(df.index, df["ma_slow"], label=f"MA{cfg.ma_slow}")
    plt.title("Price and Moving Averages")
    plt.legend()
    plt.tight_layout()

    # 2) ROC_s and Accel
    plt.figure()
    plt.plot(df.index, df["roc_s"] * 100.0, label="ROC_s (%)")
    plt.plot(df.index, df["accel"] * 100.0, label="Accel (% points)")
    plt.title("Smoothed ROC and Acceleration")
    plt.legend()
    plt.tight_layout()

    # 3) Relative and Rel ROC_s
    plt.figure()
    plt.plot(df.index, df["rel"], label="Relative price (ticker/benchmark)")
    plt.title("Relative Price")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(df.index, df["rel_roc_s"] * 100.0, label="Rel ROC_s (%)")
    plt.title("Smoothed Relative ROC")
    plt.legend()
    plt.tight_layout()

    # 4) Composite score + regime markers (textual via colorless markers)
    plt.figure()
    plt.plot(df.index, df["score"], label="Momentum score")
    # Mark regime transitions as vertical lines (no explicit colors set)
    transitions = regime != regime.shift(1)
    for t in df.index[transitions.fillna(False)]:
        plt.axvline(t, linewidth=0.5)
    plt.title("Momentum Score (vertical lines = regime changes)")
    plt.legend()
    plt.tight_layout()

    plt.show()


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)

    p.add_argument("--roc", dest="roc_period", type=int, default=20, help="ROC lookback in trading days")
    p.add_argument("--ewm", dest="ewm_span", type=int, default=10, help="EWM span for smoothing ROC series")
    p.add_argument("--zwin", type=int, default=60, help="Rolling window for z-scores")
    p.add_argument("--trend-win", type=int, default=20, help="Rolling window for score slope")
    p.add_argument("--flat-win", type=int, default=30, help="Window used to define 'flattening' and near-high checks")
    p.add_argument("--ma-fast", type=int, default=20)
    p.add_argument("--ma-slow", type=int, default=50)

    p.add_argument("--w-roc", type=float, default=0.45)
    p.add_argument("--w-accel", type=float, default=0.25)
    p.add_argument("--w-rel", type=float, default=0.30)

    p.add_argument("--flat-roc-thresh", type=float, default=0.01,
                   help="Abs(smoothed ROC) threshold for flattening (e.g., 0.01 = 1%%)")
    p.add_argument("--near-high-pct", type=float, default=0.05,
                   help="Near-high threshold (e.g., 0.05 => within 5%% of rolling max)")
    p.add_argument("--rel-lag-thresh", type=float, default=0.002,
                   help="Relative lag threshold for rel_roc_s (e.g., 0.002 = 0.2%%)")

    p.add_argument("--log-slope-win", type=int, default=30,
                   help="Window for log-price regression slope (default: 30 days)")
    p.add_argument("--r2-thresh", type=float, default=0.7,
                   help="R² threshold for 'clean trend' classification (default: 0.7)")

    p.add_argument("--plot", action="store_true")

    p.add_argument("--csv-ticker", default=None, help="Optional CSV for ticker prices (Date + Adj Close/Close)")
    p.add_argument("--csv-benchmark", default=None, help="Optional CSV for benchmark prices (Date + Adj Close/Close)")

    a = p.parse_args()

    return Config(
        ticker=a.ticker.upper(),
        benchmark=a.benchmark.upper(),
        start=a.start,
        end=a.end,
        roc_period=a.roc_period,
        ewm_span=a.ewm_span,
        zwin=a.zwin,
        trend_win=a.trend_win,
        flat_win=a.flat_win,
        ma_fast=a.ma_fast,
        ma_slow=a.ma_slow,
        w_roc=a.w_roc,
        w_accel=a.w_accel,
        w_rel=a.w_rel,
        flat_roc_thresh=a.flat_roc_thresh,
        near_high_pct=a.near_high_pct,
        rel_lag_thresh=a.rel_lag_thresh,
        log_slope_win=a.log_slope_win,
        r2_thresh=a.r2_thresh,
        plot=a.plot,
        csv_ticker=a.csv_ticker,
        csv_benchmark=a.csv_benchmark,
    )


def main() -> int:
    cfg = parse_args()

    # Load data
    if cfg.csv_ticker and cfg.csv_benchmark:
        px = fetch_from_csv(cfg.csv_ticker, cfg.ticker)
        bx = fetch_from_csv(cfg.csv_benchmark, cfg.benchmark)
    else:
        px = fetch_adj_close_yf(cfg.ticker, cfg.start, cfg.end)
        bx = fetch_adj_close_yf(cfg.benchmark, cfg.start, cfg.end)

    px, bx = align_series(px, bx)
    df = compute_features(px, bx, cfg)
    regime = classify_regime(df, cfg)

    # Output last signal + optionally dump recent rows
    print(summarize_latest(df, regime, cfg))

    # List caution signals
    caution_dates = regime[regime == "caution"].index
    if len(caution_dates) > 0:
        print(f"\n=== CAUTION SIGNALS ({len(caution_dates)} occurrences) ===")
        caution_df = df.loc[caution_dates, ["price", "roc_s", "log_r2", "rel_roc_s"]].copy()
        caution_df["roc_s"] = caution_df["roc_s"] * 100
        caution_df["rel_roc_s"] = caution_df["rel_roc_s"] * 100
        caution_df.columns = ["price", "roc_s(%)", "R²", "rel_roc_s(%)"]
        with pd.option_context("display.width", 140, "display.max_columns", 20, "display.float_format", "{:.2f}".format):
            print(caution_df.tail(20))  # Show last 20 caution signals
        if len(caution_dates) > 20:
            print(f"  ... ({len(caution_dates) - 20} earlier signals not shown)")
    else:
        print("\n=== CAUTION SIGNALS: None ===")

    print("\nRecent regimes (last 50 rows):")
    tail = pd.concat([df[["price", "roc_s", "accel", "rel_roc_s", "score"]], regime.rename("regime")], axis=1).tail(50)
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(tail)

    if cfg.plot:
        plot_df(df, regime, cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
