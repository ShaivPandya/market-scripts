#!/usr/bin/env python3
"""
Total-portfolio beta-neutral + full-portfolio volatility targeting (USD base),
with currency conversion for non-USD instruments.

What this script does:
1) Loads portfolio metadata from equities/universes/portfolio.csv.
2) Downloads daily price history for all tickers from yfinance.
3) Downloads required FX rates from yfinance (e.g., EURUSD=X for EUR-denominated stocks).
4) Converts each instrument's price series into USD (if not already USD).
5) Computes USD daily returns, covariance (Sigma), and SPX betas for ALL instruments via regression on SPY.
6) Solves a convex program:
   - beta-neutral (total portfolio vs SPY)
   - vol cap (entire portfolio)
   - gross leverage, asset-class gross caps, equity net bounds
7) Scales the solution toward a target vol (within constraints).

Data sources:
- Portfolio metadata: equities/universes/portfolio.csv
- Price data: yfinance (live download)
- FX rates: yfinance (e.g., EURUSD=X, USDJPY=X)

FX quote conventions:
- yfinance FX tickers use "=X" suffix (e.g., "EURUSD=X" for USD per EUR)
- "EURUSD=X" means USD per 1 EUR. Then USD_price = local_price * EURUSD.
- "USDJPY=X" means JPY per 1 USD. Then USD_price = local_price / USDJPY.

python3 portfolio_optimizer.py
python3 portfolio_optimizer.py --book 100000

python3 equities/portfolio/portfolio_optimizer.py
python3 equities/portfolio/portfolio_optimizer.py --book 100000
"""

import argparse
import numpy as np
import pandas as pd
import cvxpy as cp
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import yfinance as yf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from composite_signal import generate_composite_signals, DEFAULT_WEIGHTS_SHORT

console = Console()

# -----------------------------
# Configuration
# -----------------------------
PORTFOLIO_CSV = Path(__file__).parent.parent / "universes" / "portfolio.csv"
LOOKBACK_DAYS = 365  # days of price history to fetch from yfinance

BASE_CCY = "USD"
MARKET_TICKER_LONG = "SPY"            # SPY used for beta regression on long positions
MARKET_TICKER_SHORT = "IWM"           # Russell 2000 ETF for beta regression on short positions
VOL_MIN = 0.0120                      # daily
VOL_TARGET = 0.0150                   # daily
VOL_MAX = 0.0200                      # daily

# Constraints
GROSS_MAX = 4.0
EQ_NET_MIN, EQ_NET_MAX = -0.50, 1.00
FX_GROSS_MAX = 2.0
CMDTY_GROSS_MAX = 1.0
BOND_10YR_EQUIV_MAX = 3.0  # 300% in 10-year equivalent
# Beta hedging is done post-optimization via explicit SPY/IWM hedge positions
MIN_ABS_WEIGHT = 0.01  # enforce minimum absolute weight for active longs/shorts
LONG_MAX = 0.20        # max 25% for any single long position
SHORT_MIN = -0.10      # max 25% (abs) for any single short position

# Duration (in years) for bond/Treasury futures instruments
DURATION_OF_TICKER: Dict[str, float] = {
    "ZN": 6.5,   # 10-year Treasury note futures
    "ZT": 2.0,   # 2-year Treasury note futures
    "ZF": 5.0,   # 5-year Treasury note futures
    "ZB": 17.0,  # 30-year Treasury bond futures
}

# Objective tuning
GAMMA_RISK = 1e-4     # small; increases preference for lower risk while staying close to w_raw
VOL_POWER_LONG = 0.7  # power for inverse-vol weighting for longs: 1/σ^p (p < 1 reduces concentration in low-vol names)
VOL_POWER_SHORT = 1.4 # power for inverse-vol weighting for shorts: 1/σ^p


# -----------------------------
# Currency metadata for non-USD instruments
# -----------------------------
# For each non-USD instrument, specify the currency of its local price series.
CURRENCY_OF_TICKER: Dict[str, str] = {
    "METSO.HE": "EUR",  # Helsinki-listed, priced in EUR
}

# If you have FX instruments as tradables (asset == "fx"), specify their base/quote.
# Only needed if you include FX tickers as instruments you trade.
# Example:
#   "USDJPY": ("USD", "JPY")   # USDJPY quoted as JPY per USD
FX_PAIR_INFO: Dict[str, Tuple[str, str]] = {
    # Example:
    # "USDJPY": ("USD", "JPY"),
}

# -----------------------------
# Helpers
# -----------------------------
def get_required_fx_tickers(tickers: list) -> list:
    """
    Determine which FX tickers to download based on CURRENCY_OF_TICKER.
    Returns yfinance FX ticker symbols (e.g., EURUSD=X).
    """
    currencies_needed = set()
    for t in tickers:
        ccy = CURRENCY_OF_TICKER.get(t)
        if ccy and ccy != BASE_CCY:
            currencies_needed.add(ccy)

    fx_tickers = []
    for ccy in currencies_needed:
        # yfinance uses CCYUSD=X format (e.g., EURUSD=X for USD per EUR)
        fx_tickers.append(f"{ccy}{BASE_CCY}=X")
    return fx_tickers


def download_prices(tickers: list, fx_tickers: list) -> pd.DataFrame:
    """
    Download price data from yfinance for tickers and FX rates.
    """
    all_tickers = tickers + fx_tickers
    end = date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)

    px = yf.download(
        tickers=all_tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False
    )

    # yfinance returns either a single DataFrame or a column MultiIndex
    if isinstance(px.columns, pd.MultiIndex):
        prices = px["Close"].copy()
    else:
        prices = px.copy()
        prices.columns = [all_tickers[0]]

    return prices.dropna(how="all")


def fx_series_for_ccy(prices: pd.DataFrame, ccy: str) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    Returns (fx, mode) to convert local CCY prices to USD:
      mode == "CCYUSD": fx is USD per CCY (e.g., EURUSD=X). USD_price = local * fx
      mode == "USDCCY": fx is CCY per USD (e.g., USDEUR=X). USD_price = local / fx
    """
    if ccy == BASE_CCY:
        return None, None

    # yfinance uses =X suffix for FX pairs
    ccyusd = f"{ccy}{BASE_CCY}=X"  # e.g., EURUSD=X (USD per EUR)
    usdccy = f"{BASE_CCY}{ccy}=X"  # e.g., USDEUR=X (EUR per USD)

    if ccyusd in prices.columns:
        return prices[ccyusd], "CCYUSD"
    if usdccy in prices.columns:
        return prices[usdccy], "USDCCY"

    return None, None


def to_usd_price(local_price: pd.Series, ccy: str, prices_all: pd.DataFrame) -> pd.Series:
    """
    Convert a local currency price series into USD price series using available FX columns.
    """
    if ccy == BASE_CCY:
        return local_price

    fx, mode = fx_series_for_ccy(prices_all, ccy)
    if fx is None:
        raise ValueError(
            f"Missing FX rate to convert {ccy} to USD. "
            f"Add {ccy} to CURRENCY_OF_TICKER for the relevant ticker."
        )

    if mode == "CCYUSD":
        return local_price * fx
    if mode == "USDCCY":
        return local_price / fx
    raise RuntimeError("Unexpected FX mode")


def ensure_psd(Sigma: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Eigenvalue clipping to make Sigma numerically PSD.
    """
    S = 0.5 * (Sigma + Sigma.T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(vals) @ vecs.T


def fetch_yfinance_betas(tickers: list) -> pd.Series:
    """
    Fetch beta values from yfinance Ticker.info.
    Returns NaN for tickers where beta is unavailable.
    """
    betas = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            beta = info.get('beta')
            betas[t] = beta if beta is not None else np.nan
        except Exception:
            betas[t] = np.nan
    return pd.Series(betas)


def compute_betas(rets: pd.DataFrame, market_col: str) -> pd.Series:
    """
    Regression slope beta_i = Cov(r_i, r_m) / Var(r_m), for all columns.
    Fallback for when yfinance beta is unavailable.
    """
    rm = rets[market_col].dropna()
    var_m = rm.var(ddof=1)
    if var_m <= 0:
        raise ValueError("Market return variance is non-positive; check market data.")

    betas = {}
    for c in rets.columns:
        ri = rets[c].dropna()
        aligned = pd.concat([ri, rm], axis=1, join="inner").dropna()
        if aligned.shape[0] < 20:
            betas[c] = 0.0
            continue
        cov = np.cov(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values, ddof=1)[0, 1]
        betas[c] = cov / var_m
    return pd.Series(betas)


def compute_defense_volatility(prices: pd.DataFrame, tickers: list) -> pd.Series:
    """
    Compute defense volatility from log returns using an EWMA blend and a floor.
    Returns daily volatility for each ticker.

    Method:
    - Uses log returns (not simple returns)
    - Short/long EWMA variance blend
    - Floor from rolling median of long-run EWMA volatility
    """
    # Log returns
    log_rets = np.log(prices[tickers] / prices[tickers].shift(1))

    short_hl = 20
    long_hl = 120
    blend_w = 0.70
    floor_window = 252
    floor_min = 60

    short_var = (log_rets ** 2).ewm(halflife=short_hl, adjust=False).mean()
    long_var = (log_rets ** 2).ewm(halflife=long_hl, adjust=False).mean()
    blend_var = blend_w * short_var + (1.0 - blend_w) * long_var
    blend_vol = np.sqrt(blend_var)

    long_vol = np.sqrt(long_var)
    floor_vol = long_vol.rolling(floor_window, min_periods=floor_min).median()
    floor_vol = floor_vol.fillna(long_vol)

    # Defense vol = max(blended EWMA vol, long-run floor) - latest value for each ticker
    defense_vol = {}
    for t in tickers:
        series = np.maximum(blend_vol[t], floor_vol[t])
        defense_vol[t] = series.dropna().iloc[-1] if series.notna().any() else np.nan

    return pd.Series(defense_vol)


def build_raw_weights(
    meta: pd.DataFrame,
    signals: Optional[pd.Series] = None,
    G_L: float = 1.0,
    G_S: float = 1.0,
    signal_scale_equity_long: float = 1.5,
    signal_scale_equity_short: float = 1.0,
    signal_scale_other: float = 0.9,
    vol_power_long: float = 0.7,
    vol_power_short: float = 1.4,
) -> pd.Series:
    """
    Inverse-vol raw weights, optionally tilted by momentum signals.

    Args:
        meta: Portfolio metadata with 'direction', 'asset', and 'realized_vol' columns
        signals: Optional z-scored momentum signals per ticker (higher = more conviction)
        G_L: Long gross target (default: 1.0)
        G_S: Short gross target (default: 1.0)
        signal_scale_equity_long: Scaling factor for signal tilt on equity longs (default: 1.5)
        signal_scale_equity_short: Scaling factor for signal tilt on equity shorts (default: 1.0)
        signal_scale_other: Scaling factor for signal tilt on non-equities (default: 0.9)
        vol_power_long: Power for inverse-vol weighting 1/σ^p for longs (default: 0.7; use <1 to reduce low-vol concentration)
        vol_power_short: Power for inverse-vol weighting 1/σ^p for shorts (default: 1.4)

    Signal interpretation (signals are direction-agnostic: higher = stronger/better stock):
        - Positive signal on LONG = increase weight (strong stock, go longer)
        - Negative signal on SHORT = increase short conviction (weak stock, short more)
        - Signal multiplier: exp(signal_scale * signal), with signal inverted for shorts
        - Different signal scales used for equities vs other assets
    """
    w_raw = pd.Series(0.0, index=meta.index)
    longs = meta[meta["direction"].str.lower().eq("long")]
    shorts = meta[meta["direction"].str.lower().eq("short")]

    if len(longs) > 0:
        invv = 1.0 / (longs["realized_vol"].replace(0, np.nan) ** vol_power_long)
        invv = invv.fillna(0.0)
        if invv.sum() > 0:
            base_w = invv / invv.sum()

            # Apply signal tilt if provided (use different scales for equities vs other assets)
            if signals is not None:
                sig = signals.reindex(longs.index).fillna(0.0)
                # Determine signal scale based on asset type
                is_equity = longs["asset"].str.lower().eq("equity")
                signal_scale = pd.Series(
                    np.where(is_equity, signal_scale_equity_long, signal_scale_other),
                    index=longs.index
                )
                signal_mult = np.exp(signal_scale * sig)
                base_w = base_w * signal_mult
                base_w = base_w / base_w.sum()  # Re-normalize

            w_raw.loc[longs.index] = G_L * base_w

    if len(shorts) > 0:
        invv = 1.0 / (shorts["realized_vol"].replace(0, np.nan) ** vol_power_short)
        invv = invv.fillna(0.0)
        if invv.sum() > 0:
            base_w = invv / invv.sum()

            # Apply signal tilt if provided (use different scales for equities vs other assets)
            if signals is not None:
                # Invert signal for shorts: negative signal (weak stock) -> more short conviction
                sig = -signals.reindex(shorts.index).fillna(0.0)
                # Determine signal scale based on asset type
                is_equity = shorts["asset"].str.lower().eq("equity")
                signal_scale = pd.Series(
                    np.where(is_equity, signal_scale_equity_short, signal_scale_other),
                    index=shorts.index
                )
                signal_mult = np.exp(signal_scale * sig)
                base_w = base_w * signal_mult
                base_w = base_w / base_w.sum()  # Re-normalize

            w_raw.loc[shorts.index] = -G_S * base_w

    return w_raw


def exposures_by_class(w: pd.Series, meta: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for cls in ["equity", "fx", "commodity", "bond"]:
        mask = meta["asset"].str.lower().eq(cls)
        if mask.any():
            out[f"{cls}_gross"] = float(np.abs(w[mask]).sum())
            out[f"{cls}_net"] = float(w[mask].sum())
    out["total_gross"] = float(np.abs(w).sum())
    out["total_net"] = float(w.sum())
    return out


def compute_10yr_equivalent(w: pd.Series, meta: pd.DataFrame) -> float:
    """Compute total 10-year equivalent exposure for bond positions."""
    bond_mask = meta["asset"].str.lower().eq("bond")
    if not bond_mask.any():
        return 0.0

    total_10yr_equiv = 0.0
    for ticker in w[bond_mask].index:
        duration = DURATION_OF_TICKER.get(ticker, 10.0)  # default to 10 if unknown
        total_10yr_equiv += abs(w[ticker]) * (duration / 10.0)
    return total_10yr_equiv


def identify_binding_constraint(w: pd.Series, meta: pd.DataFrame) -> str:
    """Identify which constraint limits further scaling."""
    checks = []

    # Total gross
    checks.append(("Total gross (400%)", abs(w).sum(), GROSS_MAX))

    # Equity net
    eq_mask = meta["asset"].str.lower().eq("equity")
    eq_net = w[eq_mask].sum() if eq_mask.any() else 0.0
    if eq_net >= 0:
        checks.append(("Equity net long (100%)", eq_net, EQ_NET_MAX))
    else:
        checks.append(("Equity net short (-50%)", -eq_net, -EQ_NET_MIN))

    # Asset class caps
    for name, mask_col, cap in [
        ("FX gross (200%)", "fx", FX_GROSS_MAX),
        ("Commodity gross (50%)", "commodity", CMDTY_GROSS_MAX),
    ]:
        mask = meta["asset"].str.lower().eq(mask_col)
        if mask.any():
            checks.append((name, abs(w[mask]).sum(), cap))

    # Bond 10yr equivalent
    bond_10yr = compute_10yr_equivalent(w, meta)
    if bond_10yr > 0:
        checks.append(("Bond 10yr equiv (300%)", bond_10yr, BOND_10YR_EQUIV_MAX))

    # Find binding (closest to limit)
    binding = max(checks, key=lambda x: x[1] / x[2] if x[2] > 0 else 0)
    return f"{binding[0]}: {binding[1]:.2%} of {binding[2]:.0%} limit"


def max_scale_to_respect_linear_caps(w: pd.Series, meta: pd.DataFrame) -> float:
    """
    Scaling w by k preserves beta neutrality and correlations.
    Returns the max k such that linear caps remain satisfied (gross/net by class).
    """
    eps = 1e-12
    k_list = []

    total_gross = np.abs(w).sum()
    k_list.append(GROSS_MAX / max(total_gross, eps))

    # Asset-class gross caps
    def add_gross_cap(asset_name: str, cap: float):
        mask = meta["asset"].str.lower().eq(asset_name)
        if mask.any():
            g = np.abs(w[mask]).sum()
            k_list.append(cap / max(g, eps))

    add_gross_cap("fx", FX_GROSS_MAX)
    add_gross_cap("commodity", CMDTY_GROSS_MAX)

    # Bond 10-year equivalent cap
    current_10yr = compute_10yr_equivalent(w, meta)
    if current_10yr > eps:
        k_list.append(BOND_10YR_EQUIV_MAX / current_10yr)

    # Equity net bounds
    eq_mask = meta["asset"].str.lower().eq("equity")
    eq_net = float(w[eq_mask].sum()) if eq_mask.any() else 0.0
    if abs(eq_net) < eps:
        k_eq = np.inf
    elif eq_net > 0:
        k_eq = EQ_NET_MAX / eq_net
    else:
        # eq_net negative; EQ_NET_MIN is negative
        k_eq = EQ_NET_MIN / eq_net
    k_list.append(k_eq)

    return float(min(k_list))


# -----------------------------
# Main
# -----------------------------
def main(book: Optional[float] = None, debug_weights: bool = False):
    meta = pd.read_csv(PORTFOLIO_CSV)
    meta["direction"] = meta["direction"].fillna("")
    # realized_vol will be computed from price data, not loaded from CSV
    meta = meta.set_index("ticker")

    tickers = meta.index.tolist()

    # Determine required FX tickers and download all prices from yfinance
    fx_tickers = get_required_fx_tickers(tickers)
    # Include both market tickers for beta regression
    market_tickers = [MARKET_TICKER_LONG, MARKET_TICKER_SHORT]
    all_tickers_to_fetch = list(set(tickers + market_tickers))
    console.print(f"[cyan]Downloading prices for {len(all_tickers_to_fetch)} tickers + {len(fx_tickers)} FX rates...[/cyan]")
    prices_all = download_prices(all_tickers_to_fetch, fx_tickers)

    missing_cols = [t for t in tickers if t not in prices_all.columns]
    if missing_cols:
        raise ValueError(f"yfinance failed to download tickers: {missing_cols}")

    for mt in market_tickers:
        if mt not in prices_all.columns:
            raise ValueError(f"yfinance failed to download {mt} for beta regression.")

    # Convert all instrument prices to USD prices (include market tickers for beta calc)
    usd_prices = pd.DataFrame(index=prices_all.index)
    tickers_plus_market = list(set(tickers + market_tickers))
    for t in tickers_plus_market:
        local_px = prices_all[t]
        ccy = CURRENCY_OF_TICKER.get(t, BASE_CCY)  # default USD if not specified
        usd_prices[t] = to_usd_price(local_px, ccy, prices_all)

    # Compute USD returns
    # Forward-fill prices to handle misaligned trading calendars, then compute returns
    usd_prices = usd_prices.ffill()
    rets = usd_prices.pct_change(fill_method=None).dropna(how="all")
    # Ensure consistent ordering (keep only portfolio tickers, but rets still has MARKET_TICKER for beta)
    tickers = [t for t in tickers if t in rets.columns]
    meta = meta.loc[tickers]

    # Compute defense volatility (max of 20d, 60d rolling vol) from USD prices
    console.print("[cyan]Computing defense volatility (EWMA blend + floor)...[/cyan]")
    defense_vol = compute_defense_volatility(usd_prices, tickers)
    meta["realized_vol"] = defense_vol

    if len(tickers) < 2:
        raise ValueError("Need at least 2 instruments with returns to optimize.")

    # Portfolio-only returns for covariance (exclude market ticker)
    rets_portfolio = rets[tickers]

    # Covariance (daily)
    Sigma = rets_portfolio.cov().values
    Sigma = ensure_psd(Sigma, eps=1e-10)

    # Cholesky for SOC vol constraint
    L = np.linalg.cholesky(Sigma)

    # Compute betas for ALL instruments vs both benchmarks
    # SPY betas for long positions, IWM betas for short positions
    console.print("[cyan]Computing betas vs SPY (longs) and IWM (shorts)...[/cyan]")

    # SPY betas: try yfinance first, then compute missing ones manually
    yf_betas_spy = fetch_yfinance_betas(tickers)
    computed_betas_spy = compute_betas(rets, MARKET_TICKER_LONG).reindex(tickers)
    betas_spy = yf_betas_spy.combine_first(computed_betas_spy).fillna(0.0)

    # IWM betas: compute manually (yfinance doesn't provide beta vs Russell 2000)
    betas_iwm = compute_betas(rets, MARKET_TICKER_SHORT).reindex(tickers).fillna(0.0)

    # Generate composite signals for active tickers (those with direction)
    active_tickers = [t for t in tickers if meta.loc[t, "direction"].strip()]
    console.print(f"[cyan]Generating composite signals for {len(active_tickers)} active tickers...[/cyan]")
    asset_map = dict(zip(meta.index, meta["asset"]))
    direction_map = {t: meta.loc[t, "direction"].strip().lower() for t in active_tickers}
    signals_df, _ = generate_composite_signals(
        tickers=active_tickers,
        asset_map=asset_map,
        benchmark_override=MARKET_TICKER_LONG,
        direction_map=direction_map,
        weights_short=DEFAULT_WEIGHTS_SHORT,
    )
    # Extract composite signal for weighting
    signals = signals_df["composite_signal"] if not signals_df.empty else pd.Series(0.0, index=active_tickers)

    # Raw weights shape (inverse-vol by long/short buckets, tilted by signals)
    w_raw = build_raw_weights(meta, signals=signals, G_L=1.0, G_S=1.0, vol_power_long=VOL_POWER_LONG, vol_power_short=VOL_POWER_SHORT).reindex(tickers).fillna(0.0)
    w_raw_vec = w_raw.values

    # Masks
    asset = meta["asset"].str.lower()
    eq_mask = asset.eq("equity").values
    fx_mask = asset.eq("fx").values
    cmdty_mask = asset.eq("commodity").values
    bond_mask = asset.eq("bond").values

    n = len(tickers)
    w = cp.Variable(n)

    constraints = []

    # Enforce direction: longs >= MIN_ABS_WEIGHT, shorts <= -MIN_ABS_WEIGHT
    direction = meta["direction"].str.lower()
    long_mask = direction.eq("long").values
    short_mask = direction.eq("short").values
    if long_mask.any():
        constraints.append(w[long_mask] >= MIN_ABS_WEIGHT)
        constraints.append(w[long_mask] <= LONG_MAX)  # cap longs at 20%
    if short_mask.any():
        constraints.append(w[short_mask] <= -MIN_ABS_WEIGHT)
        constraints.append(w[short_mask] >= SHORT_MIN)  # floor shorts at -10%

    # Total gross leverage
    constraints.append(cp.norm1(w) <= GROSS_MAX)

    # Equity net bounds
    if eq_mask.any():
        w_eq = w[eq_mask]
        constraints.append(cp.sum(w_eq) >= EQ_NET_MIN)
        constraints.append(cp.sum(w_eq) <= EQ_NET_MAX)

    # Asset-class gross caps
    if fx_mask.any():
        constraints.append(cp.norm1(w[fx_mask]) <= FX_GROSS_MAX)
    if cmdty_mask.any():
        constraints.append(cp.norm1(w[cmdty_mask]) <= CMDTY_GROSS_MAX)
    if bond_mask.any():
        # Bond 10-year equivalent constraint: sum(|w_i| * duration_i / 10) <= limit
        bond_tickers = [tickers[i] for i in range(n) if bond_mask[i]]
        duration_coeffs = np.array([DURATION_OF_TICKER.get(t, 10.0) / 10.0 for t in bond_tickers])
        constraints.append(cp.sum(cp.multiply(duration_coeffs, cp.abs(w[bond_mask]))) <= BOND_10YR_EQUIV_MAX)

    # Note: Beta hedging is done post-optimization via explicit SPY/IWM positions
    # (not via constraints on the portfolio weights)

    # Total-portfolio vol cap (SOC form): ||L w||_2 <= VOL_MAX
    constraints.append(cp.norm(L @ w, 2) <= VOL_MAX)

    # Objective: stay close to raw + mild risk regularization
    objective = cp.Minimize(cp.sum_squares(w - w_raw_vec) + GAMMA_RISK * cp.sum_squares(L @ w))

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)  # let cvxpy choose the best available solver

    if w.value is None:
        raise RuntimeError("Optimization failed. Try solver=SCS, or check data/constraints for feasibility.")

    w_star = pd.Series(w.value, index=tickers)

    # Post-solve scaling toward VOL_TARGET, respecting linear caps
    def port_vol(w_vec: np.ndarray) -> float:
        x = L @ w_vec
        return float(np.sqrt(np.maximum(0.0, x.T @ x)))

    vol0 = port_vol(w_star.values)
    if vol0 <= 0:
        raise RuntimeError("Optimized portfolio has ~0 volatility; check inputs.")

    # Scaling cannot exceed vol cap, linear caps, or equity net bounds.
    # Also avoid scaling down below the minimum absolute weight for active positions.
    k_target = VOL_TARGET / vol0
    k_volcap = VOL_MAX / vol0
    k_linear = max_scale_to_respect_linear_caps(w_star, meta)
    k = min(k_target, k_volcap, k_linear)

    active_mask = long_mask | short_mask
    if MIN_ABS_WEIGHT > 0 and active_mask.any():
        min_abs_active = float(np.min(np.abs(w_star.values[active_mask])))
        if min_abs_active > 0:
            k_floor = MIN_ABS_WEIGHT / min_abs_active
            if k < k_floor:
                k = k_floor

    w_final = w_star * k
    vol_final = port_vol(w_final.values)

    # Compute volatility for benchmark ETFs (SPY and IWM) for informational purposes
    benchmark_vol = compute_defense_volatility(usd_prices, market_tickers)
    vol_spy = benchmark_vol.get(MARKET_TICKER_LONG, np.nan)
    vol_iwm = benchmark_vol.get(MARKET_TICKER_SHORT, np.nan)

    # Compute separate betas for longs and shorts
    beta_long_spy = float(betas_spy.values[long_mask] @ w_final.values[long_mask]) if long_mask.any() else 0.0
    beta_short_iwm = float(betas_iwm.values[short_mask] @ w_final.values[short_mask]) if short_mask.any() else 0.0

    # Calculate hedge positions:
    # - Short SPY to hedge long positions' beta exposure to SPY
    # - Long IWM to hedge short positions' beta exposure to IWM (short beta is negative, so -beta gives positive IWM weight)
    hedge_spy_weight = -beta_long_spy   # Short SPY to offset long beta
    hedge_iwm_weight = -beta_short_iwm  # Long IWM to offset short beta (shorts have negative beta contribution)

    if debug_weights:
        console.print()
        dbg = pd.DataFrame({
            "asset": meta["asset"],
            "direction": meta["direction"],
            "realized_vol": meta["realized_vol"],
            "w_raw": w_raw,
            "w_star": w_star,
            "w_final": w_final,
        }).sort_values("w_final", ascending=False)

        dbg_table = Table(title="[bold]Weight Diagnostics[/bold]", box=box.ROUNDED, show_header=True, header_style="bold yellow")
        dbg_table.add_column("Ticker", style="bold white")
        dbg_table.add_column("Asset", style="white")
        dbg_table.add_column("Direction", style="white")
        dbg_table.add_column("Vol", justify="right", style="white")
        dbg_table.add_column("w_raw", justify="right")
        dbg_table.add_column("w_star", justify="right")
        dbg_table.add_column("w_final", justify="right")

        for ticker, row in dbg.iterrows():
            dbg_table.add_row(
                str(ticker),
                row["asset"],
                row["direction"],
                f"{row['realized_vol']:.4f}" if pd.notna(row["realized_vol"]) else "nan",
                f"{row['w_raw']:+.6f}",
                f"{row['w_star']:+.6f}",
                f"{row['w_final']:+.6f}",
            )
        console.print(dbg_table)

    # If vol_final < VOL_MIN, the band may be infeasible under constraints with this universe/shape.
    feasible_band = vol_final >= VOL_MIN - 1e-6

    # Report
    console.print()
    status_color = "green" if prob.status == "optimal" else "yellow"
    feasible_text = "[green]YES[/green]" if feasible_band else "[red]NO[/red] (hit constraints before reaching VOL_MIN)"

    solution_text = (
        f"[bold]Status:[/bold]        [{status_color}]{prob.status}[/{status_color}]\n"
        f"[bold]Vol (daily):[/bold]   {vol_final:.6f}  [dim](target {VOL_TARGET:.6f}, band [{VOL_MIN:.6f}, {VOL_MAX:.6f}])[/dim]\n"
        f"[bold]SPY vol:[/bold]       {vol_spy:.6f}  [dim](for reference)[/dim]\n"
        f"[bold]IWM vol:[/bold]       {vol_iwm:.6f}  [dim](for reference)[/dim]\n"
        f"[bold]Long β to SPY:[/bold] {beta_long_spy:+.4f}  [dim]→ Hedge: {hedge_spy_weight:+.4f} {MARKET_TICKER_LONG}[/dim]\n"
        f"[bold]Short β to IWM:[/bold] {beta_short_iwm:+.4f}  [dim]→ Hedge: {hedge_iwm_weight:+.4f} {MARKET_TICKER_SHORT}[/dim]\n"
        f"[bold]Band feasible:[/bold] {feasible_text}"
    )
    console.print(Panel(solution_text, title="[bold blue]Solution[/bold blue]", border_style="blue"))

    exp = exposures_by_class(w_final, meta)
    exp_table = Table(title="[bold]Exposures[/bold]", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    exp_table.add_column("Type", style="white")
    exp_table.add_column("Value", justify="right", style="white")
    for k0 in sorted(exp.keys()):
        val = exp[k0]
        val_str = f"{val:+.4f}" if "net" in k0 else f"{val:.4f}"
        exp_table.add_row(k0, val_str)
    console.print(exp_table)

    console.print()
    out = pd.DataFrame({
        "asset": meta["asset"],
        "direction": meta["direction"],
        "signal": signals.reindex(tickers).fillna(0.0),
        "beta_to_SPY": betas_spy,
        "beta_to_IWM": betas_iwm,
        "realized_volatility": meta["realized_vol"],
        "weight": w_final,
    })
    if book is not None:
        out["dollar_weight"] = w_final * book
    out = out.sort_values("weight", ascending=False)

    # Build rich table for weights
    weights_table = Table(title="[bold]Weights (% NAV notional)[/bold]", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    weights_table.add_column("Ticker", style="bold white")
    weights_table.add_column("Asset", style="white")
    weights_table.add_column("Direction", style="white")
    weights_table.add_column("Signal", justify="right", style="white")
    weights_table.add_column("β SPY", justify="right", style="white")
    weights_table.add_column("β IWM", justify="right", style="white")
    weights_table.add_column("Vol", justify="right", style="white")
    weights_table.add_column("Weight", justify="right")
    if book is not None:
        weights_table.add_column("Dollar", justify="right")

    for ticker, row in out.iterrows():
        weight_val = row["weight"]
        weight_color = "green" if weight_val > 0 else "red" if weight_val < 0 else "white"
        weight_str = f"[{weight_color}]{weight_val:+.4f}[/{weight_color}]"

        row_data = [
            str(ticker),
            row["asset"],
            row["direction"],
            f"{row['signal']:+.2f}",
            f"{row['beta_to_SPY']:.2f}",
            f"{row['beta_to_IWM']:.2f}",
            f"{row['realized_volatility']:.4f}",
            weight_str,
        ]
        if book is not None:
            dollar_val = row["dollar_weight"]
            dollar_color = "green" if dollar_val > 0 else "red" if dollar_val < 0 else "white"
            row_data.append(f"[{dollar_color}]{dollar_val:+,.0f}[/{dollar_color}]")
        weights_table.add_row(*row_data)

    # Add separator and hedge positions
    sep_cols = ["───"] * (8 if book is None else 9)
    weights_table.add_row(*sep_cols)

    # SPY hedge (short to hedge long beta)
    spy_color = "green" if hedge_spy_weight > 0 else "red" if hedge_spy_weight < 0 else "white"
    spy_row = [
        f"[bold]{MARKET_TICKER_LONG}[/bold]",
        "hedge",
        "short" if hedge_spy_weight < 0 else "long",
        "—",
        "1.00",
        "—",
        "—",
        f"[{spy_color}]{hedge_spy_weight:+.4f}[/{spy_color}]",
    ]
    if book is not None:
        spy_dollar = hedge_spy_weight * book
        spy_row.append(f"[{spy_color}]{spy_dollar:+,.0f}[/{spy_color}]")
    weights_table.add_row(*spy_row)

    # IWM hedge (long to hedge short beta)
    iwm_color = "green" if hedge_iwm_weight > 0 else "red" if hedge_iwm_weight < 0 else "white"
    iwm_row = [
        f"[bold]{MARKET_TICKER_SHORT}[/bold]",
        "hedge",
        "long" if hedge_iwm_weight > 0 else "short",
        "—",
        "—",
        "1.00",
        "—",
        f"[{iwm_color}]{hedge_iwm_weight:+.4f}[/{iwm_color}]",
    ]
    if book is not None:
        iwm_dollar = hedge_iwm_weight * book
        iwm_row.append(f"[{iwm_color}]{iwm_dollar:+,.0f}[/{iwm_color}]")
    weights_table.add_row(*iwm_row)

    console.print(weights_table)

    # out.to_csv("optimized_weights.csv")
    # print("\nWrote: optimized_weights.csv")

    # === Max Scaled Version ===
    k_max = max_scale_to_respect_linear_caps(w_final, meta)
    w_max_scaled = w_final * k_max
    vol_max_scaled = port_vol(w_max_scaled.values)

    binding = identify_binding_constraint(w_max_scaled, meta)

    console.print()
    max_scaled_text = (
        f"[bold]Scale factor:[/bold]      {k_max:.4f}x\n"
        f"[bold]Binding constraint:[/bold] [yellow]{binding}[/yellow]\n"
        f"[bold]Vol (daily):[/bold]        {vol_max_scaled:.6f}"
    )
    console.print(Panel(max_scaled_text, title="[bold magenta]Max Scaled Portfolio[/bold magenta]", border_style="magenta"))

    exp_max = exposures_by_class(w_max_scaled, meta)
    exp_max_table = Table(title="[bold]Max Scaled Exposures[/bold]", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    exp_max_table.add_column("Type", style="white")
    exp_max_table.add_column("Value", justify="right", style="white")
    for k0 in sorted(exp_max.keys()):
        val = exp_max[k0]
        val_str = f"{val:+.4f}" if "net" in k0 else f"{val:.4f}"
        exp_max_table.add_row(k0, val_str)

    # Add 10yr equivalent for bonds
    bond_10yr = compute_10yr_equivalent(w_max_scaled, meta)
    if bond_10yr > 0:
        exp_max_table.add_row("bond_10yr_equiv", f"{bond_10yr:.4f}")

    console.print(exp_max_table)

    console.print()
    out_max = pd.DataFrame({
        "asset": meta["asset"],
        "direction": meta["direction"],
        "weight": w_max_scaled,
    })
    if book is not None:
        out_max["dollar_weight"] = w_max_scaled * book
    out_max = out_max.sort_values("weight", ascending=False)

    # Build rich table for max scaled weights
    max_weights_table = Table(title="[bold]Max Scaled Weights (% NAV notional)[/bold]", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    max_weights_table.add_column("Ticker", style="bold white")
    max_weights_table.add_column("Asset", style="white")
    max_weights_table.add_column("Direction", style="white")
    max_weights_table.add_column("Weight", justify="right")
    if book is not None:
        max_weights_table.add_column("Dollar", justify="right")

    for ticker, row in out_max.iterrows():
        weight_val = row["weight"]
        weight_color = "green" if weight_val > 0 else "red" if weight_val < 0 else "white"
        weight_str = f"[{weight_color}]{weight_val:+.4f}[/{weight_color}]"

        row_data = [
            str(ticker),
            row["asset"],
            row["direction"],
            weight_str,
        ]
        if book is not None:
            dollar_val = row["dollar_weight"]
            dollar_color = "green" if dollar_val > 0 else "red" if dollar_val < 0 else "white"
            row_data.append(f"[{dollar_color}]{dollar_val:+,.0f}[/{dollar_color}]")
        max_weights_table.add_row(*row_data)

    console.print(max_weights_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio optimizer with beta-neutral and volatility targeting.")
    parser.add_argument("--book", type=float, default=None, help="Book size in dollars to compute dollar weights")
    parser.add_argument("--debug-weights", action="store_true", help="Print raw/optimized/final weights for diagnostics")
    args = parser.parse_args()
    main(book=args.book, debug_weights=args.debug_weights)
