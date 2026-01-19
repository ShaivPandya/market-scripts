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

from momentum_signal import generate_portfolio_signals

# -----------------------------
# Configuration
# -----------------------------
PORTFOLIO_CSV = Path(__file__).parent.parent / "universes" / "portfolio.csv"
LOOKBACK_DAYS = 365  # days of price history to fetch from yfinance

BASE_CCY = "USD"
MARKET_TICKER = "SPY"                 # SPY used for beta regression; must be in prices
VOL_MIN = 0.0030                      # daily
VOL_TARGET = 0.0035                   # daily
VOL_MAX = 0.0040                      # daily

# Constraints
GROSS_MAX = 4.0
EQ_NET_MIN, EQ_NET_MAX = -0.50, 1.00
FX_GROSS_MAX = 2.0
CMDTY_GROSS_MAX = 0.50
BOND_GROSS_MAX = 3.0

# Objective tuning
GAMMA_RISK = 1e-4     # small; increases preference for lower risk while staying close to w_raw


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


def build_raw_weights(
    meta: pd.DataFrame,
    signals: Optional[pd.Series] = None,
    G_L: float = 1.0,
    G_S: float = 1.0,
    signal_scale: float = 0.4,
) -> pd.Series:
    """
    Inverse-vol raw weights, optionally tilted by momentum signals.

    Args:
        meta: Portfolio metadata with 'direction' and 'realized_vol' columns
        signals: Optional z-scored momentum signals per ticker (higher = more conviction)
        G_L: Long gross target (default: 1.0)
        G_S: Short gross target (default: 1.0)
        signal_scale: Scaling factor for signal tilt (default: 0.4)

    Signal interpretation:
        - Positive signal on LONG = increase weight
        - Positive signal on SHORT = increase short conviction (more negative)
        - Signal multiplier: exp(signal_scale * signal)
    """
    w_raw = pd.Series(0.0, index=meta.index)
    longs = meta[meta["direction"].str.lower().eq("long")]
    shorts = meta[meta["direction"].str.lower().eq("short")]

    if len(longs) > 0:
        invv = 1.0 / longs["realized_vol"].replace(0, np.nan)
        invv = invv.fillna(0.0)
        if invv.sum() > 0:
            base_w = invv / invv.sum()

            # Apply signal tilt if provided
            if signals is not None:
                sig = signals.reindex(longs.index).fillna(0.0)
                signal_mult = np.exp(signal_scale * sig)
                base_w = base_w * signal_mult
                base_w = base_w / base_w.sum()  # Re-normalize

            w_raw.loc[longs.index] = G_L * base_w

    if len(shorts) > 0:
        invv = 1.0 / shorts["realized_vol"].replace(0, np.nan)
        invv = invv.fillna(0.0)
        if invv.sum() > 0:
            base_w = invv / invv.sum()

            # Apply signal tilt if provided (higher signal = more short conviction)
            if signals is not None:
                sig = signals.reindex(shorts.index).fillna(0.0)
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
    add_gross_cap("bond", BOND_GROSS_MAX)

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
def main(book: Optional[float] = None):
    meta = pd.read_csv(PORTFOLIO_CSV)
    meta["direction"] = meta["direction"].fillna("")
    meta["realized_vol"] = pd.to_numeric(meta["realized_vol"], errors="coerce")
    meta = meta.set_index("ticker")

    tickers = meta.index.tolist()

    # Determine required FX tickers and download all prices from yfinance
    fx_tickers = get_required_fx_tickers(tickers)
    all_tickers_to_fetch = tickers + [MARKET_TICKER] if MARKET_TICKER not in tickers else tickers
    print(f"Downloading prices for {len(all_tickers_to_fetch)} tickers + {len(fx_tickers)} FX rates...")
    prices_all = download_prices(all_tickers_to_fetch, fx_tickers)

    missing_cols = [t for t in tickers if t not in prices_all.columns]
    if missing_cols:
        raise ValueError(f"yfinance failed to download tickers: {missing_cols}")

    if MARKET_TICKER not in prices_all.columns:
        raise ValueError(f"yfinance failed to download {MARKET_TICKER} for beta regression.")

    # Convert all instrument prices to USD prices (include market ticker for beta calc)
    usd_prices = pd.DataFrame(index=prices_all.index)
    tickers_plus_market = tickers + [MARKET_TICKER] if MARKET_TICKER not in tickers else tickers
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

    if len(tickers) < 2:
        raise ValueError("Need at least 2 instruments with returns to optimize.")

    # Portfolio-only returns for covariance (exclude market ticker)
    rets_portfolio = rets[tickers]

    # Covariance (daily)
    Sigma = rets_portfolio.cov().values
    Sigma = ensure_psd(Sigma, eps=1e-10)

    # Cholesky for SOC vol constraint
    L = np.linalg.cholesky(Sigma)

    # Total-portfolio betas vs market (SPY) for ALL instruments
    # Try yfinance betas first, then compute missing ones manually
    print("Fetching betas from yfinance...")
    yf_betas = fetch_yfinance_betas(tickers)
    computed_betas = compute_betas(rets, MARKET_TICKER).reindex(tickers)
    # Use yfinance beta if available, otherwise use computed
    betas = yf_betas.combine_first(computed_betas).fillna(0.0)
    beta_vec = betas.values

    # Generate momentum signals for active tickers (those with direction)
    active_tickers = [t for t in tickers if meta.loc[t, "direction"].strip()]
    print(f"Generating momentum signals for {len(active_tickers)} active tickers...")
    signals = generate_portfolio_signals(active_tickers, benchmark=MARKET_TICKER)

    # Raw weights shape (inverse-vol by long/short buckets, tilted by signals)
    w_raw = build_raw_weights(meta, signals=signals, G_L=1.0, G_S=1.0).reindex(tickers).fillna(0.0)
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

    # Enforce direction: longs >= 0, shorts <= 0
    direction = meta["direction"].str.lower()
    long_mask = direction.eq("long").values
    short_mask = direction.eq("short").values
    if long_mask.any():
        constraints.append(w[long_mask] >= 0)
    if short_mask.any():
        constraints.append(w[short_mask] <= 0)

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
        constraints.append(cp.norm1(w[bond_mask]) <= BOND_GROSS_MAX)

    # Total-portfolio beta-neutral vs market
    constraints.append(beta_vec @ w == 0.0)

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

    # Scaling cannot exceed vol cap, linear caps, or equity net bounds
    k_target = VOL_TARGET / vol0
    k_volcap = VOL_MAX / vol0
    k_linear = max_scale_to_respect_linear_caps(w_star, meta)
    k = min(k_target, k_volcap, k_linear)

    w_final = w_star * k
    vol_final = port_vol(w_final.values)
    beta_final = float(betas.values @ w_final.values)

    # If vol_final < VOL_MIN, the band may be infeasible under constraints with this universe/shape.
    feasible_band = vol_final >= VOL_MIN - 1e-6

    # Report
    print("\n=== Solution ===")
    print(f"Status: {prob.status}")
    print(f"Vol (daily): {vol_final:.6f}  (target {VOL_TARGET:.6f}, band [{VOL_MIN:.6f}, {VOL_MAX:.6f}])")
    print(f"Total beta vs {MARKET_TICKER}: {beta_final:.6e}")
    print(f"Band feasible? {'YES' if feasible_band else 'NO (hit constraints before reaching VOL_MIN)'}\n")

    exp = exposures_by_class(w_final, meta)
    print("=== Exposures ===")
    for k0 in sorted(exp.keys()):
        print(f"{k0:14s}: {exp[k0]: .4f}")

    print("\n=== Weights (% NAV notional) ===")
    out = pd.DataFrame({
        "asset": meta["asset"],
        "direction": meta["direction"],
        "signal": signals.reindex(tickers).fillna(0.0),
        "beta_to_SPY": betas,
        "weight": w_final,
    })
    if book is not None:
        out["dollar_weight"] = w_final * book
    out = out.sort_values("weight", ascending=False)
    print(out.to_string(float_format=lambda x: f"{x: .6f}"))

    out.to_csv("optimized_weights.csv")
    print("\nWrote: optimized_weights.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio optimizer with beta-neutral and volatility targeting.")
    parser.add_argument("--book", type=float, default=None, help="Book size in dollars to compute dollar weights")
    args = parser.parse_args()
    main(book=args.book)
