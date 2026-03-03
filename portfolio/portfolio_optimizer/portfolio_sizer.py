#!/usr/bin/env python3
"""
Conviction-based portfolio sizer.

Takes user-supplied conviction levels (1–5) per ticker and sizes positions
using the same CVXPY constraint framework and hedging pipeline as the
portfolio optimizer, but replaces signal generation with direct conviction
mapping.
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

try:
    from .portfolio_optimizer import (
        BASE_CCY,
        BETA_EWMA_HALFLIFE_DAYS,
        BETA_FALLBACK,
        BETA_METHOD,
        BETA_MIN_OBS,
        BETA_SHRINK_TO_ONE,
        BOND_10YR_EQUIV_MAX,
        CMDTY_GROSS_MAX,
        CURRENCY_OF_TICKER,
        DURATION_OF_TICKER,
        EQ_NET_MAX,
        EQ_NET_MIN,
        FX_GROSS_MAX,
        GROSS_MAX,
        LONG_MAX,
        MARKET_TICKER_LONG,
        MARKET_TICKER_SHORT,
        MIN_ABS_WEIGHT,
        PORTFOLIO_CSV,
        SEVERE_DD_MAX,
        SHORT_MIN,
        apply_distressed_gating,
        apply_hedges_with_gross_cap,
        apply_net_neutral,
        compute_beta_frame,
        compute_defense_volatility,
        compute_severe_drawdown_flags,
        compute_10yr_equivalent,
        download_prices,
        ensure_psd,
        exposures_by_class,
        get_required_fx_tickers,
        identify_binding_constraint,
        max_scale_to_respect_linear_caps,
        solve_joint_hedge_weights,
        to_usd_price,
    )
except ImportError:
    from portfolio_optimizer import (
        BASE_CCY,
        BETA_EWMA_HALFLIFE_DAYS,
        BETA_FALLBACK,
        BETA_METHOD,
        BETA_MIN_OBS,
        BETA_SHRINK_TO_ONE,
        BOND_10YR_EQUIV_MAX,
        CMDTY_GROSS_MAX,
        CURRENCY_OF_TICKER,
        DURATION_OF_TICKER,
        EQ_NET_MAX,
        EQ_NET_MIN,
        FX_GROSS_MAX,
        GROSS_MAX,
        LONG_MAX,
        MARKET_TICKER_LONG,
        MARKET_TICKER_SHORT,
        MIN_ABS_WEIGHT,
        PORTFOLIO_CSV,
        SEVERE_DD_MAX,
        SHORT_MIN,
        apply_distressed_gating,
        apply_hedges_with_gross_cap,
        apply_net_neutral,
        compute_beta_frame,
        compute_defense_volatility,
        compute_severe_drawdown_flags,
        compute_10yr_equivalent,
        download_prices,
        ensure_psd,
        exposures_by_class,
        get_required_fx_tickers,
        identify_binding_constraint,
        max_scale_to_respect_linear_caps,
        solve_joint_hedge_weights,
        to_usd_price,
    )

LOGGER = logging.getLogger(__name__)

CONVICTION_MIN = 1
CONVICTION_MAX = 5


def _parse_positions(
    positions: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    """Parse and validate position rows into {ticker: conviction} dict."""
    if not positions:
        raise ValueError("positions must be a non-empty list.")

    result: Dict[str, int] = {}
    for idx, row in enumerate(positions):
        ticker = str(row.get("ticker", "")).strip().upper()
        conviction_raw = row.get("conviction", 3)

        if not ticker:
            continue

        try:
            conviction = int(conviction_raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"Position '{ticker}' has an invalid conviction: {conviction_raw!r}."
            ) from None

        if conviction < CONVICTION_MIN or conviction > CONVICTION_MAX:
            raise ValueError(
                f"Position '{ticker}' conviction must be {CONVICTION_MIN}–{CONVICTION_MAX}, got {conviction}."
            )

        result[ticker] = conviction

    if not result:
        raise ValueError("No valid positions provided. Add at least one ticker with a conviction level.")

    return result


def _build_conviction_weights(
    meta: pd.DataFrame,
    convictions: Dict[str, int],
) -> pd.Series:
    """
    Map conviction levels (1–5) to raw target weights.

    Longs:  weight = LONG_MAX  * (conviction / CONVICTION_MAX)
    Shorts: weight = SHORT_MIN * (conviction / CONVICTION_MAX)
    """
    w_raw = pd.Series(0.0, index=meta.index)

    for ticker in meta.index:
        direction = str(meta.loc[ticker, "direction"]).strip().lower()
        conviction = convictions.get(ticker, 0)
        if conviction <= 0 or not direction:
            continue

        frac = conviction / CONVICTION_MAX
        if direction == "long":
            w_raw[ticker] = LONG_MAX * frac
        elif direction == "short":
            w_raw[ticker] = SHORT_MIN * frac

    return w_raw


def size_portfolio(
    positions: Sequence[Mapping[str, Any]],
    book: Optional[float] = 100_000.0,
    target_leverage: Optional[float] = 2.0,
) -> dict:
    """
    Size a portfolio from user conviction levels using CVXPY optimization.

    Args:
        positions: List of {ticker: str, conviction: int (1–5)} dicts.
        book: Book size in USD.
        target_leverage: Target gross leverage (0.5–4.0).

    Returns:
        Same output dict structure as optimize_portfolio().
    """
    try:
        convictions = _parse_positions(positions)

        # Load portfolio metadata
        meta = pd.read_csv(PORTFOLIO_CSV)
        meta["direction"] = meta["direction"].fillna("")
        meta = meta.set_index("ticker")

        # Filter to user-requested tickers that exist in CSV
        requested = list(convictions.keys())
        missing_from_csv = [t for t in requested if t not in meta.index]
        if missing_from_csv:
            raise ValueError(
                f"Tickers not found in portfolio.csv: {missing_from_csv}. "
                f"Only tickers defined in the portfolio are allowed."
            )

        tickers = [t for t in requested if t in meta.index]
        meta = meta.loc[tickers]

        if len(tickers) < 2:
            raise ValueError("Need at least 2 tickers to size a portfolio.")

        # Download prices
        fx_tickers = get_required_fx_tickers(tickers)
        market_tickers = [MARKET_TICKER_LONG, MARKET_TICKER_SHORT]
        all_tickers_to_fetch = list(set(tickers + market_tickers))
        prices_all = download_prices(all_tickers_to_fetch, fx_tickers)

        missing_cols = [t for t in tickers if t not in prices_all.columns]
        if missing_cols:
            raise ValueError(f"Failed to download ticker prices: {missing_cols}")

        for mt in market_tickers:
            if mt not in prices_all.columns:
                raise ValueError(f"Failed to download benchmark '{mt}' for beta regression.")

        # Convert to USD
        usd_prices = pd.DataFrame(index=prices_all.index)
        tickers_plus_market = list(set(tickers + market_tickers))
        for t in tickers_plus_market:
            local_px = prices_all[t]
            ccy = CURRENCY_OF_TICKER.get(t, BASE_CCY)
            usd_prices[t] = to_usd_price(local_px, ccy, prices_all)

        # Returns
        usd_prices = usd_prices.ffill()
        rets = usd_prices.pct_change(fill_method=None).dropna(how="all")
        tickers = [t for t in tickers if t in rets.columns]
        meta = meta.loc[tickers]

        # Distressed gating
        meta = apply_distressed_gating(meta, prices_all)

        # Defense volatility
        defense_vol = compute_defense_volatility(usd_prices, tickers)
        meta["realized_vol"] = defense_vol

        # Severe drawdown flags
        equity_tickers_dd = [t for t in tickers if meta.loc[t, "asset"].lower() == "equity"]
        severe_dd_flags = compute_severe_drawdown_flags(usd_prices, equity_tickers_dd)
        meta["severe_drawdown"] = pd.Series({t: severe_dd_flags.get(t, False) for t in meta.index})

        if len(tickers) < 2:
            raise ValueError("Need at least 2 instruments with returns to optimize.")

        # Covariance
        rets_portfolio = rets[tickers]
        Sigma = rets_portfolio.cov().values
        Sigma = ensure_psd(Sigma, eps=1e-10)
        L = np.linalg.cholesky(Sigma)

        # Betas
        beta_frame, betas_all_spy, betas_all_iwm = compute_beta_frame(rets, tickers)
        betas_spy = beta_frame["beta_spy"]
        betas_iwm = beta_frame["beta_iwm"]

        # Build conviction-driven raw weights
        w_raw = _build_conviction_weights(meta, convictions).reindex(tickers).fillna(0.0)
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

        # Direction constraints
        direction = meta["direction"].str.lower()
        long_mask = direction.eq("long").values
        short_mask = direction.eq("short").values
        if long_mask.any():
            constraints.append(w[long_mask] >= 0.0)
            constraints.append(w[long_mask] <= LONG_MAX)
        if short_mask.any():
            constraints.append(w[short_mask] <= -MIN_ABS_WEIGHT)
            constraints.append(w[short_mask] >= SHORT_MIN)
            severe_dd_short_mask = meta["severe_drawdown"].values & short_mask
            if severe_dd_short_mask.any():
                constraints.append(w[severe_dd_short_mask] >= -SEVERE_DD_MAX)

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
            bond_tickers = [tickers[i] for i in range(n) if bond_mask[i]]
            duration_coeffs = np.array([DURATION_OF_TICKER.get(t, 10.0) / 10.0 for t in bond_tickers])
            constraints.append(cp.sum(cp.multiply(duration_coeffs, cp.abs(w[bond_mask]))) <= BOND_10YR_EQUIV_MAX)

        # Objective: minimize deviation from conviction-driven raw weights
        objective = cp.Minimize(cp.sum_squares(w - w_raw_vec))
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=False)

        if w.value is None:
            return {"error": "Optimization failed. Check data/constraints for feasibility.", "status": "infeasible"}

        w_star = pd.Series(w.value, index=tickers)

        # Portfolio vol helper
        def port_vol(w_vec: np.ndarray) -> float:
            x = L @ w_vec
            return float(np.sqrt(np.maximum(0.0, x.T @ x)))

        vol0 = port_vol(w_star.values)
        if vol0 <= 0:
            return {"error": "Optimized portfolio has ~0 volatility; check inputs."}

        # Post-solve scaling
        k_linear = max_scale_to_respect_linear_caps(w_star, meta)
        if target_leverage is not None:
            current_gross = np.abs(w_star).sum()
            k_user = target_leverage / current_gross if current_gross > 0 else 1.0
            k = min(k_user, k_linear)
        else:
            k = k_linear

        if MIN_ABS_WEIGHT > 0 and short_mask.any():
            min_abs_short = float(np.min(np.abs(w_star.values[short_mask])))
            if min_abs_short > 0:
                k_floor = MIN_ABS_WEIGHT / min_abs_short
                if k < k_floor:
                    k = k_floor

        w_final = w_star * k
        vol_final = port_vol(w_final.values)

        # Benchmark volatility
        benchmark_vol = compute_defense_volatility(usd_prices, market_tickers)
        vol_spy = benchmark_vol.get(MARKET_TICKER_LONG, np.nan)
        vol_iwm = benchmark_vol.get(MARKET_TICKER_SHORT, np.nan)

        # Hedges
        w_final, hedge_summary = apply_hedges_with_gross_cap(
            w_final,
            betas_spy,
            betas_iwm,
            betas_all_spy,
            betas_all_iwm,
            long_mask,
            short_mask,
            eq_mask,
        )
        vol_final = port_vol(w_final.values)

        # Exposures
        exp = exposures_by_class(w_final, meta)
        exp["hedge_gross"] = hedge_summary["hedge_gross"]
        exp["total_gross"] = hedge_summary["gross_with_hedges"]

        # Constraints utilization
        constraints_util = {
            "Total Gross (400%)": {
                "limit": GROSS_MAX,
                "current": exp["total_gross"],
                "utilization": exp["total_gross"] / GROSS_MAX if GROSS_MAX > 0 else 0,
            },
            "FX Gross (200%)": {
                "limit": FX_GROSS_MAX,
                "current": exp.get("fx_gross", 0),
                "utilization": exp.get("fx_gross", 0) / FX_GROSS_MAX if FX_GROSS_MAX > 0 else 0,
            },
            "Commodity Gross (100%)": {
                "limit": CMDTY_GROSS_MAX,
                "current": exp.get("commodity_gross", 0),
                "utilization": exp.get("commodity_gross", 0) / CMDTY_GROSS_MAX if CMDTY_GROSS_MAX > 0 else 0,
            },
            "Equity Net Max (100%)": {
                "limit": EQ_NET_MAX,
                "current": exp.get("equity_net", 0),
                "utilization": max(0, exp.get("equity_net", 0)) / EQ_NET_MAX if EQ_NET_MAX > 0 else 0,
            },
            "Equity Net Min (-50%)": {
                "limit": EQ_NET_MIN,
                "current": exp.get("equity_net", 0),
                "utilization": max(0, -exp.get("equity_net", 0)) / abs(EQ_NET_MIN) if EQ_NET_MIN < 0 else 0,
            },
        }

        bond_10yr = compute_10yr_equivalent(w_final, meta)
        if bond_10yr > 0:
            constraints_util["Bond 10yr Equiv (300%)"] = {
                "limit": BOND_10YR_EQUIV_MAX,
                "current": bond_10yr,
                "utilization": bond_10yr / BOND_10YR_EQUIV_MAX if BOND_10YR_EQUIV_MAX > 0 else 0,
            }

        # Build weights DataFrame
        latest_prices = usd_prices[tickers].iloc[-1]
        weights_df = pd.DataFrame({
            "ticker": tickers,
            "asset": meta["asset"].values,
            "direction": meta["direction"].values,
            "distressed": meta["distressed"].values if "distressed" in meta.columns else False,
            "conviction": [convictions.get(t, 0) for t in tickers],
            "beta_spy": betas_spy.values,
            "beta_iwm": betas_iwm.values,
            "realized_vol": meta["realized_vol"].values,
            "weight": w_final.values,
            "price": latest_prices.values,
        })
        if book is not None:
            weights_df["dollar_weight"] = w_final.values * book
            weights_df["shares"] = (weights_df["dollar_weight"] / weights_df["price"]).round(0).astype(int)
        weights_df = weights_df.sort_values("weight", ascending=False)

        # Build hedges DataFrame
        hedge_spy_weight = hedge_summary["hedge_spy_weight"]
        hedge_iwm_weight = hedge_summary["hedge_iwm_weight"]
        spy_price = float(usd_prices[MARKET_TICKER_LONG].iloc[-1])
        iwm_price = float(usd_prices[MARKET_TICKER_SHORT].iloc[-1])
        hedges_data: Dict[str, Any] = {
            "ticker": [MARKET_TICKER_LONG, MARKET_TICKER_SHORT],
            "type": ["hedge", "hedge"],
            "direction": [
                "short" if hedge_spy_weight < 0 else "long",
                "long" if hedge_iwm_weight > 0 else "short",
            ],
            "weight": [hedge_spy_weight, hedge_iwm_weight],
            "price": [spy_price, iwm_price],
        }
        if book is not None:
            hedges_data["dollar_weight"] = [hedge_spy_weight * book, hedge_iwm_weight * book]
            hedges_data["shares"] = [
                int(round(hedge_spy_weight * book / spy_price)),
                int(round(hedge_iwm_weight * book / iwm_price)),
            ]
        hedges_df = pd.DataFrame(hedges_data)

        # Max scaled version
        k_max = max_scale_to_respect_linear_caps(w_final, meta, include_position_limits=False)
        w_max_scaled = w_final * k_max
        vol_max_scaled = port_vol(w_max_scaled.values)
        binding = identify_binding_constraint(w_max_scaled, meta, include_position_limits=False)
        exp_max = exposures_by_class(w_max_scaled, meta)

        max_scaled_weights_df = pd.DataFrame({
            "ticker": tickers,
            "asset": meta["asset"].values,
            "direction": meta["direction"].values,
            "weight": w_max_scaled.values,
            "price": latest_prices.values,
        })
        if book is not None:
            max_scaled_weights_df["dollar_weight"] = w_max_scaled.values * book
            max_scaled_weights_df["shares"] = (
                max_scaled_weights_df["dollar_weight"] / max_scaled_weights_df["price"]
            ).round(0).astype(int)
        max_scaled_weights_df = max_scaled_weights_df.sort_values("weight", ascending=False)

        return {
            "status": prob.status,
            "error": None,
            "timestamp": datetime.now(),
            "book_size": book,
            "target_leverage": target_leverage,

            # Solution metrics
            "vol_daily": vol_final,
            "vol_spy": vol_spy,
            "vol_iwm": vol_iwm,
            "gross_leverage": exp["total_gross"],

            # Beta hedging
            "beta_long_spy": hedge_summary["beta_long_spy"],
            "beta_short_spy": hedge_summary["beta_short_spy"],
            "beta_long_iwm": hedge_summary["beta_long_iwm"],
            "beta_short_iwm": hedge_summary["beta_short_iwm"],
            "net_beta_spy": hedge_summary["net_beta_spy"],
            "net_beta_iwm": hedge_summary["net_beta_iwm"],
            "post_hedge_beta_spy": hedge_summary["post_hedge_beta_spy"],
            "post_hedge_beta_iwm": hedge_summary["post_hedge_beta_iwm"],
            "hedge_spy_weight": hedge_spy_weight,
            "hedge_iwm_weight": hedge_iwm_weight,
            "beta_method": BETA_METHOD,
            "beta_halflife_days": BETA_EWMA_HALFLIFE_DAYS,
            "beta_min_obs": BETA_MIN_OBS,
            "beta_shrink_to_one": BETA_SHRINK_TO_ONE,

            # Exposures
            "exposures": exp,

            # Constraints utilization
            "constraints": constraints_util,

            # DataFrames
            "weights_df": weights_df,
            "hedges_df": hedges_df,

            # Max scaled version
            "max_scaled": {
                "scale_factor": k_max,
                "binding_constraint": binding,
                "vol_daily": vol_max_scaled,
                "weights_df": max_scaled_weights_df,
                "exposures": exp_max,
            },
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


def get_data(
    positions: Sequence[Mapping[str, Any]],
    book: float = 100_000.0,
    target_leverage: float = 2.0,
) -> dict:
    return size_portfolio(positions=positions, book=book, target_leverage=target_leverage)
