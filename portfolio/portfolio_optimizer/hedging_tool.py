#!/usr/bin/env python3
"""
Standalone hedging utility that computes SPY/IWM hedge legs for user-supplied
signed portfolio weights.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

try:
    from .portfolio_analyzer import (
        BASE_CCY,
        BETA_EWMA_HALFLIFE_DAYS,
        BETA_FALLBACK,
        BETA_METHOD,
        BETA_MIN_OBS,
        BETA_SHRINK_TO_ONE,
        CURRENCY_OF_TICKER,
        MARKET_TICKER_LONG,
        MARKET_TICKER_SHORT,
        compute_beta_frame,
        download_prices,
        get_required_fx_tickers,
        solve_joint_hedge_weights,
        to_usd_price,
    )
except ImportError:
    from portfolio_analyzer import (
        BASE_CCY,
        BETA_EWMA_HALFLIFE_DAYS,
        BETA_FALLBACK,
        BETA_METHOD,
        BETA_MIN_OBS,
        BETA_SHRINK_TO_ONE,
        CURRENCY_OF_TICKER,
        MARKET_TICKER_LONG,
        MARKET_TICKER_SHORT,
        compute_beta_frame,
        download_prices,
        get_required_fx_tickers,
        solve_joint_hedge_weights,
        to_usd_price,
    )

DEFAULT_BOOK = 100_000.0


def _normalize_positions(positions: Sequence[Mapping[str, Any]]) -> tuple[pd.Series, int]:
    """
    Parse and aggregate position rows:
    - ticker uppercased
    - duplicate tickers summed
    - blank rows skipped
    """
    if not isinstance(positions, Sequence) or len(positions) == 0:
        raise ValueError("positions must be a non-empty list of {ticker, weight} rows.")

    aggregated: dict[str, float] = {}
    input_count = 0

    for idx, row in enumerate(positions):
        if not isinstance(row, Mapping):
            raise ValueError(f"Position at index {idx} must be an object with ticker and weight.")

        ticker_raw = row.get("ticker", "")
        weight_raw = row.get("weight")

        ticker = str(ticker_raw).strip().upper()
        weight_text = "" if weight_raw is None else str(weight_raw).strip()

        # Allow UI-friendly blank rows to exist in the payload.
        if not ticker and not weight_text:
            continue

        input_count += 1

        if not ticker:
            raise ValueError(f"Position at index {idx} has an empty ticker.")

        try:
            weight = float(weight_raw)
        except (TypeError, ValueError):
            raise ValueError(f"Position '{ticker}' has an invalid numeric weight: {weight_raw!r}.") from None

        if not np.isfinite(weight):
            raise ValueError(f"Position '{ticker}' has a non-finite weight: {weight_raw!r}.")

        aggregated[ticker] = aggregated.get(ticker, 0.0) + weight

    if input_count == 0 or not aggregated:
        raise ValueError("No valid positions provided. Add at least one ticker with a numeric weight.")

    weights = pd.Series(aggregated, dtype=float).sort_index()
    return weights, input_count


def _direction_from_weight(weight: float) -> str:
    if weight > 0:
        return "long"
    if weight < 0:
        return "short"
    return "flat"


def _build_positions_df(
    weights: pd.Series,
    betas_spy: pd.Series,
    betas_iwm: pd.Series,
    latest_prices: pd.Series,
    book: float,
) -> pd.DataFrame:
    beta_contrib_spy = weights * betas_spy.reindex(weights.index).fillna(BETA_FALLBACK)
    beta_contrib_iwm = weights * betas_iwm.reindex(weights.index).fillna(BETA_FALLBACK)

    df = pd.DataFrame(
        {
            "ticker": weights.index,
            "direction": [_direction_from_weight(float(w)) for w in weights.values],
            "weight": weights.values,
            "beta_spy": betas_spy.reindex(weights.index).fillna(BETA_FALLBACK).values,
            "beta_iwm": betas_iwm.reindex(weights.index).fillna(BETA_FALLBACK).values,
            "beta_contribution_spy": beta_contrib_spy.values,
            "beta_contribution_iwm": beta_contrib_iwm.values,
            "price": latest_prices.reindex(weights.index).values,
        }
    )
    df["dollar_weight"] = df["weight"] * float(book)
    shares = np.where(
        pd.to_numeric(df["price"], errors="coerce").replace(0, np.nan).notna(),
        np.round(df["dollar_weight"] / pd.to_numeric(df["price"], errors="coerce")),
        0.0,
    )
    df["shares"] = shares.astype(int)
    df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    return df


def _build_hedges_df(
    hedge_spy_weight: float,
    hedge_iwm_weight: float,
    spy_price: float,
    iwm_price: float,
    book: float,
) -> pd.DataFrame:
    hedges = pd.DataFrame(
        {
            "ticker": [MARKET_TICKER_LONG, MARKET_TICKER_SHORT],
            "type": ["hedge", "hedge"],
            "direction": [_direction_from_weight(hedge_spy_weight), _direction_from_weight(hedge_iwm_weight)],
            "weight": [hedge_spy_weight, hedge_iwm_weight],
            "price": [spy_price, iwm_price],
        }
    )
    hedges["dollar_weight"] = hedges["weight"] * float(book)
    hedges["shares"] = [
        int(round(hedge_spy_weight * float(book) / spy_price)) if spy_price else 0,
        int(round(hedge_iwm_weight * float(book) / iwm_price)) if iwm_price else 0,
    ]
    return hedges


def compute_hedge(positions: Sequence[Mapping[str, Any]], book: float = DEFAULT_BOOK) -> dict:
    if not np.isfinite(float(book)) or float(book) <= 0:
        raise ValueError("book must be a positive number.")

    weights, input_count = _normalize_positions(positions)
    tickers = weights.index.tolist()

    fx_tickers = get_required_fx_tickers(tickers)
    market_tickers = [MARKET_TICKER_LONG, MARKET_TICKER_SHORT]
    all_tickers_to_fetch = sorted(set(tickers + market_tickers))
    prices_all = download_prices(all_tickers_to_fetch, fx_tickers)

    missing_tickers = [t for t in tickers if t not in prices_all.columns]
    if missing_tickers:
        raise ValueError(f"Failed to download ticker prices: {missing_tickers}")

    for mt in market_tickers:
        if mt not in prices_all.columns:
            raise ValueError(f"Failed to download benchmark ticker '{mt}' for beta regression.")

    usd_prices = pd.DataFrame(index=prices_all.index)
    for ticker in all_tickers_to_fetch:
        local_px = prices_all[ticker]
        ccy = CURRENCY_OF_TICKER.get(ticker, BASE_CCY)
        usd_prices[ticker] = to_usd_price(local_px, ccy, prices_all)

    usd_prices = usd_prices.ffill()
    rets = usd_prices.pct_change(fill_method=None).dropna(how="all")
    if rets.empty:
        raise ValueError("Insufficient price history to compute returns and betas.")

    if not set(tickers).issubset(set(rets.columns)):
        missing_returns = sorted(set(tickers) - set(rets.columns))
        raise ValueError(f"Missing return history for tickers: {missing_returns}")

    beta_frame, betas_all_spy, betas_all_iwm = compute_beta_frame(rets, tickers)
    betas_spy = beta_frame["beta_spy"].reindex(tickers).fillna(BETA_FALLBACK)
    betas_iwm = beta_frame["beta_iwm"].reindex(tickers).fillna(BETA_FALLBACK)

    w_vals = weights.reindex(tickers).fillna(0.0).values.astype(float)
    net_beta_spy = float(betas_spy.values @ w_vals)
    net_beta_iwm = float(betas_iwm.values @ w_vals)

    hedge_spy_weight, hedge_iwm_weight, post_hedge_beta_spy, post_hedge_beta_iwm = solve_joint_hedge_weights(
        net_beta_spy,
        net_beta_iwm,
        betas_all_spy,
        betas_all_iwm,
    )

    # Post-hedge gross and daily volatility (simple-return covariance framework).
    gross_after_hedging = float(np.abs(w_vals).sum() + abs(hedge_spy_weight) + abs(hedge_iwm_weight))
    post_weights = np.concatenate([w_vals, np.array([hedge_spy_weight, hedge_iwm_weight], dtype=float)])
    vol_cols = tickers + [MARKET_TICKER_LONG, MARKET_TICKER_SHORT]
    rets_vol = rets.reindex(columns=vol_cols).dropna(how="any")
    if rets_vol.empty:
        volatility_after_hedging = float("nan")
    else:
        sigma = rets_vol.cov().values
        variance = float(post_weights.T @ sigma @ post_weights)
        volatility_after_hedging = float(np.sqrt(max(0.0, variance)))

    latest_prices = usd_prices[tickers].iloc[-1]
    spy_price = float(usd_prices[MARKET_TICKER_LONG].iloc[-1])
    iwm_price = float(usd_prices[MARKET_TICKER_SHORT].iloc[-1])

    positions_df = _build_positions_df(
        weights=weights,
        betas_spy=betas_spy,
        betas_iwm=betas_iwm,
        latest_prices=latest_prices,
        book=float(book),
    )
    hedges_df = _build_hedges_df(
        hedge_spy_weight=hedge_spy_weight,
        hedge_iwm_weight=hedge_iwm_weight,
        spy_price=spy_price,
        iwm_price=iwm_price,
        book=float(book),
    )

    return {
        "timestamp": datetime.now(),
        "book_size": float(book),
        "input_count": int(input_count),
        "unique_ticker_count": int(len(tickers)),
        "gross_input": float(np.abs(weights.values).sum()),
        "net_input": float(weights.values.sum()),
        "net_beta_spy": net_beta_spy,
        "net_beta_iwm": net_beta_iwm,
        "post_hedge_beta_spy": post_hedge_beta_spy,
        "post_hedge_beta_iwm": post_hedge_beta_iwm,
        "hedge_spy_weight": hedge_spy_weight,
        "hedge_iwm_weight": hedge_iwm_weight,
        "hedge_spy_dollar": hedge_spy_weight * float(book),
        "hedge_iwm_dollar": hedge_iwm_weight * float(book),
        "gross_after_hedging": gross_after_hedging,
        "volatility_after_hedging": volatility_after_hedging,
        "beta_method": BETA_METHOD,
        "beta_halflife_days": BETA_EWMA_HALFLIFE_DAYS,
        "beta_min_obs": BETA_MIN_OBS,
        "beta_shrink_to_one": BETA_SHRINK_TO_ONE,
        "positions_df": positions_df,
        "hedges_df": hedges_df,
    }


def get_data(positions: Sequence[Mapping[str, Any]], book: float = DEFAULT_BOOK) -> dict:
    return compute_hedge(positions=positions, book=book)
