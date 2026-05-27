#!/usr/bin/env python3
"""
Conviction-based portfolio sizer.

Takes user-supplied conviction levels (1–5) per ticker and sizes positions
using the same CVXPY constraint framework and hedging pipeline as the
portfolio analyzer utilities, but replaces signal generation with direct conviction
mapping.
"""

from __future__ import annotations

import logging
import re
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal, cast

import cvxpy as cp
import numpy as np
import pandas as pd

from ontology.runtime_read_service import get_hedge_positions as _get_hedge_positions
from ontology.runtime_read_service import get_positions_df as _get_positions_df
from portfolio.portfolio_optimizer.portfolio_analyzer import (
    BASE_CCY,
    BETA_EWMA_HALFLIFE_DAYS,
    BETA_FALLBACK,
    BETA_METHOD,
    BETA_MIN_OBS,
    BETA_SHRINK_TO_ONE,
    BOND_10YR_EQUIV_MAX,
    CMDTY_GROSS_MAX,
    DURATION_OF_TICKER,
    EQ_NET_MAX,
    EQ_NET_MIN,
    FX_GROSS_MAX,
    GROSS_MAX,
    HEDGE_SOLVE_RIDGE,
    LONG_MAX,
    MARKET_TICKER_LONG,
    MARKET_TICKER_SHORT,
    SEVERE_DD_MAX,
    SHORT_MIN,
    apply_contrarian_gating,
    compute_10yr_equivalent,
    compute_betas,
    compute_defense_volatility,
    compute_severe_drawdown_flags,
    ensure_psd,
    exposures_by_class,
    fetch_prices_for_portfolio_symbols,
    identify_binding_constraint,
    max_scale_to_respect_linear_caps,
    prepare_instrument_metadata,
    to_usd_price,
    unit_notional_in_base,
)
from portfolio.position_groups import group_key, normalize_group_conviction, normalize_group_name

LOGGER = logging.getLogger(__name__)

CONVICTION_MIN = 1
CONVICTION_MAX = 5
BetaHedgeMode = Literal["spy", "iwm", "qqq", "spy_iwm", "spy_qqq", "iwm_qqq", "spy_iwm_qqq"]
BETA_HEDGE_MODE_SPY_IWM: BetaHedgeMode = "spy_iwm"
BETA_HEDGE_MODE_SPY: BetaHedgeMode = "spy"
MARKET_TICKER_QQQ = "QQQ"
HEDGE_TICKER_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9.^=_-]{0,31}$")
BETA_HEDGE_MODE_TICKERS: dict[str, tuple[str, ...]] = {
    "spy": (MARKET_TICKER_LONG,),
    "iwm": (MARKET_TICKER_SHORT,),
    "qqq": (MARKET_TICKER_QQQ,),
    "spy_iwm": (MARKET_TICKER_LONG, MARKET_TICKER_SHORT),
    "spy_qqq": (MARKET_TICKER_LONG, MARKET_TICKER_QQQ),
    "iwm_qqq": (MARKET_TICKER_SHORT, MARKET_TICKER_QQQ),
    "spy_iwm_qqq": (MARKET_TICKER_LONG, MARKET_TICKER_SHORT, MARKET_TICKER_QQQ),
}


def _benchmark_key(ticker: str) -> str:
    return ticker.strip().lower()


def _beta_metric_name(prefix: str, ticker: str) -> str:
    return f"{prefix}_{_benchmark_key(ticker)}"


def _normalize_beta_hedge_mode(value: str | None) -> BetaHedgeMode:
    normalized = (value or BETA_HEDGE_MODE_SPY_IWM).strip().lower()
    if normalized in BETA_HEDGE_MODE_TICKERS:
        return cast(BetaHedgeMode, normalized)
    modes = ", ".join(sorted(BETA_HEDGE_MODE_TICKERS))
    raise ValueError(f"beta_hedge_mode must be one of: {modes}.")


def _hedge_tickers_for_mode(beta_hedge_mode: BetaHedgeMode) -> tuple[str, ...]:
    return BETA_HEDGE_MODE_TICKERS[beta_hedge_mode]


def _normalize_hedge_ticker(ticker: Any) -> str:
    normalized = str(ticker or "").strip().upper()
    if not normalized:
        raise ValueError("hedge_tickers cannot contain empty tickers.")
    if not HEDGE_TICKER_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid hedge ticker '{ticker}'. Use a yfinance-compatible ticker symbol.")
    return normalized


def _normalize_hedge_tickers(
    hedge_tickers: Sequence[str] | str | None,
    beta_hedge_mode: str | None = BETA_HEDGE_MODE_SPY_IWM,
) -> tuple[str, ...]:
    if hedge_tickers is None:
        return _hedge_tickers_for_mode(_normalize_beta_hedge_mode(beta_hedge_mode))

    raw_values: Sequence[str] = [hedge_tickers] if isinstance(hedge_tickers, str) else hedge_tickers
    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        ticker = _normalize_hedge_ticker(value)
        if ticker not in seen:
            seen.add(ticker)
            normalized.append(ticker)

    if not normalized:
        raise ValueError("hedge_tickers must contain at least one ticker.")
    return tuple(normalized)


def _compute_equity_net_betas(
    w: pd.Series,
    beta_by_benchmark: dict[str, pd.Series],
    long_mask: np.ndarray,
    short_mask: np.ndarray,
    eq_mask: np.ndarray,
) -> dict[str, float]:
    exposure_mask = eq_mask
    long_exposure_mask = long_mask & exposure_mask
    short_exposure_mask = short_mask & exposure_mask

    out: dict[str, float] = {}
    for benchmark, betas in beta_by_benchmark.items():
        key = _benchmark_key(benchmark)
        beta_long = (
            float(betas.values[long_exposure_mask] @ w.values[long_exposure_mask]) if long_exposure_mask.any() else 0.0
        )
        beta_short = (
            float(betas.values[short_exposure_mask] @ w.values[short_exposure_mask])
            if short_exposure_mask.any()
            else 0.0
        )
        out[f"beta_long_{key}"] = beta_long
        out[f"beta_short_{key}"] = beta_short
        out[f"net_beta_{key}"] = beta_long + beta_short
    return out


def _solve_selected_hedge_weights(
    beta_summary: dict[str, float],
    selected_hedges: Sequence[str],
    betas_all_by_benchmark: dict[str, pd.Series],
    diagnostic_tickers: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    target_benchmarks = list(selected_hedges)
    B = np.array(
        [
            [
                float(betas_all_by_benchmark.get(benchmark, pd.Series(dtype=float)).get(hedge_ticker, BETA_FALLBACK))
                for hedge_ticker in selected_hedges
            ]
            for benchmark in target_benchmarks
        ],
        dtype=float,
    )
    target = np.array(
        [-float(beta_summary.get(_beta_metric_name("net_beta", benchmark), 0.0)) for benchmark in target_benchmarks],
        dtype=float,
    )
    ridge = HEDGE_SOLVE_RIDGE * np.eye(len(selected_hedges))
    hedge = np.linalg.solve(B.T @ B + ridge, B.T @ target)

    hedge_weights = {ticker: 0.0 for ticker in diagnostic_tickers}
    for ticker, weight in zip(selected_hedges, hedge, strict=False):
        hedge_weights[ticker] = float(weight)

    post_betas: dict[str, float] = {}
    for benchmark in diagnostic_tickers:
        benchmark_betas = betas_all_by_benchmark.get(benchmark, pd.Series(dtype=float))
        adjustment = sum(
            float(benchmark_betas.get(hedge_ticker, BETA_FALLBACK)) * float(hedge_weights.get(hedge_ticker, 0.0))
            for hedge_ticker in selected_hedges
        )
        post_betas[benchmark] = float(beta_summary.get(_beta_metric_name("net_beta", benchmark), 0.0) + adjustment)

    return hedge_weights, post_betas


def _apply_beta_hedges_with_gross_cap(
    w: pd.Series,
    beta_by_benchmark: dict[str, pd.Series],
    betas_all_by_benchmark: dict[str, pd.Series],
    long_mask: np.ndarray,
    short_mask: np.ndarray,
    eq_mask: np.ndarray,
    selected_hedges: Sequence[str],
) -> tuple[pd.Series, dict[str, Any]]:
    selected_hedges = tuple(selected_hedges)

    def _solve_for_weights(weights: pd.Series) -> dict[str, Any]:
        beta_summary = _compute_equity_net_betas(weights, beta_by_benchmark, long_mask, short_mask, eq_mask)
        hedge_weights, post_betas = _solve_selected_hedge_weights(
            beta_summary,
            selected_hedges,
            betas_all_by_benchmark,
            selected_hedges,
        )

        pre_hedge_gross = float(np.abs(weights).sum())
        hedge_gross = float(sum(abs(weight) for weight in hedge_weights.values()))
        gross_with_hedges = pre_hedge_gross + hedge_gross

        out: dict[str, Any] = dict(beta_summary)
        out.update(
            {
                "pre_hedge_gross": pre_hedge_gross,
                "hedge_gross": hedge_gross,
                "gross_with_hedges": gross_with_hedges,
                "selected_hedges": list(selected_hedges),
                "hedge_weights": hedge_weights,
            }
        )
        for benchmark in selected_hedges:
            key = _benchmark_key(benchmark)
            out[f"hedge_{key}_weight"] = float(hedge_weights.get(benchmark, 0.0))
            out[f"post_hedge_beta_{key}"] = float(post_betas.get(benchmark, 0.0))
        return out

    summary = _solve_for_weights(w)
    gross_scale_factor = 1.0

    if summary["gross_with_hedges"] > GROSS_MAX + 1e-10:
        gross_scale_factor = GROSS_MAX / summary["gross_with_hedges"]
        w = w * gross_scale_factor
        summary = _solve_for_weights(w)

    summary["gross_scale_factor"] = gross_scale_factor
    return w, summary


def _parse_positions(
    positions: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Parse and validate position rows keyed by ticker."""
    if not positions:
        raise ValueError("positions must be a non-empty list.")

    result: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(positions):  # noqa: B007
        ticker = str(row.get("ticker", "")).strip().upper()
        conviction_raw = row.get("conviction", 3)

        if not ticker:
            continue

        try:
            conviction = int(conviction_raw)
        except (TypeError, ValueError):
            raise ValueError(f"Position '{ticker}' has an invalid conviction: {conviction_raw!r}.") from None

        if conviction < CONVICTION_MIN or conviction > CONVICTION_MAX:
            raise ValueError(
                f"Position '{ticker}' conviction must be {CONVICTION_MIN}–{CONVICTION_MAX}, got {conviction}."
            )

        name = normalize_group_name(row.get("group_name"))
        gkey = group_key(name)
        group_conviction = normalize_group_conviction(row.get("group_conviction")) if gkey else None
        if gkey:
            if group_conviction is None:
                raise ValueError(f"Group '{name}' requires a group conviction.")
            group = groups.setdefault(gkey, {"name": name, "conviction": group_conviction})
            if group["conviction"] != group_conviction:
                raise ValueError(
                    f"Group '{group['name']}' has inconsistent group convictions "
                    f"({group['conviction']} and {group_conviction})."
                )

        result[ticker] = {
            "ticker": ticker,
            "conviction": conviction,
            "group_name": group["name"] if gkey else None,
            "group_key": gkey,
            "group_conviction": group_conviction,
        }

    if not result:
        raise ValueError("No valid positions provided. Add at least one ticker with a conviction level.")

    return result


def _build_conviction_weights(
    meta: pd.DataFrame,
    positions: dict[str, dict[str, Any]],
) -> pd.Series:
    """
    Map conviction levels (1–5) to raw target weights.

    Longs:  weight = LONG_MAX  * (conviction / CONVICTION_MAX)
    Shorts: weight = SHORT_MIN * (conviction / CONVICTION_MAX)
    """
    w_raw = pd.Series(0.0, index=meta.index)

    for ticker in meta.index:
        direction = str(meta.loc[ticker, "direction"]).strip().lower()
        position = positions.get(ticker) or {}
        conviction = int(position.get("conviction") or 0)
        if conviction <= 0 or not direction:
            continue
        if position.get("group_key"):
            continue

        frac = conviction / CONVICTION_MAX
        if direction == "long":
            w_raw[ticker] = LONG_MAX * frac
        elif direction == "short":
            w_raw[ticker] = SHORT_MIN * frac

    groups: dict[str, dict[str, Any]] = {}
    for ticker in meta.index:
        position = positions.get(ticker) or {}
        gkey = position.get("group_key")
        if not gkey:
            continue
        direction = str(meta.loc[ticker, "direction"]).strip().lower()
        conviction = int(position.get("conviction") or 0)
        group_conviction = int(position.get("group_conviction") or 0)
        if conviction <= 0 or group_conviction <= 0:
            continue
        group = groups.setdefault(
            str(gkey),
            {
                "name": position.get("group_name"),
                "direction": direction,
                "group_conviction": group_conviction,
                "members": [],
                "total_conviction": 0,
            },
        )
        if group["direction"] and direction and group["direction"] != direction:
            raise ValueError(f"Group '{group['name']}' cannot mix {group['direction']} and {direction} positions.")
        if group["group_conviction"] != group_conviction:
            raise ValueError(
                f"Group '{group['name']}' has inconsistent group convictions "
                f"({group['group_conviction']} and {group_conviction})."
            )
        group["members"].append((ticker, conviction))
        group["total_conviction"] += conviction

    for group in groups.values():
        total = float(group["total_conviction"])
        if total <= 0:
            continue
        frac = float(group["group_conviction"]) / CONVICTION_MAX
        if group["direction"] == "long":
            group_target = LONG_MAX * frac
        elif group["direction"] == "short":
            group_target = SHORT_MIN * frac
        else:
            continue
        for ticker, conviction in group["members"]:
            w_raw[ticker] = group_target * (float(conviction) / total)

    return w_raw


def _group_metadata_for_tickers(
    tickers: Sequence[str],
    positions: Mapping[str, Mapping[str, Any]],
    w_raw: pd.Series,
) -> dict[str, list[Any]]:
    group_targets: dict[str, float] = {}
    for ticker in tickers:
        position = positions.get(ticker) or {}
        gkey = position.get("group_key")
        if gkey:
            group_targets[str(gkey)] = group_targets.get(str(gkey), 0.0) + float(w_raw.get(ticker, 0.0))
    return {
        "group_name": [(positions.get(t) or {}).get("group_name") for t in tickers],
        "group_conviction": [(positions.get(t) or {}).get("group_conviction") for t in tickers],
        "group_raw_target": [
            group_targets.get(str((positions.get(t) or {}).get("group_key")), np.nan)
            if (positions.get(t) or {}).get("group_key")
            else np.nan
            for t in tickers
        ],
        "group_member_share": [
            abs(float(w_raw.get(t, 0.0)) / group_targets[str((positions.get(t) or {}).get("group_key"))])
            if (positions.get(t) or {}).get("group_key")
            and abs(group_targets.get(str((positions.get(t) or {}).get("group_key")), 0.0)) > 1e-12
            else np.nan
            for t in tickers
        ],
    }


def _compute_equity_beta_inputs(
    rets: pd.DataFrame,
    tickers: Sequence[str],
    market_tickers: Sequence[str],
    eq_mask: np.ndarray,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, pd.Series], list[str]]:
    """
    Compute beta regressions only for equity holdings.

    The hedge helper expects beta series aligned to the full portfolio ticker list,
    so non-equity holdings receive numeric zeroes for hedge math and NaN display
    values for the weights table.
    """
    equity_tickers = [ticker for ticker, is_equity in zip(tickers, eq_mask, strict=False) if bool(is_equity)]
    beta_columns = list(dict.fromkeys([*equity_tickers, *market_tickers]))
    beta_rets = rets[[col for col in beta_columns if col in rets.columns]]

    beta_by_benchmark: dict[str, pd.Series] = {}
    beta_display_by_benchmark: dict[str, pd.Series] = {}
    betas_all_by_benchmark: dict[str, pd.Series] = {}
    for benchmark in market_tickers:
        betas_all = compute_betas(beta_rets, benchmark)
        equity_display = betas_all.reindex(equity_tickers).fillna(BETA_FALLBACK)
        display = equity_display.reindex(tickers)
        beta_display_by_benchmark[benchmark] = display
        beta_by_benchmark[benchmark] = display.fillna(0.0)
        betas_all_by_benchmark[benchmark] = betas_all

    return beta_by_benchmark, beta_display_by_benchmark, betas_all_by_benchmark, equity_tickers


def _filter_equity_sizing_universe(
    meta: pd.DataFrame,
    requested: Sequence[str],
) -> tuple[list[str], list[str]]:
    asset = (
        meta["asset"].fillna("").astype(str).str.strip().str.lower()
        if "asset" in meta.columns
        else pd.Series([""] * len(meta), index=meta.index)
    )
    equity_tickers = [ticker for ticker in requested if ticker in meta.index and asset.get(ticker) == "equity"]
    excluded_tickers = [ticker for ticker in requested if ticker in meta.index and asset.get(ticker) != "equity"]
    return equity_tickers, excluded_tickers


def size_portfolio(
    positions: Sequence[Mapping[str, Any]],
    book: float | None = 100_000.0,
    target_leverage: float | None = 2.0,
    beta_hedge_mode: str | None = BETA_HEDGE_MODE_SPY_IWM,
    hedge_tickers: Sequence[str] | str | None = None,
) -> dict:
    """
    Size a portfolio from user conviction levels using CVXPY optimization.

    Args:
        positions: List of {ticker: str, conviction: int (1–5)} dicts.
        book: Book size in USD.
        target_leverage: Target gross leverage (0.5–4.0).
        beta_hedge_mode: Legacy hedge basket preset. Used only when
            hedge_tickers is not provided.
        hedge_tickers: Custom hedge basket. Tickers are normalized and de-duped.

    Returns:
        Same output dict structure as optimize_portfolio().
    """
    try:
        beta_hedge_mode = _normalize_beta_hedge_mode(beta_hedge_mode)
        selected_hedges = _normalize_hedge_tickers(hedge_tickers, beta_hedge_mode)
        positions_by_ticker = _parse_positions(positions)

        # Load portfolio metadata
        meta = _get_positions_df(fallback_to_csv=True)
        meta["direction"] = meta["direction"].fillna("")
        meta = meta.set_index("ticker")
        meta = prepare_instrument_metadata(meta)

        # Filter to user-requested tickers that exist in CSV
        requested = list(positions_by_ticker.keys())
        missing_from_csv = [t for t in requested if t not in meta.index]
        if missing_from_csv:
            raise ValueError(
                f"Tickers not found in portfolio.csv: {missing_from_csv}. "
                f"Only tickers defined in the portfolio are allowed."
            )

        tickers, excluded_non_equity_tickers = _filter_equity_sizing_universe(meta, requested)
        if not tickers:
            raise ValueError("No equity positions available to size. The portfolio sizer only sizes equity assets.")
        meta = meta.loc[tickers]

        if len(tickers) < 2:
            raise ValueError("Need at least 2 equity tickers to size a portfolio.")

        market_tickers = list(selected_hedges)
        prices_all, ticker_currencies, _symbol_map = fetch_prices_for_portfolio_symbols(meta, tickers, market_tickers)
        all_tickers_to_fetch = list(dict.fromkeys([*tickers, *market_tickers]))

        missing_cols = [t for t in tickers if t not in prices_all.columns]
        if missing_cols:
            raise ValueError(f"Failed to download ticker prices: {missing_cols}")

        for mt in market_tickers:
            if mt not in prices_all.columns:
                raise ValueError(f"Failed to download benchmark '{mt}' for beta regression.")

        # Convert to USD
        usd_prices = pd.DataFrame(index=prices_all.index)
        for t in all_tickers_to_fetch:
            local_px = prices_all[t]
            ccy = ticker_currencies.get(t, BASE_CCY)
            usd_prices[t] = to_usd_price(local_px, ccy, prices_all)

        # Returns
        usd_prices = usd_prices.ffill()
        rets = usd_prices.pct_change(fill_method=None).dropna(how="all")
        tickers = [t for t in tickers if t in rets.columns]
        meta = meta.loc[tickers]

        # Contrarian gating
        meta = apply_contrarian_gating(meta, prices_all)

        # Re-enable gated contrarian positions:
        # - Longs: always re-enable at 1/3
        # - Shorts: only re-enable if no_new_high_20d (hard gate stays for recent-high shorts)
        _gated_mask = (
            meta["contrarian"]
            & ~meta["contrarian_eligible"]
            & meta["direction_intended"].ne("")
            & (meta["direction_intended"].ne("short") | meta["no_new_high_20d"])
        )
        meta.loc[_gated_mask, "direction"] = meta.loc[_gated_mask, "direction_intended"]

        # Defense volatility
        defense_vol = compute_defense_volatility(usd_prices, tickers)
        meta["realized_vol"] = defense_vol

        # Severe drawdown flags
        equity_tickers_dd = [
            t
            for t in tickers
            if meta.loc[t, "asset"].lower() == "equity" and str(meta.loc[t, "instrument_type"]).lower() != "future"
        ]
        severe_dd_flags = compute_severe_drawdown_flags(usd_prices, equity_tickers_dd)
        meta["severe_drawdown"] = pd.Series({t: severe_dd_flags.get(t, False) for t in meta.index})

        if len(tickers) < 2:
            raise ValueError("Need at least 2 equity instruments with returns to optimize.")

        # Covariance
        rets_portfolio = rets[tickers]
        Sigma = rets_portfolio.cov().values
        Sigma = ensure_psd(Sigma, eps=1e-10)
        L = np.linalg.cholesky(Sigma)

        # Build conviction-driven raw weights
        w_raw = _build_conviction_weights(meta, positions_by_ticker).reindex(tickers).fillna(0.0)
        group_metadata = _group_metadata_for_tickers(tickers, positions_by_ticker, w_raw)

        # Scale unconfirmed contrarian positions to 1/3
        for t in tickers:
            if meta.loc[t, "contrarian"] and not meta.loc[t, "contrarian_eligible"]:
                w_raw[t] *= 1.0 / 3.0

        w_raw_vec = w_raw.values

        # Masks
        asset = meta["asset"].str.lower()
        eq_mask = asset.eq("equity").values
        fx_mask = asset.eq("fx").values
        cmdty_mask = asset.eq("commodity").values
        bond_mask = asset.eq("bond").values

        # Betas: restrict regressions to equities plus benchmark tickers.
        (
            beta_by_benchmark,
            beta_display_by_benchmark,
            betas_all_by_benchmark,
            equity_beta_tickers,
        ) = _compute_equity_beta_inputs(rets, tickers, market_tickers, eq_mask)

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
            constraints.append(w[short_mask] <= 0.0)
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

        w_final = w_star * k
        vol_final = port_vol(w_final.values)

        # Benchmark volatility
        benchmark_vol = compute_defense_volatility(usd_prices, market_tickers)
        vol_by_benchmark = {benchmark: benchmark_vol.get(benchmark, np.nan) for benchmark in market_tickers}

        # Hedges
        w_final, hedge_summary = _apply_beta_hedges_with_gross_cap(
            w_final,
            beta_by_benchmark,
            betas_all_by_benchmark,
            long_mask,
            short_mask,
            eq_mask,
            selected_hedges,
        )

        # Strict post-hedge leverage cap: ensure final gross (incl. hedges) <= target leverage.
        if target_leverage is not None:
            effective_target = float(max(0.0, min(target_leverage, GROSS_MAX)))
            tol = 1e-8
            max_iters = 3
            for _ in range(max_iters):
                gross_with_hedges = float(hedge_summary.get("gross_with_hedges", np.abs(w_final).sum()))
                if gross_with_hedges <= effective_target + tol or gross_with_hedges <= 0:
                    break
                scale = effective_target / gross_with_hedges
                w_final = w_final * scale
                w_final, hedge_summary = _apply_beta_hedges_with_gross_cap(
                    w_final,
                    beta_by_benchmark,
                    betas_all_by_benchmark,
                    long_mask,
                    short_mask,
                    eq_mask,
                    selected_hedges,
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
        weights_df = pd.DataFrame(
            {
                "ticker": tickers,
                "asset": meta["asset"].values,
                "instrument_type": meta["instrument_type"].values,
                "price_symbol": meta["price_symbol"].values,
                "quantity": meta["quantity"].values,
                "contract_multiplier": meta["contract_multiplier"].values,
                "fx_base_currency": meta["fx_base_currency"].values if "fx_base_currency" in meta.columns else "",
                "fx_quote_currency": meta["fx_quote_currency"].values if "fx_quote_currency" in meta.columns else "",
                "direction": meta["direction"].values,
                "contrarian": meta["contrarian"].values if "contrarian" in meta.columns else False,
                "conviction": [int((positions_by_ticker.get(t) or {}).get("conviction") or 0) for t in tickers],
                "group_name": group_metadata["group_name"],
                "group_conviction": group_metadata["group_conviction"],
                "group_raw_target": group_metadata["group_raw_target"],
                "group_member_share": group_metadata["group_member_share"],
                **{
                    _beta_metric_name("beta", benchmark): beta_display_by_benchmark[benchmark].values
                    for benchmark in selected_hedges
                },
                "realized_vol": meta["realized_vol"].values,
                "weight": w_final.values,
                "price": latest_prices.values,
            }
        )
        if book is not None:
            weights_df["dollar_weight"] = w_final.values * book
            unit_notional = unit_notional_in_base(weights_df)
            weights_df["quantity"] = (weights_df["dollar_weight"] / unit_notional).round(0).astype("Int64")
            weights_df["target_quantity"] = weights_df["quantity"]
            weights_df["contracts"] = weights_df["quantity"].where(weights_df["instrument_type"].eq("future"))
            weights_df["base_units"] = weights_df["quantity"].where(weights_df["instrument_type"].eq("spot_fx"))
            weights_df["shares"] = weights_df["quantity"]
        weights_df = weights_df.sort_values("weight", ascending=False)

        # Build hedges DataFrame
        hedge_direction_issues: list[str] = []
        hedge_weights = {
            ticker: float(hedge_summary.get(_beta_metric_name("hedge", ticker) + "_weight", 0.0))
            for ticker in selected_hedges
        }
        for ticker in selected_hedges:
            weight = hedge_weights[ticker]
            net_beta = float(hedge_summary.get(_beta_metric_name("net_beta", ticker), 0.0))
            self_beta = float(betas_all_by_benchmark.get(ticker, pd.Series(dtype=float)).get(ticker, BETA_FALLBACK))
            own_adjustment = self_beta * weight
            if abs(net_beta) > 1e-8 and abs(own_adjustment) > 1e-8 and net_beta * own_adjustment > 0:
                hedge_direction_issues.append(
                    f"{ticker} hedge leg ({weight:+.4f}) increases selected pre-hedge beta exposure "
                    f"({net_beta:+.4f}) before cross-hedge effects."
                )
        hedge_direction_warning = (
            "Potential hedge direction mismatch: " + " ".join(hedge_direction_issues)
            if hedge_direction_issues
            else None
        )
        hedge_rows = [
            {
                "ticker": ticker,
                "type": "hedge",
                "direction": "short" if hedge_weights[ticker] < 0 else "long",
                "weight": hedge_weights[ticker],
                "price": float(usd_prices[ticker].iloc[-1]),
            }
            for ticker in selected_hedges
        ]
        hedges_data: dict[str, Any] = {key: [row[key] for row in hedge_rows] for key in hedge_rows[0]}
        if book is not None:
            hedges_data["dollar_weight"] = [float(row["weight"]) * book for row in hedge_rows]
            hedges_data["shares"] = [
                int(round(float(row["weight"]) * book / float(row["price"]))) for row in hedge_rows
            ]
        hedges_df = pd.DataFrame(hedges_data)

        # Load existing hedge positions for delta computation
        existing_hedges = {h["ticker"]: h for h in _get_hedge_positions()}
        hedges_df["current_shares"] = hedges_df["ticker"].map(lambda t: existing_hedges.get(t, {}).get("shares") or 0)
        hedges_df["current_cost_basis"] = hedges_df["ticker"].map(
            lambda t: existing_hedges.get(t, {}).get("cost_basis")
        )
        if "shares" in hedges_df.columns:
            hedges_df["delta_shares"] = hedges_df["shares"] - hedges_df["current_shares"]

        # Max scaled version
        k_max = max_scale_to_respect_linear_caps(w_final, meta, include_position_limits=False)
        w_max_scaled = w_final * k_max
        vol_max_scaled = port_vol(w_max_scaled.values)
        binding = identify_binding_constraint(w_max_scaled, meta, include_position_limits=False)
        exp_max = exposures_by_class(w_max_scaled, meta)

        max_scaled_weights_df = pd.DataFrame(
            {
                "ticker": tickers,
                "asset": meta["asset"].values,
                "instrument_type": meta["instrument_type"].values,
                "price_symbol": meta["price_symbol"].values,
                "contract_multiplier": meta["contract_multiplier"].values,
                "fx_base_currency": meta["fx_base_currency"].values if "fx_base_currency" in meta.columns else "",
                "fx_quote_currency": meta["fx_quote_currency"].values if "fx_quote_currency" in meta.columns else "",
                "direction": meta["direction"].values,
                "conviction": [int((positions_by_ticker.get(t) or {}).get("conviction") or 0) for t in tickers],
                "group_name": group_metadata["group_name"],
                "group_conviction": group_metadata["group_conviction"],
                "group_raw_target": group_metadata["group_raw_target"],
                "group_member_share": group_metadata["group_member_share"],
                "weight": w_max_scaled.values,
                "price": latest_prices.values,
            }
        )
        if book is not None:
            max_scaled_weights_df["dollar_weight"] = w_max_scaled.values * book
            unit_notional = unit_notional_in_base(max_scaled_weights_df)
            max_scaled_weights_df["quantity"] = (
                (max_scaled_weights_df["dollar_weight"] / unit_notional).round(0).astype("Int64")
            )
            max_scaled_weights_df["target_quantity"] = max_scaled_weights_df["quantity"]
            max_scaled_weights_df["contracts"] = max_scaled_weights_df["quantity"].where(
                max_scaled_weights_df["instrument_type"].eq("future")
            )
            max_scaled_weights_df["base_units"] = max_scaled_weights_df["quantity"].where(
                max_scaled_weights_df["instrument_type"].eq("spot_fx")
            )
            max_scaled_weights_df["shares"] = max_scaled_weights_df["quantity"]
        max_scaled_weights_df = max_scaled_weights_df.sort_values("weight", ascending=False)

        benchmark_metrics: dict[str, Any] = {}
        net_betas: dict[str, float] = {}
        post_hedge_betas: dict[str, float] = {}
        benchmark_vols: dict[str, float] = {}
        for benchmark in selected_hedges:
            key = _benchmark_key(benchmark)
            vol = float(vol_by_benchmark.get(benchmark, np.nan))
            net_beta = float(hedge_summary.get(f"net_beta_{key}", 0.0))
            post_hedge_beta = float(hedge_summary.get(f"post_hedge_beta_{key}", 0.0))
            benchmark_vols[benchmark] = vol
            net_betas[benchmark] = net_beta
            post_hedge_betas[benchmark] = post_hedge_beta
            benchmark_metrics[f"vol_{key}"] = vol
            benchmark_metrics[f"beta_long_{key}"] = hedge_summary.get(f"beta_long_{key}", 0.0)
            benchmark_metrics[f"beta_short_{key}"] = hedge_summary.get(f"beta_short_{key}", 0.0)
            benchmark_metrics[f"net_beta_{key}"] = net_beta
            benchmark_metrics[f"post_hedge_beta_{key}"] = post_hedge_beta
            benchmark_metrics[f"hedge_{key}_weight"] = hedge_weights.get(benchmark, 0.0)

        return {
            "status": prob.status,
            "error": None,
            "timestamp": datetime.now(),
            "book_size": book,
            "target_leverage": target_leverage,
            "beta_hedge_mode": beta_hedge_mode,
            "hedge_tickers": list(selected_hedges),
            "selected_hedges": list(selected_hedges),
            "sizing_scope": "equity_only",
            "sizing_asset_classes": ["equity"],
            "excluded_sizing_tickers": excluded_non_equity_tickers,
            # Solution metrics
            "vol_daily": vol_final,
            "gross_leverage": exp["total_gross"],
            # Beta hedging
            **benchmark_metrics,
            "hedge_weights": hedge_weights,
            "net_betas": net_betas,
            "post_hedge_betas": post_hedge_betas,
            "benchmark_vols": benchmark_vols,
            "hedge_direction_warning": hedge_direction_warning,
            "hedge_direction_issues": hedge_direction_issues,
            "beta_scope": "equity_only",
            "beta_asset_classes": ["equity"],
            "beta_tickers": equity_beta_tickers,
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
    beta_hedge_mode: str = BETA_HEDGE_MODE_SPY_IWM,
    hedge_tickers: Sequence[str] | str | None = None,
) -> dict:
    return size_portfolio(
        positions=positions,
        book=book,
        target_leverage=target_leverage,
        beta_hedge_mode=beta_hedge_mode,
        hedge_tickers=hedge_tickers,
    )
