"""
Cross-module signal aggregation service.

This module computes a deterministic market regime score from multiple modules
and builds a hybrid historical regime timeline from modules with native history.

The composite score (0-100) describes current market stress. Higher = more stress.

WEIGHT RATIONALE (validated via 10-year backtest: backtest/signal_backtest.py)
=============================================================================
Backtest period: 2016-03 to 2026-03, 523 weekly observations, all 6 factors.

Factor weights:
  vix        0.20  VIX term structure ratio (VIX3M/VIX). Threshold 18 ≈ 20-yr
                   median VIX; /12 ≈ 1σ above mean. Strong contrarian signal:
                   4-week quintile spread -1.74 (high fear → higher fwd returns).
  breadth    0.20  % of SPX above 200d/20d MA + 20d lows. Strongest contrarian
                   signal: 4-week spread -1.83. Poor breadth → mean reversion.
  liquidity  0.35  US-only FRED composite (net liq, reserves, OAS, NFCI, M2/GDP).
                   ONLY factor with directional predictive power: spread +1.13.
                   Tight liquidity genuinely predicts lower forward returns.
                   Weight increased from 0.20 → 0.35 (absorbed positioning's 0.15).
  sector     0.15  SPDR sector ETF relative perf, 3M change, 200DMA distance.
                   Weak contrarian signal: spread -0.55.
  momentum   0.10  SPX constituent relative ROC bullish ratio. Moderate
                   contrarian signal: spread -1.23.

Regime thresholds:
  The composite score has mean ~20, median ~18, std ~10 over the backtest
  period. The thresholds are intentionally set high so that
  "risk-off" flags are rare and meaningful (genuine stress events like
  COVID-19 Mar 2020). For a more balanced split, use ~15/28 (≈40th/75th
  percentile), but this dilutes the signal.

Predictive interpretation:
  The composite works as a CONTRARIAN indicator. Empirically, "risk-off"
  periods (high composite) have *higher* subsequent SPX returns — classic
  "buy the fear" mean reversion. 4-week forward return spread:
    risk-on: +1.07%  |  transitional: +2.45%  |  risk-off: +10.70%
  The `forward_outlook` field in the output flips this for predictive use:
    elevated composite → "opportunity" (higher expected fwd returns)
    low composite      → "complacent"  (average/lower expected fwd returns)
=============================================================================
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import wait as cf_wait
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from utils.market_freshness import (
    build_market_cache_metadata,
    expected_market_date,
    market_cache_decision,
    metadata_from_decision,
)

DEFAULT_LOOKBACK_WEEKS = 156
DEFAULT_POSITIONING_INSTRUMENTS = "SP500,NASDAQ,RUSSELL,US10Y,EUR"

CONFIGURED_WEIGHTS: dict[str, float] = {
    "vix": 0.20,
    "breadth": 0.20,
    "liquidity": 0.35,
    "sector": 0.15,
    "momentum": 0.10,
}

HISTORY_CAPABLE_FACTORS = {"vix", "liquidity"}
MISSING_HISTORY_FACTORS = {"breadth", "sector", "momentum"}

MODULE_TIMEOUT = 90  # seconds per module in ThreadPoolExecutor
SP500_CHUNK_SIZE = 50
SP500_BATCH_DELAY = 1.0  # seconds between yfinance batch downloads

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Smart staleness cache for S&P 500 prices
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SP500_CACHE_DIR = _REPO_ROOT / "data_cache" / "signal_aggregator"
_SP500_CACHE_DATA = _SP500_CACHE_DIR / "sp500_prices.pkl"
_SP500_CACHE_META = _SP500_CACHE_DIR / "sp500_prices_meta.json"
_SP500_CACHE_TTL_SECONDS = 24 * 60 * 60
_CLOSE_PROBE_TICKER = "SPY"


def _load_sp500_cache() -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """Load cached S&P 500 prices + metadata from disk."""
    try:
        if not _SP500_CACHE_DATA.exists() or not _SP500_CACHE_META.exists():
            return None, None
        meta = json.loads(_SP500_CACHE_META.read_text(encoding="utf-8"))
        if not isinstance(meta, dict) or not isinstance(meta.get("fetched_at"), str):
            return None, None
        with open(_SP500_CACHE_DATA, "rb") as f:
            df = pickle.load(f)  # noqa: S301
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None, None
        return df, meta
    except Exception:
        return None, None


def _save_sp500_cache(df: pd.DataFrame, as_of_date: str | None) -> None:
    """Persist S&P 500 prices + metadata to disk."""
    try:
        _SP500_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_data = _SP500_CACHE_DATA.with_suffix(".tmp")
        tmp_meta = _SP500_CACHE_META.with_suffix(".tmp")
        with open(tmp_data, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        meta = {
            "fetched_at": datetime.now().isoformat(),
            "as_of_date": as_of_date,
            "rows": len(df),
        }
        tmp_meta.write_text(json.dumps(meta), encoding="utf-8")
        tmp_data.replace(_SP500_CACHE_DATA)
        tmp_meta.replace(_SP500_CACHE_META)
    except Exception:
        _log.warning("Failed to write S&P 500 price cache", exc_info=True)


def _touch_sp500_cache_meta(meta: dict[str, Any]) -> None:
    """Refresh the fetched_at timestamp without re-downloading."""
    try:
        meta["fetched_at"] = datetime.now().isoformat()
        _SP500_CACHE_META.write_text(json.dumps(meta), encoding="utf-8")
    except Exception:
        pass


def invalidate_sp500_price_cache() -> None:
    """Delete the shared S&P 500 price cache used by the signal aggregator."""
    for path in (
        _SP500_CACHE_DATA,
        _SP500_CACHE_META,
        _SP500_CACHE_DATA.with_suffix(".tmp"),
        _SP500_CACHE_META.with_suffix(".tmp"),
    ):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            _log.debug("Failed to delete S&P 500 price cache path %s", str(path), exc_info=True)


def _latest_market_close_date() -> str | None:
    """Probe the latest available close date via a lightweight SPY download."""
    from utils.retry import yf_download

    try:
        probe = yf_download(
            tickers=[_CLOSE_PROBE_TICKER],
            period="10d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if probe is None or probe.empty:
            return None
        idx = pd.to_datetime(probe.index, errors="coerce").dropna()
        if idx.empty:
            return None
        return str(idx[-1].date().isoformat())
    except Exception:
        return None


def _sp500_cache_as_of_date(df: pd.DataFrame) -> str | None:
    """Extract the latest date from a price DataFrame's index."""
    try:
        idx = pd.to_datetime(df.index, errors="coerce").dropna()
        if idx.empty:
            return None
        return str(idx[-1].date().isoformat())
    except Exception:
        return None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return []
        records = value.to_dict(orient="records")
        if not isinstance(records, list):
            return []
        normalized: list[dict[str, Any]] = []
        for row in records:
            if isinstance(row, dict):
                normalized.append({str(k): v for k, v in row.items()})
        return normalized
    if isinstance(value, list):
        return [{str(k): v for k, v in row.items()} for row in value if isinstance(row, dict)]
    return []


def _first_row(value: Any) -> dict[str, Any]:
    rows = _as_rows(value)
    if rows:
        return rows[0]
    if isinstance(value, dict):
        return value
    return {}


def _score_vix(vix_data: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    latest = _first_row(vix_data.get("latest_df"))
    ratio = _to_float(latest.get("Ratio"))  # VIX3M / VIX
    spot_vix = _to_float(latest.get("VIX"))

    # 70% weight on term structure ratio, 30% on spot VIX level.
    # Ratio < 1.0 = backwardation (near-term fear > longer-term) → score rises.
    # (1.0 - ratio) / 0.2: ratio of 0.8 → score 1.0, ratio of 1.0 → score 0.0.
    # Spot VIX offset 18.0 ≈ 20-year median VIX (Cboe); /12.0 ≈ 1σ above mean.
    # VIX 30 → score 1.0, VIX 18 → score 0.0.
    parts: list[tuple[float, float]] = []
    if ratio is not None:
        parts.append((70.0, clamp01((1.0 - ratio) / 0.2)))
    if spot_vix is not None:
        parts.append((30.0, clamp01((spot_vix - 18.0) / 12.0)))
    if not parts:
        return None, {"error": "missing ratio and vix level"}

    total_weight = sum(w for w, _ in parts)
    raw = sum(w * c for w, c in parts)
    score = raw * (100.0 / total_weight)
    return score, {"ratio": ratio, "vix": spot_vix}


def _score_breadth(breadth: dict[str, Any], top50: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    inputs = [
        (30.0, _to_float((breadth or {}).get("pct_above_200dma")), lambda v: clamp01((55.0 - v) / 35.0)),
        (20.0, _to_float((breadth or {}).get("pct_above_20dma")), lambda v: clamp01((55.0 - v) / 35.0)),
        (20.0, _to_float((breadth or {}).get("pct_at_20day_low")), lambda v: clamp01((v - 20.0) / 40.0)),
        (15.0, _to_float((top50 or {}).get("pct_below_50dma")), lambda v: clamp01((v - 35.0) / 45.0)),
        (10.0, _to_float((top50 or {}).get("pct_3plus_dist")), lambda v: clamp01((v - 25.0) / 50.0)),
        (5.0, _to_float((top50 or {}).get("pct_broke_20low")), lambda v: clamp01((v - 15.0) / 40.0)),
    ]

    used: list[tuple[float, float]] = []
    for weight, raw_val, transform in inputs:
        if raw_val is None:
            continue
        used.append((weight, transform(raw_val)))

    if not used:
        return None, {"error": "missing breadth inputs"}

    total_weight = sum(w for w, _ in used)
    score = sum(w * c for w, c in used) * (100.0 / total_weight)
    highlights = {
        "pct_above_200dma": _to_float((breadth or {}).get("pct_above_200dma")),
        "pct_above_20dma": _to_float((breadth or {}).get("pct_above_20dma")),
        "pct_at_20day_low": _to_float((breadth or {}).get("pct_at_20day_low")),
        "top50_below_50dma": _to_float((top50 or {}).get("pct_below_50dma")),
        "top50_3plus_dist": _to_float((top50 or {}).get("pct_3plus_dist")),
        "top50_broke_20low": _to_float((top50 or {}).get("pct_broke_20low")),
    }
    return score, highlights


def _score_liquidity(liquidity: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    regime = str((liquidity or {}).get("regime") or "normal").lower()
    composite = _to_float((liquidity or {}).get("composite_score"))
    base = {"ample": 20.0, "normal": 45.0, "tight": 75.0, "stress": 90.0}.get(regime, 45.0)

    if composite is None:
        return None, {"error": "missing composite_score", "regime": regime}

    score = max(0.0, min(100.0, base + (-composite * 10.0)))
    return score, {"regime": regime, "composite_score": composite}


def _has_usable_liquidity_payload(value: Any) -> bool:
    data = _as_dict(value)
    return _to_float(data.get("composite_score")) is not None


def _should_live_fill_liquidity(raw: dict[str, Any], module_status: dict[str, dict[str, Any]]) -> bool:
    if _has_usable_liquidity_payload(raw.get("liquidity")):
        return False
    state = module_status.get("liquidity")
    if not isinstance(state, dict):
        return False
    detail = str(state.get("detail") or "")
    return "Snapshot unavailable" in detail


def _fill_liquidity_snapshot_gap(raw: dict[str, Any], module_status: dict[str, dict[str, Any]]) -> None:
    """Use the live liquidity source when only the durable module snapshot is missing."""
    if not _should_live_fill_liquidity(raw, module_status):
        return

    try:
        from macro.liquidity.liquidity import get_snapshot as get_liquidity_snapshot

        liquidity = get_liquidity_snapshot()
    except Exception as exc:
        module_status["liquidity"] = {
            "status": "error",
            "detail": f"Snapshot unavailable and live liquidity fetch failed: {exc}",
        }
        return

    raw["liquidity"] = liquidity
    if _has_usable_liquidity_payload(liquidity):
        module_status["liquidity"] = {"status": "ok", "detail": "live fallback"}
    else:
        module_status["liquidity"] = {
            "status": "error",
            "detail": "Live liquidity fallback returned no composite_score",
        }


def _score_positioning(rows: Any) -> tuple[float | None, dict[str, Any]]:
    clean_rows = _as_rows(rows)
    if not clean_rows:
        return None, {"error": "no positioning rows"}

    z_components: list[float] = []
    forced_count = 0
    for row in clean_rows:
        z = _to_float(row.get("lf_z"))
        if z is not None:
            z_components.append(clamp01((abs(z) - 1.0) / 2.0))
        forced = row.get("lf_forced")
        if isinstance(forced, str) and forced.strip():
            forced_count += 1

    avg_z = _mean(z_components)
    avg_z = avg_z if avg_z is not None else 0.0
    forced_ratio = _safe_div(float(forced_count), float(len(clean_rows)))

    score = 60.0 * avg_z + 40.0 * forced_ratio
    return score, {"rows": len(clean_rows), "forced_flow_count": forced_count, "avg_abs_z_component": avg_z}


def _score_sector(rows: Any) -> tuple[float | None, dict[str, Any]]:
    clean_rows = _as_rows(rows)
    if not clean_rows:
        return None, {"error": "no sector rows"}

    weighted_sum = 0.0
    total_weight = 0.0
    for row in clean_rows:
        rel_perf = _to_float(row.get("RelPerf_3M_pp"))
        chg = _to_float(row.get("Chg_3M_pp"))
        pct_200 = _to_float(row.get("Pct_Above_200DMA"))
        comp_vals: list[float] = []
        if rel_perf is not None:
            comp_vals.append(clamp01((-rel_perf) / 8.0))
        if chg is not None:
            comp_vals.append(clamp01((-chg) / 1.5))
        if pct_200 is not None:
            comp_vals.append(clamp01((-pct_200) / 12.0))
        if not comp_vals:
            continue

        local = sum(comp_vals) / len(comp_vals)
        raw_weight = _to_float(row.get("Weight_Now"))
        weight = (raw_weight / 100.0) if raw_weight is not None and raw_weight > 0 else 0.0
        if weight <= 0:
            continue

        weighted_sum += local * weight
        total_weight += weight

    if total_weight <= 0:
        return None, {"error": "missing valid sector weights"}

    score = (weighted_sum / total_weight) * 100.0
    return score, {"weighted_sector_count": len(clean_rows), "total_weight_used": total_weight}


def _score_momentum(momentum_data: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    rows = (momentum_data or {}).get("results")
    if not isinstance(rows, list):
        return None, {"error": "missing momentum results"}

    clean_rows = [r for r in rows if isinstance(r, dict)]
    if not clean_rows:
        return None, {"error": "no momentum rows"}

    bullish = 0
    for row in clean_rows:
        avg10 = _to_float(row.get("avg10_rel_roc"))
        rel42 = _to_float(row.get("rel_roc42"))
        if avg10 is not None and rel42 is not None and avg10 > 0 and rel42 > 0:
            bullish += 1

    bullish_ratio = _safe_div(float(bullish), float(len(clean_rows)))
    score = clamp01((0.55 - bullish_ratio) / 0.55) * 100.0
    return score, {"rows": len(clean_rows), "bullish_count": bullish, "bullish_ratio": bullish_ratio}


def _regime_label(score: float) -> str:
    # Thresholds set high so "risk-off" flags are rare and meaningful.
    # At 40/65, ~94% of weeks are risk-on, ~6% transitional, <1% risk-off
    # over the 2016-2026 backtest. This is intentional: risk-off should only
    # fire during genuine stress events (e.g. COVID Mar-2020).
    # For a more balanced split, use ~15/28 (40th/75th percentile).
    if score < 40.0:
        return "risk-on"
    if score < 65.0:
        return "transitional"
    return "risk-off"


def _build_episodes(series_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not series_rows:
        return []

    episodes: list[dict[str, Any]] = []
    start = series_rows[0]
    current_label = str(start.get("label") or "transitional")
    scores = [_to_float(start.get("score")) or 0.0]
    prev_date = str(start.get("date") or "")

    for row in series_rows[1:]:
        label = str(row.get("label") or "transitional")
        if label == current_label:
            scores.append(_to_float(row.get("score")) or 0.0)
            prev_date = str(row.get("date") or prev_date)
            continue

        episodes.append(
            {
                "regime": current_label,
                "start_date": str(start.get("date") or ""),
                "end_date": prev_date,
                "duration_weeks": len(scores),
                "avg_score": round(sum(scores) / len(scores), 2),
            }
        )
        start = row
        current_label = label
        scores = [_to_float(row.get("score")) or 0.0]
        prev_date = str(row.get("date") or "")

    episodes.append(
        {
            "regime": current_label,
            "start_date": str(start.get("date") or ""),
            "end_date": prev_date,
            "duration_weeks": len(scores),
            "avg_score": round(sum(scores) / len(scores), 2),
        }
    )
    return episodes


def _normalize_weekly(series: pd.Series, lookback_weeks: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    if s.empty:
        return s
    idx = pd.to_datetime(s.index, errors="coerce")
    s.index = idx
    s = s[~s.index.isna()]
    if s.empty:
        return s
    s = s.sort_index().resample("W-FRI").last().dropna()
    return s.tail(lookback_weeks)


def _build_vix_history_series(
    lookback_weeks: int,
    preloaded: tuple[pd.DataFrame, str] | None = None,
) -> pd.Series:
    from equities.market_technicals.vix_term_structure import add_signals, load_term_structure

    if preloaded is not None:
        data, _ = preloaded
    else:
        start = (date.today() - timedelta(days=max(lookback_weeks * 7 + 45, 400))).isoformat()
        data, _ = load_term_structure(start)

    signals = add_signals(data, low=1.0, high=1.25)
    if signals.empty:
        return pd.Series(dtype=float)

    ratio_comp = ((1.0 - signals["Ratio"]) / 0.2).clip(lower=0.0, upper=1.0)
    vix_comp = ((signals["VIX"] - 18.0) / 12.0).clip(lower=0.0, upper=1.0)
    series = 70.0 * ratio_comp + 30.0 * vix_comp
    return _normalize_weekly(series, lookback_weeks)


def _build_liquidity_history_series(liquidity_raw: dict[str, Any], lookback_weeks: int) -> pd.Series:
    from macro.liquidity.liquidity import classify_regime

    composite_series = liquidity_raw.get("composite_series")
    if not isinstance(composite_series, pd.Series):
        return pd.Series(dtype=float)

    s = pd.to_numeric(composite_series, errors="coerce").dropna()
    if s.empty:
        return pd.Series(dtype=float)

    values: list[float] = []
    for value in s.values:
        regime, _ = classify_regime(float(value))
        base = {"ample": 20.0, "normal": 45.0, "tight": 75.0, "stress": 90.0}.get(str(regime), 45.0)
        values.append(max(0.0, min(100.0, base + (-float(value) * 10.0))))

    series = pd.Series(values, index=s.index)
    return _normalize_weekly(series, lookback_weeks)


def _build_positioning_history_series(
    lookback_weeks: int,
    instruments_csv: str,
    preloaded_df: pd.DataFrame | None = None,
) -> pd.Series:
    from macro.positioning.positioning import DATASETS, DEFAULT_DOMAIN, INSTRUMENTS, fetch_markets_timeseries

    aliases = [s.strip().upper() for s in (instruments_csv or "").split(",") if s.strip()]
    aliases = aliases or [s.strip() for s in DEFAULT_POSITIONING_INSTRUMENTS.split(",") if s.strip()]
    markets = [INSTRUMENTS[a] for a in aliases if a in INSTRUMENTS]
    if not markets:
        return pd.Series(dtype=float)

    if preloaded_df is not None:
        df = preloaded_df
    else:
        start = (date.today() - timedelta(weeks=lookback_weeks + 12)).isoformat()
        df = fetch_markets_timeseries(
            domain=DEFAULT_DOMAIN,
            dataset_id=DATASETS.get("tff_futures_only", "tff_futures_only"),
            app_token=os.environ.get("SODA_APP_TOKEN") or None,
            markets_exact=markets,
            start=start,
            end=None,
            groups=None,
            z_window=0,
            force_threshold=2.0,
        )
    if df.empty:
        return pd.Series(dtype=float)

    df = df.copy()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.dropna(subset=["report_date"])
    if df.empty:
        return pd.Series(dtype=float)

    scores: list[tuple[pd.Timestamp, float]] = []
    for dt, group in df.groupby("report_date"):
        rows = len(group)
        if rows == 0:
            continue
        z = pd.to_numeric(group.get("lf_z"), errors="coerce").dropna().abs()
        z_component = (((z - 1.0) / 2.0).clip(lower=0.0, upper=1.0).mean()) if not z.empty else 0.0
        forced = group.get("lf_forced")
        forced_count = 0
        if forced is not None:
            forced_count = int(forced.fillna("").astype(str).str.strip().ne("").sum())
        forced_ratio = forced_count / rows
        score = 60.0 * float(z_component) + 40.0 * float(forced_ratio)
        scores.append((pd.Timestamp(dt), score))

    if not scores:
        return pd.Series(dtype=float)

    series = pd.Series({dt: score for dt, score in scores}).sort_index()
    return _normalize_weekly(series, lookback_weeks)


def _build_history(
    lookback_weeks: int,
    instruments_csv: str,
    liquidity_raw: dict[str, Any],
    vix_preloaded: tuple[pd.DataFrame, str] | None = None,
    positioning_preloaded_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    module_status: dict[str, str] = {}
    factors: dict[str, pd.Series] = {}

    history_tasks = {
        "vix": lambda: _build_vix_history_series(lookback_weeks, preloaded=vix_preloaded),
        "liquidity": lambda: _build_liquidity_history_series(liquidity_raw, lookback_weeks),
    }

    with ThreadPoolExecutor(max_workers=len(history_tasks)) as pool:
        futures = {pool.submit(fn): name for name, fn in history_tasks.items()}
        done, not_done = cf_wait(futures.keys(), timeout=MODULE_TIMEOUT)
        for fut in done:
            name = futures[fut]
            try:
                result = fut.result()
                factors[name] = result
                module_status[name] = "ok" if not result.empty else "error"
            except Exception:
                factors[name] = pd.Series(dtype=float)
                module_status[name] = "error"
        for fut in not_done:
            name = futures[fut]
            factors[name] = pd.Series(dtype=float)
            module_status[name] = "timeout"

    available = {k: s for k, s in factors.items() if isinstance(s, pd.Series) and not s.empty}
    if not available:
        return {
            "frequency": "weekly",
            "lookback_weeks": lookback_weeks,
            "coverage": {
                "included_factors": [],
                "missing_factors": sorted(MISSING_HISTORY_FACTORS | HISTORY_CAPABLE_FACTORS),
                "module_status": module_status,
            },
            "series": [],
            "episodes": [],
            "scores": [],
        }

    merged = pd.concat(available, axis=1).sort_index()
    merged = merged.tail(lookback_weeks)

    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for idx, row in merged.iterrows():
        weighted = 0.0
        total_w = 0.0
        factor_values: dict[str, float] = {}
        for factor in ("vix", "liquidity"):
            value = _to_float(row.get(factor))
            if value is None:
                continue
            w = CONFIGURED_WEIGHTS.get(factor, 0.0)
            weighted += w * value
            total_w += w
            factor_values[factor] = round(value, 2)
        if total_w <= 0:
            continue
        composite = weighted / total_w
        label = _regime_label(composite)
        rows.append(
            {
                "date": pd.Timestamp(idx).date().isoformat(),
                "score": round(composite, 2),
                "label": label,
                "factors": factor_values,
            }
        )
        scores.append(float(composite))

    return {
        "frequency": "weekly",
        "lookback_weeks": lookback_weeks,
        "coverage": {
            "included_factors": sorted([k for k in HISTORY_CAPABLE_FACTORS if module_status.get(k) == "ok"]),
            "missing_factors": sorted(
                MISSING_HISTORY_FACTORS | {k for k in HISTORY_CAPABLE_FACTORS if module_status.get(k) != "ok"}
            ),
            "module_status": module_status,
        },
        "series": rows,
        "episodes": _build_episodes(rows),
        "scores": scores,
    }


def _download_sp500_prices_uncached() -> pd.DataFrame:
    """Download S&P 500 constituent prices from yfinance (no cache)."""
    from equities.market_technicals.market_breadth import get_sp500_tickers
    from utils.retry import yf_download

    tickers = get_sp500_tickers()
    chunks = [tickers[i : i + SP500_CHUNK_SIZE] for i in range(0, len(tickers), SP500_CHUNK_SIZE)]
    all_data: list[pd.DataFrame] = []

    for idx, chunk in enumerate(chunks, 1):
        _log.info("S&P 500 shared download batch %d/%d (%d tickers)", idx, len(chunks), len(chunk))
        try:
            df = yf_download(
                tickers=chunk,
                period="2y",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if df is not None and not df.empty:
                all_data.append(df)
            if idx < len(chunks):
                time.sleep(SP500_BATCH_DELAY)
        except Exception:
            _log.warning("S&P 500 batch %d failed, skipping", idx)

    if not all_data:
        return pd.DataFrame()

    if len(all_data) == 1:
        return all_data[0]

    # Merge chunks while preserving MultiIndex structure
    fields: set[str] = set()
    for df in all_data:
        if isinstance(df.columns, pd.MultiIndex):
            fields.update(df.columns.get_level_values(0).unique().tolist())
        else:
            fields.update(df.columns.tolist())

    merged: dict[str, pd.DataFrame] = {}
    for field in fields:
        parts: list[pd.DataFrame] = []
        for df in all_data:
            if isinstance(df.columns, pd.MultiIndex):
                if field in df.columns.get_level_values(0):
                    parts.append(df[field])
            elif field in df.columns:
                parts.append(df[[field]])
        if parts:
            merged[field] = pd.concat(parts, axis=1)

    if not merged:
        return pd.DataFrame()
    return pd.concat(merged, axis=1)


def _download_sp500_prices_with_meta() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download S&P 500 prices with smart staleness caching.

    Cache strategy (mirrors market_breadth.py pattern):
    1. If disk cache is current for the expected market date → return cached
    2. If cache is older but market hasn't updated since cached as_of_date
       → refresh TTL and return cached (avoids re-downloading on weekends/holidays)
    3. On fresh download failure → fall back to cached data (graceful degradation)
    4. On success → write new cache to disk
    """
    cached_df, cached_meta = _load_sp500_cache()
    cache_decision = None

    if cached_df is not None and cached_meta is not None:
        cached_as_of = cached_meta.get("as_of_date")
        fetched_at = cached_meta.get("fetched_at")
        cache_decision = market_cache_decision(
            cached_as_of=cached_as_of,
            fetched_at=fetched_at,
            ttl_seconds=_SP500_CACHE_TTL_SECONDS,
        )
        if cache_decision.action == "probe":
            latest_close = _latest_market_close_date()
            cache_decision = market_cache_decision(
                cached_as_of=cached_as_of,
                fetched_at=fetched_at,
                ttl_seconds=_SP500_CACHE_TTL_SECONDS,
                latest_close=latest_close,
                latest_close_probed=True,
            )
        if cache_decision.action == "use_cache":
            if cache_decision.status == "hit_unchanged":
                _log.info(
                    "S&P 500 prices: cache older than expected but latest close unchanged (cached=%s, latest=%s), reusing",
                    cache_decision.cached_as_of,
                    cache_decision.latest_close,
                )
                _touch_sp500_cache_meta(cached_meta)
            elif cache_decision.status == "stale_fallback":
                _log.warning("S&P 500 prices: using stale cache fallback (%s)", cache_decision.reason)
            else:
                _log.info(
                    "S&P 500 prices: serving current disk cache (cached=%s, expected=%s)",
                    cache_decision.cached_as_of,
                    cache_decision.expected_market_date,
                )
            return cached_df, cache_decision.metadata()

        _log.info(
            "S&P 500 prices: cache stale (cached=%s, expected=%s, latest=%s), re-downloading",
            cache_decision.cached_as_of,
            cache_decision.expected_market_date,
            cache_decision.latest_close,
        )

    # Fresh download
    try:
        df = _download_sp500_prices_uncached()
    except Exception as exc:
        if cached_df is not None:
            _log.info(
                "S&P 500 prices: fresh download failed, falling back to stale cache",
                exc_info=True,
            )
            if cache_decision is not None:
                return cached_df, metadata_from_decision(
                    cache_decision,
                    status="stale_fallback",
                    stale=True,
                    reason=f"refresh failed: {exc}",
                )
            return cached_df, build_market_cache_metadata(
                status="stale_fallback",
                stale=True,
                cached_as_of=cached_meta.get("as_of_date") if cached_meta else None,
                reason=f"refresh failed: {exc}",
                cache_ttl_seconds=_SP500_CACHE_TTL_SECONDS,
            )
        raise

    if df.empty:
        if cached_df is not None:
            _log.warning("S&P 500 prices: fresh download failed, falling back to stale cache")
            if cache_decision is not None:
                return cached_df, metadata_from_decision(
                    cache_decision,
                    status="stale_fallback",
                    stale=True,
                    reason="refresh returned empty data",
                )
            return cached_df, build_market_cache_metadata(
                status="stale_fallback",
                stale=True,
                cached_as_of=cached_meta.get("as_of_date") if cached_meta else None,
                reason="refresh returned empty data",
                cache_ttl_seconds=_SP500_CACHE_TTL_SECONDS,
            )
        return df, build_market_cache_metadata(
            status="miss",
            stale=True,
            reason="refresh returned empty data and no cache was available",
            cache_ttl_seconds=_SP500_CACHE_TTL_SECONDS,
        )

    as_of = _sp500_cache_as_of_date(df)
    _save_sp500_cache(df, as_of)
    _log.info("S&P 500 prices: fresh download complete (rows=%d, as_of=%s)", len(df), as_of)
    return df, build_market_cache_metadata(
        status="refresh",
        stale=False,
        cached_as_of=as_of,
        expected_market_date_value=expected_market_date().isoformat(),
        latest_close=cache_decision.latest_close if cache_decision is not None else None,
        reason="refreshed S&P 500 price cache",
        cache_ttl_seconds=_SP500_CACHE_TTL_SECONDS,
    )


def _fetch_current_modules(
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    from equities.market_technicals.market_breadth import get_data as get_market_breadth
    from equities.market_technicals.top50_breadth import get_data as get_top50_breadth
    from equities.market_technicals.vix_term_structure import add_signals, load_term_structure
    from equities.sector_metrics.sector_metrics import get_data as get_sector_metrics_data
    from macro.liquidity.liquidity import get_snapshot as get_liquidity_snapshot
    from portfolio.momentum.price_momentum.momentum import get_data as get_momentum_data

    raw: dict[str, Any] = {}
    module_status: dict[str, dict[str, Any]] = {}

    # ── Phase 1: Shared S&P 500 price download (serial) ──────────────
    # This replaces 3 separate concurrent yfinance downloads that caused
    # rate-limiting and 401 errors.
    sp500_prices, sp500_market_cache = _download_sp500_prices_with_meta()
    sp500_market_cache = dict(sp500_market_cache)
    prices_arg = sp500_prices if not sp500_prices.empty else None
    _log.info("Shared S&P 500 download complete (empty=%s)", sp500_prices.empty)

    # ── Pre-compute VIX parameters ──────────────────────────────────
    # VIX: use wider lookback so the same data serves current + history
    vix_start = (date.today() - timedelta(days=max(lookback_weeks * 7 + 45, 540))).isoformat()

    # ── VIX combined task (current + history data in one fetch) ───────
    def _vix_task() -> dict[str, Any]:
        data, used_vix3m = load_term_structure(vix_start)
        signals = add_signals(data, low=1.0, high=1.25)
        if signals.empty:
            return {
                "vix_term_structure": {
                    "latest_df": pd.DataFrame(),
                    "recent_df": pd.DataFrame(),
                    "hits_df": pd.DataFrame(),
                },
                "vix_raw_ts": (data, used_vix3m),
            }
        latest = signals.iloc[-1]
        latest_df = pd.DataFrame(
            [
                {
                    "Date": latest.name.date().isoformat(),
                    "VIX": float(latest["VIX"]),
                    "VIX3M": float(latest["VIX3M"]),
                    "Ratio": float(latest["Ratio"]),
                    "Signal": str(latest["Signal"]),
                    "UsedTicker": used_vix3m,
                }
            ]
        )
        recent_df = signals.tail(320).copy()
        if not recent_df.empty:
            recent_df = recent_df.reset_index().rename(columns={"index": "Date"})
            recent_df["Date"] = pd.to_datetime(recent_df["Date"]).dt.date.astype(str)
            recent_df["UsedTicker"] = used_vix3m
        hits_df = signals[signals["Signal"] != "Neutral"].copy()
        if not hits_df.empty:
            hits_df = hits_df.sort_index(ascending=False).head(20)
            hits_df = hits_df.reset_index().rename(columns={"index": "Date"})
            hits_df["Date"] = pd.to_datetime(hits_df["Date"]).dt.date.astype(str)
            hits_df["UsedTicker"] = used_vix3m
        return {
            "vix_term_structure": {"latest_df": latest_df, "recent_df": recent_df, "hits_df": hits_df},
            "vix_raw_ts": (data, used_vix3m),
        }

    # ── Phase 2: All modules in parallel ──────────────────────────────
    tasks: dict[str, Any] = {
        "vix_combined": _vix_task,
        "market_breadth": lambda: get_market_breadth(prices_df=prices_arg),
        "top50_breadth": lambda: get_top50_breadth(prices_df=prices_arg),
        "liquidity": get_liquidity_snapshot,
        "sector_metrics": lambda: get_sector_metrics_data(prices_df=prices_arg),
        "momentum": get_momentum_data,
    }

    # Map combined task names → canonical module keys
    _COMBINED_KEYS = {
        "vix_combined": ("vix_term_structure", "vix_raw_ts"),
    }

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        done, not_done = cf_wait(futures.keys(), timeout=MODULE_TIMEOUT)

        for fut in done:
            name = futures[fut]
            try:
                result = fut.result()
                if name in _COMBINED_KEYS:
                    key_main, key_extra = _COMBINED_KEYS[name]
                    raw[key_main] = result[key_main]
                    raw[key_extra] = result[key_extra]
                    module_status[key_main] = {"status": "ok"}
                else:
                    raw[name] = result
                    module_status[name] = {"status": "ok"}
                    if name in {"market_breadth", "top50_breadth", "sector_metrics"}:
                        module_status[name]["market_cache"] = sp500_market_cache
            except Exception as exc:
                if name in _COMBINED_KEYS:
                    key_main, _ = _COMBINED_KEYS[name]
                    raw[key_main] = None
                    module_status[key_main] = {"status": "error", "detail": str(exc)}
                else:
                    raw[name] = None
                    module_status[name] = {"status": "error", "detail": str(exc)}
                    if name in {"market_breadth", "top50_breadth", "sector_metrics"}:
                        module_status[name]["market_cache"] = sp500_market_cache

        for fut in not_done:
            name = futures[fut]
            _log.warning("Module %s timed out after %ds", name, MODULE_TIMEOUT)
            if name in _COMBINED_KEYS:
                key_main, _ = _COMBINED_KEYS[name]
                raw[key_main] = None
                module_status[key_main] = {"status": "error", "detail": "timeout"}
            else:
                raw[name] = None
                module_status[name] = {"status": "error", "detail": "timeout"}
                if name in {"market_breadth", "top50_breadth", "sector_metrics"}:
                    module_status[name]["market_cache"] = sp500_market_cache

    return raw, module_status


def build_signal_aggregator_from_payloads(
    raw: dict[str, Any],
    module_status: dict[str, dict[str, Any]],
    *,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    positioning_instruments: str = DEFAULT_POSITIONING_INSTRUMENTS,
    include_raw_modules: bool = False,
    include_history: bool = True,
) -> dict[str, Any]:
    """Build a signal-aggregator response from already-fetched module payloads."""
    lookback = max(26, min(int(lookback_weeks), 520))
    _fill_liquidity_snapshot_gap(raw, module_status)

    vix_data = _as_dict(raw.get("vix_term_structure"))
    breadth_data = _as_dict(raw.get("market_breadth"))
    top50_data = _as_dict(raw.get("top50_breadth"))
    liquidity_data = _as_dict(raw.get("liquidity"))
    sector_data = _as_dict(raw.get("sector_metrics"))
    momentum_data = _as_dict(raw.get("momentum"))

    # Pre-fetched data for history reuse (avoids redundant network calls)
    vix_preloaded = raw.get("vix_raw_ts")  # tuple[DataFrame, str] | None

    factor_builders = {
        "vix": lambda: _score_vix(vix_data),
        "breadth": lambda: _score_breadth(breadth_data, top50_data),
        "liquidity": lambda: _score_liquidity(liquidity_data),
        "sector": lambda: _score_sector(sector_data.get("weights_df")),
        "momentum": lambda: _score_momentum(momentum_data),
    }
    # NOTE: positioning removed — near-zero predictive power (backtest corr=-0.004).
    # Weight reallocated to liquidity (the only directional predictor).

    factors: list[dict[str, Any]] = []
    valid_scores: dict[str, float] = {}
    failed_modules: list[str] = []

    for key, fn in factor_builders.items():
        try:
            score, highlights = fn()
        except Exception as exc:
            score, highlights = None, {"error": str(exc)}
        status = "ok" if score is not None else "error"
        if score is None:
            failed_modules.append(key)
        else:
            valid_scores[key] = float(max(0.0, min(100.0, score)))

        factors.append(
            {
                "key": key,
                "status": status,
                "score": None if score is None else round(float(score), 2),
                "weight": 0.0,
                "contribution": 0.0,
                "highlights": highlights,
            }
        )

    total_configured_available = sum(CONFIGURED_WEIGHTS[k] for k in valid_scores)
    if total_configured_available <= 0:
        raise RuntimeError("All factor computations failed; no composite regime can be computed.")

    for factor in factors:
        key = factor["key"]
        if key in valid_scores:
            effective_weight = CONFIGURED_WEIGHTS[key] / total_configured_available
            contribution = effective_weight * valid_scores[key]
            factor["weight"] = round(effective_weight, 4)
            factor["contribution"] = round(contribution, 2)
        else:
            factor["weight"] = 0.0
            factor["contribution"] = 0.0

    composite = sum(float(f["contribution"]) for f in factors)
    label = _regime_label(composite)
    status = "ok" if len(valid_scores) == len(CONFIGURED_WEIGHTS) else "degraded"

    if include_history:
        history = _build_history(
            lookback,
            positioning_instruments,
            liquidity_data,
            vix_preloaded=vix_preloaded,
        )
    else:
        history = {
            "frequency": "weekly",
            "lookback_weeks": lookback,
            "coverage": {
                "included_factors": [],
                "missing_factors": sorted(MISSING_HISTORY_FACTORS | HISTORY_CAPABLE_FACTORS),
                "module_status": {"history": "skipped"},
            },
            "series": [],
            "episodes": [],
            "scores": [],
        }
    history_scores = [float(s) for s in history.get("scores", [])]
    history_pct = None
    if history_scores:
        below_or_equal = sum(1 for s in history_scores if s <= composite)
        history_pct = round((below_or_equal / len(history_scores)) * 100.0, 2)

    candidate_dates: list[pd.Timestamp] = []
    vix_latest = _first_row(vix_data.get("latest_df"))
    vix_date = vix_latest.get("Date")
    if isinstance(vix_date, str):
        candidate_dates.append(pd.to_datetime(vix_date, errors="coerce"))
    liq_date = liquidity_data.get("latest_date")
    if liq_date is not None:
        candidate_dates.append(pd.to_datetime(liq_date, errors="coerce"))
    sector_ts = sector_data.get("timestamp")
    if sector_ts is not None:
        candidate_dates.append(pd.to_datetime(sector_ts, errors="coerce"))

    candidate_dates = [d for d in candidate_dates if not pd.isna(d)]
    as_of = max(candidate_dates).date().isoformat() if candidate_dates else date.today().isoformat()

    confidence = round(total_configured_available, 4)

    # Contrarian forward outlook (validated by 10-year backtest)
    # Higher composite → historically higher subsequent SPX returns (mean reversion)
    if composite >= 35.0:
        fwd_outlook = "opportunity"
        fwd_outlook_detail = "Elevated stress historically precedes above-average forward returns"
    elif composite >= 22.0:
        fwd_outlook = "neutral"
        fwd_outlook_detail = "Moderate stress; forward return expectations near baseline"
    else:
        fwd_outlook = "complacent"
        fwd_outlook_detail = "Low stress / complacency; forward returns historically average or below"

    response: dict[str, Any] = {
        "status": status,
        "as_of": as_of,
        "regime": {
            "label": label,
            "score": round(composite, 2),
            "confidence": confidence,
            "history_percentile": history_pct,
        },
        "forward_outlook": {
            "label": fwd_outlook,
            "detail": fwd_outlook_detail,
            "basis": "contrarian (10-year backtest: high stress → higher fwd returns)",
        },
        "weights": {
            "configured": CONFIGURED_WEIGHTS,
            "effective": {f["key"]: f["weight"] for f in factors},
        },
        "factors": factors,
        "module_status": module_status,
        "failed_modules": sorted(set(failed_modules)),
        "history": {
            "frequency": history.get("frequency"),
            "lookback_weeks": history.get("lookback_weeks"),
            "coverage": history.get("coverage"),
            "series": history.get("series"),
            "episodes": history.get("episodes"),
        },
    }

    if include_raw_modules:
        # Exclude internal preloaded keys and large series from raw output
        _internal_keys = {"vix_raw_ts"}
        raw_modules = {}
        for key, value in raw.items():
            if key in _internal_keys:
                continue
            if key == "liquidity" and isinstance(value, dict):
                raw_modules[key] = {k: v for k, v in value.items() if k not in ("df_weekly", "composite_series")}
            else:
                raw_modules[key] = value
        response["raw_modules"] = raw_modules

    return response


def build_signal_aggregator(
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    positioning_instruments: str = DEFAULT_POSITIONING_INSTRUMENTS,
    include_raw_modules: bool = False,
    include_history: bool = True,
) -> dict[str, Any]:
    lookback = max(26, min(int(lookback_weeks), 520))
    raw, module_status = _fetch_current_modules(lookback_weeks=lookback)
    return build_signal_aggregator_from_payloads(
        raw,
        module_status,
        lookback_weeks=lookback,
        positioning_instruments=positioning_instruments,
        include_raw_modules=include_raw_modules,
        include_history=include_history,
    )
