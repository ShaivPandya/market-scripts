"""
Cross-module signal aggregation service.

This module computes a deterministic market regime score from multiple modules
and builds a hybrid historical regime timeline from modules with native history.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import wait as cf_wait
from datetime import date, timedelta
from typing import Any

import pandas as pd

DEFAULT_LOOKBACK_WEEKS = 156
DEFAULT_POSITIONING_INSTRUMENTS = "SP500,NASDAQ,RUSSELL,US10Y,EUR"

CONFIGURED_WEIGHTS: dict[str, float] = {
    "vix": 0.20,
    "breadth": 0.20,
    "liquidity": 0.20,
    "positioning": 0.15,
    "sector": 0.15,
    "momentum": 0.10,
}

HISTORY_CAPABLE_FACTORS = {"vix", "liquidity", "positioning"}
MISSING_HISTORY_FACTORS = {"breadth", "sector", "momentum"}

MODULE_TIMEOUT = 90  # seconds per module in ThreadPoolExecutor
SP500_CHUNK_SIZE = 50
SP500_BATCH_DELAY = 1.0  # seconds between yfinance batch downloads

_log = logging.getLogger(__name__)


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
    ratio = _to_float(latest.get("Ratio"))
    spot_vix = _to_float(latest.get("VIX"))

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
    from vix_term_structure import add_signals, load_term_structure

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
    from liquidity import classify_regime

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
    from positioning import DATASETS, DEFAULT_DOMAIN, INSTRUMENTS, fetch_markets_timeseries

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
        "positioning": lambda: _build_positioning_history_series(
            lookback_weeks, instruments_csv, preloaded_df=positioning_preloaded_df
        ),
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
        for factor in ("vix", "liquidity", "positioning"):
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


def _parse_instrument_csv(instruments_csv: str) -> str:
    from positioning import INSTRUMENTS

    aliases = [s.strip().upper() for s in (instruments_csv or "").split(",") if s.strip()]
    if not aliases:
        aliases = [s.strip().upper() for s in DEFAULT_POSITIONING_INSTRUMENTS.split(",") if s.strip()]
    keep = [a for a in aliases if a in INSTRUMENTS]
    if not keep:
        keep = [s.strip().upper() for s in DEFAULT_POSITIONING_INSTRUMENTS.split(",") if s.strip()]
    return ",".join(keep)


def _download_sp500_prices() -> pd.DataFrame:
    """Download S&P 500 constituent prices once for all modules."""
    import yfinance as yf
    from market_breadth import get_sp500_tickers

    tickers = get_sp500_tickers()
    chunks = [tickers[i : i + SP500_CHUNK_SIZE] for i in range(0, len(tickers), SP500_CHUNK_SIZE)]
    all_data: list[pd.DataFrame] = []

    for idx, chunk in enumerate(chunks, 1):
        _log.info("S&P 500 shared download batch %d/%d (%d tickers)", idx, len(chunks), len(chunk))
        try:
            df = yf.download(
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


def _fetch_current_modules(
    positioning_instruments_csv: str,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    from liquidity import get_snapshot as get_liquidity_snapshot
    from market_breadth import get_data as get_market_breadth
    from momentum import get_data as get_momentum_data
    from positioning import DATASETS, DEFAULT_DOMAIN, INSTRUMENTS, fetch_markets_timeseries
    from sector_metrics import get_data as get_sector_metrics_data
    from top50_breadth import get_data as get_top50_breadth
    from vix_term_structure import add_signals, load_term_structure

    raw: dict[str, Any] = {}
    module_status: dict[str, dict[str, Any]] = {}

    # ── Phase 1: Shared S&P 500 price download (serial) ──────────────
    # This replaces 3 separate concurrent yfinance downloads that caused
    # rate-limiting and 401 errors.
    sp500_prices = _download_sp500_prices()
    prices_arg = sp500_prices if not sp500_prices.empty else None
    _log.info("Shared S&P 500 download complete (empty=%s)", sp500_prices.empty)

    # ── Pre-compute VIX and positioning parameters ────────────────────
    # VIX: use wider lookback so the same data serves current + history
    vix_start = (date.today() - timedelta(days=max(lookback_weeks * 7 + 45, 540))).isoformat()

    # Positioning: resolve instrument aliases once
    pos_aliases = [s.strip().upper() for s in (positioning_instruments_csv or "").split(",") if s.strip()]
    pos_aliases = pos_aliases or [s.strip().upper() for s in DEFAULT_POSITIONING_INSTRUMENTS.split(",") if s.strip()]
    alias_to_market = {a: INSTRUMENTS[a] for a in pos_aliases if a in INSTRUMENTS}
    pos_markets = list(alias_to_market.values())

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

    # ── Positioning combined task (current + history in one fetch) ────
    def _positioning_task() -> dict[str, Any]:
        if not pos_markets:
            return {"positioning": [], "positioning_df": pd.DataFrame()}
        df_all = fetch_markets_timeseries(
            domain=DEFAULT_DOMAIN,
            dataset_id=DATASETS.get("tff_futures_only", "tff_futures_only"),
            app_token=os.environ.get("SODA_APP_TOKEN") or None,
            markets_exact=pos_markets,
            start="2015-01-01",
            end=None,
            groups=None,
            z_window=0,
            force_threshold=2.0,
        )
        # Extract latest row per instrument (replicates fetch_multiple_instruments)
        results: list[dict[str, Any]] = []
        for alias, market_name in alias_to_market.items():
            try:
                mdf = df_all[df_all["market_and_exchange_names"] == market_name]
                row = mdf.dropna(subset=["report_date"]).iloc[-1]
                results.append(
                    {
                        "instrument": alias,
                        "report_date": row["report_date"],
                        "lf_net": row["lf_net"],
                        "lf_net_pct_oi": row.get("lf_net_pct_oi"),
                        "lf_z": row.get("lf_z"),
                        "lf_deleveraging_z": row.get("lf_deleveraging_z"),
                        "lf_forced": row.get("lf_forced"),
                    }
                )
            except Exception:
                pass
        return {"positioning": results, "positioning_df": df_all}

    # ── Phase 2: All modules in parallel ──────────────────────────────
    tasks: dict[str, Any] = {
        "vix_combined": _vix_task,
        "market_breadth": lambda: get_market_breadth(prices_df=prices_arg),
        "top50_breadth": lambda: get_top50_breadth(prices_df=prices_arg),
        "liquidity": get_liquidity_snapshot,
        "sector_metrics": lambda: get_sector_metrics_data(prices_df=prices_arg),
        "momentum": get_momentum_data,
        "positioning_combined": _positioning_task,
    }

    # Map combined task names → canonical module keys
    _COMBINED_KEYS = {
        "vix_combined": ("vix_term_structure", "vix_raw_ts"),
        "positioning_combined": ("positioning", "positioning_df"),
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
            except Exception as exc:
                if name in _COMBINED_KEYS:
                    key_main, _ = _COMBINED_KEYS[name]
                    raw[key_main] = None
                    module_status[key_main] = {"status": "error", "detail": str(exc)}
                else:
                    raw[name] = None
                    module_status[name] = {"status": "error", "detail": str(exc)}

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

    return raw, module_status


def build_signal_aggregator(
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    positioning_instruments: str = DEFAULT_POSITIONING_INSTRUMENTS,
    include_raw_modules: bool = False,
) -> dict[str, Any]:
    lookback = max(26, min(int(lookback_weeks), 520))
    instruments_csv = _parse_instrument_csv(positioning_instruments)

    raw, module_status = _fetch_current_modules(instruments_csv, lookback_weeks=lookback)
    vix_data = _as_dict(raw.get("vix_term_structure"))
    breadth_data = _as_dict(raw.get("market_breadth"))
    top50_data = _as_dict(raw.get("top50_breadth"))
    liquidity_data = _as_dict(raw.get("liquidity"))
    sector_data = _as_dict(raw.get("sector_metrics"))
    momentum_data = _as_dict(raw.get("momentum"))
    positioning_rows = raw.get("positioning")

    # Pre-fetched data for history reuse (avoids redundant network calls)
    vix_preloaded = raw.get("vix_raw_ts")  # tuple[DataFrame, str] | None
    positioning_preloaded_df = raw.get("positioning_df")  # DataFrame | None

    factor_builders = {
        "vix": lambda: _score_vix(vix_data),
        "breadth": lambda: _score_breadth(breadth_data, top50_data),
        "liquidity": lambda: _score_liquidity(liquidity_data),
        "positioning": lambda: _score_positioning(positioning_rows),
        "sector": lambda: _score_sector(sector_data.get("weights_df")),
        "momentum": lambda: _score_momentum(momentum_data),
    }

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

    history = _build_history(
        lookback,
        instruments_csv,
        liquidity_data,
        vix_preloaded=vix_preloaded,
        positioning_preloaded_df=positioning_preloaded_df,
    )
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

    pos_rows = _as_rows(positioning_rows)
    if pos_rows:
        dates = [pd.to_datetime(r.get("report_date"), errors="coerce") for r in pos_rows]
        dates = [d for d in dates if not pd.isna(d)]
        if dates:
            candidate_dates.append(max(dates))

    candidate_dates = [d for d in candidate_dates if not pd.isna(d)]
    as_of = max(candidate_dates).date().isoformat() if candidate_dates else date.today().isoformat()

    confidence = round(total_configured_available, 4)
    response: dict[str, Any] = {
        "status": status,
        "as_of": as_of,
        "regime": {
            "label": label,
            "score": round(composite, 2),
            "confidence": confidence,
            "history_percentile": history_pct,
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
        _internal_keys = {"vix_raw_ts", "positioning_df"}
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
