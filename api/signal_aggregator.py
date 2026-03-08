"""
Cross-module signal aggregation service.

This module computes a deterministic market regime score from multiple modules
and builds a hybrid historical regime timeline from modules with native history.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _score_vix(vix_data: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    latest = ((vix_data or {}).get("latest_df") or [{}])[0]
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


def _score_positioning(rows: list[dict[str, Any]]) -> tuple[float | None, dict[str, Any]]:
    clean_rows = [r for r in rows if isinstance(r, dict)]
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


def _score_sector(rows: list[dict[str, Any]]) -> tuple[float | None, dict[str, Any]]:
    clean_rows = [r for r in rows if isinstance(r, dict)]
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


def _build_vix_history_series(lookback_weeks: int) -> pd.Series:
    from vix_term_structure import add_signals, load_term_structure

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


def _build_positioning_history_series(lookback_weeks: int, instruments_csv: str) -> pd.Series:
    from positioning import DATASETS, DEFAULT_DOMAIN, INSTRUMENTS, fetch_markets_timeseries

    aliases = [s.strip().upper() for s in (instruments_csv or "").split(",") if s.strip()]
    aliases = aliases or [s.strip() for s in DEFAULT_POSITIONING_INSTRUMENTS.split(",") if s.strip()]
    markets = [INSTRUMENTS[a] for a in aliases if a in INSTRUMENTS]
    if not markets:
        return pd.Series(dtype=float)

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


def _build_history(lookback_weeks: int, instruments_csv: str, liquidity_raw: dict[str, Any]) -> dict[str, Any]:
    module_status: dict[str, str] = {}
    factors: dict[str, pd.Series] = {}

    history_tasks = {
        "vix": lambda: _build_vix_history_series(lookback_weeks),
        "liquidity": lambda: _build_liquidity_history_series(liquidity_raw, lookback_weeks),
        "positioning": lambda: _build_positioning_history_series(lookback_weeks, instruments_csv),
    }

    with ThreadPoolExecutor(max_workers=len(history_tasks)) as pool:
        futures = {pool.submit(fn): name for name, fn in history_tasks.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                result = fut.result()
                factors[name] = result
                module_status[name] = "ok" if not result.empty else "error"
            except Exception:
                factors[name] = pd.Series(dtype=float)
                module_status[name] = "error"

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


def _fetch_current_modules(positioning_instruments_csv: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    from liquidity import get_snapshot as get_liquidity_snapshot
    from market_breadth import get_data as get_market_breadth
    from momentum import get_data as get_momentum_data
    from positioning import DATASETS, DEFAULT_DOMAIN, fetch_multiple_instruments
    from sector_metrics import get_data as get_sector_metrics_data
    from top50_breadth import get_data as get_top50_breadth
    from vix_term_structure import get_data as get_vix_data

    raw: dict[str, Any] = {}
    module_status: dict[str, dict[str, Any]] = {}

    tasks: dict[str, Any] = {
        "vix_term_structure": lambda: get_vix_data(start=(date.today() - timedelta(days=540)).isoformat(), tail=320),
        "market_breadth": get_market_breadth,
        "top50_breadth": get_top50_breadth,
        "liquidity": get_liquidity_snapshot,
        "sector_metrics": get_sector_metrics_data,
        "momentum": get_momentum_data,
        "positioning": lambda: fetch_multiple_instruments(
            domain=DEFAULT_DOMAIN,
            dataset_id=DATASETS.get("tff_futures_only", "tff_futures_only"),
            app_token=os.environ.get("SODA_APP_TOKEN") or None,
            instruments=[s.strip() for s in positioning_instruments_csv.split(",") if s.strip()],
            start="2015-01-01",
            end=None,
            groups=None,
            z_window=0,
            force_threshold=2.0,
        ),
    }

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                raw[name] = fut.result()
                module_status[name] = {"status": "ok"}
            except Exception as exc:
                raw[name] = None
                module_status[name] = {"status": "error", "detail": str(exc)}

    return raw, module_status


def build_signal_aggregator(
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    positioning_instruments: str = DEFAULT_POSITIONING_INSTRUMENTS,
    include_raw_modules: bool = False,
) -> dict[str, Any]:
    lookback = max(26, min(int(lookback_weeks), 520))
    instruments_csv = _parse_instrument_csv(positioning_instruments)

    raw, module_status = _fetch_current_modules(instruments_csv)

    factor_builders = {
        "vix": lambda: _score_vix(raw.get("vix_term_structure") or {}),
        "breadth": lambda: _score_breadth(raw.get("market_breadth") or {}, raw.get("top50_breadth") or {}),
        "liquidity": lambda: _score_liquidity(raw.get("liquidity") or {}),
        "positioning": lambda: _score_positioning(raw.get("positioning") or []),
        "sector": lambda: _score_sector((raw.get("sector_metrics") or {}).get("weights_df") or []),
        "momentum": lambda: _score_momentum(raw.get("momentum") or {}),
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

    history = _build_history(lookback, instruments_csv, raw.get("liquidity") or {})
    history_scores = [float(s) for s in history.get("scores", [])]
    history_pct = None
    if history_scores:
        below_or_equal = sum(1 for s in history_scores if s <= composite)
        history_pct = round((below_or_equal / len(history_scores)) * 100.0, 2)

    candidate_dates: list[pd.Timestamp] = []
    vix_latest = ((raw.get("vix_term_structure") or {}).get("latest_df") or [{}])[0]
    vix_date = vix_latest.get("Date")
    if isinstance(vix_date, str):
        candidate_dates.append(pd.to_datetime(vix_date, errors="coerce"))
    liq_date = (raw.get("liquidity") or {}).get("latest_date")
    if liq_date is not None:
        candidate_dates.append(pd.to_datetime(liq_date, errors="coerce"))
    sector_ts = (raw.get("sector_metrics") or {}).get("timestamp")
    if sector_ts is not None:
        candidate_dates.append(pd.to_datetime(sector_ts, errors="coerce"))

    pos_rows = raw.get("positioning") or []
    if isinstance(pos_rows, list) and pos_rows:
        dates = [pd.to_datetime((r or {}).get("report_date"), errors="coerce") for r in pos_rows if isinstance(r, dict)]
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
        raw_modules = {}
        for key, value in raw.items():
            if key == "liquidity" and isinstance(value, dict):
                raw_modules[key] = {k: v for k, v in value.items() if k not in ("df_weekly", "composite_series")}
            else:
                raw_modules[key] = value
        response["raw_modules"] = raw_modules

    return response
