from __future__ import annotations

from typing import Any

W_VOLATILITY = 0.35
W_BREADTH = 0.25
W_SECTOR = 0.25
W_MACRO = 0.15


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_volatility_cluster(
    vix_term_structure: dict[str, Any] | None, sentiment: dict[str, Any] | None
) -> tuple[float, list[dict[str, Any]]]:
    latest = ((vix_term_structure or {}).get("latest_df") or [{}])[0]
    ratio = _to_float(latest.get("Ratio"))
    vix_signal = str(latest.get("Signal") or "Neutral")
    vix_level = _to_float(latest.get("VIX"))

    if vix_signal == "Fear":
        base = 0.9
    elif vix_signal == "Complacency":
        base = 0.6
    else:
        base = 0.45

    sentiment_vol = (sentiment or {}).get("volatility")
    vvix = None
    if isinstance(sentiment_vol, list) and sentiment_vol:
        vvix = _to_float((sentiment_vol[-1] or {}).get("vvix"))

    adjustment = 0.0
    if vix_level is not None:
        if vix_level >= 30:
            adjustment += 0.2
        elif vix_level >= 20:
            adjustment += 0.1
    if vvix is not None and vvix >= 110:
        adjustment += 0.1

    score = clamp01(base + adjustment)
    evidence = [
        {
            "name": "VIX 3M/1M Ratio",
            "source": "vix_term_structure",
            "value": ratio,
            "threshold": "< 1.00 => fear",
            "direction": "deteriorating" if vix_signal == "Fear" else "stable",
            "raw_signal": vix_signal,
        },
        {
            "name": "Spot VIX",
            "source": "vix_term_structure",
            "value": vix_level,
            "threshold": ">= 20 elevated",
            "direction": "deteriorating" if (vix_level or 0) >= 20 else "stable",
            "raw_signal": "elevated" if (vix_level or 0) >= 20 else "normal",
        },
    ]
    if vvix is not None:
        evidence.append(
            {
                "name": "VVIX",
                "source": "sentiment",
                "value": vvix,
                "threshold": ">= 110 elevated vol-of-vol",
                "direction": "deteriorating" if vvix >= 110 else "stable",
                "raw_signal": "elevated" if vvix >= 110 else "normal",
            }
        )

    return score, evidence


def compute_breadth_stress(
    market_breadth: dict[str, Any] | None,
    top50_breadth: dict[str, Any] | None,
) -> tuple[float, list[dict[str, Any]]]:
    m = market_breadth or {}
    t = top50_breadth or {}

    p200 = _to_float(m.get("pct_above_200dma"))
    p20 = _to_float(m.get("pct_above_20dma"))
    p20l = _to_float(m.get("pct_at_20day_low"))
    p52l = _to_float(m.get("pct_at_52wk_low"))

    t50 = _to_float(t.get("pct_below_50dma"))
    tdist = _to_float(t.get("pct_3plus_dist"))
    tbroke = _to_float(t.get("pct_broke_20low"))

    components = []
    if p200 is not None:
        components.append(clamp01((55 - p200) / 55))
    if p20 is not None:
        components.append(clamp01((55 - p20) / 55))
    if p20l is not None:
        components.append(clamp01((p20l - 20) / 40))
    if p52l is not None:
        components.append(clamp01((p52l - 10) / 30))
    if t50 is not None:
        components.append(clamp01((t50 - 35) / 45))
    if tdist is not None:
        components.append(clamp01((tdist - 25) / 50))
    if tbroke is not None:
        components.append(clamp01((tbroke - 15) / 40))

    score = sum(components) / len(components) if components else 0.5

    evidence = [
        {
            "name": "% Above 200DMA",
            "source": "market_breadth",
            "value": p200,
            "threshold": "< 55% weak breadth",
            "direction": "deteriorating" if (p200 or 100) < 55 else "stable",
            "raw_signal": "weak" if (p200 or 100) < 55 else "healthy",
        },
        {
            "name": "% At 20-Day Lows",
            "source": "market_breadth",
            "value": p20l,
            "threshold": "> 50% capitulation",
            "direction": "deteriorating" if (p20l or 0) > 50 else "stable",
            "raw_signal": "stress" if (p20l or 0) > 50 else "normal",
        },
        {
            "name": "Top50 % Below 50DMA",
            "source": "top50_breadth",
            "value": t50,
            "threshold": "> 35% leadership damage",
            "direction": "deteriorating" if (t50 or 0) > 35 else "stable",
            "raw_signal": "damage" if (t50 or 0) > 35 else "healthy",
        },
    ]
    return clamp01(score), evidence


def compute_sector_stress_map(
    sector_metrics: dict[str, Any] | None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = (sector_metrics or {}).get("weights_df") or []
    if not isinstance(rows, list):
        rows = []

    out: dict[str, float] = {}
    evidence: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        sector = str(row.get("Sector") or "").strip()
        if not sector:
            continue

        rel3m = _to_float(row.get("RelPerf_3M_pp"))
        chg3m = _to_float(row.get("Chg_3M_pp"))
        pct200 = _to_float(row.get("Pct_Above_200DMA"))

        components = []
        if rel3m is not None:
            components.append(clamp01((-rel3m) / 8))
        if chg3m is not None:
            components.append(clamp01((-chg3m) / 1.5))
        if pct200 is not None:
            components.append(clamp01((-pct200) / 12))

        score = sum(components) / len(components) if components else 0.5
        out[sector] = clamp01(score)

        evidence.append(
            {
                "name": f"{sector} sector stress",
                "source": "sector_metrics",
                "value": round(score, 4),
                "threshold": "higher => weaker rel perf / breadth",
                "direction": "deteriorating" if score >= 0.6 else "stable",
                "raw_signal": "deteriorating" if score >= 0.6 else "stable",
                "sector": sector,
            }
        )

    if not out:
        out["Unknown Equity"] = 0.5

    return out, evidence


def compute_macro_regime(
    liquidity: dict[str, Any] | None,
    positioning: dict[str, Any] | None,
    economic_growth: dict[str, Any] | None,
    labor_market: dict[str, Any] | None,
) -> tuple[float, list[dict[str, Any]]]:
    liq = liquidity or {}
    regime = str(liq.get("regime") or "normal").lower()
    base = {
        "stress": 1.0,
        "tight": 0.8,
        "normal": 0.45,
        "ample": 0.25,
    }.get(regime, 0.5)

    score = base
    evidence: list[dict[str, Any]] = [
        {
            "name": "Liquidity Regime",
            "source": "liquidity",
            "value": regime,
            "threshold": "stress/tight => deteriorating",
            "direction": "deteriorating" if regime in {"stress", "tight"} else "stable",
            "raw_signal": regime,
        }
    ]

    pos_rows = _flatten_positioning_rows(positioning)
    crowded = 0
    for row in pos_rows:
        z = _to_float(row.get("lf_z"))
        if z is not None and abs(z) >= 2:
            crowded += 1
    if crowded:
        score += min(0.2, crowded * 0.05)
        evidence.append(
            {
                "name": "Crowded Positioning Count",
                "source": "positioning",
                "value": crowded,
                "threshold": ">= 3 crowded markets",
                "direction": "deteriorating" if crowded >= 3 else "stable",
                "raw_signal": "crowded" if crowded >= 3 else "contained",
            }
        )

    growth_adjustment = _growth_risk_adjustment(economic_growth)
    if growth_adjustment is not None:
        score += growth_adjustment
        evidence.append(
            {
                "name": "Growth Breadth Proxy",
                "source": "economic_growth",
                "value": round(growth_adjustment, 4),
                "threshold": "negative cross-asset growth trends",
                "direction": "deteriorating" if growth_adjustment > 0 else "stable",
                "raw_signal": "weaker" if growth_adjustment > 0 else "mixed",
            }
        )

    labor_adjustment = _labor_risk_adjustment(labor_market)
    if labor_adjustment is not None:
        score += labor_adjustment
        evidence.append(
            {
                "name": "Labor Market Risk",
                "source": "labor_market",
                "value": round(labor_adjustment, 4),
                "threshold": "rising claims / softer labor",
                "direction": "deteriorating" if labor_adjustment > 0 else "stable",
                "raw_signal": "softening" if labor_adjustment > 0 else "stable",
            }
        )

    return clamp01(score), evidence


def score_position(
    volatility_cluster: float,
    breadth_stress: float,
    sector_stress: float,
    macro_regime: float,
) -> float:
    score = (
        W_VOLATILITY * clamp01(volatility_cluster)
        + W_BREADTH * clamp01(breadth_stress)
        + W_SECTOR * clamp01(sector_stress)
        + W_MACRO * clamp01(macro_regime)
    )
    return clamp01(score)


def risk_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _flatten_positioning_rows(positioning: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(positioning, dict):
        return []
    out: list[dict[str, Any]] = []
    for v in positioning.values():
        if isinstance(v, dict) and "lf_z" in v:
            out.append(v)
        elif isinstance(v, list):
            for row in v:
                if isinstance(row, dict) and "lf_z" in row:
                    out.append(row)
    return out


def _growth_risk_adjustment(data: dict[str, Any] | None) -> float | None:
    if not isinstance(data, dict):
        return None
    # Handle common return tables with period keys and float values.
    negatives = 0
    total = 0
    for category in ("commodities", "equities", "currencies"):
        bucket = data.get(category)
        if not isinstance(bucket, dict):
            continue
        for _, periods in bucket.items():
            if not isinstance(periods, dict):
                continue
            for p in ("1M", "3M", "6M"):
                val = _to_float(periods.get(p))
                if val is None:
                    continue
                total += 1
                if val < 0:
                    negatives += 1
    if total == 0:
        return None
    ratio = negatives / total
    return clamp01((ratio - 0.45) * 0.4)


def _labor_risk_adjustment(data: dict[str, Any] | None) -> float | None:
    if not isinstance(data, dict):
        return None
    latest = data.get("latest")
    if not isinstance(latest, dict):
        return None

    claims = latest.get("initial_claims")
    if not isinstance(claims, dict):
        return None

    change = _to_float(claims.get("change"))
    if change is None:
        return None
    return clamp01(max(0.0, change) / 25.0)
