import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.cache import get_cached, long_cache, set_cached
from api.exceptions import ConfigurationError, DataFetchError
from api.serializers import serialize_dataframe, serialize_value
from llm_utils import MODEL_HAIKU_4_5, call_claude_text

router = APIRouter()


@router.get("/positioning/summary")
def get_positioning_summary(
    instruments: str = "SP500,NASDAQ,RUSSELL,US10Y,EUR",
    start: str = "2015-01-01",
    end: str | None = None,
    groups: str | None = None,
    z_window: int = 0,
    force_threshold: float = 2.0,
    app_token: str | None = None,
):
    resolved_token = app_token or os.environ.get("SODA_APP_TOKEN") or None
    key = f"positioning_summary:{instruments}:{start}:{end}:{groups}:{z_window}:{force_threshold}"
    cached = get_cached(long_cache, key)
    if cached is not None:
        return cached
    try:
        from positioning import (
            DATASETS,
            DEFAULT_DOMAIN,
            INSTRUMENTS,
            fetch_multiple_instruments,
        )

        instrument_list = [i.strip() for i in instruments.split(",") if i.strip()]
        results = fetch_multiple_instruments(
            domain=DEFAULT_DOMAIN,
            dataset_id=DATASETS.get("tff_futures_only", "tff_futures_only"),
            app_token=resolved_token,
            instruments=instrument_list,
            start=start,
            end=end,
            groups=groups or None,
            z_window=z_window,
            force_threshold=force_threshold,
        )
    except Exception as e:
        raise DataFetchError(source="positioning", detail=str(e)) from e

    result = serialize_value(results)
    set_cached(long_cache, key, result)
    return result


@router.get("/positioning/timeseries")
def get_positioning_timeseries(
    market: str,
    start: str = "2015-01-01",
    end: str | None = None,
    groups: str | None = None,
    z_window: int = 0,
    force_threshold: float = 2.0,
    app_token: str | None = None,
):
    resolved_token = app_token or os.environ.get("SODA_APP_TOKEN") or None
    key = f"positioning_ts:{market}:{start}:{end}:{groups}:{z_window}:{force_threshold}"
    cached = get_cached(long_cache, key)
    if cached is not None:
        return cached
    try:
        from positioning import DATASETS, DEFAULT_DOMAIN, fetch_market_timeseries

        df = fetch_market_timeseries(
            domain=DEFAULT_DOMAIN,
            dataset_id=DATASETS.get("tff_futures_only", "tff_futures_only"),
            app_token=resolved_token,
            market_exact=market,
            start=start,
            end=end,
            groups=groups or None,
            z_window=z_window,
            force_threshold=force_threshold,
        )
    except Exception as e:
        raise DataFetchError(source="positioning", detail=str(e)) from e

    result = serialize_dataframe(df.reset_index(drop=True))
    set_cached(long_cache, key, result)
    return result


@router.get("/positioning/instruments")
def get_positioning_instruments():
    """Return available instrument aliases."""
    try:
        from positioning import INSTRUMENTS

        return {"instruments": INSTRUMENTS}
    except Exception as e:
        raise DataFetchError(source="positioning", detail=str(e)) from e


class PositioningAnalyzeRequest(BaseModel):
    rows: list[dict]


def _fmt(v, fmt=".1f", suffix=""):
    if v is None:
        return "N/A"
    try:
        return f"{float(v):{fmt}}{suffix}"
    except (TypeError, ValueError):
        return str(v)


@router.post("/positioning/analyze")
def analyze_positioning(req: PositioningAnalyzeRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ConfigurationError("ANTHROPIC_API_KEY")

    # Sort rows by absolute z-score descending so most extreme positions appear first
    rows = sorted(
        req.rows,
        key=lambda r: abs(float(r.get("lf_z") or 0)) if r.get("lf_z") is not None else 0,
        reverse=True,
    )

    # Build a readable table
    header = (
        f"{'Instrument':<12}  {'Report Date':<12}  {'Net % OI':>8}  {'Pos Z':>7}  {'Delev Z':>7}  {'Forced Flow':<18}"
    )
    divider = "-" * len(header)
    lines = [header, divider]
    for r in rows:
        forced = str(r.get("lf_forced") or "").replace("_", " ").title() or "—"
        lines.append(
            f"{str(r.get('instrument', '')):<12}  "
            f"{str(r.get('report_date', '')):<12}  "
            f"{_fmt(r.get('lf_net_pct_oi'), '.1f', '%'):>8}  "
            f"{_fmt(r.get('lf_z'), '+.2f'):>7}  "
            f"{_fmt(r.get('lf_deleveraging_z'), '+.2f'):>7}  "
            f"{forced:<18}"
        )
    table = "\n".join(lines)

    prompt = f"""You are an experienced macro strategist specializing in CFTC Commitments of Traders (COT) data. Analyze the following leveraged fund positioning snapshot and provide a concise but insightful interpretation of what it signals about market sentiment and potential risks.

LEVERAGED FUND POSITIONING SUMMARY (sorted by most extreme z-score first):
- Net % OI = leveraged funds net position as a percentage of open interest
- Pos Z = z-score of net positioning vs history (how crowded; ±2 is historically extreme)
- Delev Z = z-score of position reduction toward flat (positive = unusual deleveraging)
- Forced Flow = "Long Liquidation" (forced selling of longs) or "Short Covering" (forced buying to close shorts)

{table}

Write 4-5 flowing paragraphs of plain text (no bullet points, no markdown, no headers). Cover:
1. The most crowded positions — which instruments have extreme z-scores (positive = very net long, negative = very net short) and what this historically implies for tail risk
2. Any active forced-flow signals — which instruments are experiencing long liquidation or short covering, what is driving it, and how large the deleveraging z-score is
3. Cross-asset read — compare equity index positioning (SP500, NASDAQ, RUSSELL) vs bond (US10Y) vs currencies (EUR, JPY, AUD, CAD, GBP) to characterize the aggregate risk-on/risk-off tilt
4. Potential positioning risks — if crowded longs/shorts unwind, what is the likely market impact and which instruments are most vulnerable to a squeeze or cascade
5. Overall positioning conclusion — where are leveraged funds leaning, and what does this imply for near-term market dynamics

Be specific about the numbers. Write for a professional investor audience."""

    try:
        analysis, _citations, _resp = call_claude_text(
            prompt=prompt,
            model=MODEL_HAIKU_4_5,
            api_key=api_key,
            max_tokens=4096,
        )
        if not analysis:
            raise ValueError("Claude returned empty response")
    except Exception as exc:
        raise DataFetchError(source="ai_analysis", detail=str(exc)) from exc

    return {"analysis": analysis}
