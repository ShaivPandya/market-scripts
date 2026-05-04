import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.cache import get_cached, long_cache, set_cached
from api.exceptions import ConfigurationError, DataFetchError, SnapshotUnavailableError
from api.serializers import serialize_value
from api.snapshot_keys import SNAPSHOT_SECTOR_METRICS
from api.snapshot_store import get_snapshot_response, snapshots_required
from equities.sector_metrics.payload import normalize_sector_metrics_payload
from llm_utils import MODEL_LOW, api_key_env, call_llm_text, has_llm_api_key

router = APIRouter()


@router.get("/sector-metrics")
def get_sector_metrics():
    key = "sector_metrics"
    cached = get_cached(long_cache, key)
    if cached is not None:
        return normalize_sector_metrics_payload(cached)
    snapshot = get_snapshot_response(SNAPSHOT_SECTOR_METRICS)
    if snapshot is not None:
        snapshot = normalize_sector_metrics_payload(snapshot)
        set_cached(long_cache, key, snapshot)
        return snapshot
    if snapshots_required():
        raise SnapshotUnavailableError(SNAPSHOT_SECTOR_METRICS)
    try:
        from equities.sector_metrics.sector_metrics import get_data

        data = get_data()
    except Exception as e:
        raise DataFetchError(source="sector_metrics", detail=str(e)) from e

    normalized = normalize_sector_metrics_payload(data)
    weights_df = normalized.get("weights_df")
    result = {
        "weights_df": weights_df if isinstance(weights_df, list) else [],
        "d_1m": data.get("d_1m"),
        "d_3m": data.get("d_3m"),
        "d_6m": data.get("d_6m"),
        "timestamp": serialize_value(data.get("timestamp")),
    }
    set_cached(long_cache, key, result)
    return result


class SectorMetricsAnalyzeRequest(BaseModel):
    rows: list[dict]
    timestamp: str | None = None


def _fmt(v, fmt=".2f", suffix=""):
    if v is None:
        return "N/A"
    try:
        val = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{val:{fmt}}{suffix}"


def _build_sector_table(rows: list[dict]) -> str:
    sorted_rows = sorted(
        rows,
        key=lambda r: float(r.get("Weight_Now") or 0),
        reverse=True,
    )
    header = (
        f"{'Sector':<24}  {'Weight':>8}  {'1M Chg':>8}  {'3M Chg':>8}  "
        f"{'6M Chg':>8}  {'Rel 3M':>8}  {'Rel 12M':>8}  {'%>200DMA':>9}"
    )
    divider = "-" * len(header)
    lines = [header, divider]
    for r in sorted_rows:
        lines.append(
            f"{str(r.get('Sector', '')):<24}  "
            f"{_fmt(r.get('Weight_Now'), '.1f', '%'):>8}  "
            f"{_fmt(r.get('Chg_1M_pp'), '+.2f', 'pp'):>8}  "
            f"{_fmt(r.get('Chg_3M_pp'), '+.2f', 'pp'):>8}  "
            f"{_fmt(r.get('Chg_6M_pp'), '+.2f', 'pp'):>8}  "
            f"{_fmt(r.get('RelPerf_3M_pp'), '+.2f', 'pp'):>8}  "
            f"{_fmt(r.get('RelPerf_12M_pp'), '+.2f', 'pp'):>8}  "
            f"{_fmt(r.get('Pct_Above_200DMA'), '.1f', '%'):>9}"
        )
    return "\n".join(lines)


@router.post("/sector-metrics/analyze")
def analyze_sector_metrics(req: SectorMetricsAnalyzeRequest):
    if not has_llm_api_key():
        raise ConfigurationError(api_key_env())
    if not req.rows:
        raise HTTPException(status_code=400, detail="No sector metric rows provided")

    table = _build_sector_table(req.rows)
    as_of = f"As of: {req.timestamp}\n\n" if req.timestamp else ""

    prompt = f"""You are an experienced equity strategist. Analyze the following S&P 500 sector metrics and provide a concise but insightful overview of market leadership, risk appetite, and trend quality.

{as_of}SECTOR METRICS:
- Weight = current sector weight in the S&P 500 proxy
- Chg (pp) = change in sector weight over the lookback window in percentage points
- Rel Perf (pp) = sector performance relative to the S&P 500 over the lookback window
- %>200DMA = sector ETF distance above/below its 200-day moving average

{table}

Write 2-3 flowing paragraphs of plain text (no bullet points, no markdown, no headers). Be concise. Cover:
1. Which sectors dominate weight and whether concentration is rising or broadening, plus what weight shifts imply about rotating risk appetite
2. Where relative performance is strongest/weakest across cyclical vs defensive groups and what trend indicators suggest
3. An overall regime takeaway and what to watch next

Be specific about the numbers. Write for a professional investor audience."""

    try:
        analysis, _citations, _resp = call_llm_text(
            prompt=prompt,
            model=MODEL_LOW,
            api_key=None,
            max_tokens=2048,
        )
        if not analysis:
            raise ValueError("LLM returned empty response")
    except Exception as exc:
        raise DataFetchError(source="ai_analysis", detail=str(exc)) from exc

    return {"analysis": analysis}
