import os

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from api.cache import daily_cache
from api.exceptions import ConfigurationError, DataFetchError
from api.macro_snapshots import get_snapshot_backed_response
from api.serializers import serialize_response
from api.snapshot_keys import SNAPSHOT_HOUSING
from llm_utils import MODEL_LOW, api_key_env, call_llm_text, has_llm_api_key

router = APIRouter()


def load_housing_payload() -> dict:
    try:
        from macro.housing.housing import get_data

        data = get_data()
    except Exception as e:
        raise DataFetchError(source="housing", detail=str(e)) from e

    return serialize_response(data)


@router.get("/housing")
def get_housing(force_refresh: bool = Query(False)):
    key = "housing"
    return get_snapshot_backed_response(
        snapshot_key=SNAPSHOT_HOUSING,
        cache=daily_cache,
        cache_key=key,
        source="housing",
        load_payload=load_housing_payload,
        force_refresh=force_refresh,
    )


class HousingAnalyzeRequest(BaseModel):
    latest: dict = Field(default_factory=dict)
    series_labels: dict = Field(default_factory=dict)
    series_units: dict = Field(default_factory=dict)
    timestamp: str | None = None


def _fmt(v, fmt=".2f", suffix=""):
    if v is None:
        return "N/A"
    try:
        return f"{float(v):{fmt}}{suffix}"
    except (TypeError, ValueError):
        return str(v)


def _build_snapshot_table(req: HousingAnalyzeRequest) -> str:
    header = f"{'Indicator':<32}  {'Latest':>12}  {'Change':>10}  {'Unit':<12}  {'As of':<12}"
    divider = "-" * len(header)
    lines = [header, divider]
    for key, info in (req.latest or {}).items():
        label = req.series_labels.get(key, key)
        unit = req.series_units.get(key, "")
        val = info.get("value")
        chg = info.get("change")
        date = info.get("date", "N/A")
        chg_str = f"{'+' if chg and chg >= 0 else ''}{_fmt(chg)}" if chg is not None else "N/A"
        lines.append(f"{str(label)[:32]:<32}  {_fmt(val):>12}  {chg_str:>10}  {unit:<12}  {str(date):<12}")
    return "\n".join(lines)


@router.post("/housing/analyze")
def analyze_housing(req: HousingAnalyzeRequest):
    if not has_llm_api_key():
        raise ConfigurationError(api_key_env())
    if not req.latest:
        raise HTTPException(status_code=400, detail="No housing data provided")

    snapshot_table = _build_snapshot_table(req)

    prompt = f"""You are an experienced macro strategist. Analyze the following US housing market dashboard snapshot and provide a concise but insightful interpretation for a professional investor audience.

As of: {req.timestamp or "N/A"}

Housing market indicators:
{snapshot_table}

Indicator guide:
- Housing Starts (HOUST): new residential construction projects begun (thousands, SAAR). Rising = expanding supply pipeline.
- Building Permits (PERMIT): authorized new housing units (thousands, SAAR). Leading indicator of future starts.
- NAHB Housing Market Index (NAHBHMI): builder confidence survey (0-100). Above 50 = more builders view conditions as good. Below 50 = pessimistic.
- Existing Home Sales (EXHOSLUSM495S): completed sales of existing homes (millions, SAAR). Reflects demand and affordability conditions.

Write 2-3 flowing paragraphs of plain text (no bullet points, no markdown, no headers). Be concise. Cover:
1. Overall characterization of the housing cycle and what starts, permits, and builder confidence signal about the construction pipeline
2. What existing home sales indicate about demand and affordability, plus key macro risks to watch
3. A clear bottom-line assessment for risk assets

Be specific about the numbers."""

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
