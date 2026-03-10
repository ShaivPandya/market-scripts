import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.cache import get_cached, set_cached, short_cache
from api.exceptions import ConfigurationError, DataFetchError
from api.serializers import serialize_response
from llm_utils import MODEL_HAIKU, call_claude_text

router = APIRouter()


@router.get("/labor-market")
def get_labor_market():
    key = "labor_market"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from macro.labor_market.labor_market import get_data

        data = get_data()
    except Exception as e:
        raise DataFetchError(source="labor_market", detail=str(e)) from e

    result = serialize_response(data)
    set_cached(short_cache, key, result)
    return result


class LaborMarketAnalyzeRequest(BaseModel):
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


def _build_snapshot_table(req: LaborMarketAnalyzeRequest) -> str:
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


@router.post("/labor-market/analyze")
def analyze_labor_market(req: LaborMarketAnalyzeRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ConfigurationError("ANTHROPIC_API_KEY")
    if not req.latest:
        raise HTTPException(status_code=400, detail="No labor market data provided")

    snapshot_table = _build_snapshot_table(req)

    prompt = f"""You are an experienced macro strategist. Analyze the following US labor market dashboard snapshot and provide a concise but insightful interpretation for a professional investor audience.

As of: {req.timestamp or "N/A"}

Labor market indicators:
{snapshot_table}

Indicator guide:
- Initial Jobless Claims (ICSA): weekly new unemployment insurance filings (thousands). Rising = labor market weakening.
- Continuing Claims (CCSA): workers still receiving unemployment benefits (thousands). Rising = difficulty finding work.
- Median Weeks Unemployed (UEMPMED): median duration of unemployment spells. Rising = structural labor market slack.
- Avg Weekly Hours Worked (AWHAETP): average hours worked per week across private sector. Falling = employers cutting hours before layoffs.
- Wage Growth YoY (AHETPI): year-over-year % change in average hourly earnings. High = inflationary pressure; low = cooling.
- Job Openings JOLTS (JTSJOL): total unfilled job openings (thousands). High = strong labor demand; falling = tightening.

Write 2-3 flowing paragraphs of plain text (no bullet points, no markdown, no headers). Be concise. Cover:
1. Overall characterization of labor market conditions and what claims data signals about near-term layoff momentum
2. What wages, hours worked, and job openings reveal about inflationary pressure and structural dynamics
3. Key macro risks to watch and a clear bottom-line assessment for risk assets

Be specific about the numbers."""

    try:
        analysis, _citations, _resp = call_claude_text(
            prompt=prompt,
            model=MODEL_HAIKU,
            api_key=api_key,
            max_tokens=2048,
        )
        if not analysis:
            raise ValueError("Claude returned empty response")
    except Exception as exc:
        raise DataFetchError(source="ai_analysis", detail=str(exc)) from exc

    return {"analysis": analysis}
