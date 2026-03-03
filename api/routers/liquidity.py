import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from api.cache import short_cache, get_cached, set_cached
from api.serializers import serialize_response

router = APIRouter()


@router.get("/liquidity")
def get_liquidity(skip_ecb: bool = False):
    key = f"liquidity:{skip_ecb}"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from liquidity import get_snapshot
        data = get_snapshot(skip_ecb=skip_ecb)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Drop large DataFrame/Series objects that React doesn't need
    # (composite_series and df_weekly are internal computation artifacts)
    filtered = {
        k: v for k, v in data.items()
        if k not in ("df_weekly", "composite_series")
    }
    result = serialize_response(filtered)
    set_cached(short_cache, key, result)
    return result


class LiquidityAnalyzeRequest(BaseModel):
    composite_score: float | None = None
    regime: str | None = None
    latest_date: str | None = None
    regional_scores: dict = Field(default_factory=dict)
    components: list[dict] = Field(default_factory=list)
    changes: dict = Field(default_factory=dict)
    skip_ecb: bool = False


def _fmt(v, fmt=".2f", suffix=""):
    if v is None:
        return "N/A"
    try:
        return f"{float(v):{fmt}}{suffix}"
    except (TypeError, ValueError):
        return str(v)


def _to_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _build_components_table(rows: list[dict]) -> str:
    sorted_rows = sorted(
        rows,
        key=lambda r: abs(_to_float(r.get("contribution"))),
        reverse=True,
    )
    header = (
        f"{'Region':<8}  {'Component':<26}  {'Value':>11}  {'Z':>7}  "
        f"{'Weight':>7}  {'Contrib':>8}  {'Signal':<11}"
    )
    divider = "-" * len(header)
    lines = [header, divider]
    for r in sorted_rows:
        signal = "supportive" if _to_float(r.get("z_score")) >= 0 else "tightening"
        lines.append(
            f"{str(r.get('region', '')):<8}  "
            f"{str(r.get('label', ''))[:26]:<26}  "
            f"{_fmt(r.get('value'), '.2f'):>11}  "
            f"{_fmt(r.get('z_score'), '+.2f'):>7}  "
            f"{_fmt(_to_float(r.get('weight')) * 100, '.0f', '%'):>7}  "
            f"{_fmt(r.get('contribution'), '+.2f'):>8}  "
            f"{signal:<11}"
        )
    return "\n".join(lines)


def _build_changes_table(changes: dict) -> str:
    header = f"{'Series':<28}  {'1W':>10}  {'1M':>10}  {'3M':>10}  {'Polarity':>8}"
    divider = "-" * len(header)
    lines = [header, divider]
    for series, info in (changes or {}).items():
        row = info if isinstance(info, dict) else {}
        lines.append(
            f"{str(series)[:28]:<28}  "
            f"{_fmt(row.get('1w'), '+.2f'):>10}  "
            f"{_fmt(row.get('1m'), '+.2f'):>10}  "
            f"{_fmt(row.get('3m'), '+.2f'):>10}  "
            f"{_fmt(row.get('polarity'), '.0f'):>8}"
        )
    return "\n".join(lines)


@router.post("/liquidity/analyze")
def analyze_liquidity(req: LiquidityAnalyzeRequest):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    if not req.components and not req.changes:
        raise HTTPException(status_code=400, detail="No liquidity data provided")

    components_table = _build_components_table(req.components)
    changes_table = _build_changes_table(req.changes)

    regional_lines = []
    for region, info in (req.regional_scores or {}).items():
        r = info if isinstance(info, dict) else {}
        regional_lines.append(
            f"- {region}: score={_fmt(r.get('score'), '+.2f')}, regime={str(r.get('regime', 'unknown')).upper()}"
        )
    regional_summary = "\n".join(regional_lines) if regional_lines else "- N/A"

    prompt = f"""You are an experienced macro strategist. Analyze the following global liquidity dashboard snapshot and provide a concise but insightful interpretation for a professional investor audience.

As of: {req.latest_date or "N/A"}
ECB data excluded: {"yes" if req.skip_ecb else "no"}
Composite liquidity score: {_fmt(req.composite_score, '+.2f')}
Current regime: {str(req.regime or "unknown").upper()}

Regional scores:
{regional_summary}

Liquidity components:
{components_table}

Historical changes:
{changes_table}

Interpretation guide:
- Positive z-scores and contributions are liquidity-supportive; negative values are tightening.
- Weight indicates each component's impact in the composite score.
- For change rows, polarity=1 means positive changes are supportive; polarity=-1 means negative changes are supportive.

Write 4-5 flowing paragraphs of plain text (no bullet points, no markdown, no headers). Cover:
1. What the composite and regional scores imply about the current global liquidity backdrop
2. Which components are driving conditions most (supportive or tightening) and why that matters for risk assets
3. What 1W/1M/3M changes suggest about short-term momentum versus medium-term trend
4. Key macro risks or reversal triggers to monitor
5. A clear bottom-line regime assessment

Be specific about the numbers."""

    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.responses.create(model="gpt-5", input=prompt)
        analysis = (resp.output_text or "").strip()
        if not analysis:
            raise ValueError("OpenAI returned empty response")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {exc}")

    return {"analysis": analysis}
