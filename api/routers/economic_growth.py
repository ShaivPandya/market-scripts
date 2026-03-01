import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.cache import short_cache, get_cached, set_cached
from api.serializers import serialize_response

router = APIRouter()


@router.get("/economic-growth")
def get_economic_growth():
    key = "economic_growth"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from economic_growth import get_data
        data = get_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    result = serialize_response(data)
    set_cached(short_cache, key, result)
    return result


class EconomicGrowthAnalyzeRequest(BaseModel):
    commodities: dict
    equities: dict
    currencies: dict
    equity_periods: list[str]
    currency_periods: list[str]


def _format_table(data: dict, periods: list[str]) -> str:
    lines = []
    header = "  ".join(f"{p:>8}" for p in periods)
    lines.append(f"{'Asset':<28}  {header}")
    lines.append("-" * (30 + 10 * len(periods)))
    for name, returns in data.items():
        vals = "  ".join(
            f"{returns.get(p):>+8.1f}" if returns.get(p) is not None else f"{'N/A':>8}"
            for p in periods
        )
        lines.append(f"{name:<28}  {vals}")
    return "\n".join(lines)


@router.post("/economic-growth/analyze")
def analyze_economic_growth(req: EconomicGrowthAnalyzeRequest):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    commodities_table = _format_table(req.commodities, req.equity_periods)
    equities_table = _format_table(req.equities, req.equity_periods)
    currencies_table = _format_table(req.currencies, req.currency_periods)

    prompt = f"""You are an experienced macro strategist. Analyze the following market performance data and provide a concise but insightful overview of what it indicates about the current global economic growth environment.

The data shows percentage returns over multiple time periods.

COMMODITIES (returns %):
{commodities_table}

EQUITIES vs BENCHMARK (returns %):
S&P 500 and STOXX 600 are the benchmarks. Green = outperforming benchmark, red = underperforming.
{equities_table}

CURRENCY PAIRS (returns %):
AUD/JPY and CAD/JPY are commodity-currency risk sentiment proxies. Rising = risk-on, falling = risk-off.
{currencies_table}

Write 4-6 flowing paragraphs of plain text (no bullet points, no markdown, no headers). Cover:
1. What commodity moves (Copper, GSCI index, CRB Industrial) signal about industrial demand and inflation pressures
2. What equity breadth signals (small caps, transport, banks vs large-cap S&P 500) indicate about growth depth, credit conditions, and risk appetite
3. What European and EM signals (STOXX 600, Europe Banks, MSCI Korea) suggest about global growth synchronization
4. What the currency pairs imply about risk sentiment and commodity demand
5. An overall conclusion about where we are in the growth cycle and what to watch

Be specific about the numbers. Write for a professional investor audience."""

    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.responses.create(model="gpt-5-mini", input=prompt)
        analysis = (resp.output_text or "").strip()
        if not analysis:
            raise ValueError("OpenAI returned empty response")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {exc}")

    return {"analysis": analysis}
