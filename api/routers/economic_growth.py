import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.cache import get_cached, set_cached, short_cache
from api.exceptions import ConfigurationError, DataFetchError
from api.serializers import serialize_response
from llm_utils import MODEL_HAIKU, call_claude_text

router = APIRouter()
REQUIRED_CURRENCY_PERIODS = ["1-mo", "3-mo", "6-mo", "1-yr"]


def _normalize_currency_payload(payload: dict) -> dict:
    periods = payload.get("currency_periods")
    normalized_periods = [p for p in periods if isinstance(p, str)] if isinstance(periods, list) else []
    for period in REQUIRED_CURRENCY_PERIODS:
        if period not in normalized_periods:
            normalized_periods.append(period)
    payload["currency_periods"] = normalized_periods

    currencies = payload.get("currencies")
    if isinstance(currencies, dict):
        for returns in currencies.values():
            if isinstance(returns, dict):
                for period in normalized_periods:
                    returns.setdefault(period, None)
    return payload


@router.get("/economic-growth")
def get_economic_growth():
    key = "economic_growth"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return _normalize_currency_payload(cached)
    try:
        from macro.economic_growth.economic_growth import get_data

        data = get_data()
    except Exception as e:
        raise DataFetchError(source="economic_growth", detail=str(e)) from e
    result = _normalize_currency_payload(serialize_response(data))
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
        vals = "  ".join(f"{returns.get(p):>+8.1f}" if returns.get(p) is not None else f"{'N/A':>8}" for p in periods)
        lines.append(f"{name:<28}  {vals}")
    return "\n".join(lines)


@router.post("/economic-growth/analyze")
def analyze_economic_growth(req: EconomicGrowthAnalyzeRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ConfigurationError("ANTHROPIC_API_KEY")

    commodities_table = _format_table(req.commodities, req.equity_periods)
    equities_table = _format_table(req.equities, req.equity_periods)
    currencies_table = _format_table(req.currencies, req.currency_periods)

    prompt = f"""You are an experienced macro strategist. Analyze the following market performance data and provide a concise but insightful overview of what it indicates about the current global economic growth environment.

The data shows percentage returns over 1-month (30 days), 3-month (91 days), 6-month (182 days), and 1-year (365 days) periods. US equities are benchmarked against S&P 500, Europe Banks against STOXX 600. Outperformance = bullish growth signal, underperformance = bearish.

COMMODITIES (returns %):
{commodities_table}

Key context:
- Copper ("Dr. Copper"): Highly sensitive to global economic activity due to widespread use in construction, electrical equipment, and manufacturing. Rising prices signal expansion, falling prices suggest contraction.
- CRB Industrial Spot Index: A broad index of non-traded industrial commodities (metals, textiles, agricultural inputs). Less influenced by investor speculation than futures-based indices, making it a purer measure of real industrial demand.
- GS Commodity Index (GSG): Broad commodity exposure including energy, metals, and agriculture. Identifies inflationary pressures and global demand trends.

EQUITIES vs BENCHMARK (returns %):
{equities_table}

Key context:
- Russell 2000 (IWM) & S&P 600 (IJR): Small-cap stocks with less diversified revenue, more domestic economic dependence, and less pricing power. Outperformance vs S&P 500 signals risk-on sentiment and economic optimism.
- DJ Transport (IYT): Direct beneficiary of goods movement. Dow Theory holds that transports should confirm trends in industrials — divergence signals economic weakness ahead.
- KBW Banks (KBWB) & Europe Banks (EXV1.DE): Highly cyclical financials whose profitability depends on loan demand, net interest margins, and credit quality. Strong performance signals confidence in growth and credit conditions.
- US Retail (XRT): Consumer discretionary spending indicator sensitive to household confidence and income growth.
- US Staples (XLP): Defensive sector that typically underperforms during expansions and outperforms during slowdowns. Relative strength signals defensive positioning and economic concern.
- US Utilities (XLU): Another defensive sector; outperformance suggests investors are seeking safety and yield over growth.
- MSCI Korea (EWY): Export-dependent and cyclical, sensitive to global manufacturing (semiconductors, electronics, autos), China's economic health, and global trade volumes. Used by macro investors as a proxy for global economic optimism.
- STOXX 600 (^STOXX): European equity benchmark for gauging European economic health and investor sentiment toward the region.

CURRENCY PAIRS (returns %):
{currencies_table}

Key context:
- AUD/JPY & CAD/JPY: Classic risk-on/risk-off indicators. JPY is a safe-haven currency; AUD and CAD are commodity currencies. Rising pairs suggest risk appetite and commodity demand (economic growth). Falling pairs suggest risk aversion and economic uncertainty. Both correlate with global risk sentiment and commodity cycles.

BULLISH GROWTH SIGNALS: Small-caps outperforming S&P 500, banks outperforming benchmarks, copper and CRB rising, transports strong, Korea outperforming, staples/utilities underperforming, AUD/JPY and CAD/JPY rising.
BEARISH/DEFENSIVE SIGNALS: Small-caps underperforming, banks underperforming, commodities falling, staples/utilities outperforming S&P 500, Korea underperforming, currency pairs falling (yen strength).

Write 2-3 flowing paragraphs of plain text (no bullet points, no markdown, no headers). Be concise. Cover:
1. What commodity moves and equity breadth signals indicate about industrial demand, growth depth, and risk appetite
2. What European/EM signals and currency pairs imply about global growth synchronization and risk sentiment
3. An overall conclusion about where we are in the growth cycle and what to watch

Be specific about the numbers. Write for a professional investor audience."""

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
