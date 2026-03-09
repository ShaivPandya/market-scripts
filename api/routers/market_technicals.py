import os
from datetime import date, timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.cache import get_cached, set_cached, short_cache
from api.exceptions import ConfigurationError, DataFetchError
from api.serializers import serialize_response
from llm_utils import MODEL_HAIKU, call_claude_text

router = APIRouter()


@router.get("/market-breadth")
def get_market_breadth():
    key = "market_breadth"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from equities.market_technicals.market_breadth import get_data

        data = get_data()
    except Exception as e:
        raise DataFetchError(source="market_breadth", detail=str(e)) from e
    result = serialize_response(data)
    set_cached(short_cache, key, result)
    return result


@router.get("/top50-breadth")
def get_top50_breadth():
    key = "top50_breadth"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from equities.market_technicals.top50_breadth import get_data

        data = get_data()
    except Exception as e:
        raise DataFetchError(source="top50_breadth", detail=str(e)) from e
    result = serialize_response(data)
    set_cached(short_cache, key, result)
    return result


@router.get("/price-volume-signals")
def get_price_volume_signals():
    key = "price_volume_signals"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from equities.market_technicals.price_volume_signals import get_data

        data = get_data()
    except Exception as e:
        raise DataFetchError(source="price_volume_signals", detail=str(e)) from e
    result = serialize_response(data)
    set_cached(short_cache, key, result)
    return result


@router.get("/vix-term-structure")
def get_vix_term_structure():
    key = "vix_term_structure"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from equities.market_technicals.vix_term_structure import get_data

        start = (date.today() - timedelta(days=400)).isoformat()
        data = get_data(tail=252, signals_count=20, start=start)
    except Exception as e:
        raise DataFetchError(source="vix_term_structure", detail=str(e)) from e
    result = serialize_response(data)
    set_cached(short_cache, key, result)
    return result


class MarketTechnicalsAnalyzeRequest(BaseModel):
    market_breadth: dict
    top50_breadth: dict
    vix_term_structure: dict
    price_volume_signals: dict


def _format_breadth(data: dict) -> str:
    total = data.get("total_analyzed", "?")
    lines = [f"Total stocks analyzed: {total}"]
    metrics = [
        ("Above 200-DMA", "pct_above_200dma", "above_200dma"),
        ("Above 20-DMA", "pct_above_20dma", "above_20dma"),
        ("At 20-Day Highs", "pct_at_20day_high", "at_20day_high"),
        ("At 20-Day Lows", "pct_at_20day_low", "at_20day_low"),
        ("At 52-Week Highs", "pct_at_52wk_high", "at_52wk_high"),
        ("At 52-Week Lows", "pct_at_52wk_low", "at_52wk_low"),
        ("At 24-Week Highs", "pct_at_24wk_high", "at_24wk_high"),
        ("At 24-Week Lows", "pct_at_24wk_low", "at_24wk_low"),
    ]
    for label, pct_key, count_key in metrics:
        pct = data.get(pct_key)
        cnt = data.get(count_key)
        if pct is not None:
            lines.append(f"  {label}: {pct:.1f}% ({cnt}/{total})")
    return "\n".join(lines)


def _format_top50(data: dict) -> str:
    lines = [f"Universe size: {data.get('universe_size', '?')}"]
    for key, label, tickers_key in [
        ("pct_below_50dma", "% Below 50-DMA", "tickers_below_50dma"),
        ("pct_3plus_dist", "% with 3+ Distribution Days", "tickers_3plus_dist"),
        ("pct_broke_20low", "% Broke 20-Day Low", "tickers_broke_20low"),
    ]:
        val = data.get(key)
        tickers = data.get(tickers_key, [])
        if val is not None:
            lines.append(f"  {label}: {val:.1f}%")
            if tickers:
                lines.append(f"    Tickers: {', '.join(tickers)}")
    return "\n".join(lines)


def _format_vix(data: dict) -> str:
    latest = (data.get("latest_df") or [{}])[0] if data.get("latest_df") else {}
    if not latest:
        return "No VIX data available"
    lines = [
        f"  VIX: {latest.get('VIX', 'N/A')}",
        f"  VIX3M: {latest.get('VIX3M', 'N/A')}",
        f"  3M/1M Ratio: {latest.get('Ratio', 'N/A')}",
        f"  Signal: {latest.get('Signal', 'Normal')}",
        f"  Date: {latest.get('Date', 'N/A')}",
    ]
    return "\n".join(lines)


def _format_price_volume(data: dict) -> str:
    rows = data.get("latest_df") or []
    if not rows:
        return "No price/volume signal data available"
    lines = []
    for r in rows:
        market = r.get("Market", r.get("MarketName", "?"))
        lines.append(
            f"  {market} ({r.get('Date', '?')}): "
            f"Close={r.get('Close', 'N/A')}, "
            f"Ret%={r.get('RetPct', 'N/A')}, "
            f"DownsideRecVol={'YES' if r.get('DownsideRecordVol') is True else 'no'}, "
            f"NewHi/LoVol={'YES' if r.get('NewHigh_LowVol') is True else 'no'}, "
            f"HiVolChurn={'YES' if r.get('HiVol_Churn') is True else 'no'}"
        )
    hits = data.get("hits_df") or []
    if hits:
        lines.append(f"\n  Recent signal hits ({len(hits)} events):")
        for r in hits[:5]:
            lines.append(f"    {r.get('Date', '?')} - {r.get('MarketName', '?')}")
    return "\n".join(lines)


@router.post("/market-technicals/analyze")
def analyze_market_technicals(req: MarketTechnicalsAnalyzeRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ConfigurationError("ANTHROPIC_API_KEY")

    breadth_text = _format_breadth(req.market_breadth)
    top50_text = _format_top50(req.top50_breadth)
    vix_text = _format_vix(req.vix_term_structure)
    pv_text = _format_price_volume(req.price_volume_signals)

    prompt = f"""You are an experienced market technician and macro strategist. Analyze the following market technical data and provide a concise but insightful overview of what it indicates about the current market environment.

S&P 500 MARKET BREADTH:
{breadth_text}

TOP 50 S&P 500 PERFORMERS — LEADERSHIP BREADTH:
{top50_text}

VIX TERM STRUCTURE (3M / 1M):
{vix_text}

PRICE/VOLUME SIGNALS:
{pv_text}

Write 4-6 flowing paragraphs of plain text (no bullet points, no markdown, no headers). Cover:
1. What the breadth readings (% above 200-DMA, 20-DMA, new highs vs new lows) say about overall market participation and health
2. What the top 50 leadership breadth signals (distribution days, breaks of 20-day lows, % below 50-DMA) suggest about trend durability and whether market leaders are holding up
3. What the VIX term structure ratio and signal imply about volatility expectations and investor sentiment
4. What the price/volume signals (downside record volume, high-volume churn, new high/low volume) indicate about institutional distribution or accumulation
5. An overall technical assessment and what to watch for

Be specific about the numbers. Write for a professional investor audience."""

    try:
        analysis, _citations, _resp = call_claude_text(
            prompt=prompt,
            model=MODEL_HAIKU,
            api_key=api_key,
            max_tokens=4096,
        )
        if not analysis:
            raise ValueError("Claude returned empty response")
    except Exception as exc:
        raise DataFetchError(source="ai_analysis", detail=str(exc)) from exc

    return {"analysis": analysis}
