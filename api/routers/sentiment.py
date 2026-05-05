import os

from fastapi import APIRouter
from pydantic import BaseModel

from api.cache import get_or_set_cached, long_cache, short_cache
from api.exceptions import ConfigurationError, DataFetchError
from api.serializers import serialize_value
from llm_utils import MODEL_LOW, api_key_env, call_llm_text, has_llm_api_key

router = APIRouter()


class SentimentAnalyzeRequest(BaseModel):
    put_call: dict
    surveys: dict
    volatility: list


def _format_put_call(data: dict) -> str:
    equity = data.get("equity") or {}
    spy = data.get("spy") or {}
    qqq = data.get("qqq") or {}
    iwm = data.get("iwm") or {}
    lines = []
    if equity:
        lines.append(f"  Equity Aggregate P/C Ratio: {equity.get('ratio', 'N/A')}")
    for ticker, d in [("SPY", spy), ("QQQ", qqq), ("IWM", iwm)]:
        if d:
            lines.append(
                f"  {ticker}: ratio={d.get('ratio', 'N/A')}, calls={d.get('calls', 'N/A')}, puts={d.get('puts', 'N/A')}"
            )
    if not lines:
        return "No put/call data available"
    return "\n".join(lines)


def _format_surveys(data: dict) -> str:
    aaii = data.get("aaii") or []
    naaim = data.get("naaim") or []
    lines = []
    if aaii:
        latest = aaii[-1]
        lines.append(
            f"  AAII (latest): Bull={latest.get('bull', 'N/A')}%, "
            f"Bear={latest.get('bear', 'N/A')}%, "
            f"Neutral={latest.get('neutral', 'N/A')}%, "
            f"Spread={latest.get('spread', 'N/A')}%"
        )
        if len(aaii) >= 4:
            spreads = [r.get("spread") for r in aaii[-4:] if r.get("spread") is not None]
            if spreads:
                lines.append(f"  AAII Bull-Bear spread (last 4 weeks): {[round(s, 1) for s in spreads]}")
    if naaim:
        latest = naaim[-1]
        lines.append(f"  NAAIM Exposure (latest, week of {latest.get('date', '?')}): {latest.get('exposure', 'N/A')}")
    if not lines:
        return "No survey data available"
    return "\n".join(lines)


def _format_volatility(data: list) -> str:
    if not data:
        return "No volatility data available"
    latest = data[-1]
    return (
        f"  VIX={latest.get('vix', 'N/A')}, "
        f"VXN={latest.get('vxn', 'N/A')}, "
        f"VVIX={latest.get('vvix', 'N/A')} "
        f"(as of {latest.get('date', '?')})"
    )


@router.post("/sentiment/analyze")
def analyze_sentiment(req: SentimentAnalyzeRequest):
    if not has_llm_api_key():
        raise ConfigurationError(api_key_env())

    pc_text = _format_put_call(req.put_call)
    surveys_text = _format_surveys(req.surveys)
    vol_text = _format_volatility(req.volatility)

    prompt = f"""You are an experienced market strategist and sentiment analyst. Analyze the following market sentiment data and provide a concise but insightful overview of what it signals about the current investor psychology and risk environment.

PUT/CALL RATIOS (options market sentiment):
{pc_text}

INVESTOR SURVEYS (AAII & NAAIM):
{surveys_text}

VOLATILITY INDICES (VIX, VXN, VVIX):
{vol_text}

Write 2-3 flowing paragraphs of plain text (no bullet points, no markdown, no headers). Be concise. Cover:
1. What put/call ratios and volatility indices indicate about hedging demand and near-term fear or complacency
2. What AAII sentiment and NAAIM exposure say about retail and institutional positioning
3. An overall sentiment assessment — whether the market appears fearful, complacent, or neutral, and any contrarian signals

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


@router.get("/sentiment/put-call")
def get_put_call(lookback_days: int = 180):
    key = f"sentiment_put_call:{lookback_days}"

    def loader():
        try:
            from macro.sentiment.sentiment import get_put_call

            data = get_put_call(lookback_days=lookback_days)
        except Exception as e:
            raise DataFetchError(source="sentiment_put_call", detail=str(e)) from e
        return serialize_value(data)

    return get_or_set_cached(short_cache, key, loader)


@router.get("/sentiment/surveys")
def get_surveys():
    key = "sentiment_surveys"

    def loader():
        try:
            from macro.sentiment.sentiment import get_surveys

            data = get_surveys()
        except Exception as e:
            raise DataFetchError(source="sentiment_surveys", detail=str(e)) from e
        return serialize_value(data)

    return get_or_set_cached(long_cache, key, loader)


@router.get("/sentiment/volatility")
def get_volatility(lookback_days: int = 365):
    key = f"sentiment_volatility:{lookback_days}"

    def loader():
        try:
            from macro.sentiment.sentiment import get_volatility

            data = get_volatility(lookback_days=lookback_days)
        except Exception as e:
            raise DataFetchError(source="sentiment_volatility", detail=str(e)) from e
        return serialize_value(data)

    return get_or_set_cached(short_cache, key, loader)
