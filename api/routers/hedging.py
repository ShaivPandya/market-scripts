from __future__ import annotations

import math
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.async_job_runner import enqueue_registered_job, enqueue_response, poll_registered_job
from api.decision_state import analysis_metadata
from api.exceptions import ConfigurationError, DataFetchError
from api.serializers import serialize_dataframe, serialize_value
from llm_utils import MODEL_LOW, api_key_env, call_llm_text, has_llm_api_key

router = APIRouter()


class HedgingPosition(BaseModel):
    ticker: str = ""
    weight: float


class HedgingRequest(BaseModel):
    book: float = 100_000
    positions: list[HedgingPosition] = Field(default_factory=list)


def _canonical_positions(req: HedgingRequest) -> list[tuple[str, float]]:
    aggregated: dict[str, float] = {}
    for idx, row in enumerate(req.positions):
        ticker = str(row.ticker).strip().upper()
        weight = float(row.weight)
        if not math.isfinite(weight):
            raise ValueError(f"Position '{ticker}' has a non-finite weight.")
        if not ticker:
            raise ValueError(f"Position at index {idx} has an empty ticker.")
        aggregated[ticker] = aggregated.get(ticker, 0.0) + weight
    if not aggregated:
        raise ValueError("No valid positions provided.")
    return sorted(aggregated.items(), key=lambda x: x[0])


def _cache_key(req: HedgingRequest) -> str:
    strategy_version = "v1_signed_all_positions"
    canonical = _canonical_positions(req)
    token = "|".join(f"{ticker}:{weight:.12g}" for ticker, weight in canonical) or "none"
    return f"hedging_tool:{strategy_version}:book={float(req.book):.4f}:positions={token}"


def _compute_hedging_result(req: HedgingRequest) -> dict[str, Any]:
    try:
        from portfolio.portfolio_optimizer.hedging_tool import get_data

        payload = [row.model_dump() for row in req.positions]
        data = get_data(positions=payload, book=float(req.book))
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(str(e)) from e

    if "error" in data and data["error"]:
        raise RuntimeError(str(data["error"]))

    import pandas as pd

    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index())
        else:
            result[k] = serialize_value(v)
    result.update(analysis_metadata(quality_state="ok"))
    return result


@router.post("/hedging-tool")
def run_hedging_tool(req: HedgingRequest):
    try:
        key = _cache_key(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904
    row, _disposition = enqueue_registered_job("hedging", req.model_dump(), cache_key=key)
    return enqueue_response(row, "/api/v1/hedging-tool/async/{job_id}")


@router.post("/hedging-tool/async")
def start_hedging_tool(req: HedgingRequest):
    try:
        key = _cache_key(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904
    row, _disposition = enqueue_registered_job("hedging", req.model_dump(), cache_key=key)
    return enqueue_response(row, "/api/v1/hedging-tool/async/{job_id}")


@router.get("/hedging-tool/async/{job_id}")
def get_hedging_tool_job(job_id: str):
    try:
        return poll_registered_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job_id")  # noqa: B904


@router.get("/hedging-tool/prefill")
def get_hedging_tool_prefill():
    try:
        from portfolio.portfolio_db import get_positions_df

        df = get_positions_df()
        if "ticker" not in df.columns:
            raise ValueError("Portfolio database is missing required 'ticker' column.")

        tickers = df["ticker"].astype(str).str.strip().str.upper()
        tickers = [t for t in tickers.tolist() if t]
        # Preserve order while deduplicating.
        deduped = list(dict.fromkeys(tickers))

        return {
            "positions": [{"ticker": t, "weight": 0.0} for t in deduped],
            "source": "portfolio.db",
            "count": len(deduped),
        }
    except Exception as e:
        raise DataFetchError(source="hedging_tool", detail=str(e)) from e


@router.get("/hedging-tool/portfolio-weights")
def get_portfolio_weights(book: float = 100_000):
    """Derive portfolio weights from the portfolio DB for use in the hedging tool."""
    try:
        from portfolio.portfolio_optimizer.hedging_tool import derive_portfolio_weights

        positions, metadata, suggested_book = derive_portfolio_weights(book)
        return {
            "positions": positions,
            "metadata": metadata,
            "book": suggested_book,
            "source": "portfolio_db",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise DataFetchError(source="hedging_tool", detail=str(e)) from e


class HedgingRecommendRequest(BaseModel):
    net_beta_spy: float | None = None
    net_beta_iwm: float | None = None
    post_hedge_beta_spy: float | None = None
    post_hedge_beta_iwm: float | None = None
    gross_input: float | None = None
    net_input: float | None = None
    gross_after_hedging: float | None = None
    volatility_after_hedging: float | None = None
    hedge_spy_weight: float | None = None
    hedge_iwm_weight: float | None = None
    positions_df: list[dict] = Field(default_factory=list)
    hedges_df: list[dict] = Field(default_factory=list)
    book_size: float | None = None


def _fmt_pct(v: float | None, signed: bool = True) -> str:
    if v is None:
        return "N/A"
    pct = v * 100
    sign = "+" if signed and pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def _build_recommend_prompt(req: HedgingRecommendRequest) -> str:
    lines: list[str] = []
    lines.append("## Portfolio Hedging Analysis\n")

    lines.append("### Summary Metrics")
    lines.append(f"- Gross Exposure (Pre-Hedge): {_fmt_pct(req.gross_input, signed=False)}")
    lines.append(f"- Net Exposure (Pre-Hedge): {_fmt_pct(req.net_input)}")
    lines.append(
        f"- Pre-Hedge Beta SPY: {req.net_beta_spy:.4f}" if req.net_beta_spy is not None else "- Pre-Hedge Beta SPY: N/A"
    )
    lines.append(
        f"- Pre-Hedge Beta IWM: {req.net_beta_iwm:.4f}" if req.net_beta_iwm is not None else "- Pre-Hedge Beta IWM: N/A"
    )
    lines.append(
        f"- Post-Hedge Beta SPY: {req.post_hedge_beta_spy:.4f}"
        if req.post_hedge_beta_spy is not None
        else "- Post-Hedge Beta SPY: N/A"
    )
    lines.append(
        f"- Post-Hedge Beta IWM: {req.post_hedge_beta_iwm:.4f}"
        if req.post_hedge_beta_iwm is not None
        else "- Post-Hedge Beta IWM: N/A"
    )
    lines.append(f"- Gross After Hedging: {_fmt_pct(req.gross_after_hedging, signed=False)}")
    lines.append(f"- Daily Volatility (Post-Hedge): {_fmt_pct(req.volatility_after_hedging, signed=False)}")
    if req.book_size:
        lines.append(f"- Book Size: ${req.book_size:,.0f}")
    lines.append("")

    if req.positions_df:
        lines.append("### Position Details")
        header = f"{'Ticker':<8} {'Dir':<6} {'Weight':>8} {'Beta SPY':>10} {'Beta IWM':>10} {'Beta Ctb SPY':>14} {'Beta Ctb IWM':>14}"
        lines.append(header)
        lines.append("-" * len(header))
        for p in req.positions_df:
            ticker = str(p.get("ticker", ""))
            direction = str(p.get("direction", ""))
            weight = p.get("weight")
            beta_spy = p.get("beta_spy")
            beta_iwm = p.get("beta_iwm")
            beta_c_spy = p.get("beta_contribution_spy")
            beta_c_iwm = p.get("beta_contribution_iwm")
            col_bs = f"{beta_spy:>10.3f}" if isinstance(beta_spy, (int, float)) else f"{'N/A':>10}"
            col_bi = f"{beta_iwm:>10.3f}" if isinstance(beta_iwm, (int, float)) else f"{'N/A':>10}"
            col_cs = f"{beta_c_spy:>14.4f}" if isinstance(beta_c_spy, (int, float)) else f"{'N/A':>14}"
            col_ci = f"{beta_c_iwm:>14.4f}" if isinstance(beta_c_iwm, (int, float)) else f"{'N/A':>14}"
            lines.append(f"{ticker:<8} {direction:<6} {_fmt_pct(weight):>8} {col_bs} {col_bi} {col_cs} {col_ci}")
        lines.append("")

    if req.hedges_df:
        lines.append("### Hedge Legs")
        for h in req.hedges_df:
            ticker = str(h.get("ticker", ""))
            weight = h.get("weight")
            shares = h.get("shares")
            dollar = h.get("dollar_weight")
            lines.append(
                f"- {ticker}: weight={_fmt_pct(weight)}, "
                f"shares={int(shares) if isinstance(shares, (int, float)) else 'N/A'}, "
                f"dollar={f'${dollar:,.0f}' if isinstance(dollar, (int, float)) else 'N/A'}"
            )
        lines.append("")

    data_block = "\n".join(lines)

    return f"""You are a portfolio risk analyst advising a professional investor. Based on the hedging analysis below, provide candidate hedge adjustments for decision support.

{data_block}

Provide 4-6 concise decision-support notes covering:
1. Whether the hedge sizing is appropriate and any adjustments needed
2. Positions with outsized beta contributions that could be reviewed for a possible internal proposal
3. Whether portfolio directionality (net long/short bias) is well-balanced
4. Concentration risk — any single position dominating the risk profile
5. Post-hedge volatility assessment and whether it is acceptable
6. Candidate adjustments to review, using internal proposal language rather than order or broker execution language

Be direct and specific. Reference actual tickers and numbers. Do not describe any item as an executable trade or order. Write for a professional investor who wants signal, not noise. Use plain text paragraphs, no markdown headers or bullet points."""


@router.post("/hedging-tool/recommend")
def recommend_hedging_adjustments(req: HedgingRecommendRequest):
    if not has_llm_api_key():
        raise ConfigurationError(api_key_env())

    if not req.positions_df:
        raise HTTPException(status_code=400, detail="No position data provided for recommendations.")

    prompt = _build_recommend_prompt(req)

    try:
        analysis, _citations, _resp = call_llm_text(
            prompt=prompt,
            model=MODEL_LOW,
            api_key=None,
            max_tokens=4096,
        )
        if not analysis:
            raise ValueError("LLM returned empty response")
    except Exception as exc:
        raise DataFetchError(source="hedging_recommend", detail=str(exc)) from exc

    return {
        "analysis": analysis,
        **analysis_metadata(quality_state="ok"),
    }
