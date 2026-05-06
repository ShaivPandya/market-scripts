from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from api.async_job_runner import enqueue_registered_job, enqueue_response, poll_registered_job
from api.decision_state import analysis_metadata
from api.exceptions import DataFetchError
from api.portfolio_settings import get_portfolio_book_size
from api.serializers import serialize_dataframe, serialize_value
from ontology.runtime_read_service import OntologyRuntimeReadService

router = APIRouter()


class SizerPosition(BaseModel):
    ticker: str = ""
    conviction: int = 3


class SizerRequest(BaseModel):
    book: float | None = None
    target_leverage: float = 2.0
    positions: list[SizerPosition] = Field(default_factory=list)

    @model_validator(mode="after")
    def _default_book_size(self) -> SizerRequest:
        if self.book is None:
            self.book = float(get_portfolio_book_size())
        return self


def _effective_book(req: SizerRequest) -> float:
    return float(req.book) if req.book is not None else float(get_portfolio_book_size())


def _canonical_positions(req: SizerRequest) -> list[tuple[str, int]]:
    aggregated: dict[str, int] = {}
    for idx, row in enumerate(req.positions):  # noqa: B007
        ticker = str(row.ticker).strip().upper()
        conviction = int(row.conviction)
        if not ticker:
            continue
        if conviction < 1 or conviction > 5:
            raise ValueError(f"Position '{ticker}' conviction must be 1-5, got {conviction}.")
        # Take the max conviction for duplicate tickers
        aggregated[ticker] = max(aggregated.get(ticker, 0), conviction)
    if not aggregated:
        raise ValueError("No valid positions provided.")
    return sorted(aggregated.items(), key=lambda x: x[0])


def _cache_key(req: SizerRequest) -> str:
    strategy_version = "v2_conviction_sizing_equity_beta"
    canonical = _canonical_positions(req)
    token = "|".join(f"{ticker}:{conviction}" for ticker, conviction in canonical) or "none"
    return (
        f"portfolio_sizer:{strategy_version}:book={_effective_book(req):.4f}:"
        f"lev={float(req.target_leverage):.4f}:positions={token}"
    )


def _compute_sizer_result(req: SizerRequest) -> dict[str, Any]:
    try:
        from portfolio.portfolio_optimizer.portfolio_sizer import get_data

        payload = [row.model_dump() for row in req.positions]
        data = get_data(
            positions=payload,
            book=_effective_book(req),
            target_leverage=float(req.target_leverage),
        )
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(str(e)) from e

    if "error" in data and data["error"]:
        raise RuntimeError(str(data["error"]))

    import pandas as pd

    result: dict[str, Any] = {}
    for k, v in data.items():
        if k == "max_scaled" and isinstance(v, dict):
            inner: dict[str, Any] = {}
            for ik, iv in v.items():
                if isinstance(iv, pd.DataFrame):
                    inner[ik] = serialize_dataframe(iv.reset_index())
                else:
                    inner[ik] = serialize_value(iv)
            result[k] = inner
        elif isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index())
        else:
            result[k] = serialize_value(v)
    result.update(analysis_metadata(quality_state="ok"))
    return result


@router.post("/portfolio-sizer")
def run_portfolio_sizer(req: SizerRequest):
    try:
        key = _cache_key(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904

    row, _disposition = enqueue_registered_job("sizer", req.model_dump(), cache_key=key)
    return enqueue_response(row, "/api/v1/portfolio-sizer/async/{job_id}")


@router.post("/portfolio-sizer/async")
def start_portfolio_sizer(req: SizerRequest):
    try:
        key = _cache_key(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904

    row, _disposition = enqueue_registered_job("sizer", req.model_dump(), cache_key=key)
    return enqueue_response(row, "/api/v1/portfolio-sizer/async/{job_id}")


@router.get("/portfolio-sizer/async/{job_id}")
def get_portfolio_sizer_job(job_id: str):
    try:
        return poll_registered_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job_id")  # noqa: B904


@router.get("/portfolio-sizer/prefill")
def get_sizer_prefill():
    try:
        df = OntologyRuntimeReadService().positions_df()
        if "ticker" not in df.columns:
            raise ValueError("Ontology positions are missing required 'ticker' column.")

        tickers = df["ticker"].astype(str).str.strip().str.upper()
        directions = (
            df["direction"].astype(str).str.strip().str.lower()
            if "direction" in df.columns
            else pd.Series([""] * len(df))
        )
        convictions = (
            pd.to_numeric(df["conviction"], errors="coerce").fillna(3).astype(int).clip(1, 5)
            if "conviction" in df.columns
            else pd.Series([3] * len(df))
        )
        instrument_types = (
            df["instrument_type"].astype(str).str.strip().str.lower()
            if "instrument_type" in df.columns
            else pd.Series(["security"] * len(df))
        )

        deduped_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for ticker, direction, conviction, instrument_type in zip(  # noqa: B905
            tickers.tolist(),
            directions.tolist(),
            convictions.tolist(),
            instrument_types.tolist(),
        ):
            if ticker and ticker not in seen:
                seen.add(ticker)
                deduped_rows.append(
                    {
                        "ticker": ticker,
                        "conviction": conviction,
                        "direction": direction,
                        "instrument_type": instrument_type,
                    }
                )

        return {
            "positions": deduped_rows,
            "book_size": get_portfolio_book_size(),
            "source": "ontology",
            "count": len(deduped_rows),
        }
    except Exception as e:
        raise DataFetchError(source="portfolio_sizer", detail=str(e)) from e
