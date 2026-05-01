from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from api.async_job_runner import enqueue_registered_job, enqueue_response, poll_registered_job
from api.serializers import serialize_dataframe, serialize_value

router = APIRouter()


class AnalyzerRequest(BaseModel):
    # Legacy optimizer fields are accepted for backward compatibility and ignored by analyzer logic.
    book: float | None = None
    target_leverage: float | None = None
    beta_neutral: bool | None = None


def _cache_key(_req: AnalyzerRequest) -> str:
    strategy_version = "v1_signal_factor_table"
    return f"portfolio_analyzer:{strategy_version}"


def _compute_analyzer_result(req: AnalyzerRequest) -> dict[str, Any]:
    try:
        from portfolio.portfolio_optimizer.portfolio_analyzer import get_data

        data = get_data(
            book=req.book,
            target_leverage=req.target_leverage,
            beta_neutral=True if req.beta_neutral is None else req.beta_neutral,
        )
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
    return result


@router.post("/portfolio-analyzer")
@router.post("/portfolio-optimizer")
def run_analyzer(req: AnalyzerRequest = Body(default_factory=AnalyzerRequest)):  # noqa: B008
    row, _disposition = enqueue_registered_job("analyzer", req.model_dump(), cache_key=_cache_key(req))
    return enqueue_response(row, "/api/v1/portfolio-analyzer/async/{job_id}")


@router.post("/portfolio-analyzer/async")
@router.post("/portfolio-optimizer/async")
def start_analyzer(req: AnalyzerRequest = Body(default_factory=AnalyzerRequest)):  # noqa: B008
    """
    Start an analyzer job and return a job_id quickly.
    """
    row, _disposition = enqueue_registered_job("analyzer", req.model_dump(), cache_key=_cache_key(req))
    return enqueue_response(row, "/api/v1/portfolio-analyzer/async/{job_id}")


@router.get("/portfolio-analyzer/async/{job_id}")
@router.get("/portfolio-optimizer/async/{job_id}")
def get_analyzer_job(job_id: str):
    try:
        return poll_registered_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job_id")  # noqa: B904
