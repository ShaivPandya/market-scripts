from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.serializers import serialize_dataframe, serialize_value

router = APIRouter()


class ChartRequest(BaseModel):
    ticker: str
    lookback: str = "2Y"


class RatioChartRequest(BaseModel):
    symbol_a: str
    symbol_b: str
    start_date: str | None = None
    end_date: str | None = None
    method: str = "price_ratio"


@router.post("/chart")
def run_chart(req: ChartRequest):
    ticker = req.ticker.strip().upper()
    try:
        from technical_analysis import get_data
        data = get_data(ticker, lookback=req.lookback)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "error" in data and data["error"]:
        raise HTTPException(status_code=400, detail=data["error"])

    import pandas as pd

    result = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index())
        else:
            result[k] = serialize_value(v)
    return result


@router.post("/chart/ratio")
def run_chart_ratio(req: RatioChartRequest):
    symbol_a = req.symbol_a.strip().upper()
    symbol_b = req.symbol_b.strip().upper()
    if not symbol_a or not symbol_b:
        raise HTTPException(status_code=400, detail="Both symbol_a and symbol_b are required.")

    try:
        from technical_analysis import get_ratio_data

        data = get_ratio_data(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            start_date=req.start_date,
            end_date=req.end_date,
            method=req.method,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "error" in data and data["error"]:
        raise HTTPException(status_code=400, detail=data["error"])

    import pandas as pd

    result = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index())
        else:
            result[k] = serialize_value(v)
    return result
