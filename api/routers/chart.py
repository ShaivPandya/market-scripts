from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.serializers import serialize_dataframe, serialize_value

router = APIRouter()


class ChartRequest(BaseModel):
    ticker: str
    lookback: str = "2Y"


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
