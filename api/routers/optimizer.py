from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.serializers import serialize_dataframe, serialize_value

router = APIRouter()


class OptimizerRequest(BaseModel):
    book: int = 100_000
    target_leverage: float = 2.0


@router.post("/portfolio-optimizer")
def run_optimizer(req: OptimizerRequest):
    try:
        from portfolio_optimizer.portfolio_optimizer import get_data
        data = get_data(book=req.book, target_leverage=req.target_leverage)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "error" in data and data["error"]:
        raise HTTPException(status_code=500, detail=data["error"])

    import pandas as pd

    result = {}
    for k, v in data.items():
        if k == "max_scaled" and isinstance(v, dict):
            # max_scaled contains its own weights_df
            inner = {}
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
    return result
