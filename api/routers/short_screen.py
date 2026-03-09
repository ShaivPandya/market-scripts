from fastapi import APIRouter
from pydantic import BaseModel

from api.cache import stamp_fresh
from api.exceptions import DataFetchError
from api.serializers import serialize_dataframe, serialize_value

router = APIRouter()


class ShortScreenRequest(BaseModel):
    pb_threshold: float = 3.0
    loss_type: str = "Gross Loss"
    check_issuance: bool = False


@router.post("/short-screen")
def run_short_screen(req: ShortScreenRequest):
    try:
        from equities.short_screen.short_screen import get_data

        data = get_data(
            pb_threshold=req.pb_threshold,
            loss_type=req.loss_type,
            check_issuance=req.check_issuance,
        )
    except Exception as e:
        raise DataFetchError(source="short_screen", detail=str(e)) from e

    if data.get("error"):
        raise DataFetchError(source="short_screen", detail=data["error"])

    import pandas as pd

    result = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index(drop=True))
        else:
            result[k] = serialize_value(v)
    return stamp_fresh(result)
