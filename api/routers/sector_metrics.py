from fastapi import APIRouter, HTTPException
from api.cache import long_cache, get_cached, set_cached
from api.serializers import serialize_dataframe, serialize_value

router = APIRouter()


@router.get("/sector-metrics")
def get_sector_metrics():
    key = "sector_metrics"
    cached = get_cached(long_cache, key)
    if cached is not None:
        return cached
    try:
        from sector_metrics import get_data
        data = get_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    import pandas as pd
    weights_df = data.get("weights_df")
    result = {
        "weights_df": serialize_dataframe(weights_df.reset_index()) if isinstance(weights_df, pd.DataFrame) else [],
        "d_1m": data.get("d_1m"),
        "d_3m": data.get("d_3m"),
        "d_6m": data.get("d_6m"),
        "timestamp": serialize_value(data.get("timestamp")),
    }
    set_cached(long_cache, key, result)
    return result
