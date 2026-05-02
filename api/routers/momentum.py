from fastapi import APIRouter

from api.cache import get_cached, set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_dataframe, serialize_value

router = APIRouter()


@router.get("/momentum")
def get_momentum():
    key = "momentum"
    cached = get_cached(short_cache, key)
    if cached is not None:
        return cached
    try:
        from portfolio.momentum.price_momentum.momentum import get_data

        data = get_data()
    except Exception as e:
        raise DataFetchError(source="momentum", detail=str(e)) from e

    result = {}
    for k, v in data.items():
        import pandas as pd

        if k == "results" and isinstance(v, list):
            cleaned_rows = []
            for row in v:
                if isinstance(row, dict):
                    cleaned = {rk: rv for rk, rv in row.items() if rk not in {"avg20_vol_roc63", "roc63"}}
                    cleaned_rows.append(cleaned)
                else:
                    cleaned_rows.append(row)
            result[k] = serialize_value(cleaned_rows)
        elif isinstance(v, pd.DataFrame):
            df = v.reset_index()
            drop_columns = [col for col in ("avg20_vol_roc63", "roc63") if col in df.columns]
            if drop_columns:
                df = df.drop(columns=drop_columns)
            result[k] = serialize_dataframe(df)
        else:
            result[k] = serialize_value(v)

    set_cached(short_cache, key, result)
    return result
