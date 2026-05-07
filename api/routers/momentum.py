import hashlib
import json
from typing import Any

from fastapi import APIRouter

from api.cache import get_or_set_cached, short_cache
from api.exceptions import DataFetchError
from api.serializers import serialize_dataframe, serialize_value

router = APIRouter()


def _current_position_records() -> list[dict[str, Any]]:
    try:
        from ontology.runtime_read_service import get_positions_df

        df = get_positions_df()
        if df.empty:
            return []

        columns = [col for col in ("ticker", "direction", "price_symbol", "instrument_type") if col in df.columns]
        if not columns:
            return []
        records = serialize_value(df[columns].fillna("").to_dict(orient="records"))
        return records if isinstance(records, list) else []
    except Exception:
        return []


def _positions_cache_token() -> str:
    try:
        encoded = json.dumps(_current_position_records(), sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        encoded = repr(_current_position_records())
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


@router.get("/momentum")
def get_momentum():
    key = f"momentum:v2:{_positions_cache_token()}"

    def loader():
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

        return result

    return get_or_set_cached(short_cache, key, loader)
