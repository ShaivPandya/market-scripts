"""
Serialize Python / pandas objects to JSON-compatible types.

All get_data() functions return dicts that may contain:
  - pd.DataFrame → list[dict]  (records orientation)
  - pd.Series    → list[{date, value}]
  - datetime / pd.Timestamp / date → ISO string
  - np.integer / np.floating → int / float  (NaN → None)
  - nested dicts / lists → recurse
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


def serialize_value(v: Any) -> Any:
    """Recursively convert a value to a JSON-serializable type."""
    if isinstance(v, pd.DataFrame):
        return serialize_dataframe(v)
    if isinstance(v, pd.Series):
        return serialize_series(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, dict):
        return {str(k): serialize_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [serialize_value(i) for i in v]
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [serialize_value(i) for i in v.tolist()]
    return v


def serialize_dataframe(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of records with serialized values.

    If the DataFrame has a named index (e.g. Ticker, Sector), reset_index()
    should be called *before* passing here so the index becomes a column.
    """
    records = []
    for _, row in df.iterrows():
        records.append({col: serialize_value(row[col]) for col in df.columns})
    return records


def serialize_series(s: pd.Series) -> list[dict]:
    """Convert a pd.Series to [{date, value}, ...] for React charting."""
    records = []
    for idx, val in s.items():
        date_str = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
        records.append({"date": date_str, "value": serialize_value(val)})
    return records


def serialize_response(data: dict) -> dict:
    """Walk every key of a get_data() return dict and serialize values."""
    return {k: serialize_value(v) for k, v in data.items()}
