"""Tests for api/serializers.py — NaN, Inf, DataFrame, Series edge cases."""

import math
from datetime import date, datetime

import numpy as np
import pandas as pd

from api.serializers import serialize_dataframe, serialize_series, serialize_value


class TestSerializeValue:
    def test_nan_float_returns_none(self):
        assert serialize_value(float("nan")) is None

    def test_inf_float_returns_none(self):
        assert serialize_value(float("inf")) is None

    def test_neg_inf_float_returns_none(self):
        assert serialize_value(float("-inf")) is None

    def test_normal_float_passes_through(self):
        assert serialize_value(3.14) == 3.14

    def test_numpy_nan_returns_none(self):
        assert serialize_value(np.float64("nan")) is None

    def test_numpy_int_converts(self):
        result = serialize_value(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_float_converts(self):
        result = serialize_value(np.float64(2.5))
        assert result == 2.5
        assert isinstance(result, float)

    def test_numpy_bool_converts(self):
        assert serialize_value(np.bool_(True)) is True
        assert isinstance(serialize_value(np.bool_(False)), bool)

    def test_datetime_to_iso(self):
        dt = datetime(2024, 1, 15, 12, 30, 0)
        assert serialize_value(dt) == "2024-01-15T12:30:00"

    def test_date_to_iso(self):
        d = date(2024, 6, 1)
        assert serialize_value(d) == "2024-06-01"

    def test_pd_timestamp_to_iso(self):
        ts = pd.Timestamp("2024-03-01")
        assert "2024-03-01" in serialize_value(ts)

    def test_nested_dict(self):
        data = {"a": np.float64("nan"), "b": 42, "c": {"d": float("inf")}}
        result = serialize_value(data)
        assert result == {"a": None, "b": 42, "c": {"d": None}}

    def test_list_with_mixed_types(self):
        data = [1, np.int64(2), float("nan"), "hello"]
        result = serialize_value(data)
        assert result == [1, 2, None, "hello"]

    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, np.nan])
        result = serialize_value(arr)
        assert result == [1.0, 2.0, None]

    def test_string_passes_through(self):
        assert serialize_value("hello") == "hello"

    def test_none_passes_through(self):
        assert serialize_value(None) is None


class TestSerializeDataframe:
    def test_simple_dataframe(self):
        df = pd.DataFrame({"ticker": ["AAPL", "GOOG"], "price": [150.0, 140.0]})
        result = serialize_dataframe(df)
        assert len(result) == 2
        assert result[0]["ticker"] == "AAPL"
        assert result[1]["price"] == 140.0

    def test_dataframe_with_nan(self):
        df = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]})
        result = serialize_dataframe(df)
        assert result[0]["a"] == 1.0
        assert result[0]["b"] is None
        assert result[1]["a"] is None
        assert result[1]["b"] == 2.0

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = serialize_dataframe(df)
        assert result == []


class TestSerializeSeries:
    def test_date_indexed_series(self):
        idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
        s = pd.Series([100.0, 101.5], index=idx)
        result = serialize_series(s)
        assert len(result) == 2
        assert "2024-01-01" in result[0]["date"]
        assert result[0]["value"] == 100.0

    def test_series_with_nan(self):
        s = pd.Series([1.0, np.nan, 3.0], index=["a", "b", "c"])
        result = serialize_series(s)
        assert result[1]["value"] is None
