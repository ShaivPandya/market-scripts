"""Simple forecast model wrappers for aluminum backtests."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class AluminumForecastModel(Protocol):
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> AluminumForecastModel: ...

    def predict(self, X_test: pd.DataFrame) -> np.ndarray: ...


class ZeroReturnForecast:
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> ZeroReturnForecast:
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X_test), dtype=float)


class SklearnForecastModel:
    def __init__(self, estimator) -> None:
        self.estimator = estimator

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> SklearnForecastModel:
        self.estimator.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(X_test), dtype=float)


def make_model(model_type: str, *, random_seed: int = 42) -> AluminumForecastModel:
    normalized = model_type.strip().lower().replace("-", "_")
    if normalized in {"zero", "zero_return", "baseline"}:
        return ZeroReturnForecast()

    if normalized == "ridge":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return SklearnForecastModel(
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                Ridge(alpha=1.0, random_state=random_seed),
            )
        )

    if normalized in {"random_forest", "randomforest", "rf"}:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline

        return SklearnForecastModel(
            make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=3,
                    random_state=random_seed,
                    n_jobs=1,
                ),
            )
        )

    raise ValueError(f"Unsupported aluminum model type: {model_type}")
