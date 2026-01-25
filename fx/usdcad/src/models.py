import numpy as np
import pandas as pd
import statsmodels.api as sm

class ModelDataError(ValueError):
    pass

def fit_horizon_ols(df: pd.DataFrame, horizon: int, feature_cols: list, target_col: str = "logS"):
    d = df.copy()
    d = d.replace([np.inf, -np.inf], np.nan)

    missing = [c for c in ([target_col] + list(feature_cols)) if c not in d.columns]
    if missing:
        raise ModelDataError(f"Missing required columns: {missing}")

    d[f"y_fwd{horizon}"] = d[target_col].shift(-horizon) - d[target_col]
    X = d[feature_cols].shift(1)
    y = d[f"y_fwd{horizon}"]
    est = pd.concat([y, X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()

    if est.empty:
        last_non_na = {}
        for c in [f"y_fwd{horizon}"] + list(feature_cols):
            s = pd.concat([y, X], axis=1)[c]
            last_non_na[c] = None if s.dropna().empty else str(s.dropna().index.max().date())
        raise ModelDataError(
            "No usable rows after shifting/dropping missing values for "
            f"horizon={horizon}. Last non-missing dates: {last_non_na}"
        )

    y_train = est[f"y_fwd{horizon}"]
    X_train = sm.add_constant(est[feature_cols], has_constant="add")

    if int(y_train.shape[0]) <= (len(feature_cols) + 1):
        raise ModelDataError(
            f"Too few observations to fit OLS for horizon={horizon}: "
            f"n={int(y_train.shape[0])}, features={len(feature_cols)}"
        )

    res = sm.OLS(y_train, X_train).fit(cov_type="HAC", cov_kwds={"maxlags": max(3, horizon // 6)})
    return res, est

def predict_latest(df: pd.DataFrame, res, feature_cols: list) -> float:
    d = df.replace([np.inf, -np.inf], np.nan)
    x_all = d[feature_cols].dropna()
    if x_all.empty:
        raise ModelDataError("No non-missing feature rows available for prediction.")
    x = x_all.iloc[[-1]]
    Xp = sm.add_constant(x, has_constant="add")
    return float(res.predict(Xp).iloc[0])

def bootstrap_forecast_distribution(res, x_row: pd.DataFrame, draws: int = 2000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b = res.params.values
    x_row = x_row.replace([np.inf, -np.inf], np.nan)
    if x_row.isna().any().any():
        raise ModelDataError("x_row contains missing values; cannot bootstrap forecast distribution.")
    Xp = sm.add_constant(x_row, has_constant="add").values
    base = float(Xp @ b)
    resid = np.asarray(res.resid)
    if resid.size < 10:
        return np.full(draws, base)
    e = rng.choice(resid, size=draws, replace=True)
    return base + e
