"""Walk-forward price forecasting with honest baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .indicators import add_indicators

MODELS = {
    "Ridge (linear)": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=10, n_jobs=-1, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42),
}


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = add_indicators(df)
    f = pd.DataFrame(index=d.index)
    for lag in (1, 2, 3, 5, 10, 21):
        f[f"ret_{lag}"] = d.Close.pct_change(lag)
    f["rsi"] = d.RSI
    f["macd"], f["macd_sig"] = d.MACD / d.Close, d.MACD_signal / d.Close
    f["bb_pos"] = (d.Close - d.BB_low) / (d.BB_up - d.BB_low)
    f["sma20_gap"], f["sma50_gap"] = d.Close / d.SMA20 - 1, d.Close / d.SMA50 - 1
    f["vol20"] = d.Ret.rolling(20).std()
    f["atr"] = d.ATR / d.Close
    f["vol_chg"] = np.log1p(d.Volume).diff(5)
    return f.replace([np.inf, -np.inf], np.nan)


def forecast(df: pd.DataFrame, model_name: str, horizon: int = 5, test_frac: float = 0.2) -> dict:
    X = make_features(df)
    y = np.log(df.Close.shift(-horizon) / df.Close).rename("y")
    data = X.join(y)
    live_row = X.iloc[[-1]]
    lab = data.dropna()
    if len(lab) < 250 or live_row.isna().any().any():
        raise ValueError("Need at least ~1.5 years of daily data for a forecast.")

    split = int(len(lab) * (1 - test_frac))
    train, test = lab.iloc[: split - horizon], lab.iloc[split:]  # gap = no label leakage
    model = MODELS[model_name]()
    model.fit(train[X.columns], train.y)
    pred = model.predict(test[X.columns])

    pos = df.index.get_indexer(test.index)
    close_t = df.Close.iloc[pos].to_numpy()
    actual = close_t * np.exp(test.y.to_numpy())
    predicted = close_t * np.exp(pred)
    mape = lambda a, p: float(np.mean(np.abs(a - p) / a))
    backtest = pd.DataFrame({"Actual": actual, "Predicted": predicted, "Naive (no change)": close_t},
                            index=df.index[pos + horizon])
    metrics = {
        "Model MAPE": mape(actual, predicted),
        "Naive MAPE": mape(actual, close_t),
        "Directional accuracy": float(np.mean(np.sign(pred) == np.sign(test.y.to_numpy()))),
        "Test samples": len(test),
    }
    resid = test.y.to_numpy() - pred

    final = MODELS[model_name]()
    final.fit(lab[X.columns], lab.y)
    p = float(final.predict(live_row)[0])
    last = float(df.Close.iloc[-1])
    s = float(resid.std())
    est = final[-1] if hasattr(final, "steps") else final
    imp = getattr(est, "feature_importances_", None)
    if imp is None:
        imp = np.abs(est.coef_)
    return {
        "metrics": metrics, "backtest": backtest,
        "last_close": last, "price": last * np.exp(p), "low": last * np.exp(p - 1.28 * s),
        "high": last * np.exp(p + 1.28 * s),
        "target_date": pd.bdate_range(df.index[-1], periods=horizon + 1)[-1],
        "importance": pd.Series(imp, index=X.columns).sort_values(),
    }
