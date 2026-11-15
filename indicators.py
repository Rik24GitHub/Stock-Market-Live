"""Technical indicators and risk statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d, c = df.copy(), df["Close"]
    for w in (20, 50, 200):
        d[f"SMA{w}"] = c.rolling(w).mean()
    d["EMA12"], d["EMA26"] = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["MACD"] = d.EMA12 - d.EMA26
    d["MACD_signal"] = d.MACD.ewm(span=9, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["RSI"] = 100 - 100 / (1 + gain / loss)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["BB_up"], d["BB_low"] = mid + 2 * sd, mid - 2 * sd
    tr = pd.concat([d.High - d.Low, (d.High - c.shift()).abs(), (d.Low - c.shift()).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["Ret"] = c.pct_change()
    return d


def drawdown(close: pd.Series) -> pd.Series:
    return close / close.cummax() - 1


def risk_stats(close: pd.Series, bench: pd.Series | None = None, rf: float = 0.065) -> dict:
    r = close.pct_change().dropna()
    cagr = (close.iloc[-1] / close.iloc[0]) ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    out = {
        "Total return": close.iloc[-1] / close.iloc[0] - 1,
        "CAGR": cagr,
        "Annualised volatility": vol,
        "Sharpe (rf 6.5%)": (cagr - rf) / vol if vol else np.nan,
        "Max drawdown": drawdown(close).min(),
        "1-day VaR (95%)": r.quantile(0.05),
        "Best day": r.max(),
        "Worst day": r.min(),
    }
    if bench is not None:
        j = pd.concat([r, bench.pct_change()], axis=1, join="inner").dropna()
        j.columns = ["s", "b"]
        if len(j) > 30:
            beta = j.cov().iloc[0, 1] / j.b.var()
            out["Beta vs Nifty"] = beta
            out["Alpha (annualised)"] = (j.s.mean() - beta * j.b.mean()) * 252
            out["Correlation vs Nifty"] = j.corr().iloc[0, 1]
    return out
