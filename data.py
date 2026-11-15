"""Data loading: live (Yahoo Finance) or the Kaggle Nifty-50 CSV."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

# Constituents change over time - edit this list whenever NSE rebalances the index.
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INFY", "INDIGO", "ITC", "JIOFIN", "JSWSTEEL",
    "KOTAKBANK", "LT", "M&M", "MARUTI", "MAXHEALTH", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN", "SUNPHARMA", "TCS",
    "TATACONSUMER", "TATAMOTORS", "TATASTEEL", "TECHM", "TITAN", "TRENT",
    "ULTRACEMCO", "WIPRO",
]
INDEX_TICKER = "^NSEI"  # NIFTY 50 index on Yahoo Finance
DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "NIFTY50_all.csv"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def to_yahoo(sym: str) -> str:
    sym = sym.strip().upper()
    return sym if sym.startswith("^") or "." in sym else sym + ".NS"


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[[c for c in OHLCV if c in df.columns]].dropna(subset=["Close"]).copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


# ---------------------------------------------------------------- live data
@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_live(symbol: str, period: str = "5y") -> pd.DataFrame:
    raw = yf.download(to_yahoo(symbol), period=period, interval="1d",
                      auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for '{symbol}'. Check the ticker.")
    return _tidy(raw)


@st.cache_data(ttl=600, show_spinner="Fetching prices for several stocks...")
def fetch_close_matrix_live(symbols: tuple, period: str = "1y") -> pd.DataFrame:
    tickers = [to_yahoo(s) for s in symbols]
    raw = yf.download(tickers, period=period, interval="1d", auto_adjust=True,
                      progress=False, threads=True)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(symbols[0])
    raw = raw.dropna(how="all")
    raw.columns = [str(c).replace(".NS", "") for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    return raw


# ------------------------------------------------------------- Kaggle data
@st.cache_data(show_spinner="Reading CSV...")
def load_kaggle(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), parse_dates=["Date"])
    missing = {"Date", "Symbol", "Close"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}. "
                         "Use NIFTY50_all.csv from the Kaggle dataset.")
    for c in ("Open", "High", "Low"):
        if c not in df:
            df[c] = df["Close"]
    if "Volume" not in df:
        df["Volume"] = 0
    return df.sort_values(["Symbol", "Date"]).reset_index(drop=True)


def kaggle_stock(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    return _tidy(df[df.Symbol == symbol].set_index("Date")[OHLCV])


def kaggle_close_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="Date", columns="Symbol", values="Close")


def kaggle_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """The Kaggle file has no index series, so build an equal-weighted Nifty proxy."""
    ret = kaggle_close_matrix(df).pct_change(fill_method=None).mean(axis=1).fillna(0)
    return ((1 + ret).cumprod() * 1000).to_frame("Close")
