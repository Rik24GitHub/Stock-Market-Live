# 📈 Nifty 50 Analytics & Price Forecast (Streamlit)

Interactive dashboard for Nifty 50 stocks: technical charts, comparison against the Nifty 50 index,
risk analytics, multi-stock screener, and an ML price forecast that is **always shown next to a naive baseline**.

## Features
| Tab | What it does |
|---|---|
| Technical | Candlesticks, SMA/EMA, Bollinger, RSI, MACD, volume |
| vs Nifty 50 | Rebased performance, relative strength, rolling beta, return scatter |
| Risk & returns | CAGR, volatility, Sharpe, max drawdown, VaR, alpha/beta, monthly heatmap |
| Multi-stock | Compare stocks, correlation matrix, Nifty 50 screener (1M/3M/6M/1Y) |
| Forecast | Ridge / Random Forest / Gradient Boosting, chronological backtest, 80% range |
| Data | Table + CSV download |

## Data sources
1. **Live** (default): Yahoo Finance via `yfinance`. NSE quotes are ~15 min delayed. Cached 5 min.
2. **Kaggle CSV**: *NIFTY-50 Stock Market Data (2000 - 2021)* by Rohan Rao
   (`rohanrao/nifty50-stock-market-data`). Download `NIFTY50_all.csv`, then either upload it in the
   sidebar or save it as `data/NIFTY50_all.csv` and commit it. It ends in April 2021, so use it for
   historical analysis; use Live mode for current prices.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Push to GitHub
```bash
git init
git add .
git commit -m "Nifty 50 analytics app"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
(Create the empty repo on github.com first.)

## Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Pick your repo, branch `main`, main file path `app.py`.
4. (Optional) **Advanced settings** → choose Python 3.11 or 3.12.
5. Click **Deploy**. First build takes a few minutes. Every `git push` redeploys automatically.

## Project layout
```
app.py              UI (tabs, charts)
src/data.py         Yahoo Finance + Kaggle loaders, Nifty 50 ticker list
src/indicators.py   Indicators, drawdown, risk stats
src/models.py       Features, chronological backtest, forecast
requirements.txt    Dependencies
.streamlit/config.toml
data/               Put NIFTY50_all.csv here (optional)
```

## Notes & limitations
- Nifty 50 constituents change. Edit `NIFTY50` in `src/data.py`; any NSE ticker can also be typed in the sidebar.
- If Yahoo rate-limits the shared Streamlit Cloud IP, click **Refresh** after a minute or use Kaggle mode.
- The Kaggle file has no index series, so Kaggle mode compares against an equal-weighted proxy of the 50 stocks.
- The forecast predicts an N-day log-return from technical features. Short-term stock prediction is very hard;
  trust it only if it beats the naive baseline on the backtest. Not investment advice.
