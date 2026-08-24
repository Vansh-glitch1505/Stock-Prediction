# 📈 Stock Signal Predictor

A machine learning pipeline that predicts short-term stock price direction (UP/DOWN) using technical indicators and XGBoost, deployed as an interactive Streamlit dashboard.

> ⚠️ **Disclaimer:** Built for educational purposes only. Not financial advice.

---

## 🧠 How It Works

1. **Data Collection** — Downloads historical OHLCV data for any stock ticker via `yfinance` (NSE, NASDAQ, etc.)
2. **Feature Engineering** — Computes technical indicators as model features:
   - **RSI** (Relative Strength Index, 14-day)
   - **MACD** & MACD Signal Line
   - **SMA-20** (20-day Simple Moving Average)
3. **Labeling** — Defines the target as whether the stock's closing price is higher 10 trading days later (binary UP/DOWN classification)
4. **Model Training** — Trains an `XGBoostClassifier` with:
   - Time-aware train/test split (no shuffling — respects chronological order)
   - Balanced class weighting to handle skewed UP/DOWN distributions
5. **Evaluation** — Benchmarks the model against a majority-class baseline using accuracy, precision, recall, F1, and ROC-AUC
6. **Deployment** — Serves live predictions through a Streamlit dashboard with price charts and a BUY/SELL/HOLD signal

---
![](https://github.com/Vansh-glitch1505/Stock-Prediction/blob/main/WhatsApp%20Image%202026-08-25%20at%2003.00.29.jpeg?raw=true)
![](https://github.com/Vansh-glitch1505/Stock-Prediction/blob/main/WhatsApp%20Image%202026-08-25%20at%2003.00.29%20(1).jpeg?raw=true)
## 🛠 Tech Stack

`Python` · `XGBoost` · `scikit-learn` · `yfinance` · `ta` (technical analysis) · `Streamlit` · `Pandas` / `NumPy`

---

## 📊 Model Performance

Evaluated on a chronological hold-out test set (20% of history), benchmarked against a naive majority-class baseline:

| Ticker | Baseline Accuracy | Model Accuracy | Precision (UP) | Recall (UP) | ROC-AUC |
|--------|-------------------|-----------------|------------------|--------------|---------|
| TCS.NS | 50.21% | **55.19%** | 55.50% | 54.29% | 0.557 |
| RELIANCE.NS | 53.07% | 47.99% | 58.97% | 6.56% | 0.582 |
| AAPL | 56.91% | 43.84% | 61.36% | 3.56% | 0.485 |

**Takeaway:** Performance is ticker-dependent. On TCS.NS, the model beats a naive baseline by ~5 points with balanced precision/recall, showing the technical-indicator feature set (RSI, MACD, SMA-20) captures a modest but real directional signal. On RELIANCE.NS and AAPL, the model underperforms the baseline — likely because these indicators alone aren't sufficient in all market regimes, and the low recall suggests the model defaults toward predicting "DOWN" more often on those tickers.

**Next steps to improve robustness:** add volume/volatility-based features, test alternate prediction horizons, and evaluate across a larger, sector-diversified basket of tickers rather than relying on any single ticker's result.

---

## 🚀 Running Locally

### 1. Clone the repo
```bash
git clone https://github.com/Vansh-glitch1505/Stock-Prediction.git
cd Stock-Prediction
```

### 2. Install dependencies
```bash
pip install streamlit yfinance pandas numpy ta scikit-learn xgboost
```

### 3. Launch the dashboard
```bash
streamlit run main.py
```

### 4. (Optional) Run model evaluation directly
To see accuracy/precision/recall/ROC-AUC printed to the terminal without the UI:
```bash
python predict.py
```

---

## 📂 Project Structure
```
Stock-Prediction/
├── main.py       # Streamlit dashboard (UI, charts, signal display)
├── predict.py    # Data pipeline, feature engineering, model training & evaluation
└── README.md
```

---

## 📌 Notes

- Predictions are generated fresh on each run (no persisted/cached model) — every request retrains on the latest available data.
- The dashboard accepts any Yahoo Finance–compatible ticker (e.g., `RELIANCE.NS`, `TCS.NS`, `AAPL`, `MSFT`).
