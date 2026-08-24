import yfinance as yf
import pandas as pd
import numpy as np
import datetime

from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier


def predict_stock(symbol):

    # =========================
    # 1. DOWNLOAD DATA
    # =========================
    today = datetime.datetime.today()
    start_date = "2000-01-01"
    end_date = today.strftime("%Y-%m-%d")

    data = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return {
            "signal": "NO DATA",
            "recent": pd.DataFrame(),
            "accuracy": None,
            "report": None
        }

    data.columns = data.columns.get_level_values(0)

    # Keep full copy for display later
    full_data = data.copy()

    # =========================
    # 2. FEATURE ENGINEERING (FOR TRAINING)
    # =========================
    data["RSI"] = RSIIndicator(data["Close"], window=14).rsi()

    macd = MACD(data["Close"])
    data["MACD"] = macd.macd()
    data["MACD_signal"] = macd.macd_signal()

    data["SMA_20"] = SMAIndicator(data["Close"], window=20).sma_indicator()

    # =========================
    # 3. 10-DAY LABEL
    # =========================
    data["future_return"] = data["Close"].shift(-10) / data["Close"] - 1
    data["direction"] = np.where(data["future_return"] > 0, 1, 0)

    train_data = data.dropna().copy()

    if len(train_data) < 300:
        return {
            "signal": "NOT ENOUGH DATA",
            "recent": pd.DataFrame(),
            "accuracy": None,
            "report": None
        }

    # =========================
    # 4. FEATURES & TARGET
    # =========================
    features = ["RSI", "MACD", "MACD_signal", "SMA_20"]
    X = train_data[features]
    y = train_data["direction"]

    # =========================
    # 5. SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # =========================
    # 6. SCALE
    # =========================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =========================
    # 7. CLASS WEIGHTS
    # =========================
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    sample_weights = y_train.map(dict(zip(classes, weights)))

    # =========================
    # 8. MODEL
    # =========================
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

    # =========================
    # 9. EVALUATION
    # =========================
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["DOWN", "UP"]
    )

    # =========================
    # 9b. EXTRA METRICS
    # =========================
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    precision_up = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall_up = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_up = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== {symbol} — MODEL PERFORMANCE =====")
    print(f"Baseline accuracy (majority class): {baseline_acc*100:.2f}%")
    print(f"Model accuracy: {accuracy*100:.2f}%")
    print(f"Lift over baseline: {(accuracy-baseline_acc)*100:.2f} pts")
    print(f"Precision (BUY reliability): {precision_up*100:.2f}%")
    print(f"Recall: {recall_up*100:.2f}%")
    print(f"F1: {f1_up*100:.2f}%")
    print(f"ROC-AUC: {auc:.3f}")
    print(f"Confusion Matrix:\n{cm}")
    print("=========================================\n")

    # =========================
    # 10. DISPLAY USING FULL DATA
    # =========================
    full_data["RSI"] = RSIIndicator(full_data["Close"], window=14).rsi()
    macd_full = MACD(full_data["Close"])
    full_data["MACD"] = macd_full.macd()
    full_data["MACD_signal"] = macd_full.macd_signal()
    full_data["SMA_20"] = SMAIndicator(full_data["Close"], window=20).sma_indicator()

    recent = full_data.iloc[-30:].copy()

    # Predict only where features are available
    valid_rows = recent.dropna(subset=features)

    recent.loc[valid_rows.index, "Prediction"] = model.predict(
        scaler.transform(valid_rows[features])
    )

    label_map = {0: "DOWN", 1: "UP"}
    recent["Prediction"] = recent["Prediction"].map(label_map)

    # =========================
    # 11. TODAY SIGNAL (TRUE LAST DATE)
    # =========================
    latest = full_data.iloc[-1:].copy()

    latest["RSI"] = full_data["RSI"].iloc[-1]
    latest["MACD"] = full_data["MACD"].iloc[-1]
    latest["MACD_signal"] = full_data["MACD_signal"].iloc[-1]
    latest["SMA_20"] = full_data["SMA_20"].iloc[-1]

    latest_scaled = scaler.transform(latest[features])
    today_pred = model.predict(latest_scaled)[0]

    return {
        "signal": label_map[today_pred],
        "recent": recent[["Close", "Prediction"]],
        "accuracy": round(accuracy * 100, 2),
        "baseline_accuracy": round(baseline_acc * 100, 2),
        "precision": round(precision_up * 100, 2),
        "recall": round(recall_up * 100, 2),
        "f1": round(f1_up * 100, 2),
        "roc_auc": round(auc, 3),
        "confusion_matrix": cm.tolist(),
        "report": report
    }

if __name__ == "__main__":
    for symbol in ["RELIANCE.NS", "TCS.NS", "AAPL"]:
        predict_stock(symbol)
