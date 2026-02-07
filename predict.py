import yfinance as yf
import pandas as pd
import numpy as np

from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier


def predict_stock(symbol):
    # =========================
    # 1. DOWNLOAD DATA
    # =========================
    data = yf.download(symbol, period="max", progress=False)
    data.columns = data.columns.get_level_values(0)

    if len(data) < 300:
        print("❌ Not enough data")
        return {
            "signal": "NOT ENOUGH DATA",
            "recent": pd.DataFrame(),
            "accuracy": None,
            "report": None
        }

    close = data["Close"]

    # =========================
    # 2. FEATURE ENGINEERING
    # =========================
    data["RSI"] = RSIIndicator(close, window=14).rsi()

    macd = MACD(close)
    data["MACD"] = macd.macd()
    data["MACD_signal"] = macd.macd_signal()

    data["SMA_20"] = SMAIndicator(close, window=20).sma_indicator()

    # =========================
    # 3. 5-DAY DIRECTION LABEL
    # =========================
    data["future_return"] = data["Close"].shift(-10) / data["Close"] - 1

    # Binary direction
    data["direction"] = np.where(data["future_return"] > 0, 1, 0)
    # 1 = UP, 0 = DOWN

    data.dropna(inplace=True)

    # =========================
    # 4. FEATURES & TARGET
    # =========================
    features = ["RSI", "MACD", "MACD_signal", "SMA_20"]
    X = data[features]
    y = data["direction"]

    # =========================
    # 5. TIME-AWARE SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # =========================
    # 6. SCALE FEATURES
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

    # ✅ PRINT TO TERMINAL
    print("\n================ MODEL PERFORMANCE ================")
    print(f"Symbol: {symbol}")
    print(f"5-Day Directional Accuracy: {accuracy * 100:.2f}%")
    print("---------------------------------------------------")
    print(report)
    print("===================================================\n")

    # =========================
    # 10. LAST 30 DAYS PREDICTION
    # =========================
    label_map = {0: "DOWN", 1: "UP"}

    recent = data.iloc[-30:].copy()
    recent["Prediction"] = model.predict(
        scaler.transform(recent[features])
    )
    recent["Prediction"] = recent["Prediction"].map(label_map)

    # =========================
    # 11. TODAY SIGNAL
    # =========================
    latest_features = pd.DataFrame(
        [recent.iloc[-1][features]],
        columns=features
    )
    latest_scaled = scaler.transform(latest_features)
    today_pred = model.predict(latest_scaled)[0]

    # =========================
    # RETURN FOR STREAMLIT
    # =========================
    return {
        "signal": label_map[today_pred],
        "recent": recent[["Close", "Prediction"]],
        "accuracy": round(accuracy * 100, 2),
        "report": report
    }
