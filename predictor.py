import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Stock Price Predictor (v2.0)
# -----------------------------

# 1. User input
ticker = input("Enter stock ticker (default: AAPL): ") or "AAPL"
start_date = input("Start date (YYYY-MM-DD, default: 2020-01-01): ") or "2020-01-01"
end_date = input("End date (YYYY-MM-DD, default: 2024-01-01): ") or "2024-01-01"

print(f"\n📥 Downloading {ticker} stock data...")
data = yf.download(ticker, start=start_date, end=end_date)

# 2. Prepare features and target
data["Tomorrow"] = data["Close"].shift(-1)  # next-day close
data = data.dropna()

X = data[["Open", "High", "Low", "Close", "Volume"]]
y = data["Tomorrow"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 4. Train models
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# 5. Evaluate models
lr_mae = mean_absolute_error(y_test, lr_preds)
rf_mae = mean_absolute_error(y_test, rf_preds)

print("\n📊 Model Performance:")
print(f"Linear Regression MAE: {lr_mae:.2f}")
print(f"Random Forest MAE:     {rf_mae:.2f}")

# 6. Show next day prediction
latest = X.iloc[[-1]]
lr_next = lr.predict(latest)[0]
rf_next = rf.predict(latest)[0]

print(f"\n🔮 Next day prediction (Linear Regression): {lr_next:.2f}")
print(f"🌲 Next day prediction (Random Forest):     {rf_next:.2f}")

# 7. Plot results
plt.figure(figsize=(12,6))
plt.plot(y_test.values, label="Actual", linewidth=2)
plt.plot(lr_preds, label="Linear Regression", linestyle="--")
plt.plot(rf_preds, label="Random Forest", linestyle="--")
plt.legend()
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
