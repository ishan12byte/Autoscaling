import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

TRAIN_FILE = "data/ml_train.csv"
TEST_FILE = "data/ml_test.csv"

MODEL_FILE = "models/rf_dynamics.pkl"

FEATURES = [
    "cpu_smoothed",
    "request_ratio",
    "network_ratio"
]

TARGET = "cpu_next"

def main():
    print("Loading data...")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    X_train = train_df[FEATURES].astype(float)
    y_train = train_df[TARGET].astype(float)

    X_test = test_df[FEATURES].astype(float)
    y_test = test_df[TARGET].astype(float)

    print("\nTraining Random Forest (clean dynamics)...")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nMetrics:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    print("\nFeature Importance:")