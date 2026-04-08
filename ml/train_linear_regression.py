import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

TRAIN_FILE = "data/ml_train.csv"
TEST_FILE = "data/ml_test.csv"

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

    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    print("\nTraining OLS (clean dynamics)...")

    model = sm.OLS(y_train, X_train).fit()

    print("\nOLS Summary:")
    print(model.summary())

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nMetrics:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")


if __name__ == "__main__":
    main()