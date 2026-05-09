import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

TRAIN_FILE = "data/ml_train.csv"
TEST_FILE = "data/ml_test.csv"

# Saved Linear Regression model
MODEL_FILE = "models/lr_dynamics.pkl"

FEATURES = [
    "cpu_smoothed",
    "request_ratio",
    "network_ratio"
]

TARGET = "cpu_next"


def main():
    print("Loading data...")

    # Load datasets
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # Training data
    X_train = train_df[FEATURES].astype(float)
    y_train = train_df[TARGET].astype(float)

    # Testing data
    X_test = test_df[FEATURES].astype(float)
    y_test = test_df[TARGET].astype(float)

    # Add constant for OLS intercept
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    print("\nTraining OLS (clean dynamics)...")

    # Train Linear Regression / OLS
    model = sm.OLS(y_train, X_train).fit()

    # Print summary
    print("\nOLS Summary:")
    print(model.summary())

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nMetrics:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    # Coefficients
    print("\nModel Coefficients:")
    for feature, coef in zip(X_train.columns, model.params):
        print(f"{feature}: {coef:.4f}")

    # Save trained model
    joblib.dump(model, MODEL_FILE)

    print(f"\nModel saved: {MODEL_FILE}")


if __name__ == "__main__":
    main()
