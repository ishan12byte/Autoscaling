import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "data"
MODEL_DIR = "models"

TRAIN_INPUT = os.path.join(DATA_DIR, "train_baseline.csv")
TEST_INPUT = os.path.join(DATA_DIR, "test_baseline.csv")

TRAIN_OUTPUT = os.path.join(DATA_DIR, "ml_train_dataset.csv")
TEST_OUTPUT = os.path.join(DATA_DIR, "ml_test_dataset.csv")

SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")


FEATURES = [
    "requests",
    "net_throughput",
    "instances",
    "cpu_smoothed",
    "queue_length",
    "latency_ms",
    "request_ratio",
    "network_ratio",
    "cpu_trend",
    "queue_trend",
    "cpu_per_instance",
    "high_streak",
    "low_streak"
]


def engineer_features(df):

    df["cpu_trend"] = df["cpu_smoothed"].diff().fillna(0)

    df["queue_trend"] = df["queue_length"].diff().fillna(0)

    df["cpu_per_instance"] = df["cpu_smoothed"] / df["instances"]

    df["cpu_next"] = df["cpu_smoothed"].shift(-1)

    df = df.dropna()

    return df


def process_dataset(df, scaler, fit=False):

    X = df[FEATURES]
    y = df["cpu_next"]

    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=FEATURES)

    dataset = X_scaled.copy()
    dataset["cpu_next"] = y.values

    return dataset


def main():

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading train dataset...")

    train_df = pd.read_csv(TRAIN_INPUT)
    train_df = engineer_features(train_df)

    if os.path.exists(SCALER_FILE):

        print("Scaler already exists — loading existing scaler")

        scaler = joblib.load(SCALER_FILE)

        train_dataset = process_dataset(train_df, scaler, fit=False)

    else:

        print("Fitting new scaler")

        scaler = StandardScaler()

        train_dataset = process_dataset(train_df, scaler, fit=True)

        joblib.dump(scaler, SCALER_FILE)

    train_dataset.to_csv(TRAIN_OUTPUT, index=False)

    print("Train dataset saved:", TRAIN_OUTPUT)

    print("Loading test dataset...")

    test_df = pd.read_csv(TEST_INPUT)
    test_df = engineer_features(test_df)

    test_dataset = process_dataset(test_df, scaler, fit=False)

    test_dataset.to_csv(TEST_OUTPUT, index=False)

    print("Test dataset saved:", TEST_OUTPUT)

    print("Scaler location:", SCALER_FILE)


if __name__ == "__main__":
    main()