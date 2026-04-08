import os
import pandas as pd
DATA_DIR = "data"

TRAIN_INPUT = os.path.join(DATA_DIR, "train_baseline.csv")
TEST_INPUT = os.path.join(DATA_DIR, "test_baseline.csv")

# ✅ Updated output names (clean ML dataset)
TRAIN_OUTPUT = os.path.join(DATA_DIR, "ml_train.csv")
TEST_OUTPUT = os.path.join(DATA_DIR, "ml_test.csv")


def prepare(df):
    df = df.copy()

    # =========================================================
    # ✅ KEEP ONLY REQUIRED FEATURES (system state)
    # =========================================================
    df = df[[
        "cpu_smoothed",
        "request_ratio",
        "network_ratio"
    ]]

    # =========================================================
    # ✅ TARGET: NEXT CPU (THIS IS CRITICAL)
    # =========================================================
    df["cpu_next"] = df["cpu_smoothed"].shift(-1)

    # =========================================================
    # ✅ CLEAN DATA
    # =========================================================
    df = df.dropna().reset_index(drop=True)

    return df


def main():
    print("Preparing FINAL ML dataset...")

    train_df = pd.read_csv(TRAIN_INPUT)
    test_df = pd.read_csv(TEST_INPUT)

    train_df = prepare(train_df)
    test_df = prepare(test_df)

    train_df.to_csv(TRAIN_OUTPUT, index=False)
    test_df.to_csv(TEST_OUTPUT, index=False)

    print("Saved:")
    print(TRAIN_OUTPUT)
    print(TEST_OUTPUT)


if __name__ == "__main__":
    main()
