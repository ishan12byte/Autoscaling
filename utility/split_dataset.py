import os
import pandas as pd

TRAIN_RATIO = 0.7
SNAPSHOT_DIR = "snapshots"
OUTPUT_DIR = "data"


def get_latest_snapshot():

    if not os.path.exists(SNAPSHOT_DIR):
        raise FileNotFoundError("snapshots directory not found")

    snapshots = [
        os.path.join(SNAPSHOT_DIR, d)
        for d in os.listdir(SNAPSHOT_DIR)
        if d.startswith("snapshot_")
    ]

    if not snapshots:
        raise RuntimeError("No snapshots found")

    snapshots.sort(reverse=True)

    latest = snapshots[0]

    print(f"Using latest snapshot: {latest}")

    return latest


def main():

    snapshot_dir = get_latest_snapshot()

    state_file = os.path.join(snapshot_dir, "state.csv")
    requests_file = os.path.join(snapshot_dir, "requests.csv")

    if not os.path.exists(state_file):
        raise FileNotFoundError("state.csv missing in snapshot")

    if not os.path.exists(requests_file):
        raise FileNotFoundError("requests.csv missing in snapshot")

    print("Loading snapshot data...")

    state_df = pd.read_csv(state_file)
    req_df = pd.read_csv(requests_file)

    state_df["timestamp"] = pd.to_datetime(state_df["timestamp"])
    req_df["timestamp"] = pd.to_datetime(req_df["timestamp"])

    state_df = state_df.sort_values("timestamp")
    req_df = req_df.sort_values("timestamp")

    merged = pd.merge_asof(
        state_df,
        req_df,
        on="timestamp",
        direction="nearest"
    )

    total_rows = len(merged)
    split_idx = int(total_rows * TRAIN_RATIO)

    print(f"Total rows: {total_rows}")
    print(f"Train rows: {split_idx}")
    print(f"Test rows: {total_rows - split_idx}")

    train_df = merged.iloc[:split_idx]
    test_df = merged.iloc[split_idx:]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df[state_df.columns].to_csv(
        os.path.join(OUTPUT_DIR, "train_state.csv"),
        index=False
    )

    train_df[req_df.columns].to_csv(
        os.path.join(OUTPUT_DIR, "train_requests.csv"),
        index=False
    )

    test_df[state_df.columns].to_csv(
        os.path.join(OUTPUT_DIR, "test_state.csv"),
        index=False
    )

    test_df[req_df.columns].to_csv(
        os.path.join(OUTPUT_DIR, "test_requests.csv"),
        index=False
    )

    print("\nDataset split complete!")
    print("Files created:")
    print("data/train_state.csv")
    print("data/train_requests.csv")
    print("data/test_state.csv")
    print("data/test_requests.csv")


if __name__ == "__main__":
    main()