#!/usr/bin/env python3
"""
evaluation/eval_baseline.py

Compute evaluation metrics for baseline CSV produced by run_rule_based_sim.py.
Handles new sustained overload and queue-growth fields; if they are missing,
it attempts to reconstruct sensible measures from available signals.
"""
import argparse
import json
import pandas as pd
from datetime import datetime

parser = argparse.ArgumentParser(description="Evaluate baseline CSV")
parser.add_argument("--input-csv", type=str, required=True)
parser.add_argument("--out-json", type=str, default=None)
args = parser.parse_args()

df = pd.read_csv(args.input_csv)
n_rows = len(df)

# ensure timestamp is datetime
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# New columns available?
has_overload_flag = "overload_flag" in df.columns
has_overload_streak = "overload_streak" in df.columns
has_queue_growth_flag = "queue_growth_flag" in df.columns

# If not available, try to reconstruct basic signals
if not has_overload_flag:
    # try to reconstruct from cpu_smoothed, latency_ms, queue_length if present
    cpu_col = "cpu_smoothed" if "cpu_smoothed" in df.columns else None
    latency_col = "latency_ms" if "latency_ms" in df.columns else None
    queue_col = "queue_length" if "queue_length" in df.columns else None

    # default thresholds (same as simulator defaults)
    cpu_thr = 90.0
    lat_thr = 1000.0
    q_thr = 10.0

    conds = []
    if cpu_col:
        conds.append(df[cpu_col] > cpu_thr)
    if latency_col:
        conds.append(df[latency_col] > lat_thr)
    if queue_col:
        conds.append(df[queue_col] > q_thr)

    if conds:
        import numpy as np
        combined = np.zeros(len(df), dtype=bool)
        for c in conds:
            combined = combined | c
        df["recon_overload_cond"] = combined
        # sustained: require 3 consecutive
        df["recon_overload_flag"] = False
        streak = 0
        for i, val in enumerate(df["recon_overload_cond"]):
            if val:
                streak += 1
            else:
                streak = 0
            if streak >= 3:
                df.at[i, "recon_overload_flag"] = True
        has_overload_flag = True
        df["overload_flag"] = df["recon_overload_flag"]
        print("Note: reconstructed overload_flag from available signals")
    else:
        # fallback: consider request_ratio > 1 if present
        if "request_ratio" in df.columns:
            df["overload_flag"] = df["request_ratio"] > 1.0
            has_overload_flag = True
            print("Note: fallback overload_flag from request_ratio > 1")

# compute sustained events and streaks using overload_flag column
overload_rows = int(df["overload_flag"].astype(int).sum())
# find longest consecutive streak of overload_flag==1
longest_streak = 0
cur = 0
for v in df["overload_flag"].astype(int):
    if v:
        cur += 1
        if cur > longest_streak:
            longest_streak = cur
    else:
        cur = 0

# count event starts (0 -> 1 transitions)
event_starts = 0
prev = 0
for v in df["overload_flag"].astype(int):
    if v and not prev:
        event_starts += 1
    prev = v

# Oscillation count (if present)
oscillations = int(df["oscillation_flag"].sum()) if "oscillation_flag" in df.columns else None

# instances / cost
avg_instances = df["instances"].mean() if "instances" in df.columns else None
total_instance_minutes = None
if avg_instances is not None:
    # sampling interval not present in CSV; assume 60s if not available
    sampling_seconds = 60
    if "sampling_interval_seconds" in df.columns:
        sampling_seconds = int(df["sampling_interval_seconds"].iloc[0])
    total_instance_minutes = (df["instances"].sum() * (sampling_seconds / 60.0)) if "instances" in df.columns else None

# queue metrics
avg_queue = df["queue_length"].mean() if "queue_length" in df.columns else None
peak_queue = df["queue_length"].max() if "queue_length" in df.columns else None

# latency
avg_latency = df["latency_ms"].mean() if "latency_ms" in df.columns else None
peak_latency = df["latency_ms"].max() if "latency_ms" in df.columns else None

# queue growth events
queue_growth_events = int(df["queue_growth_flag"].sum()) if "queue_growth_flag" in df.columns else None

summary = {
    "rows": int(n_rows),
    "sustained_overload_rows": int(overload_rows),
    "sustained_overload_event_starts": int(event_starts),
    "longest_sustained_overload_streak": int(longest_streak),
    "queue_growth_events": queue_growth_events,
    "oscillation_events": oscillations,
    "avg_instances": float(avg_instances) if avg_instances is not None else None,
    "total_instance_minutes": float(total_instance_minutes) if total_instance_minutes is not None else None,
    "avg_queue_length": float(avg_queue) if avg_queue is not None else None,
    "peak_queue_length": float(peak_queue) if peak_queue is not None else None,
    "avg_latency_ms": float(avg_latency) if avg_latency is not None else None,
    "peak_latency_ms": float(peak_latency) if peak_latency is not None else None,
    "evaluated_at": datetime.utcnow().isoformat() + "Z",
    "input_csv": args.input_csv
}

out_json = args.out_json or args.input_csv.replace(".csv", "_eval_summary.json")
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)

# pretty print
print("\n Baseline Evaluation Summary")
print("---------------------------------------")
print(f"Rows: {summary['rows']}")
print(f"Sustained overload rows: {summary['sustained_overload_rows']} ({summary['sustained_overload_rows'] / summary['rows'] * 100:.2f}%)")
print(f"Sustained overload event starts: {summary['sustained_overload_event_starts']}")
print(f"Longest sustained overload streak: {summary['longest_sustained_overload_streak']}")
print(f"Queue growth events: {summary['queue_growth_events']}")
print(f"Oscillation events: {summary['oscillation_events']}")
print(f"Average instances: {summary['avg_instances']}")
print(f"Instance-minutes (proxy): {summary['total_instance_minutes']}")
print(f"Avg queue length: {summary['avg_queue_length']}, Peak queue: {summary['peak_queue_length']}")
print(f"Avg latency (ms): {summary['avg_latency_ms']}, Peak latency (ms): {summary['peak_latency_ms']}")
print(f"Saved JSON summary: {out_json}")
