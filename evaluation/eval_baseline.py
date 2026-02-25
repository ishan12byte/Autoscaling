#!/usr/bin/env python3
# evaluation/eval_baseline.py

import sys
import os
import json
import argparse
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings as cfg  # type: ignore

parser = argparse.ArgumentParser(description="Evaluate baseline simulation output.")
parser.add_argument("--input-csv", type=str, default=os.path.join(getattr(cfg, "DATA_DIR", "data"), "baseline_simulation.csv"))
parser.add_argument("--output-json", type=str, default=os.path.join(getattr(cfg, "DATA_DIR", "data"), "baseline_eval_summary.json"))
args = parser.parse_args()

INPUT = args.input_csv
OUT_JSON = args.output_json

if not os.path.exists(INPUT):
    raise FileNotFoundError(f"Baseline CSV not found: {INPUT}")

df = pd.read_csv(INPUT)

# required columns check
required = ["instances", "overload_flag", "oscillation_flag", "queue_length", "latency_ms"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Evaluator expected columns missing from {INPUT}: {missing}")

report = {}

# Overload metrics
report["rows"] = len(df)
report["overload_events"] = int(df["overload_flag"].sum())
report["overload_pct"] = float(100.0 * df["overload_flag"].sum() / len(df))

# Longest overload streak
streak = 0
max_streak = 0
for f in df["overload_flag"]:
    if int(f) == 1:
        streak += 1
        max_streak = max(max_streak, streak)
    else:
        streak = 0
report["longest_overload_streak"] = int(max_streak)

# Cost metrics
report["avg_instances"] = float(df["instances"].mean())
report["peak_instances"] = int(df["instances"].max())
# instance-minutes proxy from rows (assuming sampling interval is consistent)
report["instance_minutes_proxy"] = float(df["instances"].sum())

# Stability metrics
report["oscillation_events"] = int(df["oscillation_flag"].sum())
report["oscillation_rate_pct"] = float(100.0 * df["oscillation_flag"].sum() / len(df))

# Queue & latency
report["avg_queue_length"] = float(df["queue_length"].mean())
report["peak_queue_length"] = float(df["queue_length"].max())
report["avg_latency_ms"] = float(df["latency_ms"].mean())
report["peak_latency_ms"] = float(df["latency_ms"].max())

# Save JSON
with open(OUT_JSON, "w") as f:
    json.dump(report, f, indent=2)

# Print summary
print("\n Baseline Evaluation Summary")
print("---------------------------------------")
print(f"Rows: {report['rows']}")
print(f"Overload events: {report['overload_events']} ({report['overload_pct']:.2f}%)")
print(f"Longest overload streak: {report['longest_overload_streak']}")
print(f"Average instances: {report['avg_instances']:.2f}")
print(f"Peak instances: {report['peak_instances']}")
print(f"Instance-minutes (proxy): {report['instance_minutes_proxy']}")
print(f"Oscillation events: {report['oscillation_events']} ({report['oscillation_rate_pct']:.2f}%)")
print(f"Avg queue length: {report['avg_queue_length']:.2f}, Peak queue: {report['peak_queue_length']:.2f}")
print(f"Avg latency (ms): {report['avg_latency_ms']:.2f}, Peak latency (ms): {report['peak_latency_ms']:.2f}")
print(f"Saved JSON summary: {OUT_JSON}")