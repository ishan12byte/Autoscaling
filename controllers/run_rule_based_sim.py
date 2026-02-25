#!/usr/bin/env python3
# controllers/run_rule_based_sim.py

import sys
import os
import math
import csv
import json
import warnings
from datetime import datetime, timezone

import pandas as pd
import argparse

# allow importing config from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings as cfg  # type: ignore

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="Baseline simulator (controller).")
parser.add_argument("--state-file", type=str, default=getattr(cfg, "STATE_FILE", "data/state.csv"))
parser.add_argument("--requests-file", type=str, default=getattr(cfg, "REQUESTS_FILE", "data/requests.csv"))
parser.add_argument("--output-csv", type=str, default=os.path.join(getattr(cfg, "DATA_DIR", "data"), "baseline_simulation.csv"))
parser.add_argument("--metrics-out", type=str, default=os.path.join(getattr(cfg, "DATA_DIR", "data"), "baseline_metrics.json"))
parser.add_argument("--max-instances", type=int, default=getattr(cfg, "MAX_INSTANCES", 4))
parser.add_argument("--min-instances", type=int, default=getattr(cfg, "MIN_INSTANCES", 1))
parser.add_argument("--workers-per-instance", type=int, default=getattr(cfg, "WORKERS_PER_INSTANCE", 1))
parser.add_argument("--k", type=float, default=getattr(cfg, "K", 1.25))
parser.add_argument("--alpha", type=float, default=getattr(cfg, "ALPHA", 0.25))
parser.add_argument("--inefficiency", type=float, default=getattr(cfg, "INEFFICIENCY", 0.85))
parser.add_argument("--scale-up-streak", type=int, default=getattr(cfg, "SCALE_UP_STREAK", 3))
parser.add_argument("--scale-down-streak", type=int, default=getattr(cfg, "SCALE_DOWN_STREAK", 5))
parser.add_argument("--cooldown-steps", type=int, default=getattr(cfg, "COOLDOWN_STEPS", 3))
parser.add_argument("--base-latency-ms", type=float, default=getattr(cfg, "BASE_LATENCY", 50.0))
parser.add_argument("--sampling-interval-seconds", type=int, default=getattr(cfg, "SAMPLING_INTERVAL", 60))
parser.add_argument("--merge-tolerance-multiplier", type=float, default=1.1)
parser.add_argument("--queue-persistence", action="store_true")
parser.add_argument("--clamp-load-ratio", type=float, default=5.0)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

# ---------- params ----------
STATE_FILE = args.state_file
REQUESTS_FILE = args.requests_file
OUTPUT_FILE = args.output_csv
METRICS_OUT = args.metrics_out

MAX_INSTANCES = max(1, args.max_instances)
MIN_INSTANCES = max(1, min(args.min_instances, MAX_INSTANCES))
WORKERS_PER_INSTANCE = max(1, args.workers_per_instance)

K = float(args.k)
ALPHA = float(args.alpha)
INEFFICIENCY = float(args.inefficiency)

SCALE_UP_STREAK = int(args.scale_up_streak)
SCALE_DOWN_STREAK = int(args.scale_down_streak)
COOLDOWN_STEPS = int(args.cooldown_steps)

BASE_LATENCY = float(args.base_latency_ms)
SAMPLING_INTERVAL = int(args.sampling_interval_seconds)
MERGE_TOL = pd.Timedelta(seconds=int(SAMPLING_INTERVAL * args.merge_tolerance_multiplier))

QUEUE_PERSISTENCE = bool(args.queue_persistence)
LOAD_RATIO_CLAMP = float(args.clamp_load_ratio)
VERBOSE = bool(args.verbose)

# ---------- defensive checks ----------
if not os.path.exists(STATE_FILE):
    raise FileNotFoundError(f"state CSV not found: {STATE_FILE}")

if INEFFICIENCY <= 0 or INEFFICIENCY > 1.0:
    raise ValueError("inefficiency must be in (0,1].")

# ---------- load & align ----------
state_df = pd.read_csv(STATE_FILE)
if "timestamp" not in state_df.columns:
    raise KeyError("state CSV must include 'timestamp' column")
state_df["timestamp"] = pd.to_datetime(state_df["timestamp"], utc=True)
state_df = state_df.sort_values("timestamp").reset_index(drop=True)

if os.path.exists(REQUESTS_FILE):
    req_df = pd.read_csv(REQUESTS_FILE)
    if "timestamp" not in req_df.columns or "requests_per_min" not in req_df.columns:
        raise KeyError("requests CSV must include 'timestamp' and 'requests_per_min'")
    req_df["timestamp"] = pd.to_datetime(req_df["timestamp"], utc=True)
    req_df = req_df.sort_values("timestamp").reset_index(drop=True)
else:
    req_df = pd.DataFrame({"timestamp": state_df["timestamp"], "requests_per_min": 0.0})
    if VERBOSE:
        print("requests file missing — using zero requests for all timestamps.")

merged = pd.merge_asof(state_df, req_df, on="timestamp", direction="nearest", tolerance=MERGE_TOL)
n_na = merged["requests_per_min"].isna().sum()
if n_na:
    warnings.warn(f"{n_na} rows had no nearby requests within tolerance ({MERGE_TOL}); filling with 0.")
merged["requests_per_min"] = merged["requests_per_min"].fillna(0.0)

required_cols = ["avg_cpu_5", "avg_net_in_5", "avg_net_out_5", "requests_per_min"]
nan_rows = merged[required_cols].isna().any(axis=1).sum()
if nan_rows:
    warnings.warn(f"Dropping {nan_rows} rows with NaNs in required numeric columns.")
    merged = merged.dropna(subset=required_cols).reset_index(drop=True)
if merged.empty:
    raise RuntimeError("No rows available after merging/cleaning — nothing to simulate.")

# ---------- simulation state ----------
instances = max(MIN_INSTANCES, 1)
cpu_prev = None
high_streak = 0
low_streak = 0
cooldown_timer = 0
prev_action = None
queue_prev = 0.0

oscillation_count = 0
overload_count = 0
total_instance_minutes = 0.0
peak_queue = 0.0
latency_accum = 0.0

n_rows = len(merged)

# ---------- prepare output ----------
headers = [
    "timestamp",
    "requests",
    "instances",
    "capacity",
    "effective_capacity",
    "load_ratio",
    "cpu_target",
    "cpu_smoothed",
    "latency_ms",
    "queue_length",
    "high_streak",
    "low_streak",
    "cooldown_timer",
    "action",
    "prev_action",
    "oscillation_flag",
    "overload_flag"
]
file_exists = os.path.exists(OUTPUT_FILE)
out_f = open(OUTPUT_FILE, "a", newline="")
writer = csv.writer(out_f)
if not file_exists:
    writer.writerow(headers)

# ---------- init cpu_prev ----------
first_row = merged.iloc[0]
init_requests = float(first_row.get("requests_per_min", 0.0))
init_capacity = instances * WORKERS_PER_INSTANCE
init_effective_capacity = max(0.1, init_capacity * INEFFICIENCY)
init_load_ratio = init_requests / init_effective_capacity if init_effective_capacity > 0 else 0.0
init_load_ratio = min(init_load_ratio, LOAD_RATIO_CLAMP)
init_cpu_target = 100.0 * (1.0 - math.exp(-K * init_load_ratio))
cpu_prev = min(100.0, init_cpu_target)

# ---------- main loop ----------
for idx, row in merged.iterrows():
    timestamp = row["timestamp"]
    requests = float(row.get("requests_per_min", 0.0))

    capacity = instances * WORKERS_PER_INSTANCE
    effective_capacity = max(0.1, capacity * INEFFICIENCY)

    load_ratio = requests / effective_capacity if effective_capacity > 0 else float("inf")
    load_ratio = min(load_ratio, LOAD_RATIO_CLAMP)

    cpu_target = 100.0 * (1.0 - math.exp(-K * load_ratio))
    cpu_target = min(cpu_target, 100.0)

    cpu_smoothed = ALPHA * cpu_target + (1.0 - ALPHA) * cpu_prev
    cpu_smoothed = min(max(cpu_smoothed, 0.0), 100.0)

    # queue persistence optional
    if QUEUE_PERSISTENCE:
        queue_length = max(0.0, queue_prev + requests - effective_capacity)
    else:
        queue_length = max(0.0, requests - effective_capacity)
    queue_prev = queue_length

    latency_ms = BASE_LATENCY * (1.0 + load_ratio ** 2)

    # streak logic
    if cpu_smoothed > 70.0:
        high_streak += 1
        low_streak = 0
    elif cpu_smoothed < 30.0:
        low_streak += 1
        high_streak = 0
    else:
        high_streak = 0
        low_streak = 0

    action = "hold"
    # cooldown handling
    if cooldown_timer > 0:
        cooldown_timer -= 1
    else:
        if high_streak >= SCALE_UP_STREAK:
            new_instances = min(MAX_INSTANCES, instances + 1)
            if new_instances != instances:
                action = "scale_up"
            instances = new_instances
            high_streak = 0
            cooldown_timer = COOLDOWN_STEPS
        elif low_streak >= SCALE_DOWN_STREAK:
            new_instances = max(MIN_INSTANCES, instances - 1)
            if new_instances != instances:
                action = "scale_down"
            instances = new_instances
            low_streak = 0
            cooldown_timer = COOLDOWN_STEPS

    # oscillation detection
    oscillation_flag = 0
    if prev_action is not None:
        if (prev_action == "scale_up" and action == "scale_down") or \
           (prev_action == "scale_down" and action == "scale_up"):
            oscillation_flag = 1
            oscillation_count += 1

    # set prev_action when an action occurs
    if action != "hold":
        prev_action = action

    overload_flag = int(load_ratio > 1.0)
    overload_count += overload_flag

    total_instance_minutes += instances * (SAMPLING_INTERVAL / 60.0)
    peak_queue = max(peak_queue, queue_length)
    latency_accum += latency_ms

    writer.writerow([
        timestamp.isoformat(),
        round(requests, 4),
        instances,
        capacity,
        round(effective_capacity, 4),
        round(load_ratio, 4),
        round(cpu_target, 4),
        round(cpu_smoothed, 4),
        round(latency_ms, 4),
        round(queue_length, 4),
        int(high_streak),
        int(low_streak),
        int(cooldown_timer),
        action,
        prev_action if prev_action is not None else "",
        int(oscillation_flag),
        int(overload_flag)
    ])

    cpu_prev = cpu_smoothed

out_f.close()

# ---------- summary ----------
avg_instances = (total_instance_minutes / ((SAMPLING_INTERVAL/60.0) * n_rows)) if n_rows > 0 else 0.0
avg_latency = (latency_accum / n_rows) if n_rows > 0 else 0.0

summary = {
    "rows_simulated": n_rows,
    "overload_events": int(overload_count),
    "oscillation_count": int(oscillation_count),
    "total_instance_minutes": round(total_instance_minutes, 3),
    "avg_instances": round(avg_instances, 3),
    "peak_queue_length": round(peak_queue, 3),
    "avg_latency_ms": round(avg_latency, 3),
    "parameters": {
        "max_instances": MAX_INSTANCES,
        "min_instances": MIN_INSTANCES,
        "workers_per_instance": WORKERS_PER_INSTANCE,
        "k": K,
        "alpha": ALPHA,
        "inefficiency": INEFFICIENCY,
        "scale_up_streak": SCALE_UP_STREAK,
        "scale_down_streak": SCALE_DOWN_STREAK,
        "cooldown_steps": COOLDOWN_STEPS,
        "queue_persistence": QUEUE_PERSISTENCE,
        "sampling_interval_seconds": SAMPLING_INTERVAL
    },
    "output_csv": OUTPUT_FILE
}

with open(METRICS_OUT, "w") as mf:
    json.dump(summary, mf, indent=2)

print("\n✅ Baseline simulation finished")
print("----------------------------------------")
print(f"Rows simulated:         {n_rows}")
print(f"Overload events:        {summary['overload_events']}")
print(f"Oscillation_count:      {summary['oscillation_count']}")
print(f"Total instance-minutes: {summary['total_instance_minutes']}")
print(f"Average instances:      {summary['avg_instances']}")
print(f"Peak queue length:      {summary['peak_queue_length']}")
print(f"Average latency (ms):   {summary['avg_latency_ms']}")
print(f"CSV output:             {OUTPUT_FILE}")
print(f"Summary JSON:           {METRICS_OUT}")
if VERBOSE:
    print("\nParameters used:")
    print(json.dumps(summary["parameters"], indent=2))
