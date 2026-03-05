#!/usr/bin/env python3
# controllers/run_rule_based_sim.py
"""
Net-aware rule-based baseline simulator (NO ML / NO OLS).
Reads state + requests, simulates capacity & CPU, applies rule-based scaling,
logs per-step CSV and produces a summary JSON.

Usage example:
python controllers/run_rule_based_sim.py \
  --state-file data/train_state.csv \
  --requests-file data/train_requests.csv \
  --output-csv data/train_baseline.csv
"""
import sys
import os
import math
import csv
import json
import warnings
from datetime import datetime, timezone

import pandas as pd
import argparse

# allow import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings as cfg  # type: ignore

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="Net-aware rule-based baseline simulator (no ML).")
parser.add_argument("--state-file", type=str, default=getattr(cfg, "STATE_FILE", "data/state.csv"))
parser.add_argument("--requests-file", type=str, default=getattr(cfg, "REQUESTS_FILE", "data/requests.csv"))
parser.add_argument("--output-csv", type=str, default=os.path.join(getattr(cfg, "DATA_DIR", "data"), "baseline_simulation.csv"))
parser.add_argument("--metrics-out", type=str, default=os.path.join(getattr(cfg, "DATA_DIR", "data"), "baseline_metrics.json"))
parser.add_argument("--max-instances", type=int, default=getattr(cfg, "MAX_INSTANCES", 4))
parser.add_argument("--min-instances", type=int, default=getattr(cfg, "MIN_INSTANCES", 1))
parser.add_argument("--workers-per-instance", type=int, default=getattr(cfg, "WORKERS_PER_INSTANCE", 1))
parser.add_argument("--k-req", type=float, default=1.25, help="K for request-driven CPU")
parser.add_argument("--k-net", type=float, default=1.0, help="K for network-driven CPU")
parser.add_argument("--w-req", type=float, default=0.7, help="Weight for request CPU contribution")
parser.add_argument("--w-net", type=float, default=0.3, help="Weight for network CPU contribution")
parser.add_argument("--alpha", type=float, default=getattr(cfg, "ALPHA", 0.25))
parser.add_argument("--inefficiency", type=float, default=getattr(cfg, "INEFFICIENCY", 0.85))
parser.add_argument("--bandwidth-per-instance", type=float, default=None,
                    help="Bandwidth capacity per instance in same units as avg_net_* columns. If not set, auto-estimated.")
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

K_REQ = float(args.k_req)
K_NET = float(args.k_net)
W_REQ = float(args.w_req)
W_NET = float(args.w_net)
ALPHA = float(args.alpha)
INEFFICIENCY = float(args.inefficiency)

BANDWIDTH_PER_INSTANCE = args.bandwidth_per_instance  # None => auto-estimate

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
if not (0.0 <= W_REQ <= 1.0 and 0.0 <= W_NET <= 1.0):
    raise ValueError("w-req and w-net must be between 0 and 1")

# normalize weights
w_sum = W_REQ + W_NET
if w_sum == 0:
    W_REQ, W_NET = 0.7, 0.3
else:
    W_REQ, W_NET = W_REQ / w_sum, W_NET / w_sum

# ---------- load & align ----------
state_df = pd.read_csv(STATE_FILE)
if "timestamp" not in state_df.columns:
    raise KeyError("state CSV must include 'timestamp' column")
state_df["timestamp"] = pd.to_datetime(state_df["timestamp"], utc=True)
state_df = state_df.sort_values("timestamp").reset_index(drop=True)

# ensure net columns exist
if "avg_net_in_5" not in state_df.columns:
    state_df["avg_net_in_5"] = 0.0
if "avg_net_out_5" not in state_df.columns:
    state_df["avg_net_out_5"] = 0.0

if os.path.exists(REQUESTS_FILE):
    req_df = pd.read_csv(REQUESTS_FILE)
    if "timestamp" not in req_df.columns or "requests_per_min" not in req_df.columns:
        raise KeyError("requests CSV must include 'timestamp' and 'requests_per_min'")
    req_df["timestamp"] = pd.to_datetime(req_df["timestamp"], utc=True)
    req_df = req_df.sort_values("timestamp").reset_index(drop=True)
else:
    req_df = pd.DataFrame({"timestamp": state_df["timestamp"], "requests_per_min": 0.0})
    if VERBOSE:
        print("requests file missing — using zero requests")

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

n_rows = len(merged)
if VERBOSE:
    print(f"Rows for simulation: {n_rows}")

# ---------- estimate bandwidth_per_instance if not provided ----------
if BANDWIDTH_PER_INSTANCE is None:
    net_throughput_series = (merged["avg_net_in_5"].fillna(0.0) + merged["avg_net_out_5"].fillna(0.0))
    mean_throughput = float(net_throughput_series.mean()) if len(net_throughput_series) > 0 else 1.0
    BANDWIDTH_PER_INSTANCE = max(1.0, mean_throughput / max(1, MAX_INSTANCES))
    if VERBOSE:
        print(f"Auto-estimated bandwidth_per_instance = {BANDWIDTH_PER_INSTANCE:.3f}")

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

# ---------- output CSV ----------
headers = [
    "timestamp",
    "requests",
    "instances",
    "capacity",
    "effective_capacity",
    "request_ratio",
    "net_throughput",
    "network_capacity",
    "network_ratio",
    "cpu_req_target",
    "cpu_net_target",
    "cpu_target_combined",
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
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
file_exists = os.path.exists(OUTPUT_FILE)
out_f = open(OUTPUT_FILE, "w", newline="")
writer = csv.writer(out_f)
writer.writerow(headers)

# ---------- init cpu_prev ----------
first_row = merged.iloc[0]
init_requests = float(first_row.get("requests_per_min", 0.0))
init_capacity = instances * WORKERS_PER_INSTANCE
init_effective_capacity = max(0.1, init_capacity * INEFFICIENCY)
init_request_ratio = init_requests / init_effective_capacity if init_effective_capacity > 0 else 0.0
init_request_ratio = min(init_request_ratio, LOAD_RATIO_CLAMP)
init_net_throughput = float(first_row.get("avg_net_in_5", 0.0)) + float(first_row.get("avg_net_out_5", 0.0))
init_network_capacity = max(0.1, instances * BANDWIDTH_PER_INSTANCE * INEFFICIENCY)
init_network_ratio = init_net_throughput / init_network_capacity if init_network_capacity > 0 else 0.0
init_network_ratio = min(init_network_ratio, LOAD_RATIO_CLAMP)

init_cpu_req = 100.0 * (1.0 - math.exp(-K_REQ * init_request_ratio))
init_cpu_net = 100.0 * (1.0 - math.exp(-K_NET * init_network_ratio))
init_cpu_combined = W_REQ * init_cpu_req + W_NET * init_cpu_net
cpu_prev = min(100.0, init_cpu_combined)

# ---------- main loop ----------
for idx, row in merged.iterrows():
    timestamp = row["timestamp"]
    requests = float(row.get("requests_per_min", 0.0))
    net_in = float(row.get("avg_net_in_5", 0.0))
    net_out = float(row.get("avg_net_out_5", 0.0))
    net_throughput = net_in + net_out

    capacity = instances * WORKERS_PER_INSTANCE
    effective_capacity = max(0.1, capacity * INEFFICIENCY)

    # request-driven ratio
    request_ratio = requests / effective_capacity if effective_capacity > 0 else float("inf")
    request_ratio = min(request_ratio, LOAD_RATIO_CLAMP)

    # network capacity and ratio
    network_capacity = max(0.1, instances * BANDWIDTH_PER_INSTANCE * INEFFICIENCY)
    network_ratio = net_throughput / network_capacity if network_capacity > 0 else float("inf")
    network_ratio = min(network_ratio, LOAD_RATIO_CLAMP)

    # CPU contributions
    cpu_req_target = 100.0 * (1.0 - math.exp(-K_REQ * request_ratio))
    cpu_req_target = min(cpu_req_target, 100.0)

    cpu_net_target = 100.0 * (1.0 - math.exp(-K_NET * network_ratio))
    cpu_net_target = min(cpu_net_target, 100.0)

    cpu_target_combined = W_REQ * cpu_req_target + W_NET * cpu_net_target
    cpu_target_combined = min(cpu_target_combined, 100.0)

    # smoothing
    cpu_smoothed = ALPHA * cpu_target_combined + (1.0 - ALPHA) * cpu_prev
    cpu_smoothed = min(max(cpu_smoothed, 0.0), 100.0)

    # queue persistence optional
    if QUEUE_PERSISTENCE:
        queue_length = max(0.0, queue_prev + requests - effective_capacity)
    else:
        queue_length = max(0.0, requests - effective_capacity)
    queue_prev = queue_length

    # latency proxy
    latency_ms = BASE_LATENCY * (1.0 + (request_ratio + network_ratio) ** 2)

    # streak/hysteresis
    if cpu_smoothed > 70.0:
        high_streak += 1
        low_streak = 0
    elif cpu_smoothed < 30.0:
        low_streak += 1
        high_streak = 0
    else:
        high_streak = 0
        low_streak = 0

    # action decision with cooldown
    action = "hold"
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

    prev_action_logged = prev_action if prev_action is not None else ""
    if action != "hold":
        prev_action = action

    # overload if either resource ratio > 1
    overload_flag = int((request_ratio > 1.0) or (network_ratio > 1.0))
    overload_count += overload_flag

    # costs & accumulators
    total_instance_minutes += instances * (SAMPLING_INTERVAL / 60.0)
    peak_queue = max(peak_queue, queue_length)
    latency_accum += latency_ms

    # write row
    writer.writerow([
        timestamp.isoformat(),
        round(requests, 4),
        instances,
        capacity,
        round(effective_capacity, 4),
        round(request_ratio, 4),
        round(net_throughput, 4),
        round(network_capacity, 4),
        round(network_ratio, 4),
        round(cpu_req_target, 4),
        round(cpu_net_target, 4),
        round(cpu_target_combined, 4),
        round(cpu_smoothed, 4),
        round(latency_ms, 4),
        round(queue_length, 4),
        int(high_streak),
        int(low_streak),
        int(cooldown_timer),
        action,
        prev_action_logged,
        int(oscillation_flag),
        int(overload_flag)
    ])

    # advance
    cpu_prev = cpu_smoothed

# close and summarize
out_f.close()

avg_instances = (total_instance_minutes / ((SAMPLING_INTERVAL / 60.0) * n_rows)) if n_rows > 0 else 0.0
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
        "k_req": K_REQ,
        "k_net": K_NET,
        "w_req": W_REQ,
        "w_net": W_NET,
        "alpha": ALPHA,
        "inefficiency": INEFFICIENCY,
        "bandwidth_per_instance": BANDWIDTH_PER_INSTANCE,
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

print("\n Baseline simulation finished")
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

