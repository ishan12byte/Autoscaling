#!/usr/bin/env python3
# controllers/run_rule_based_sim.py
"""
Net-aware baseline simulator + two-pass bootstrap OLS predictor.

Workflow:
 - Merge state + requests
 - PASS 1: simulate with trend predictor (no OLS). Produces simulated instances series.
 - Train OLS using merged data and PASS1 instances series.
 - PASS 2: simulate again with OLS predictor enabled (uses trained coeffs).
 - Write final CSV and metrics JSON (summary includes predictor MAE).

This keeps things simple, deterministic, and ensures the predictor sees the actual simulated
instances used during scaling.
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

# optional numpy for OLS predictor; fallback allowed
try:
    import numpy as np
except Exception:
    np = None

# allow import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings as cfg  # type: ignore

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="Net-aware baseline simulator with two-pass OLS bootstrap.")
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
parser.add_argument("--pred-horizon", type=int, default=5)
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
PRED_HORIZON = int(args.pred_horizon)
VERBOSE = bool(args.verbose)

# ---------- defensive checks ----------
if not os.path.exists(STATE_FILE):
    raise FileNotFoundError(f"state CSV not found: {STATE_FILE}")
if INEFFICIENCY <= 0 or INEFFICIENCY > 1.0:
    raise ValueError("inefficiency must be in (0,1].")
if PRED_HORIZON < 1:
    raise ValueError("pred-horizon must be >= 1")
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

# ---------- helper: simulation pass (returns row list + instances series) ----------
def simulate_pass(use_ols, coeffs=None, training_instances=None, write_rows=False):
    """
    Run a simulation pass.
    - use_ols: whether predict_cpu_for_row will use coeffs
    - coeffs: numpy array (or None)
    - training_instances: list/array used only for training building when required (not used inside pass)
    - write_rows: if True, write rows to OUTPUT_FILE; otherwise return rows in memory
    Returns:
      rows_out: list of dicts (per-row)
      instances_series: list of instances used per timestep
      pred_mae_accum, pred_count (for OLS MAE calc during pass)
    """
    instances_local = max(MIN_INSTANCES, 1)
    cpu_prev_local = None
    high_streak_local = 0
    low_streak_local = 0
    cooldown_timer_local = 0
    prev_action_local = None
    queue_prev_local = 0.0

    oscillation_count_local = 0
    overload_count_local = 0
    total_instance_minutes_local = 0.0
    peak_queue_local = 0.0
    latency_accum_local = 0.0

    rows_out = []
    instances_series = []
    pred_mae_acc = 0.0
    pred_cnt = 0

    for idx, row in merged.iterrows():
        timestamp = row["timestamp"]
        requests = float(row.get("requests_per_min", 0.0))
        net_in = float(row.get("avg_net_in_5", 0.0))
        net_out = float(row.get("avg_net_out_5", 0.0))
        net_throughput = net_in + net_out

        capacity = instances_local * WORKERS_PER_INSTANCE
        effective_capacity = max(0.1, capacity * INEFFICIENCY)

        # request-driven ratio
        request_ratio = requests / effective_capacity if effective_capacity > 0 else float("inf")
        request_ratio = min(request_ratio, LOAD_RATIO_CLAMP)

        # network capacity and ratio
        network_capacity = max(0.1, instances_local * BANDWIDTH_PER_INSTANCE * INEFFICIENCY)
        network_ratio = net_throughput / network_capacity if network_capacity > 0 else float("inf")
        network_ratio = min(network_ratio, LOAD_RATIO_CLAMP)

        # CPU contributions
        cpu_req_target = 100.0 * (1.0 - math.exp(-K_REQ * request_ratio))
        cpu_req_target = min(cpu_req_target, 100.0)

        cpu_net_target = 100.0 * (1.0 - math.exp(-K_NET * network_ratio))
        cpu_net_target = min(cpu_net_target, 100.0)

        cpu_target_combined = W_REQ * cpu_req_target + W_NET * cpu_net_target
        cpu_target_combined = min(cpu_target_combined, 100.0)

        if cpu_prev_local is None:
            cpu_prev_local = cpu_target_combined

        cpu_smoothed_local = ALPHA * cpu_target_combined + (1.0 - ALPHA) * cpu_prev_local
        cpu_smoothed_local = min(max(cpu_smoothed_local, 0.0), 100.0)

        # queue persistence optional
        if QUEUE_PERSISTENCE:
            queue_length = max(0.0, queue_prev_local + requests - effective_capacity)
        else:
            queue_length = max(0.0, requests - effective_capacity)
        queue_prev_local = queue_length

        # latency proxy
        latency_ms = BASE_LATENCY * (1.0 + (request_ratio + network_ratio) ** 2)

        # streak/hysteresis
        if cpu_smoothed_local > 70.0:
            high_streak_local += 1
            low_streak_local = 0
        elif cpu_smoothed_local < 30.0:
            low_streak_local += 1
            high_streak_local = 0
        else:
            high_streak_local = 0
            low_streak_local = 0

        # action decision with cooldown
        action = "hold"
        if cooldown_timer_local > 0:
            cooldown_timer_local -= 1
        else:
            if high_streak_local >= SCALE_UP_STREAK:
                new_instances_local = min(MAX_INSTANCES, instances_local + 1)
                if new_instances_local != instances_local:
                    action = "scale_up"
                instances_local = new_instances_local
                high_streak_local = 0
                cooldown_timer_local = COOLDOWN_STEPS
            elif low_streak_local >= SCALE_DOWN_STREAK:
                new_instances_local = max(MIN_INSTANCES, instances_local - 1)
                if new_instances_local != instances_local:
                    action = "scale_down"
                instances_local = new_instances_local
                low_streak_local = 0
                cooldown_timer_local = COOLDOWN_STEPS

        # oscillation detection
        oscillation_flag = 0
        if prev_action_local is not None:
            if (prev_action_local == "scale_up" and action == "scale_down") or \
               (prev_action_local == "scale_down" and action == "scale_up"):
                oscillation_flag = 1
                oscillation_count_local += 1

        prev_action_logged = prev_action_local if prev_action_local is not None else ""
        if action != "hold":
            prev_action_local = action

        # overload if either resource ratio > 1
        overload_flag = int((request_ratio > 1.0) or (network_ratio > 1.0))
        overload_count_local += overload_flag

        # predictor: if use_ols and coeffs provided, use them; otherwise fallback trend
        if use_ols and (coeffs is not None) and (np is not None):
            # Build feature vector: [1, avg_cpu_5(t), requests(t), net_in(t), net_out(t), instances(t)]
            try:
                a_cpu = float(merged.iloc[idx]["avg_cpu_5"])
                r = float(requests)
                nin = float(net_in)
                nout = float(net_out)
                inst = float(instances_local)
                x = np.array([1.0, a_cpu, r, nin, nout, inst])
                predicted_cpu = float(np.dot(coeffs, x))
                predicted_cpu = max(0.0, min(100.0, predicted_cpu))
            except Exception:
                # fallback to trend
                if idx >= 1:
                    prev_cpu = float(merged.iloc[idx-1]["avg_cpu_5"])
                    cur_cpu = float(merged.iloc[idx]["avg_cpu_5"])
                    trend = cur_cpu - prev_cpu
                    predicted_cpu = float(cur_cpu + trend * PRED_HORIZON)
                else:
                    predicted_cpu = float(merged.iloc[idx]["avg_cpu_5"])
                predicted_cpu = max(0.0, min(100.0, predicted_cpu))
        else:
            # trend fallback predictor
            if idx >= 1:
                prev_cpu = float(merged.iloc[idx-1]["avg_cpu_5"])
                cur_cpu = float(merged.iloc[idx]["avg_cpu_5"])
                trend = cur_cpu - prev_cpu
                predicted_cpu = float(cur_cpu + trend * PRED_HORIZON)
            else:
                predicted_cpu = float(merged.iloc[idx]["avg_cpu_5"])
            predicted_cpu = max(0.0, min(100.0, predicted_cpu))

        # if we can compute MAE in pass when using coeffs and future exists
        if use_ols and (np is not None) and (idx + PRED_HORIZON < n_rows) and (coeffs is not None):
            actual_future = float(merged.iloc[idx + PRED_HORIZON]["avg_cpu_5"])
            pred_mae_acc += abs(predicted_cpu - actual_future)
            pred_cnt += 1

        # costs & accumulators
        total_instance_minutes_local += instances_local * (SAMPLING_INTERVAL / 60.0)
        peak_queue_local = max(peak_queue_local, queue_length)
        latency_accum_local += latency_ms

        # collect row
        row_out = {
            "timestamp": timestamp.isoformat(),
            "requests": round(requests, 4),
            "instances": instances_local,
            "capacity": capacity,
            "effective_capacity": round(effective_capacity, 4),
            "request_ratio": round(request_ratio, 4),
            "net_throughput": round(net_throughput, 4),
            "network_capacity": round(network_capacity, 4),
            "network_ratio": round(network_ratio, 4),
            "cpu_req_target": round(cpu_req_target, 4),
            "cpu_net_target": round(cpu_net_target, 4),
            "cpu_target_combined": round(cpu_target_combined, 4),
            "cpu_smoothed": round(cpu_smoothed_local, 4),
            "predicted_cpu_5": round(predicted_cpu, 4),
            "latency_ms": round(latency_ms, 4),
            "queue_length": round(queue_length, 4),
            "high_streak": int(high_streak_local),
            "low_streak": int(low_streak_local),
            "cooldown_timer": int(cooldown_timer_local),
            "action": action,
            "prev_action": prev_action_logged,
            "oscillation_flag": int(oscillation_flag),
            "overload_flag": int(overload_flag)
        }

        rows_out.append(row_out)
        instances_series.append(instances_local)

        # advance
        cpu_prev_local = cpu_smoothed_local

    # return everything
    return rows_out, instances_series, pred_mae_acc, pred_cnt, {
        "oscillation_count": oscillation_count_local,
        "overload_count": overload_count_local,
        "total_instance_minutes": total_instance_minutes_local,
        "peak_queue": peak_queue_local,
        "latency_accum": latency_accum_local
    }

# ---------- PASS 1: run simulation without OLS (trend predictor) to get instances_series ----------
if VERBOSE:
    print("PASS 1 — running initial simulation (no OLS) to collect instances history...")
rows1, instances1, _, _, stats1 = simulate_pass(use_ols=False, coeffs=None, write_rows=False)

if VERBOSE:
    print(f"PASS 1 complete. Example head rows: {rows1[:2]}")

# ---------- Train OLS using PASS1 instances as the instances column ----------
use_ols = False
coeffs = None
pred_mae_accum = 0.0
pred_count = 0

# Build training arrays using merged and instances1
if np is not None and n_rows > PRED_HORIZON + 10:
    try:
        h = PRED_HORIZON
        rows_X = n_rows - h
        intercept = np.ones(rows_X)
        cpu_now = merged["avg_cpu_5"].values[:-h]
        req_now = merged["requests_per_min"].values[:-h]
        net_in = merged["avg_net_in_5"].values[:-h]
        net_out = merged["avg_net_out_5"].values[:-h]
        inst_now = np.array(instances1[:-h], dtype=float)

        X_full = np.column_stack([intercept, cpu_now, req_now, net_in, net_out, inst_now])
        y_full = merged["avg_cpu_5"].values[h:]
        coeffs, *_ = np.linalg.lstsq(X_full, y_full, rcond=None)
        use_ols = True
        if VERBOSE:
            print("PASS 1-based OLS trained. coeffs:", coeffs.tolist())
    except Exception as e:
        warnings.warn(f"OLS training on PASS1 data failed: {e}. Falling back to trend predictor.")
        use_ols = False
else:
    if VERBOSE:
        print("Not enough rows or numpy missing — skipping OLS training and using trend predictor.")

# ---------- PASS 2: run final simulation with trained OLS (if available) and write final CSV ----------
if VERBOSE:
    print("PASS 2 — running final simulation with trained predictor (if available) and writing CSV...")

# run final pass (use trained coeffs if use_ols True)
rows2, instances2, pred_mae_accum, pred_count, stats2 = simulate_pass(use_ols=use_ols, coeffs=coeffs)

# write final CSV (overwrite or append? we'll append as before but better to create fresh file)
# create/overwrite output CSV to ensure results come from PASS2 only
headers = [
    "timestamp", "requests", "instances", "capacity", "effective_capacity", "request_ratio",
    "net_throughput", "network_capacity", "network_ratio", "cpu_req_target", "cpu_net_target",
    "cpu_target_combined", "cpu_smoothed", "predicted_cpu_5", "latency_ms", "queue_length",
    "high_streak", "low_streak", "cooldown_timer", "action", "prev_action", "oscillation_flag",
    "overload_flag"
]

# Ensure directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for r in rows2:
        writer.writerow([
            r["timestamp"], r["requests"], r["instances"], r["capacity"], r["effective_capacity"],
            r["request_ratio"], r["net_throughput"], r["network_capacity"], r["network_ratio"],
            r["cpu_req_target"], r["cpu_net_target"], r["cpu_target_combined"], r["cpu_smoothed"],
            r["predicted_cpu_5"], r["latency_ms"], r["queue_length"], r["high_streak"],
            r["low_streak"], r["cooldown_timer"], r["action"], r["prev_action"],
            r["oscillation_flag"], r["overload_flag"]
        ])

# compute PASS2 summary
n_rows2 = len(rows2)
total_instance_minutes = stats2["total_instance_minutes"]
avg_instances = (total_instance_minutes / ((SAMPLING_INTERVAL / 60.0) * n_rows2)) if n_rows2 > 0 else 0.0
avg_latency = (stats2["latency_accum"] / n_rows2) if n_rows2 > 0 else 0.0
pred_mae = (pred_mae_accum / pred_count) if pred_count > 0 else None

summary = {
    "rows_simulated": n_rows2,
    "overload_events": int(stats2["overload_count"]) if "overload_count" in stats2 else None,
    "oscillation_count": int(stats2["oscillation_count"]) if "oscillation_count" in stats2 else None,
    "total_instance_minutes": round(total_instance_minutes, 3),
    "avg_instances": round(avg_instances, 3),
    "peak_queue_length": round(stats2["peak_queue"], 3),
    "avg_latency_ms": round(avg_latency, 3),
    "prediction": {
        "method": "ols" if use_ols else "trend_fallback",
        "horizon_steps": PRED_HORIZON,
        "mae": round(pred_mae, 3) if pred_mae is not None else None
    },
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

print("\n✅ Baseline simulation (two-pass) finished")
print("----------------------------------------")
print(f"Rows simulated (final):  {n_rows2}")
print(f"Total instance-minutes:  {summary['total_instance_minutes']}")
print(f"Average instances:       {summary['avg_instances']}")
print(f"Prediction method:       {summary['prediction']['method']}")
print(f"Prediction MAE:          {summary['prediction']['mae']}")
print(f"CSV output:              {OUTPUT_FILE}")
print(f"Summary JSON:            {METRICS_OUT}")

if VERBOSE:
    print("\nPASS1 stats (example):")
    print(stats1)
    print("\nPASS2 stats (example):")
    print(stats2)
    if use_ols:
        print("Trained coeffs (OLS):", coeffs.tolist())
