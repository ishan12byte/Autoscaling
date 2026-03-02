#!/usr/bin/env python3
# evaluation/eval_baseline.py

import sys
import os
import json
import argparse
from statistics import mean

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings as cfg  # type: ignore

parser = argparse.ArgumentParser(description="Evaluate baseline simulation output (net+instances aware).")
parser.add_argument("--input-csv", type=str, default=os.path.join(getattr(cfg, "DATA_DIR", "data"), "baseline_simulation.csv"))
parser.add_argument("--metrics-json", type=str, default=os.path.join(getattr(cfg, "DATA_DIR", "data"), "baseline_metrics.json"))
parser.add_argument("--output-json", type=str, default=os.path.join(getattr(cfg, "DATA_DIR", "data"), "baseline_eval_summary.json"))
parser.add_argument("--pred-horizon", type=int, default=5)
args = parser.parse_args()

INPUT = args.input_csv
METRICS_JSON = args.metrics_json
OUT_JSON = args.output_json
PRED_HORIZON = int(args.pred_horizon)

if not os.path.exists(INPUT):
    raise FileNotFoundError(f"Baseline CSV not found: {INPUT}")

df = pd.read_csv(INPUT)

# required core columns
core_required = ["instances", "overload_flag", "oscillation_flag", "queue_length", "latency_ms", "cpu_smoothed"]
missing_core = [c for c in core_required if c not in df.columns]
if missing_core:
    raise KeyError(f"Evaluator expected core columns missing from {INPUT}: {missing_core}")

report = {}
report["rows"] = len(df)

# Overload
report["overload_events"] = int(df["overload_flag"].sum())
report["overload_pct"] = float(100.0 * df["overload_flag"].sum() / len(df))

# Longest overload streak
streak = 0
max_streak = 0
for flag in df["overload_flag"]:
    if int(flag) == 1:
        streak += 1
        if streak > max_streak:
            max_streak = streak
    else:
        streak = 0
report["longest_overload_streak"] = int(max_streak)

# Cost
report["avg_instances"] = float(df["instances"].mean())
report["peak_instances"] = int(df["instances"].max())
report["instance_minutes_proxy"] = float(df["instances"].sum())

# Stability
report["oscillation_events"] = int(df["oscillation_flag"].sum()) if "oscillation_flag" in df.columns else 0
report["oscillation_rate_pct"] = float(100.0 * report["oscillation_events"] / len(df)) if len(df) > 0 else 0.0

# Queue & latency
report["avg_queue_length"] = float(df["queue_length"].mean()) if "queue_length" in df.columns else 0.0
report["peak_queue_length"] = float(df["queue_length"].max()) if "queue_length" in df.columns else 0.0
report["avg_latency_ms"] = float(df["latency_ms"].mean()) if "latency_ms" in df.columns else 0.0
report["peak_latency_ms"] = float(df["latency_ms"].max()) if "latency_ms" in df.columns else 0.0

# Network metrics: throughput and ratio
if "net_throughput" in df.columns:
    report["avg_net_throughput"] = float(df["net_throughput"].mean())
    report["peak_net_throughput"] = float(df["net_throughput"].max())
else:
    report["avg_net_throughput"] = None
    report["peak_net_throughput"] = None

if "network_ratio" in df.columns or "network_ratio" in df.columns:
    # prefer explicit column name if present (older controllers used 'network_ratio' naming)
    if "network_ratio" in df.columns:
        report["avg_network_ratio"] = float(df["network_ratio"].mean())
        report["peak_network_ratio"] = float(df["network_ratio"].max())
    else:
        report["avg_network_ratio"] = None
        report["peak_network_ratio"] = None
else:
    # try to compute from avg_net_in_5 & avg_net_out_5
    if "avg_net_in_5" in df.columns and "avg_net_out_5" in df.columns:
        ratios = []
        for nin, nout in zip(df["avg_net_in_5"].fillna(0).values, df["avg_net_out_5"].fillna(0).values):
            ratios.append(nout / (nin + 1e-9))
        report["avg_network_ratio"] = float(mean(ratios)) if ratios else None
        report["peak_network_ratio"] = float(max(ratios)) if ratios else None
    else:
        report["avg_network_ratio"] = None
        report["peak_network_ratio"] = None

# CPU breakdown averages if available
if "cpu_req_target" in df.columns and "cpu_net_target" in df.columns:
    report["avg_cpu_req_target"] = float(df["cpu_req_target"].mean())
    report["avg_cpu_net_target"] = float(df["cpu_net_target"].mean())
    report["avg_cpu_combined_target"] = float(df["cpu_target_combined"].mean()) if "cpu_target_combined" in df.columns else None
else:
    report["avg_cpu_req_target"] = None
    report["avg_cpu_net_target"] = None
    report["avg_cpu_combined_target"] = None

# Predictor MAE: prefer metrics JSON if it contains mae, otherwise compute local shifted MAE
pred_mae = None
pred_method = None
if os.path.exists(METRICS_JSON):
    try:
        with open(METRICS_JSON, "r") as mf:
            m = json.load(mf)
        if "prediction" in m and isinstance(m["prediction"], dict) and "mae" in m["prediction"]:
            pred_mae = m["prediction"]["mae"]
            pred_method = m["prediction"].get("method", None)
            report["prediction_horizon_steps"] = m["prediction"].get("horizon_steps", None)
    except Exception:
        pred_mae = None

# compute local MAE if needed and possible
if pred_mae is None and "predicted_cpu_5" in df.columns and "cpu_smoothed" in df.columns:
    ph = report.get("prediction_horizon_steps", PRED_HORIZON) if 'PRED_HORIZON' in globals() else PRED_HORIZON
    errors = []
    n = len(df)
    for i in range(0, n):
        j = i + ph
        if j < n:
            try:
                pred = float(df.iloc[i]["predicted_cpu_5"])
                actual = float(df.iloc[j]["cpu_smoothed"])
                errors.append(abs(pred - actual))
            except Exception:
                pass
    if errors:
        pred_mae = float(sum(errors) / len(errors))
        pred_method = "local_shift_mae"
        report["prediction_horizon_steps"] = ph

report["prediction_mae"] = pred_mae
report["prediction_method"] = pred_method

# Save JSON summary
with open(OUT_JSON, "w") as f:
    json.dump(report, f, indent=2)

# Print summary
print("\n✅ Baseline Evaluation Summary")
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
if report["avg_net_throughput"] is not None:
    print(f"Avg net throughput: {report['avg_net_throughput']:.2f}, Peak: {report['peak_net_throughput']:.2f}")
if report["avg_network_ratio"] is not None:
    print(f"Avg network_ratio: {report['avg_network_ratio']:.3f}, Peak network_ratio: {report['peak_network_ratio']:.3f}")
if report["avg_cpu_req_target"] is not None:
    print(f"Avg cpu_req_target: {report['avg_cpu_req_target']:.2f}, Avg cpu_net_target: {report['avg_cpu_net_target']:.2f}")
if report["prediction_mae"] is not None:
    print(f"Prediction MAE: {report['prediction_mae']} (method: {report['prediction_method']})")
print(f"Saved JSON summary: {OUT_JSON}")
