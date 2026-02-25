import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import panda as pd
import csv
from datetime import datetime, timezone

from controller.rule_based import decide
from config.settings import STATE_FILE, DATA_DIR

DECISIONS_FILE = os.path.join(DATA_DIR, "decisions_rule.csv")
PERF_FILE = os.path.join(DATA_DIR, "baseline_performance.csv")

MAX_INSTANCES = 2
MIN_INSTANCES = 1

df = pd.read_csv(STATE_FILE)

instances = 1
prev_action = None
oscillation_count = 0

decisions_exists = os.path.exists(DECISIONS_FILE)
perf_exists = os.path.exists(PERF_FILE)

with open(DECISIONS_FILE, "a", newline="") as dfile, \
     open(PERF_FILE, "a", newline="") as pfile:

    d_writer = csv.writer(dfile)
    p_writer = csv.writer(pfile)

    # ✅ Headers
    if not decisions_exists:
        d_writer.writerow([
            "timestamp",
            "action",
            "reason",
            "avg_cpu_5",
            "instances"
        ])

    if not perf_exists:
        p_writer.writerow([
            "timestamp",
            "avg_cpu_5",
            "instances",
            "overload_flag"
        ])

    for _, row in df.iterrows():

        cpu = float(row["avg_cpu_5"])

        action, reason = decide(row)

        # -------------------------
        # Scaling logic
        # -------------------------
        if action == "scale_up":
            instances = min(MAX_INSTANCES, instances + 1)

        elif action == "scale_down":
            instances = max(MIN_INSTANCES, instances - 1)

        # -------------------------
        # Overload metric (CRITICAL)
        # -------------------------
        overload_flag = int(cpu > 90)

        # -------------------------
        # Instability / Oscillation
        # -------------------------
        if prev_action:

            if (
                prev_action == "scale_up" and action == "scale_down"
            ) or (
                prev_action == "scale_down" and action == "scale_up"
            ):
                oscillation_count += 1

        prev_action = action

        timestamp = datetime.now(timezone.utc).isoformat()

        # -------------------------
        # Decision log
        # -------------------------
        d_writer.writerow([
            timestamp,
            action,
            reason,
            cpu,
            instances
        ])

        # -------------------------
        # Performance log
        # -------------------------
        p_writer.writerow([
            timestamp,
            cpu,
            instances,
            overload_flag
        ])

print("Baseline run complete.")
print("Oscillation events detected:", oscillation_count)  