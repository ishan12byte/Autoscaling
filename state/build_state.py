import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from datetime import datetime, timezone
import statistics

from config.settings import METRICS_FILE, REQUESTS_FILE, STATE_FILE

# Exit if metrics file missing
if not os.path.exists(METRICS_FILE):
    sys.exit(0)

# Read metrics
rows = []
with open(METRICS_FILE, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Need minimum history
if len(rows) < 5:
    sys.exit(0)

# ✅ CRITICAL FIX — Ensure proper ordering
rows = sorted(rows, key=lambda r: r["timestamp"])

last_5 = rows[-5:]

# CPU features
avg_cpu_5 = sum(float(r["cpu"]) for r in last_5) / 5
cpu_trend = float(last_5[-1]["cpu"]) - float(last_5[0]["cpu"])

cpu_values = [float(r["cpu"]) for r in last_5]
volatility = statistics.pstdev(cpu_values)

# Network features
avg_net_in_5 = sum(float(r["net_in"]) for r in last_5) / 5
avg_net_out_5 = sum(float(r["net_out"]) for r in last_5) / 5

net_ratio = (
    avg_net_out_5 / avg_net_in_5
    if avg_net_in_5 > 0 else 0
)

# Requests signal
requests_per_min = 0

if os.path.exists(REQUESTS_FILE):
    with open(REQUESTS_FILE, "r") as f:
        lines = f.readlines()
        if len(lines) > 1:
            requests_per_min = int(lines[-1].split(",")[1])

# Write state
state_exists = os.path.exists(STATE_FILE)

with open(STATE_FILE, "a", newline="") as f:
    writer = csv.writer(f)

    if not state_exists:
        writer.writerow([
            "timestamp",
            "avg_cpu_5",
            "cpu_trend",
            "volatility",
            "avg_net_in_5",
            "avg_net_out_5",
            "net_ratio",
            "requests_per_min",
        ])

    writer.writerow([
        datetime.now(timezone.utc).isoformat(),
        round(avg_cpu_5, 2),
        round(cpu_trend, 2),
        round(volatility, 2),
        round(avg_net_in_5, 2),
        round(avg_net_out_5, 2),
        round(net_ratio, 2),
        requests_per_min,
    ])
