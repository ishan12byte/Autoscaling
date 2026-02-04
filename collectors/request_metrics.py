import requests
import csv
import os
from datetime import datetime

from config.settings import METRICS_URL, REQUESTS_FILE

resp = requests.get(METRICS_URL).json()
current_total = resp["requests"]

prev_total = None

if os.path.exists(REQUESTS_FILE):
    with open(REQUESTS_FILE, "r") as f:
        lines = f.readlines()
        if len(lines) > 1:
            prev_total = int(lines[-1].split(",")[2])

requests_per_min = 0 if prev_total is None else max(0, current_total - prev_total)

file_exists = os.path.exists(REQUESTS_FILE)

with open(REQUESTS_FILE, "a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow(["timestamp", "requests_per_min", "requests_total"])

    writer.writerow([
        datetime.utcnow().isoformat(),
        requests_per_min,
        current_total,
    ])