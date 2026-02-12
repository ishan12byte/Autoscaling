import boto3
import csv
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import os

from config.settings import INSTANCE_ID, REGION, METRICS_FILE

cw = boto3.client("cloudwatch", region_name=REGION)

end = datetime.now(timezone.utc)
start = end - timedelta(minutes=10)

def get_metric(name):
    return cw.get_metric_statistics(
        Namespace="AWS/EC2",
        MetricName=name,
        Dimensions=[{"Name": "InstanceId", "Value": INSTANCE_ID}],
        StartTime=start,
        EndTime=end,
        Period=60,
        Statistics=["Average"],
    )

cpu = get_metric("CPUUtilization")
net_in = get_metric("NetworkIn")
net_out = get_metric("NetworkOut")

data = defaultdict(lambda: {"cpu": 0, "net_in": 0, "net_out": 0})

for p in cpu["Datapoints"]:
    data[p["Timestamp"]]["cpu"] = p["Average"]

for p in net_in["Datapoints"]:
    data[p["Timestamp"]]["net_in"] = p["Average"]

for p in net_out["Datapoints"]:
    data[p["Timestamp"]]["net_out"] = p["Average"]

if not data:
    exit(0)

latest_ts = max(data.keys())
latest = data[latest_ts]

file_exists = os.path.exists(METRICS_FILE)

with open(METRICS_FILE, "a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow(["timestamp", "cpu", "net_in", "net_out"])

    writer.writerow([
        latest_ts.isoformat(),
        round(latest["cpu"], 2),
        round(latest["net_in"], 2),
        round(latest["net_out"], 2),
    ])



