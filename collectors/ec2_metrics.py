import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
import csv
from datetime import datetime, timedelta, timezone

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


def latest_datapoint(metric):
    if not metric["Datapoints"]:
        return None
    return max(metric["Datapoints"], key=lambda x: x["Timestamp"])


cpu = latest_datapoint(get_metric("CPUUtilization"))
net_in = latest_datapoint(get_metric("NetworkIn"))
net_out = latest_datapoint(get_metric("NetworkOut"))

# If CPU missing → skip write (core metric)
if cpu is None:
    exit(0)

latest_ts = cpu["Timestamp"]

cpu_val = cpu["Average"]
net_in_val = net_in["Average"] if net_in else 0
net_out_val = net_out["Average"] if net_out else 0

file_exists = os.path.exists(METRICS_FILE)

with open(METRICS_FILE, "a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow(["timestamp", "cpu", "net_in", "net_out"])

    writer.writerow([
        latest_ts.isoformat(),
        round(cpu_val, 2),
        round(net_in_val, 2),
        round(net_out_val, 2),
    ])
