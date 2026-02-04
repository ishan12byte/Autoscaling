import os

# ---------- AWS ----------
INSTANCE_ID = "i-04347365325c9ac7b"
REGION = "eu-north-1"

# ---------- URLs ----------
METRICS_URL = "http://127.0.0.1:8000/metrics"

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

METRICS_FILE = os.path.join(DATA_DIR, "metrics.csv")
REQUESTS_FILE = os.path.join(DATA_DIR, "requests.csv")
STATE_FILE = os.path.join(DATA_DIR, "state.csv")

os.makedirs(DATA_DIR, exist_ok=True)