from flask import Flask, jsonify
import time
import random
import threading

app = Flask(__name__)

# -------------------------
# In-memory counters
# -------------------------
REQUEST_COUNT = 0
ERROR_COUNT = 0

lock = threading.Lock()

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return jsonify({
        "service": "compute-service",
        "endpoints": ["/compute", "/metrics"]
    })


@app.route("/compute")
def compute():
    global REQUEST_COUNT, ERROR_COUNT

    with lock:
        REQUEST_COUNT += 1

    try:
        x = 0
        for i in range(5_000_000):
            x += i * random.random()

        time.sleep(random.random())

        return jsonify({
            "status": "ok",
            "result": x
        })

    except Exception:
        with lock:
            ERROR_COUNT += 1

        return jsonify({
            "status": "error"
        }), 500


@app.route("/metrics")
def metrics():
    return jsonify({
        "requests": REQUEST_COUNT,
        "errors": ERROR_COUNT
    })


# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
