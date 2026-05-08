# controllers/decision_engine.py

def decision_engine(state, baseline_action, cpu_pred):
    """
    state: dict containing current system metrics
    baseline_action: action from existing rule-based system
    cpu_pred: predicted next CPU from ML model

    returns: final action (scale_up / scale_down / hold)
    """

    cpu = state["cpu"]
    queue = state["queue"]
    latency = state["latency"]
    instances = state["instances"]

    # -------------------------
    #  1. SAFETY FIRST (never break system)
    # -------------------------
    if latency > 800 or queue > 10:
        return "scale_up"

    # -------------------------
    #  2. EARLY SCALE UP (your biggest advantage)
    # -------------------------
    if cpu_pred > 75 and cpu < 65:
        return "scale_up"

    # -------------------------
    #  3. PREVENT UNNECESSARY SCALE UP
    # -------------------------
    if cpu > 75 and cpu_pred < 65:
        return "hold"

    # -------------------------
    #  4. SMART SCALE DOWN
    # -------------------------
    if cpu_pred < 40 and queue < 5 and latency < 300 and instances > 1:
        return "scale_down"

    # -------------------------
    #  5. FALLBACK TO BASELINE
    # -------------------------
    return baseline_action
