CPU_UP = 70
CPU_DOWN = 30

def decide(row):
    cpu = float(row["avg_cpu_5"])

    if cpu > CPU_UP:
        return "scale_up", "cpu_above_threshold"
    
    if cpu < CPU_DOWN:
        return "scale_down", "cpu_below_threshold"
    
    return "do_nothing", "within_band"