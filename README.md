# Learning-Based Autoscaling: A Sequential Decision and Control Framework
A Control-System & Machine Learning Approach to Infrastructure Scaling

🚀 Overview

Modern infrastructure scaling is typically reactive, heuristic-driven, and brittle under uncertainty. This project reframes autoscaling as a sequential decision-making problem, combining:

✔ Control-system modelling
✔ Predictive machine learning
✔ Anomaly detection
✔ Reinforcement learning
✔ Safe, auditable execution logic

Instead of static threshold rules, the system treats scaling as a dynamic feedback process driven by workload, capacity, and system state.

🎯 Objective

Build a production-credible simulation environment that:

• Models realistic infrastructure dynamics
• Provides ML-ready state representations
• Enables RL policy learning
• Produces measurable evaluation metrics
• Demonstrates stability, cost-awareness, and safety

🧠 Core Concept

Infrastructure behaviour is simulated using:

Demand → Capacity → Utilization

Rather than naïve adjustments like:

cpu / instances

the simulator models:

✔ Effective capacity
✔ Inefficiency losses
✔ Nonlinear CPU saturation
✔ Work queue persistence
✔ Latency pressure

🏗️ System Architecture
Synthetic Workload (hey / metrics)
            ↓
Metrics Collection (CloudWatch + App)
            ↓
State Construction
            ↓
Baseline Controller / Simulator
            ↓
Evaluation Engine
            ↓
ML / RL Models
📦 Project Structure
project_root/
│
├── app/
│   └── app.py                  # CPU-bound workload service
│
├── collectors/
│   ├── ec2_metrics.py          # CloudWatch metrics collector
│   └── requests_metrics.py     # Workload / request collector
│
├── controllers/
│   ├── rule_based.py           # Static scaling policy
│   └── run_rule_based_sim.py   # Infrastructure dynamics simulator
│
├── evaluation/
│   └── eval_baseline.py        # Performance evaluator
│
├── config/
│   └── settings.py             # Central configuration
│
├── data/
│   ├── metrics.csv
│   ├── requests.csv
│   ├── state.csv
│   ├── baseline_simulation.csv
│   └── baseline_metrics.json
│
└── README.md
⚙️ Key Modelling Features
✅ Capacity Model
capacity = instances × workers_per_instance
effective_capacity = capacity × inefficiency

Models real-world inefficiencies:

✔ Scheduling overhead
✔ Context switching
✔ System losses

✅ CPU Response Curve

Nonlinear saturation:

cpu_target = 100 × (1 − exp(−k × load_ratio))

Why nonlinear?

✔ Smooth saturation
✔ No linear artefacts
✔ RL-friendly gradients

✅ CPU Inertia (Realistic Dynamics)
cpu_t = α × cpu_target + (1 − α) × cpu_previous

Prevents unrealistic jumps.

✅ Queue Persistence (Backpressure)

Optional modelling:

queue_t = max(0, queue_prev + requests − effective_capacity)

Simulates:

✔ Unserved demand
✔ Latency pressure
✔ Scaling urgency

✅ Latency Proxy
latency ∝ (1 + load_ratio²)

Models nonlinear degradation under load.

✅ Scaling Stability Mechanisms

✔ Hysteresis streaks
✔ Cooldown timers
✔ Instance guardrails

Prevents oscillations and flapping.

📊 Evaluation Metrics

The system evaluates scaling quality via:

🎯 Performance / SLA Metrics

• Overload events
• Overload percentage
• Longest overload streak

💰 Cost Proxy

• Average instances
• Peak instances
• Instance-minutes

🔄 Stability Metrics

• Oscillation events
• Oscillation rate

⏳ Pressure Signals

• Queue length behaviour
• Latency behaviour

🧪 Baseline Policy

Static rule controller:

IF cpu > 70 → scale_up
IF cpu < 30 → scale_down
ELSE → hold

Enhanced by:

✔ Sustained pressure detection
✔ Cooldown guardrails

Serves as the benchmark for ML / RL policies.

🧠 Machine Learning Extensions

This simulator provides a stable environment for:

✅ Predictive Models

• CPU forecasting
• Demand prediction
• Trend modelling

✅ Anomaly Detection

• Isolation Forest
• Statistical deviation models

✅ Reinforcement Learning

Infrastructure framed as an MDP:

State → Action → Transition → Reward

RL agents learn:

✔ Cost-performance tradeoffs
✔ Stability-aware scaling
✔ Proactive decisions

🛡️ Safety & Guardrails

Simulation enforces:

✔ Maximum replicas
✔ Minimum replicas
✔ Cooldown periods
✔ No destructive actions

Designed to mimic production-safe scaling behaviour.

🚀 Running the System
1️⃣ Start Workload Service
gunicorn app.app:app --bind 0.0.0.0:8000 --workers 1
2️⃣ Generate Synthetic Load

Using hey load generator.

3️⃣ Collect Metrics

Run collectors via cron or manual execution.

4️⃣ Run Baseline Simulator
python3 controllers/run_rule_based_sim.py
5️⃣ Evaluate Performance
python3 evaluation/eval_baseline.py
🎯 Demo Narrative

This project demonstrates:

✔ Infrastructure as a control system
✔ Scaling as sequential decision-making
✔ Limitations of static heuristics
✔ Benefits of predictive & RL policies
✔ Stability & safety awareness

✅ Success Criteria

Improved controllers should show:

✔ Fewer overload events
✔ Comparable or lower cost proxy
✔ Reduced oscillations
✔ Better queue / latency behaviour

🔮 Future Extensions

Planned enhancements:

• Multi-instance simulation
• Warmup delays
• Demand burst modelling
• Anomaly-aware controllers
• RL policy training
• Executor integration
• Visualization dashboard

📚 Key Ideas Explored

This project bridges:

✔ Cloud infrastructure behaviour
✔ Control theory
✔ Queueing dynamics
✔ Machine learning
✔ Reinforcement learning

🧑‍💻 Author

Ishan Gupta
Machine Learning / Systems Modelling
