# Learning-Based Autoscaling: A Sequential Decision and Control Framework
A Control-System & Machine Learning Approach to Infrastructure Scaling

🚀 Overview

Modern infrastructure scaling is typically reactive, heuristic-driven, and brittle under uncertainty.
This project reframes autoscaling as a sequential decision-making problem, combining:

✔ Control-system modelling
✔ Predictive machine learning
✔ Anomaly detection
✔ Reinforcement learning
✔ Safe, auditable execution logic

Instead of static threshold rules, the system treats scaling as a dynamic feedback process driven by workload, capacity, and system state.

📌 Project Status

This project is an active research and engineering effort.

Current progress

✔ Infrastructure simulator implemented
✔ Realistic scaling dynamics modelled
✔ Baseline rule-based controller implemented
✔ Evaluation framework completed
✔ Dataset generation pipeline created

In progress

• Machine learning prediction layer
• Reinforcement learning policy training
• Learning-based autoscaling policies

Planned

• RL policy evaluation vs baseline
• Real-time policy execution layer
• Visualization and experiment dashboard

🎯 Objective

Build a production-credible simulation environment that:

• Models realistic infrastructure dynamics
• Provides ML-ready state representations
• Enables RL policy learning
• Produces measurable evaluation metrics
• Demonstrates stability, cost-awareness, and safety

🧠 Core Concept

Infrastructure behaviour is simulated using the relationship:

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
🛠️ Tech Stack

Languages & Libraries

• Python
• NumPy / Pandas
• scikit-learn (planned)
• Reinforcement learning libraries (planned)

Infrastructure

• AWS EC2
• AWS CloudWatch
• Gunicorn

Tools

• hey load generator
• cron for metric collection

📦 Project Structure
project_root/

├── app/
│   └── app.py                     # CPU-bound workload service

├── collectors/
│   ├── ec2_metrics.py             # CloudWatch metrics collector
│   └── requests_metrics.py        # Workload / request collector

├── controllers/
│   ├── rule_based.py              # Static scaling policy
│   └── run_rule_based_sim.py      # Infrastructure dynamics simulator

├── evaluation/
│   └── eval_baseline.py           # Performance evaluator

├── config/
│   └── settings.py                # Central configuration

├── data/
│   ├── metrics.csv
│   ├── requests.csv
│   ├── state.csv
│   ├── baseline_simulation.csv
│   └── baseline_metrics.json

└── README.md
📊 Generated Dataset

The simulator produces structured datasets suitable for machine learning and reinforcement learning.

Key dataset files:

• metrics.csv — infrastructure metrics collected from CloudWatch
• requests.csv — workload request statistics
• state.csv — constructed state representation used by controllers
• baseline_simulation.csv — simulator output during baseline execution

Typical state features include:

• CPU utilization
• Request rate
• Instance count
• Effective capacity
• Queue length
• Latency proxy
• Load ratio

These datasets form the basis for training predictive models and reinforcement learning policies.

⚙️ Key Modelling Features
✅ Capacity Model
capacity = instances × workers_per_instance
effective_capacity = capacity × inefficiency

Models real-world inefficiencies:

✔ Scheduling overhead
✔ Context switching
✔ System losses

✅ CPU Response Curve

Nonlinear saturation model:

cpu_target = 100 × (1 − exp(−k × load_ratio))

Why nonlinear?

✔ Smooth saturation
✔ No linear artefacts
✔ RL-friendly gradients

✅ CPU Inertia (Realistic Dynamics)
cpu_t = α × cpu_target + (1 − α) × cpu_previous

Prevents unrealistic jumps in utilization.

✅ Queue Persistence (Backpressure)

Optional modelling:

queue_t = max(0, queue_prev + requests − effective_capacity)

Simulates:

✔ Unserved demand
✔ Latency pressure
✔ Scaling urgency

✅ Latency Proxy
latency ∝ (1 + load_ratio²)

Models nonlinear degradation under heavy load.

✅ Scaling Stability Mechanisms

✔ Hysteresis streaks
✔ Cooldown timers
✔ Instance guardrails

Prevents oscillations and scaling flapping.

📊 Evaluation Metrics

The system evaluates scaling quality through the following metrics.

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

Static rule-based controller:

IF cpu > 70 → scale_up
IF cpu < 30 → scale_down
ELSE → hold

Enhanced with:

✔ Sustained pressure detection
✔ Cooldown guardrails

This baseline serves as the benchmark for ML / RL policies.

🧪 Experiment Pipeline

The project follows a structured experimentation workflow for evaluating autoscaling strategies.

Infrastructure Metrics
        ↓
Snapshot Collection
        ↓
State Construction
        ↓
Dataset Split (Training / Testing)
        ↓
Baseline Controller Simulation
        ↓
Evaluation Metrics
        ↓
Machine Learning Models
        ↓
Reinforcement Learning Policies
Step 1 — Snapshot Collection

Infrastructure metrics and workload statistics are collected and stored as datasets.

Generated files:

metrics.csv
requests.csv

These snapshots represent realistic workload conditions over time.

Step 2 — State Construction

Collected metrics are transformed into ML-ready state representations.

The constructed state includes signals such as:

• CPU utilization
• Request rate
• Instance count
• Effective capacity
• Queue length
• Latency proxy
• Load ratio

The resulting dataset:

data/state.csv
Step 3 — Dataset Split

The state dataset is divided into:

Training set
Testing set

Typical split:

70% training
30% testing

This enables fair evaluation of learned policies.

Step 4 — Baseline Simulation

Run the rule-based simulator:

python3 controllers/run_rule_based_sim.py

The simulator models:

• CPU dynamics
• queue persistence
• latency pressure
• scaling guardrails

Output:

baseline_simulation.csv
Step 5 — Evaluation

Evaluate the controller:

python3 evaluation/eval_baseline.py

Metrics generated include:

• overload frequency
• scaling stability
• cost proxy
• latency pressure

Results are stored in:

baseline_metrics.json
🧠 Machine Learning Extensions

This simulator provides a stable environment for experimenting with:

✅ Predictive Models

• CPU forecasting
• Demand prediction
• Trend modelling

✅ Anomaly Detection

• Isolation Forest
• Statistical deviation models

🎮 Reinforcement Learning Formulation

Autoscaling is framed as a Markov Decision Process (MDP).

State

• CPU utilization
• Request rate
• Instance count
• Queue length
• Latency proxy
• Load ratio

Actions

scale_down
hold
scale_up

Transitions

The simulator models infrastructure responses including:

• CPU inertia
• queue dynamics
• nonlinear capacity response

Reward

Rewards balance performance and cost:

• penalize overload and latency pressure
• penalize unnecessary scaling
• encourage stable instance usage

The goal is to learn a stable, cost-efficient autoscaling policy.

📈 Example Baseline Results

Example metrics from the baseline controller:

• Total simulated rows: ~7800
• Sustained overload events: 10
• Maximum overload streak: 18
• Oscillation events: 2
• Queue growth events: 0

These results establish a reference point for evaluating future ML and RL controllers.

🛡️ Safety & Guardrails

The simulator enforces:

✔ Maximum replicas
✔ Minimum replicas
✔ Cooldown periods
✔ No destructive actions

Designed to mimic production-safe scaling behaviour.

🚀 Running the System
1️⃣ Start Workload Service
gunicorn app.app:app --bind 0.0.0.0:8000 --workers 1
2️⃣ Generate Synthetic Load

Use the hey load generator.

3️⃣ Collect Metrics

Run collectors via cron jobs or manual execution.

4️⃣ Run Baseline Simulator
python3 controllers/run_rule_based_sim.py
5️⃣ Evaluate Performance
python3 evaluation/eval_baseline.py
💡 Motivation

Traditional autoscaling systems rely on static threshold rules such as:

scale when CPU > 70%

These heuristics often fail under:

• bursty workloads
• delayed scaling responses
• nonlinear infrastructure behavior

This project explores whether learning-based policies can produce:

• more stable scaling behaviour
• lower infrastructure cost
• improved performance under dynamic workloads

🔮 Future Extensions

Planned enhancements include:

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
Machine Learning & Systems Modeling
