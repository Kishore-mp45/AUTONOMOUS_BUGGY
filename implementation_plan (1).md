# 🤖 Implementation Plan: Local Planner Controller Comparison
### Autonomous Campus Buggy — Kishore | 3rd Year Project

---

## 📋 Project Summary

Build a **pure Python 2D simulation** (Windows-compatible) that compares three local motion planning controllers:
- **DWA** — Dynamic Window Approach
- **RPP** — Regulated Pure Pursuit
- **MPPI** — Model Predictive Path Integral

Across **5 scripted obstacle scenarios**, each run **10 times per controller**, recording:
- Minimum distance to obstacle
- Completion time
- Peak acceleration & jerk
- Failures (collision / timeout / stuck)

**Output:** Scenario files, results CSV table, path plots, acceleration/jerk plots, and a final recommendation.

---

## 🛠️ Tech Stack

### Core Python Libraries (Must Install)
```
pip install numpy matplotlib scipy pandas pygame
```

| Library | Role |
|---|---|
| `numpy` | Robot kinematics, math, vector ops |
| `matplotlib` | Path plots, metric graphs, results visualization |
| `scipy` | Signal processing for jerk/acceleration |
| `pandas` | Results table, CSV logging |
| `pygame` | Live 2D visualization during simulation |

### Optional / Exploration Libraries (Evaluate for 3D or Agent-Based Visualization)

| Library | Role | Notes |
|---|---|---|
| `vpython` | 3D real-time visualization in browser | `pip install vpython` — good for 3D buggy view |
| `panda3d` | Full 3D game-engine level simulation | Heavy but powerful for realistic rendering |
| `PyOpenGL` | Low-level OpenGL 3D graphics | Needs GLUT; more complex setup |
| `simulate` (HuggingFace) | Intelligent agent environments | Experimental; try if time permits |

> **Decision rule for agent:** Start with `pygame` + `matplotlib` for 2D. Add `vpython` only if 3D visualization is needed for the write-up.

---

## 📁 Project Folder Structure

```
campus_buggy_sim/
│
├── main.py                  # Entry point — runs all controllers × all scenarios
├── config.py                # All tunable parameters in one place
│
├── robot/
│   ├── __init__.py
│   └── bicycle_model.py     # Car-like robot kinematic model
│
├── controllers/
│   ├── __init__.py
│   ├── dwa.py               # Dynamic Window Approach
│   ├── rpp.py               # Regulated Pure Pursuit
│   └── mppi.py              # Model Predictive Path Integral
│
├── scenarios/
│   ├── __init__.py
│   ├── scenario_base.py     # Base scenario class
│   ├── s1_static_box.py     # Scenario 1: Static obstacle on path
│   ├── s2_person_crossing.py # Scenario 2: Moving person crosses path
│   ├── s3_narrow_gap.py     # Scenario 3: Narrow corridor
│   ├── s4_two_obstacles.py  # Scenario 4: Two obstacles side by side
│   └── s5_sudden_obstacle.py # Scenario 5: Obstacle spawns mid-run
│
├── metrics/
│   ├── __init__.py
│   └── recorder.py          # Records all metrics per run
│
├── visualization/
│   ├── __init__.py
│   ├── pygame_viz.py        # Live 2D simulation view
│   └── plot_results.py      # Post-run matplotlib plots
│
├── results/
│   ├── raw/                 # Per-run CSVs (auto-generated)
│   └── plots/               # Saved PNG plots (auto-generated)
│
├── requirements.txt
└── README.md
```

---

## 🧱 Phase 1: Robot Model (`robot/bicycle_model.py`)

### Task
Implement a **kinematic bicycle model** — the standard car-like robot model.

### What to implement

```python
class BicycleModel:
    """
    State: [x, y, theta, v]
    Control input: [steering_angle, acceleration]
    """
    def __init__(self, wheelbase=0.5, max_speed=1.5, max_steer=0.6):
        # wheelbase: distance between front and rear axle (metres)
        # max_speed: m/s
        # max_steer: radians

    def update(self, state, control, dt=0.05):
        # Apply kinematic equations:
        # x_dot = v * cos(theta)
        # y_dot = v * sin(theta)
        # theta_dot = v * tan(delta) / L
        # v_dot = acceleration
        # Return new state

    def get_footprint(self, state):
        # Return bounding box corners for collision checking
```

### Parameters to expose in `config.py`
```python
WHEELBASE = 0.5          # metres
MAX_SPEED = 1.5          # m/s
MAX_ACCEL = 1.0          # m/s²
MAX_STEER = 0.6          # radians (~34 degrees)
DT = 0.05                # simulation timestep (seconds)
ROBOT_RADIUS = 0.3       # metres (for collision detection)
```

---

## 🧱 Phase 2: Controllers

### 2A — Regulated Pure Pursuit (`controllers/rpp.py`)

**Concept:** Look ahead a fixed distance on the path, steer toward that point. Speed is regulated (reduced) near obstacles.

```python
class RegulatedPurePursuit:
    def __init__(self, lookahead_dist=1.0, max_speed=1.5, min_speed=0.3):
        ...

    def compute_control(self, state, path, costmap):
        # 1. Find the lookahead point on path
        # 2. Compute curvature to that point: kappa = 2*sin(alpha) / L_d
        # 3. Regulate speed based on curvature and proximity to obstacles
        # 4. Return (steering_angle, velocity)
```

**Key tunable params:**
```python
RPP_LOOKAHEAD = 1.0        # metres
RPP_MAX_SPEED = 1.5        # m/s
RPP_MIN_SPEED = 0.3        # m/s (near obstacles)
RPP_OBSTACLE_SLOWDOWN = True
RPP_CURVATURE_SLOWDOWN = True
```

---

### 2B — Dynamic Window Approach (`controllers/dwa.py`)

**Concept:** Sample velocity commands within a "dynamic window" (reachable in next timestep), score each by goal progress + obstacle clearance + speed, pick best.

```python
class DWA:
    def __init__(self, max_speed, max_accel, max_steer, dt):
        ...

    def compute_control(self, state, goal, obstacles):
        # 1. Build dynamic window: [v_min, v_max] x [w_min, w_max]
        # 2. Sample N velocity pairs (v, omega) within window
        # 3. For each sample:
        #    - Simulate trajectory forward T seconds
        #    - Score: heading_score + clearance_score + velocity_score
        #    - Discard if collision detected
        # 4. Return best (v, steering) pair
```

**Key tunable params:**
```python
DWA_V_SAMPLES = 20         # velocity samples
DWA_W_SAMPLES = 30         # angular velocity samples
DWA_SIM_TIME = 2.0         # seconds to simulate each trajectory
DWA_HEADING_WEIGHT = 0.8
DWA_CLEARANCE_WEIGHT = 0.2
DWA_SPEED_WEIGHT = 0.4
```

---

### 2C — MPPI (`controllers/mppi.py`)

**Concept:** Sample K random trajectory perturbations, evaluate their cost (obstacle hit + goal distance + smoothness), compute weighted average of best trajectories.

```python
class MPPI:
    def __init__(self, K=500, T=20, sigma=0.5, lam=1.0):
        # K: number of trajectory samples
        # T: horizon steps
        # sigma: noise std dev
        # lam: temperature (controls how sharply best is selected)
        ...

    def compute_control(self, state, path, obstacles, costmap):
        # 1. Sample K noise sequences: epsilon ~ N(0, sigma²)
        # 2. For each sample k:
        #    - Simulate trajectory using U + epsilon_k
        #    - Compute cost: obstacle_cost + path_cost + control_cost
        # 3. Compute weights: w_k = exp(-cost_k / lam)
        # 4. Update control: U = sum(w_k * epsilon_k) / sum(w_k)
        # 5. Return first control action
```

**Key tunable params:**
```python
MPPI_K = 500               # trajectory samples
MPPI_T = 20                # horizon steps
MPPI_SIGMA = 0.5           # noise standard deviation
MPPI_LAMBDA = 1.0          # temperature
MPPI_OBSTACLE_COST = 1e6   # high cost for hitting obstacle
MPPI_PATH_COST = 1.0
MPPI_CONTROL_COST = 0.1
```

---

## 🧱 Phase 3: Scenarios (`scenarios/`)

### Base Scenario Class (`scenario_base.py`)

```python
class ScenarioBase:
    def __init__(self):
        self.start = None          # (x, y, theta)
        self.goal = None           # (x, y)
        self.global_path = []      # list of (x,y) waypoints
        self.obstacles = []        # list of obstacle objects
        self.map_bounds = (0, 0, 20, 20)  # xmin, ymin, xmax, ymax
        self.time_limit = 60.0     # seconds

    def reset(self):
        # Reset dynamic obstacles to start position

    def update(self, t):
        # Update moving obstacles at time t

    def get_obstacles(self):
        # Return current obstacle list
```

---

### Scenario 1 — Static Box (`s1_static_box.py`)

```
Map layout (20x20 grid):
Start: (1, 10)  →  Goal: (19, 10)
Path: straight line, y=10

Obstacle: Box at (10, 10), size 1.5x1.5m

Robot must detect and go around or stop.
```

```python
OBSTACLES = [
    {"type": "static", "x": 10, "y": 10, "w": 1.5, "h": 1.5}
]
```

---

### Scenario 2 — Person Crossing (`s2_person_crossing.py`)

```
Map layout:
Start: (1, 10)  →  Goal: (19, 10)
Path: straight line, y=10

Person (0.4m radius circle) starts at (10, 5),
moves upward: y increases at 0.5 m/s
Crosses path at t ≈ 10s

Robot must: slow down or stop, not collide.
```

```python
PERSON = {
    "start": (10, 5), "velocity": (0, 0.5),
    "radius": 0.4, "type": "dynamic"
}
```

---

### Scenario 3 — Narrow Gap (`s3_narrow_gap.py`)

```
Map layout:
Start: (1, 10)  →  Goal: (19, 10)

Wall 1: x=10, y=0  to  y=8.5   (blocks left)
Wall 2: x=10, y=11.5  to  y=20 (blocks right)
Gap width: 3.0m (robot width ~0.6m, so feasible but tight)
```

```python
WALLS = [
    {"x": 10, "y_start": 0, "y_end": 8.5, "thickness": 0.3},
    {"x": 10, "y_start": 11.5, "y_end": 20, "thickness": 0.3}
]
```

---

### Scenario 4 — Two Obstacles (`s4_two_obstacles.py`)

```
Map layout:
Start: (1, 10)  →  Goal: (19, 10)

Box 1: (9, 10.5),  size 1.2x1.2m
Box 2: (11, 9.2),  size 1.2x1.2m

Staggered placement — robot must weave through.
```

```python
OBSTACLES = [
    {"type": "static", "x": 9,  "y": 10.5, "w": 1.2, "h": 1.2},
    {"type": "static", "x": 11, "y": 9.2,  "w": 1.2, "h": 1.2}
]
```

---

### Scenario 5 — Sudden Obstacle (`s5_sudden_obstacle.py`)

```
Map layout:
Start: (1, 10)  →  Goal: (19, 10)
Path: clear initially.

At t = 8.0 seconds:
  A box (1.0x1.0m) spawns at (12, 10) — directly on path.

Robot must react in real-time. Tests reaction speed of each controller.
```

```python
SPAWN_TIME = 8.0
SPAWN_OBSTACLE = {"type": "static", "x": 12, "y": 10, "w": 1.0, "h": 1.0}
```

---

## 🧱 Phase 4: Metrics Recorder (`metrics/recorder.py`)

```python
class MetricsRecorder:
    def __init__(self, controller_name, scenario_name, run_id):
        self.positions = []        # list of (x, y, t)
        self.velocities = []       # list of (v, t)
        self.min_obstacle_dist = float('inf')
        self.start_time = None
        self.end_time = None
        self.failed = False
        self.failure_reason = ""   # "collision" | "timeout" | "stuck"

    def step(self, state, obstacles, t):
        # Record position, velocity
        # Compute distance to nearest obstacle → update min
        # Detect collision (dist < robot_radius)

    def finish(self, success):
        # Record end time, compute completion time
        # Compute peak acceleration: diff(velocity) / dt
        # Compute peak jerk: diff(acceleration) / dt

    def to_dict(self):
        # Return all metrics as a flat dict for pandas

    def save_csv(self, path):
        # Save this run's time-series data to CSV
```

### Output CSV columns (one row per run):
```
controller, scenario, run_id, completed, failure_reason,
completion_time, min_obstacle_dist,
peak_acceleration, peak_jerk, path_length
```

---

## 🧱 Phase 5: Main Runner (`main.py`)

```python
CONTROLLERS = {
    "RPP":  RegulatedPurePursuit(...),
    "DWA":  DWA(...),
    "MPPI": MPPI(...)
}

SCENARIOS = {
    "s1_static_box":      S1StaticBox(),
    "s2_person_crossing": S2PersonCrossing(),
    "s3_narrow_gap":      S3NarrowGap(),
    "s4_two_obstacles":   S4TwoObstacles(),
    "s5_sudden_obstacle": S5SuddenObstacle()
}

RUNS_PER_COMBO = 10

# Loop structure:
for scenario_name, scenario in SCENARIOS.items():
    for ctrl_name, controller in CONTROLLERS.items():
        for run_id in range(RUNS_PER_COMBO):
            scenario.reset()
            recorder = MetricsRecorder(ctrl_name, scenario_name, run_id)
            run_simulation(scenario, controller, recorder)
            all_results.append(recorder.to_dict())

df = pd.DataFrame(all_results)
df.to_csv("results/all_results.csv", index=False)
```

---

## 🧱 Phase 6: Visualization

### 6A — Live Pygame Visualization (`visualization/pygame_viz.py`)

```python
class PygameViz:
    # Window: 800x800 pixels, maps to 20x20m world
    # Draw: robot (blue rectangle), path (grey), obstacles (red),
    #       goal (green circle), trajectory trail (cyan dots)
    # HUD: current controller name, scenario, run #, speed, time
    # Optional: toggle with --viz flag in main.py
```

### 6B — Post-run Matplotlib Plots (`visualization/plot_results.py`)

Generate the following plots and save to `results/plots/`:

#### Plot 1: Path Comparison (per scenario)
```
One subplot per scenario (5 total).
Overlay paths of RPP (blue), DWA (green), MPPI (orange).
Show obstacles in red. Mark start (★) and goal (◎).
```

#### Plot 2: Min Obstacle Distance (box plot)
```
X-axis: Scenarios (s1 to s5)
Y-axis: Minimum distance to obstacle (metres)
Grouped bars: RPP | DWA | MPPI per scenario
Danger zone line at y=0.3m (robot radius)
```

#### Plot 3: Completion Time (bar chart)
```
Mean completion time ± std deviation
Grouped by controller, one color per scenario
```

#### Plot 4: Peak Acceleration & Jerk (line plot)
```
Two subplots: acceleration | jerk
X-axis: scenarios, Y-axis: peak values
One line per controller
Comfort threshold lines (e.g. accel < 1.5 m/s², jerk < 3.0 m/s³)
```

#### Plot 5: Failure Heatmap
```
Rows: controllers (RPP, DWA, MPPI)
Cols: scenarios (s1–s5)
Cell value: number of failures out of 10 runs
Color: green=0, yellow=1-3, red=4+
```

---

## 🧱 Phase 7: Results Table

After all runs, generate a **summary table** printed to console and saved as `results/summary_table.csv`:

| Controller | Scenario | Completion Rate | Avg Time (s) | Min Dist (m) | Peak Accel | Peak Jerk | Failures |
|---|---|---|---|---|---|---|---|
| RPP | s1_static_box | 10/10 | 18.3 | 0.52 | 0.8 | 1.2 | 0 |
| DWA | s1_static_box | 10/10 | 17.1 | 0.41 | 1.1 | 2.1 | 0 |
| MPPI | s1_static_box | 9/10 | 16.8 | 0.38 | 0.9 | 1.5 | 1 |
| ... | ... | ... | ... | ... | ... | ... | ... |

---

## 🧱 Phase 8: Config File (`config.py`)

All parameters in one place — agent should use this as the single source of truth:

```python
# ── Robot ──────────────────────────────────────────────
WHEELBASE    = 0.5    # m
ROBOT_RADIUS = 0.3    # m
MAX_SPEED    = 1.5    # m/s
MAX_ACCEL    = 1.0    # m/s²
MAX_STEER    = 0.6    # rad
DT           = 0.05   # s (simulation timestep)

# ── Map ────────────────────────────────────────────────
MAP_W = 20.0          # metres
MAP_H = 20.0          # metres

# ── Experiment ─────────────────────────────────────────
RUNS_PER_COMBO  = 10
TIME_LIMIT      = 60.0  # seconds per run
HEADLESS        = False # True = no pygame window (faster)

# ── RPP ────────────────────────────────────────────────
RPP_LOOKAHEAD   = 1.0
RPP_MAX_SPEED   = 1.5
RPP_MIN_SPEED   = 0.3

# ── DWA ────────────────────────────────────────────────
DWA_V_SAMPLES   = 20
DWA_W_SAMPLES   = 30
DWA_SIM_TIME    = 2.0
DWA_HEADING_W   = 0.8
DWA_CLEARANCE_W = 0.2
DWA_SPEED_W     = 0.4

# ── MPPI ───────────────────────────────────────────────
MPPI_K          = 500
MPPI_T          = 20
MPPI_SIGMA      = 0.5
MPPI_LAMBDA     = 1.0
MPPI_OBS_COST   = 1e6
MPPI_PATH_COST  = 1.0
MPPI_CTRL_COST  = 0.1

# ── Comfort Thresholds ─────────────────────────────────
MAX_COMFORTABLE_ACCEL = 1.5   # m/s²
MAX_COMFORTABLE_JERK  = 3.0   # m/s³
MIN_SAFE_DISTANCE     = 0.3   # m (robot radius)
```

---

## 🧱 Phase 9: requirements.txt

```
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
pandas>=2.0.0
pygame>=2.4.0
vpython>=7.6.4
```

---

## 🧱 Phase 10: README.md

The agent must also generate a `README.md` with:

```markdown
# Campus Buggy — Local Planner Comparison

## What this is
## How to install
## How to run
## How to run headless (no pygame, faster)
## Output files explained
## Controller descriptions (RPP, DWA, MPPI)
## Scenario descriptions
## Results summary (filled in after runs)
## Recommendation
## Known failures
```

---

## ✅ Agent Execution Order (Step-by-Step)

The agent should implement files in this exact order to avoid import errors:

```
Step 1:  config.py
Step 2:  requirements.txt
Step 3:  robot/bicycle_model.py
Step 4:  scenarios/scenario_base.py
Step 5:  scenarios/s1_static_box.py
Step 6:  scenarios/s2_person_crossing.py
Step 7:  scenarios/s3_narrow_gap.py
Step 8:  scenarios/s4_two_obstacles.py
Step 9:  scenarios/s5_sudden_obstacle.py
Step 10: metrics/recorder.py
Step 11: controllers/rpp.py
Step 12: controllers/dwa.py
Step 13: controllers/mppi.py
Step 14: visualization/pygame_viz.py
Step 15: visualization/plot_results.py
Step 16: main.py
Step 17: README.md
Step 18: Run: python main.py
Step 19: Check results/all_results.csv and results/plots/
```

---

## ⚠️ Edge Cases the Agent Must Handle

| Case | How to handle |
|---|---|
| Robot doesn't reach goal in TIME_LIMIT | Mark as `failure_reason = "timeout"` |
| Robot distance to obstacle < ROBOT_RADIUS | Mark as `failure_reason = "collision"`, stop run |
| Robot stops moving for >5s | Mark as `failure_reason = "stuck"` |
| MPPI samples all collide | Fall back to zero control (stop) |
| Division by zero in RPP curvature | Clamp or skip that step |
| Obstacle spawns inside robot (Scenario 5) | Adjust spawn position slightly |

---

## 📊 Expected Deliverables (What You Submit)

```
results/
├── all_results.csv           # Raw data: 150 rows (3 controllers × 5 scenarios × 10 runs)
├── summary_table.csv         # Aggregated mean/std per controller-scenario combo
└── plots/
    ├── path_comparison.png   # Paths overlaid per scenario
    ├── min_distance.png      # Box plot of clearances
    ├── completion_time.png   # Bar chart
    ├── accel_jerk.png        # Line plot with comfort threshold
    └── failure_heatmap.png   # Controller × scenario failure counts
```

---

## 🏁 Final Recommendation Template (For Write-up)

After results, fill this in:

```
Best overall controller: [RPP / DWA / MPPI]
Reason: [fastest / safest / smoothest]

Fails in scenario: [s2 / s5 / ...]
Failure mode: [too slow to react / oscillates in narrow gap / ...]

Second choice: [...]
Use when: [...]

Not recommended: [...]
Because: [...]
```

---

*This plan is self-contained. Give it entirely to your coding agent. The agent should implement every file in the order listed, test with `python main.py`, and verify that `results/all_results.csv` is populated before generating plots.*
