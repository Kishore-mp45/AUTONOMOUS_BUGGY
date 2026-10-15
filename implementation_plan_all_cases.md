# 🚗 Upgrade Implementation Plan — Autonomous Campus Buggy
## Modifying Existing Codebase → Production-Grade End-to-End System
### Kishore | 3rd Year | October 2026

---

> **IMPORTANT — How to read this plan:**
> This plan is NOT a rebuild from scratch.
> The existing base system (Phases 1–10 of the original plan) is already implemented.
> Every phase below is a **modification, extension, or addition** to that existing code.
> Files marked 🔧 MODIFY mean edit the existing file.
> Files marked ➕ ADD NEW mean create a new file that didn't exist before.
> Files marked ✅ KEEP mean no changes needed.

---

## 📊 Gap Analysis — What Exists vs What's Missing

### What the existing base system has (already built):
| Component | File | Status |
|---|---|---|
| Robot model | robot/bicycle_model.py | ✅ Basic kinematic bicycle |
| Controller | controllers/rpp.py | ✅ Basic RPP |
| Controller | controllers/dwa.py | ✅ Basic DWA |
| Controller | controllers/mppi.py | ✅ Basic MPPI |
| Scenario 1 | scenarios/s1_static_box.py | ✅ Static obstacle |
| Scenario 2 | scenarios/s2_person_crossing.py | ✅ Person crossing |
| Scenario 3 | scenarios/s3_narrow_gap.py | ✅ Narrow gap |
| Scenario 4 | scenarios/s4_two_obstacles.py | ✅ Two obstacles |
| Scenario 5 | scenarios/s5_sudden_obstacle.py | ✅ Sudden spawn |
| Metrics | metrics/recorder.py | ✅ Basic CSV recording |
| Visualization | visualization/pygame_viz.py | ✅ Basic pygame |
| Visualization | visualization/plot_results.py | ✅ Basic 5 plots |
| Entry point | main.py | ✅ Basic runner |
| Config | config.py | ✅ Basic parameters |

### What is MISSING (to be added in this upgrade plan):
| What | Why it matters |
|---|---|
| Proper 2D costmap with inflation | Controllers currently use raw obstacle lists — no cost gradients |
| MPPI cost critics (5 components) | Current MPPI cost is too simple — needs obstacle + path + jerk + goal + speed critics |
| Ackermann constraint in DWA | DWA uses (v, omega) but buggy needs (v, delta) — wrong for car-like model |
| RPP adaptive lookahead | Fixed lookahead fails in narrow gap and at high speed |
| Edge case handling (8 cases) | No recovery behavior, no oscillation detection, no dead-end handling |
| Seed-based reproducibility | Runs are not reproducible — no seed control |
| Reaction time metric | No measurement of how fast each controller reacts to sudden obstacle |
| Controller health monitor | No detection of stuck/oscillation/hunting mid-run |
| Per-timestep raw CSV | Only summary CSV exists — no timestep-level data for analysis |
| Automated report generator | No auto-generation of summary table and all 5 plots in one command |
| Edge case scenario files | No crowd, dead-end, sensor noise, or tight-turn scenarios |
| README auto-populated | README template exists but not filled with actual results |
| Unit tests | No tests for vehicle model or controllers |
| Config validation | No check that config values are physically consistent |

---

## 🗂️ Upgrade Folder Structure

```
campus_buggy_sim/                          ← existing root (unchanged)
│
├── config.py                              🔧 MODIFY — add new params
├── main.py                                🔧 MODIFY — add seed, phases, edge cases
├── requirements.txt                       🔧 MODIFY — add scipy, tqdm
├── README.md                              🔧 MODIFY — auto-populate after runs
│
├── robot/
│   └── bicycle_model.py                   🔧 MODIFY — add footprint, jerk tracking, Ackermann constraint
│
├── controllers/
│   ├── rpp.py                             🔧 MODIFY — adaptive lookahead, obstacle regulation
│   ├── dwa.py                             🔧 MODIFY — fix to Ackermann (v, delta) window
│   ├── mppi.py                            🔧 MODIFY — add 5 cost critics, numpy vectorization
│   └── controller_base.py                 ➕ ADD NEW — shared base class for all controllers
│
├── environment/                           ➕ ADD NEW folder
│   ├── __init__.py
│   ├── costmap.py                         ➕ ADD NEW — 2D grid with inflation layer
│   └── obstacle.py                        ➕ ADD NEW — unified obstacle class (static + dynamic)
│
├── scenarios/
│   ├── scenario_base.py                   🔧 MODIFY — add seed, costmap integration, is_complete()
│   ├── s1_static_box.py                   🔧 MODIFY — integrate costmap, add seed
│   ├── s2_person_crossing.py              🔧 MODIFY — add frozen person edge case
│   ├── s3_narrow_gap.py                   🔧 MODIFY — tighten gap, add centerline metric
│   ├── s4_two_obstacles.py                🔧 MODIFY — add offset stagger, double-weave test
│   ├── s5_sudden_obstacle.py              🔧 MODIFY — add reaction time measurement
│   └── edge_cases/                        ➕ ADD NEW folder
│       ├── __init__.py
│       ├── ec1_dead_end.py
│       ├── ec2_oscillation_trigger.py
│       ├── ec3_tight_turn.py
│       ├── ec4_late_braking.py
│       └── ec5_crowd.py
│
├── metrics/
│   ├── recorder.py                        🔧 MODIFY — add timestep CSV, reaction time, jerk sequence
│   └── report_generator.py                ➕ ADD NEW — auto-generates all plots + summary table
│
├── health/                                ➕ ADD NEW folder
│   ├── __init__.py
│   └── monitor.py                         ➕ ADD NEW — detects stuck/oscillation/collision mid-run
│
├── visualization/
│   ├── pygame_viz.py                      🔧 MODIFY — add MPPI trajectory samples display, HUD
│   └── plot_results.py                    🔧 MODIFY — add reaction time plot, edge case plots
│
├── tests/                                 ➕ ADD NEW folder
│   ├── test_bicycle_model.py
│   ├── test_mppi.py
│   ├── test_dwa.py
│   ├── test_rpp.py
│   └── test_costmap.py
│
└── results/
    ├── raw/                               ✅ exists — now also gets timestep CSVs
    ├── plots/                             ✅ exists — now gets 8 plots instead of 5
    └── edge_case_results/                 ➕ ADD NEW subfolder
```

---

## ═══════════════════════════════════════════
## PHASE A — Foundation Fixes
## (Fix the base before adding anything new)
## ═══════════════════════════════════════════

**Why first:** Controllers share the robot model and config. If those have bugs,
every downstream result is wrong. Fix the foundation before extending it.

---

### A1 — 🔧 MODIFY `config.py`

**What to add** (append to existing file, do not remove existing params):

```python
# ── Costmap (NEW) ───────────────────────────────────────
COSTMAP_RESOLUTION  = 0.05    # metres per cell
INFLATION_RADIUS    = 0.5     # metres — expand obstacles outward
INFLATION_DECAY     = 3.0     # decay rate for cost falloff
LETHAL_COST         = 99      # cost value = collision
INSCRIBED_COST      = 80      # cost inside inflation radius

# ── MPPI extended critics (NEW) ────────────────────────
MPPI_OBSTACLE_WEIGHT  = 1.0
MPPI_GOAL_WEIGHT      = 2.0
MPPI_PATH_WEIGHT      = 1.5
MPPI_JERK_WEIGHT      = 0.5
MPPI_SPEED_WEIGHT     = 0.3

# ── DWA Ackermann fix (NEW) ────────────────────────────
DWA_STEER_SAMPLES   = 30      # steering angle samples (replaces W_SAMPLES)
DWA_STEER_RATE      = 0.3     # max steering change per timestep (rad/s)
DWA_MIN_TURN_RADIUS = 0.5     # metres — physical Ackermann constraint

# ── RPP adaptive lookahead (NEW) ───────────────────────
RPP_LOOKAHEAD_GAIN  = 0.5     # L_d = base + gain * speed
RPP_MIN_LOOKAHEAD   = 0.4     # metres
RPP_MAX_LOOKAHEAD   = 3.0     # metres
RPP_STEER_LOWPASS   = 0.7     # low-pass filter coefficient (0=heavy, 1=off)

# ── Health monitor (NEW) ───────────────────────────────
STUCK_TIMEOUT       = 5.0     # seconds not moving → stuck
STUCK_SPEED_THRESH  = 0.05    # m/s — below this = not moving
OSCILLATION_WINDOW  = 10      # timesteps to check for oscillation
OSCILLATION_THRESH  = 3       # sign flips in steering → oscillating

# ── Edge case scenarios (NEW) ──────────────────────────
CROWD_N_PEDESTRIANS = 3       # number of moving people in crowd scenario
SENSOR_NOISE_POS    = 0.05    # metres — Gaussian position noise
SENSOR_NOISE_HDG    = 0.01    # radians — Gaussian heading noise

# ── Experiment reproducibility (NEW) ───────────────────
BASE_SEED           = 42      # run_id + BASE_SEED = per-run seed

# ── Comfort thresholds (upgrade existing values) ───────
MAX_COMFORTABLE_ACCEL = 1.5   # m/s²
MAX_COMFORTABLE_JERK  = 3.0   # m/s³
MIN_SAFE_DISTANCE     = 0.3   # m

# ── Config self-validation (NEW) ───────────────────────
def validate():
    assert MAX_SPEED > 0, "MAX_SPEED must be positive"
    assert DT > 0 and DT < 0.5, "DT must be small positive value"
    assert MPPI_K > 0, "Need at least 1 MPPI sample"
    assert INFLATION_RADIUS >= ROBOT_RADIUS, \
        "Inflation must be >= robot radius or robot can pass through lethal zone"
    assert RPP_MIN_LOOKAHEAD < RPP_MAX_LOOKAHEAD, "Lookahead bounds inverted"
    print("[config] All parameters valid.")
```

**Call `validate()` at the top of `main.py` before anything else.**

---

### A2 — 🔧 MODIFY `robot/bicycle_model.py`

**What to add to existing class:**

```python
# ADD: Jerk tracking (needed for metrics and MPPI cost)
def update(self, state, control, dt):
    # EXISTING: kinematic equations — keep as-is
    # ADD AFTER existing update:
    prev_accel = self._prev_accel
    current_accel = (new_v - state[3]) / dt
    self.jerk = (current_accel - prev_accel) / dt
    self._prev_accel = current_accel
    return new_state

# ADD: Ackermann minimum turning radius enforcement
def _enforce_ackermann(self, v, delta):
    """
    Ackermann constraint: turning radius = L / tan(delta)
    Must not go below DWA_MIN_TURN_RADIUS.
    """
    if abs(delta) > 1e-6:
        radius = self.wheelbase / np.tan(abs(delta))
        if radius < MIN_TURN_RADIUS:
            delta = np.sign(delta) * np.arctan(self.wheelbase / MIN_TURN_RADIUS)
    return v, delta

# ADD: Footprint as 4 corners (for collision checking)
def get_footprint(self, state):
    x, y, theta = state[0], state[1], state[2]
    L, W = 0.6, 0.4   # buggy length and width
    corners_local = [(-L/2,-W/2),(L/2,-W/2),(L/2,W/2),(-L/2,W/2)]
    c, s = np.cos(theta), np.sin(theta)
    return [(x + c*cx - s*cy, y + s*cx + c*cy) for cx,cy in corners_local]

# ADD: Reset internal state between runs
def reset(self):
    self._prev_accel = 0.0
    self.jerk = 0.0
```

---

## ═══════════════════════════════════════════
## PHASE B — New Environment Layer
## (Costmap + Obstacle unification)
## ═══════════════════════════════════════════

**Why:** The existing controllers use raw obstacle lists with manual distance checks.
This is fragile and inconsistent between DWA, RPP and MPPI.
A shared costmap gives all three controllers the same environment representation —
exactly how Nav2 does it.

---

### B1 — ➕ ADD NEW `environment/obstacle.py`

**Purpose:** Unified obstacle class replacing scattered dicts in each scenario.

```python
class Obstacle:
    """
    Single obstacle — static or dynamic.
    All scenarios create Obstacle objects; costmap reads them.
    """
    def __init__(self, x, y, radius=None, width=None, height=None,
                 dynamic=False, vx=0.0, vy=0.0, label=""):
        self.x = x
        self.y = y
        self.radius = radius        # circular obstacle (person, bike)
        self.width  = width         # rectangular obstacle (box, wall)
        self.height = height
        self.dynamic = dynamic
        self.vx = vx                # velocity (dynamic only)
        self.vy = vy
        self.label = label          # "person" | "box" | "wall" | "bike"
        self.active = True          # False = not yet spawned (Scenario 5)

    def step(self, dt):
        """Move dynamic obstacle one timestep."""
        if self.dynamic and self.active:
            self.x += self.vx * dt
            self.y += self.vy * dt

    def dist_to_point(self, px, py):
        """Minimum distance from point (px,py) to this obstacle surface."""
        if self.radius is not None:
            return max(0.0, np.hypot(px - self.x, py - self.y) - self.radius)
        else:
            # Rectangle: clamp point to rectangle, measure distance
            cx = np.clip(px, self.x - self.width/2, self.x + self.width/2)
            cy = np.clip(py, self.y - self.height/2, self.y + self.height/2)
            return np.hypot(px - cx, py - cy)
```

---

### B2 — ➕ ADD NEW `environment/costmap.py`

**Purpose:** 2D grid costmap with inflation — shared by all three controllers.

```python
from scipy.ndimage import distance_transform_edt
import numpy as np
from config import (COSTMAP_RESOLUTION, INFLATION_RADIUS,
                    INFLATION_DECAY, LETHAL_COST, MAP_W, MAP_H)

class Costmap2D:
    """
    2D occupancy grid with 3 layers:
      Layer 1 — Static:   walls, fixed structures
      Layer 2 — Obstacle: dynamic obstacles stamped each timestep
      Layer 3 — Inflation: cost gradient around all obstacles

    All controllers call costmap.query(x, y) to get the cost at a world point.
    """

    def __init__(self):
        self.res  = COSTMAP_RESOLUTION
        self.cols = int(MAP_W / self.res)
        self.rows = int(MAP_H / self.res)
        self.static_layer   = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.obstacle_layer = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.inflated       = np.zeros((self.rows, self.cols), dtype=np.float32)

    # ── World ↔ Grid conversion ─────────────────────────
    def world_to_cell(self, x, y):
        col = int(np.clip(x / self.res, 0, self.cols - 1))
        row = int(np.clip(y / self.res, 0, self.rows - 1))
        return row, col

    def cell_to_world(self, row, col):
        return col * self.res + self.res/2, row * self.res + self.res/2

    # ── Static layer ────────────────────────────────────
    def add_static_obstacle(self, obstacle):
        """Call once at scenario setup for walls and fixed boxes."""
        self._stamp_obstacle(self.static_layer, obstacle)

    # ── Dynamic update (called every timestep) ──────────
    def update(self, obstacles):
        """
        Clear obstacle layer, re-stamp all active obstacles,
        then recompute inflation. Called once per simulation step.
        """
        self.obstacle_layer[:] = 0.0
        for obs in obstacles:
            if obs.active:
                self._stamp_obstacle(self.obstacle_layer, obs)
        self._inflate()

    def _stamp_obstacle(self, layer, obs):
        """Mark cells occupied by this obstacle as LETHAL_COST."""
        if obs.radius is not None:
            # Circular obstacle
            cx, cy = obs.x, obs.y
            r_cells = int(obs.radius / self.res) + 1
            row0, col0 = self.world_to_cell(cx, cy)
            for dr in range(-r_cells, r_cells+1):
                for dc in range(-r_cells, r_cells+1):
                    r, c = row0+dr, col0+dc
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        wx, wy = self.cell_to_world(r, c)
                        if np.hypot(wx-cx, wy-cy) <= obs.radius:
                            layer[r, c] = LETHAL_COST
        else:
            # Rectangular obstacle
            for r in range(self.rows):
                for c in range(self.cols):
                    wx, wy = self.cell_to_world(r, c)
                    if (abs(wx - obs.x) <= obs.width/2 and
                            abs(wy - obs.y) <= obs.height/2):
                        layer[r, c] = LETHAL_COST

    def _inflate(self):
        """
        Inflate combined (static + obstacle) grid using distance transform.
        Cost falls off exponentially from obstacle surface.
        """
        combined = np.maximum(self.static_layer, self.obstacle_layer)
        free_mask = (combined < LETHAL_COST).astype(np.float32)
        dist_cells = distance_transform_edt(free_mask)
        dist_m = dist_cells * self.res
        decay_cost = 100.0 * np.exp(-INFLATION_DECAY * dist_m)
        self.inflated = np.where(combined >= LETHAL_COST, LETHAL_COST, decay_cost)

    # ── Query (called by all controllers) ───────────────
    def query(self, x, y):
        """Return cost [0–100] at world position (x,y). 100 = lethal."""
        r, c = self.world_to_cell(x, y)
        return float(self.inflated[r, c])

    def is_free(self, x, y):
        return self.query(x, y) < LETHAL_COST

    def nearest_obstacle_dist(self, x, y, obstacles):
        """Fast exact distance to nearest obstacle surface (for metrics)."""
        if not obstacles:
            return float('inf')
        return min(obs.dist_to_point(x, y) for obs in obstacles if obs.active)

    # ── Goal safety check (Edge Case 8) ─────────────────
    def project_goal_to_free(self, gx, gy, search_radius=2.0):
        """
        If goal is inside inflation radius, find nearest free cell.
        Returns (gx, gy) unchanged if already free.
        """
        if self.is_free(gx, gy):
            return gx, gy
        best_dist = float('inf')
        best = (gx, gy)
        steps = int(search_radius / self.res)
        gr, gc = self.world_to_cell(gx, gy)
        for dr in range(-steps, steps+1):
            for dc in range(-steps, steps+1):
                r, c = gr+dr, gc+dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    if self.inflated[r,c] < LETHAL_COST:
                        wx, wy = self.cell_to_world(r, c)
                        d = np.hypot(wx-gx, wy-gy)
                        if d < best_dist:
                            best_dist = d
                            best = (wx, wy)
        print(f"[costmap] Goal adjusted from ({gx:.2f},{gy:.2f}) to ({best[0]:.2f},{best[1]:.2f})")
        return best
```

---

## ═══════════════════════════════════════════
## PHASE C — Controller Upgrades
## (Modify existing controllers to production quality)
## ═══════════════════════════════════════════

---

### C0 — ➕ ADD NEW `controllers/controller_base.py`

**Purpose:** Shared base class. All three controllers inherit from this.

```python
class ControllerBase:
    """
    Common interface for RPP, DWA, MPPI.
    All controllers must implement compute_control().
    """
    name = "Base"

    def compute_control(self, state, path, goal, costmap, obstacles):
        """
        Args:
            state     : [x, y, theta, v]
            path      : list of (x,y) waypoints (reference path)
            goal      : (gx, gy)
            costmap   : Costmap2D instance
            obstacles : list of Obstacle objects
        Returns:
            (steering_angle [rad], target_speed [m/s])
        """
        raise NotImplementedError

    def reset(self):
        """Called at the start of each run. Reset internal state."""
        pass
```

**Update all three controllers:** change class definition to:
```python
class RegulatedPurePursuit(ControllerBase): ...
class DWA(ControllerBase): ...
class MPPI(ControllerBase): ...
```

And update `compute_control` signature in all three to match base class.

---

### C1 — 🔧 MODIFY `controllers/rpp.py`

**What to change:**

```
EXISTING: Fixed lookahead distance (RPP_LOOKAHEAD = 1.0)
UPGRADE:  Adaptive lookahead based on speed

EXISTING: Speed regulation based on simple obstacle distance threshold
UPGRADE:  Curvature-based speed regulation + low-pass filter on steering

EXISTING: No path end detection
UPGRADE:  Detect when near goal → stop cleanly
```

**Specific code changes:**

```python
# CHANGE 1: Adaptive lookahead (replace fixed lookahead)
def _get_lookahead(self, v):
    """
    Lookahead grows with speed. At low speed → short (precise).
    At high speed → longer (stable). Clamped to [min, max].
    """
    L_d = RPP_LOOKAHEAD + RPP_LOOKAHEAD_GAIN * abs(v)
    return np.clip(L_d, RPP_MIN_LOOKAHEAD, RPP_MAX_LOOKAHEAD)

# CHANGE 2: Low-pass filter on steering (prevents oscillation)
def _filter_steer(self, new_steer):
    """
    Exponential moving average: prevents sudden steering jumps.
    alpha=0.7 → 30% smoothing.
    """
    self._prev_steer = (RPP_STEER_LOWPASS * self._prev_steer
                        + (1 - RPP_STEER_LOWPASS) * new_steer)
    return self._prev_steer

# CHANGE 3: Updated compute_control signature (add costmap)
def compute_control(self, state, path, goal, costmap, obstacles):
    x, y, theta, v = state
    L_d = self._get_lookahead(v)

    # Find lookahead point (EXISTING logic — keep)
    lookahead_pt = self._find_lookahead_point(state, path, L_d)
    if lookahead_pt is None:
        return 0.0, 0.0   # reached end of path

    # Compute steering (EXISTING logic — keep)
    alpha = np.arctan2(lookahead_pt[1]-y, lookahead_pt[0]-x) - theta
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # wrap to [-pi, pi]
    steer = np.arctan2(2 * WHEELBASE * np.sin(alpha), L_d)
    steer = np.clip(steer, -MAX_STEER, MAX_STEER)

    # NEW: Apply low-pass filter
    steer = self._filter_steer(steer)

    # NEW: Curvature-based speed regulation
    curvature = abs(np.tan(steer) / WHEELBASE)
    speed = RPP_MAX_SPEED / (1 + 2.0 * curvature)  # slow on sharp turns
    speed = max(speed, RPP_MIN_SPEED)

    # NEW: Obstacle proximity regulation (uses costmap)
    cost_at_pos = costmap.query(x, y)
    if cost_at_pos > 50:
        speed *= (1.0 - cost_at_pos / 100.0)  # reduce speed in high-cost zone
    speed = max(speed, RPP_MIN_SPEED)

    return steer, speed

# CHANGE 4: reset() to clear filter state between runs
def reset(self):
    self._prev_steer = 0.0
```

---

### C2 — 🔧 MODIFY `controllers/dwa.py`

**Critical fix:** Existing DWA uses `(v, omega)` — angular velocity.
The buggy is Ackermann-steered, so the correct window is `(v, delta)` — steering angle.
This is not cosmetic — it changes the entire dynamic window and scoring.

```
EXISTING: Sample (v, omega) pairs → simulate as differential drive
UPGRADE:  Sample (v, delta) pairs → simulate as kinematic bicycle
          Add Ackermann minimum turning radius constraint
          Add steering rate limit (can't change steer instantly)
```

**Specific code changes:**

```python
# CHANGE 1: Build Ackermann dynamic window
def _build_window(self, state):
    """
    Reachable (v, delta) in one timestep given:
      - acceleration limits on v
      - steering rate limit on delta
    """
    x, y, theta, v = state
    v_min = max(0.0, v - MAX_ACCEL * DT)
    v_max = min(MAX_SPEED, v + MAX_ACCEL * DT)
    d_min = max(-MAX_STEER, self._prev_steer - DWA_STEER_RATE * DT)
    d_max = min(MAX_STEER,  self._prev_steer + DWA_STEER_RATE * DT)
    return v_min, v_max, d_min, d_max

# CHANGE 2: Simulate trajectory using bicycle model (not differential drive)
def _simulate(self, state, v, delta):
    """
    Forward simulate bicycle model for DWA_SIM_TIME seconds.
    Returns list of (x, y) positions.
    """
    traj = []
    s = list(state)
    steps = int(DWA_SIM_TIME / DT)
    for _ in range(steps):
        s[0] += v * np.cos(s[2]) * DT
        s[1] += v * np.sin(s[2]) * DT
        s[2] += (v / WHEELBASE) * np.tan(delta) * DT
        s[3]  = v
        traj.append((s[0], s[1]))
    return traj

# CHANGE 3: Ackermann constraint check (ADD before scoring)
def _is_ackermann_feasible(self, v, delta):
    """Reject (v, delta) if turning radius is below physical minimum."""
    if abs(delta) < 1e-6:
        return True  # going straight — always feasible
    radius = abs(WHEELBASE / np.tan(delta))
    return radius >= DWA_MIN_TURN_RADIUS

# CHANGE 4: Updated compute_control
def compute_control(self, state, path, goal, costmap, obstacles):
    v_min, v_max, d_min, d_max = self._build_window(state)

    v_samples     = np.linspace(v_min, v_max, DWA_V_SAMPLES)
    steer_samples = np.linspace(d_min, d_max, DWA_STEER_SAMPLES)

    best_score = -float('inf')
    best_v, best_d = 0.0, 0.0

    for v in v_samples:
        for delta in steer_samples:
            # NEW: Ackermann feasibility check
            if not self._is_ackermann_feasible(v, delta):
                continue

            traj = self._simulate(state, v, delta)

            # Collision check via costmap
            collision = any(costmap.query(px, py) >= LETHAL_COST
                            for px, py in traj)
            if collision:
                continue

            # Scoring (EXISTING logic — keep headings/clearance/speed)
            heading_score   = self._heading_score(traj[-1], goal)
            clearance_score = self._clearance_score(traj, costmap)
            speed_score     = v / MAX_SPEED

            score = (DWA_HEADING_W   * heading_score +
                     DWA_CLEARANCE_W * clearance_score +
                     DWA_SPEED_W     * speed_score)

            if score > best_score:
                best_score = score
                best_v, best_d = v, delta

    # Edge case: no valid trajectory found → emergency stop
    if best_score == -float('inf'):
        print("[DWA] No valid trajectory — emergency stop")
        return 0.0, 0.0

    self._prev_steer = best_d
    return best_d, best_v

# ADD: reset()
def reset(self):
    self._prev_steer = 0.0
```

---

### C3 — 🔧 MODIFY `controllers/mppi.py`

**What to change:**

```
EXISTING: Single cost = obstacle_cost + path_cost + control_cost
UPGRADE:  5-critic cost system (obstacle, goal, path alignment, jerk, speed)
          Vectorized numpy rollout (all K samples at once — faster)
          Warm-starting: shift nominal control sequence each step
          Fallback: if all K samples collide → emergency stop
```

**Specific code changes:**

```python
# CHANGE 1: Vectorized rollout (replaces Python for-loop over K)
def _rollout_batch(self, state, U_noisy):
    """
    state   : shape (4,)
    U_noisy : shape (K, T, 2) — [steer, accel] for each sample and step
    Returns : shape (K, T, 4) — full state trajectories
    """
    K, T, _ = U_noisy.shape
    states = np.zeros((K, T+1, 4))
    states[:, 0, :] = state

    for t in range(T):
        x, y, th, v = (states[:, t, 0], states[:, t, 1],
                        states[:, t, 2], states[:, t, 3])
        delta = np.clip(U_noisy[:, t, 0], -MAX_STEER, MAX_STEER)
        a     = np.clip(U_noisy[:, t, 1], -MAX_ACCEL, MAX_ACCEL)
        states[:, t+1, 0] = x + v * np.cos(th) * DT
        states[:, t+1, 1] = y + v * np.sin(th) * DT
        states[:, t+1, 2] = th + (v / WHEELBASE) * np.tan(delta) * DT
        states[:, t+1, 3] = np.clip(v + a * DT, 0.0, MAX_SPEED)

    return states[:, 1:, :]  # shape (K, T, 4)

# CHANGE 2: 5-critic cost function (replaces single cost)
def _compute_costs(self, trajectories, goal, path, costmap):
    """
    trajectories : shape (K, T, 4)
    Returns      : shape (K,) total cost per sample
    """
    K, T, _ = trajectories.shape
    costs = np.zeros(K)

    gx, gy = goal

    for t in range(T):
        xs = trajectories[:, t, 0]
        ys = trajectories[:, t, 1]
        vs = trajectories[:, t, 3]

        # Critic 1: Obstacle cost (highest priority)
        for k in range(K):
            cell_cost = costmap.query(xs[k], ys[k])
            if cell_cost >= LETHAL_COST:
                costs[k] += MPPI_OBSTACLE_COST     # lethal → huge penalty
            else:
                costs[k] += MPPI_OBSTACLE_WEIGHT * cell_cost

        # Critic 2: Goal cost (distance to goal at each step)
        goal_dist = np.hypot(xs - gx, ys - gy)
        costs += MPPI_GOAL_WEIGHT * goal_dist

        # Critic 3: Path alignment (cross-track error to reference path)
        if path:
            path_arr = np.array(path)
            dists = np.hypot(xs[:,None] - path_arr[:,0],
                             ys[:,None] - path_arr[:,1])
            cte = np.min(dists, axis=1)
            costs += MPPI_PATH_WEIGHT * cte

        # Critic 4: Speed cost (penalize stopping unnecessarily)
        costs += MPPI_SPEED_WEIGHT * (MAX_SPEED - vs)**2

    # Critic 5: Jerk cost over full trajectory (comfort)
    if T > 2:
        accels = np.diff(trajectories[:, :, 3], axis=1) / DT  # (K, T-1)
        jerks  = np.diff(accels, axis=1) / DT                  # (K, T-2)
        costs += MPPI_JERK_WEIGHT * np.sum(jerks**2, axis=1)

    return costs

# CHANGE 3: Updated compute_control with warm-starting and fallback
def compute_control(self, state, path, goal, costmap, obstacles):
    # Sample noise
    eps = np.random.randn(MPPI_K, MPPI_T, 2)
    eps[:, :, 0] *= MPPI_SIGMA    # steering noise
    eps[:, :, 1] *= MPPI_SIGMA * 0.5  # acceleration noise

    # Perturb nominal control
    U_noisy = self.U[None, :, :] + eps    # shape (K, T, 2)
    U_noisy[:, :, 0] = np.clip(U_noisy[:, :, 0], -MAX_STEER, MAX_STEER)
    U_noisy[:, :, 1] = np.clip(U_noisy[:, :, 1], -MAX_ACCEL, MAX_ACCEL)

    # Rollout all K samples
    trajs = self._rollout_batch(state, U_noisy)

    # Score all samples
    costs = self._compute_costs(trajs, goal, path, costmap)

    # Fallback: all samples collide → emergency stop
    min_cost = np.min(costs)
    if min_cost >= MPPI_OBSTACLE_COST:
        print("[MPPI] All samples collide — emergency stop")
        self.U[:] = 0.0
        return 0.0, 0.0

    # MPPI weight update
    beta = np.min(costs)
    weights = np.exp(-(costs - beta) / MPPI_LAMBDA)
    weights /= np.sum(weights)

    # Control update
    for t in range(MPPI_T):
        self.U[t] += np.sum(weights[:, None] * eps[:, t, :], axis=0)

    # Warm-start: shift nominal sequence forward
    u0 = self.U[0].copy()
    self.U[:-1] = self.U[1:]
    self.U[-1]  = 0.0

    steer = np.clip(u0[0], -MAX_STEER, MAX_STEER)
    speed = np.clip(state[3] + u0[1] * DT, 0.0, MAX_SPEED)
    return steer, speed

# ADD: reset() between runs
def reset(self):
    self.U = np.zeros((MPPI_T, 2))   # reset nominal control sequence
```

---

## ═══════════════════════════════════════════
## PHASE D — Scenario Upgrades
## (Integrate costmap; add seed; strengthen each scenario)
## ═══════════════════════════════════════════

---

### D0 — 🔧 MODIFY `scenarios/scenario_base.py`

**What to add:**

```python
# ADD: Seed support (reproducibility)
def reset(self, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # ... existing reset logic

# ADD: Costmap integration
def build_costmap(self):
    """
    Each scenario builds its own costmap at reset.
    Called once per run after reset().
    Returns a Costmap2D with static obstacles pre-stamped.
    """
    from environment.costmap import Costmap2D
    cm = Costmap2D()
    for obs in self.static_obstacles:
        cm.add_static_obstacle(obs)
    return cm

# ADD: is_complete() — unified termination check
def is_complete(self, state, t):
    """
    Returns (done: bool, success: bool, reason: str)
    Scenarios override this for custom completion logic.
    """
    x, y = state[0], state[1]
    gx, gy = self.goal
    if np.hypot(x - gx, y - gy) < 0.5:    # within 0.5m of goal
        return True, True, ""
    if t >= self.time_limit:
        return True, False, "timeout"
    return False, False, ""

# ADD: Use Obstacle class (replace raw dicts)
# All self.obstacles must be Obstacle objects (not dicts)
```

---

### D1 — 🔧 MODIFY `scenarios/s1_static_box.py`

```
CHANGE: Replace obstacle dict with Obstacle object
CHANGE: Call build_costmap() at reset
CHANGE: Add seed parameter to reset()
ADD:    A second static scenario variant (box off-center) for robustness
```

```python
from environment.obstacle import Obstacle

def reset(self, seed=None):
    super().reset(seed)
    self.obstacles = [
        Obstacle(x=10, y=10, width=1.5, height=1.5, label="box")
    ]
    self.static_obstacles = self.obstacles
    self.costmap = self.build_costmap()
    return self._initial_state()
```

---

### D2 — 🔧 MODIFY `scenarios/s2_person_crossing.py`

```
CHANGE: Person is now an Obstacle(dynamic=True, vx=..., vy=...)
CHANGE: Costmap updated each timestep as person moves
ADD:    Frozen person edge case — person stops at x=10, y=10 (mid-path)
        Triggered when run_id % 3 == 0 (every 3rd run tests frozen case)
ADD:    Person reverses direction at t=12s (unpredictable pedestrian)
```

```python
def reset(self, seed=None):
    super().reset(seed)
    np.random.seed(seed)
    self.frozen = (seed is not None and seed % 3 == 0)  # every 3rd run
    self.person = Obstacle(
        x=10, y=5, radius=0.4, dynamic=True,
        vx=0.0, vy=0.5, label="person"
    )
    self.obstacles = [self.person]
    self.costmap = self.build_costmap()

def update(self, t):
    if self.frozen and self.person.y >= 10.0:
        self.person.vy = 0.0  # freeze mid-crossing
    elif t > 12.0:
        self.person.vy = -0.3  # reverse direction at t=12s
    self.person.step(DT)
    self.costmap.update(self.obstacles)
```

---

### D3 — 🔧 MODIFY `scenarios/s3_narrow_gap.py`

```
CHANGE: Replace wall dicts with Obstacle objects
CHANGE: Add centerline deviation metric to scenario
ADD:    Gap width test variants — 3.0m (easy), 2.0m (medium), 1.8m (hard)
        Selected by run_id: runs 0-3=easy, 4-6=medium, 7-9=hard
ADD:    Lateral deviation from centerline logged as extra metric
```

```python
def reset(self, seed=None):
    super().reset(seed)
    # Gap width varies by run
    if seed is not None and seed % 10 < 4:
        gap = 3.0    # easy
    elif seed is not None and seed % 10 < 7:
        gap = 2.0    # medium
    else:
        gap = 1.8    # hard (most realistic campus scenario)

    half = gap / 2
    self.obstacles = [
        Obstacle(x=10, y=(20-half)/2,    width=0.3, height=20-half-gap, label="wall_top"),
        Obstacle(x=10, y=-(20-half)/2+gap, width=0.3, height=20-half-gap, label="wall_bot"),
    ]
    self.static_obstacles = self.obstacles
    self.costmap = self.build_costmap()
    self.gap_width = gap   # stored for reporting
```

---

### D4 — 🔧 MODIFY `scenarios/s4_two_obstacles.py`

```
CHANGE: Use Obstacle objects
ADD:    Second obstacle is semi-dynamic — drifts sideways at 0.1 m/s
        Tests whether controller handles slowly-moving obstacle mid-weave
ADD:    Log number of control replans for each controller
```

---

### D5 — 🔧 MODIFY `scenarios/s5_sudden_obstacle.py`

```
CHANGE: Use Obstacle with active=False initially, set active=True at spawn time
ADD:    Reaction time metric:
          reaction_time = first timestep where |accel| > 0.3 m/s² after spawn
          Passed to recorder as a dedicated field
ADD:    Two spawn distances: 2m (hard) and 4m (easy)
        run_id < 5 → hard (2m), run_id >= 5 → easy (4m)
```

```python
def update(self, t):
    if t >= SPAWN_TIME and not self.obstacle.active:
        self.obstacle.active = True
        self.spawn_timestep = int(t / DT)
        self.costmap.update(self.obstacles)
    if self.obstacle.active:
        self.costmap.update(self.obstacles)
```

---

### D6 — ➕ ADD NEW `scenarios/edge_cases/`

---

#### EC1 — `ec1_dead_end.py`

```
What: Buggy enters a U-shaped dead end. All forward paths blocked.
How:  Three walls form a U at x=12–15. Goal is at x=19.
Expected: Controller must trigger recovery — rotate and reverse.
Failure if: Stuck for >5s without moving.
Log: "recovery_triggered" flag in results.
```

```python
OBSTACLES = [
    Obstacle(x=12, y=8,  width=0.3, height=4.0, label="wall_left"),
    Obstacle(x=15, y=8,  width=0.3, height=4.0, label="wall_right"),
    Obstacle(x=13.5, y=10, width=3.3, height=0.3, label="wall_top"),
]
```

---

#### EC2 — `ec2_oscillation_trigger.py`

```
What: Long straight path with no obstacles — tests if RPP oscillates at speed.
How:  20m straight path, no obstacles. Run at MAX_SPEED.
Expected: RPP may hunt (oscillate) without low-pass filter.
Metric: Count steering sign changes per 10-step window.
         If > OSCILLATION_THRESH → log "oscillation_detected".
```

---

#### EC3 — `ec3_tight_turn.py`

```
What: Path requires a 90-degree turn at x=10 (L-shaped route).
How:  Path goes (1,1) → (10,1) → (10,19). Sharp left turn.
Expected: Tests Ackermann minimum turning radius enforcement.
Failure: Cutting the corner into the wall.
Metric: Min distance to corner wall during the turn.
```

---

#### EC4 — `ec4_late_braking.py`

```
What: Obstacle at 4m, buggy approaching at MAX_SPEED.
How:  Clear path until t=1.0s, then obstacle spawns 4m ahead.
      Buggy is at full speed — braking distance at 1.5 m/s = ~0.45m.
Expected: Emergency stop must trigger within one planning cycle.
Metric: Time to deceleration, final closest approach distance.
```

---

#### EC5 — `ec5_crowd.py`

```
What: Three pedestrians cross at different times and angles.
How:
  Person 1: crosses at x=8,  y: 5→15, speed=0.8 m/s, starts at t=3s
  Person 2: crosses at x=12, y: 15→5, speed=1.2 m/s, starts at t=6s
  Person 3: crosses at x=15, y: 5→15, speed=0.6 m/s, starts at t=9s
Expected: MPPI handles all three. DWA may freeze. RPP slows sharply.
Metric: Total hesitation time (v < 0.1 m/s while not at goal).
```

---

## ═══════════════════════════════════════════
## PHASE E — Health Monitor (NEW)
## (Detect problems mid-run, trigger recovery)
## ═══════════════════════════════════════════

### E1 — ➕ ADD NEW `health/monitor.py`

**Purpose:** Called every timestep inside the simulation loop.
Detects stuck, oscillation, and collision BEFORE they become silent failures.

```python
class HealthMonitor:
    """
    Monitors controller health every timestep.
    Returns action: "ok" | "stuck" | "oscillating" | "collision"
    """

    def __init__(self):
        self.speed_history  = []    # recent speeds
        self.steer_history  = []    # recent steering commands
        self.collision_flag = False

    def check(self, state, control, costmap):
        """
        Call every timestep. Returns (status, message).
        """
        x, y, theta, v = state
        steer, speed = control

        # ── Collision check ────────────────────────────
        if costmap.query(x, y) >= LETHAL_COST:
            self.collision_flag = True
            return "collision", f"Collision at ({x:.2f},{y:.2f})"

        # ── Stuck check ────────────────────────────────
        self.speed_history.append(abs(v))
        if len(self.speed_history) > int(STUCK_TIMEOUT / DT):
            self.speed_history.pop(0)
            if max(self.speed_history) < STUCK_SPEED_THRESH:
                return "stuck", "Robot not moving for >5s"

        # ── Oscillation check ─────────────────────────
        self.steer_history.append(steer)
        if len(self.steer_history) > OSCILLATION_WINDOW:
            self.steer_history.pop(0)
            sign_flips = sum(
                1 for i in range(1, len(self.steer_history))
                if self.steer_history[i] * self.steer_history[i-1] < 0
            )
            if sign_flips >= OSCILLATION_THRESH:
                return "oscillating", f"{sign_flips} steer sign flips in {OSCILLATION_WINDOW} steps"

        return "ok", ""

    def reset(self):
        self.speed_history  = []
        self.steer_history  = []
        self.collision_flag = False
```

---

## ═══════════════════════════════════════════
## PHASE F — Metrics Recorder Upgrade
## ═══════════════════════════════════════════

### F1 — 🔧 MODIFY `metrics/recorder.py`

**What to add to existing recorder:**

```python
# ADD FIELDS to __init__:
self.timestep_log   = []       # full timestep-level data
self.reaction_time  = None     # seconds from event to deceleration
self.oscillations   = 0        # count of oscillation events detected
self.recoveries     = 0        # count of recovery maneuvers triggered
self.gap_width      = None     # for S3 narrow gap runs
self.spawn_timestep = None     # for S5 reaction time calculation

# ADD to step() method:
def step(self, state, control, obstacles, costmap, t):
    # EXISTING: positions, velocities, min_dist — keep

    # ADD: per-timestep structured log
    self.timestep_log.append({
        "t":             t,
        "x":             state[0],
        "y":             state[1],
        "theta":         state[2],
        "v":             state[3],
        "steer":         control[0],
        "target_speed":  control[1],
        "costmap_cost":  costmap.query(state[0], state[1]),
        "min_obs_dist":  costmap.nearest_obstacle_dist(
                             state[0], state[1], obstacles),
    })

# ADD: save_timestep_csv()
def save_timestep_csv(self, out_dir="results/raw"):
    import pandas as pd, os
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{out_dir}/{self.controller}_{self.scenario}_run{self.run_id:02d}.csv"
    pd.DataFrame(self.timestep_log).to_csv(fname, index=False)

# ADD: reaction_time recording (for S5)
def record_spawn_event(self, spawn_timestep):
    self.spawn_timestep = spawn_timestep

def compute_reaction_time(self):
    if self.spawn_timestep is None:
        return None
    # Find first timestep after spawn where |accel| > 0.3 m/s²
    for i, row in enumerate(self.timestep_log):
        if i < self.spawn_timestep:
            continue
        if i > 0:
            dv = abs(self.timestep_log[i]["v"] - self.timestep_log[i-1]["v"])
            a  = dv / DT
            if a > 0.3:
                self.reaction_time = (i - self.spawn_timestep) * DT
                return self.reaction_time
    self.reaction_time = None
    return None

# MODIFY: to_dict() — add new fields
def to_dict(self):
    # EXISTING fields — keep all
    d = { ... }   # keep existing
    # ADD:
    d["reaction_time_s"] = self.reaction_time
    d["oscillation_count"] = self.oscillations
    d["recovery_count"] = self.recoveries
    d["gap_width_m"] = self.gap_width
    return d
```

---

### F2 — ➕ ADD NEW `metrics/report_generator.py`

**Purpose:** Single command generates all 8 plots + summary table from CSV.
Run after `main.py` completes.

```python
"""
Usage: python -m metrics.report_generator
Reads:  results/all_results.csv + results/raw/*.csv
Writes: results/summary_table.csv
        results/plots/plot1_path_comparison.png
        results/plots/plot2_min_distance.png
        results/plots/plot3_completion_time.png
        results/plots/plot4_accel_jerk.png
        results/plots/plot5_failure_heatmap.png
        results/plots/plot6_reaction_time.png   (NEW)
        results/plots/plot7_path_smoothness.png (NEW)
        results/plots/plot8_edge_case_summary.png (NEW)
"""

def plot6_reaction_time(df):
    """Bar chart of mean reaction time for S5 per controller."""
    s5 = df[df["scenario"] == "s5_sudden_obstacle"]
    ...

def plot7_path_smoothness(raw_dir):
    """Jerk over time for each controller — one line per scenario."""
    ...

def plot8_edge_case_summary(ec_df):
    """Success rate per controller per edge case scenario."""
    ...
```

---

## ═══════════════════════════════════════════
## PHASE G — Main Runner Upgrade
## ═══════════════════════════════════════════

### G1 — 🔧 MODIFY `main.py`

**What to change:**

```python
# CHANGE 1: Add config validation at top
from config import validate
validate()

# CHANGE 2: Add seed-based reproducibility
for run_id in range(RUNS_PER_COMBO):
    seed = BASE_SEED + run_id
    scenario.reset(seed=seed)
    controller.reset()          # NEW: reset controller state each run
    monitor.reset()             # NEW: reset health monitor each run

# CHANGE 3: Integrate costmap into simulation loop
costmap = scenario.build_costmap()

for step_i in range(MAX_STEPS):
    t = step_i * DT

    # Update dynamic obstacles + costmap
    scenario.update(t)
    costmap.update(scenario.get_obstacles())

    # Controller decision (new signature with costmap)
    control = controller.compute_control(
        state, scenario.global_path, scenario.goal, costmap, scenario.get_obstacles()
    )

    # Health check (NEW)
    status, msg = monitor.check(state, control, costmap)
    if status == "collision":
        recorder.finish(success=False, reason="collision")
        break
    elif status == "stuck":
        recorder.finish(success=False, reason="stuck")
        break
    elif status == "oscillating":
        recorder.oscillations += 1   # log but keep running

    # Step vehicle
    state = robot.update(state, control, DT)

    # Record
    recorder.step(state, control, scenario.get_obstacles(), costmap, t)

    # Termination
    done, success, reason = scenario.is_complete(state, t)
    if done:
        recorder.finish(success=success, reason=reason)
        break

# Save timestep CSV (NEW)
recorder.save_timestep_csv()

# CHANGE 4: Add edge case scenarios to the run loop (separate block)
EDGE_SCENARIOS = {
    "ec1_dead_end":         EC1DeadEnd(),
    "ec2_oscillation":      EC2OscillationTrigger(),
    "ec3_tight_turn":       EC3TightTurn(),
    "ec4_late_braking":     EC4LateBraking(),
    "ec5_crowd":            EC5Crowd(),
}
# Same run structure, results saved to results/edge_case_results/

# CHANGE 5: Add --headless and --edge-only flags
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless",   action="store_true")
parser.add_argument("--edge-only",  action="store_true")
parser.add_argument("--report",     action="store_true",
                    help="Generate report after runs")
args = parser.parse_args()
```

---

## ═══════════════════════════════════════════
## PHASE H — Visualization Upgrades
## ═══════════════════════════════════════════

### H1 — 🔧 MODIFY `visualization/pygame_viz.py`

```
ADD: MPPI trajectory samples display
     Draw top-20 weighted trajectories as faint cyan lines behind robot
     Shows HOW MPPI thinks — useful for write-up screenshots

ADD: Health status HUD
     Show current: controller | scenario | run | status (ok / stuck / oscillating)

ADD: Costmap overlay (toggle with 'C' key)
     Show costmap inflation as colour gradient (green=free, yellow=near, red=lethal)

ADD: Reaction marker for S5
     Red flash at the moment the sudden obstacle spawns
```

### H2 — 🔧 MODIFY `visualization/plot_results.py`

```
ADD: Plot 6 — Reaction Time bar chart (S5 only)
ADD: Plot 7 — Path smoothness (jerk over time, all controllers, all scenarios)
ADD: Plot 8 — Edge case summary bar chart
MODIFY: Plot 5 failure heatmap — include edge case scenarios as extra columns
```

---

## ═══════════════════════════════════════════
## PHASE I — Unit Tests (NEW)
## ═══════════════════════════════════════════

### I1 — ➕ ADD NEW `tests/` folder

These are run with: `python -m pytest tests/ -v`

#### `tests/test_bicycle_model.py`
```python
def test_straight_line():
    # v=1.0, delta=0 → should move along x-axis only
def test_turning():
    # delta > 0 → heading should increase (left turn)
def test_speed_clamp():
    # accel = +100 → speed must not exceed MAX_SPEED
def test_steer_clamp():
    # steer = +100 → must not exceed MAX_STEER
def test_ackermann_constraint():
    # delta > arctan(L/R_min) → should be clamped
def test_jerk_tracking():
    # two consecutive accelerations → jerk computed correctly
```

#### `tests/test_costmap.py`
```python
def test_lethal_at_obstacle():
    # cell at obstacle center → cost == LETHAL_COST
def test_free_far_from_obstacle():
    # cell 5m from obstacle → cost should be near 0
def test_inflation_monotone():
    # cost increases as you approach obstacle center
def test_goal_projection():
    # goal inside obstacle → projected to nearest free cell
def test_update_clears_old_obstacle():
    # move dynamic obstacle → old cell should no longer be lethal
```

#### `tests/test_mppi.py`
```python
def test_produces_valid_control():
    # output steer in [-MAX_STEER, MAX_STEER]
def test_fallback_on_all_collide():
    # if every sample is lethal → returns (0, 0) not crash
def test_nominal_shifts_each_step():
    # call twice → U[0] at step 2 == U[1] at step 1
def test_cost_decreases_with_clear_path():
    # on clear path, cost should be lower than on blocked path
```

#### `tests/test_dwa.py`
```python
def test_ackermann_feasibility():
    # tight delta rejected when radius < DWA_MIN_TURN_RADIUS
def test_no_valid_trajectory_returns_stop():
    # all paths blocked → (0, 0) returned
def test_window_respects_speed_limits():
    # sampled v values all within [v_min, v_max]
```

#### `tests/test_rpp.py`
```python
def test_lookahead_grows_with_speed():
    # v=1.5 → L_d > v=0.5
def test_low_pass_filter_smooths_steer():
    # sudden steer change → filtered output should be gradual
def test_goal_reached_returns_zero():
    # state at goal → returns (0, 0)
```

---

## ═══════════════════════════════════════════
## PHASE J — Complete Edge Case & Critical Conditions Table
## ═══════════════════════════════════════════

All edge cases the system must handle. Each has a detection method,
a handler, and a log entry.

| # | Condition | Detection | Handler | Log Field |
|---|---|---|---|---|
| EC-1 | Dead end — no valid path | All MPPI samples collide / DWA no valid (v,d) / RPP lookahead blocked | Emergency stop, log recovery | failure_reason = "dead_end" |
| EC-2 | Controller oscillating | Steering sign flips > OSCILLATION_THRESH in window | Increase RPP lookahead, log | oscillation_count++ |
| EC-3 | Ackermann violation | v * tan(delta) / L > curvature_max | Hard-clamp delta | logged per step |
| EC-4 | Late braking — won't stop in time | Obstacle distance < braking_distance at current v | Emergency decel = MAX_DECEL | failure_reason = "near_miss" |
| EC-5 | Crowd — multiple pedestrians | 3+ dynamic obstacles in costmap | No special handler — tests controller naturally | hesitation_time added |
| EC-6 | Frozen pedestrian mid-path | Obstacle stopped at goal position | Same as dead-end path | failure_reason = "blocked_by_person" |
| EC-7 | Goal inside inflation radius | costmap.query(goal) >= INSCRIBED_COST | project_goal_to_free() | goal_adjusted = True |
| EC-8 | Stuck (not moving) | max(speed_history) < STUCK_SPEED_THRESH for STUCK_TIMEOUT | Mark failure | failure_reason = "stuck" |
| EC-9 | Timeout | t >= TIME_LIMIT | Mark failure | failure_reason = "timeout" |
| EC-10 | Collision | costmap.query(pos) >= LETHAL_COST | Stop run immediately | failure_reason = "collision" |
| EC-11 | Obstacle spawns on robot | spawn pos overlaps robot footprint | Offset spawn by 0.5m | logged as warning |
| EC-12 | MPPI division by zero | sum(weights) == 0 | catch ZeroDivisionError, use uniform weights | logged as warning |
| EC-13 | RPP lookahead off-path | No waypoint found within MAX_LOOKAHEAD | Return to nearest path point | logged as warning |
| EC-14 | DWA all samples discard | No (v,d) survives collision check | Return (0, 0) — full stop | failure_reason = "dwa_no_valid" |

---

## ═══════════════════════════════════════════
## PHASE K — Final Deliverables
## ═══════════════════════════════════════════

### K1 — Results files

```
results/
├── all_results.csv              # 150 rows: 3 controllers × 5 scenarios × 10 runs
│                                 # + edge case runs appended
├── summary_table.csv            # Mean ± std per controller-scenario combo
├── raw/                         # 150 timestep-level CSVs (one per run)
│   └── MPPI_s1_static_box_run00.csv ... (150 files)
├── edge_case_results/           # Edge case run summaries
│   └── all_edge_results.csv
└── plots/
    ├── plot1_path_comparison.png
    ├── plot2_min_distance.png
    ├── plot3_completion_time.png
    ├── plot4_accel_jerk.png
    ├── plot5_failure_heatmap.png
    ├── plot6_reaction_time.png    (NEW)
    ├── plot7_path_smoothness.png  (NEW)
    └── plot8_edge_case_summary.png (NEW)
```

### K2 — Final summary table template

| Controller | Scenario | Success | Avg Time (s) | Min Dist (m) | Peak Accel | Peak Jerk | React (s) | Fails |
|---|---|---|---|---|---|---|---|---|
| MPPI | s1 | /10 | | | | | — | |
| DWA  | s1 | /10 | | | | | — | |
| RPP  | s1 | /10 | | | | | — | |
| MPPI | s5 | /10 | | | | | ← key | |
| DWA  | s5 | /10 | | | | | ← key | |
| RPP  | s5 | /10 | | | | | ← key | |

---

## ═══════════════════════════════════════════
## AGENT EXECUTION ORDER
## (Exact sequence to avoid import errors)
## ═══════════════════════════════════════════

```
── PHASE A: Foundation Fixes ──────────────────────────
Step A1:  config.py                    [MODIFY — add new params + validate()]
Step A2:  robot/bicycle_model.py       [MODIFY — jerk, footprint, reset, Ackermann]

── PHASE B: Environment Layer ─────────────────────────
Step B1:  environment/__init__.py      [ADD NEW — empty]
Step B2:  environment/obstacle.py      [ADD NEW — full Obstacle class]
Step B3:  environment/costmap.py       [ADD NEW — full Costmap2D class]

── PHASE C: Controller Upgrades ───────────────────────
Step C0:  controllers/controller_base.py  [ADD NEW — base class]
Step C1:  controllers/rpp.py              [MODIFY — adaptive lookahead, filter, costmap]
Step C2:  controllers/dwa.py              [MODIFY — Ackermann fix, new window]
Step C3:  controllers/mppi.py             [MODIFY — 5 critics, vectorized, warm-start]

── PHASE D: Scenario Upgrades ─────────────────────────
Step D0:  scenarios/scenario_base.py   [MODIFY — seed, costmap, is_complete]
Step D1:  scenarios/s1_static_box.py   [MODIFY — Obstacle class, costmap]
Step D2:  scenarios/s2_person_crossing.py [MODIFY — frozen, reverse edge cases]
Step D3:  scenarios/s3_narrow_gap.py   [MODIFY — gap variants, centerline metric]
Step D4:  scenarios/s4_two_obstacles.py [MODIFY — dynamic second obstacle]
Step D5:  scenarios/s5_sudden_obstacle.py [MODIFY — reaction time measurement]
Step D6a: scenarios/edge_cases/__init__.py      [ADD NEW]
Step D6b: scenarios/edge_cases/ec1_dead_end.py [ADD NEW]
Step D6c: scenarios/edge_cases/ec2_oscillation_trigger.py [ADD NEW]
Step D6d: scenarios/edge_cases/ec3_tight_turn.py [ADD NEW]
Step D6e: scenarios/edge_cases/ec4_late_braking.py [ADD NEW]
Step D6f: scenarios/edge_cases/ec5_crowd.py [ADD NEW]

── PHASE E: Health Monitor ────────────────────────────
Step E1:  health/__init__.py           [ADD NEW — empty]
Step E2:  health/monitor.py            [ADD NEW — HealthMonitor class]

── PHASE F: Metrics Upgrades ──────────────────────────
Step F1:  metrics/recorder.py          [MODIFY — timestep log, reaction time, new fields]
Step F2:  metrics/report_generator.py  [ADD NEW — 8 plots + summary table]

── PHASE G: Main Runner ───────────────────────────────
Step G1:  main.py                      [MODIFY — seed, costmap loop, health, edge cases, flags]

── PHASE H: Visualization ─────────────────────────────
Step H1:  visualization/pygame_viz.py  [MODIFY — MPPI samples, costmap overlay, HUD]
Step H2:  visualization/plot_results.py [MODIFY — add plots 6, 7, 8]

── PHASE I: Tests ─────────────────────────────────────
Step I1:  tests/__init__.py            [ADD NEW — empty]
Step I2:  tests/test_bicycle_model.py  [ADD NEW]
Step I3:  tests/test_costmap.py        [ADD NEW]
Step I4:  tests/test_mppi.py           [ADD NEW]
Step I5:  tests/test_dwa.py            [ADD NEW]
Step I6:  tests/test_rpp.py            [ADD NEW]

── PHASE K: Run & Verify ──────────────────────────────
Step K1:  python -m pytest tests/ -v
          (All tests must pass before running experiments)

Step K2:  python main.py --headless
          (Full 150-run experiment, no pygame window, fastest)

Step K3:  python -m metrics.report_generator
          (Generate all 8 plots + summary_table.csv)

Step K4:  python main.py --edge-only --headless
          (Run 5 edge case scenarios, 10 runs each)

Step K5:  Verify results/all_results.csv has 150 rows
          Verify results/plots/ has 8 PNG files
          Verify results/edge_case_results/all_edge_results.csv exists
```

---

*Upgrade plan version: 2.0 | Built on top of base plan v1.0 | October 2026*
*Every step modifies or extends the existing codebase — zero rebuild from scratch.*
