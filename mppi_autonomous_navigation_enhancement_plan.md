# 🚗 MPPI-Based Autonomous Navigation: Enhancement Plan
## Autonomous Campus Buggy — MPPI as Sole Controller
### Kishore | 3rd Year | October 2026

---

> **How to read this document:**
> This plan upgrades the existing codebase (which has RPP + DWA + basic MPPI already built)
> into an MPPI-only, production-grade navigation system with Indian left-hand traffic rules.
>
> Tags used throughout:
> - 🗑️ DELETE — remove this file or block entirely
> - 🔧 MODIFY — change specific parts of an existing file
> - ➕ ADD NEW — create a file that does not yet exist
> - ✅ KEEP — no changes needed
> - ⚠️ VERIFIED — confirmed from real papers/repos
> - 📌 ASSUMPTION — inferred from context, not directly confirmed

---

## 📚 Section 1 — MPPI Research Foundation

### 1.1 Mathematical Formulation (Verified from Williams et al. 2017)

MPPI is a sampling-based stochastic optimal control algorithm derived from
information-theoretic principles (KL divergence and free energy duality).

**Core problem:**
Minimize the expected cost of a trajectory over a finite horizon T:

```
J(U) = E[ phi(x_T) + sum_{t=0}^{T-1} L(x_t, u_t) ]

where:
  phi(x_T)      = terminal cost at final state
  L(x_t, u_t)  = running cost at each step
  U             = control sequence [u_0, u_1, ..., u_{T-1}]
```

**Information-theoretic weight:**
```
w(V_k) = exp( -(1/lambda) * S(V_k) )   normalized across K samples

where:
  S(V_k)  = total trajectory cost for sample k
  lambda  = temperature parameter
             low lambda  → greedy (picks cheapest trajectory)
             high lambda → explores more (flatter distribution)
```

**Control update rule:**
```
U* = U_nominal + sum_k [ w(V_k) * epsilon_k ] / sum_k [ w(V_k) ]

where:
  epsilon_k ~ N(0, Sigma)   — Gaussian noise perturbation
  V_k = U_nominal + epsilon_k
```

**Key insight:** MPPI does NOT require the cost function to be differentiable.
This makes it ideal for discontinuous costs like collision detection,
lane boundary violations, and traffic rule penalties.

### 1.2 Bicycle Model Dynamics (Ackermann)

State: x = [x_pos, y_pos, theta, v]
Control: u = [delta (steering), a (acceleration)]

```
x_dot     = v * cos(theta)
y_dot     = v * sin(theta)
theta_dot = (v / L) * tan(delta)
v_dot     = a

Constraints:
  |delta| <= MAX_STEER    (0.6 rad)
  0 <= v <= MAX_SPEED     (1.5 m/s)
  |a| <= MAX_ACCEL        (1.0 m/s^2)
  turning_radius = L / tan(|delta|) >= R_min
```

### 1.3 MPPI Algorithm Loop (Per Timestep)

```
STEP 1 — SAMPLE
  Draw K noise sequences: epsilon_k ~ N(0, Sigma), shape (K, T, 2)
  Sigma diagonal: [sigma_steer^2, sigma_accel^2]

STEP 2 — PERTURB
  V_k = U_nominal + epsilon_k
  Clip V_k to control bounds

STEP 3 — ROLLOUT (vectorized)
  For each k in K, simulate T steps using bicycle model
  Result: trajectories, shape (K, T, 4)

STEP 4 — SCORE
  For each k, compute S(k) = sum of all critic costs over T steps

STEP 5 — WEIGHT
  beta = min(S)                        ← numerical stability offset
  w_k = exp( -(S_k - beta) / lambda )
  w_k = w_k / sum(w_k)                 ← normalize
  Guard: if sum(w_k) < 1e-10 → use uniform weights (all-collide fallback)

STEP 6 — UPDATE
  For t in T: U_nominal[t] += sum_k( w_k * epsilon_k[t] )
  Clip U_nominal to control bounds

STEP 7 — EXECUTE
  Apply U_nominal[0] to vehicle model
  Record state

STEP 8 — WARM-START (SHIFT)
  U_nominal[0:T-1] = U_nominal[1:T]
  U_nominal[T-1]   = 0.0
  (New timestep starts with previous plan shifted forward)
```

### 1.4 Known MPPI Limitations (Verified from Literature)

| Limitation | Source | Mitigation in this system |
|---|---|---|
| **Chattering** — Gaussian sampling at every step introduces high-frequency noise in control commands | Kim et al. (2022) arXiv:2112.09988 | Exponential moving average (EMA) filter on U_nominal output |
| **Infeasible rollouts** — If nominal trajectory enters obstacle, all K samples may be lethal; MPPI can diverge | o-MPPI arXiv:2309.13201 | Fallback to zero control; increase sigma temporarily for recovery |
| **Lambda sensitivity** — Wrong lambda → too greedy (stuck) or too random (wanders) | Williams et al. 2017 | Adaptive lambda: increase near obstacles, decrease on clear path |
| **Sample starvation** — Low K makes the weighted average noisy at tight corridors | Vlahov et al. 2024 arXiv:2404.03094 | Minimum K=300 enforced; narrow gap scenario uses K=600 |
| **Computational cost** — K*T rollouts per timestep scales O(K*T) | Nav2 MPPI docs | Numpy vectorized rollout — no Python loops over K |
| **Horizon too short** — T steps not enough to see past obstacle → controller walks into wall | Shield-MPPI arXiv:2302.11719 | Adaptive horizon: increase T near detected obstacles |
| **High-frequency noise** propagates through warm-start | Low Freq Sampling paper 2024 | Colored noise option (apply low-pass to epsilon before use) |

### 1.5 GitHub Reference Implementations

| Repository | What it provides | Used for |
|---|---|---|
| `MizuhoAOKI/python_simple_mppi` | Pure numpy MPPI path tracking + obstacle avoidance, no ROS | Reference for vectorized rollout, cost structure |
| `UM-ARM-Lab/pytorch_mppi` | GPU-accelerated MPPI with Tube-MPPI and RMPPI variants | Reference for adaptive covariance, fallback logic |
| `Shunichi09/PythonLinearNonlinearControl` | MPPIWilliams and MPPINagabandi in pure Python with scipy | Reference for filter_coefs (beta_0, beta_1, beta_2) smoothing |
| `autompc` (readthedocs) | MPPICopy class with num_path, H, sigma, lmda params | Reference for parameter naming and seeding |
| `Nav2 nav2_mppi_controller` | Production MPPI at 50+ Hz on i5; Ackermann support; critic plugins | Reference for critic architecture and Ackermann config |
| `360lab-public/mppi` (Uni Luxembourg) | MPPI for real AV with circular obstacle approximation | Reference for obstacle cost formulation |

---

## 📚 Section 2 — Indian Traffic Rules for Left-Hand Navigation

### 2.1 Lane Discipline Rules (Verified from Indian Motor Vehicles Act + GeoDrive-Bench)

```
RULE 1 — KEEP LEFT BY DEFAULT
  The buggy must stay in the left half of the road at all times.
  Reference centerline at y = MAP_H/2.
  Buggy target lane center = MAP_H/2 - LANE_WIDTH/2 (left lane center).

RULE 2 — OVERTAKE ON RIGHT ONLY
  If buggy must cross centerline (obstacle on left lane),
  it may temporarily move right, then return left.
  Right-lane cost is non-zero but not lethal (soft penalty).

RULE 3 — LEFT TURN: STAY LEFT
  For a left turn, move to leftmost position before the turn point.
  No wide-swing into right lane.

RULE 4 — RIGHT TURN: MOVE RIGHT, THEN RETURN LEFT
  Before right turn, cross to right lane.
  After turn is complete, return to left lane.

RULE 5 — U-TURN
  Move to rightmost position available.
  Execute U-turn arc (minimum turning radius enforced).
  Return to left lane after completing 180-degree heading change.

RULE 6 — PEDESTRIAN CROSSING
  Yield to pedestrian crossing from left.
  Pedestrian on left side has priority.
```

### 2.2 Road Layout for Simulation

```
MAP (20m x 20m world, road runs along x-axis):

  y=20 ┌──────────────────────────────┐
       │  ← PAVEMENT / BOUNDARY       │
  y=15 ├──────────────────────────────┤  ← right boundary wall
       │                              │
  y=12 │    RIGHT LANE (y=12–15)      │  ← right lane (buggy avoids)
       │                              │
  y=10 ├ ── ── ── ── ── ── ── ── ── ─┤  ← road centerline (y=10)
       │                              │
  y=7  │    LEFT LANE  (y=7–10)       │  ← buggy default lane
       │                              │
  y=5  ├──────────────────────────────┤  ← left boundary wall
       │  ← PAVEMENT / BOUNDARY       │
  y=0  └──────────────────────────────┘

  LEFT LANE CENTER:  y = 8.5   (target for MPPI path cost)
  RIGHT LANE CENTER: y = 12.5  (temporary overtake only)
```

---

## 📊 Section 3 — Gap Analysis: Existing vs Required

### 3.1 What exists (from uploaded plan v2.0)

| File | Current State |
|---|---|
| controllers/rpp.py | ✅ Implemented (to be deleted) |
| controllers/dwa.py | ✅ Implemented (to be deleted) |
| controllers/mppi.py | ✅ Basic — needs major upgrade |
| controllers/controller_base.py | ✅ Exists |
| environment/obstacle.py | ✅ Exists |
| environment/costmap.py | ✅ Exists |
| robot/bicycle_model.py | ✅ Upgraded (jerk, footprint, Ackermann) |
| scenarios/ (s1–s5) | ✅ Upgraded with seed + costmap |
| scenarios/edge_cases/ (ec1–ec5) | ✅ Exists |
| health/monitor.py | ✅ Exists |
| metrics/recorder.py | ✅ Upgraded |
| visualization/pygame_viz.py | ✅ Upgraded |
| config.py | ✅ Upgraded |
| tests/ (test_dwa, test_rpp) | ✅ Exist but target deleted code |

### 3.2 What is MISSING for MPPI-only system

| Missing | Why needed |
|---|---|
| Lane cost critic | MPPI has no awareness of left-hand lane discipline |
| Right-lane penalty critic | No cost for wrong-lane entry |
| Maneuver mode state machine | No logic to handle turn / U-turn / reverse modes |
| Adaptive lambda | Fixed lambda fails near tight obstacles |
| Adaptive horizon T | Short horizon misses obstacles far ahead |
| Colored noise / EMA filter | Chattering not suppressed |
| MPPI-specific edge case handlers | All-collide fallback, sigma expansion, recovery |
| Updated tests (remove DWA/RPP) | test_dwa.py and test_rpp.py are now dead |
| Lane boundary cost in costmap | Costmap only has physical obstacles, not lane boundaries |
| Maneuver scenarios | No test for U-turn, reverse, right turn, lane return |
| Updated config (remove DWA/RPP params) | Dead parameters create confusion |
| Updated main.py (remove controller loop) | main.py loops over 3 controllers — wrong now |

---

## 🗂️ Section 4 — Final Folder Structure

```
campus_buggy_sim/
│
├── config.py                        🔧 MODIFY — remove RPP/DWA params, add MPPI/lane params
├── main.py                          🔧 MODIFY — single controller, maneuver modes, lane flags
├── requirements.txt                 ✅ KEEP
├── README.md                        🔧 MODIFY — MPPI-only description
│
├── robot/
│   └── bicycle_model.py             ✅ KEEP (already upgraded in v2.0)
│
├── controllers/
│   ├── controller_base.py           🔧 MODIFY — remove RPP/DWA references
│   ├── mppi.py                      🔧 MODIFY — major upgrade (see Phase C)
│   ├── rpp.py                       🗑️ DELETE
│   └── dwa.py                       🗑️ DELETE
│
├── environment/
│   ├── __init__.py                  ✅ KEEP
│   ├── obstacle.py                  ✅ KEEP
│   ├── costmap.py                   🔧 MODIFY — add lane boundary layers
│   └── lane_map.py                  ➕ ADD NEW — lane geometry, boundaries, centerlines
│
├── mppi_critics/                    ➕ ADD NEW folder — one file per critic
│   ├── __init__.py
│   ├── obstacle_critic.py
│   ├── goal_critic.py
│   ├── path_critic.py
│   ├── jerk_critic.py
│   ├── speed_critic.py
│   ├── lane_critic.py               ← NEW: keeps buggy in left lane
│   └── maneuver_critic.py           ← NEW: adjusts cost during turns/U-turns
│
├── maneuvers/                       ➕ ADD NEW folder
│   ├── __init__.py
│   ├── maneuver_state_machine.py    ← controls which mode is active
│   ├── left_turn.py
│   ├── right_turn.py
│   ├── u_turn.py
│   └── reverse.py
│
├── scenarios/
│   ├── scenario_base.py             ✅ KEEP
│   ├── s1_static_box.py             ✅ KEEP
│   ├── s2_person_crossing.py        ✅ KEEP
│   ├── s3_narrow_gap.py             ✅ KEEP
│   ├── s4_two_obstacles.py          ✅ KEEP
│   ├── s5_sudden_obstacle.py        ✅ KEEP
│   └── edge_cases/
│       ├── ec1_dead_end.py          ✅ KEEP
│       ├── ec2_oscillation_trigger.py  🗑️ DELETE (RPP-specific test)
│       ├── ec3_tight_turn.py        ✅ KEEP
│       ├── ec4_late_braking.py      ✅ KEEP
│       ├── ec5_crowd.py             ✅ KEEP
│       ├── ec6_wrong_lane_entry.py  ➕ ADD NEW
│       ├── ec7_u_turn_blocked.py    ➕ ADD NEW
│       ├── ec8_missing_lane_mark.py ➕ ADD NEW
│       ├── ec9_sensor_noise.py      ➕ ADD NEW
│       └── ec10_all_samples_collide.py ➕ ADD NEW
│
├── health/
│   └── monitor.py                   🔧 MODIFY — remove RPP oscillation check, add MPPI-specific checks
│
├── metrics/
│   ├── recorder.py                  🔧 MODIFY — add lane_violation_count, maneuver_success
│   └── report_generator.py         🔧 MODIFY — remove DWA/RPP comparisons, MPPI-only plots
│
├── visualization/
│   ├── pygame_viz.py                🔧 MODIFY — add lane overlay, maneuver state display
│   └── plot_results.py             🔧 MODIFY — MPPI-only result plots
│
├── tests/
│   ├── test_bicycle_model.py        ✅ KEEP
│   ├── test_mppi.py                 🔧 MODIFY — add critic tests, lane tests, maneuver tests
│   ├── test_costmap.py              ✅ KEEP
│   ├── test_dwa.py                  🗑️ DELETE
│   ├── test_rpp.py                  🗑️ DELETE
│   └── test_lane_map.py             ➕ ADD NEW
│
└── results/
    ├── raw/                         ✅ KEEP
    ├── plots/                       ✅ KEEP
    └── edge_case_results/           ✅ KEEP
```

---

## ═══════════════════════════════════════════════════
## PHASE 0 — Safe Removal of RPP and DWA
## ═══════════════════════════════════════════════════

**Do this first. Everything else depends on a clean codebase.**

### 0.1 — 🗑️ DELETE `controllers/rpp.py`

Safe to delete because:
- No other file imports from it after this phase
- All RPP-specific config params will be removed from config.py in Phase 1
- test_rpp.py is also deleted in this phase

**Before deleting:** Search entire codebase for `from controllers.rpp` and `import rpp`.
Expected locations: `main.py`, `controllers/__init__.py`.
Remove those import lines FIRST, then delete the file.

### 0.2 — 🗑️ DELETE `controllers/dwa.py`

Same procedure as rpp.py.
Search for `from controllers.dwa` and `import dwa` and remove those lines first.

### 0.3 — 🗑️ DELETE `tests/test_dwa.py` and `tests/test_rpp.py`

These test files target deleted code. Delete them.

### 0.4 — 🗑️ DELETE `scenarios/edge_cases/ec2_oscillation_trigger.py`

This scenario was designed to expose RPP oscillation.
Not relevant for MPPI-only system.
Remove its import from `scenarios/edge_cases/__init__.py`.

### 0.5 — 🔧 MODIFY `controllers/__init__.py`

Remove:
```python
# DELETE these lines:
from .rpp import RegulatedPurePursuit
from .dwa import DWA
```

Keep:
```python
from .mppi import MPPI
from .controller_base import ControllerBase
```

### 0.6 — Verify removal

After all deletions, run:
```bash
python -c "from controllers.mppi import MPPI; print('OK')"
```
This must succeed. Any ImportError means a dangling reference was missed.

---

## ═══════════════════════════════════════════════════
## PHASE 1 — Config Cleanup and MPPI Extension
## ═══════════════════════════════════════════════════

### 1.1 — 🔧 MODIFY `config.py`

**Remove (delete these blocks entirely):**
```python
# DELETE — RPP params
RPP_LOOKAHEAD       = ...
RPP_LOOKAHEAD_GAIN  = ...
RPP_MIN_LOOKAHEAD   = ...
RPP_MAX_LOOKAHEAD   = ...
RPP_STEER_LOWPASS   = ...
RPP_MAX_SPEED       = ...
RPP_MIN_SPEED       = ...
RPP_OBSTACLE_SLOWDOWN = ...
RPP_CURVATURE_SLOWDOWN = ...

# DELETE — DWA params
DWA_V_SAMPLES       = ...
DWA_W_SAMPLES       = ...
DWA_STEER_SAMPLES   = ...
DWA_STEER_RATE      = ...
DWA_SIM_TIME        = ...
DWA_HEADING_W       = ...
DWA_CLEARANCE_W     = ...
DWA_SPEED_W         = ...
DWA_MIN_TURN_RADIUS = ...
```

**Keep (all existing MPPI params — already good):**
```python
MPPI_K        = 500
MPPI_T        = 20
MPPI_SIGMA    = 0.5
MPPI_LAMBDA   = 1.0
MPPI_OBS_COST = 1e6
MPPI_PATH_COST = 1.0
MPPI_CTRL_COST = 0.1
# + all critic weights added in v2.0
```

**Add (new MPPI-only and lane params):**
```python
# ── MPPI Advanced Controls ───────────────────────────────
MPPI_MIN_K          = 300       # minimum samples (narrow gap uses 600)
MPPI_NARROW_K       = 600       # samples for narrow-gap scenarios
MPPI_SIGMA_STEER    = 0.3       # steering noise std (separate from accel)
MPPI_SIGMA_ACCEL    = 0.2       # acceleration noise std
MPPI_EMA_ALPHA      = 0.85      # EMA filter for output smoothing (0=heavy, 1=off)
MPPI_ITER           = 1         # iterations per timestep (increase for complex env)

# ── Adaptive MPPI ────────────────────────────────────────
MPPI_LAMBDA_NEAR_OBS   = 0.5    # lower lambda when in high-cost zone (greedier)
MPPI_LAMBDA_FREE       = 1.5    # higher lambda on clear path (more exploratory)
MPPI_LAMBDA_ADAPT_COST = 40.0   # costmap value above which lambda switches
MPPI_HORIZON_NEAR_OBS  = 30     # longer horizon near obstacle
MPPI_HORIZON_DEFAULT   = 20     # default horizon

# ── Lane geometry (Indian left-hand traffic) ─────────────
ROAD_Y_MIN          = 5.0       # road lower boundary (left kerb)
ROAD_Y_MAX          = 15.0      # road upper boundary (right kerb)
ROAD_CENTERLINE_Y   = 10.0      # road centre
LANE_WIDTH          = 2.5       # metres per lane
LEFT_LANE_CENTER_Y  = 8.5       # target y for left lane (MPPI follows this)
RIGHT_LANE_CENTER_Y = 12.5      # right lane centre (temporary overtake only)
LANE_COST_LEFT      = 0.0       # cost for being in left lane (reward = zero cost)
LANE_COST_RIGHT     = 30.0      # soft penalty for being in right lane
LANE_COST_BOUNDARY  = 1e6       # lethal cost for crossing road boundary

# ── Maneuver modes ───────────────────────────────────────
MANEUVER_LEFT_TURN_RADIUS   = 1.5   # metres
MANEUVER_RIGHT_TURN_RADIUS  = 2.5   # metres
MANEUVER_UTURN_RADIUS       = 3.0   # metres
MANEUVER_REVERSE_MAX_SPEED  = 0.5   # m/s
MANEUVER_GOAL_ARRIVAL_DIST  = 0.5   # metres — when to declare goal reached

# ── MPPI Recovery ────────────────────────────────────────
MPPI_RECOVERY_SIGMA_MULT    = 2.0   # multiply sigma when all-collide detected
MPPI_RECOVERY_MAX_ATTEMPTS  = 5     # timesteps to try before emergency stop

# ── Remove oscillation threshold (RPP-only) ──────────────
# DELETE: OSCILLATION_THRESH and OSCILLATION_WINDOW

# ── Update validate() to remove RPP/DWA assertions ───────
def validate():
    assert MAX_SPEED > 0
    assert DT > 0 and DT < 0.5
    assert MPPI_K >= MPPI_MIN_K, "K must be >= minimum sample count"
    assert MPPI_LAMBDA > 0
    assert INFLATION_RADIUS >= ROBOT_RADIUS
    assert ROAD_Y_MIN < LEFT_LANE_CENTER_Y < ROAD_CENTERLINE_Y
    assert ROAD_CENTERLINE_Y < RIGHT_LANE_CENTER_Y < ROAD_Y_MAX
    print("[config] All parameters valid — MPPI-only mode")
```

---

## ═══════════════════════════════════════════════════
## PHASE 2 — Lane Map (New Module)
## ═══════════════════════════════════════════════════

### 2.1 — ➕ ADD NEW `environment/lane_map.py`

**Purpose:** Encode road geometry and lane discipline as a queryable cost surface.
MPPI lane_critic reads from this. Costmap reads from this to add lane boundary walls.

```python
"""
LaneMap: encodes Indian left-hand road geometry.

Provides:
  - lane_cost(x, y)       : penalty cost for being at (x, y) relative to road
  - in_left_lane(x, y)    : True if y is in left lane band
  - in_right_lane(x, y)   : True if y is in right lane band
  - boundary_cost(x, y)   : lethal if outside road boundaries
  - get_lane_center_y(mode): returns target y for given maneuver mode
"""

from config import (ROAD_Y_MIN, ROAD_Y_MAX, ROAD_CENTERLINE_Y,
                    LANE_WIDTH, LEFT_LANE_CENTER_Y, RIGHT_LANE_CENTER_Y,
                    LANE_COST_LEFT, LANE_COST_RIGHT, LANE_COST_BOUNDARY)

class LaneMap:

    def lane_cost(self, x, y):
        """
        Returns lane discipline cost at position (x, y).
        Used by LaneCritic inside MPPI.
        """
        # Outside road entirely → lethal
        if y <= ROAD_Y_MIN or y >= ROAD_Y_MAX:
            return LANE_COST_BOUNDARY

        # Left lane (correct lane for Indian traffic)
        if ROAD_Y_MIN < y <= ROAD_CENTERLINE_Y:
            # Quadratic penalty increasing toward road edge
            dist_from_center = abs(y - LEFT_LANE_CENTER_Y)
            return LANE_COST_LEFT + 2.0 * dist_from_center

        # Right lane (wrong lane — soft penalty)
        if ROAD_CENTERLINE_Y < y < ROAD_Y_MAX:
            dist_from_center = abs(y - RIGHT_LANE_CENTER_Y)
            return LANE_COST_RIGHT + 2.0 * dist_from_center

        return 0.0

    def boundary_cost(self, x, y):
        """Lethal if y exits road limits."""
        if y <= ROAD_Y_MIN or y >= ROAD_Y_MAX:
            return LANE_COST_BOUNDARY
        return 0.0

    def get_target_lane_y(self, maneuver_mode="normal"):
        """
        Returns the y-coordinate MPPI path_critic should target.
        Changes during maneuvers:
          "normal"       → LEFT_LANE_CENTER_Y (default)
          "overtake"     → RIGHT_LANE_CENTER_Y (temporary)
          "return_left"  → LEFT_LANE_CENTER_Y
          "right_turn"   → RIGHT_LANE_CENTER_Y (pre-turn positioning)
          "left_turn"    → LEFT_LANE_CENTER_Y
          "u_turn"       → RIGHT_LANE_CENTER_Y (widest arc position)
          "reverse"      → LEFT_LANE_CENTER_Y
        """
        mode_map = {
            "normal":       LEFT_LANE_CENTER_Y,
            "overtake":     RIGHT_LANE_CENTER_Y,
            "return_left":  LEFT_LANE_CENTER_Y,
            "right_turn":   RIGHT_LANE_CENTER_Y,
            "left_turn":    LEFT_LANE_CENTER_Y,
            "u_turn":       RIGHT_LANE_CENTER_Y,
            "reverse":      LEFT_LANE_CENTER_Y,
        }
        return mode_map.get(maneuver_mode, LEFT_LANE_CENTER_Y)

    def in_left_lane(self, x, y):
        return ROAD_Y_MIN < y <= ROAD_CENTERLINE_Y

    def in_right_lane(self, x, y):
        return ROAD_CENTERLINE_Y < y < ROAD_Y_MAX

    def wrong_lane_violation(self, x, y, maneuver_mode="normal"):
        """
        Returns True if buggy is in wrong lane for current maneuver.
        Used by metrics recorder to count violations.
        """
        if maneuver_mode in ("normal", "left_turn", "return_left", "reverse"):
            return self.in_right_lane(x, y)
        return False  # overtake/right_turn/u_turn allows right lane temporarily
```

### 2.2 — 🔧 MODIFY `environment/costmap.py`

Add lane boundary walls as static obstacles at reset:

```python
# ADD to __init__ or to a new method: add_lane_boundaries()
def add_lane_boundaries(self):
    """
    Stamp road boundaries (y < ROAD_Y_MIN and y > ROAD_Y_MAX) as lethal.
    These are static — added once at scenario setup.
    Also stamp road centerline as a soft penalty layer (NOT lethal).
    """
    from config import ROAD_Y_MIN, ROAD_Y_MAX, LETHAL_COST, MAP_W
    for c in range(self.cols):
        wx, _ = self.cell_to_world(0, c)
        # Left boundary (below road)
        for row in range(self.rows):
            _, wy = self.cell_to_world(row, c)
            if wy <= ROAD_Y_MIN or wy >= ROAD_Y_MAX:
                self.static_layer[row, c] = LETHAL_COST
    self._inflate()
```

---

## ═══════════════════════════════════════════════════
## PHASE 3 — MPPI Critics Architecture
## ═══════════════════════════════════════════════════

Each critic is a separate file in `mppi_critics/`.
MPPI calls all critics and sums their outputs.
This makes it easy to add, remove or tune individual critics.

### 3.1 — ➕ ADD NEW `mppi_critics/__init__.py`

```python
from .obstacle_critic  import ObstacleCritic
from .goal_critic      import GoalCritic
from .path_critic      import PathCritic
from .jerk_critic      import JerkCritic
from .speed_critic     import SpeedCritic
from .lane_critic      import LaneCritic
from .maneuver_critic  import ManeuverCritic
```

### 3.2 — ➕ ADD NEW `mppi_critics/obstacle_critic.py`

**Purpose:** Penalize trajectories that pass through or near obstacles.
Highest-priority critic — lethal cost stops the vehicle.

```python
class ObstacleCritic:
    """
    Queries costmap at each trajectory point.
    Points inside obstacle → MPPI_OBS_COST (makes sample weight near zero).
    Points near obstacle → proportional penalty.
    """
    name = "obstacle"

    def score(self, trajectories, costmap):
        """
        trajectories : shape (K, T, 4)
        Returns      : shape (K,) cost contribution
        """
        K, T, _ = trajectories.shape
        costs = np.zeros(K)
        for t in range(T):
            for k in range(K):
                x, y = trajectories[k, t, 0], trajectories[k, t, 1]
                cell_cost = costmap.query(x, y)
                if cell_cost >= LETHAL_COST:
                    costs[k] += MPPI_OBS_COST
                else:
                    costs[k] += MPPI_OBSTACLE_WEIGHT * cell_cost
        return costs
```

### 3.3 — ➕ ADD NEW `mppi_critics/goal_critic.py`

**Purpose:** Drive buggy toward goal. Decreases as distance to goal decreases.

```python
class GoalCritic:
    """
    Cost = distance from final trajectory state to goal.
    Also adds smaller cost at intermediate steps to keep moving toward goal.
    """
    name = "goal"

    def score(self, trajectories, goal):
        gx, gy = goal
        K, T, _ = trajectories.shape
        costs = np.zeros(K)
        # Terminal cost (heaviest weight)
        costs += MPPI_GOAL_WEIGHT * np.hypot(
            trajectories[:, -1, 0] - gx,
            trajectories[:, -1, 1] - gy
        )
        # Intermediate progress cost (lighter)
        for t in range(T):
            costs += 0.1 * np.hypot(
                trajectories[:, t, 0] - gx,
                trajectories[:, t, 1] - gy
            )
        return costs
```

### 3.4 — ➕ ADD NEW `mppi_critics/path_critic.py`

**Purpose:** Keep buggy near the reference path. Cross-track error penalized.

```python
class PathCritic:
    """
    Cost = minimum distance from each trajectory state to reference path.
    Uses vectorized nearest-neighbor on path array.
    """
    name = "path"

    def score(self, trajectories, reference_path, lane_map, maneuver_mode):
        if not reference_path:
            return np.zeros(trajectories.shape[0])

        path_arr = np.array(reference_path)          # shape (N, 2)
        target_y = lane_map.get_target_lane_y(maneuver_mode)

        K, T, _ = trajectories.shape
        costs = np.zeros(K)

        for t in range(T):
            xs = trajectories[:, t, 0]               # shape (K,)
            ys = trajectories[:, t, 1]

            # Cross-track error to path
            dists = np.hypot(
                xs[:, None] - path_arr[:, 0],
                ys[:, None] - path_arr[:, 1]
            )                                         # shape (K, N)
            cte = np.min(dists, axis=1)               # shape (K,)
            costs += MPPI_PATH_WEIGHT * cte

            # Lane centering cost (lateral deviation from target y)
            lane_dev = np.abs(ys - target_y)
            costs += 0.5 * lane_dev

        return costs
```

### 3.5 — ➕ ADD NEW `mppi_critics/jerk_critic.py`

**Purpose:** Penalize high jerk. Ensures passenger comfort.
Jerk = rate of change of acceleration.

```python
class JerkCritic:
    """
    Computes jerk from trajectory velocity sequence.
    Cost = sum of squared jerks across horizon.
    """
    name = "jerk"

    def score(self, trajectories):
        K, T, _ = trajectories.shape
        if T < 3:
            return np.zeros(K)

        vs     = trajectories[:, :, 3]               # shape (K, T)
        accels = np.diff(vs, axis=1) / DT            # shape (K, T-1)
        jerks  = np.diff(accels, axis=1) / DT        # shape (K, T-2)
        return MPPI_JERK_WEIGHT * np.sum(jerks ** 2, axis=1)
```

### 3.6 — ➕ ADD NEW `mppi_critics/speed_critic.py`

**Purpose:** Prevent unnecessary stopping. Buggy should maintain reasonable speed.

```python
class SpeedCritic:
    """
    Penalizes speed far from desired speed.
    Near obstacles: desired speed is lower.
    Clear path: desired speed is MAX_SPEED.
    """
    name = "speed"

    def score(self, trajectories, costmap):
        K, T, _ = trajectories.shape
        costs = np.zeros(K)
        for t in range(T):
            vs = trajectories[:, t, 3]
            # Desired speed: reduce near obstacles
            for k in range(K):
                x, y = trajectories[k, t, 0], trajectories[k, t, 1]
                c = costmap.query(x, y)
                v_desired = MAX_SPEED * max(0.2, 1.0 - c / 100.0)
                costs[k] += MPPI_SPEED_WEIGHT * (v_desired - vs[k]) ** 2
        return costs
```

### 3.7 — ➕ ADD NEW `mppi_critics/lane_critic.py`

**Purpose:** Enforce Indian left-hand lane discipline.
This is the key new critic not in the existing plan.

```python
class LaneCritic:
    """
    Applies lane discipline cost at every trajectory point.
    Uses LaneMap.lane_cost(x, y).

    Normal mode:    left lane = zero cost, right lane = LANE_COST_RIGHT per step
    Overtake mode:  right lane temporarily allowed (cost = 0 for crossing)
    U-turn mode:    rightmost position preferred before arc
    """
    name = "lane"

    def __init__(self, lane_map):
        self.lane_map = lane_map

    def score(self, trajectories, maneuver_mode="normal"):
        K, T, _ = trajectories.shape
        costs = np.zeros(K)

        for t in range(T):
            for k in range(K):
                x, y = trajectories[k, t, 0], trajectories[k, t, 1]

                if maneuver_mode in ("normal", "left_turn",
                                     "return_left", "reverse"):
                    # Full lane cost: left lane = free, right = penalty
                    costs[k] += self.lane_map.lane_cost(x, y)

                elif maneuver_mode in ("right_turn", "u_turn"):
                    # Only penalize going too far right (past road edge)
                    costs[k] += self.lane_map.boundary_cost(x, y)

                elif maneuver_mode == "overtake":
                    # In right lane: penalize going outside road
                    costs[k] += self.lane_map.boundary_cost(x, y)
                    # Add small cost to return left as soon as obstacle passed
                    costs[k] += 5.0 * max(0.0, y - ROAD_CENTERLINE_Y)

        return costs
```

### 3.8 — ➕ ADD NEW `mppi_critics/maneuver_critic.py`

**Purpose:** Guide MPPI during specific maneuvers (turns, U-turn, reverse).
Replaces goal direction with maneuver-specific target waypoint.

```python
class ManeuverCritic:
    """
    Active only during maneuver modes.
    Adds cost for deviation from the maneuver's intermediate waypoints.

    Maneuver waypoints are pre-calculated by the ManeuverStateMachine
    and passed to MPPI each step as maneuver_target (x, y, theta).
    """
    name = "maneuver"

    def score(self, trajectories, maneuver_target, maneuver_mode):
        if maneuver_target is None or maneuver_mode == "normal":
            return np.zeros(trajectories.shape[0])

        tx, ty, t_theta = maneuver_target
        K, T, _ = trajectories.shape
        costs = np.zeros(K)

        # Position cost to maneuver target
        costs += 3.0 * np.hypot(
            trajectories[:, -1, 0] - tx,
            trajectories[:, -1, 1] - ty
        )

        # Heading cost to target heading (for U-turn completion check)
        if maneuver_mode == "u_turn":
            final_theta = trajectories[:, -1, 2]
            heading_err = np.abs(
                np.arctan2(np.sin(final_theta - t_theta),
                           np.cos(final_theta - t_theta))
            )
            costs += 2.0 * heading_err

        return costs
```

---

## ═══════════════════════════════════════════════════
## PHASE 4 — MPPI Controller Core Upgrade
## ═══════════════════════════════════════════════════

### 4.1 — 🔧 MODIFY `controllers/mppi.py`

This is the central upgrade. Replace the existing compute_control with the
full production-grade version below.

**Complete new MPPI class structure:**

```python
import numpy as np
from config import *
from mppi_critics import (ObstacleCritic, GoalCritic, PathCritic,
                           JerkCritic, SpeedCritic, LaneCritic, ManeuverCritic)
from environment.lane_map import LaneMap
from controllers.controller_base import ControllerBase

class MPPI(ControllerBase):
    name = "MPPI"

    def __init__(self):
        self.U = np.zeros((MPPI_T, 2))       # nominal control sequence [steer, accel]
        self.lane_map    = LaneMap()
        self.critics     = {
            "obstacle":  ObstacleCritic(),
            "goal":      GoalCritic(),
            "path":      PathCritic(),
            "jerk":      JerkCritic(),
            "speed":     SpeedCritic(),
            "lane":      LaneCritic(self.lane_map),
            "maneuver":  ManeuverCritic(),
        }
        self._ema_u      = np.zeros(2)        # EMA filter state
        self._recovery_count = 0
        self._current_sigma  = np.array([MPPI_SIGMA_STEER, MPPI_SIGMA_ACCEL])
        self.maneuver_mode   = "normal"
        self.maneuver_target = None

    # ── Adaptive lambda ─────────────────────────────────
    def _get_lambda(self, state, costmap):
        """
        Lower lambda near obstacles (greedy — commit to safe trajectory).
        Higher lambda on clear path (exploratory — find best route).
        """
        cost_here = costmap.query(state[0], state[1])
        if cost_here > MPPI_LAMBDA_ADAPT_COST:
            return MPPI_LAMBDA_NEAR_OBS
        return MPPI_LAMBDA_FREE

    # ── Adaptive horizon ────────────────────────────────
    def _get_horizon(self, state, costmap):
        """
        Longer horizon near obstacles so MPPI can see past them.
        """
        cost_here = costmap.query(state[0], state[1])
        if cost_here > MPPI_LAMBDA_ADAPT_COST:
            return MPPI_HORIZON_NEAR_OBS
        return MPPI_HORIZON_DEFAULT

    # ── Vectorized rollout ───────────────────────────────
    def _rollout(self, state, U_noisy, T):
        """
        state   : (4,)
        U_noisy : (K, T, 2)
        Returns : (K, T, 4)
        """
        K = U_noisy.shape[0]
        states = np.zeros((K, T + 1, 4))
        states[:, 0, :] = state

        for t in range(T):
            x, y, th, v = (states[:, t, 0], states[:, t, 1],
                           states[:, t, 2], states[:, t, 3])
            delta = np.clip(U_noisy[:, t, 0], -MAX_STEER, MAX_STEER)
            a     = np.clip(U_noisy[:, t, 1], -MAX_ACCEL,  MAX_ACCEL)

            # Ackermann turning radius enforcement
            # Minimum radius: R = L / tan(|delta|) >= R_min
            # If violation, clamp delta
            nonzero = np.abs(delta) > 1e-6
            radii   = np.where(nonzero,
                               WHEELBASE / np.abs(np.tan(np.clip(np.abs(delta), 1e-6, np.pi/2))),
                               1e6)
            too_tight = nonzero & (radii < MANEUVER_LEFT_TURN_RADIUS)
            delta[too_tight] = np.sign(delta[too_tight]) * np.arctan(
                WHEELBASE / MANEUVER_LEFT_TURN_RADIUS)

            states[:, t+1, 0] = x + v * np.cos(th) * DT
            states[:, t+1, 1] = y + v * np.sin(th) * DT
            states[:, t+1, 2] = th + (v / WHEELBASE) * np.tan(delta) * DT
            states[:, t+1, 3] = np.clip(v + a * DT, 0.0, MAX_SPEED)

        return states[:, 1:, :]   # (K, T, 4)

    # ── Cost computation ─────────────────────────────────
    def _compute_costs(self, trajs, goal, path, costmap):
        costs = np.zeros(trajs.shape[0])
        costs += self.critics["obstacle"].score(trajs, costmap)
        costs += self.critics["goal"].score(trajs, goal)
        costs += self.critics["path"].score(trajs, path, self.lane_map, self.maneuver_mode)
        costs += self.critics["jerk"].score(trajs)
        costs += self.critics["speed"].score(trajs, costmap)
        costs += self.critics["lane"].score(trajs, self.maneuver_mode)
        costs += self.critics["maneuver"].score(trajs, self.maneuver_target, self.maneuver_mode)
        return costs

    # ── Main compute_control ─────────────────────────────
    def compute_control(self, state, path, goal, costmap, obstacles):
        T      = self._get_horizon(state, costmap)
        lam    = self._get_lambda(state, costmap)
        K      = MPPI_NARROW_K if self._is_narrow_context(costmap, state) else MPPI_K

        # Resize U_nominal if horizon changed
        if self.U.shape[0] != T:
            self.U = np.resize(self.U, (T, 2))

        # Sample noise
        eps           = np.random.randn(K, T, 2)
        eps[:, :, 0] *= self._current_sigma[0]
        eps[:, :, 1] *= self._current_sigma[1]

        U_noisy = np.clip(
            self.U[None, :T, :] + eps,
            [[-MAX_STEER, -MAX_ACCEL]],
            [[MAX_STEER,  MAX_ACCEL]]
        )

        # Rollout
        trajs = self._rollout(state, U_noisy, T)

        # Score
        costs = self._compute_costs(trajs, goal, path, costmap)

        # MPPI weight
        beta    = np.min(costs)
        weights = np.exp(-(costs - beta) / lam)
        w_sum   = np.sum(weights)

        # ── All-collide fallback (Edge Case EC-10) ───────
        if w_sum < 1e-10 or np.min(costs) >= MPPI_OBS_COST:
            self._recovery_count += 1
            if self._recovery_count >= MPPI_RECOVERY_MAX_ATTEMPTS:
                print("[MPPI] Emergency stop — no valid trajectory")
                self.U[:] = 0.0
                self._recovery_count = 0
                self._current_sigma *= 1.0   # reset sigma
                return 0.0, 0.0
            # Expand sigma for next attempt (more exploration)
            self._current_sigma = np.array([MPPI_SIGMA_STEER, MPPI_SIGMA_ACCEL]) * \
                                  MPPI_RECOVERY_SIGMA_MULT
            print(f"[MPPI] Recovery attempt {self._recovery_count} — widening sigma")
            return 0.0, 0.1   # creep forward to escape local trap
        else:
            self._recovery_count = 0
            self._current_sigma  = np.array([MPPI_SIGMA_STEER, MPPI_SIGMA_ACCEL])

        weights /= w_sum

        # Update nominal control
        for t in range(T):
            self.U[t] += np.sum(weights[:, None] * eps[:, t, :], axis=0)
        self.U = np.clip(self.U, [[-MAX_STEER, -MAX_ACCEL]], [[MAX_STEER, MAX_ACCEL]])

        # Extract first control
        u0    = self.U[0].copy()
        steer = u0[0]
        accel = u0[1]

        # ── EMA smoothing (reduces chattering) ───────────
        self._ema_u = MPPI_EMA_ALPHA * self._ema_u + (1 - MPPI_EMA_ALPHA) * u0
        steer = np.clip(self._ema_u[0], -MAX_STEER, MAX_STEER)
        speed = np.clip(state[3] + self._ema_u[1] * DT, 0.0, MAX_SPEED)

        # Warm-start shift
        self.U[:-1] = self.U[1:]
        self.U[-1]  = 0.0

        return steer, speed

    # ── Helper: narrow context detection ─────────────────
    def _is_narrow_context(self, costmap, state):
        """
        Returns True if lateral scan shows walls close on both sides.
        Used to increase K for narrow gap scenarios.
        """
        x, y = state[0], state[1]
        left_clear  = costmap.query(x, y - 1.0) < LETHAL_COST
        right_clear = costmap.query(x, y + 1.0) < LETHAL_COST
        return not (left_clear and right_clear)

    def reset(self):
        self.U               = np.zeros((MPPI_T, 2))
        self._ema_u          = np.zeros(2)
        self._recovery_count = 0
        self._current_sigma  = np.array([MPPI_SIGMA_STEER, MPPI_SIGMA_ACCEL])
        self.maneuver_mode   = "normal"
        self.maneuver_target = None

    def set_maneuver(self, mode, target=None):
        """Called by ManeuverStateMachine to switch mode."""
        self.maneuver_mode   = mode
        self.maneuver_target = target
```

---

## ═══════════════════════════════════════════════════
## PHASE 5 — Maneuver State Machine
## ═══════════════════════════════════════════════════

### 5.1 — ➕ ADD NEW `maneuvers/maneuver_state_machine.py`

**Purpose:** Decides which maneuver mode MPPI should use at each timestep.
Reads from the global path and current vehicle state.
Calls `mppi.set_maneuver(mode, target)` each step.

```python
class ManeuverStateMachine:
    """
    States:
      NORMAL       — following path in left lane
      PRE_LEFT     — approaching left turn, staying left
      IN_LEFT      — executing left turn arc
      PRE_RIGHT    — moving to right lane before right turn
      IN_RIGHT     — executing right turn arc
      RETURN_LEFT  — returning to left lane after right turn/overtake
      PRE_UTURN    — moving to rightmost position
      IN_UTURN     — executing 180-degree arc
      REVERSING    — moving backward
      OVERTAKING   — temporarily in right lane to pass obstacle

    Transitions are triggered by:
      - Upcoming turn waypoints on global path
      - Obstacles blocking left lane
      - Explicit command from scenario
    """

    STATES = [
        "NORMAL", "PRE_LEFT", "IN_LEFT",
        "PRE_RIGHT", "IN_RIGHT", "RETURN_LEFT",
        "PRE_UTURN", "IN_UTURN",
        "REVERSING", "OVERTAKING"
    ]

    def __init__(self, mppi_controller):
        self.mppi   = mppi_controller
        self.state  = "NORMAL"
        self.target = None            # current maneuver intermediate target

    def update(self, vehicle_state, path, costmap, scenario_command=None):
        """
        Called every timestep from main.py.
        Computes next maneuver state and sets MPPI accordingly.
        """
        x, y, theta, v = vehicle_state

        # External command overrides (from scenario)
        if scenario_command == "u_turn":
            self._enter_uturn(vehicle_state)
            return
        if scenario_command == "reverse":
            self._enter_reverse()
            return

        # Path-based detection: check upcoming waypoint for sharp turn
        upcoming_heading_change = self._detect_upcoming_turn(path, vehicle_state)

        if self.state == "NORMAL":
            if self._left_lane_blocked(vehicle_state, costmap):
                self._enter_overtake(vehicle_state)
            elif upcoming_heading_change < -0.5:    # sharp left (>30 deg)
                self.state = "PRE_LEFT"
                self.mppi.set_maneuver("left_turn",
                    self._left_turn_target(vehicle_state))
            elif upcoming_heading_change > 0.5:     # sharp right
                self.state = "PRE_RIGHT"
                self.mppi.set_maneuver("right_turn",
                    self._right_turn_target(vehicle_state))
            else:
                self.mppi.set_maneuver("normal")

        elif self.state == "OVERTAKING":
            if not self._left_lane_blocked(vehicle_state, costmap):
                self.state = "RETURN_LEFT"
                self.mppi.set_maneuver("return_left")

        elif self.state == "RETURN_LEFT":
            if y <= ROAD_CENTERLINE_Y + 0.3:        # close enough to left lane
                self.state = "NORMAL"
                self.mppi.set_maneuver("normal")

        elif self.state in ("IN_LEFT", "IN_RIGHT"):
            if self._turn_complete(vehicle_state):
                self.state = "NORMAL"
                self.mppi.set_maneuver("normal")

        elif self.state == "IN_UTURN":
            if self._uturn_complete(vehicle_state):
                self.state = "RETURN_LEFT"
                self.mppi.set_maneuver("return_left")

    def _left_lane_blocked(self, state, costmap):
        """True if costmap shows obstacle in left lane ahead of buggy."""
        x, y, theta, v = state
        ahead_x = x + 2.0 * np.cos(theta)
        ahead_y = y + 2.0 * np.sin(theta)
        return costmap.query(ahead_x, ahead_y) >= LETHAL_COST

    def _detect_upcoming_turn(self, path, state):
        """Returns heading change to next waypoint on path."""
        if len(path) < 2:
            return 0.0
        x, y = state[0], state[1]
        # Find nearest path point, look 3 waypoints ahead
        dists = [np.hypot(p[0]-x, p[1]-y) for p in path]
        idx   = int(np.argmin(dists))
        far   = min(idx + 3, len(path) - 1)
        dx    = path[far][0] - path[idx][0]
        dy    = path[far][1] - path[idx][1]
        target_heading = np.arctan2(dy, dx)
        current_heading = state[2]
        return np.arctan2(np.sin(target_heading - current_heading),
                          np.cos(target_heading - current_heading))

    def _enter_overtake(self, state):
        self.state = "OVERTAKING"
        self.mppi.set_maneuver("overtake")

    def _enter_uturn(self, state):
        self.state  = "PRE_UTURN"
        target_x    = state[0]
        target_y    = ROAD_Y_MAX - 0.5       # far right before arc
        target_theta = state[2] + np.pi      # 180-degree turn
        self.mppi.set_maneuver("u_turn", (target_x, target_y, target_theta))
        self.state = "IN_UTURN"

    def _enter_reverse(self):
        self.state = "REVERSING"
        self.mppi.set_maneuver("reverse")

    def _turn_complete(self, state):
        if self.target is None:
            return True
        return np.hypot(state[0]-self.target[0], state[1]-self.target[1]) < 0.5

    def _uturn_complete(self, state):
        if self.target is None:
            return False
        heading_diff = abs(np.arctan2(
            np.sin(state[2] - self.target[2]),
            np.cos(state[2] - self.target[2])
        ))
        return heading_diff < 0.2    # within ~12 degrees of target heading

    def _left_turn_target(self, state):
        return (state[0] + 3.0, LEFT_LANE_CENTER_Y, state[2] - np.pi/2)

    def _right_turn_target(self, state):
        return (state[0] + 3.0, RIGHT_LANE_CENTER_Y, state[2] + np.pi/2)
```

---

## ═══════════════════════════════════════════════════
## PHASE 6 — Scenario Updates for Maneuvers
## ═══════════════════════════════════════════════════

**Existing 5 scenarios (s1–s5) are kept unchanged.**
Add the following new edge case scenarios.

### 6.1 — ➕ ADD NEW `scenarios/edge_cases/ec6_wrong_lane_entry.py`

```
Purpose: Obstacle blocks left lane. Buggy must temporarily use right lane.
         After passing obstacle, must return to left lane.
Metrics: wrong_lane_violation count before and after overtake
         Time to return to left lane after clearing obstacle
Pass condition: returns to left lane within 5m of clearing obstacle
```

### 6.2 — ➕ ADD NEW `scenarios/edge_cases/ec7_u_turn_blocked.py`

```
Purpose: Buggy must execute a U-turn (road ends, goal is behind).
         Tests ManeuverStateMachine "u_turn" mode.
Metrics: Heading change achieved, min distance to road boundary during arc
Pass condition: heading rotates 170–190 degrees, no boundary collision
```

### 6.3 — ➕ ADD NEW `scenarios/edge_cases/ec8_missing_lane_mark.py`

```
Purpose: Lane markings absent (lane_cost = 0 for all y in road range).
         Tests whether MPPI can navigate without lane guidance.
Expected: Buggy may drift toward right lane. Tracks lateral drift.
Metrics: Mean lateral deviation from LEFT_LANE_CENTER_Y
```

### 6.4 — ➕ ADD NEW `scenarios/edge_cases/ec9_sensor_noise.py`

```
Purpose: Gaussian noise added to vehicle state before passing to MPPI.
         pos_noise ~ N(0, 0.05m),  heading_noise ~ N(0, 0.01 rad)
Expected: MPPI more robust than RPP was (sampling inherently noisy)
Metrics: Cross-track error variance across 10 runs
```

### 6.5 — ➕ ADD NEW `scenarios/edge_cases/ec10_all_samples_collide.py`

```
Purpose: Place obstacles all around buggy to trigger the all-collide fallback.
         Box on path + two side boxes at 0.8m lateral distance.
Expected: MPPI detects all-collide, expands sigma, finds escape after N attempts
Metrics: recovery_count, time to escape, final outcome (escaped / emergency stop)
```

---

## ═══════════════════════════════════════════════════
## PHASE 7 — Health Monitor Update
## ═══════════════════════════════════════════════════

### 7.1 — 🔧 MODIFY `health/monitor.py`

**Remove:**
```python
# DELETE: oscillation check (was RPP-specific)
self.steer_history = []
# DELETE: oscillation detection block
```

**Add:**
```python
# ADD: MPPI-specific health checks

# Check 1: MPPI recovery escalation
def check_mppi_recovery(self, recovery_count):
    if recovery_count >= MPPI_RECOVERY_MAX_ATTEMPTS:
        return "mppi_emergency_stop", "MPPI exhausted recovery attempts"
    if recovery_count > 0:
        return "mppi_recovering", f"Recovery attempt {recovery_count}"
    return "ok", ""

# Check 2: Wrong lane violation counter
def check_lane_discipline(self, state, maneuver_mode, lane_map):
    if lane_map.wrong_lane_violation(state[0], state[1], maneuver_mode):
        self.lane_violation_count += 1
        return "lane_violation", f"Wrong lane at ({state[0]:.1f},{state[1]:.1f})"
    return "ok", ""

# Check 3: Road boundary violation (separate from costmap)
def check_boundary(self, state):
    y = state[1]
    if y <= ROAD_Y_MIN or y >= ROAD_Y_MAX:
        return "boundary_exit", f"Exited road at y={y:.2f}"
    return "ok", ""

# Modify reset() to include new fields
def reset(self):
    self.speed_history      = []
    self.collision_flag     = False
    self.lane_violation_count = 0     # NEW
    self.boundary_exit      = False   # NEW
```

---

## ═══════════════════════════════════════════════════
## PHASE 8 — Metrics Recorder Update
## ═══════════════════════════════════════════════════

### 8.1 — 🔧 MODIFY `metrics/recorder.py`

**Add to __init__:**
```python
self.lane_violation_count = 0      # timesteps spent in wrong lane
self.maneuver_success     = {}     # dict of maneuver_type: bool
self.wrong_lane_durations = []     # list of (start_t, end_t) wrong-lane intervals
self.recovery_count_total = 0      # total MPPI recovery attempts across run
```

**Add to step():**
```python
# Track lane violations
if lane_map.wrong_lane_violation(state[0], state[1], maneuver_mode):
    self.lane_violation_count += 1

# Track MPPI recovery attempts
self.recovery_count_total += mppi_recovery_count
```

**Update to_dict():**
```python
d["lane_violations"]       = self.lane_violation_count
d["recovery_count_total"]  = self.recovery_count_total
d["maneuver_success"]      = str(self.maneuver_success)
# Remove: d["oscillation_count"] (RPP-specific)
```

---

## ═══════════════════════════════════════════════════
## PHASE 9 — Main Runner Update
## ═══════════════════════════════════════════════════

### 9.1 — 🔧 MODIFY `main.py`

**Remove:**
```python
# DELETE: controller comparison loop
CONTROLLERS = {"RPP": ..., "DWA": ..., "MPPI": ...}
for ctrl_name, controller in CONTROLLERS.items():
    ...
```

**Replace with single-controller loop:**
```python
from config import validate
from controllers.mppi import MPPI
from maneuvers.maneuver_state_machine import ManeuverStateMachine
from environment.lane_map import LaneMap

validate()

mppi      = MPPI()
lane_map  = LaneMap()
maneuver  = ManeuverStateMachine(mppi)

SCENARIOS = {
    "s1_static_box":      S1StaticBox(),
    "s2_person_crossing": S2PersonCrossing(),
    "s3_narrow_gap":      S3NarrowGap(),
    "s4_two_obstacles":   S4TwoObstacles(),
    "s5_sudden_obstacle": S5SuddenObstacle(),
}
EDGE_SCENARIOS = { ... }   # ec1 through ec10 (remove ec2)

RUNS_PER_SCENARIO = 10

all_results = []

for scenario_name, scenario in {**SCENARIOS, **EDGE_SCENARIOS}.items():
    for run_id in range(RUNS_PER_SCENARIO):
        seed  = BASE_SEED + run_id
        state = scenario.reset(seed=seed)
        mppi.reset()
        maneuver.state = "NORMAL"
        costmap = scenario.build_costmap()
        costmap.add_lane_boundaries()              # NEW: add lane walls
        recorder = MetricsRecorder("MPPI", scenario_name, run_id)
        monitor.reset()

        for step_i in range(MAX_STEPS):
            t = step_i * DT

            # Update environment
            scenario.update(t)
            costmap.update(scenario.get_obstacles())

            # Update maneuver state
            scenario_command = scenario.get_maneuver_command(t)  # NEW method
            maneuver.update(state, scenario.global_path, costmap, scenario_command)

            # MPPI compute
            control = mppi.compute_control(
                state, scenario.global_path, scenario.goal,
                costmap, scenario.get_obstacles()
            )

            # Health check
            status, msg = monitor.check(state, control, costmap)
            lane_status, _ = monitor.check_lane_discipline(
                state, mppi.maneuver_mode, lane_map)
            boundary_status, _ = monitor.check_boundary(state)

            if status == "collision" or boundary_status == "boundary_exit":
                recorder.finish(False, reason=status)
                break
            if status == "stuck":
                recorder.finish(False, reason="stuck")
                break

            # Step vehicle
            state = robot.update(state, control, DT)

            # Record
            recorder.step(state, control, scenario.get_obstacles(), costmap, t,
                          mppi_recovery=mppi._recovery_count,
                          maneuver_mode=mppi.maneuver_mode,
                          lane_map=lane_map)

            done, success, reason = scenario.is_complete(state, t)
            if done:
                recorder.finish(success, reason)
                break

        recorder.save_timestep_csv()
        all_results.append(recorder.to_dict())

import pandas as pd
df = pd.DataFrame(all_results)
df.to_csv("results/all_results.csv", index=False)
print(f"Done. {len(df)} runs. Results in results/all_results.csv")
```

**Command-line flags (keep existing, add new):**
```bash
python main.py --headless               # no pygame (fastest)
python main.py --edge-only --headless   # only edge case scenarios
python main.py --report                 # generate plots after runs
python main.py --scenarios s1 s3 s5    # run specific scenarios only (NEW)
python main.py --vis-mppi-samples       # show MPPI trajectory samples in pygame (NEW)
```

---

## ═══════════════════════════════════════════════════
## PHASE 10 — Visualization Update
## ═══════════════════════════════════════════════════

### 10.1 — 🔧 MODIFY `visualization/pygame_viz.py`

**Remove:**
- Controller selection display (was showing RPP/DWA/MPPI, now always MPPI)

**Add:**
```python
# ADD 1: Lane overlay (always visible)
def _draw_lane_overlay(self, surface):
    """
    Draw road boundaries, centerline and left-lane target.
    """
    # Road boundaries (solid red)
    pygame.draw.line(surface, RED,
        self._world_to_screen(0, ROAD_Y_MIN),
        self._world_to_screen(MAP_W, ROAD_Y_MIN), 3)
    pygame.draw.line(surface, RED,
        self._world_to_screen(0, ROAD_Y_MAX),
        self._world_to_screen(MAP_W, ROAD_Y_MAX), 3)

    # Road centerline (dashed yellow)
    for x in range(0, 20, 2):
        pygame.draw.line(surface, YELLOW,
            self._world_to_screen(x, ROAD_CENTERLINE_Y),
            self._world_to_screen(x+1, ROAD_CENTERLINE_Y), 2)

    # Left lane target (dashed green)
    for x in range(0, 20, 2):
        pygame.draw.line(surface, GREEN,
            self._world_to_screen(x, LEFT_LANE_CENTER_Y),
            self._world_to_screen(x+1, LEFT_LANE_CENTER_Y), 1)

# ADD 2: Maneuver state display in HUD
def _draw_hud(self, surface, state, maneuver_mode, recovery_count, lane_violations):
    texts = [
        f"Controller: MPPI",
        f"Mode: {maneuver_mode}",
        f"Speed: {state[3]:.2f} m/s",
        f"Recovery: {recovery_count}",
        f"Lane violations: {lane_violations}",
    ]
    for i, t in enumerate(texts):
        surf = self.font.render(t, True, WHITE)
        surface.blit(surf, (10, 10 + i*20))

# ADD 3: MPPI sample trajectories (toggle with --vis-mppi-samples)
def _draw_mppi_samples(self, surface, sample_trajs, weights):
    """
    Draw top-20 weighted sample trajectories as faint cyan lines.
    Opacity proportional to weight.
    """
    top_idx = np.argsort(weights)[-20:]
    for k in top_idx:
        alpha = int(weights[k] * 255)
        for t in range(len(sample_trajs[k]) - 1):
            p1 = self._world_to_screen(*sample_trajs[k][t][:2])
            p2 = self._world_to_screen(*sample_trajs[k][t+1][:2])
            pygame.draw.line(surface, (0, 200, 200), p1, p2, 1)
```

### 10.2 — 🔧 MODIFY `visualization/plot_results.py`

**Remove:**
- Multi-controller comparison plots (no longer applicable)

**Add/replace with MPPI-only plots:**
```python
# Plot 1: Path overlay per scenario (MPPI only, 10 runs)
# Plot 2: Min obstacle distance per scenario (box plot, 10 runs)
# Plot 3: Completion time per scenario (bar with error bars)
# Plot 4: Peak acceleration and jerk per scenario
# Plot 5: Failure heatmap — scenarios vs edge cases
# Plot 6: Lane discipline — wrong-lane time per scenario (NEW)
# Plot 7: Maneuver success rate (NEW — for ec6, ec7)
# Plot 8: MPPI recovery count — which scenarios trigger recovery (NEW)
```

---

## ═══════════════════════════════════════════════════
## PHASE 11 — Tests Update
## ═══════════════════════════════════════════════════

### 11.1 — 🗑️ DELETE `tests/test_dwa.py` and `tests/test_rpp.py`
(Already covered in Phase 0)

### 11.2 — 🔧 MODIFY `tests/test_mppi.py`

**Remove:**
```python
# DELETE: comparison tests that relied on DWA/RPP existing
```

**Add:**
```python
# ADD: Critic tests
def test_obstacle_critic_lethal():
    # trajectory through lethal cell → cost >= MPPI_OBS_COST

def test_lane_critic_left_lane_free():
    # trajectory in left lane → lane critic cost near 0

def test_lane_critic_right_lane_penalized():
    # trajectory in right lane → lane critic cost >= LANE_COST_RIGHT * T

def test_lane_critic_boundary_lethal():
    # trajectory outside ROAD_Y_MIN → cost == LANE_COST_BOUNDARY * T

def test_maneuver_critic_uturn():
    # u_turn mode + target heading 180° away → heading cost penalizes wrong heading

def test_adaptive_lambda_near_obstacle():
    # high costmap value → lambda returned < MPPI_LAMBDA_FREE

def test_ema_filter_smooths_output():
    # sudden control jump → EMA output changes gradually

def test_all_collide_fallback():
    # all K trajectories lethal → returns (0.0, 0.0) after max attempts

def test_recovery_widens_sigma():
    # first recovery attempt → sigma > MPPI_SIGMA_STEER

def test_warm_start_shifts():
    # call twice → U[0] at step 2 equals U[1] at step 1 (before eps update)

def test_narrow_context_increases_K():
    # walls within 1m both sides → _is_narrow_context() returns True

def test_ackermann_clamp_in_rollout():
    # very large delta in U_noisy → rollout clamps to feasible radius
```

### 11.3 — ➕ ADD NEW `tests/test_lane_map.py`

```python
def test_left_lane_zero_cost():
    # y = LEFT_LANE_CENTER_Y → lane_cost = 0

def test_right_lane_penalized():
    # y = RIGHT_LANE_CENTER_Y → lane_cost = LANE_COST_RIGHT

def test_boundary_lethal():
    # y = ROAD_Y_MIN - 0.1 → lane_cost = LANE_COST_BOUNDARY

def test_get_target_y_normal():
    # mode="normal" → returns LEFT_LANE_CENTER_Y

def test_get_target_y_uturn():
    # mode="u_turn" → returns RIGHT_LANE_CENTER_Y

def test_wrong_lane_detection_normal_mode():
    # y = 12.5, mode="normal" → wrong_lane_violation = True

def test_wrong_lane_detection_overtake_mode():
    # y = 12.5, mode="overtake" → wrong_lane_violation = False
```

---

## ═══════════════════════════════════════════════════
## PHASE 12 — Complete Edge Case & Critical Condition Table
## ═══════════════════════════════════════════════════

| # | Condition | Detection | MPPI Handler | Log Field |
|---|---|---|---|---|
| EC-1 | Dead end — all paths blocked | All K samples collide | Recovery: widen sigma, try N times, then emergency stop | failure_reason="dead_end" |
| EC-2 | ~~Controller oscillation~~ | ~~RPP-specific~~ | ~~Deleted~~ | — |
| EC-3 | Ackermann violation | v*tan(delta)/L > curvature_max | Clamp delta in rollout | logged per step |
| EC-4 | Late braking | Obstacle dist < braking_distance | Obstacle critic cost spikes → MPPI decelerates | failure_reason="near_miss" |
| EC-5 | Crowd — 3 pedestrians | 3+ dynamic obstacles in costmap | Obstacle critic handles all 3 simultaneously | hesitation_time |
| EC-6 | Wrong lane entry | lane_map.wrong_lane_violation() | Lane critic guides back to left lane | lane_violation_count |
| EC-7 | Goal inside inflation | costmap.query(goal) >= INSCRIBED_COST | project_goal_to_free() | goal_adjusted=True |
| EC-8 | Stuck | max(speed_history) < STUCK_SPEED_THRESH | Mark failure | failure_reason="stuck" |
| EC-9 | Timeout | t >= TIME_LIMIT | Mark failure | failure_reason="timeout" |
| EC-10 | Collision | costmap.query(pos) >= LETHAL_COST | Stop run | failure_reason="collision" |
| EC-11 | Obstacle spawns on robot | spawn pos overlaps footprint | Offset spawn 0.5m | warning logged |
| EC-12 | MPPI weights all zero | sum(weights) < 1e-10 | Uniform weights fallback | warning logged |
| EC-13 | All samples collide | min(costs) >= MPPI_OBS_COST | Sigma expansion → recovery → emergency stop | recovery_count_total |
| EC-14 | Road boundary exit | y <= ROAD_Y_MIN or y >= ROAD_Y_MAX | Stop run (boundary_exit) | failure_reason="boundary_exit" |
| EC-15 | U-turn infeasible (too narrow) | Road width < 2*MANEUVER_UTURN_RADIUS | Return failure_reason="uturn_infeasible" | failure_reason |
| EC-16 | Maneuver target unreachable | Maneuver target inside obstacle | Abort maneuver, revert to NORMAL | maneuver_success=False |
| EC-17 | Missing lane markings | lane_map disabled for scenario | Buggy navigates by obstacle + path critics only | lateral_drift_mean |
| EC-18 | Sensor noise / state uncertainty | Noise added to state before MPPI | MPPI inherently robust (sampling is stochastic) | cross_track_error_var |
| EC-19 | Frozen pedestrian mid-path | Person stops at y=10 (centerline) | Maneuver: overtake → right lane | failure_reason if stuck |
| EC-20 | Narrow gap + wrong lane | Obstacle right-of-center in narrow gap | Increase K, obstacle + lane critics cooperate | narrow_gap_violation |

---

## ═══════════════════════════════════════════════════
## PHASE 13 — Agent Execution Order
## (Exact sequence — no import errors if followed precisely)
## ═══════════════════════════════════════════════════

```
── PHASE 0: Safe Removal ──────────────────────────────────────────
Step 0.1: Search and remove all RPP/DWA imports from main.py and __init__.py
Step 0.2: DELETE controllers/rpp.py
Step 0.3: DELETE controllers/dwa.py
Step 0.4: DELETE tests/test_dwa.py
Step 0.5: DELETE tests/test_rpp.py
Step 0.6: DELETE scenarios/edge_cases/ec2_oscillation_trigger.py
Step 0.7: Remove ec2 from scenarios/edge_cases/__init__.py
Step 0.8: Verify: python -c "from controllers.mppi import MPPI; print('OK')"

── PHASE 1: Config ────────────────────────────────────────────────
Step 1.1: config.py — remove RPP/DWA params, add MPPI advanced + lane params
Step 1.2: Verify: python -c "from config import validate; validate()"

── PHASE 2: Lane Map ──────────────────────────────────────────────
Step 2.1: environment/lane_map.py            [ADD NEW]
Step 2.2: environment/costmap.py             [MODIFY — add add_lane_boundaries()]
Step 2.3: Verify: python -c "from environment.lane_map import LaneMap; LaneMap()"

── PHASE 3: MPPI Critics ──────────────────────────────────────────
Step 3.1: mppi_critics/__init__.py           [ADD NEW]
Step 3.2: mppi_critics/obstacle_critic.py   [ADD NEW]
Step 3.3: mppi_critics/goal_critic.py       [ADD NEW]
Step 3.4: mppi_critics/path_critic.py       [ADD NEW]
Step 3.5: mppi_critics/jerk_critic.py       [ADD NEW]
Step 3.6: mppi_critics/speed_critic.py      [ADD NEW]
Step 3.7: mppi_critics/lane_critic.py       [ADD NEW]
Step 3.8: mppi_critics/maneuver_critic.py   [ADD NEW]
Step 3.9: Verify: python -c "from mppi_critics import ObstacleCritic; print('OK')"

── PHASE 4: MPPI Core Upgrade ────────────────────────────────────
Step 4.1: controllers/controller_base.py     [MODIFY — remove RPP/DWA references]
Step 4.2: controllers/mppi.py                [MODIFY — full rewrite of class body]
Step 4.3: Verify: python -c "from controllers.mppi import MPPI; MPPI()"

── PHASE 5: Maneuver System ──────────────────────────────────────
Step 5.1: maneuvers/__init__.py              [ADD NEW]
Step 5.2: maneuvers/maneuver_state_machine.py [ADD NEW]
Step 5.3: maneuvers/left_turn.py             [ADD NEW]
Step 5.4: maneuvers/right_turn.py            [ADD NEW]
Step 5.5: maneuvers/u_turn.py                [ADD NEW]
Step 5.6: maneuvers/reverse.py               [ADD NEW]
Step 5.7: Verify: python -c "from maneuvers.maneuver_state_machine import ManeuverStateMachine; print('OK')"

── PHASE 6: Edge Case Scenarios ──────────────────────────────────
Step 6.1: scenarios/edge_cases/ec6_wrong_lane_entry.py  [ADD NEW]
Step 6.2: scenarios/edge_cases/ec7_u_turn_blocked.py    [ADD NEW]
Step 6.3: scenarios/edge_cases/ec8_missing_lane_mark.py [ADD NEW]
Step 6.4: scenarios/edge_cases/ec9_sensor_noise.py      [ADD NEW]
Step 6.5: scenarios/edge_cases/ec10_all_samples_collide.py [ADD NEW]
Step 6.6: Update scenarios/edge_cases/__init__.py to include ec6–ec10

── PHASE 7: Health Monitor ───────────────────────────────────────
Step 7.1: health/monitor.py  [MODIFY — remove oscillation, add lane/boundary checks]

── PHASE 8: Metrics ──────────────────────────────────────────────
Step 8.1: metrics/recorder.py         [MODIFY — add lane_violations, recovery_count]
Step 8.2: metrics/report_generator.py [MODIFY — MPPI-only plots, add lane plot]

── PHASE 9: Main Runner ──────────────────────────────────────────
Step 9.1: main.py  [MODIFY — single MPPI controller, maneuver loop, lane tracking]

── PHASE 10: Visualization ───────────────────────────────────────
Step 10.1: visualization/pygame_viz.py   [MODIFY — lane overlay, maneuver HUD]
Step 10.2: visualization/plot_results.py [MODIFY — MPPI-only plots]

── PHASE 11: Tests ───────────────────────────────────────────────
Step 11.1: tests/test_mppi.py            [MODIFY — add critic/lane/maneuver tests]
Step 11.2: tests/test_lane_map.py        [ADD NEW]

── PHASE 12: Run & Verify ────────────────────────────────────────
Step 12.1: python -m pytest tests/ -v
           Expected: test_bicycle_model, test_mppi, test_costmap, test_lane_map all pass
           test_dwa and test_rpp do NOT exist (confirm deleted)

Step 12.2: python main.py --headless
           Expected: 50 rows in results/all_results.csv
           (5 core scenarios × the default 10 runs each)
           With --edge-cases, expect 140 total rows (5 core + 9 edge scenarios,
           each run 10 times); with --edge-only, expect 90 edge rows.

Step 12.3: python -m metrics.report_generator
           Expected: 8 PNG files in results/plots/

Step 12.4: python main.py --edge-only --headless
           Expected: 90 rows in results/edge_case_results/all_edge_results.csv
           (9 registered edge scenarios × the default 10 runs each)

Step 12.5: Checklist:
  [ ] results/all_results.csv — no "RPP" or "DWA" in controller column
  [ ] results/plots/plot6_lane_discipline.png exists
  [ ] results/plots/plot7_maneuver_success.png exists
  [ ] results/plots/plot8_mppi_recovery.png exists
  [ ] No import of rpp or dwa anywhere in codebase
      (run: grep -r "from controllers.rpp\|from controllers.dwa" . → no results)
```

---

## 📋 Section 14 — Final Deliverables

### Results files

```
results/
├── all_results.csv                   # 50 core rows by default; MPPI only
├── summary_table.csv                 # Mean ± std per scenario
├── raw/                              # Timestep-level CSVs
│   └── MPPI__s1_static_box__run00.csv ... (50 default core-run files)
├── edge_case_results/
│   └── all_edge_results.csv          # 90 rows for the default edge-only run
└── plots/
    ├── plot1_path_comparison.png     # 10 MPPI runs per scenario
    ├── plot2_min_distance.png        # Clearance box plot
    ├── plot3_completion_time.png     # Time per scenario
    ├── plot4_accel_jerk.png          # Comfort metrics
    ├── plot5_failure_heatmap.png     # Scenarios × edge cases
    ├── plot6_lane_discipline.png     # NEW: wrong-lane time per scenario
    ├── plot7_maneuver_success.png    # NEW: maneuver outcomes
    └── plot8_mppi_recovery.png       # NEW: recovery count per scenario
```

### Summary table (MPPI only)

| Scenario | Success | Avg Time (s) | Min Dist (m) | Peak Jerk | Lane Viols | Recoveries | Fails |
|---|---|---|---|---|---|---|---|
| s1_static_box | /10 | | | | | | |
| s2_person_crossing | /10 | | | | | | |
| s3_narrow_gap | /10 | | | | | | |
| s4_two_obstacles | /10 | | | | | | |
| s5_sudden_obstacle | /10 | | | | | | |
| ec1_dead_end | /10 | | | | | | |
| ec3_tight_turn | /10 | | | | | | |
| ec4_late_braking | /10 | | | | | | |
| ec5_crowd | /10 | | | | | | |
| ec6_wrong_lane_entry | /10 | | | | | | |
| ec7_u_turn_blocked | /10 | | | | | | |
| ec8_missing_lane_mark | /10 | | | | | | |
| ec9_sensor_noise | /10 | | | | | | |
| ec10_all_samples_collide | /10 | | | | | | |

---

## 📋 Section 15 — Verified References

| Source | Type | Used for |
|---|---|---|
| Williams et al. (2017) "Information Theoretic MPC" IEEE RA-L | Paper (⚠️ VERIFIED) | MPPI mathematical foundation, update rule, lambda |
| Kim et al. (2022) arXiv:2112.09988 "Smooth MPPI without Smoothing" | Paper (⚠️ VERIFIED) | Chattering problem, EMA filter motivation |
| Vlahov et al. (2024) arXiv:2404.03094 "Low Frequency Sampling in MPPI" | Paper (⚠️ VERIFIED) | Gaussian sampling limitation, colored noise alternative |
| Shield-MPPI arXiv:2302.11719 | Paper (⚠️ VERIFIED) | Infeasible rollout fallback, safety guarantees |
| o-MPPI arXiv:2309.13201 | Paper (⚠️ VERIFIED) | All-samples-collide failure mode |
| Testouri et al. (2023) Biomimetics "Safe MPPI for ADS" | Paper (⚠️ VERIFIED) | Circular obstacle approximation in MPPI cost |
| Nav2 MPPI Controller (Macenski/Budyakov) | Production code (⚠️ VERIFIED) | Critic plugin architecture, 50Hz benchmark, Ackermann mode |
| MizuhoAOKI/python_simple_mppi | GitHub (⚠️ VERIFIED) | Numpy rollout reference, path tracking + obstacle avoidance |
| UM-ARM-Lab/pytorch_mppi | GitHub (⚠️ VERIFIED) | Adaptive covariance, recovery logic |
| Shunichi09/PythonLinearNonlinearControl | GitHub (⚠️ VERIFIED) | filter_coefs smoothing pattern |
| GeoDrive-Bench arXiv:2606.02774 | Paper (⚠️ VERIFIED) | Indian traffic rules — keep left, left turn on red, yield rules |
| Indian Motor Vehicles Act | Legal (📌 ASSUMPTION — cross-ref GeoDrive-Bench) | Lane discipline, overtake rules |
| OpenPodCar arXiv:2205.04454 | Paper (⚠️ VERIFIED) | Campus mobility vehicle model, Ackermann config, narrow passages |

---

*Plan version: 3.0 (MPPI-only) | Built on v2.0 | October 2026*
*RPP and DWA safely removed. MPPI is the sole active controller.*
*All changes are modifications or additions to existing codebase — not a rebuild.*
