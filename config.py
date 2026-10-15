# =============================================================================
# config.py  —  Campus Buggy Simulation — Single Source of Truth
# =============================================================================
# All tunable parameters live here.
# Every module should import from this file rather than hardcoding values.

# ── Robot ──────────────────────────────────────────────────────────────────
WHEELBASE    = 0.5    # metres  — distance between front and rear axle
ROBOT_RADIUS = 0.3    # metres  — used for collision detection / footprint
MAX_SPEED    = 1.5    # m/s
MAX_ACCEL    = 1.0    # m/s²
MAX_STEER    = 0.6    # radians (~34 degrees)
DT           = 0.05   # seconds — simulation timestep

# ── Map ────────────────────────────────────────────────────────────────────
MAP_W = 20.0          # metres
MAP_H = 20.0          # metres

# ── Experiment ─────────────────────────────────────────────────────────────
RUNS_PER_COMBO  = 10
TIME_LIMIT      = 60.0   # seconds per run
HEADLESS        = False  # True = skip pygame window (much faster batch runs)

# ── Regulated Pure Pursuit (RPP) ───────────────────────────────────────────
RPP_LOOKAHEAD          = 1.0   # metres — look-ahead distance on path
RPP_MAX_SPEED          = 1.5   # m/s
RPP_MIN_SPEED          = 0.3   # m/s — minimum speed near obstacles
RPP_OBSTACLE_SLOWDOWN  = True  # reduce speed when obstacle is close
RPP_CURVATURE_SLOWDOWN = True  # reduce speed when path curvature is high

# ── Dynamic Window Approach (DWA) ──────────────────────────────────────────
DWA_V_SAMPLES    = 20    # number of linear velocity samples
DWA_W_SAMPLES    = 30    # number of angular velocity samples
DWA_SIM_TIME     = 2.0   # seconds to simulate each candidate trajectory
DWA_HEADING_W    = 0.8   # weight: progress toward goal
DWA_CLEARANCE_W  = 0.2   # weight: clearance from obstacles
DWA_SPEED_W      = 0.4   # weight: preference for higher speed

# ── MPPI ────────────────────────────────────────────────────────────────────
MPPI_K         = 500    # number of trajectory samples
MPPI_T         = 80     # horizon steps (4.0 s @ DT=0.05 — long enough for a full
                        # detour around the s1 box to fit inside the horizon;
                        # at T=40 the crash-avoidance cost made freezing beat detouring)
MPPI_SIGMA     = 0.5    # noise standard deviation
MPPI_LAMBDA    = 1.0    # temperature — controls how sharply the best is selected
MPPI_OBS_COST  = 1e6    # high cost for hitting an obstacle
MPPI_PATH_COST = 1.0    # cost weight for deviation from reference path
MPPI_CTRL_COST = 0.1    # cost weight for large control effort
MPPI_FLUSH_ON_LETHAL = True  # reset warm-start U when a lethal cell is detected ahead

# ── Comfort / Safety Thresholds ────────────────────────────────────────────
MAX_COMFORTABLE_ACCEL = 1.5   # m/s²  — above this = uncomfortable
MAX_COMFORTABLE_JERK  = 3.0   # m/s³  — above this = uncomfortable
MIN_SAFE_DISTANCE     = 0.3   # m      — same as ROBOT_RADIUS (collision boundary)

# ── Stuck Detection ────────────────────────────────────────────────────────
STUCK_TIME_THRESHOLD  = 5.0   # seconds — if robot hasn't moved, it's "stuck"
STUCK_DIST_THRESHOLD  = 0.05  # metres  — movement less than this = not moving

# ── Goal Reach Tolerance ──────────────────────────────────────────────────
GOAL_TOLERANCE = 0.5   # metres — robot is "at goal" if within this distance

# ── Costmap (NEW) ─────────────────────────────────────────────────────────
COSTMAP_RESOLUTION  = 0.05    # metres per cell
INFLATION_RADIUS    = 0.8     # metres — expand obstacles outward (>= ROBOT_RADIUS)
INFLATION_DECAY     = 3.0     # decay rate for cost falloff
LETHAL_COST         = 99      # cost value = collision
INSCRIBED_COST      = 80      # cost inside inflation radius

# ── MPPI extended critics (NEW) ──────────────────────────────────────────
MPPI_OBSTACLE_WEIGHT  = 1.0
MPPI_GOAL_WEIGHT      = 2.0
MPPI_PATH_WEIGHT      = 1.5
MPPI_JERK_WEIGHT      = 0.5
MPPI_SPEED_WEIGHT     = 0.3

# ── DWA Ackermann fix (NEW) ──────────────────────────────────────────────
DWA_STEER_SAMPLES   = 30      # steering angle samples (replaces W_SAMPLES)
DWA_STEER_RATE      = 0.3     # max steering change per timestep (rad/s)
DWA_MIN_TURN_RADIUS = 0.5     # metres — physical Ackermann constraint

# ── RPP adaptive lookahead (NEW) ─────────────────────────────────────────
RPP_LOOKAHEAD_GAIN  = 0.5     # L_d = base + gain * speed
RPP_MIN_LOOKAHEAD   = 0.4     # metres
RPP_MAX_LOOKAHEAD   = 3.0     # metres
RPP_STEER_LOWPASS   = 0.7     # low-pass filter coefficient (0=heavy, 1=off)

# ── Health monitor (NEW) ─────────────────────────────────────────────────
STUCK_TIMEOUT       = 5.0     # seconds not moving → stuck
STUCK_SPEED_THRESH  = 0.05    # m/s — below this = not moving
OSCILLATION_WINDOW  = 10      # timesteps to check for oscillation
OSCILLATION_THRESH  = 3       # sign flips in steering → oscillating

# ── Edge case scenarios (NEW) ────────────────────────────────────────────
CROWD_N_PEDESTRIANS = 3       # number of moving people in crowd scenario
SENSOR_NOISE_POS    = 0.05    # metres — Gaussian position noise
SENSOR_NOISE_HDG    = 0.01    # radians — Gaussian heading noise

# ── Experiment reproducibility (NEW) ─────────────────────────────────────
BASE_SEED           = 42      # run_id + BASE_SEED = per-run seed

# ── Config self-validation (NEW) ─────────────────────────────────────────
def validate():
    assert MAX_SPEED > 0, "MAX_SPEED must be positive"
    assert DT > 0 and DT < 0.5, "DT must be small positive value"
    assert MPPI_K > 0, "Need at least 1 MPPI sample"
    assert INFLATION_RADIUS >= ROBOT_RADIUS, \
        "Inflation must be >= robot radius or robot can pass through lethal zone"
    assert RPP_MIN_LOOKAHEAD < RPP_MAX_LOOKAHEAD, "Lookahead bounds inverted"
    horizon_reach = MPPI_T * DT * MAX_SPEED
    assert horizon_reach >= 2.0, \
        f"MPPI horizon only covers {horizon_reach:.1f}m; must cover >=2m for S5 hard spawn"
    print("[config] All parameters valid.")

