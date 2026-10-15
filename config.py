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
MPPI_T         = 20     # horizon steps
MPPI_SIGMA     = 0.5    # noise standard deviation
MPPI_LAMBDA    = 1.0    # temperature — controls how sharply the best is selected
MPPI_OBS_COST  = 1e6    # high cost for hitting an obstacle
MPPI_PATH_COST = 1.0    # cost weight for deviation from reference path
MPPI_CTRL_COST = 0.1    # cost weight for large control effort

# ── Comfort / Safety Thresholds ────────────────────────────────────────────
MAX_COMFORTABLE_ACCEL = 1.5   # m/s²  — above this = uncomfortable
MAX_COMFORTABLE_JERK  = 3.0   # m/s³  — above this = uncomfortable
MIN_SAFE_DISTANCE     = 0.3   # m      — same as ROBOT_RADIUS (collision boundary)

# ── Stuck Detection ────────────────────────────────────────────────────────
STUCK_TIME_THRESHOLD  = 5.0   # seconds — if robot hasn't moved, it's "stuck"
STUCK_DIST_THRESHOLD  = 0.05  # metres  — movement less than this = not moving

# ── Goal Reach Tolerance ──────────────────────────────────────────────────
GOAL_TOLERANCE = 0.5   # metres — robot is "at goal" if within this distance
