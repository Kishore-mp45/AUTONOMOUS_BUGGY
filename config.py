# =============================================================================
# config.py  —  Campus Buggy Simulation — Single Source of Truth
# =============================================================================
# All tunable parameters live here.
# Every module should import from this file rather than hardcoding values.
#
# MPPI-only configuration (v3.0): RPP and DWA parameters removed.

# ── Robot ──────────────────────────────────────────────────────────────────
WHEELBASE    = 0.5    # metres  — distance between front and rear axle
ROBOT_RADIUS = 0.3    # metres  — used for collision detection / footprint
MAX_SPEED    = 1.5    # m/s
MAX_ACCEL    = 1.0    # m/s²
MAX_STEER    = 0.6    # radians (~34 degrees)
DT           = 0.05   # seconds — simulation timestep

# Ackermann physical minimum turning radius (was DWA_MIN_TURN_RADIUS; the
# constraint itself is robot physics, not a controller choice)
MIN_TURN_RADIUS = 0.5     # metres

# ── Map ────────────────────────────────────────────────────────────────────
MAP_W = 20.0          # metres
MAP_H = 20.0          # metres

# ── Experiment ─────────────────────────────────────────────────────────────
RUNS_PER_COMBO  = 10
TIME_LIMIT      = 60.0   # seconds per run
HEADLESS        = False  # True = skip pygame window (much faster batch runs)
BASE_SEED       = 42     # run_id + BASE_SEED = per-run seed

# ── MPPI core ──────────────────────────────────────────────────────────────
MPPI_K         = 500    # number of trajectory samples
MPPI_LAMBDA    = 1.0    # temperature — controls how sharply the best is selected
                        # (base value; adaptive override in MPPI_LAMBDA_*)
MPPI_T         = 80     # horizon steps (4.0 s @ DT=0.05). Empirically required:
                        # at T=40 (2 s) a detour around the s1 box does not fit
                        # inside the horizon and MPPI freezes in front of it.
MPPI_OBS_COST  = 1e6    # high cost for hitting an obstacle
MPPI_PATH_COST = 1.0    # cost weight for deviation from reference path
MPPI_CTRL_COST = 0.1    # cost weight for large control effort

# ── MPPI Advanced Controls ─────────────────────────────────────────────────
MPPI_MIN_K          = 300       # minimum samples (narrow gap uses 600)
MPPI_NARROW_K       = 600       # samples for narrow-gap scenarios
MPPI_SIGMA_STEER    = 0.08      # lower steering noise reduces clear-path zigzag
MPPI_SIGMA_ACCEL    = 0.2       # acceleration noise std
MPPI_EMA_ALPHA      = 0.85      # EMA filter for output smoothing (0=heavy, 1=off)
MPPI_ITER           = 1         # iterations per timestep (increase for complex env)
MPPI_NOISE_ALPHA    = 0.9       # coloured-noise low-pass on epsilon along the
                                # horizon (plan §1.4). With iid per-step noise no
                                # sample sustains acceleration, so the accel
                                # gradient is swamped by lateral cost noise and
                                # the buggy never gets moving.

# ── Adaptive MPPI ──────────────────────────────────────────────────────────
MPPI_LAMBDA_NEAR_OBS   = 0.5    # lower lambda when in high-cost zone (greedier)
MPPI_LAMBDA_FREE       = 1.5    # higher lambda on clear path (more exploratory)
MPPI_LAMBDA_ADAPT_COST = 40.0   # costmap value above which lambda switches
MPPI_HORIZON_NEAR_OBS  = 80     # longer horizon near obstacle
MPPI_HORIZON_DEFAULT   = 60     # default horizon

# ── MPPI critics (weights used by the critics package) ────────────────────
MPPI_OBSTACLE_WEIGHT  = 0.1     # soft near-obstacle cost scale. Empirically
                                # capped at 0.1×cell-cost: at 1.0 the integrated
                                # inflation-zone cost exceeds the stop cost and
                                # MPPI freezes in front of the s1 box instead
                                # of detouring (verified in earlier tuning).
MPPI_GOAL_WEIGHT      = 10.0   # terminal goal-distance weight. Must dominate
                               # the lane-centering pull near the goal, else the
                               # buggy orbits the goal instead of entering the
                               # 0.5 m tolerance ring (verified failure mode).
MPPI_PATH_WEIGHT      = 1.5
MPPI_JERK_WEIGHT      = 0.05   # squared-jerk weight. The plan's 0.5 assumes
                               # smooth sampled controls; with stochastic accel
                               # noise (±0.13/step even when coloured) velocity-
                               # derived jerk ≈2.5 and 0.5×Σjerk² ≈ 180-460,
                               # drowning the ~40-cost acceleration gradient and
                               # freezing the buggy. 0.05 keeps comfort pressure
                               # without masking progress.
MPPI_SPEED_WEIGHT     = 1.0    # per-step speed-tracking weight. 0.3 (plan value)
                               # gives a ~40-cost spread for stopping over the 60-
                               # step horizon, too weak against lateral cost noise;
                               # 1.0 makes progress clearly cheaper than freezing.

# ── MPPI Recovery ──────────────────────────────────────────────────────────
MPPI_RECOVERY_SIGMA_MULT    = 2.0   # multiply sigma when all-collide detected
MPPI_RECOVERY_MAX_ATTEMPTS  = 5     # timesteps to try before emergency stop

# ── Lane geometry (Indian left-hand traffic) ───────────────────────────────
ROAD_Y_MIN          = 5.0       # road lower boundary (left kerb)
ROAD_Y_MAX          = 15.0      # road upper boundary (right kerb)
ROAD_CENTERLINE_Y   = 10.0      # road centre
LANE_WIDTH          = 2.5       # metres per lane
LEFT_LANE_CENTER_Y  = 8.5       # target y for left lane (MPPI follows this)
RIGHT_LANE_CENTER_Y = 12.5      # right lane centre (temporary overtake only)
LANE_COST_LEFT      = 0.0       # cost for being in left lane (reward = zero cost)
LANE_COST_RIGHT     = 30.0      # soft penalty for being in right lane
LANE_COST_BOUNDARY  = 1e6       # lethal cost for crossing road boundary

# ── Maneuver modes ─────────────────────────────────────────────────────────
MANEUVER_LEFT_TURN_RADIUS   = 1.5   # metres
MANEUVER_RIGHT_TURN_RADIUS  = 2.5   # metres
MANEUVER_UTURN_RADIUS       = 3.0   # metres
MANEUVER_REVERSE_MAX_SPEED  = 0.5   # m/s
MANEUVER_GOAL_ARRIVAL_DIST  = 0.5   # metres — when to declare goal reached

# ── Comfort / Safety Thresholds ────────────────────────────────────────────
MAX_COMFORTABLE_ACCEL = 1.5   # m/s²  — above this = uncomfortable
MAX_COMFORTABLE_JERK  = 3.0   # m/s³  — above this = uncomfortable
MIN_SAFE_DISTANCE     = 0.3   # m      — same as ROBOT_RADIUS (collision boundary)

# ── Stuck Detection ────────────────────────────────────────────────────────
STUCK_TIME_THRESHOLD  = 5.0   # seconds — if robot hasn't moved, it's "stuck"
STUCK_DIST_THRESHOLD  = 0.05  # metres  — movement less than this = not moving
STUCK_TIMEOUT         = 5.0   # seconds not moving → stuck
STUCK_SPEED_THRESH    = 0.05  # m/s — below this = not moving

# ── Goal Reach Tolerance ──────────────────────────────────────────────────
GOAL_TOLERANCE = 0.5   # metres — robot is "at goal" if within this distance

# ── Costmap ────────────────────────────────────────────────────────────────
COSTMAP_RESOLUTION  = 0.05    # metres per cell
INFLATION_RADIUS    = 0.8     # metres — expand obstacles outward (>= ROBOT_RADIUS)
INFLATION_DECAY     = 3.0     # decay rate for cost falloff
LETHAL_COST         = 99      # cost value = collision
INSCRIBED_COST      = 80      # cost inside inflation radius

# ── Edge case scenarios ────────────────────────────────────────────────────
CROWD_N_PEDESTRIANS = 3       # number of moving people in crowd scenario
SENSOR_NOISE_POS    = 0.05    # metres — Gaussian position noise
SENSOR_NOISE_HDG    = 0.01    # radians — Gaussian heading noise

# ── Config self-validation ─────────────────────────────────────────────────
def validate():
    assert MAX_SPEED > 0, "MAX_SPEED must be positive"
    assert DT > 0 and DT < 0.5, "DT must be small positive value"
    assert MPPI_K >= MPPI_MIN_K, "K must be >= minimum sample count"
    assert MPPI_LAMBDA > 0, "MPPI_LAMBDA must be positive"
    assert INFLATION_RADIUS >= ROBOT_RADIUS, \
        "Inflation must be >= robot radius or robot can pass through lethal zone"
    horizon_reach = MPPI_T * DT * MAX_SPEED
    assert horizon_reach >= 2.0, \
        f"MPPI horizon only covers {horizon_reach:.1f}m; must cover >=2m for S5 hard spawn"
    assert ROAD_Y_MIN < LEFT_LANE_CENTER_Y < ROAD_CENTERLINE_Y, \
        "Left lane centre must lie between road edge and centreline"
    assert ROAD_CENTERLINE_Y < RIGHT_LANE_CENTER_Y < ROAD_Y_MAX, \
        "Right lane centre must lie between centreline and road edge"
    assert LANE_COST_LEFT < LANE_COST_RIGHT, \
        "Right lane must cost more than left lane"
    print("[config] All parameters valid — MPPI-only mode")
