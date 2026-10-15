"""
metrics/recorder.py
====================
MetricsRecorder — records all per-run data and computes final metrics.

Per-step recording (call once per simulation timestep):
    recorder.step(state, obstacles, t)

At run end (call once):
    recorder.finish(success=True/False)

Extract results:
    d   = recorder.to_dict()      # flat dict — one row for pandas DataFrame
    recorder.save_csv(path)       # time-series CSV for this run

Output CSV columns (summary, one row per run):
    controller, scenario, run_id,
    completed, failure_reason,
    completion_time, path_length,
    min_obstacle_dist,
    peak_acceleration, peak_jerk,
    mean_speed, final_x, final_y

Time-series CSV columns (one row per timestep):
    t, x, y, theta, v, min_obs_dist, accel, jerk
"""

import os
import math
import csv
import time
from typing import List, Dict, Any, Optional

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class MetricsRecorder:
    """
    Records all metrics for a single controller × scenario × run combination.

    Parameters
    ----------
    controller_name : str   e.g. "RPP", "DWA", "MPPI"
    scenario_name   : str   e.g. "s1_static_box"
    run_id          : int   0-indexed run number
    """

    def __init__(
        self,
        controller_name: str,
        scenario_name:   str,
        run_id:          int,
    ):
        self.controller_name = controller_name
        self.scenario_name   = scenario_name
        self.run_id          = run_id

        # ── Time-series buffers ────────────────────────────────────────
        self.times:       List[float] = []   # t  (s)
        self.positions:   List[tuple] = []   # (x, y)
        self.thetas:      List[float] = []   # heading (rad)
        self.velocities:  List[float] = []   # v (m/s)
        self.obs_dists:   List[float] = []   # min dist to obstacle each step

        # ── Running scalar metrics ─────────────────────────────────────
        self.min_obstacle_dist: float = float("inf")
        self.start_time:        Optional[float] = None   # wall-clock (unused for completion_time)
        self.end_time:          Optional[float] = None
        self._sim_start_t:      Optional[float] = None   # sim time at first step
        self._sim_end_t:        Optional[float] = None   # sim time at finish

        # ── Outcome ───────────────────────────────────────────────────
        self.completed:       bool = False
        self.failed:          bool = False
        self.failure_reason:  str  = ""   # "collision" | "timeout" | "stuck" | ""

        # ── Derived (computed in finish()) ────────────────────────────
        self.completion_time:   float = float("nan")
        self.path_length:       float = 0.0
        self.peak_acceleration: float = 0.0
        self.peak_jerk:         float = 0.0
        self.mean_speed:        float = 0.0

        # ── Stuck detection state ─────────────────────────────────────
        self._stuck_last_pos:  Optional[tuple] = None
        self._stuck_accum_t:   float = 0.0   # seconds without meaningful movement

    # ------------------------------------------------------------------
    # Per-step recording
    # ------------------------------------------------------------------

    def step(
        self,
        state:     np.ndarray,
        obstacles: List[Dict[str, Any]],
        t:         float,
    ) -> None:
        """
        Record one simulation timestep.

        Parameters
        ----------
        state     : np.ndarray [x, y, theta, v]
        obstacles : current obstacle list (for distance computation)
        t         : simulation time in seconds
        """
        x, y, theta, v = float(state[0]), float(state[1]), float(state[2]), float(state[3])

        # ── Initialise start time on first call ───────────────────────
        if self._sim_start_t is None:
            self._sim_start_t = t
            self.start_time   = time.time()

        # ── Record position / heading / speed ─────────────────────────
        self.times.append(t)
        self.positions.append((x, y))
        self.thetas.append(theta)
        self.velocities.append(v)

        # ── Distance to nearest obstacle ──────────────────────────────
        min_d = _min_obstacle_dist(x, y, obstacles)
        self.obs_dists.append(min_d)
        if min_d < self.min_obstacle_dist:
            self.min_obstacle_dist = min_d

        # ── Collision detection ───────────────────────────────────────
        if min_d < config.ROBOT_RADIUS and not self.failed:
            self._mark_failure("collision", t)

        # ── Stuck detection ───────────────────────────────────────────
        self._check_stuck(x, y, t)

    def mark_timeout(self, t: float) -> None:
        """Call when TIME_LIMIT is exceeded without reaching the goal."""
        if not self.failed:
            self._mark_failure("timeout", t)

    def mark_stuck(self, t: float) -> None:
        """Call when the robot has been stationary for STUCK_TIME_THRESHOLD."""
        if not self.failed:
            self._mark_failure("stuck", t)

    # ------------------------------------------------------------------
    # Run end
    # ------------------------------------------------------------------

    def finish(self, success: bool, t: float = None) -> None:
        """
        Finalise the run.  Call once when the run ends (goal reached or failed).

        Parameters
        ----------
        success : bool   True = goal reached;  False = failure of some kind
        t       : float  sim time at finish (uses last recorded time if None)
        """
        self.end_time    = time.time()
        self._sim_end_t  = t if t is not None else (self.times[-1] if self.times else 0.0)
        self.completed   = success

        if success and not self.failed:
            self.failure_reason = ""
        # If failed was already set, keep the failure_reason

        # ── Completion time ───────────────────────────────────────────
        if self._sim_start_t is not None:
            self.completion_time = self._sim_end_t - self._sim_start_t
        else:
            self.completion_time = 0.0

        # ── Path length  (sum of Euclidean segment lengths) ───────────
        self.path_length = _compute_path_length(self.positions)

        # ── Mean speed ────────────────────────────────────────────────
        if self.velocities:
            self.mean_speed = float(np.mean(self.velocities))

        # ── Peak acceleration & jerk ──────────────────────────────────
        if len(self.velocities) >= 2:
            dt = config.DT
            v_arr    = np.array(self.velocities, dtype=float)
            accel    = np.diff(v_arr) / dt                    # m/s²
            jerk     = np.diff(accel)  / dt                   # m/s³
            self.peak_acceleration = float(np.max(np.abs(accel))) if len(accel) > 0 else 0.0
            self.peak_jerk         = float(np.max(np.abs(jerk)))  if len(jerk)  > 0 else 0.0
        else:
            self.peak_acceleration = 0.0
            self.peak_jerk         = 0.0

        # Keep min_obstacle_dist as inf → convert to a large float for CSV
        if self.min_obstacle_dist == float("inf"):
            self.min_obstacle_dist = 999.0

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Return all summary metrics as a flat dict suitable for a pandas row.

        Keys match the plan's CSV column spec:
            controller, scenario, run_id,
            completed, failure_reason,
            completion_time, path_length,
            min_obstacle_dist,
            peak_acceleration, peak_jerk,
            mean_speed, final_x, final_y
        """
        final_x, final_y = self.positions[-1] if self.positions else (float("nan"), float("nan"))
        return {
            "controller":       self.controller_name,
            "scenario":         self.scenario_name,
            "run_id":           self.run_id,
            "completed":        int(self.completed),
            "failure_reason":   self.failure_reason,
            "completion_time":  round(self.completion_time, 4),
            "path_length":      round(self.path_length, 4),
            "min_obstacle_dist":round(self.min_obstacle_dist, 4),
            "peak_acceleration":round(self.peak_acceleration, 4),
            "peak_jerk":        round(self.peak_jerk, 4),
            "mean_speed":       round(self.mean_speed, 4),
            "final_x":          round(final_x, 4),
            "final_y":          round(final_y, 4),
        }

    def save_csv(self, path: str) -> None:
        """
        Save this run's time-series data to a CSV file.

        Columns: t, x, y, theta, v, min_obs_dist, accel, jerk
        The first row has accel=NaN (no previous velocity) and
        the first two rows have jerk=NaN.

        Parameters
        ----------
        path : str   Absolute or relative path to the output .csv file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        v_arr  = np.array(self.velocities, dtype=float)
        dt     = config.DT
        n      = len(v_arr)

        # Acceleration: diff(v)/dt, prepend NaN for first row
        if n >= 2:
            accel = np.concatenate([[float("nan")], np.diff(v_arr) / dt])
        else:
            accel = np.full(n, float("nan"))

        # Jerk: diff(accel)/dt, prepend two NaNs
        valid_accel = accel[1:]   # skip first NaN
        if len(valid_accel) >= 2:
            jerk_vals = np.diff(valid_accel) / dt
            jerk = np.concatenate([[float("nan"), float("nan")], jerk_vals])
        else:
            jerk = np.full(n, float("nan"))

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "x", "y", "theta", "v",
                             "min_obs_dist", "accel", "jerk"])
            for i in range(n):
                x_i, y_i = self.positions[i]
                writer.writerow([
                    round(self.times[i], 4),
                    round(x_i, 4),
                    round(y_i, 4),
                    round(self.thetas[i], 4),
                    round(v_arr[i], 4),
                    round(self.obs_dists[i], 4) if i < len(self.obs_dists) else "",
                    "" if math.isnan(accel[i]) else round(accel[i], 4),
                    "" if math.isnan(jerk[i])  else round(jerk[i], 4),
                ])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mark_failure(self, reason: str, t: float) -> None:
        self.failed         = True
        self.completed      = False
        self.failure_reason = reason
        self._sim_end_t     = t

    def _check_stuck(self, x: float, y: float, t: float) -> None:
        """
        Accumulate time during which the robot moves < STUCK_DIST_THRESHOLD
        per step.  Mark as 'stuck' if accumulated >= STUCK_TIME_THRESHOLD.
        """
        if self._stuck_last_pos is None:
            self._stuck_last_pos = (x, y)
            return

        dx = x - self._stuck_last_pos[0]
        dy = y - self._stuck_last_pos[1]
        moved = math.hypot(dx, dy)

        if moved < config.STUCK_DIST_THRESHOLD:
            self._stuck_accum_t += config.DT
        else:
            self._stuck_accum_t = 0.0   # reset on any real movement
            self._stuck_last_pos = (x, y)

        if self._stuck_accum_t >= config.STUCK_TIME_THRESHOLD and not self.failed:
            self._mark_failure("stuck", t)

    def __repr__(self) -> str:
        status = "completed" if self.completed else (
            f"failed({self.failure_reason})" if self.failed else "in-progress"
        )
        return (f"MetricsRecorder("
                f"{self.controller_name}/{self.scenario_name}/run{self.run_id} "
                f"[{status}])")


# ── Module-level helpers ──────────────────────────────────────────────────

def _min_obstacle_dist(
    rx: float,
    ry: float,
    obstacles: List[Dict[str, Any]],
) -> float:
    """Minimum surface distance from (rx, ry) to any obstacle. inf if none."""
    if not obstacles:
        return float("inf")

    min_d = float("inf")
    for obs in obstacles:
        obs_type = obs.get("type", "static")

        if obs_type == "circle" or "radius" in obs:
            d = max(0.0, math.hypot(rx - obs["x"], ry - obs["y"]) - obs["radius"])
        else:
            # Axis-aligned rectangle
            ox, oy = obs["x"], obs["y"]
            hw = obs.get("w", 1.0) / 2.0
            hh = obs.get("h", 1.0) / 2.0
            cx = max(ox - hw, min(rx, ox + hw))
            cy = max(oy - hh, min(ry, oy + hh))
            d  = math.hypot(rx - cx, ry - cy)

        if d < min_d:
            min_d = d

    return min_d


def _compute_path_length(positions: List[tuple]) -> float:
    """Sum of Euclidean distances between consecutive recorded positions."""
    if len(positions) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        total += math.hypot(dx, dy)
    return total


# ── Quick self-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("=== MetricsRecorder self-test ===")

    rec = MetricsRecorder("RPP", "s1_static_box", run_id=0)
    print(rec)

    # Simulate 40 steps: robot moves East from (1,10) at accelerating speed
    obstacles = [{"type": "static", "x": 10, "y": 10, "w": 1.5, "h": 1.5}]
    state = np.array([1.0, 10.0, 0.0, 0.0])
    t = 0.0
    for step in range(40):
        state[0] += state[3] * config.DT         # x += v*dt
        state[3]  = min(state[3] + 0.05, 1.5)    # accelerate gently
        rec.step(state, obstacles, t)
        t += config.DT

    print(f"  Steps recorded: {len(rec.times)}")
    print(f"  Min obs dist:   {rec.min_obstacle_dist:.3f} m")
    assert len(rec.times) == 40
    assert rec.min_obstacle_dist < 999.0

    # Finish as success
    rec.finish(success=True, t=t)
    d = rec.to_dict()
    print(f"  to_dict keys:   {list(d.keys())}")
    print(f"  completion_time:{d['completion_time']:.3f} s")
    print(f"  path_length:    {d['path_length']:.3f} m")
    print(f"  peak_accel:     {d['peak_acceleration']:.4f} m/s2")
    print(f"  peak_jerk:      {d['peak_jerk']:.4f} m/s3")
    assert d["completed"] == 1
    assert d["failure_reason"] == ""
    assert d["path_length"] > 0

    # Save time-series CSV to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "run.csv")
        rec.save_csv(csv_path)
        with open(csv_path) as f:
            lines = f.readlines()
        print(f"  CSV rows:       {len(lines) - 1} data rows + 1 header")
        assert len(lines) == 41   # 1 header + 40 data rows

    # Test failure detection — stuck robot
    rec2 = MetricsRecorder("DWA", "s1_static_box", run_id=1)
    stuck_state = np.array([5.0, 10.0, 0.0, 0.0])
    t2 = 0.0
    # Run for STUCK_TIME_THRESHOLD + buffer
    n_stuck = int(config.STUCK_TIME_THRESHOLD / config.DT) + 10
    for _ in range(n_stuck):
        rec2.step(stuck_state, [], t2)
        t2 += config.DT
    print(f"  Stuck failure:  '{rec2.failure_reason}'  (expect 'stuck')")
    assert rec2.failure_reason == "stuck", f"Got: {rec2.failure_reason}"

    # Test collision marking
    rec3 = MetricsRecorder("MPPI", "s1_static_box", run_id=2)
    collision_state = np.array([10.0, 10.0, 0.0, 1.0])   # inside obstacle
    rec3.step(collision_state, obstacles, 0.0)
    print(f"  Collision flag: failure_reason='{rec3.failure_reason}'  (expect 'collision')")
    assert rec3.failure_reason == "collision"

    print("\n[PASS]  MetricsRecorder OK")
