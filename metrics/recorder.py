"""
metrics/recorder.py
====================
MetricsRecorder — records all per-run data and computes final metrics.
Upgraded with:
  - Full timestep structured log (timestep_log)
  - Reaction time computation for sudden obstacles (S5)
  - Oscillation and recovery counters
  - Scenario gap width metadata
  - High-precision distance calculation supporting Obstacle objects and Costmap2D
"""

import os
import math
import csv
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class MetricsRecorder:
    """
    Records all metrics for a single controller × scenario × run combination.
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
        self.times:       List[float] = []   # t (s)
        self.positions:   List[tuple] = []   # (x, y)
        self.thetas:      List[float] = []   # heading (rad)
        self.velocities:  List[float] = []   # v (m/s)
        self.obs_dists:   List[float] = []   # min dist to obstacle each step

        # ── Extended Phase F fields ───────────────────────────────────
        self.timestep_log:   List[Dict[str, Any]] = []
        self.reaction_time:  Optional[float] = None
        self.oscillations:   int = 0
        self.recoveries:     int = 0
        self.gap_width:      Optional[float] = None
        self.spawn_timestep: Optional[int] = None

        # ── Running scalar metrics ─────────────────────────────────────
        self.min_obstacle_dist: float = float("inf")
        self.start_time:        Optional[float] = None
        self.end_time:          Optional[float] = None
        self._sim_start_t:      Optional[float] = None
        self._sim_end_t:        Optional[float] = None

        # ── Outcome ───────────────────────────────────────────────────
        self.completed:       bool = False
        self.failed:          bool = False
        self.failure_reason:  str  = ""

        # ── Derived (computed in finish()) ────────────────────────────
        self.completion_time:   float = float("nan")
        self.path_length:       float = 0.0
        self.peak_acceleration: float = 0.0
        self.peak_jerk:         float = 0.0
        self.mean_speed:        float = 0.0

        # ── Stuck detection state ─────────────────────────────────────
        self._stuck_last_pos:  Optional[tuple] = None
        self._stuck_accum_t:   float = 0.0

    # ------------------------------------------------------------------
    # Per-step recording
    # ------------------------------------------------------------------

    def step(
        self,
        state:     np.ndarray,
        obstacles: list,
        t:         float,
        control:   Optional[Tuple[float, float]] = None,
        costmap = None,
    ) -> None:
        """
        Record one simulation timestep.
        """
        x, y, theta, v = float(state[0]), float(state[1]), float(state[2]), float(state[3])

        if self._sim_start_t is None:
            self._sim_start_t = t
            self.start_time   = time.time()

        self.times.append(t)
        self.positions.append((x, y))
        self.thetas.append(theta)
        self.velocities.append(v)

        # Distance to nearest obstacle
        min_d = _min_obstacle_dist(x, y, obstacles)
        self.obs_dists.append(min_d)
        if min_d < self.min_obstacle_dist:
            self.min_obstacle_dist = min_d

        # Costmap cost lookup
        cell_cost = costmap.query(x, y) if costmap is not None else 0.0

        # Extended structured log
        steer_cmd = float(control[0]) if control is not None else 0.0
        target_v  = float(control[1]) if control is not None else v

        self.timestep_log.append({
            "t":             round(t, 4),
            "x":             round(x, 4),
            "y":             round(y, 4),
            "theta":         round(theta, 4),
            "v":             round(v, 4),
            "steer":         round(steer_cmd, 4),
            "target_speed":  round(target_v, 4),
            "costmap_cost":  round(float(cell_cost), 2),
            "min_obs_dist":  round(float(min_d), 4),
        })

        # Collision detection (using robot radius)
        if min_d < config.ROBOT_RADIUS and not self.failed:
            self._mark_failure("collision", t)

        # Stuck detection
        self._check_stuck(x, y, t)

    def mark_timeout(self, t: float) -> None:
        if not self.failed:
            self._mark_failure("timeout", t)

    def mark_stuck(self, t: float) -> None:
        if not self.failed:
            self._mark_failure("stuck", t)

    def record_spawn_event(self, spawn_timestep: int):
        """Record the timestep when a sudden obstacle spawned."""
        self.spawn_timestep = int(spawn_timestep)

    def compute_reaction_time(self) -> Optional[float]:
        """
        Reaction time = seconds from sudden spawn until vehicle initiates
        braking (|accel| > 0.3 m/s²).
        """
        if self.spawn_timestep is None:
            return None

        dt = config.DT
        for i in range(self.spawn_timestep, len(self.timestep_log)):
            if i > 0:
                dv = abs(self.timestep_log[i]["v"] - self.timestep_log[i - 1]["v"])
                a = dv / dt
                if a > 0.3:
                    self.reaction_time = round((i - self.spawn_timestep) * dt, 4)
                    return self.reaction_time
        return None

    # ------------------------------------------------------------------
    # Run end
    # ------------------------------------------------------------------

    def finish(self, success: bool, t: float = None, reason: str = "") -> None:
        """
        Finalise the run.
        """
        self.end_time    = time.time()
        self._sim_end_t  = t if t is not None else (self.times[-1] if self.times else 0.0)
        self.completed   = success

        if reason and not self.failed:
            self._mark_failure(reason, self._sim_end_t)
        elif success and not self.failed:
            self.failure_reason = ""

        # Completion time
        if self._sim_start_t is not None:
            self.completion_time = self._sim_end_t - self._sim_start_t
        else:
            self.completion_time = 0.0

        # Path length
        self.path_length = _compute_path_length(self.positions)

        # Mean speed
        if self.velocities:
            self.mean_speed = float(np.mean(self.velocities))

        # Peak acceleration and jerk
        if len(self.velocities) >= 2:
            dt = config.DT
            v_arr = np.array(self.velocities, dtype=float)
            accel = np.diff(v_arr) / dt
            jerk  = np.diff(accel) / dt
            self.peak_acceleration = float(np.max(np.abs(accel))) if len(accel) > 0 else 0.0
            self.peak_jerk         = float(np.max(np.abs(jerk)))  if len(jerk)  > 0 else 0.0
        else:
            self.peak_acceleration = 0.0
            self.peak_jerk         = 0.0

        if self.min_obstacle_dist == float("inf"):
            self.min_obstacle_dist = 999.0

        # Compute reaction time if applicable
        self.compute_reaction_time()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return flat dict of summary metrics."""
        final_x, final_y = self.positions[-1] if self.positions else (float("nan"), float("nan"))
        return {
            "controller":        self.controller_name,
            "scenario":          self.scenario_name,
            "run_id":            self.run_id,
            "completed":         int(self.completed),
            "failure_reason":    self.failure_reason,
            "completion_time":   round(self.completion_time, 4),
            "path_length":       round(self.path_length, 4),
            "min_obstacle_dist": round(self.min_obstacle_dist, 4),
            "peak_acceleration": round(self.peak_acceleration, 4),
            "peak_jerk":         round(self.peak_jerk, 4),
            "mean_speed":        round(self.mean_speed, 4),
            "final_x":           round(final_x, 4),
            "final_y":           round(final_y, 4),
            "reaction_time_s":   self.reaction_time,
            "oscillation_count": self.oscillations,
            "recovery_count":    self.recoveries,
            "gap_width_m":       self.gap_width,
        }

    def save_csv(self, path: str) -> None:
        """Save this run's time-series data to a CSV file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        v_arr = np.array(self.velocities, dtype=float)
        dt = config.DT
        n = len(v_arr)

        if n >= 2:
            accel = np.concatenate([[float("nan")], np.diff(v_arr) / dt])
        else:
            accel = np.full(n, float("nan"))

        valid_accel = accel[1:]
        if len(valid_accel) >= 2:
            jerk_vals = np.diff(valid_accel) / dt
            jerk = np.concatenate([[float("nan"), float("nan")], jerk_vals])
        else:
            jerk = np.full(n, float("nan"))

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "x", "y", "theta", "v", "min_obs_dist", "accel", "jerk"])
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

    def save_timestep_csv(self, out_dir: str = "results/raw") -> str:
        """Save extended structured timestep log (Phase F)."""
        import pandas as pd
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"{self.controller_name}__{self.scenario_name}__run{self.run_id:02d}.csv")
        pd.DataFrame(self.timestep_log).to_csv(fname, index=False)
        return fname

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mark_failure(self, reason: str, t: float) -> None:
        self.failed         = True
        self.completed      = False
        self.failure_reason = reason
        self._sim_end_t     = t

    def _check_stuck(self, x: float, y: float, t: float) -> None:
        if self._stuck_last_pos is None:
            self._stuck_last_pos = (x, y)
            return

        dx = x - self._stuck_last_pos[0]
        dy = y - self._stuck_last_pos[1]
        moved = math.hypot(dx, dy)

        if moved < config.STUCK_DIST_THRESHOLD:
            self._stuck_accum_t += config.DT
        else:
            self._stuck_accum_t = 0.0
            self._stuck_last_pos = (x, y)

        if self._stuck_accum_t >= config.STUCK_TIME_THRESHOLD and not self.failed:
            self._mark_failure("stuck", t)

    def __repr__(self) -> str:
        status = "completed" if self.completed else (
            f"failed({self.failure_reason})" if self.failed else "in-progress"
        )
        return (f"MetricsRecorder({self.controller_name}/{self.scenario_name}/run{self.run_id} [{status}])")


def _min_obstacle_dist(rx: float, ry: float, obstacles: list) -> float:
    if not obstacles:
        return float("inf")
    min_d = float("inf")
    for obs in obstacles:
        if hasattr(obs, "dist_to_point"):
            d = obs.dist_to_point(rx, ry)
        elif isinstance(obs, dict):
            if obs.get("type") == "circle" or "radius" in obs:
                d = max(0.0, math.hypot(rx - obs["x"], ry - obs["y"]) - obs["radius"])
            else:
                hw = obs.get("w", 1.0) / 2.0
                hh = obs.get("h", 1.0) / 2.0
                cx = float(np.clip(rx, obs["x"] - hw, obs["x"] + hw))
                cy = float(np.clip(ry, obs["y"] - hh, obs["y"] + hh))
                d = math.hypot(rx - cx, ry - cy)
        else:
            d = float("inf")
        if d < min_d:
            min_d = d
    return min_d


def _compute_path_length(positions: List[tuple]) -> float:
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
    print("=== MetricsRecorder Upgraded Self-Test ===")
    rec = MetricsRecorder("MPPI", "s5_sudden_obstacle", run_id=0)
    rec.record_spawn_event(spawn_timestep=10)

    for i in range(25):
        t = i * 0.05
        state = np.array([float(i) * 0.05, 10.0, 0.0, 1.0 if i < 15 else 0.5])
        rec.step(state, [], t, control=(0.0, 1.0))

    rec.finish(success=True)
    d = rec.to_dict()
    print("Reaction time:", d["reaction_time_s"])
    assert d["reaction_time_s"] is not None
    print("[PASS] MetricsRecorder verified!")
